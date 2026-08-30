# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a standalone OGPO behavior-cloning checkpoint on ManiSkill demos."""

from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from maniskill_bc_dataset import ManiSkillBCDataset
from rlinf.algorithms.losses import compute_flow_matching_bc_loss
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.metric_logger import MetricLogger


def _validate_bc_config(cfg) -> None:
    positive_integer_fields = (
        "batch_size",
        "max_updates",
        "log_interval",
        "eval_interval",
        "eval_episodes",
        "eval_num_envs",
        "eval_max_episode_steps",
    )
    for field in positive_integer_fields:
        if int(cfg.bc[field]) <= 0:
            raise ValueError(f"bc.{field} must be positive.")
    if int(cfg.bc.num_workers) < 0:
        raise ValueError("bc.num_workers must be non-negative.")
    if float(cfg.bc.lr) <= 0:
        raise ValueError("bc.lr must be positive.")
    if float(cfg.bc.weight_decay) < 0:
        raise ValueError("bc.weight_decay must be non-negative.")
    if float(cfg.bc.clip_grad_norm) <= 0:
        raise ValueError("bc.clip_grad_norm must be positive.")

    success_threshold = float(cfg.bc.success_threshold)
    if not 0.0 <= success_threshold < 1.0:
        raise ValueError("bc.success_threshold must be in [0, 1).")
    if int(cfg.bc.eval_episodes) % int(cfg.bc.eval_num_envs) != 0:
        raise ValueError("bc.eval_episodes must be divisible by bc.eval_num_envs.")
    if bool(cfg.bc.eval_save_video):
        if int(cfg.bc.eval_video_interval) <= 0:
            raise ValueError("bc.eval_video_interval must be positive.")
        if int(cfg.bc.eval_video_num_envs) <= 0:
            raise ValueError("bc.eval_video_num_envs must be positive.")
        if int(cfg.bc.eval_video_interval) % int(cfg.bc.eval_interval) != 0:
            raise ValueError(
                "bc.eval_video_interval must be a multiple of bc.eval_interval."
            )

    if str(cfg.actor.model.model_type) != "flow_policy":
        raise ValueError("OGPO BC requires actor.model.model_type=flow_policy.")
    if str(cfg.actor.model.input_type) != "state":
        raise ValueError("OGPO PickCube BC requires actor.model.input_type=state.")
    if str(cfg.actor.model.flow_actor_type) != "OGPOFlowActor":
        raise ValueError("OGPO BC requires actor.model.flow_actor_type=OGPOFlowActor.")
    if int(cfg.actor.model.num_action_chunks) <= 0:
        raise ValueError("actor.model.num_action_chunks must be positive.")
    if int(cfg.actor.model.obs_dim) <= 0 or int(cfg.actor.model.action_dim) <= 0:
        raise ValueError("actor.model obs_dim and action_dim must be positive.")
    if float(cfg.algorithm.ogpo.constant_noise_std) <= 0:
        raise ValueError("algorithm.ogpo.constant_noise_std must be positive.")
    if float(cfg.algorithm.ogpo.get("randn_clip_value", 3.0)) <= 0:
        raise ValueError("algorithm.ogpo.randn_clip_value must be positive.")
    if float(cfg.algorithm.ogpo.get("denoised_clip_value", 1.0)) <= 0:
        raise ValueError("algorithm.ogpo.denoised_clip_value must be positive.")


def _build_model(cfg, device: torch.device):
    model_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.actor.model, resolve=True)
    )
    model_cfg.load_to_device = False
    model = get_model(model_cfg)
    if model is None:
        raise ValueError(f"Unsupported actor model type: {model_cfg.model_type}")
    return model.to(device)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BC config requests CUDA, but torch.cuda.is_available() is false.")
    return device


def _build_eval_env(
    cfg,
    *,
    num_envs: int | None = None,
    record_video: bool = False,
    video_sampling_method: str | None = None,
):
    from rlinf.envs.maniskill.maniskill_env import ManiskillEnv

    num_envs = int(cfg.bc.eval_num_envs) if num_envs is None else int(num_envs)
    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.eval, resolve=True))
    env_cfg.total_num_envs = num_envs
    env_cfg.auto_reset = False
    env_cfg.ignore_terminations = True
    env_cfg.use_fixed_reset_state_ids = True
    env_cfg.video_cfg.save_video = record_video
    if record_video:
        if video_sampling_method not in {"sde", "ode"}:
            raise ValueError(
                "video_sampling_method must be 'sde' or 'ode' when recording."
            )
        env_cfg.video_cfg.video_base_dir = str(
            Path(str(cfg.bc.eval_video_dir)) / video_sampling_method
        )
    env_cfg.init_params.num_envs = num_envs
    env_cfg.init_params.obs_mode = "state"
    env_cfg.init_params.control_mode = str(cfg.bc.expected_control_mode)
    env_cfg.init_params.sim_backend = str(cfg.bc.eval_sim_backend)
    env_cfg.init_params.max_episode_steps = int(cfg.bc.eval_max_episode_steps)
    env = ManiskillEnv(
        cfg=env_cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )
    action_shape = tuple(env.env.unwrapped.single_action_space.shape)
    expected_action_shape = (int(cfg.actor.model.action_dim),)
    if action_shape != expected_action_shape:
        env.env.close()
        raise ValueError(
            f"Evaluation action space has shape {action_shape}; expected "
            f"{expected_action_shape}. Check policy_setup/control_mode."
        )
    if record_video:
        from rlinf.envs.wrappers import RecordVideo

        return RecordVideo(env, env_cfg.video_cfg)
    return env


def _fixed_eval_batches(
    env,
    cfg,
    *,
    eval_episodes: int | None = None,
    num_envs: int | None = None,
) -> list[tuple[int, torch.Tensor]]:
    eval_episodes = (
        int(cfg.bc.eval_episodes) if eval_episodes is None else int(eval_episodes)
    )
    num_envs = int(cfg.bc.eval_num_envs) if num_envs is None else int(num_envs)
    if eval_episodes % num_envs != 0:
        raise ValueError("eval_episodes must be divisible by num_envs.")
    base_env = env.unwrapped
    generator = torch.Generator()
    generator.manual_seed(int(cfg.bc.eval_seed))
    episode_ids = torch.randint(
        low=0,
        high=base_env.total_num_group_envs,
        size=(eval_episodes,),
        generator=generator,
    )
    return [
        (int(cfg.bc.eval_seed) + batch_index, chunk.to(base_env.device))
        for batch_index, chunk in enumerate(episode_ids.split(num_envs))
    ]


def _close_eval_env(env) -> None:
    """Wait for pending video writes, then close the ManiSkill simulator."""
    base_env = env.unwrapped
    env.close()
    base_env.env.close()


def _sample_eval_action_chunk(model, obs, cfg, sampling_method: str) -> torch.Tensor:
    model_device = next(model.parameters()).device
    model_obs = {"states": obs["states"].to(model_device)}
    if sampling_method == "ode":
        flat_actions = model.ogpo_ode_forward(
            model_obs,
            num_samples=1,
            clip_intermediate=bool(
                cfg.algorithm.ogpo.get("clip_intermediate_actions", True)
            ),
            clip_value=float(
                cfg.algorithm.ogpo.get("denoised_clip_value", 1.0)
            ),
        )[0]
    elif sampling_method != "sde":
        raise ValueError(f"Unsupported evaluation sampling method: {sampling_method}")
    else:
        candidates, _, _ = model(
            forward_type=ForwardType.OGPO_SAMPLE,
            obs=model_obs,
            num_samples=1,
            noise_std=float(cfg.algorithm.ogpo.constant_noise_std),
            normalize_horizon=bool(
                cfg.algorithm.ogpo.get("normalize_denoising_horizon", True)
            ),
            normalize_dimension=bool(
                cfg.algorithm.ogpo.get("normalize_action_dimension", True)
            ),
            randn_clip_value=float(cfg.algorithm.ogpo.get("randn_clip_value", 3.0)),
            use_tapered_noise=bool(
                cfg.algorithm.ogpo.get("use_tapered_noise", False)
            ),
            ignore_last=bool(cfg.algorithm.ogpo.get("ignore_last", True)),
            error_correct_sde_to_ode=bool(
                cfg.algorithm.ogpo.get("error_correct_sde_to_ode", True)
            ),
            clip_intermediate=bool(
                cfg.algorithm.ogpo.get("clip_intermediate_actions", True)
            ),
            clip_value=float(
                cfg.algorithm.ogpo.get("denoised_clip_value", 1.0)
            ),
        )
        flat_actions = candidates[0]

    num_action_chunks = int(cfg.actor.model.num_action_chunks)
    action_dim = int(cfg.actor.model.action_dim)
    expected_flat_dim = num_action_chunks * action_dim
    if flat_actions.ndim != 2 or flat_actions.shape[1] != expected_flat_dim:
        raise ValueError(
            f"Sampled actions have shape {tuple(flat_actions.shape)}; expected "
            f"[B, {expected_flat_dim}]."
        )
    return flat_actions.reshape(-1, num_action_chunks, action_dim)


@torch.no_grad()
def evaluate_single_sample(model, env, eval_batches, cfg, sampling_method: str) -> float:
    """Evaluate one SDE or ODE action chunk per observation without Q selection."""
    was_training = model.training
    model.eval()
    successes: list[torch.Tensor] = []
    model_device = next(model.parameters()).device
    env_device = env.unwrapped.device

    cuda_devices = []
    if model_device.type == "cuda":
        cuda_devices = [
            model_device.index
            if model_device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(cfg.bc.eval_seed))
        for reset_seed, episode_ids in eval_batches:
            obs, _ = env.reset(
                seed=reset_seed, options={"episode_id": episode_ids}
            )
            if obs["states"].shape[-1] != int(cfg.actor.model.obs_dim):
                raise ValueError(
                    f"Evaluation states have shape {tuple(obs['states'].shape)}; "
                    f"expected final dimension {int(cfg.actor.model.obs_dim)}."
                )
            success_once = torch.zeros(
                episode_ids.shape[0], dtype=torch.bool, device=env_device
            )
            steps = 0
            max_steps = int(cfg.bc.eval_max_episode_steps)
            while steps < max_steps:
                action_chunk = _sample_eval_action_chunk(
                    model, obs, cfg, sampling_method
                ).to(env_device)
                for action in action_chunk.unbind(dim=1):
                    if steps >= max_steps:
                        break
                    obs, _, _, _, infos = env.step(action)
                    episode_metrics = infos.get("episode", {})
                    if "success_once" in episode_metrics:
                        success_once |= episode_metrics["success_once"].bool()
                    steps += 1
            successes.append(success_once.cpu())

    model.train(was_training)
    return torch.cat(successes).float().mean().item()


def evaluate_single_sample_sde(model, env, eval_batches, cfg) -> float:
    """Evaluate one SDE chunk per observation without Q selection or best-of-N."""
    return evaluate_single_sample(model, env, eval_batches, cfg, "sde")


def evaluate_single_sample_ode(model, env, eval_batches, cfg) -> float:
    """Evaluate one ODE chunk per observation without Q selection or best-of-N."""
    return evaluate_single_sample(model, env, eval_batches, cfg, "ode")


def _cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def _save_and_verify_checkpoint(model, cfg, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = _cpu_state_dict(model)
    torch.save(state_dict, output_path)

    verification_model = _build_model(cfg, torch.device("cpu"))
    verification_model.load_state_dict(
        torch.load(output_path, map_location="cpu", weights_only=True), strict=True
    )


@hydra.main(version_base="1.1", config_path="config", config_name="maniskill_ogpo_bc")
def main(cfg) -> None:
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    _validate_bc_config(cfg)
    _set_seed(int(cfg.bc.seed))
    device = _resolve_device(str(cfg.bc.device))

    dataset = ManiSkillBCDataset(
        cfg.bc.dataset_path,
        obs_dim=int(cfg.actor.model.obs_dim),
        action_dim=int(cfg.actor.model.action_dim),
        num_action_chunks=int(cfg.actor.model.num_action_chunks),
        expected_env_id=str(cfg.bc.expected_env_id),
        expected_control_mode=str(cfg.bc.expected_control_mode),
        success_only=bool(cfg.bc.success_only),
    )
    print(
        f"Loaded {len(dataset)} action chunks from {dataset.kept_episodes}/"
        f"{dataset.total_episodes} episodes."
    )
    data_generator = torch.Generator()
    data_generator.manual_seed(int(cfg.bc.seed))
    data_loader = DataLoader(
        dataset,
        batch_size=int(cfg.bc.batch_size),
        shuffle=True,
        num_workers=int(cfg.bc.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=data_generator,
    )

    model = _build_model(cfg, device)
    if hasattr(model, "q_head"):
        model.q_head.requires_grad_(False)
    actor_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("q_head.")
    ]
    optimizer = torch.optim.AdamW(
        actor_parameters,
        lr=float(cfg.bc.lr),
        weight_decay=float(cfg.bc.weight_decay),
    )
    logger = MetricLogger(cfg)
    eval_env = _build_eval_env(cfg)
    eval_batches = _fixed_eval_batches(eval_env, cfg)
    video_envs = {}
    video_eval_batches = {}
    if bool(cfg.bc.eval_save_video):
        video_num_envs = int(cfg.bc.eval_video_num_envs)
        for sampling_method in ("sde", "ode"):
            video_envs[sampling_method] = _build_eval_env(
                cfg,
                num_envs=video_num_envs,
                record_video=True,
                video_sampling_method=sampling_method,
            )
            video_eval_batches[sampling_method] = _fixed_eval_batches(
                video_envs[sampling_method],
                cfg,
                eval_episodes=video_num_envs,
                num_envs=video_num_envs,
            )
    output_path = Path(str(cfg.bc.output_path)).expanduser().resolve()

    update = 0
    data_iterator = iter(data_loader)
    last_success_rate = 0.0
    try:
        model.train()
        while update < int(cfg.bc.max_updates):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                batch = next(data_iterator)

            states = batch["states"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            predicted_velocity, target_velocity = model(
                forward_type=ForwardType.OGPO_BC,
                obs={"states": states},
                actions=actions,
            )
            loss, metrics = compute_flow_matching_bc_loss(
                predicted_velocity, target_velocity
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                actor_parameters, float(cfg.bc.clip_grad_norm)
            )
            optimizer.step()
            update += 1

            if update % int(cfg.bc.log_interval) == 0:
                log_data = {
                    **{key: float(value) for key, value in metrics.items()},
                    "actor/grad_norm": float(grad_norm),
                }
                logger.log(log_data, step=update)
                print(
                    f"[BC] update={update} loss={float(loss):.6f} "
                    f"grad_norm={float(grad_norm):.4f}"
                )

            should_eval = (
                update % int(cfg.bc.eval_interval) == 0
                or update == int(cfg.bc.max_updates)
            )
            if should_eval:
                sde_success_rate = evaluate_single_sample_sde(
                    model, eval_env, eval_batches, cfg
                )
                ode_success_rate = evaluate_single_sample_ode(
                    model, eval_env, eval_batches, cfg
                )
                last_success_rate = sde_success_rate
                logger.log(
                    {
                        "eval/sde_single_sample_success_once": sde_success_rate,
                        "eval/ode_single_sample_success_once": ode_success_rate,
                    },
                    step=update,
                )
                print(
                    f"[EVAL SDE-1] update={update} "
                    f"success_once={sde_success_rate:.4f}"
                )
                print(
                    f"[EVAL ODE-1] update={update} "
                    f"success_once={ode_success_rate:.4f}"
                )
                should_record_video = bool(video_envs) and (
                    update % int(cfg.bc.eval_video_interval) == 0
                    or update == int(cfg.bc.max_updates)
                )
                if should_record_video:
                    video_evaluators = {
                        "sde": evaluate_single_sample_sde,
                        "ode": evaluate_single_sample_ode,
                    }
                    for sampling_method, video_env in video_envs.items():
                        video_success_rate = video_evaluators[sampling_method](
                            model,
                            video_env,
                            video_eval_batches[sampling_method],
                            cfg,
                        )
                        video_env.flush_video(
                            video_sub_dir=f"update_{update:08d}"
                        )
                        print(
                            f"[EVAL {sampling_method.upper()} VIDEO] update={update} "
                            f"success_once={video_success_rate:.4f}"
                        )
                if last_success_rate > float(cfg.bc.success_threshold):
                    _save_and_verify_checkpoint(model, cfg, output_path)
                    print(f"Saved BC checkpoint to {output_path}")
                    break
        if last_success_rate <= float(cfg.bc.success_threshold):
            if bool(cfg.bc.save_last):
                _save_and_verify_checkpoint(model, cfg, output_path)
                print(
                    "BC did not reach the success threshold; saved the final "
                    f"checkpoint to {output_path}."
                )
            else:
                print("BC did not reach the success threshold; no checkpoint saved.")
    finally:
        logger.finish()
        _close_eval_env(eval_env)
        for video_env in video_envs.values():
            _close_eval_env(video_env)


if __name__ == "__main__":
    main()
