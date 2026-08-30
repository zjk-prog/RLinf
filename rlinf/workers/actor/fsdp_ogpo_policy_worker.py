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

import os
from math import ceil
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.algorithms.advantages import compute_group_q_advantages
from rlinf.algorithms.losses import (
    compute_flow_matching_bc_loss,
    compute_q_td_loss,
)
from rlinf.algorithms.registry import policy_loss
from rlinf.algorithms.utils import aggregate_q_values, compute_one_step_td_target
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import clear_memory, collect_param_names_need_sync
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedOGPOFSDPPolicy(EmbodiedFSDPActor):
    """FSDP worker for Torch-native online OGPO."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.target_model = None
        self.replay_buffer = None
        self.qf_optimizer = None
        self.qf_lr_scheduler = None
        self.update_step = 0
        self.ogpo_cfg = self.cfg.algorithm.get("ogpo", {})

    def init_worker(self) -> None:
        self.setup_model_and_optimizer()
        self.setup_ogpo_buffers()
        self.update_target_models(actor_tau=1.0, critic_tau=1.0)
        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def setup_model_and_optimizer(self) -> None:
        online_module = self.model_provider_func()
        target_module = self.model_provider_func()
        self.param_names_need_sync = collect_param_names_need_sync(target_module)
        self.model = self._strategy.wrap_model(
            model=online_module, device_mesh=self._device_mesh
        )
        self.target_model = self._strategy.wrap_model(
            model=target_module, device_mesh=self._device_mesh
        )
        self.target_model.requires_grad_(False)
        self.target_model.eval()
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype

        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters={"critic": ["q_head"]},
            filtered_optim_config={"critic": self.cfg.actor.critic_optim},
        )
        self.optimizer, self.qf_optimizer = optimizers
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )

        scaler_cfg = self.cfg.actor.fsdp_config.get("grad_scaler", {})
        scaler_kwargs = {
            key: scaler_cfg[key]
            for key in ("init_scale", "growth_interval")
            if scaler_cfg.get(key) is not None
        }
        self.grad_scaler = self.build_grad_scaler(
            scaler_cfg.get("enabled", False), **scaler_kwargs
        )

    def _new_buffer(self, name: str) -> TrajectoryReplayBuffer:
        buffer_cfg = self.cfg.algorithm.replay_buffer
        auto_save_path = buffer_cfg.get("auto_save_path")
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, name, f"rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(
                auto_save_path, name, f"rank_{self._rank}"
            )
        return TrajectoryReplayBuffer(
            seed=self.cfg.actor.get("seed", 1234),
            enable_cache=buffer_cfg.get("enable_cache", True),
            cache_size=buffer_cfg.get("cache_size", 100),
            sample_window_size=buffer_cfg.get("sample_window_size", 100),
            auto_save=buffer_cfg.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=buffer_cfg.get("trajectory_format", "pt"),
        )

    def setup_ogpo_buffers(self) -> None:
        self.replay_buffer = self._new_buffer("replay_buffer")
        seed = int(self.cfg.actor.get("seed", 1234))
        self.q_sample_generator = torch.Generator(device=self.device)
        self.q_sample_generator.manual_seed(seed + self._rank)

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """Receive variable-count complete episodes in fixed channel messages."""
        clear_memory(sync=False)
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        episodes: list[Trajectory] = []
        for _ in range(split_num):
            message = await input_channel.get(async_op=True).async_wait()
            if isinstance(message, list):
                episodes.extend(message)
            elif isinstance(message, Trajectory):
                episodes.append(message)
            else:
                raise TypeError(f"Unsupported OGPO replay message: {type(message)}")

        self._ingest_episodes(episodes)

    def _ingest_episodes(self, episodes: list[Trajectory]) -> None:
        """Add complete episodes to the replay buffer."""
        self.replay_buffer.add_trajectories(episodes)

    @torch.no_grad()
    def update_target_models(
        self,
        actor_tau: float | None = None,
        critic_tau: float | None = None,
    ) -> None:
        actor_tau = (
            float(self.ogpo_cfg.get("actor_tau", 0.005))
            if actor_tau is None
            else actor_tau
        )
        critic_tau = (
            float(self.ogpo_cfg.get("critic_tau", 0.005))
            if critic_tau is None
            else critic_tau
        )
        for (online_name, online_param), (target_name, target_param) in zip(
            self.model.named_parameters(), self.target_model.named_parameters()
        ):
            if online_name != target_name:
                raise RuntimeError(
                    f"Online/target parameter mismatch: {online_name} != {target_name}"
                )
            is_critic = "q_head" in online_name
            tau = critic_tau if is_critic else actor_tau
            target_param.data.lerp_(online_param.data, tau)

    def get_rollout_state_dict(self) -> dict[str, torch.Tensor]:
        return self._strategy.get_model_state_dict(
            self.target_model, cpu_offload=False, full_state_dict=False
        )

    def _sample_sequences(
        self, batch_size: int, *, success_only: bool = False
    ) -> dict[str, Any]:
        return self.replay_buffer.sample_sequences(
            batch_size,
            sequence_length=int(self.cfg.actor.model.num_action_chunks),
            discount=float(self.cfg.algorithm.gamma),
            success_only=success_only,
        )

    def _sample_train_micro_batches(
        self,
        *,
        success_only: bool = False,
    ) -> list[dict[str, Any]]:
        per_rank_batch = self.cfg.actor.global_batch_size // self._world_size
        batch = self._sample_sequences(per_rank_batch, success_only=success_only)
        actual_batch_size = int(batch["actions"].shape[0])
        micro_batch_size = int(self.cfg.actor.micro_batch_size)
        split_count = max(1, ceil(actual_batch_size / micro_batch_size))
        return [
            put_tensor_device(micro_batch, device=self.device)
            for micro_batch in split_dict_to_chunk(batch, split_count)
        ]

    def _success_q_ready(self) -> bool:
        """Return whether every data-parallel rank has a full success batch."""
        per_rank_batch = self.cfg.actor.global_batch_size // self._world_size
        ready = torch.tensor(
            int(
                self.replay_buffer.get_sampleable_count(success_only=True)
                >= per_rank_batch
            ),
            device=self.device,
            dtype=torch.int32,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
        return bool(ready.item())

    def _replay_buffer_ready(self, min_buffer_size: int) -> bool:
        """Return whether every actor rank has enough replay trajectories."""
        ready = torch.tensor(
            int(self.replay_buffer.is_ready(min_buffer_size)),
            device=self.device,
            dtype=torch.int32,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
        return bool(ready.item())

    def _aggregate_q(self, q_values: torch.Tensor) -> torch.Tensor:
        return aggregate_q_values(
            q_values.float(),
            method=self.ogpo_cfg.get("td_q_aggregation", "mean"),
            ensemble_dim=-1,
            subsample_size=int(self.ogpo_cfg.get("td_q_subsample_size", 2)),
            generator=self.q_sample_generator,
        )

    @staticmethod
    def _repeat_obs(
        obs: dict[str, torch.Tensor], repeats: int
    ) -> dict[str, torch.Tensor]:
        """Repeat state observations in candidate-major order."""
        return {
            key: value.unsqueeze(0)
            .expand(repeats, *value.shape)
            .reshape(repeats * value.shape[0], *value.shape[1:])
            for key, value in obs.items()
        }

    def _candidate_q(
        self,
        model,
        obs: dict[str, torch.Tensor],
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        num_samples, batch_size = candidates.shape[:2]
        flow_action_dim = (
            int(self.cfg.actor.model.num_action_chunks)
            * int(self.cfg.actor.model.action_dim)
        )
        flat_q = model(
            forward_type=ForwardType.SAC_Q,
            obs=self._repeat_obs(obs, num_samples),
            actions=candidates.reshape(
                num_samples * batch_size, flow_action_dim
            ),
        )
        return flat_q.reshape(num_samples, batch_size, -1)

    @Worker.timer("ogpo_forward_critic")
    def forward_critic(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"].reshape(batch["actions"].shape[0], -1)
        batch_size = actions.shape[0]
        horizon = int(self.cfg.actor.model.num_action_chunks)
        rewards = batch["rewards"].float().reshape(batch_size, horizon, -1)[
            :, -1, 0
        ]
        terminations = batch["terminations"].bool().reshape(
            batch_size, horizon, -1
        )[:, -1, 0]
        valid = batch["valid"].bool()[:, -1]

        with torch.no_grad():
            next_actions, _, _ = self.target_model(
                forward_type=ForwardType.OGPO_SAMPLE,
                obs=next_obs,
                num_samples=1,
                noise_std=float(self.ogpo_cfg.constant_noise_std),
                normalize_horizon=bool(
                    self.ogpo_cfg.get("normalize_denoising_horizon", True)
                ),
                normalize_dimension=bool(
                    self.ogpo_cfg.get("normalize_action_dimension", True)
                ),
                randn_clip_value=float(self.ogpo_cfg.get("randn_clip_value", 3.0)),
                clip_randn=bool(self.ogpo_cfg.get("clip_randn", True)),
                use_tapered_noise=bool(
                    self.ogpo_cfg.get("use_tapered_noise", False)
                ),
                ignore_last=bool(self.ogpo_cfg.get("ignore_last", True)),
                error_correct_sde_to_ode=bool(
                    self.ogpo_cfg.get("error_correct_sde_to_ode", True)
                ),
                clip_intermediate=bool(
                    self.ogpo_cfg.get("clip_intermediate_actions", True)
                ),
                clip_value=float(
                    self.ogpo_cfg.get("denoised_clip_value", 1.0)
                ),
            )
            next_q_heads = self.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=next_obs,
                actions=next_actions[0],
            )
            next_q = self._aggregate_q(next_q_heads)
            target_q = compute_one_step_td_target(
                rewards,
                terminations,
                next_q,
                gamma=float(self.cfg.algorithm.gamma) ** horizon,
            )

        if target_q.ndim != 1:
            raise RuntimeError(
                f"Expected TD target shape [B], got {target_q.shape}"
            )

        q_values = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=actions,
        ).float()
        td_loss, metrics = compute_q_td_loss(
            q_values,
            target_q,
            loss_type=self.ogpo_cfg.get("q_loss_type", "mse"),
            huber_delta=float(self.ogpo_cfg.get("q_huber_delta", 1.0)),
            valid_mask=valid,
        )
        metrics["critic/total_loss"] = td_loss.detach()
        return td_loss, metrics

    def _get_bc_batch(self, batch_size: int) -> dict[str, Any] | None:
        if self.replay_buffer.get_sampleable_count(success_only=True) > 0:
            batch = self._sample_sequences(batch_size, success_only=True)
            return put_tensor_device(batch, device=self.device)
        return None

    def _compute_bc_loss(
        self, batch: dict[str, Any] | None
    ) -> tuple[torch.Tensor | None, dict]:
        if batch is None:
            return None, {}
        actions = batch["actions"].reshape(
            batch["actions"].shape[0],
            int(self.cfg.actor.model.num_action_chunks),
            int(self.cfg.actor.model.action_dim),
        )
        valid_mask = (
            batch["valid"]
            .unsqueeze(-1)
            .expand_as(actions)
            .reshape(actions.shape[0], -1)
        )
        predicted_velocity, target_velocity = self.model(
            forward_type=ForwardType.OGPO_BC,
            obs=batch["curr_obs"],
            actions=actions,
        )
        return compute_flow_matching_bc_loss(
            predicted_velocity,
            target_velocity,
            valid_mask=valid_mask,
        )

    @Worker.timer("ogpo_forward_actor")
    def forward_actor(
        self,
        batch: dict[str, Any],
        *,
        bc_only: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        batch_size = batch["actions"].shape[0]
        bc_batch = self._get_bc_batch(batch_size)
        bc_loss, bc_metrics = self._compute_bc_loss(bc_batch)
        if bc_only:
            if bc_loss is None:
                return torch.zeros((), device=self.device), {}
            return bc_loss, bc_metrics

        with torch.no_grad():
            candidates, chains, old_logprobs = self.target_model(
                forward_type=ForwardType.OGPO_SAMPLE,
                obs=batch["curr_obs"],
                num_samples=int(self.cfg.algorithm.group_size),
                noise_std=float(self.ogpo_cfg.constant_noise_std),
                normalize_horizon=bool(
                    self.ogpo_cfg.get("normalize_denoising_horizon", True)
                ),
                normalize_dimension=bool(
                    self.ogpo_cfg.get("normalize_action_dimension", True)
                ),
                randn_clip_value=float(self.ogpo_cfg.get("randn_clip_value", 3.0)),
                clip_randn=bool(self.ogpo_cfg.get("clip_randn", True)),
                use_tapered_noise=bool(
                    self.ogpo_cfg.get("use_tapered_noise", False)
                ),
                ignore_last=bool(self.ogpo_cfg.get("ignore_last", True)),
                error_correct_sde_to_ode=bool(
                    self.ogpo_cfg.get("error_correct_sde_to_ode", True)
                ),
                clip_intermediate=bool(
                    self.ogpo_cfg.get("clip_intermediate_actions", True)
                ),
                clip_value=float(
                    self.ogpo_cfg.get("denoised_clip_value", 1.0)
                ),
            )
            q_ensemble = self._candidate_q(
                self.target_model, batch["curr_obs"], candidates
            ).float()
            q_values = self._aggregate_q(q_ensemble)
            advantages = compute_group_q_advantages(
                q_values=q_values,
                q_ensemble=q_ensemble,
                strategy=self.ogpo_cfg.get(
                    "advantage_q_strategy", "vanilla"
                ),
                normalize_group=bool(
                    self.ogpo_cfg.get("normalize_group", True)
                ),
                advantage_min=self.ogpo_cfg.get("advantage_min"),
                group_dim=0,
            )

        logprobs, entropy, flow_metrics = self.model(
            forward_type=ForwardType.OGPO_LOG_PROB,
            obs=batch["curr_obs"],
            chains=chains,
            noise_std=float(self.ogpo_cfg.constant_noise_std),
            normalize_horizon=bool(
                self.ogpo_cfg.get("normalize_denoising_horizon", True)
            ),
            normalize_dimension=bool(
                self.ogpo_cfg.get("normalize_action_dimension", True)
            ),
            use_tapered_noise=bool(
                self.ogpo_cfg.get("use_tapered_noise", False)
            ),
            ignore_last=bool(self.ogpo_cfg.get("ignore_last", True)),
            error_correct_sde_to_ode=bool(
                self.ogpo_cfg.get("error_correct_sde_to_ode", True)
            ),
            clip_intermediate=bool(
                self.ogpo_cfg.get("clip_intermediate_actions", True)
            ),
            clip_value=float(self.ogpo_cfg.get("denoised_clip_value", 1.0)),
        )
        loss, metrics = policy_loss(
            task_type="embodied",
            loss_type="embodied_ogpo",
            logprobs=logprobs.float(),
            old_logprobs=old_logprobs.float(),
            advantages=advantages.float(),
            loss_mask=torch.ones_like(logprobs, dtype=torch.bool),
            clip_epsilon=float(self.ogpo_cfg.get("clip_epsilon", 0.2)),
            entropy=entropy,
            entropy_coeff=float(self.ogpo_cfg.get("entropy_coeff", 0.0)),
            bc_loss=bc_loss,
            bc_coeff=float(self.ogpo_cfg.get("bc_coeff", 0.0)),
        )
        metrics.update(
            {f"actor/flow_{key}": value for key, value in flow_metrics.items()}
        )
        metrics["actor/q_mean"] = q_values.mean().detach()
        metrics["actor/advantage_mean"] = advantages.mean().detach()
        return loss, metrics

    def _micro_batch_update(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler,
        micro_batches: list[dict[str, Any]],
        forward_fn: Callable[
            [dict[str, Any]], tuple[torch.Tensor, dict[str, Any]]
        ],
        clip_grad: float,
    ) -> tuple[dict[str, Any], torch.Tensor] | None:
        total_batch_size = sum(
            int(micro_batch["actions"].shape[0])
            for micro_batch in micro_batches
        )
        self.model.zero_grad(set_to_none=True)
        metrics: dict[str, Any] = {}
        has_loss = False
        for micro_batch in micro_batches:
            loss, micro_metrics = forward_fn(micro_batch)
            if not micro_metrics:
                continue
            weight = micro_batch["actions"].shape[0] / total_batch_size
            (loss * weight).backward()
            has_loss = True
            for key, value in micro_metrics.items():
                weighted_value = value * weight
                metrics[key] = metrics.get(key, 0.0) + weighted_value
        if not has_loss:
            return None
        grad_norm = self.model.clip_grad_norm_(max_norm=clip_grad)
        optimizer.step()
        scheduler.step()
        return metrics, grad_norm

    @Worker.timer("ogpo_update_one_epoch")
    def update_one_epoch(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        use_success_buffer_q = bool(
            self.ogpo_cfg.get("use_success_buffer_q", False)
        ) and self._success_q_ready()
        append_to_dict(
            metrics,
            {"critic/success_buffer_q_active": float(use_success_buffer_q)},
        )

        for _ in range(int(self.ogpo_cfg.get("q_utd", 1))):
            micro_batches = self._sample_train_micro_batches()
            critic_update = self._micro_batch_update(
                self.qf_optimizer,
                self.qf_lr_scheduler,
                micro_batches,
                self.forward_critic,
                float(self.cfg.actor.critic_optim.clip_grad),
            )
            if critic_update is None:
                raise RuntimeError("OGPO critic update produced no loss")
            critic_metrics, critic_grad = critic_update
            critic_metrics["critic/grad_norm"] = critic_grad
            append_to_dict(metrics, critic_metrics)

            if use_success_buffer_q:
                success_micro_batches = self._sample_train_micro_batches(
                    success_only=True
                )
                success_critic_update = self._micro_batch_update(
                    self.qf_optimizer,
                    self.qf_lr_scheduler,
                    success_micro_batches,
                    self.forward_critic,
                    float(self.cfg.actor.critic_optim.clip_grad),
                )
                if success_critic_update is None:
                    raise RuntimeError(
                        "OGPO success-only critic update produced no loss"
                    )
                success_critic_metrics, success_critic_grad = (
                    success_critic_update
                )
                success_critic_metrics = {
                    f"critic/success_buffer_{key.removeprefix('critic/')}": value
                    for key, value in success_critic_metrics.items()
                }
                success_critic_metrics["critic/success_buffer_grad_norm"] = (
                    success_critic_grad
                )
                append_to_dict(metrics, success_critic_metrics)

        q_warmup_steps = int(self.ogpo_cfg.get("q_warmup_steps", 0))
        if self.update_step >= q_warmup_steps:
            for _ in range(int(self.ogpo_cfg.get("pi_utd", 1))):
                micro_batches = self._sample_train_micro_batches()
                actor_update = self._micro_batch_update(
                    self.optimizer,
                    self.lr_scheduler,
                    micro_batches,
                    self.forward_actor,
                    float(self.cfg.actor.optim.clip_grad),
                )
                if actor_update is None:
                    raise RuntimeError("OGPO actor update produced no loss")
                actor_metrics, actor_grad = actor_update
                actor_metrics["actor/grad_norm"] = actor_grad
                append_to_dict(metrics, actor_metrics)

            for _ in range(int(self.ogpo_cfg.get("bc_refine_updates", 0))):
                bc_update = self._micro_batch_update(
                    self.optimizer,
                    self.lr_scheduler,
                    micro_batches,
                    lambda micro_batch: self.forward_actor(
                        micro_batch, bc_only=True
                    ),
                    float(self.cfg.actor.optim.clip_grad),
                )
                if bc_update is not None:
                    bc_metrics, bc_grad = bc_update
                    bc_metrics["actor/bc_refine_grad_norm"] = bc_grad
                    append_to_dict(metrics, bc_metrics)

        if self.update_step % int(
            self.ogpo_cfg.get("critic_target_update_freq", 1)
        ) == 0:
            self.update_target_models(actor_tau=0.0)
        if self.update_step % int(
            self.ogpo_cfg.get("actor_target_update_freq", 1)
        ) == 0:
            self.update_target_models(critic_tau=0.0)
        return metrics

    def process_train_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        append_to_dict(
            metrics,
            {
                f"replay_buffer/{key}": value
                for key, value in self.replay_buffer.get_stats().items()
            },
        )
        averaged = {}
        for key, value in metrics.items():
            values = value if isinstance(value, list) else [value]
            numeric = [
                item.detach().float().cpu().item()
                if isinstance(item, torch.Tensor)
                else item
                for item in values
            ]
            averaged[key] = float(np.mean(numeric))
        return all_reduce_dict(averaged, op=torch.distributed.ReduceOp.AVG)

    @Worker.timer("run_training")
    def run_training(self) -> dict[str, Any]:
        if self.enable_offload:
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        min_buffer_size = int(
            self.cfg.algorithm.replay_buffer.get("min_buffer_size", 1)
        )
        if not self._replay_buffer_ready(min_buffer_size):
            self.log_on_first_rank(
                f"OGPO replay size {len(self.replay_buffer)} < "
                f"{min_buffer_size}; skipping update"
            )
            return {}

        self.model.train()
        metrics: dict[str, Any] = {}
        for _ in range(int(self.cfg.algorithm.get("update_epoch", 1))):
            epoch_metrics = self.update_one_epoch()
            for key, values in epoch_metrics.items():
                if isinstance(values, list):
                    metrics.setdefault(key, []).extend(values)
                else:
                    metrics.setdefault(key, []).append(values)
            self.update_step += 1
        result = self.process_train_metrics(metrics)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return result

    def compute_advantages_and_returns(self) -> dict:
        """OGPO computes candidate-action advantages inside actor updates."""
        return {}

    def save_checkpoint(self, save_base_path: str, step: int) -> None:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format=(
                "local_shard"
                if self.cfg.actor.fsdp_config.use_orig_params
                else "dcp"
            ),
        )

        component_path = os.path.join(save_base_path, "ogpo_components")
        target_path = os.path.join(component_path, "target_model")
        os.makedirs(target_path, exist_ok=True)
        target_state = self._strategy.get_model_state_dict(
            self.target_model, cpu_offload=False, full_state_dict=True
        )
        torch.save(
            target_state,
            os.path.join(target_path, f"checkpoint_rank_{self._rank}.pt"),
        )
        torch.save(
            {"update_step": self.update_step},
            os.path.join(component_path, f"state_rank_{self._rank}.pt"),
        )
        if self.cfg.algorithm.replay_buffer.get("save_buffer_checkpoint", True):
            self.replay_buffer.save_checkpoint(
                os.path.join(
                    component_path, "replay_buffer", f"rank_{self._rank}"
                )
            )

    def load_checkpoint(self, load_base_path: str) -> None:
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format=(
                "local_shard"
                if self.cfg.actor.fsdp_config.use_orig_params
                else "dcp"
            ),
        )
        component_path = os.path.join(load_base_path, "ogpo_components")
        target_state = torch.load(
            os.path.join(
                component_path,
                "target_model",
                f"checkpoint_rank_{self._rank}.pt",
            ),
            map_location=self.device,
        )
        self._strategy.load_model_with_state_dict(
            self.target_model,
            target_state,
            cpu_offload=False,
            full_state_dict=True,
        )
        state = torch.load(
            os.path.join(component_path, f"state_rank_{self._rank}.pt"),
            map_location="cpu",
        )
        self.update_step = int(state.get("update_step", 0))
        if self.cfg.algorithm.replay_buffer.get("save_buffer_checkpoint", True):
            buffer_path = os.path.join(
                component_path, "replay_buffer", f"rank_{self._rank}"
            )
            if os.path.isdir(buffer_path):
                self.replay_buffer.load_checkpoint(buffer_path)
            else:
                self.log_on_first_rank(
                    "Skipping replay_buffer restore: checkpoint directory does not "
                    f"exist at {buffer_path}."
                )
