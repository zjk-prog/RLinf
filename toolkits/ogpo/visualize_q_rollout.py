#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Save successful LIBERO actor rollouts with synchronized, per-head Q curves.

Run with the training Python environment. Configuration is discovered beside the
checkpoint in tensorboard/config.yaml. The default comparison negates every
coordinate of the normalized action chunk consumed by the trained critic.
Use --actor-checkpoint and --critic-checkpoint to score a successful actor's
actions with another checkpoint's critic. Environment settings and default
output location come from the critic checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

# Support `python toolkits/ogpo/visualize_q_rollout.py` from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def positive_int(value: str) -> int:
    """Parse a strictly positive CLI integer."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse checkpoint, successful episode count and visualization options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt", type=Path,
                        help="Shared checkpoint fallback for actor and critic")
    parser.add_argument("--actor-checkpoint", type=Path,
                        help="Checkpoint used only to generate rollout actions")
    parser.add_argument("--critic-checkpoint", type=Path,
                        help="Checkpoint providing Q heads and their feature encoder")
    parser.add_argument("--episodes", required=True, type=positive_int,
                        help="Number of SUCCESSFUL trajectory videos to save")
    parser.add_argument("--config", type=Path,
                        help="Shared saved Hydra config fallback for both models")
    parser.add_argument("--actor-config", type=Path,
                        help="Override actor config discovery")
    parser.add_argument("--critic-config", type=Path,
                        help="Override critic config discovery (also sets environment)")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: <critic-checkpoint>/q_visualizations/<timestamp>")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--critic-device",
                        help="Optional separate critic device; defaults to --device")
    parser.add_argument("--env-split", choices=("train", "eval"), default="train",
                        help="Saved environment config to reuse (default: train)")
    parser.add_argument("--mode", choices=("train", "eval"), default="train",
                        help="train uses training SDE sampling; eval uses ODE")
    parser.add_argument("--negate-space", choices=("normalized", "env"),
                        default="normalized", help="Negate critic actions directly, "
                        "or negate decoded env actions and re-encode them")
    parser.add_argument("--max-attempts", type=positive_int,
                        help="Default: 100 * episodes; fail if successes are insufficient")
    parser.add_argument("--seed", type=int, help="Default: saved actor seed")
    parser.add_argument("--fps", type=positive_int, default=20)
    args = parser.parse_args(argv)
    args.actor_checkpoint = args.actor_checkpoint or args.checkpoint
    args.critic_checkpoint = args.critic_checkpoint or args.checkpoint
    if args.actor_checkpoint is None or args.critic_checkpoint is None:
        parser.error("Provide --checkpoint, or both --actor-checkpoint and "
                     "--critic-checkpoint. Each explicit checkpoint overrides "
                     "the shared --checkpoint independently.")
    args.actor_config = args.actor_config or args.config
    args.critic_config = args.critic_config or args.config
    args.critic_device = args.critic_device or args.device
    return args


def find_config(checkpoint: Path, explicit: Path | None) -> Path:
    """Find the closest saved training configuration above a checkpoint."""
    if explicit is not None:
        return explicit.expanduser().resolve(strict=True)
    start = checkpoint if checkpoint.is_dir() else checkpoint.parent
    for parent in (start, *start.parents):
        for relative in ("tensorboard/config.yaml", ".hydra/config.yaml", "config.yaml"):
            candidate = parent / relative
            if candidate.is_file():
                return candidate
        if parent == REPO_ROOT:
            break
    raise FileNotFoundError("No saved config found above checkpoint; supply --config")


def load_policy(cfg: Any, checkpoint: Path, device: Any, *, role: str) -> Any:
    """Load all actor and critic weights strictly, retaining saved transforms."""
    import torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_rlinf import get_model
    from rlinf.models.embodiment.openpi_rlinf.checkpoint import (
        _convert_official_openpi_keys,
        _normalize_state_dict,
        resolve_full_weights,
    )
    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict
    from rlinf.utils.logging import get_logger

    logger = get_logger()
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.actor.model, resolve=True))
    if model_cfg.model_type != "openpi_rlinf" or model_cfg.openpi.task != "ogpo":
        raise ValueError("Expected actor.model.model_type=openpi_rlinf, openpi.task=ogpo")
    weights = resolve_full_weights(checkpoint)
    if weights is None:
        raise FileNotFoundError(f"No actor full_weights.pt under {checkpoint}")
    # These are the user's own training checkpoints, using the repository's format.
    loaded = torch.load(weights, map_location="cpu", weights_only=False)
    state = _convert_official_openpi_keys(
        _normalize_state_dict(as_state_dict(loaded)), weights
    )
    head_ids = sorted({int(key.split(".")[2]) for key in state
                       if key.startswith("q_head.qs.")})
    if not head_ids or head_ids != list(range(len(head_ids))):
        raise ValueError("Checkpoint must contain contiguous trained q_head.qs.N weights")
    configured = OmegaConf.select(model_cfg, "openpi.num_q_heads",
                                 default=model_cfg.get("num_q_heads", len(head_ids)))
    if int(configured) != len(head_ids):
        logger.warning("Config requests %s Q heads, checkpoint contains %s; "
                       "visualizing every saved head", configured, len(head_ids))
    # The factory reads openpi.num_q_heads; older logs stored only num_q_heads.
    model_cfg.openpi.num_q_heads = len(head_ids)
    model_cfg.num_q_heads = len(head_ids)
    # Keep original model_path for normalization assets. get_model also preserves
    # the training configuration's selective BF16/FP32 boundaries.
    policy = get_model(model_cfg)
    policy.load_state_dict(state, strict=True)
    del state, loaded
    policy.requires_grad_(False)
    policy.to(device=device)
    policy.eval()
    logger.info("Loaded %s model and all %d Q heads from %s on %s",
                role, len(head_ids), weights, device)
    return policy


def validate_model_pair(actor: Any, critic: Any) -> None:
    """Require a common action chunk shape without silently padding/truncating."""
    actor_shape = (actor.action_chunk, actor.action_env_dim)
    critic_shape = (critic.action_chunk, critic.action_env_dim)
    if actor_shape != critic_shape:
        raise ValueError(f"Actor action chunk {actor_shape} differs from critic "
                         f"action chunk {critic_shape}")


def score_actions(
    actor: Any, critic: Any, obs: dict[str, Any], decoded: Any,
    actor_action: Any, negate_space: str,
) -> tuple[Any, Any, Any]:
    """Score actor commands using the critic's own action encoding and features."""
    import torch

    if actor is critic:
        # Preserve the old single-model diagnostic exactly, including actions
        # outside quantile bounds which a decode/encode round-trip would clip.
        action = actor_action.to(device=critic.device, dtype=torch.float32)
    else:
        # Normalization assets can differ between runs. Convert the commands
        # actually sent to LIBERO into this critic's training action coordinates.
        action = critic.encode_replay_actions(obs, decoded).reshape(1, -1)
        action = action.to(device=critic.device, dtype=torch.float32)
    if negate_space == "normalized":
        negated = -action
    elif negate_space == "env":
        negated = critic.encode_replay_actions(obs, -decoded).reshape(1, -1)
        negated = negated.to(device=critic.device, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown negate space: {negate_space}")
    expected = (1, critic.action_chunk * critic.action_env_dim)
    if tuple(action.shape) != expected or tuple(negated.shape) != expected:
        raise ValueError(f"Critic encoded actions must have shape {expected}")
    if not torch.isfinite(action).all() or not torch.isfinite(negated).all():
        raise FloatingPointError("Non-finite critic action encoding")
    # Never pass the actor's cached prefix or state embedding to this call.
    # The critic's SAC_Q path builds its own image/text prefix and state_proj.
    values = critic.sac_q_forward(
        obs=obs, actions=torch.stack((action, negated), dim=0)
    )[:, 0].float().cpu().numpy()
    return action, negated, values


def make_environment(cfg: Any, split: str) -> Any:
    """Reuse training reset, image, state and success semantics in one env."""
    from omegaconf import OmegaConf

    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env[split], resolve=True))
    if env_cfg.env_type != "libero":
        raise ValueError("This visualization supports the LIBERO environment")
    env_cfg.total_num_envs = 1
    env_cfg.group_size = 1
    env_cfg.auto_reset = False
    env_cfg.ignore_terminations = False
    env_cfg.skip_intermediate_renders = False
    env_cfg.video_cfg.save_video = False
    # Use the exact saved specific_reset_id, task filters, seed, init_params and
    # reset_gripper_open. LiberoEnv also performs its training 15-step settling.
    return LiberoEnv(env_cfg, num_envs=1, seed_offset=0,
                     total_num_processes=1, worker_info=None)


def rollout(actor: Any, critic: Any, env: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Record Q(s,a) and Q(s,-a) before each chunk; execute only actor actions."""
    import numpy as np
    import torch

    obs, _ = env.reset()
    # A global reset ID selects both the task and its initial-state index.
    # Verify every attempt, including resets after failed trajectories.
    expected_reset_id = env.cfg.get("specific_reset_id", None)
    if expected_reset_id is not None:
        actual_task = int(env.task_ids[0])
        actual_trial = int(env.trial_ids[0])
        offset = int(env.cumsum_trial_id_bins[actual_task - 1]) if actual_task else 0
        if offset + actual_trial != int(expected_reset_id):
            raise RuntimeError(
                f"Reset drift: expected global reset ID {expected_reset_id}, "
                f"got task {actual_task}, trial {actual_trial}"
            )
    frames = [obs["main_images"][0].cpu().numpy().copy()]
    records = []
    step = 0
    success = False
    limit = int(env.cfg.max_episode_steps)
    while step < limit:
        with torch.inference_mode():
            decoded, result = actor.predict_action_batch(
                obs, mode=args.mode, compute_values=False
            )
            actor_action = result["forward_inputs"]["action"].float()
            if tuple(actor_action.shape) != (1, actor.action_chunk * actor.action_env_dim):
                raise ValueError(f"Unexpected actor action shape: {tuple(actor_action.shape)}")
            commands = decoded.detach().float().cpu().numpy()
            if tuple(commands.shape) != (1, actor.action_chunk, actor.action_env_dim):
                raise ValueError(f"Unexpected decoded action shape: {commands.shape}")
            if not np.isfinite(commands).all():
                raise FloatingPointError("Non-finite actor actions")
            action, negated, values = score_actions(
                actor, critic, obs, decoded, actor_action, args.negate_space
            )
        # A numerically collapsed critic is useful diagnostic output. Preserve
        # its NaN/Inf status rather than discarding an otherwise valid rollout.
        q_status = [[str(float(value)) for value in row] for row in values]
        q_serializable = [[float(value) if np.isfinite(value) else None
                           for value in row] for row in values]
        record = {
            "inference": len(records), "env_step": step,
            "q_actor": q_serializable[0], "q_negated": q_serializable[1],
            "q_value_text": q_status,
            "actor_normalized_action": actor_action[0].cpu().tolist(),
            "critic_action": action[0].cpu().tolist(),
            "critic_action_negated": negated[0].cpu().tolist(),
            "env_action_chunk": commands[0].tolist(), "executed_steps": 0,
        }
        records.append(record)
        for index in range(min(actor.action_chunk, limit - step)):
            obs, _, terminated, truncated, info = env.step(
                commands[:, index], auto_reset=False
            )
            step += 1
            record["executed_steps"] += 1
            frames.append(obs["main_images"][0].cpu().numpy().copy())
            success = bool(info["episode"]["success_once"][0].item())
            if success or bool(terminated[0].item()) or bool(truncated[0].item()):
                break
        if success or bool(terminated[0].item()) or bool(truncated[0].item()):
            break
    return {"success": success, "frames": frames, "records": records,
            "steps": step, "task_id": int(env.task_ids[0]),
            "trial_id": int(env.trial_ids[0]),
            "task_description": str(obs["task_descriptions"][0])}


def save_video(path: Path, episode: dict[str, Any], args: argparse.Namespace) -> None:
    """Stream an MP4 with rollout on the left and one paired plot per Q head."""
    import imageio.v2 as imageio
    import matplotlib
    import numpy as np
    from PIL import Image

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    records = episode["records"]
    x = np.asarray([row["env_step"] for row in records])
    actor = np.asarray([row["q_actor"] for row in records], dtype=float)
    negated = np.asarray([row["q_negated"] for row in records], dtype=float)
    heads = actor.shape[1]
    columns = min(2, heads)
    rows = math.ceil(heads / columns)
    height = max(720, rows * 180)
    width = 960
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(width / 100, height / 100), dpi=100)
    lines, cursors, titles = [], [], []
    label = "Q(s, -a_normalized)" if args.negate_space == "normalized" else "Q(s, encode(-a_env))"
    for head, ax in enumerate(axes.flat):
        if head >= heads:
            ax.set_visible(False)
            continue
        first, = ax.plot([], [], color="tab:blue", lw=1.5, marker=".", label="Q(s, a)")
        second, = ax.plot([], [], color="tab:orange", lw=1.5, ls="--", marker=".", label=label)
        finite = np.concatenate((actor[:, head], negated[:, head]))
        finite = finite[np.isfinite(finite)]
        low = finite.min() if finite.size else -1.0
        high = finite.max() if finite.size else 1.0
        margin = max((high - low) * .1, abs(high) * .01, 1e-5)
        ax.set_ylim(low - margin, high + margin)
        ax.set_xlim(0, max(1, episode["steps"]))
        ax.set_xlabel("Environment step (Q sampled once per inference)", fontsize=8)
        ax.set_ylabel("Q")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc="best")
        titles.append(ax.set_title(f"Q head {head}", fontsize=10))
        cursors.append(ax.axvline(0, color="gray", alpha=.6, lw=1))
        lines.append((first, second))
    fig.subplots_adjust(left=.08, right=.98, bottom=.06, top=.91,
                        hspace=.75, wspace=.3)
    heading = fig.suptitle("", fontsize=11)
    temp = path.with_name(path.stem + ".partial.mp4")
    try:
        with imageio.get_writer(str(temp), fps=args.fps, codec="libx264",
                                macro_block_size=2) as writer:
            for step, frame in enumerate(episode["frames"]):
                count = int(np.searchsorted(x, step, side="right"))
                current = max(0, count - 1)
                for head, (first, second) in enumerate(lines):
                    first.set_data(x[:count], actor[:count, head])
                    second.set_data(x[:count], negated[:count, head])
                    cursors[head].set_xdata([step, step])
                    raw = records[current]["q_value_text"]
                    titles[head].set_text(
                        f"Q{head}: a={float(raw[0][head]):.4g}, "
                        f"negated={float(raw[1][head]):.4g}"
                    )
                heading.set_text(
                    f"{episode['task_description']}\n"
                    f"step {step}/{episode['steps']} | inference {current} | "
                    f"{args.mode} | {'SUCCESS' if step == episode['steps'] else 'actor rollout'}"
                )
                fig.canvas.draw()
                panel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
                # Letterbox, preserving the training camera's aspect ratio.
                left = np.zeros((height, 640, 3), dtype=np.uint8)
                picture = Image.fromarray(frame).convert("RGB")
                picture.thumbnail((640, height), Image.Resampling.LANCZOS)
                image = np.asarray(picture)
                y0, x0 = (height-image.shape[0])//2, (640-image.shape[1])//2
                left[y0:y0+image.shape[0], x0:x0+image.shape[1]] = image
                writer.append_data(np.concatenate((left, panel), axis=1))
        temp.replace(path)
    finally:
        plt.close(fig)
        temp.unlink(missing_ok=True)


def main() -> None:
    """Collect exactly the requested successes, with a bounded attempt budget."""
    args = parse_args()
    from datetime import datetime

    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_rlinf.checkpoint import resolve_full_weights
    from rlinf.utils.logging import get_logger

    args.actor_checkpoint = args.actor_checkpoint.expanduser().resolve(strict=True)
    args.critic_checkpoint = args.critic_checkpoint.expanduser().resolve(strict=True)
    actor_config_path = find_config(args.actor_checkpoint, args.actor_config)
    critic_config_path = find_config(args.critic_checkpoint, args.critic_config)
    actor_cfg = OmegaConf.load(actor_config_path)
    critic_cfg = OmegaConf.load(critic_config_path)
    OmegaConf.resolve(actor_cfg)
    OmegaConf.resolve(critic_cfg)
    for key in ("env_type", "task_suite_name"):
        if actor_cfg.env[args.env_split][key] != critic_cfg.env[args.env_split][key]:
            raise ValueError(f"Actor and critic configs disagree on environment {key}")
    logger = get_logger()
    seed = int(actor_cfg.actor.get("seed", 0) if args.seed is None else args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    critic_device = torch.device(args.critic_device)
    checkpoint_dir = (
        args.critic_checkpoint if args.critic_checkpoint.is_dir()
        else args.critic_checkpoint.parent
    )
    # A full_weights.pt input still belongs to its enclosing global_step_N.
    checkpoint_dir = next(
        (parent for parent in (checkpoint_dir, *checkpoint_dir.parents)
         if parent.name.startswith("global_step_")),
        checkpoint_dir,
    )
    output = args.output_dir or (
        checkpoint_dir / "q_visualizations" /
        datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    output = output.expanduser().resolve()
    # A run directory must be new, so repeated runs cannot overwrite old videos.
    output.mkdir(parents=True, exist_ok=False)
    attempts_limit = args.max_attempts or 100 * args.episodes
    # Keep legacy fields as aliases for the model whose Q values are plotted.
    manifest = {"checkpoint": str(args.critic_checkpoint),
                "config": str(critic_config_path),
                "actor_checkpoint": str(args.actor_checkpoint),
                "critic_checkpoint": str(args.critic_checkpoint),
                "actor_config": str(actor_config_path),
                "critic_config": str(critic_config_path),
                "actor_device": str(device), "critic_device": str(critic_device),
                "environment_config_source": "critic",
                "episodes_requested": args.episodes, "seed": seed,
                "env_split": args.env_split, "mode": args.mode,
                "negate_space": args.negate_space, "fps": args.fps,
                "max_attempts": attempts_limit, "attempts": []}
    OmegaConf.save(critic_cfg, output / "training_config.yaml")
    OmegaConf.save(actor_cfg, output / "actor_training_config.yaml")
    OmegaConf.save(critic_cfg, output / "critic_training_config.yaml")
    actor = load_policy(actor_cfg, args.actor_checkpoint, device, role="actor")
    actor_weights = resolve_full_weights(args.actor_checkpoint)
    critic_weights = resolve_full_weights(args.critic_checkpoint)
    same_model = (
        actor_weights is not None and critic_weights is not None
        and actor_weights.resolve() == critic_weights.resolve()
        and OmegaConf.to_container(actor_cfg.actor.model, resolve=True)
        == OmegaConf.to_container(critic_cfg.actor.model, resolve=True)
        and args.device == args.critic_device
    )
    critic = actor if same_model else load_policy(
        critic_cfg, args.critic_checkpoint, critic_device, role="critic"
    )
    validate_model_pair(actor, critic)
    manifest["num_q_heads"] = critic.q_head.num_q_heads
    manifest["critic_action_source"] = (
        "actor_normalized" if same_model else "critic_encoded_env_action"
    )
    logger.info("Rollout actor=%s; scoring critic=%s; environment config=%s",
                args.actor_checkpoint, args.critic_checkpoint, critic_config_path)
    env = make_environment(critic_cfg, args.env_split)
    successes = 0
    try:
        OmegaConf.save(env.cfg, output / "visualization_env_config.yaml")
        for attempt in range(1, attempts_limit + 1):
            episode = rollout(actor, critic, env, args)
            entry = {key: episode[key] for key in
                     ("success", "steps", "task_id", "trial_id", "task_description")}
            entry["attempt"] = attempt
            if episode["success"]:
                name = (f"episode_{successes+1:04d}_task{episode['task_id']:02d}"
                        f"_trial{episode['trial_id']:03d}")
                video = output / f"{name}.mp4"
                save_video(video, episode, args)
                data = {**entry, "negate_space": args.negate_space,
                        "actor_checkpoint": str(args.actor_checkpoint),
                        "critic_checkpoint": str(args.critic_checkpoint),
                        "critic_action_source": manifest["critic_action_source"],
                        "records": episode["records"]}
                (output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
                entry["video"] = video.name
                successes += 1
            manifest["attempts"].append(entry)
            manifest["episodes_saved"] = successes
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            logger.info("Attempt %d/%d: success=%s, saved %d/%d, output=%s",
                        attempt, attempts_limit, episode["success"],
                        successes, args.episodes, output)
            if successes == args.episodes:
                break
    finally:
        # LiberoEnv inherits gym.Env.close (a no-op); close its subprocess owner.
        env.env.close()
    if successes < args.episodes:
        raise RuntimeError(f"Saved {successes}/{args.episodes} successful videos after "
                           f"{attempts_limit} attempts. Existing results: {output}")


if __name__ == "__main__":
    main()
