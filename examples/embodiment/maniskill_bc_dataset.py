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

"""State-action dataset for the standalone ManiSkill BC entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class ManiSkillBCDataset(Dataset):
    """Load state and action-chunk windows from ManiSkill trajectories."""

    def __init__(
        self,
        trajectory_path: str | Path,
        *,
        obs_dim: int,
        action_dim: int,
        num_action_chunks: int = 1,
        expected_env_id: str = "PickCube-v1",
        expected_control_mode: str = "pd_ee_delta_pos",
        success_only: bool = True,
    ) -> None:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional embodied dependency
            raise ImportError(
                "ManiSkillBCDataset requires h5py. Install the embodied dependencies."
            ) from exc

        if num_action_chunks <= 0:
            raise ValueError("num_action_chunks must be positive.")

        self.trajectory_path = Path(trajectory_path).expanduser().resolve()
        self.num_action_chunks = num_action_chunks
        if not self.trajectory_path.is_file():
            raise FileNotFoundError(
                f"ManiSkill trajectory file does not exist: {self.trajectory_path}"
            )

        metadata = self._load_metadata()
        env_info = metadata.get("env_info", {})
        env_id = env_info.get("env_id")
        if env_id is not None and env_id != expected_env_id:
            raise ValueError(
                f"Expected ManiSkill env_id={expected_env_id!r}, got {env_id!r}."
            )
        env_control_mode = env_info.get("env_kwargs", {}).get("control_mode")
        if (
            env_control_mode is not None
            and env_control_mode != expected_control_mode
        ):
            raise ValueError(
                f"ManiSkill metadata uses control_mode={env_control_mode!r}; "
                f"expected {expected_control_mode!r}."
            )

        episode_metadata = {
            int(episode["episode_id"]): episode
            for episode in metadata.get("episodes", [])
            if "episode_id" in episode
        }
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        kept_episodes = 0
        total_episodes = 0

        with h5py.File(self.trajectory_path, "r") as trajectory_file:
            trajectory_names = sorted(
                (name for name in trajectory_file if name.startswith("traj_")),
                key=self._trajectory_id,
            )
            if not trajectory_names:
                raise ValueError(
                    f"No traj_<episode_id> groups found in {self.trajectory_path}."
                )

            for trajectory_name in trajectory_names:
                total_episodes += 1
                episode_id = self._trajectory_id(trajectory_name)
                trajectory = trajectory_file[trajectory_name]
                episode_info = episode_metadata.get(episode_id, {})
                control_mode = episode_info.get("control_mode")
                if (
                    control_mode is not None
                    and control_mode != expected_control_mode
                ):
                    raise ValueError(
                        f"Trajectory {trajectory_name} uses control_mode="
                        f"{control_mode!r}; expected {expected_control_mode!r}."
                    )

                if success_only and not self._episode_succeeded(
                    trajectory, episode_info
                ):
                    continue
                if "obs" not in trajectory or "actions" not in trajectory:
                    raise ValueError(
                        f"Trajectory {trajectory_name} must contain obs and actions. "
                        "Replay the downloaded demo with '-o state --save-traj'."
                    )
                if not isinstance(trajectory["obs"], h5py.Dataset):
                    raise ValueError(
                        f"Trajectory {trajectory_name}/obs is not a flat state dataset. "
                        "Replay the demo with '-o state'."
                    )

                episode_states = np.asarray(trajectory["obs"], dtype=np.float32)
                episode_actions = np.asarray(
                    trajectory["actions"], dtype=np.float32
                )
                if episode_states.ndim != 2 or episode_states.shape[1] != obs_dim:
                    raise ValueError(
                        f"Trajectory {trajectory_name} has state shape "
                        f"{episode_states.shape}; expected [T + 1, {obs_dim}]."
                    )
                if (
                    episode_actions.ndim != 2
                    or episode_actions.shape[1] != action_dim
                ):
                    raise ValueError(
                        f"Trajectory {trajectory_name} has action shape "
                        f"{episode_actions.shape}; expected [T, {action_dim}]."
                    )
                if episode_states.shape[0] != episode_actions.shape[0] + 1:
                    raise ValueError(
                        f"Trajectory {trajectory_name} must contain one more state "
                        "than action."
                    )

                num_windows = episode_actions.shape[0] - num_action_chunks + 1
                if num_windows <= 0:
                    continue

                states.append(episode_states[:num_windows])
                actions.append(
                    np.stack(
                        [
                            episode_actions[start : start + num_action_chunks]
                            for start in range(num_windows)
                        ],
                        axis=0,
                    )
                )
                kept_episodes += 1

        if not states:
            qualifier = " successful" if success_only else ""
            raise ValueError(
                f"No{qualifier} action chunks of length {num_action_chunks} "
                f"found in {self.trajectory_path}."
            )

        self.states = torch.from_numpy(np.concatenate(states, axis=0))
        self.actions = torch.from_numpy(np.concatenate(actions, axis=0))
        self.total_episodes = total_episodes
        self.kept_episodes = kept_episodes

    @staticmethod
    def _trajectory_id(name: str) -> int:
        try:
            return int(name.removeprefix("traj_"))
        except ValueError as exc:
            raise ValueError(f"Invalid ManiSkill trajectory group name: {name!r}") from exc

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self.trajectory_path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"ManiSkill metadata file must accompany the trajectory: {metadata_path}"
            )
        with metadata_path.open(encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    @staticmethod
    def _episode_succeeded(trajectory: Any, episode_info: dict[str, Any]) -> bool:
        if "success" in trajectory:
            return bool(np.asarray(trajectory["success"]).any())
        final_info = episode_info.get("info", {})
        for key in ("success_once", "success_at_end", "success"):
            if key in final_info:
                return bool(np.asarray(final_info[key]).any())
        # Official motion-planning demonstrations do not always store success labels.
        return True

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"states": self.states[index], "actions": self.actions[index]}
