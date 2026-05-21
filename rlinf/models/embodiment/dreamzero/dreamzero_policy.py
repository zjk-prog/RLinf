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

import types
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig
from groot.vla.model.dreamzero.modules.wan2_1_submodule import sinusoidal_embedding_1d
from tianshou.data import Batch
from transformers.configuration_utils import PretrainedConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


@dataclass
class DreamZeroConfig(VLAConfig):
    model_type = "dreamzero"
    backbone_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Backbone configuration."}
    )

    action_head_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    env_action_dim: int = field(
        default=None, metadata={"help": "Environment action dimension."}
    )
    num_action_chunks: int = field(
        default=8, metadata={"help": "Number of action chunks."}
    )

    relative_action: bool = field(default=False, metadata={"help": "Relative action."})
    relative_action_per_horizon: bool = field(
        default=False, metadata={"help": "Relative action per horizon."}
    )
    relative_action_keys: list = field(
        default_factory=list, metadata={"help": "Relative action keys."}
    )

    data_transforms: ComposedModalityTransform = field(
        default=None,
        metadata={
            "help": "Transforming data modalities, e.g. video frame augmentation or action normalization."
        },
    )

    gradient_checkpointing: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class DreamZeroPolicy(VLA, BasePolicy):
    """Lightweight DreamZero action model: IdentityBackbone + WANPolicyHead."""

    _no_split_modules = [
        "CausalWanAttentionBlock",  # action head
    ]

    def __init__(
        self,
        config: DreamZeroConfig,
    ):
        super().__init__(config)
        self.config = config
        # `signed`: gripper in {-1, 1}; `zero_one`: gripper in {0, 1}.
        self._gripper_action_mode = "signed"
        self._gripper_binary_threshold = 0.0
        self._refresh_gripper_action_mode_from_metadata()
        try:
            diffusion_model = getattr(getattr(self, "action_head", None), "model", None)
            self._patch_causal_wan_model_forward_train(diffusion_model)
            enabled = self.config.gradient_checkpointing
            if diffusion_model is not None:
                if hasattr(diffusion_model, "_set_gradient_checkpointing"):
                    diffusion_model._set_gradient_checkpointing(
                        diffusion_model, enabled
                    )
                elif hasattr(diffusion_model, "gradient_checkpointing"):
                    diffusion_model.gradient_checkpointing = enabled
        except Exception:
            pass

    def apply(self, batch: Batch, **kwargs) -> Batch:
        """Normalize inputs"""
        obs = batch.obs
        normalized_input = self.config.data_transforms(obs)
        batch.normalized_obs = normalized_input
        return batch

    def unapply(self, batch: Batch, obs: Optional[dict] = None, **kwargs):
        """Unnormalize actions and convert relative actions to absolute if needed"""
        unnormalized_action = self.config.data_transforms.unapply(
            {"action": batch.normalized_action.cpu()}
        )

        # Check if relative_action is enabled and convert relative to absolute
        relative_action = self.config.relative_action
        relative_action_per_horizon = self.config.relative_action_per_horizon
        relative_action_keys = self.config.relative_action_keys
        if (
            (relative_action or relative_action_per_horizon)
            and relative_action_keys
            and obs is not None
        ):
            for key in relative_action_keys:
                action_key = f"action.{key}"
                state_key = f"state.{key}"

                if action_key not in unnormalized_action:
                    continue

                # Try to find the state data - check multiple possible key formats
                last_state = None

                # Format 1: Direct key like "state.joint_position"
                if state_key in obs:
                    last_state = obs[state_key]
                else:
                    # Format 2: Search for keys containing both "state" and the key name
                    for obs_key in obs.keys():
                        if "state" in obs_key and key in obs_key:
                            last_state = obs[obs_key]
                            break

                    # Format 3: If key is "joint_position" and obs has "state" key directly
                    # This handles cases where the observation uses modality-level keys
                    if last_state is None and "state" in obs:
                        state_data = obs["state"]
                        # Check if the state data shape matches the action shape
                        action_dim = unnormalized_action[action_key].shape[-1]
                        if torch.is_tensor(state_data):
                            state_dim = state_data.shape[-1]
                        elif isinstance(state_data, np.ndarray):
                            state_dim = state_data.shape[-1]
                        else:
                            state_dim = None

                        if state_dim == action_dim:
                            last_state = state_data

                if last_state is None:
                    continue

                if torch.is_tensor(last_state):
                    last_state = last_state.cpu().numpy()

                # Shape is (B, T, D) or (T, D), we want the last timestep
                # After indexing: (B, D) or (D,)
                if len(last_state.shape) >= 2:
                    last_state = last_state[..., -1, :]  # Get the last timestep

                # Action shape is (horizon, D) or (B, horizon, D)
                # Expand dims to broadcast: (D,) -> (1, D) or (B, D) -> (B, 1, D)
                if len(unnormalized_action[action_key].shape) > len(last_state.shape):
                    last_state = np.expand_dims(
                        last_state, axis=-2
                    )  # Add horizon dimension

                # Add state to relative action to get absolute action
                unnormalized_action[action_key] = (
                    unnormalized_action[action_key] + last_state
                )

        batch.act = unnormalized_action
        return batch

    def _process_batch(self, batch: Batch) -> Batch:
        """Process batch."""
        # Normalize / transform
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs
        # If the normalized input is still a Batch, flatten it into a pure dict
        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()
        # Do dtype cast if needed
        target_dtype = next(self.parameters()).dtype
        for k, v in normalized_input.items():
            if (
                torch.is_tensor(v)
                and v.dtype == torch.float32
                and target_dtype != torch.float32
            ):
                normalized_input[k] = v.to(dtype=target_dtype)
        return normalized_input

    def _refresh_gripper_action_mode_from_metadata(self) -> None:
        """Infer gripper action convention from checkpoint metadata statistics."""
        self._gripper_action_mode = "signed"
        self._gripper_binary_threshold = 0.0
        try:
            metadata = getattr(self.config.data_transforms, "metadata", None)
            if metadata is None:
                metadata_dict = None
            elif isinstance(metadata, dict):
                metadata_dict = metadata
            elif hasattr(metadata, "model_dump"):
                metadata_dict = metadata.model_dump()
            else:
                metadata_dict = None

            # Some metadata files are keyed by embodiment tag at the top level,
            # e.g. {"real_panda_single_arm": {"statistics": ...}}.
            if isinstance(metadata_dict, dict) and "statistics" not in metadata_dict:
                embodiment_tag = getattr(self.config, "embodiment_tag", None)
                if (
                    isinstance(embodiment_tag, str)
                    and embodiment_tag in metadata_dict
                    and isinstance(metadata_dict[embodiment_tag], dict)
                ):
                    metadata_dict = metadata_dict[embodiment_tag]
                elif len(metadata_dict) == 1:
                    only_value = next(iter(metadata_dict.values()))
                    if isinstance(only_value, dict):
                        metadata_dict = only_value

            if isinstance(metadata_dict, dict):
                action_stats = (
                    metadata_dict.get("statistics", {})
                    .get("action", {})
                    .get("actions", {})
                )
                mins = action_stats.get("min")
                maxs = action_stats.get("max")
                if mins is not None and maxs is not None:
                    mins_arr = np.asarray(mins).reshape(-1)
                    maxs_arr = np.asarray(maxs).reshape(-1)
                    if mins_arr.size > 0 and maxs_arr.size > 0:
                        min_g = float(mins_arr[-1])
                        max_g = float(maxs_arr[-1])
                        if min_g >= -1e-6 and max_g <= 1.0 + 1e-6:
                            self._gripper_action_mode = "zero_one"
                            self._gripper_binary_threshold = 0.5

            # Optional explicit override from config:
            # - "auto": use metadata inference
            # - "signed" or "zero_one": force the convention
            configured_mode = str(
                getattr(self.config, "gripper_action_mode", "auto")
            ).lower()
            if configured_mode in {"signed", "zero_one"}:
                self._gripper_action_mode = configured_mode
                self._gripper_binary_threshold = (
                    0.5 if configured_mode == "zero_one" else 0.0
                )
        except Exception:
            # Keep signed fallback to preserve backward compatibility.
            self._gripper_action_mode = "signed"
            self._gripper_binary_threshold = 0.0

    def _binarize_gripper_action(self, actions: np.ndarray) -> np.ndarray:
        """Apply checkpoint-aligned binary convention for gripper action."""
        if self._gripper_action_mode == "zero_one":
            actions[..., -1] = np.where(
                actions[..., -1] >= self._gripper_binary_threshold,
                1.0,
                0.0,
            ).astype(actions.dtype)
        else:
            actions[..., -1] = np.where(
                actions[..., -1] > self._gripper_binary_threshold,
                1.0,
                -1.0,
            ).astype(actions.dtype)
        return actions

    def _observation_convert(self, env_obs: dict) -> dict:
        """Convert environment observation to model input for end-effector control"""
        main = env_obs["main_images"]
        wrist = env_obs.get("wrist_images", None)
        if wrist is None:
            extra_views = env_obs.get("extra_view_images", None)
            if extra_views is not None:
                if torch.is_tensor(extra_views):
                    extra_views = extra_views.detach().cpu().numpy()
                else:
                    extra_views = np.asarray(extra_views)
                # RealWorld outputs [B, N, H, W, C] for multi-camera extra views.
                if extra_views.ndim == 5:
                    wrist = extra_views[:, 0]
                elif extra_views.ndim == 4:
                    wrist = extra_views
        states = env_obs.get("states", None)
        prompts = env_obs.get("task_descriptions", None)
        if torch.is_tensor(main):
            main = main.detach().cpu().numpy()
        else:
            main = np.asarray(main)
        B = main.shape[0]
        if wrist is not None:
            if torch.is_tensor(wrist):
                wrist = wrist.detach().cpu().numpy()
            else:
                wrist = np.asarray(wrist)

        expected_state_dim = None
        try:
            metadata = getattr(self.config.data_transforms, "metadata", None)
            if metadata is not None:
                if isinstance(metadata, dict):
                    shape = (
                        metadata.get("modalities", {})
                        .get("state", {})
                        .get("state", {})
                        .get("shape", None)
                    )
                    if isinstance(shape, list) and len(shape) > 0:
                        expected_state_dim = int(shape[-1])
                else:
                    modalities = getattr(metadata, "modalities", None)
                    state_mod = getattr(modalities, "state", None)
                    state_state = getattr(state_mod, "state", None)
                    shape = getattr(state_state, "shape", None)
                    if isinstance(shape, (list, tuple)) and len(shape) > 0:
                        expected_state_dim = int(shape[-1])
        except Exception:
            pass

        def _ensure_bt_hwc_uint8(x):
            arr = np.asarray(x)
            if arr.ndim == 3:
                arr = arr[None, ...]
            if arr.ndim != 4:
                raise ValueError(
                    "DreamZero expects image tensors with shape [B,H,W,C] (or [H,W,C]), "
                    f"but got shape {arr.shape}."
                )
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            return arr

        main = _ensure_bt_hwc_uint8(main)
        if wrist is not None:
            wrist = _ensure_bt_hwc_uint8(wrist)
        if main.ndim == 4:
            main = main[:, None, ...]
        if wrist is not None and wrist.ndim == 4:
            wrist = wrist[:, None, ...]
        if states is None:
            expected_shape = (
                f"[B,{expected_state_dim}]"
                if expected_state_dim is not None
                else "[B,D]"
            )
            raise ValueError(
                "DreamZero requires env_obs['states'] for policy inference, "
                f"but got None. Expected shape {expected_shape}."
            )

        if torch.is_tensor(states):
            s_np = states.detach().cpu().numpy()
        else:
            s_np = np.asarray(states)

        if s_np.ndim == 1:
            s_np = s_np[None, :]
        elif s_np.ndim == 3 and s_np.shape[1] == 1:
            s_np = s_np[:, 0, :]
        elif s_np.ndim != 2:
            raise ValueError(
                "DreamZero expects states with shape [B,D] (or [B,1,D]), "
                f"but got shape {s_np.shape}."
            )

        if s_np.shape[0] != B:
            raise ValueError(
                "DreamZero batch size mismatch between images and states: "
                f"images batch={B}, states batch={s_np.shape[0]}."
            )

        current_state_dim = s_np.shape[-1]
        if expected_state_dim is not None and current_state_dim != expected_state_dim:
            raise ValueError(
                "DreamZero state dimension mismatch: "
                f"got {current_state_dim}, expected {expected_state_dim}. "
                "Refusing to silently truncate/pad state; please ensure metadata and env state are aligned."
            )

        s_np = s_np.astype(np.float32)
        state_bt = s_np[:, None, :]
        prompts = prompts if prompts is not None else [""] * B
        if isinstance(prompts, str):
            prompts = [prompts] * B
        converted_obs = {
            "video.image": main,  # [B,H,W,C]
            "video.wrist_image": wrist,  # [B,H,W,C]
            "state.state": state_bt,  # [B,1,D]
            "annotation.language.action_text": list(prompts),  # list[str], len=B
        }
        print(f"{main.shape=}")
        print(f"{wrist.shape=}")
        print(f"{state_bt.shape=}")
        print(f"{prompts=}")
        return converted_obs

    def predict_action_batch(self, env_obs, mode, **kwargs) -> np.ndarray:
        """
        input:
            env_obs:
                - main_images: [B,H,W,C] uint8
                - extra_view_images: [B,H,W,C]
                - states: [B,D]
                - task_descriptions: list[str] or None
        output:
            actions: np.ndarray [B, num_action_chunks, 8]  # 6ee + 1 gripper
            result: dict  # compatible with rollout interface"""

        converted_obs = self._observation_convert(env_obs)
        batch = Batch(obs=converted_obs)
        # ---------- DreamZero inference ----------
        normalized_input = self._process_batch(batch)
        with torch.no_grad():
            model_pred = self.lazy_joint_video_action_causal(normalized_input)

        normalized_action = model_pred["action_pred"].float()

        # Unnormalize actions (pass obs for relative action normalization)
        unnormalized_action = self.config.data_transforms.unapply(
            {"action": normalized_action.cpu()}
        )
        batch.act = unnormalized_action

        actions = batch.act["action.actions"]
        print("----- DreamZero raw action output -----")
        print(actions)
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions[..., -1] = np.where(actions[..., -1] > 0.5, 1.0, 0).astype(
            actions.dtype
        )

        assert actions.shape[-1] == self.config.env_action_dim, (
            f"Action shape mismatch: {actions.shape} != {self.config.env_action_dim}"
        )

        flat = (
            torch.as_tensor(actions, dtype=torch.float32)
            .reshape(actions.shape[0], -1)
            .cpu()
        )
        forward_inputs = {"action": flat}
        result = {
            "prev_logprobs": torch.zeros_like(flat, dtype=torch.float32),
            "prev_values": torch.zeros((flat.shape[0], 1), dtype=torch.float32),
            "forward_inputs": forward_inputs,
        }
        print("----- DreamZero final action output -----")
        #actions = actions[:,:16,:]
        print(actions)
        return actions, result

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data=None, **kwargs):
        if data is None:
            data = kwargs.get("data")
        if data is None:
            raise ValueError("sft_forward requires `data` from the SFT dataloader.")
        outputs = super().forward(data)
        if "loss" not in outputs:
            raise ValueError("sft_forward requires `loss` in the outputs.")
        return outputs

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Default forward pass."""
        raise NotImplementedError

    def _patch_causal_wan_model_forward_train(self, model: torch.nn.Module) -> bool:
        """
        Monkey-patch DreamZero CausalWanModel._forward_train to support:
        - micro-batch (B) > 1

        Returns True if patched, False otherwise.
        """
        if model is None or not hasattr(model, "_forward_train"):
            return False

        def _forward_train_patched(
            self,
            x,
            timestep,
            timestep_action,
            context,
            seq_len,
            clean_x=None,
            aug_t=None,
            y=None,
            clip_feature=None,
            action=None,
            state=None,
            embodiment_id=None,
        ):
            # This is a minimally-edited copy of DreamZero's CausalWanModel._forward_train.
            # The only intentional behavioral change is checkpoint invocation.
            if self.model_type == "i2v":
                assert clip_feature is not None and y is not None

            if y is not None and self.concat_first_frame_latent:
                x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

            x = self.patch_embedding(x)
            grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
            freqs = self._create_freqs(
                grid_size=grid_size,
                start_frame=0,
            )

            x = x.flatten(start_dim=2).transpose(1, 2)
            assert x.shape[1] == seq_len

            B = x.shape[0]
            F = timestep.shape[1]

            if action is not None:
                embodiment_id = (
                    torch.tensor([0]).repeat(x.shape[0]).to(device=embodiment_id.device)
                )
                action_features = self.action_encoder(
                    action, timestep_action, embodiment_id
                )
                action_length = action_features.shape[1]
                state_features = self.state_encoder(state, embodiment_id)
                action_register = torch.cat([action_features, state_features], dim=1)
                action_register_length = action_register.shape[1]
                x = torch.cat([x, action_register], dim=1)
            else:
                action_features = None
                action_length = None
                state_features = None
                action_register = None
                action_register_length = None

            timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
            timestep_original = timestep.clone()

            if action is not None:
                assert timestep_action is not None
                assert state_features is not None
                stride = timestep_action.shape[1] // state_features.shape[1]
                timestep_state = timestep_action[:, ::stride]
                timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x)
            )
            e = e.unflatten(dim=0, sizes=(B, -1))
            e0 = self.time_projection(e)
            e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

            assert context.shape[1] == self.text_len
            context = self.text_embedding(context)
            if clip_feature is not None:
                clip_embedding = self.img_emb(clip_feature)
                context = torch.cat([clip_embedding, context], dim=1)

            if clean_x is not None:
                if y is not None and self.concat_first_frame_latent:
                    clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
                clean_x = self.patch_embedding(clean_x)
                clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
                assert clean_x.shape[1] == seq_len

                x = torch.cat([clean_x, x], dim=1)

                if aug_t is None:
                    aug_t = torch.zeros_like(timestep_original)

                e_clean = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x)
                )
                e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
                e0_clean = self.time_projection(e_clean)
                e0_clean = e0_clean.unflatten(dim=2, sizes=(6, self.dim))
                e0 = torch.cat([e0_clean, e0], dim=1)

            kwargs = {
                "e": e0,
                "freqs": freqs,
                "freqs_action": self.freqs_action,
                "freqs_state": self.freqs_state,
                "action_register_length": action_register_length,
                "context": context,
                "is_tf": clean_x is not None,
            }

            def create_custom_forward(module):
                def custom_forward(*inputs, **kwargs):
                    outputs, updated_kv_cache = module(*inputs, **kwargs)
                    assert updated_kv_cache is None
                    return outputs

                return custom_forward

            for block in self.blocks:
                use_ckpt = (
                    torch.is_grad_enabled()
                    and self.gradient_checkpointing
                    and not (action_register_length is not None and x.shape[0] > 1)
                )

                if use_ckpt:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        **kwargs,
                        use_reentrant=False,
                    )
                else:
                    x, _ = block(x, **kwargs)

            if clean_x is not None:
                x = x[:, clean_x.shape[1] :]

            if action is not None:
                action_noise_pred = x[:, seq_len : seq_len + action_length]
                action_noise_pred = self.action_decoder(
                    action_noise_pred, embodiment_id
                )
            else:
                action_noise_pred = None

            x_video = x[:, :seq_len]
            e_video = e[:, :seq_len]
            x_video = self.head(x_video, e_video.unsqueeze(2))
            video_noise_pred = self.unpatchify(x_video, grid_size)
            return video_noise_pred, action_noise_pred

        model._forward_train = types.MethodType(_forward_train_patched, model)
        return True
