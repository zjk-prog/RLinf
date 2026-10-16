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

"""Pi0 OGPO with PPO sampling/log-probabilities and executed-chunk Q/BC training."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Sequence

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_rlinf.modules import gemma
from rlinf.models.embodiment.openpi_rlinf.modules.model import (
    Observation,
    preprocess_observation,
)
from rlinf.models.embodiment.openpi_rlinf.modules.ogpo_q_head import OGPOQHead
from rlinf.models.embodiment.openpi_rlinf.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi_rlinf.tasks.rl import Pi0RL, Pi0RLConfig
from rlinf.models.embodiment.openpi_rlinf.transforms.env import repack_env_obs


def _default_critic_gemma3_config() -> dict[str, Any]:
    return {
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 3,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "max_position_embeddings": 1024,
        "attention_dropout": 0.0,
        "hidden_activation": "gelu_pytorch_tanh",
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
    }


@dataclasses.dataclass(frozen=True)
class Pi0OGPOConfig:
    train_expert_only: bool = True
    num_q_heads: int = 2
    critic_independent_state_proj: bool = False
    critic_prefix_length: int = 816
    critic_mlp_hidden_dims: tuple[int, ...] = (256, 128)
    critic_q_output_init_std: float = 1e-3
    critic_gemma3: dict[str, Any] = dataclasses.field(default_factory=dict)


class Pi0OGPO(Pi0RL):
    """PPO action distribution with OGPO advantages and independent Q heads."""

    def __init__(
        self,
        config: Pi0Config,
        *,
        rl_cfg: Pi0RLConfig,
        num_steps: int = 10,
        action_env_dim: int | None = None,
        action_chunk: int | None = None,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
        ogpo_cfg: Pi0OGPOConfig | None = None,
    ):
        if config.pi05:
            raise ValueError("Pi0 OGPO currently supports Pi0 only")
        if config.discrete_state_input:
            raise ValueError("Pi0 OGPO requires continuous state inputs")
        if not action_chunk or not 0 < action_chunk <= config.action_horizon:
            raise ValueError("OGPO requires 0 < action_chunk <= action_horizon")
        if not action_env_dim or not 0 < action_env_dim <= config.action_dim:
            raise ValueError("OGPO action_env_dim must fit the model action dimension")
        if num_steps < 2:
            raise ValueError("OGPO requires at least two denoising steps")
        super().__init__(
            config,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            action_chunk=action_chunk,
            config_name=config_name,
            state_indices=state_indices,
            rl_cfg=rl_cfg,
        )
        self.model_action_dim = self.action_dim
        self.ogpo_cfg = ogpo_cfg or Pi0OGPOConfig()
        action_width = gemma.get_config(config.action_expert_variant).width
        context_width = gemma.get_config(config.paligemma_variant).width
        critic_kwargs = _default_critic_gemma3_config()
        critic_kwargs.update(self.ogpo_cfg.critic_gemma3)
        self.q_head = OGPOQHead(
            num_q_heads=self.ogpo_cfg.num_q_heads,
            action_dim=self.action_env_dim,
            state_dim=action_width,
            state_input_proj=(
                self.state_proj if self.ogpo_cfg.critic_independent_state_proj else None
            ),
            context_dim=context_width,
            prefix_length=self.ogpo_cfg.critic_prefix_length,
            mlp_hidden_dims=self.ogpo_cfg.critic_mlp_hidden_dims,
            q_output_init_std=self.ogpo_cfg.critic_q_output_init_std,
            **critic_kwargs,
        ).float()
        self._mark_fsdp_wrap_names()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Base/legacy checkpoints have no per-Q input projections. Seed them
        # from the checkpoint's policy projection, not the random constructor.
        # Existing per-Q weights take precedence when restoring trained critics.
        if self.ogpo_cfg.critic_independent_state_proj:
            for index, network in enumerate(self.q_head.qs):
                projection_prefix = f"{prefix}q_head.qs.{index}.state_input_proj."
                projection_keys = network.state_input_proj.state_dict().keys()
                if not any(
                    projection_prefix + key in state_dict for key in projection_keys
                ):
                    for key in projection_keys:
                        source_key = f"{prefix}state_proj.{key}"
                        if source_key in state_dict:
                            state_dict[projection_prefix + key] = state_dict[
                                source_key
                            ].clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def setup_transforms(
        self, transforms: Sequence, output_transforms: Sequence
    ) -> None:
        """Reuse the policy's action transforms for replay window encoding."""
        from openpi import transforms as openpi_transforms

        super().setup_transforms(transforms, output_transforms)
        self._replay_action_transform = openpi_transforms.compose(
            [
                transform
                for transform in transforms
                if isinstance(
                    transform,
                    (openpi_transforms.DeltaActions, openpi_transforms.Normalize),
                )
            ]
        )

    def encode_replay_actions(
        self, obs: dict[str, torch.Tensor], env_actions: torch.Tensor
    ) -> torch.Tensor:
        """Encode CPU environment actions relative to each sampled window's state."""
        self._ensure_transforms()
        state = self._select_configured_state(obs["states"])
        data = {
            "state": state.detach().float().cpu().numpy(),
            # DeltaActions modifies its input; never mutate the sampled commands.
            "actions": env_actions.detach().float().cpu().numpy().copy(),
        }
        encoded = self._replay_action_transform(data)
        return torch.from_numpy(encoded["actions"]).float()

    @property
    def _no_split_modules(self) -> list[str] | None:
        # Preserve all upstream FP32/BF16 boundaries. Each Q network owns its
        # value token and projections separately from the policy's parameters.
        return [
            *(super()._no_split_modules or []),
            "Gemma3DecoderLayer",
            "OGPOQNetwork",
        ]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.ogpo_cfg.train_expert_only:
            self.img.eval()
        return self

    @staticmethod
    def _repeat_tensor(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
        return (
            tensor.unsqueeze(0)
            .expand(repeats, *tensor.shape)
            .reshape(repeats * tensor.shape[0], *tensor.shape[1:])
        )

    def _repeat_observation(
        self, observation: Observation, repeats: int
    ) -> Observation:
        return Observation(
            images={
                k: self._repeat_tensor(v, repeats)
                for k, v in observation.images.items()
            },
            image_masks={
                k: self._repeat_tensor(v, repeats)
                for k, v in observation.image_masks.items()
            },
            state=self._repeat_tensor(observation.state, repeats),
            tokenized_prompt=self._repeat_tensor(observation.tokenized_prompt, repeats)
            if observation.tokenized_prompt is not None
            else None,
            tokenized_prompt_mask=self._repeat_tensor(
                observation.tokenized_prompt_mask, repeats
            )
            if observation.tokenized_prompt_mask is not None
            else None,
            token_ar_mask=self._repeat_tensor(observation.token_ar_mask, repeats)
            if observation.token_ar_mask is not None
            else None,
            token_loss_mask=self._repeat_tensor(observation.token_loss_mask, repeats)
            if observation.token_loss_mask is not None
            else None,
            pcd_xyz=self._repeat_tensor(observation.pcd_xyz, repeats)
            if observation.pcd_xyz is not None
            else None,
        )

    def _observation(self, obs: dict[str, Any]) -> Observation:
        """Apply identical transforms to rollout and tensor-only replay observations."""
        if "main_images" in obs:
            if "task_descriptions" in obs:
                repacked = repack_env_obs(
                    self.config_name, obs, select_state=self._select_configured_state
                )
            else:
                # Pi0 prompts are task-dependent, not state-dependent. Replay
                # keeps the sampled task tokens alongside every atomic image.
                repacked = {
                    "observation/image": obs["main_images"],
                    "observation/state": self._select_configured_state(obs["states"]),
                    "tokenized_prompt": obs["tokenized_prompt"],
                    "tokenized_prompt_mask": obs["tokenized_prompt_mask"],
                }
                for source, target in (
                    ("wrist_images", "observation/wrist_image"),
                    ("extra_view_images", "observation/extra_view_image"),
                ):
                    if source in obs:
                        repacked[target] = obs[source]
        else:
            repacked = obs
        processed = self.input_transform(repacked, transpose=False)
        return preprocess_observation(
            self._observation_dict_to_device(processed), train=False
        )

    def _condition(self, observation: Observation) -> dict[str, Any]:
        if self.ogpo_cfg.train_expert_only:
            with torch.no_grad():
                prefix, mask, cache = self.build_prefix_cache(observation)
        else:
            prefix, mask, cache = self.build_prefix_cache(observation)
        # Independent critics cache the normalized raw state and project it
        # inside each Q forward, where the critic loss can train the projection.
        if self.ogpo_cfg.critic_independent_state_proj:
            state = observation.state.float()[:, None]
        else:
            with torch.no_grad():
                state = self.state_proj(observation.state.float())[:, None]
        return {
            "prefix": prefix.detach(),
            "mask": mask.detach(),
            "cache": cache,
            "state": state.detach(),
        }

    def _repeat_cache(self, cache, repeats: int):
        if isinstance(cache, tuple):
            return tuple(self._repeat_cache(x, repeats) for x in cache)
        if isinstance(cache, list):
            return [self._repeat_cache(x, repeats) for x in cache]
        if torch.is_tensor(cache):
            return self._repeat_tensor(cache, repeats)
        return cache

    def _repeat_prefix_cache(self, condition: dict[str, Any], repeats: int) -> tuple:
        return (
            self._repeat_tensor(condition["prefix"], repeats),
            self._repeat_tensor(condition["mask"], repeats),
            self._repeat_cache(condition["cache"], repeats),
        )

    def _sample(
        self,
        observation: Observation,
        *,
        num_samples: int,
        condition: dict[str, Any] | None = None,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ):
        if num_samples <= 0:
            raise ValueError("OGPO num_samples must be positive")
        batch_size = observation.state.shape[0]
        if condition is None:
            condition = self._condition(observation)
        repeated = self._repeat_observation(observation, num_samples)
        shape = (num_samples * batch_size, self.action_horizon, self.model_action_dim)
        if noise is None:
            noise = torch.randn(
                shape, device=self.device, dtype=torch.float32, generator=rng
            )
        elif tuple(noise.shape) != shape:
            raise ValueError(f"OGPO initial noise must have shape {shape}")
        else:
            noise = noise.to(device=self.device, dtype=torch.float32)
        _, result = self._predict_train(
            repeated,
            noise=noise,
            rng=rng,
            compute_values=False,
            prefix_cache=self._repeat_prefix_cache(condition, num_samples),
        )
        forward_inputs = result["forward_inputs"]
        return (
            forward_inputs["chains"].reshape(
                num_samples, batch_size, self.num_steps + 1,
                self.action_horizon, self.model_action_dim,
            ),
            forward_inputs["denoise_inds"].reshape(
                num_samples, batch_size, self.num_steps
            ),
            result["prev_logprobs"].reshape(
                num_samples, batch_size, self.action_chunk, self.action_env_dim
            ),
        )

    def ogpo_sample_forward(
        self,
        obs: dict[str, Any],
        num_samples: int,
        *,
        return_q_condition: bool = False,
        **kwargs,
    ):
        """Return candidates, chains, denoise indices, and coordinate log-probs.

        The leading axes are [N,B]. Optional Q conditions belong to this model.
        """
        observation = self._observation(obs)
        condition = self._condition(observation)
        chains, denoise_inds, log_prob = self._sample(
            observation,
            num_samples=num_samples,
            condition=condition,
            noise=kwargs.get("noise"),
            rng=kwargs.get("rng"),
        )
        final = chains[:, :, -1]
        candidates = final[:, :, : self.action_chunk, : self.action_env_dim].reshape(
            num_samples, observation.state.shape[0], -1
        )
        if return_q_condition:
            q_condition = {
                key: condition[key] for key in ("prefix", "mask", "state")
            }
            return candidates, chains, denoise_inds, log_prob, q_condition
        return candidates, chains, denoise_inds, log_prob

    def ogpo_log_prob_forward(
        self,
        obs: dict[str, Any],
        chains: torch.Tensor,
        denoise_inds: torch.Tensor,
        **kwargs,
    ):
        """Recompute PPO likelihoods at the target policy's saved denoise steps."""
        observation = self._observation(obs)
        if chains.ndim != 5:
            raise ValueError("OGPO chains must have shape [N,B,S+1,H,D]")
        n, batch_size, chain_len, horizon, action_dim = chains.shape
        if batch_size != observation.state.shape[0] or (
            chain_len, horizon, action_dim
        ) != (self.num_steps + 1, self.action_horizon, self.model_action_dim):
            raise ValueError("OGPO chain shape does not match the observation/model")
        if denoise_inds.shape != (n, batch_size, self.num_steps):
            raise ValueError("OGPO denoise_inds must have shape [N,B,S]")
        condition = self._condition(observation)
        repeated = self._repeat_observation(observation, n)
        output = self.recompute_chain_logprobs(
            repeated,
            chains.detach().reshape(n * batch_size, chain_len, horizon, action_dim),
            denoise_inds.reshape(n * batch_size, self.num_steps),
            compute_values=False,
            prefix_cache=self._repeat_prefix_cache(condition, n),
        )
        return {
            "logprobs": output["logprobs"].reshape(
                n, batch_size, self.action_chunk, self.action_env_dim
            ),
            "entropy": output["entropy"].reshape(n, batch_size, -1),
        }

    def ogpo_bc_forward(self, obs: dict[str, Any], actions: torch.Tensor, **kwargs):
        """Return full-horizon velocities; the worker masks unexecuted positions."""
        observation = self._observation(obs)
        batch_size = observation.state.shape[0]
        actions = actions.to(device=self.device, dtype=torch.float32).reshape(
            batch_size, self.action_chunk, self.action_env_dim
        )
        # The tail is computational padding, not observed future actions.
        # Loss masking preserves Pi0 attention rather than masking its inputs.
        target = torch.zeros(
            batch_size, self.action_horizon, self.model_action_dim,
            device=self.device, dtype=torch.float32,
        )
        target[:, :self.action_chunk, :self.action_env_dim] = actions
        noise = torch.randn_like(target)
        time = torch.rand(
            batch_size, 1, 1, device=self.device, dtype=torch.float32
        )
        interpolated = time * noise + (1.0 - time) * target
        condition = self._condition(observation)
        suffix = self.run_suffix(
            observation, interpolated, time[:, 0, 0],
            condition["cache"], condition["mask"],
        )
        predicted = self.velocity_from_suffix(suffix).float()
        return predicted, noise - target

    def sac_q_forward(
        self,
        obs: dict[str, Any],
        actions: torch.Tensor,
        *,
        q_condition: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        """Evaluate actions using optional conditioning from this model."""
        if q_condition is None:
            observation = self._observation(obs)
            with torch.no_grad():
                q_condition = self._condition(observation)
        condition = q_condition
        batch_size = condition["state"].shape[0]
        expected = self.action_chunk * self.action_env_dim
        candidate_shape = actions.ndim == 3
        if candidate_shape:
            n, b, d = actions.shape
            if b != batch_size or d != expected:
                raise ValueError("OGPO candidate action shape is invalid")
            actions = actions.reshape(n * b, self.action_chunk, self.action_env_dim)
            condition = {
                "prefix": self._repeat_tensor(condition["prefix"], n),
                "mask": self._repeat_tensor(condition["mask"], n),
                "state": self._repeat_tensor(condition["state"], n),
            }
        elif actions.ndim == 2:
            if tuple(actions.shape) != (batch_size, expected):
                raise ValueError("OGPO Q action shape is invalid")
            actions = actions.reshape(
                actions.shape[0], self.action_chunk, self.action_env_dim
            )
        else:
            raise ValueError("OGPO Q actions must be rank 2 or 3")
        q = self.q_head(
            actions,
            state_embedding=condition["state"],
            condition_tokens=condition["prefix"],
            condition_mask=condition["mask"],
        )
        if candidate_shape:
            return q.reshape(n, b, self.q_head.num_q_heads)
        return q

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = False,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        **kwargs,
    ):
        observation = self._observation(env_obs)
        if mode == "train":
            chains, _, log_prob = self._sample(
                observation,
                num_samples=1,
                noise=noise,
                rng=rng,
            )
            selected_model = chains[0, :, -1]
            selected_logprob = log_prob[0]
        else:
            selected_model = self.sample_actions(
                observation, num_steps=self.num_steps, noise=noise, rng=rng
            )
            selected_logprob = torch.zeros(
                observation.state.shape[0], self.action_chunk, self.action_env_dim,
                device=self.device,
            )
        actions = self.decode_actions(selected_model, observation.state)
        replay_obs = {}
        replay_obs.update(
            {
                "action": selected_model[:, : self.action_chunk, : self.action_env_dim]
                .reshape(observation.state.shape[0], -1)
                .contiguous(),
                "model_action": selected_model.reshape(
                    observation.state.shape[0], -1
                ).contiguous(),
                "tokenized_prompt": observation.tokenized_prompt.contiguous(),
                "tokenized_prompt_mask": observation.tokenized_prompt_mask.contiguous(),
            }
        )
        return actions, {
            "prev_logprobs": selected_logprob,
            "prev_values": torch.zeros(
                observation.state.shape[0], 1, device=self.device
            ),
            "forward_inputs": replay_obs,
        }

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.OGPO_SAMPLE:
            return self.ogpo_sample_forward(**kwargs)
        if forward_type == ForwardType.OGPO_LOG_PROB:
            return self.ogpo_log_prob_forward(**kwargs)
        if forward_type == ForwardType.OGPO_BC:
            return self.ogpo_bc_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        raise NotImplementedError(
            f"Unsupported OpenPI OGPO forward type: {forward_type}"
        )
