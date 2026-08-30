# Copyright 2025 The RLinf Authors.
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

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .batch_renorm import BatchRenorm


def flow_score_from_velocity(
    velocity: torch.Tensor,
    actions: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """Recover the CondOT marginal score from a flow velocity field.

    For the conditional optimal-transport path ``alpha_t=t`` and
    ``beta_t=1-t``, the score is
    ``grad_x log p_t(x) = (t * v(x, t) - x) / (1 - t)``.
    Callers using constant noise must not evaluate this expression at ``t=1``.
    """
    return (time * velocity - actions) / (1.0 - time)


def sde_drift_correction(
    velocity: torch.Tensor,
    actions: torch.Tensor,
    time: torch.Tensor,
    sigma_base: float | torch.Tensor,
    use_tapered_noise: bool,
) -> torch.Tensor:
    """Compute the score drift that preserves the flow ODE marginals.

    The marginal-preserving SDE adds
    ``(sigma_t**2 / 2) * grad_x log p_t(x)`` to the learned flow velocity.
    With tapered noise, ``sigma_t=sigma_base*sqrt(1-t)``, the ``1-t`` term
    cancels analytically and avoids a division near the endpoint.
    """
    sigma = torch.as_tensor(
        sigma_base,
        device=velocity.device,
        dtype=velocity.dtype,
    )
    time_velocity_minus_actions = time * velocity - actions
    if use_tapered_noise:
        return 0.5 * sigma.square() * time_velocity_minus_actions
    return 0.5 * sigma.square() * flow_score_from_velocity(
        velocity,
        actions,
        time,
    )


class FlowTActor(nn.Module):
    """
    Transformer-based Flow Matching Actor for SAC
    Uses transformer architecture with cross-attention between action and observation
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        d_model=64,
        n_head=4,
        n_layers=2,
        denoising_steps=4,
        use_batch_norm=False,
        batch_norm_momentum=0.99,
        action_scale=None,
        action_bias=None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.log_std_min = -5
        self.log_std_max = 2
        self.use_batch_norm = use_batch_norm

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.d_model // 2),
            nn.SiLU(),  # SiLU is PyTorch's Swish/silu
            nn.Linear(self.d_model // 2, self.d_model),
        )

        # Action input projection (projects action_dim -> d_model)
        self.action_proj = nn.Linear(self.action_dim, self.d_model)

        # Time embedding (projects 1 -> d_model)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.d_model // 4),
            nn.SiLU(),
            nn.Linear(self.d_model // 4, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, self.d_model),
        )

        # Transformer decoder layers
        # We use nn.TransformerDecoderLayer which includes self-attn, cross-attn, and FFN
        decoder_layers = []
        for _ in range(self.n_layers):
            decoder_layers.append(
                nn.TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=self.n_head,
                    dim_feedforward=self.d_model * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
            )
        self.transformer_layers = nn.ModuleList(decoder_layers)

        # Velocity output heads
        self.velocity_mean_head = nn.Linear(self.d_model, self.action_dim)
        self.velocity_log_std_head = nn.Linear(self.d_model, self.action_dim)

        if self.use_batch_norm:
            self.bn_obs = BatchRenorm(self.obs_dim, momentum=batch_norm_momentum)
            self.bn_action = BatchRenorm(self.action_dim, momentum=batch_norm_momentum)

        # --- Action Scaling ---
        if action_scale is not None and action_bias is not None:
            self.register_buffer("action_scale", action_scale)
            self.register_buffer("action_bias", action_bias)
        else:
            # Default to [-1, 1] range
            self.register_buffer("action_scale", torch.ones(action_dim))
            self.register_buffer("action_bias", torch.zeros(action_dim))

        self._init_weights()

        self.grad_norms = {}

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, train=False, log_grad=False):
        if log_grad:
            self.grad_norms.clear()

        batch_size = obs.shape[0]
        device = obs.device

        # Flow Matching time step size
        DELTA_T = 1.0 / self.denoising_steps

        # 1. Observation encoding (memory)
        # obs: [batch_size, obs_dim] -> obs_emb: [batch_size, d_model]
        if self.use_batch_norm:
            obs = self.bn_obs(obs, train)
        obs_emb = self.obs_encoder(obs)
        # Add sequence dimension: [batch_size, 1, d_model]
        obs_emb = obs_emb.unsqueeze(1)

        x_current = torch.randn((batch_size, self.action_dim), device=device)

        # Calculate x0 log probability under N(0, I) using torch.distributions
        initial_dist = Normal(torch.zeros_like(x_current), torch.ones_like(x_current))
        total_log_prob = initial_dist.log_prob(x_current).sum(dim=1, keepdim=True)

        # 3. Flow Matching iterative refinement
        for step in range(self.denoising_steps):
            # 3a. Project current action to embedding space
            # x_current: [batch_size, action_dim] -> x_input: [batch_size, 1, action_dim]
            if self.use_batch_norm:
                x_bn = self.bn_action(x_current, train)
            else:
                x_bn = x_current
            x_input_bn = x_bn.unsqueeze(1)
            # action_emb: [batch_size, 1, d_model]
            action_emb = self.action_proj(x_input_bn)

            # 3b. Add time embedding
            time_value = torch.full(
                (batch_size, 1, 1),
                step / self.denoising_steps,
                device=device,
                dtype=torch.float32,
            )
            # time_emb: [batch_size, 1, d_model]
            time_emb = self.time_embedding(time_value)

            # 3c. Combine action and time to form query (tgt)
            # input_emb: [batch_size, 1, d_model]
            input_emb = action_emb + time_emb

            # 3d. Create diagonal mask
            # For a single query position (seq_len=1), we don't need
            # to mask future positions. A mask of 0s is fine.
            # PyTorch's mask should be (L, L) -> (1, 1)
            diagonal_mask = torch.zeros(1, 1, device=device)

            # 3e. Transformer forward pass
            output = input_emb
            for layer in self.transformer_layers:
                # tgt=output, memory=obs_emb, tgt_mask=diagonal_mask
                output = layer(output, obs_emb, tgt_mask=diagonal_mask)

            # Output is [batch_size, 1, d_model], squeeze to [batch_size, d_model]
            output = output.squeeze(1)

            # 3f. Predict velocity
            velocity_mean = self.velocity_mean_head(output)
            velocity_log_std = self.velocity_log_std_head(output)

            # Clamp log_std
            velocity_log_std = torch.tanh(velocity_log_std)
            velocity_log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
            ) * (velocity_log_std + 1)
            velocity_std = torch.exp(velocity_log_std)

            # 3g. Sample velocity
            u_dist = Normal(velocity_mean, velocity_std)
            predicted_velocity = u_dist.rsample()

            velocity_log_prob = u_dist.log_prob(predicted_velocity).sum(
                dim=-1, keepdim=True
            )
            total_log_prob += velocity_log_prob

            # 3i. Flow Matching update: x_{t+1} = x_t + v_t * Δt
            x_current = x_current + predicted_velocity * DELTA_T

            # Add gradient logging hook in style of Actor
            if log_grad:
                current_step_for_hook = step
                x_current.register_hook(
                    lambda grad, s=current_step_for_hook: self.grad_norms.update(
                        {s: grad.norm().item()}
                    )
                )

        # 4. Apply tanh transformation and scaling
        y_t = torch.tanh(x_current)
        action = y_t * self.action_scale + self.action_bias

        # 5. Add Jacobian correction for tanh
        # Use 1e-6 to match JAX implementation
        tanh_correction = torch.sum(
            torch.log(self.action_scale * (1 - y_t**2) + 1e-6), dim=-1, keepdim=True
        )
        total_log_prob -= tanh_correction

        return action, total_log_prob.detach()


class SinusoidalTimeEmbedding(nn.Module):
    """Embed a scalar flow timestep with sinusoidal features and a small MLP."""

    def __init__(self, embed_dim: int = 32, max_frequency: float = 10000.0):
        super().__init__()
        if embed_dim < 4 or embed_dim % 2 != 0:
            raise ValueError("embed_dim must be an even integer greater than or equal to 4")

        half_dim = embed_dim // 2
        frequencies = torch.exp(
            torch.arange(half_dim, dtype=torch.float32)
            * -(math.log(max_frequency) / (half_dim - 1))
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Encode timesteps shaped ``[..., 1]`` or ``[...]``."""
        if time.ndim > 0 and time.shape[-1] == 1:
            time = time.squeeze(-1)
        angles = time.unsqueeze(-1) * self.frequencies.to(dtype=time.dtype)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.projection(embedding)


class OGPOFlowMixin:
    """Transformer-independent OGPO sampling and flow-matching operations."""

    action_dim: int
    denoising_steps: int
    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def _ogpo_env_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        """Map bounded normalized actions back to the environment action space."""
        return normalized_actions.clamp(-1.0, 1.0) * self.action_scale + self.action_bias

    def _ogpo_velocity(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the actor-specific deterministic velocity field."""
        raise NotImplementedError

    @staticmethod
    def _ogpo_sde_drift(
        velocity: torch.Tensor,
        actions: torch.Tensor,
        time: torch.Tensor,
        noise_std: float,
        use_tapered_noise: bool,
        ignore_last: bool,
        error_correct_sde_to_ode: bool,
        is_last_step: bool,
    ) -> torch.Tensor:
        """Build the SDE drift shared by sampling and log-prob evaluation."""
        apply_correction = error_correct_sde_to_ode and (
            use_tapered_noise or not ignore_last or not is_last_step
        )
        if not apply_correction:
            return velocity
        return velocity + sde_drift_correction(
            velocity,
            actions,
            time,
            noise_std,
            use_tapered_noise,
        )

    @staticmethod
    def _ogpo_sde_sigma(
        actions: torch.Tensor,
        time: torch.Tensor,
        noise_std: float,
        use_tapered_noise: bool,
        ignore_last: bool,
        is_last_step: bool,
    ) -> torch.Tensor:
        """Build the noise schedule shared by sampling and log-prob evaluation."""
        if use_tapered_noise:
            taper = torch.sqrt(torch.clamp(1.0 - time, min=0.0))
            return torch.ones_like(actions) * noise_std * taper
        if is_last_step and ignore_last:
            return torch.zeros_like(actions)
        return torch.full_like(actions, noise_std)

    def _ogpo_sde_step_statistics(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        time: torch.Tensor,
        noise_std: float,
        use_tapered_noise: bool,
        ignore_last: bool,
        error_correct_sde_to_ode: bool,
        is_last_step: bool,
        clip_intermediate: bool,
        clip_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return velocity, drift, transition mean, and sigma for one SDE step."""
        velocity = self._ogpo_velocity(obs, actions, time)
        drift = self._ogpo_sde_drift(
            velocity,
            actions,
            time,
            noise_std,
            use_tapered_noise,
            ignore_last,
            error_correct_sde_to_ode,
            is_last_step,
        )
        mean_next = actions + drift / self.denoising_steps
        if clip_intermediate:
            mean_next = mean_next.clamp(-clip_value, clip_value)
        sigma = self._ogpo_sde_sigma(
            actions,
            time,
            noise_std,
            use_tapered_noise,
            ignore_last,
            is_last_step,
        )
        return velocity, drift, mean_next, sigma

    @staticmethod
    def _normalize_ogpo_statistic(
        value: torch.Tensor,
        contributing_steps: int,
        action_dim: int,
        normalize_horizon: bool,
        normalize_dimension: bool,
    ) -> torch.Tensor:
        if normalize_horizon:
            value = value / max(contributing_steps, 1)
        if normalize_dimension:
            value = value / action_dim
        return value

    def sample_ogpo_sde(
        self,
        obs: torch.Tensor,
        num_samples: int,
        noise_std: float,
        normalize_horizon: bool = True,
        normalize_dimension: bool = True,
        randn_clip_value: float = 3.0,
        clip_randn: bool = True,
        use_tapered_noise: bool = False,
        ignore_last: bool = True,
        error_correct_sde_to_ode: bool = True,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample marginal-preserving SDE chains without changing parameters."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")
        if clip_value <= 0:
            raise ValueError("clip_value must be positive")

        batch_size = obs.shape[0]
        expanded_obs = (
            obs.unsqueeze(0)
            .expand(num_samples, *obs.shape)
            .reshape(num_samples * batch_size, -1)
        )
        actions = torch.randn(
            num_samples * batch_size,
            self.action_dim,
            device=obs.device,
            dtype=obs.dtype,
        )
        chains = [actions]
        initial = Normal(torch.zeros_like(actions), torch.ones_like(actions))
        log_prob = initial.log_prob(actions).sum(dim=-1)
        for step in range(self.denoising_steps):
            is_last_step = step == self.denoising_steps - 1
            time = torch.full(
                (actions.shape[0], 1),
                step / self.denoising_steps,
                device=actions.device,
                dtype=actions.dtype,
            )
            _, _, mean_next, sigma = self._ogpo_sde_step_statistics(
                expanded_obs,
                actions,
                time,
                noise_std,
                use_tapered_noise,
                ignore_last,
                error_correct_sde_to_ode,
                is_last_step,
                clip_intermediate,
                clip_value,
            )
            deterministic_last_step = (
                is_last_step and ignore_last and not use_tapered_noise
            )
            if deterministic_last_step:
                actions = mean_next
            else:
                noise = torch.randn_like(actions)
                if clip_randn:
                    noise = noise.clamp(-randn_clip_value, randn_clip_value)
                actions = mean_next + sigma * noise
                if is_last_step:
                    actions = actions.clamp(-1.0, 1.0)
                log_prob = log_prob + Normal(mean_next, sigma).log_prob(actions).sum(
                    dim=-1
                )
            if deterministic_last_step:
                actions = actions.clamp(-1.0, 1.0)
            chains.append(actions)

        contributing_steps = self.denoising_steps + int(
            use_tapered_noise or not ignore_last
        )
        log_prob = self._normalize_ogpo_statistic(
            log_prob,
            contributing_steps,
            self.action_dim,
            normalize_horizon,
            normalize_dimension,
        )
        env_actions = self._ogpo_env_actions(actions)
        return (
            env_actions.reshape(num_samples, batch_size, self.action_dim),
            torch.stack(chains, dim=1).reshape(
                num_samples,
                batch_size,
                self.denoising_steps + 1,
                self.action_dim,
            ),
            log_prob.reshape(num_samples, batch_size),
        )

    def sample_ogpo_ode(
        self,
        obs: torch.Tensor,
        num_samples: int = 1,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
    ) -> torch.Tensor:
        """Sample actions with deterministic Euler integration after initialization."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if clip_value <= 0:
            raise ValueError("clip_value must be positive")

        batch_size = obs.shape[0]
        expanded_obs = (
            obs.unsqueeze(0)
            .expand(num_samples, *obs.shape)
            .reshape(num_samples * batch_size, -1)
        )
        actions = torch.randn(
            num_samples * batch_size,
            self.action_dim,
            device=obs.device,
            dtype=obs.dtype,
        )
        dt = 1.0 / self.denoising_steps

        for step in range(self.denoising_steps):
            time = torch.full(
                (actions.shape[0], 1),
                step / self.denoising_steps,
                device=actions.device,
                dtype=actions.dtype,
            )
            actions = actions + self._ogpo_velocity(
                expanded_obs, actions, time
            ) * dt
            if clip_intermediate:
                actions = actions.clamp(-clip_value, clip_value)

        env_actions = self._ogpo_env_actions(actions)
        return env_actions.reshape(num_samples, batch_size, self.action_dim)

    def ogpo_log_prob(
        self,
        obs: torch.Tensor,
        chains: torch.Tensor,
        noise_std: float,
        normalize_horizon: bool = True,
        normalize_dimension: bool = True,
        use_tapered_noise: bool = False,
        ignore_last: bool = True,
        error_correct_sde_to_ode: bool = True,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Recompute current-policy probability on fixed target-policy chains."""
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")
        if clip_value <= 0:
            raise ValueError("clip_value must be positive")
        if chains.ndim != 4:
            raise ValueError("chains must have shape [N, B, S + 1, A]")
        num_samples, batch_size, chain_length, action_dim = chains.shape
        if chain_length != self.denoising_steps + 1 or action_dim != self.action_dim:
            raise ValueError("chain shape does not match the flow actor")

        expanded_obs = (
            obs.unsqueeze(0)
            .expand(num_samples, *obs.shape)
            .reshape(num_samples * batch_size, -1)
        )
        flat_chains = chains.reshape(
            num_samples * batch_size, chain_length, action_dim
        )
        initial = flat_chains[:, 0]
        initial_dist = Normal(torch.zeros_like(initial), torch.ones_like(initial))
        log_prob = initial_dist.log_prob(initial).sum(dim=-1)
        entropy = initial_dist.entropy().sum(dim=-1)
        velocity_means = []
        drift_means = []
        sigma_means = []

        for step in range(self.denoising_steps):
            is_last_step = step == self.denoising_steps - 1
            actions = flat_chains[:, step]
            time = torch.full(
                (actions.shape[0], 1),
                step / self.denoising_steps,
                device=actions.device,
                dtype=actions.dtype,
            )
            velocity, drift, mean_next, sigma = self._ogpo_sde_step_statistics(
                expanded_obs,
                actions,
                time,
                noise_std,
                use_tapered_noise,
                ignore_last,
                error_correct_sde_to_ode,
                is_last_step,
                clip_intermediate,
                clip_value,
            )
            velocity_means.append(velocity.mean())
            drift_means.append(drift.mean())
            sigma_means.append(sigma.mean())
            deterministic_last_step = (
                is_last_step and ignore_last and not use_tapered_noise
            )
            if deterministic_last_step:
                continue
            transition = Normal(mean_next, sigma)
            log_prob = log_prob + transition.log_prob(
                flat_chains[:, step + 1]
            ).sum(dim=-1)
            entropy = entropy + transition.entropy().sum(dim=-1)

        contributing_steps = self.denoising_steps + int(
            use_tapered_noise or not ignore_last
        )
        log_prob = self._normalize_ogpo_statistic(
            log_prob,
            contributing_steps,
            self.action_dim,
            normalize_horizon,
            normalize_dimension,
        )
        entropy = self._normalize_ogpo_statistic(
            entropy,
            contributing_steps,
            self.action_dim,
            normalize_horizon,
            normalize_dimension,
        )
        shape = (num_samples, batch_size)
        return log_prob.reshape(shape), entropy.reshape(shape), {
            "velocity_mean": torch.stack(velocity_means).mean().detach(),
            "drift_mean": torch.stack(drift_means).mean().detach(),
            "sigma_mean": torch.stack(sigma_means).mean().detach(),
        }

    def ogpo_flow_matching(
        self,
        obs: torch.Tensor,
        env_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the bounded-action flow target used by the public OGPO actor."""
        target_actions = (
            env_actions - self.action_bias
        ) / self.action_scale.clamp_min(1e-6)
        target_actions = target_actions.clamp(-1.0, 1.0)
        noise = torch.randn_like(target_actions)
        time = torch.rand(
            target_actions.shape[0],
            1,
            device=target_actions.device,
            dtype=target_actions.dtype,
        )
        interpolated = (1.0 - time) * noise + time * target_actions
        target_velocity = target_actions - noise
        return self._ogpo_velocity(obs, interpolated, time), target_velocity


class JaxFlowTActor(nn.Module):
    """
    JAX-style Flow Matching Actor (uses noise sampling instead of distribution sampling)
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        d_model=64,
        n_head=4,
        n_layers=2,
        denoising_steps=4,
        use_batch_norm=False,
        batch_norm_momentum=0.99,
        action_scale=None,
        action_bias=None,
        noise_std_head=False,
        log_std_min_train=-5,
        log_std_max_train=2,
        log_std_min_rollout=-5,
        log_std_max_rollout=2,
        noise_std_train=0.3,
        noise_std_rollout=0.02,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.use_batch_norm = use_batch_norm
        # Whether to use fixed noise std, otherwise predict std via velocity_log_std_head
        self.noise_std_head = noise_std_head
        # Different noise std for train/rollout, smaller noise during rollout.
        self.log_std_min_train = log_std_min_train
        self.log_std_max_train = log_std_max_train
        self.log_std_min_rollout = log_std_min_rollout
        self.log_std_max_rollout = log_std_max_rollout
        # Fixed noise std added directly to actions
        self.noise_std_train = noise_std_train
        self.noise_std_rollout = noise_std_rollout

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.d_model // 2),
            nn.SiLU(),  # SiLU is PyTorch's Swish/silu
            nn.Linear(self.d_model // 2, self.d_model),
        )

        # Action input projection (projects action_dim -> d_model)
        self.action_proj = nn.Linear(self.action_dim, self.d_model)

        # Time embedding (projects 1 -> d_model)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.d_model // 4),
            nn.SiLU(),
            nn.Linear(self.d_model // 4, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, self.d_model),
        )

        # Transformer decoder layers
        # We use nn.TransformerDecoderLayer which includes self-attn, cross-attn, and FFN
        decoder_layers = []
        for _ in range(self.n_layers):
            decoder_layers.append(
                nn.TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=self.n_head,
                    dim_feedforward=self.d_model * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
            )
        self.transformer_layers = nn.ModuleList(decoder_layers)

        # Velocity output heads
        self.velocity_mean_head = nn.Linear(self.d_model, self.action_dim)
        # Use a specific head to predict velocity_log_std
        if self.noise_std_head:
            self.velocity_log_std_head = nn.Linear(self.d_model, self.action_dim)

        if self.use_batch_norm:
            self.bn_obs = BatchRenorm(self.obs_dim, momentum=batch_norm_momentum)
            self.bn_action = BatchRenorm(self.action_dim, momentum=batch_norm_momentum)

        # --- Action Scaling ---
        if action_scale is not None and action_bias is not None:
            self.register_buffer("action_scale", action_scale)
            self.register_buffer("action_bias", action_bias)
        else:
            # Default to [-1, 1] range
            self.register_buffer("action_scale", torch.ones(action_dim))
            self.register_buffer("action_bias", torch.zeros(action_dim))

        self._init_weights()

        self.grad_norms = {}

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, train=False, log_grad=False):
        if log_grad:
            self.grad_norms.clear()

        batch_size = obs.shape[0]
        device = obs.device

        # Flow Matching time step size
        DELTA_T = 1.0 / self.denoising_steps

        # 1. Observation encoding (memory)
        # obs: [batch_size, obs_dim] -> obs_emb: [batch_size, d_model]
        if self.use_batch_norm:
            obs = self.bn_obs(obs, train)
        obs_emb = self.obs_encoder(obs)
        # Add sequence dimension: [batch_size, 1, d_model]
        obs_emb = obs_emb.unsqueeze(1)

        x_current = torch.randn((batch_size, self.action_dim), device=device)

        # Calculate x0 log probability under N(0, I) using torch.distributions
        initial_dist = Normal(torch.zeros_like(x_current), torch.ones_like(x_current))
        total_log_prob = initial_dist.log_prob(x_current).sum(dim=1, keepdim=True)

        if self.noise_std_head:
            log_std_min = self.log_std_min_train if train else self.log_std_min_rollout
            log_std_max = self.log_std_max_train if train else self.log_std_max_rollout
        else:
            noise_std = self.noise_std_train if train else self.noise_std_rollout

        # 3. Flow Matching iterative refinement
        for step in range(self.denoising_steps):
            # 3a. Project current action to embedding space
            # x_current: [batch_size, action_dim] -> x_input: [batch_size, 1, action_dim]
            if self.use_batch_norm:
                x_bn = self.bn_action(x_current, train)
            else:
                x_bn = x_current
            x_input_bn = x_bn.unsqueeze(1)
            # action_emb: [batch_size, 1, d_model]
            action_emb = self.action_proj(x_input_bn)

            # 3b. Add time embedding
            time_value = torch.full(
                (batch_size, 1, 1),
                step / self.denoising_steps,
                device=device,
                dtype=torch.float32,
            )
            # time_emb: [batch_size, 1, d_model]
            time_emb = self.time_embedding(time_value)

            # 3c. Combine action and time to form query (tgt)
            # input_emb: [batch_size, 1, d_model]
            input_emb = action_emb + time_emb

            # 3d. Create diagonal mask
            # For a single query position (seq_len=1), we don't need
            # to mask future positions. A mask of 0s is fine.
            # PyTorch's mask should be (L, L) -> (1, 1)
            diagonal_mask = torch.zeros(1, 1, device=device)

            # 3e. Transformer forward pass
            output = input_emb
            for layer in self.transformer_layers:
                # tgt=output, memory=obs_emb, tgt_mask=diagonal_mask
                output = layer(output, obs_emb, tgt_mask=diagonal_mask)

            # Output is [batch_size, 1, d_model], squeeze to [batch_size, d_model]
            output = output.squeeze(1)

            # Choice A: use NN predicted velocity_log_std, add noise to velocity
            if self.noise_std_head:
                # 3f. Predict velocity
                velocity_mean = self.velocity_mean_head(output)
                velocity_log_std = self.velocity_log_std_head(output)

                # Clamp log_std
                velocity_log_std = torch.tanh(velocity_log_std)
                velocity_log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (
                    velocity_log_std + 1
                )
                velocity_std = torch.exp(velocity_log_std)

                # 3g. Sample velocity (JAX style: sample noise first, then add)
                noise_dist = Normal(0, 1)
                noise = noise_dist.rsample()

                predicted_velocity = velocity_mean + velocity_std * noise

                velocity_log_prob = noise_dist.log_prob(noise).sum(dim=-1, keepdim=True)
                total_log_prob += velocity_log_prob

                # 3i. Flow Matching update: x_{t+1} = x_t + v_t * Δt
                x_current = x_current + predicted_velocity * DELTA_T

            # Choice B: use fixed noise_std, add noise to action
            else:
                # 3f. Predict velocity
                velocity_mean = self.velocity_mean_head(output)

                # 3g. Euler step (no noise, deterministic)
                x_next_mean = x_current + velocity_mean * DELTA_T

                # 3h. Add noise to actions (not velocity)
                noise_dist = Normal(0, 1)
                noise = noise_dist.rsample((batch_size, self.action_dim)).to(device)
                x_current = x_next_mean + noise_std * noise

                # 3i. calculate log prob
                step_log_prob = (
                    Normal(x_next_mean, noise_std)
                    .log_prob(x_current)
                    .sum(dim=-1, keepdim=True)
                )
                total_log_prob += step_log_prob

            # Add gradient logging hook in style of Actor
            if log_grad:
                current_step_for_hook = step
                x_current.register_hook(
                    lambda grad, s=current_step_for_hook: self.grad_norms.update(
                        {s: grad.norm().item()}
                    )
                )

        # 4. Apply tanh transformation and scaling
        y_t = torch.tanh(x_current)
        action = y_t * self.action_scale + self.action_bias

        # 5. Add Jacobian correction for tanh
        tanh_correction = torch.sum(
            torch.log(self.action_scale * (1 - y_t**2) + 1e-6), dim=-1, keepdim=True
        )
        total_log_prob -= tanh_correction

        return action, total_log_prob


class OGPOFlowActor(OGPOFlowMixin, nn.Module):
    """OGPO-style MLP velocity field over observation, action, and time."""

    _ACTIVATIONS = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "gelu",
        layer_norm: bool = False,
        time_embedding_type: str = "sinusoidal",
        time_embedding_dim: int = 32,
        denoising_steps: int = 4,
        use_batch_norm: bool = False,
        batch_norm_momentum: float = 0.99,
        action_scale: torch.Tensor | None = None,
        action_bias: torch.Tensor | None = None,
        noise_std_train: float = 0.3,
        noise_std_rollout: float = 0.02,
    ):
        super().__init__()
        if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if denoising_steps < 1:
            raise ValueError("denoising_steps must be positive")

        activation = activation.lower()
        if activation not in self._ACTIVATIONS:
            supported = ", ".join(sorted(self._ACTIVATIONS))
            raise ValueError(f"Unsupported activation {activation!r}; choose {supported}")

        time_embedding_type = time_embedding_type.lower()
        if time_embedding_type == "scalar":
            self.time_embedding = nn.Identity()
            time_feature_dim = 1
        elif time_embedding_type == "sinusoidal":
            self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
            time_feature_dim = time_embedding_dim
        else:
            raise ValueError(
                "time_embedding_type must be either 'scalar' or 'sinusoidal'"
            )

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.use_batch_norm = use_batch_norm
        self.noise_std_train = noise_std_train
        self.noise_std_rollout = noise_std_rollout

        if self.use_batch_norm:
            self.bn_obs = BatchRenorm(self.obs_dim, momentum=batch_norm_momentum)
            self.bn_action = BatchRenorm(self.action_dim, momentum=batch_norm_momentum)

        input_dim = obs_dim + action_dim + time_feature_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._ACTIVATIONS[activation]())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.velocity_net = nn.Sequential(*layers)

        if action_scale is not None and action_bias is not None:
            self.register_buffer("action_scale", action_scale)
            self.register_buffer("action_bias", action_bias)
        else:
            self.register_buffer("action_scale", torch.ones(action_dim))
            self.register_buffer("action_bias", torch.zeros(action_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _ogpo_velocity(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_batch_norm:
            obs = self.bn_obs(obs, self.training)
            actions = self.bn_action(actions, self.training)
        time_features = self.time_embedding(time)
        inputs = torch.cat((obs, actions, time_features), dim=-1)
        return self.velocity_net(inputs)

    def forward(
        self,
        obs: torch.Tensor,
        train: bool = False,
        log_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one action while preserving the existing flow-actor interface."""
        noise_std = self.noise_std_train if train else self.noise_std_rollout
        actions, _, log_probs = self.sample_ogpo_sde(
            obs,
            num_samples=1,
            noise_std=noise_std,
            normalize_horizon=False,
            normalize_dimension=False,
        )
        return actions[0], log_probs[0].unsqueeze(-1)
