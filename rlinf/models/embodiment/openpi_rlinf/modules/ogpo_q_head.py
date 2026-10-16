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

"""Independent OGPO Q heads conditioned on action chunks and context tokens."""

import copy
from collections.abc import Sequence

import torch
from torch import nn
from transformers import Gemma3TextConfig, Gemma3TextModel


class OGPOQNetwork(nn.Module):
    """Evaluate an action chunk with a bidirectional Gemma3 backbone."""

    def __init__(
        self,
        *,
        action_dim: int,
        state_dim: int,
        context_dim: int,
        prefix_length: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        attention_dropout: float,
        hidden_activation: str,
        initializer_range: float,
        rms_norm_eps: float,
        rope_theta: float,
        mlp_hidden_dims: Sequence[int],
        q_output_init_std: float,
        state_input_proj: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        if len(mlp_hidden_dims) != 2:
            raise ValueError("The OGPO Q-head requires exactly two hidden dimensions")

        self.prefix_length = prefix_length
        self.max_position_embeddings = max_position_embeddings
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.prefix_proj = nn.Linear(context_dim, hidden_size)
        self.state_proj = nn.Linear(state_dim, hidden_size)
        # Each Q owns its raw-state encoder; the existing projection still
        # adapts the action-expert embedding width to the critic hidden width.
        self.state_input_proj = copy.deepcopy(state_input_proj)
        self.value_token = nn.Parameter(torch.empty(1, 1, hidden_size))

        gemma3_config = Gemma3TextConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_activation=hidden_activation,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=False,
            pad_token_id=0,
            eos_token_id=0,
            bos_token_id=0,
            tie_word_embeddings=False,
            rope_theta=rope_theta,
            attention_bias=False,
            attention_dropout=attention_dropout,
            query_pre_attn_scalar=head_dim,
            sliding_window=max_position_embeddings,
            layer_types=["full_attention"] * num_hidden_layers,
        )
        self.backbone = Gemma3TextModel(gemma3_config)
        self.backbone.embed_tokens = None
        self.backbone.config._attn_implementation = "sdpa"  # noqa: SLF001

        dims = (hidden_size, *mlp_hidden_dims, 1)
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(dims[:-1], dims[1:], strict=True)
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dims) - 2:
                layers.append(nn.GELU())
        self.q_head = nn.Sequential(*layers)

        nn.init.normal_(self.value_token, mean=0.0, std=initializer_range)
        self._init_q_head_weights(q_output_init_std)

    def _init_q_head_weights(self, output_std: float) -> None:
        linear_layers = [
            module for module in self.q_head if isinstance(module, nn.Linear)
        ]
        for module in linear_layers[:-1]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.normal_(linear_layers[-1].weight, mean=0.0, std=output_std)
        nn.init.zeros_(linear_layers[-1].bias)

    @staticmethod
    def _position_ids(valid_mask: torch.Tensor) -> torch.Tensor:
        return valid_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)

    @staticmethod
    def _bidirectional_attention_mask(
        valid_mask: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        mask = torch.zeros(
            (valid_mask.shape[0], 1, 1, valid_mask.shape[1]),
            dtype=dtype,
            device=valid_mask.device,
        )
        mask = mask.masked_fill(~valid_mask[:, None, None, :], torch.finfo(dtype).min)
        return mask.expand(-1, 1, valid_mask.shape[1], -1)

    def forward(
        self,
        actions: torch.Tensor,
        *,
        state_embedding: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return one Q value while preserving gradients through actions."""
        if actions.ndim != 3 or actions.shape[2] != self.action_proj.in_features:
            raise ValueError("OGPO critic actions must have shape [B, H, A]")
        batch_size, action_chunk = actions.shape[:2]
        state_input_dim = (
            self.state_proj.in_features
            if self.state_input_proj is None
            else self.state_input_proj.in_features
        )
        expected_state_shape = (batch_size, 1, state_input_dim)
        if tuple(state_embedding.shape) != expected_state_shape:
            raise ValueError(
                "OGPO critic state embedding has shape "
                f"{tuple(state_embedding.shape)}; expected {expected_state_shape}"
            )
        expected_prefix_shape = (
            batch_size,
            self.prefix_length,
            self.prefix_proj.in_features,
        )
        if tuple(condition_tokens.shape) != expected_prefix_shape:
            raise ValueError(
                "OGPO critic prefix has shape "
                f"{tuple(condition_tokens.shape)}; expected {expected_prefix_shape}"
            )

        if condition_mask is None:
            condition_mask = torch.ones(
                (batch_size, self.prefix_length),
                dtype=torch.bool,
                device=actions.device,
            )
        elif condition_mask.shape != condition_tokens.shape[:2]:
            raise ValueError("OGPO critic prefix mask has an invalid shape")
        condition_mask = condition_mask.to(device=actions.device, dtype=torch.bool)

        network_dtype = self.action_proj.weight.dtype
        action_tokens = self.action_proj(actions.to(dtype=network_dtype))
        prefix_tokens = self.prefix_proj(
            condition_tokens.detach().to(dtype=network_dtype)
        )
        state_features = state_embedding.detach()
        if self.state_input_proj is not None:
            state_features = self.state_input_proj(
                state_features.to(dtype=self.state_input_proj.weight.dtype)
            )
        state_token = self.state_proj(state_features.to(dtype=network_dtype))
        value_token = self.value_token.expand(batch_size, -1, -1)
        input_tokens = torch.cat(
            (action_tokens, prefix_tokens, state_token, value_token), dim=1
        )
        if input_tokens.shape[1] > self.max_position_embeddings:
            raise ValueError("OGPO critic sequence exceeds max_position_embeddings")

        valid_mask = torch.cat(
            (
                torch.ones(
                    (batch_size, action_chunk),
                    dtype=torch.bool,
                    device=actions.device,
                ),
                condition_mask,
                torch.ones((batch_size, 2), dtype=torch.bool, device=actions.device),
            ),
            dim=1,
        )
        attention_mask = self._bidirectional_attention_mask(
            valid_mask, dtype=input_tokens.dtype
        )
        hidden = self.backbone(
            inputs_embeds=input_tokens,
            attention_mask=attention_mask,
            position_ids=self._position_ids(valid_mask),
            use_cache=False,
        ).last_hidden_state
        return self.q_head(hidden[:, -1])


class OGPOQHead(nn.Module):
    """Configurable independent OGPO Q networks."""

    def __init__(self, num_q_heads: int = 2, **network_kwargs) -> None:
        super().__init__()
        if num_q_heads < 1:
            raise ValueError("num_q_heads must be positive")
        self.num_q_heads = num_q_heads
        self.qs = nn.ModuleList(
            OGPOQNetwork(**network_kwargs) for _ in range(num_q_heads)
        )

    def forward(self, actions: torch.Tensor, **condition) -> torch.Tensor:
        return torch.cat([q(actions, **condition) for q in self.qs], dim=-1)
