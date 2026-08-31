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

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from rlinf.algorithms.utils import aggregate_bon_q_values, select_best_of_n
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.flow_actor import (
    FlowTActor,
    JaxFlowTActor,
    OGPOFlowActor,
)
from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.models.embodiment.modules.resnet_utils import ResNetEncoder
from rlinf.models.embodiment.modules.utils import init_mlp_weights, layer_init, make_mlp
from rlinf.models.embodiment.modules.value_head import ValueHead


@dataclass
class FlowConfig:
    image_size: list[int] = field(default_factory=list)
    image_num: int = 1
    action_dim: int = 4
    state_dim: int = 29
    num_action_chunks: int = 1
    backbone: str = "resnet"
    model_path: Optional[str] = None  # used as dir actually!
    encoder_config: dict[str, Any] = field(
        default_factory=dict
    )  # 'extra_config' rename to 'encoder_config'
    add_value_head: bool = False
    add_q_head: bool = False
    q_head_type: str = "default"  # same as cnn_policy.py

    state_latent_dim: int = 64
    action_scale = None
    final_tanh = True
    std_range = None  # same as cnn_policy.py
    logstd_range = None  # same as cnn_policy.py

    num_q_heads: int = 2  # same as cnn_policy.py

    # -- Flow Matching specific parameters --##
    denoising_steps: int = 4
    d_model: int = 96
    n_head: int = 4
    n_layers: int = 2
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99
    flow_actor_type: str = "JaxFlowTActor"  # "FlowTActor" or "JaxFlowTActor"
    # Whether to use a separate head to predict noise_std
    noise_std_head: bool = False
    # Min/Max log std for training (if using noise_std_head)
    log_std_min_train: float = -5
    log_std_max_train: float = 2
    # Min/Max log std for rollout (if using noise_std_head)
    log_std_min_rollout: float = -20
    log_std_max_rollout: float = 0
    # Fixed noise std for training (if not using noise_std_head)
    noise_std_train: float = 0.3
    # Fixed noise std for rollout (if not using noise_std_head)
    noise_std_rollout: float = 0.02

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
        self._update_info()

    def _update_info(self):
        if self.add_q_head:
            if self.action_scale is None:
                self.action_scale = -1, 1
            self.final_tanh = True
            if self.backbone == "resnet":
                self.std_range = (1e-5, 5)

        assert self.model_path is not None, "Please specify the model_path."
        assert "ckpt_name" in self.encoder_config, (
            "Please specify the ckpt_name in encoder_config to load pretrained encoder weights."
        )
        ckpt_path = os.path.join(self.model_path, self.encoder_config["ckpt_name"])
        assert os.path.exists(ckpt_path), (
            f"Pretrained encoder weights not found at {ckpt_path} with model path {self.model_path} and encoder ckpt name {self.encoder_config['ckpt_name']}"
        )
        self.encoder_config["ckpt_path"] = ckpt_path


class FlowPolicy(nn.Module, BasePolicy):
    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channels = self.cfg.image_size[0]

        # Step1: Init Image encoders (same as CNNPolicy)
        self.encoders = nn.ModuleList()
        encoder_out_dim = 0
        if self.cfg.backbone == "resnet":
            sample_x = torch.randn(1, *self.cfg.image_size)
            for img_id in range(self.cfg.image_num):
                self.encoders.append(
                    ResNetEncoder(
                        sample_x, out_dim=256, encoder_cfg=self.cfg.encoder_config
                    )
                )
                encoder_out_dim += self.encoders[img_id].out_dim
        else:
            raise NotImplementedError

        if self.cfg.backbone == "resnet":
            self.state_proj = nn.Sequential(
                *make_mlp(
                    in_channels=self.cfg.state_dim,
                    mlp_channels=[
                        self.cfg.state_latent_dim,
                    ],
                    act_builder=nn.Tanh,
                    last_act=True,
                    use_layer_norm=True,
                )
            )
            init_mlp_weights(self.state_proj, nonlinearity="tanh")
            self.mix_proj = nn.Sequential(
                *make_mlp(
                    in_channels=encoder_out_dim + self.cfg.state_latent_dim,
                    mlp_channels=[256, 256],
                    act_builder=nn.Tanh,
                    last_act=True,
                    use_layer_norm=True,
                )
            )
            init_mlp_weights(self.mix_proj, nonlinearity="tanh")

        # --- Step2: Create flow actor --- #
        # FlowTActor will receive mix_feature (256 dim) as obs input
        # So we set obs_dim to 256 (output of mix_proj)
        flow_obs_dim = 256

        # Action scaling for flow actor
        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            action_scale = torch.tensor((h - l) / 2.0, dtype=torch.float32)
            action_bias = torch.tensor((h + l) / 2.0, dtype=torch.float32)
        else:
            # Default to [-1, 1] range
            action_scale = torch.ones(self.cfg.action_dim, dtype=torch.float32)
            action_bias = torch.zeros(self.cfg.action_dim, dtype=torch.float32)

        if self.cfg.flow_actor_type == "FlowTActor":
            self.flow_actor = FlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=self.cfg.action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias,
            )
        elif self.cfg.flow_actor_type == "JaxFlowTActor":
            self.flow_actor = JaxFlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=self.cfg.action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias,
                noise_std_head=self.cfg.noise_std_head,
                log_std_min_train=self.cfg.log_std_min_train,
                log_std_max_train=self.cfg.log_std_max_train,
                log_std_min_rollout=self.cfg.log_std_min_rollout,
                log_std_max_rollout=self.cfg.log_std_max_rollout,
                noise_std_train=self.cfg.noise_std_train,
                noise_std_rollout=self.cfg.noise_std_rollout,
            )
        else:
            raise ValueError(f"Unknown flow_actor_type: {self.cfg.flow_actor_type}")

        # --- Step3: Create Q-head for SAC --- #
        assert self.cfg.add_value_head + self.cfg.add_q_head <= 1
        if self.cfg.add_value_head:
            self.value_head = ValueHead(
                input_dim=256, hidden_sizes=(256, 256, 256), activation="relu"
            )
        if self.cfg.add_q_head:
            if self.cfg.backbone == "resnet":  # Now only "resnet" backbone is supported
                hidden_size = encoder_out_dim + self.cfg.state_latent_dim
                hidden_dims = [256, 256, 256]
            if self.cfg.q_head_type == "default":
                self.q_head = MultiQHead(
                    hidden_size=hidden_size,
                    hidden_dims=hidden_dims,
                    num_q_heads=self.cfg.num_q_heads,  # pass from actor.model.num_q_heads
                    action_feature_dim=self.cfg.action_dim,
                )

        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            self.register_buffer(
                "action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32)
            )
            self.register_buffer(
                "action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32)
            )
        else:
            self.action_scale = None

    @property
    def num_action_chunks(self):
        return self.cfg.num_action_chunks

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        processed_env_obs = {}
        processed_env_obs["states"] = env_obs["states"].clone().to(device)
        processed_env_obs["main_images"] = (
            env_obs["main_images"].clone().to(device).float() / 255.0
        )
        if env_obs.get("extra_view_images", None) is not None:
            processed_env_obs["extra_view_images"] = (
                env_obs["extra_view_images"].clone().to(device).float() / 255.0
            )
        return processed_env_obs

    def get_feature(self, obs):
        """Extract features from observations (images + states)"""
        visual_features = []
        # from image_keys to image_num
        for img_id in range(self.cfg.image_num):
            if img_id == 0:
                images = obs["main_images"]
            else:
                images = obs["extra_view_images"][:, img_id - 1]
            if images.shape[3] == 3:
                # [B, H, W, C] -> [B, C, H, W]
                images = images.permute(0, 3, 1, 2)
            visual_features.append(self.encoders[img_id](images))
        visual_feature = torch.cat(visual_features, dim=-1)

        state_feature = self.state_proj(obs["states"])
        full_feature = torch.cat([visual_feature, state_feature], dim=-1)

        return full_feature, visual_feature

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        obs = kwargs.get("obs")
        if obs is not None:
            obs = self.preprocess_env_obs(obs)
            kwargs.update({"obs": obs})
        next_obs = kwargs.get("next_obs", None)
        if next_obs is not None:
            next_obs = self.preprocess_env_obs(next_obs)
            kwargs.update({"next_obs": next_obs})

        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        elif forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def sac_forward(self, obs, **kwargs):
        """SAC forward pass using Flow Matching actor"""
        full_feature, visual_feature = self.get_feature(obs)
        mix_feature = self.mix_proj(full_feature)

        # Use flow actor to generate actions
        # FlowTActor expects obs as input, we pass mix_feature as the observation
        action, log_prob = self.flow_actor(mix_feature, train=True, log_grad=False)

        return action, log_prob, full_feature

    def get_q_values(self, obs, actions, shared_feature=None, detach_encoder=False):
        """Get Q-values for given observations and actions"""
        if shared_feature is None:
            shared_feature, visual_feature = self.get_feature(obs)
        if detach_encoder:
            shared_feature = shared_feature.detach()
        return self.q_head(shared_feature, actions)

    # use get_q_values() as sac_q_forward()
    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        if shared_feature is None:
            shared_feature, visual_feature = self.get_feature(obs)
        if detach_encoder:
            shared_feature = shared_feature.detach()
        return self.q_head(shared_feature, actions)

    def default_forward(
        self,
        forward_inputs,
        compute_entropy=False,
        compute_values=False,
        **kwargs,
    ):
        """Default forward pass"""

        obs = {
            "main_images": forward_inputs["main_images"],
            "states": forward_inputs["states"],
        }
        if "extra_view_images" in forward_inputs:
            obs["extra_view_images"] = forward_inputs["extra_view_images"]
        obs = self.preprocess_env_obs(obs)

        full_feature, visual_feature = self.get_feature(obs)
        mix_feature = self.mix_proj(full_feature)

        # Use flow actor
        action, log_prob = self.flow_actor(mix_feature, train=False, log_grad=False)

        output_dict = {
            "action": action,
            "log_prob": log_prob,  # key 'log_prob' or 'logprobs' as used in both cnn_policy.py??
        }

        if compute_entropy:
            # For flow matching, entropy is computed from log_prob
            # Approximate entropy as negative log_prob (this is a simplification)
            entropy = -log_prob
            output_dict.update(entropy=entropy)
        if compute_values:
            if getattr(self, "value_head", None):
                values = self.value_head(mix_feature)
                output_dict.update(values=values)
            else:
                raise NotImplementedError
        return output_dict

    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        return_shared_feature=False,
        **kwargs,
    ):
        """Predict actions in batch"""
        env_obs = self.preprocess_env_obs(env_obs)

        full_feature, visual_feature = self.get_feature(env_obs)
        mix_feature = self.mix_proj(full_feature)

        # Use flow actor
        action, log_prob = self.flow_actor(mix_feature, train=False, log_grad=False)

        # chunk_actions is always torch tensor
        chunk_actions = action.reshape(
            -1, self.cfg.num_action_chunks, self.cfg.action_dim
        )

        if hasattr(self, "value_head") and calculate_values:
            chunk_values = self.value_head(mix_feature)
        else:
            chunk_values = torch.zeros_like(log_prob[..., :1])

        forward_inputs = {"action": action}
        if return_obs:
            # x1. image indexing logic changed
            forward_inputs["main_images"] = env_obs["main_images"]
            forward_inputs["states"] = env_obs["states"]
            if "extra_view_images" in env_obs:
                forward_inputs["extra_view_images"] = env_obs["extra_view_images"]

        result = {
            "prev_logprobs": log_prob,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if return_shared_feature:
            result["shared_feature"] = visual_feature
        return chunk_actions, result


@dataclass
class FlowStateConfig:
    action_dim: int = 4
    obs_dim: int = 29
    num_action_chunks: int = 1
    encoder_config: dict[str, Any] = field(default_factory=dict)
    use_state_encoder: bool = True
    add_value_head: bool = False  # No visual_feature -> No mix_feature -> No value_head -> add_value_head must be false !
    add_q_head: bool = False
    q_head_type: str = "default"
    num_q_heads: int = 2
    q_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    q_activation: str = "tanh"
    q_layer_norm: bool = True
    q_initializer: str = "legacy"

    action_scale = None
    final_tanh = True

    # Flow Matching specific parameters
    denoising_steps: int = 4
    d_model: int = 96
    n_head: int = 4
    n_layers: int = 2
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99
    flow_actor_type: str = "JaxFlowTActor"  # Also supports OGPOFlowActor.
    flow_mlp_hidden_dims: list[int] = field(
        default_factory=lambda: [256, 256, 256]
    )
    flow_mlp_activation: str = "gelu"
    flow_mlp_layer_norm: bool = False
    time_embedding_type: str = "sinusoidal"
    time_embedding_dim: int = 32
    # Whether to use a separate head to predict noise_std
    noise_std_head: bool = False
    # Min/Max log std for training (if using noise_std_head)
    log_std_min_train: float = -5
    log_std_max_train: float = 2
    # Min log std for rollout (if using noise_std_head)
    log_std_min_rollout: float = -20
    log_std_max_rollout: float = 0
    # Fixed noise std for training (if not using noise_std_head)
    noise_std_train: float = 0.3
    # Fixed noise std for rollout (if not using noise_std_head)
    noise_std_rollout: float = 0.02

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
        self._update_info()

    def _update_info(self):
        if self.add_q_head:
            if self.action_scale is None:
                self.action_scale = -1, 1
            self.final_tanh = True


class FlowStatePolicy(nn.Module, BasePolicy):
    def __init__(self, cfg: FlowStateConfig):
        super().__init__()
        self.cfg = cfg
        if self.cfg.num_action_chunks <= 0:
            raise ValueError("num_action_chunks must be positive")

        if self.cfg.use_state_encoder:
            self.backbone = nn.Sequential(
                layer_init(nn.Linear(self.cfg.obs_dim, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
            )
            flow_obs_dim = 256
        else:
            self.backbone = nn.Identity()
            flow_obs_dim = self.cfg.obs_dim
        flow_action_dim = self.cfg.action_dim * self.cfg.num_action_chunks

        # Action scaling for flow actor
        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            action_scale = torch.full(
                (flow_action_dim,), (h - l) / 2.0, dtype=torch.float32
            )
            action_bias = torch.full(
                (flow_action_dim,), (h + l) / 2.0, dtype=torch.float32
            )
        else:
            # Default to [-1, 1] range
            action_scale = torch.ones(flow_action_dim, dtype=torch.float32)
            action_bias = torch.zeros(flow_action_dim, dtype=torch.float32)

        if self.cfg.flow_actor_type == "FlowTActor":
            self.flow_actor = FlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=flow_action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias,
            )
        elif self.cfg.flow_actor_type == "JaxFlowTActor":
            self.flow_actor = JaxFlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=flow_action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias,
                noise_std_head=self.cfg.noise_std_head,
                log_std_min_train=self.cfg.log_std_min_train,
                log_std_max_train=self.cfg.log_std_max_train,
                log_std_min_rollout=self.cfg.log_std_min_rollout,
                log_std_max_rollout=self.cfg.log_std_max_rollout,
                noise_std_train=self.cfg.noise_std_train,
                noise_std_rollout=self.cfg.noise_std_rollout,
            )
        elif self.cfg.flow_actor_type == "OGPOFlowActor":
            self.flow_actor = OGPOFlowActor(
                obs_dim=flow_obs_dim,
                action_dim=flow_action_dim,
                hidden_dims=self.cfg.flow_mlp_hidden_dims,
                activation=self.cfg.flow_mlp_activation,
                layer_norm=self.cfg.flow_mlp_layer_norm,
                time_embedding_type=self.cfg.time_embedding_type,
                time_embedding_dim=self.cfg.time_embedding_dim,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias,
                noise_std_train=self.cfg.noise_std_train,
                noise_std_rollout=self.cfg.noise_std_rollout,
            )
        else:
            raise ValueError(f"Unknown flow_actor_type: {self.cfg.flow_actor_type}")

        # Q-head for SAC
        assert self.cfg.add_value_head + self.cfg.add_q_head <= 1
        if self.cfg.add_value_head:
            self.value_head = ValueHead(
                input_dim=256, hidden_sizes=(256, 256, 256), activation="relu"
            )
        if self.cfg.add_q_head:
            self.q_head = MultiQHead(
                hidden_size=self.cfg.obs_dim,
                hidden_dims=self.cfg.q_hidden_dims,
                num_q_heads=self.cfg.num_q_heads,
                action_feature_dim=flow_action_dim,
                activation=self.cfg.q_activation,
                use_layer_norm=self.cfg.q_layer_norm,
                initializer=self.cfg.q_initializer,
            )

        self.register_buffer("action_scale", action_scale.clone())
        self.register_buffer("action_bias", action_bias.clone())

    # added num_action_chunks property
    @property
    def num_action_chunks(self):
        return self.cfg.num_action_chunks

    @property
    def flow_action_dim(self):
        return self.cfg.num_action_chunks * self.cfg.action_dim

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        return {"states": env_obs["states"].to(device)}

    def sac_forward(self, obs, **kwargs):
        """SAC forward pass using Flow Matching actor"""
        feat = self.backbone(obs["states"])

        # Use the selected flow actor to generate actions from state features.
        action, log_prob = self.flow_actor(feat, train=True, log_grad=False)

        return action, log_prob, None

    def get_q_values(self, obs, actions, shared_feature=None, detach_encoder=False):
        """Get Q-values for given observations and actions"""
        return self.q_head(obs["states"], actions)

    # use get_q_values() as sac_q_forward()
    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        return self.q_head(obs["states"], actions)

    def ogpo_sample_forward(
        self,
        obs,
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
        **kwargs,
    ):
        """Sample OGPO chains with the existing state encoder and flow actor."""
        features = self.backbone(obs["states"])
        return self.flow_actor.sample_ogpo_sde(
            features,
            num_samples=num_samples,
            noise_std=noise_std,
            normalize_horizon=normalize_horizon,
            normalize_dimension=normalize_dimension,
            randn_clip_value=randn_clip_value,
            clip_randn=clip_randn,
            use_tapered_noise=use_tapered_noise,
            ignore_last=ignore_last,
            error_correct_sde_to_ode=error_correct_sde_to_ode,
            clip_intermediate=clip_intermediate,
            clip_value=clip_value,
        )

    def ogpo_ode_forward(
        self,
        obs,
        num_samples: int = 1,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
    ):
        """Sample actions from the BC flow with deterministic ODE integration."""
        features = self.backbone(obs["states"])
        return self.flow_actor.sample_ogpo_ode(
            features,
            num_samples=num_samples,
            clip_intermediate=clip_intermediate,
            clip_value=clip_value,
        )

    def ogpo_log_prob_forward(
        self,
        obs,
        chains,
        noise_std: float,
        normalize_horizon: bool = True,
        normalize_dimension: bool = True,
        use_tapered_noise: bool = False,
        ignore_last: bool = True,
        error_correct_sde_to_ode: bool = True,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
        **kwargs,
    ):
        """Evaluate fixed target-policy chains with the online actor."""
        features = self.backbone(obs["states"])
        return self.flow_actor.ogpo_log_prob(
            features,
            chains,
            noise_std=noise_std,
            normalize_horizon=normalize_horizon,
            normalize_dimension=normalize_dimension,
            use_tapered_noise=use_tapered_noise,
            ignore_last=ignore_last,
            error_correct_sde_to_ode=error_correct_sde_to_ode,
            clip_intermediate=clip_intermediate,
            clip_value=clip_value,
        )

    def ogpo_bc_forward(self, obs, actions, **kwargs):
        """Construct flattened action-chunk flow-matching targets for BC."""
        expected_shape = (
            obs["states"].shape[0],
            self.cfg.num_action_chunks,
            self.cfg.action_dim,
        )
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"BC actions have shape {tuple(actions.shape)}; expected "
                f"{expected_shape}."
            )
        features = self.backbone(obs["states"])
        return self.flow_actor.ogpo_flow_matching(
            features, actions.reshape(actions.shape[0], -1)
        )

    def _ogpo_candidate_q(self, obs, candidates):
        num_samples, batch_size = candidates.shape[:2]
        states = (
            obs["states"]
            .unsqueeze(0)
            .expand(num_samples, *obs["states"].shape)
            .reshape(num_samples * batch_size, -1)
        )
        return self.q_head(
            states, candidates.reshape(num_samples * batch_size, self.flow_action_dim)
        ).reshape(num_samples, batch_size, -1)

    # 10. add unified forward()
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        obs = kwargs.get("obs")
        if obs is not None:
            obs = self.preprocess_env_obs(obs)
            kwargs.update({"obs": obs})
        next_obs = kwargs.get("next_obs", None)
        if next_obs is not None:
            next_obs = self.preprocess_env_obs(next_obs)
            kwargs.update({"next_obs": next_obs})

        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)  # originally exists
        elif forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)  # use get_q_values()
        elif forward_type == ForwardType.OGPO_SAMPLE:
            return self.ogpo_sample_forward(**kwargs)
        elif forward_type == ForwardType.OGPO_LOG_PROB:
            return self.ogpo_log_prob_forward(**kwargs)
        elif forward_type == ForwardType.OGPO_BC:
            return self.ogpo_bc_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)  # NOT USED (NO get_feature)
        else:
            raise NotImplementedError

    def default_forward(
        self, obs, compute_entropy=False, compute_values=False, **kwargs
    ):
        """
        Default forward pass for FlowStatePolicy.

        This method is not supported for FlowStatePolicy because it relies on features
        (e.g., get_feature, mix_proj) that are not defined for this class.
        It should not be used; kept only for compatibility.
        """
        raise NotImplementedError(
            "FlowStatePolicy.default_forward is not supported. "
            "Use FlowStatePolicy.forward with the appropriate forward_type instead."
        )

    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,  # NOT USED, unlike FlowPolicy
        return_obs=True,
        return_shared_feature=False,  # NOT USED, unlike FlowPolicy
        **kwargs,
    ):
        """
        Predict actions in batch.
        Called by MultiStepRolloutWorker for rollout
        """
        env_obs = self.preprocess_env_obs(env_obs)

        if kwargs.get("ogpo_use_sde", False):
            candidates, _, candidate_logprobs = self.ogpo_sample_forward(
                env_obs,
                num_samples=int(kwargs.get("ogpo_num_samples", 1)),
                noise_std=float(kwargs["ogpo_noise_std"]),
                normalize_horizon=bool(
                    kwargs.get("ogpo_normalize_horizon", True)
                ),
                normalize_dimension=bool(
                    kwargs.get("ogpo_normalize_dimension", True)
                ),
                randn_clip_value=float(kwargs.get("ogpo_randn_clip_value", 3.0)),
                clip_randn=bool(kwargs.get("ogpo_clip_randn", True)),
                use_tapered_noise=bool(
                    kwargs.get("ogpo_use_tapered_noise", False)
                ),
                ignore_last=bool(kwargs.get("ogpo_ignore_last", True)),
                error_correct_sde_to_ode=bool(
                    kwargs.get("ogpo_error_correct_sde_to_ode", True)
                ),
                clip_intermediate=bool(
                    kwargs.get("ogpo_clip_intermediate_actions", True)
                ),
                clip_value=float(kwargs.get("ogpo_denoised_clip_value", 1.0)),
            )
            if candidates.shape[0] == 1:
                action = candidates[0]
                log_prob = candidate_logprobs[0].unsqueeze(-1)
            else:
                q_ensemble = self._ogpo_candidate_q(env_obs, candidates)
                q_values = aggregate_bon_q_values(
                    q_ensemble,
                    subsample_size=kwargs.get("ogpo_bon_subsample_heads", 2),
                    ensemble_dim=-1,
                    generator=kwargs.get("ogpo_bon_generator"),
                    fallback_method=kwargs.get(
                        "ogpo_bon_fallback_aggregation", "min"
                    ),
                )
                action, _, best_indices = select_best_of_n(candidates, q_values)
                batch_indices = torch.arange(action.shape[0], device=action.device)
                log_prob = candidate_logprobs[
                    best_indices, batch_indices
                ].unsqueeze(-1)
        elif "ogpo_use_sde" in kwargs:
            action = self.ogpo_ode_forward(
                env_obs,
                num_samples=1,
                clip_intermediate=bool(
                    kwargs.get("ogpo_clip_intermediate_actions", True)
                ),
                clip_value=float(kwargs.get("ogpo_denoised_clip_value", 1.0)),
            )[0]
            log_prob = torch.zeros(
                (action.shape[0], 1), device=action.device, dtype=action.dtype
            )
        else:
            feat = self.backbone(env_obs["states"])
            action, log_prob = self.flow_actor(feat, train=False, log_grad=False)

        # chunk_actions is always torch tensor
        chunk_actions = action.reshape(
            -1, self.cfg.num_action_chunks, self.cfg.action_dim
        )

        chunk_values = torch.zeros_like(log_prob[..., :1])

        forward_inputs = {"action": action}
        if return_obs:
            forward_inputs["states"] = env_obs[
                "states"
            ]  # add 'states' to forward_inputs instead of 'obs/{key}'

        result = {
            "prev_logprobs": log_prob,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result
