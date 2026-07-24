#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import Tensor, nn

from lerobot.policies.diffusion_jepa.configuration_diffusion_jepa import DiffusionJEPAConfig


def build_action_block_causal_attention_mask(
    num_steps: int,
    tokens_per_state: int,
    num_action_tokens: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Return a block-causal mask with full attention inside each time step.

    PyTorch attention uses ``True`` for disallowed entries. Tokens at time ``t``
    can therefore attend to all action/state tokens at times ``<= t``.
    """
    tokens_per_step = tokens_per_state + num_action_tokens
    time_ids = torch.arange(num_steps, device=device).repeat_interleave(tokens_per_step)
    return time_ids.unsqueeze(0) > time_ids.unsqueeze(1)


class LeWorldModelPredictor(nn.Module):
    """Token-level action-conditioned causal world model.

    Each time block interleaves ``K`` latent-action tokens with all V-JEPA2
    spatial state tokens. This mirrors the attention structure in VLA-JEPA while
    keeping the implementation native to LeRobot.
    """

    def __init__(
        self,
        config: DiffusionJEPAConfig,
        *,
        state_dim: int,
        num_steps: int,
        tokens_per_state: int,
    ):
        super().__init__()
        predictor_dim = config.jepa_latent_dim
        self.state_dim = state_dim
        self.num_steps = num_steps
        self.tokens_per_state = tokens_per_state
        self.num_action_tokens = config.jepa_action_tokens_per_transition

        self.state_encoder = nn.Linear(state_dim, predictor_dim)
        self.action_encoder = nn.Linear(config.jepa_action_latent_dim, predictor_dim)
        self.time_embedding = nn.Parameter(torch.zeros(1, num_steps, 1, predictor_dim))
        self.state_position_embedding = nn.Parameter(
            torch.zeros(1, 1, tokens_per_state, predictor_dim)
        )
        self.action_position_embedding = nn.Parameter(
            torch.zeros(1, 1, self.num_action_tokens, predictor_dim)
        )
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.state_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.action_position_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=config.jepa_predictor_heads,
            dim_feedforward=int(predictor_dim * config.jepa_predictor_mlp_ratio),
            dropout=config.jepa_predictor_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer,
            num_layers=config.jepa_predictor_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(predictor_dim)
        self.state_projector = nn.Linear(predictor_dim, state_dim)

        causal_mask = build_action_block_causal_attention_mask(
            num_steps,
            tokens_per_state,
            self.num_action_tokens,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, states: Tensor, latent_actions: Tensor) -> Tensor:
        """Predict the next V-JEPA2 state tokens under teacher forcing.

        Args:
            states: Current state tokens ``(B, T, P, D_state)``.
            latent_actions: Latent-action tokens ``(B, T, K, D_action)``.
        Returns:
            Predicted next-state tokens ``(B, T, P, D_state)``.
        """
        if states.ndim != 4 or latent_actions.ndim != 4:
            raise ValueError(
                "World model expects states (B, T, P, D) and latent actions (B, T, K, D)."
            )
        batch_size, num_steps, tokens_per_state, state_dim = states.shape
        if (num_steps, tokens_per_state, state_dim) != (
            self.num_steps,
            self.tokens_per_state,
            self.state_dim,
        ):
            raise ValueError(
                "Unexpected world-state token shape. "
                f"Expected (*, {self.num_steps}, {self.tokens_per_state}, {self.state_dim}), "
                f"got {tuple(states.shape)}."
            )
        expected_action_prefix = (batch_size, self.num_steps, self.num_action_tokens)
        if latent_actions.shape[:3] != expected_action_prefix:
            raise ValueError(
                f"Expected latent-action prefix {expected_action_prefix}, "
                f"got {tuple(latent_actions.shape[:3])}."
            )

        time_embedding = self.time_embedding[:, :num_steps]
        action_tokens = (
            self.action_encoder(latent_actions)
            + self.action_position_embedding
            + time_embedding
        )
        state_tokens = (
            self.state_encoder(states)
            + self.state_position_embedding
            + time_embedding
        )
        tokens = torch.cat([action_tokens, state_tokens], dim=2).flatten(1, 2)
        tokens = self.blocks(tokens, mask=self.causal_mask)
        tokens = tokens.view(
            batch_size,
            num_steps,
            self.num_action_tokens + self.tokens_per_state,
            -1,
        )
        predicted_states = self.final_norm(tokens[:, :, self.num_action_tokens :])
        return self.state_projector(predicted_states)
