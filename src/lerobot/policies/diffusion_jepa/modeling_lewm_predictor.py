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
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.diffusion_jepa.configuration_diffusion_jepa import DiffusionJEPAConfig


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def _safe_batch_norm(norm: nn.BatchNorm1d, x: Tensor) -> Tensor:
    """Apply BatchNorm while keeping single-item smoke tests well-defined."""
    if norm.training and x.shape[0] == 1:
        return F.batch_norm(
            x,
            norm.running_mean,
            norm.running_var,
            norm.weight,
            norm.bias,
            training=False,
            momentum=0.0,
            eps=norm.eps,
        )
    return norm(x)


def _stepwise_batch_norm(norm: nn.BatchNorm1d, x: Tensor) -> Tensor:
    """Normalize every time index over the batch without leaking future tokens."""
    if x.ndim != 3:
        raise ValueError("Step-wise BatchNorm expects a (B, T, D) tensor.")
    return torch.stack([_safe_batch_norm(norm, x[:, step]) for step in range(x.shape[1])], dim=1)


class ActionAdaLNBlock(nn.Module):
    """Causal Transformer block conditioned on Diffusion latent actions."""

    def __init__(self, config: DiffusionJEPAConfig):
        super().__init__()
        dim = config.jepa_latent_dim
        action_dim = config.jepa_action_latent_dim
        mlp_dim = int(dim * config.jepa_predictor_mlp_ratio)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            dim,
            config.jepa_predictor_heads,
            dropout=config.jepa_predictor_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.jepa_predictor_dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(config.jepa_predictor_dropout),
        )
        self.action_modulation = nn.Sequential(nn.SiLU(), nn.Linear(action_dim, 6 * dim))

        modulation = self.action_modulation[-1]
        nn.init.zeros_(modulation.weight)
        nn.init.zeros_(modulation.bias)

    def forward(self, x: Tensor, latent_actions: Tensor, causal_mask: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.action_modulation(
            latent_actions
        ).chunk(6, dim=-1)

        attention_input = _modulate(self.norm1(x), shift_attn, scale_attn)
        attention_output = self.attention(
            attention_input,
            attention_input,
            attention_input,
            attn_mask=causal_mask,
            need_weights=False,
        )[0]
        # A zero action modulation recovers an ordinary Transformer block. Action
        # influence then grows continuously away from this stable initialization.
        x = x + (1 + gate_attn) * attention_output
        x = x + (1 + gate_mlp) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class LeWorldModelPredictor(nn.Module):
    """Causal world model conditioned on Diffusion's latent action tokens."""

    def __init__(self, config: DiffusionJEPAConfig):
        super().__init__()
        self.config = config
        dim = config.jepa_latent_dim
        max_sequence_length = config.jepa_prediction_horizon

        self.position_embedding = nn.Parameter(torch.zeros(1, max_sequence_length, dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        # This explicit token path gives the alignment loss a non-zero gradient
        # into Diffusion from the first update. AdaLN below provides additional
        # per-block conditioning as its zero-initialized modulation learns.
        self.latent_action_embedding = nn.Linear(config.jepa_action_latent_dim, dim)
        self.blocks = nn.ModuleList([ActionAdaLNBlock(config) for _ in range(config.jepa_predictor_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.projector = nn.Linear(dim, dim, bias=False)
        self.projector_norm = nn.BatchNorm1d(dim)

    def forward(self, latents: Tensor, latent_actions: Tensor) -> Tensor:
        """Predict the latent following every aligned state/latent-action pair."""
        if latents.ndim != 3 or latent_actions.ndim != 3:
            raise ValueError(
                "LeWorldModel predictor expects latents and latent actions shaped (B, T, D)."
            )
        if latents.shape[:2] != latent_actions.shape[:2]:
            raise ValueError(
                "State and latent-action sequences must have matching batch/time axes. "
                f"Got {latents.shape[:2]} and {latent_actions.shape[:2]}."
            )
        sequence_length = latents.shape[1]
        if sequence_length > self.position_embedding.shape[1]:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds configured maximum "
                f"{self.position_embedding.shape[1]}."
            )

        x = (
            latents
            + self.position_embedding[:, :sequence_length]
            + self.latent_action_embedding(latent_actions)
        )
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=latents.device,
        ).triu(diagonal=1)
        for block in self.blocks:
            x = block(x, latent_actions, causal_mask)

        x = self.projector(self.final_norm(x))
        return _stepwise_batch_norm(self.projector_norm, x)
