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


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer using real-valued ECF operations.

    Embeddings are projected onto random unit directions. Each one-dimensional
    empirical characteristic function is matched to that of N(0, 1) using a
    discretized Epps-Pulley statistic. The implementation avoids complex tensors
    so it can run on accelerator backends with limited complex-op support.
    """

    def __init__(
        self,
        num_projections: int = 512,
        num_frequencies: int = 17,
        min_frequency: float = 0.2,
        max_frequency: float = 4.0,
    ):
        super().__init__()
        if num_projections <= 0:
            raise ValueError("`num_projections` must be positive.")
        if num_frequencies < 2:
            raise ValueError("`num_frequencies` must be at least 2.")
        self.num_projections = num_projections
        self.num_frequencies = num_frequencies
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

    def _single_step_loss(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError("SIGReg expects a (samples, features) tensor per time step.")
        if embeddings.shape[0] < 2:
            return embeddings.sum() * 0

        values = embeddings.float()
        directions = torch.randn(
            values.shape[-1],
            self.num_projections,
            device=values.device,
            dtype=values.dtype,
        )
        directions = directions / directions.square().sum(dim=0, keepdim=True).clamp_min(1e-12).sqrt()
        projected = values @ directions

        frequencies = torch.linspace(
            self.min_frequency,
            self.max_frequency,
            self.num_frequencies,
            device=values.device,
            dtype=values.dtype,
        )
        phases = projected.unsqueeze(-1) * frequencies
        empirical_real = phases.cos().mean(dim=0)
        empirical_imag = phases.sin().mean(dim=0)
        target_real = (-0.5 * frequencies.square()).exp().unsqueeze(0)
        weight = (-0.5 * frequencies.square()).exp().unsqueeze(0)
        integrand = weight * ((empirical_real - target_real).square() + empirical_imag.square())

        spacing = (self.max_frequency - self.min_frequency) / (self.num_frequencies - 1)
        integral = spacing * (0.5 * integrand[:, 0] + integrand[:, 1:-1].sum(dim=-1) + 0.5 * integrand[:, -1])
        # Keep the statistic in FP32 under autocast; casting it back to FP16 can
        # underflow when the empirical distribution is already close to normal.
        return integral.mean()

    def forward(self, embeddings: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        """Compute step-wise SIGReg for embeddings shaped ``(B, T, D)``."""
        if embeddings.ndim != 3:
            raise ValueError("SIGReg expects embeddings shaped (B, T, D).")
        if valid_mask is None:
            valid_mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        if valid_mask.shape != embeddings.shape[:2]:
            raise ValueError(f"SIGReg mask must have shape {embeddings.shape[:2]}. Got {valid_mask.shape}.")

        losses = []
        for step in range(embeddings.shape[1]):
            step_embeddings = embeddings[:, step][valid_mask[:, step]]
            if step_embeddings.shape[0] >= 2:
                losses.append(self._single_step_loss(step_embeddings))
        if not losses:
            return embeddings.sum() * 0
        return torch.stack(losses).mean()
