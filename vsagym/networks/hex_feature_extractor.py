import numpy as np
import gymnasium as gym
import torch
from torch import nn as nn
from gymnasium.spaces import Space
from typing import Any, Optional, Union

from vsagym.spaces import HexagonalSSPSpace, RandomSSPSpace
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .ssp_feature_extractor import SSPFeaturesExtractor, ModulationMLP

def skew_symmetric_matrix(params, n):
    # Convert the parameter vector into a skew-symmetric matrix
    A = torch.zeros(n, n, device=params.device, dtype=params.dtype)
    idx_upper = torch.triu_indices(n, n, offset=1)
    A[idx_upper[0], idx_upper[1]] = params
    A = A - A.T  # Make it skew-symmetric
    return A

def exponential_map(A):
    # Compute the matrix exponential of the skew-symmetric matrix
    return torch.matrix_exp(A)

class RotationLearner(torch.nn.Module):
    def __init__(self, n):
        super(RotationLearner, self).__init__()
        # We need (n choose 2) parameters for a skew-symmetric matrix in n dimensions
        self.n = n
        self.params = torch.nn.Parameter(torch.randn((n * (n - 1)) // 2))

    def forward(self):
        A = skew_symmetric_matrix(self.params, self.n)
        R = exponential_map(A)
        return R

class SSPHexFeaturesExtractor(SSPFeaturesExtractor):
    """Hexagonal SSP feature extractor with a structured phase matrix
    (per-scale, per-rotation parametrization).

    ``ls_mode`` behaves as in :class:`SSPFeaturesExtractor`. ``phase_mode``:
      - ``'fixed'``: scales and rotations are frozen,
      - ``'learn'`` (default): scales and rotations are free parameters,
      - ``'nn'``: the scales are output by a small MLP taking ``x`` as input
        (rotations remain free parameters), so the phase matrix is an
        input-dependent function while preserving the hexagonal structure.
    """
    def __init__(self, observation_space: Space[Any],
                 features_dim: int,
                 rng: Optional[Union[int, np.random.Generator]] = None,
                 input_dim: Optional[int] = None,
                 learn_ls: bool = True,
                 ls_mode: Optional[str] = None,
                 phase_mode: Optional[str] = None,
                 nn_hidden_dim: int = 32,
                 nn_n_layers: int = 1,
                 **kwargs):
        # The base class builds a plain (half) phase matrix parameter; replace it
        # below with the structured scale/rotation parametrization.
        super().__init__(observation_space,
                         features_dim, 'hex', rng, input_dim,
                         True, learn_ls,
                         ls_mode=ls_mode, phase_mode='learn',
                         nn_hidden_dim=nn_hidden_dim, nn_n_layers=nn_n_layers,
                         **kwargs)
        delattr(self, 'phase_matrix')

        if phase_mode is None:
            phase_mode = 'learn'
        if phase_mode not in ('fixed', 'learn', 'nn'):
            raise ValueError('Invalid phase_mode {phase_mode}'.format(phase_mode=phase_mode))
        self.phase_mode = phase_mode

        ssp_space = HexagonalSSPSpace(self.input_dim, features_dim, rng=rng, **kwargs)
        self.phase_base = torch.Tensor(ssp_space.phases_hex)

        self._features_dim = ssp_space.ssp_dim
        self.ssp_dim = ssp_space.ssp_dim
        self.n_rotations = int(np.sqrt((self.ssp_dim-1)/(2*(self.input_dim+1))))
        self.n_params = (self.ssp_dim-1)//2

        init_scales = torch.Tensor(ssp_space.scales)
        if self.phase_mode == 'nn':
            self.scale_net = ModulationMLP(self.input_dim,
                                           init_scales.shape[0],
                                           hidden_dim=nn_hidden_dim,
                                           n_layers=nn_n_layers,
                                           init_output=init_scales)
        else:
            self.scales = nn.Parameter(init_scales,
                                       requires_grad=(self.phase_mode == 'learn'))
        self.rotation_learners = torch.nn.ModuleList([RotationLearner(self.input_dim) for _ in range(self.n_rotations)])
        if self.phase_mode == 'fixed':
            for rl in self.rotation_learners:
                rl.params.requires_grad_(False)

    def _phases_from_scales(self, scales, device):
        """Scale the base simplex phases and apply the learned rotations.

        ``scales``: ``(n_scales,)`` or per-sample ``(batch, n_scales)``.
        Returns ``(n_params, d)`` or ``(batch, n_params, d)`` respectively.
        """
        phase_base = self.phase_base.to(device)
        phases_scaled = scales[..., :, None, None] * phase_base   # (..., n_scales, d+1, d)
        phases_scaled = phases_scaled.reshape(scales.shape[:-1] + (-1, self.input_dim))
        R = torch.stack([rl() for rl in self.rotation_learners])  # (n_rotations, d, d)
        rotated = torch.einsum('...kd,rde->...rke', phases_scaled, R)
        return rotated.reshape(scales.shape[:-1] + (self.n_params, self.input_dim))

    def _compute_phase_matrix(self, x):
        if self.phase_mode == 'nn':
            scales = self.scale_net(x)                            # (batch, n_scales)
        else:
            scales = self.scales
        return self._conjsym(self._phases_from_scales(scales, x.device))

    def _get_phase_matrix(self, device):
        phases_scaled = torch.vstack([self.phase_base.to(device) * i for i in self.scales])
        rotated_matrices = []
        for i in range(self.n_rotations):
            R = self.rotation_learners[i]()
            rotated_matrices.append(torch.matmul(phases_scaled, R))
        phases = torch.vstack(rotated_matrices)
        F = torch.zeros((self.ssp_dim, self.input_dim)).to(device)
        F[1:(self.n_params + 1), :] = phases
        F[(self.n_params + 1):, :] = -torch.flip(phases, dims=(0,))
        return F
