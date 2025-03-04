import numpy as np
import gymnasium as gym
import torch
from torch import nn as nn
from gymnasium.spaces import Space
from typing import Any, Optional, Union

from vsagym.spaces import HexagonalSSPSpace, RandomSSPSpace
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .ssp_feature_extractor import SSPFeaturesExtractor

def skew_symmetric_matrix(params, n):
    # Convert the parameter vector into a skew-symmetric matrix
    A = torch.zeros(n, n)
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
    def __init__(self, observation_space: Space[Any],
                 features_dim: int,
                 rng: Optional[Union[int, np.random.Generator]] = None,
                 input_dim: Optional[int] = None,
                 learn_ls: bool = True,
                 **kwargs):
        super().__init__(observation_space,
                         features_dim, 'hex', rng, input_dim,
                         True, learn_ls, **kwargs)
        delattr(self, 'phase_matrix')

        ssp_space = HexagonalSSPSpace(self.input_dim, features_dim, rng=rng, **kwargs)
        self.phase_base = torch.Tensor(ssp_space.phases_hex)
        
        self._features_dim = ssp_space.ssp_dim
        self.ssp_dim = ssp_space.ssp_dim
        self.n_rotations = int(np.sqrt((self.ssp_dim-1)/(2*(self.input_dim+1))))

        self.scales = nn.Parameter(torch.Tensor(ssp_space.scales),
                                   requires_grad=True)
        self.rotation_learners = torch.nn.ModuleList([RotationLearner(self.input_dim) for _ in range(self.n_rotations)])
        self.n_params = (self.ssp_dim-1)//2

    def _get_phase_matrix(self, device):
        phases_scaled = torch.vstack([self.phase_base * i for i in self.scales])
        rotated_matrices = []
        for i in range(self.n_rotations):
            R = self.rotation_learners[i]()
            rotated_matrices.append(torch.matmul(phases_scaled, R))
        phases = torch.vstack(rotated_matrices)
        F = torch.zeros((self.ssp_dim, self.input_dim)).to(device)
        F[1:(self.n_params + 1), :] = phases
        F[(self.n_params + 1):, :] = -torch.flip(phases, dims=(0,))
        return F
