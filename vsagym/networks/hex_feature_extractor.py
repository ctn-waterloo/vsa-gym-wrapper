import numpy as np
import gymnasium as gym
import torch
from torch import nn as nn
from gymnasium.spaces import Space
from typing import Any, Optional, Union

from vsagym.spaces import HexagonalSSPSpace, RandomSSPSpace
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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
        self.params = torch.nn.Parameter(torch.randn((n * (n - 1)) // 2))

    def forward(self):
        A = skew_symmetric_matrix(self.params, len(self.params))
        R = exponential_map(A)
        return R
    
class SSPHexFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space[Any],
                 features_dim: int,
                 rng: Optional[Union[int, np.random.Generator]] = None,
                 input_dim: Optional[int] = None,
                 learn_ls: bool = True,
                 **kwargs):
        super().__init__(observation_space, features_dim=1)
        
        if input_dim is None:
            self.input_dim = observation_space.shape[0]
        else:
            self.input_dim = input_dim

        if 'length_scale' in kwargs:
            initial_ls = kwargs['length_scale']
            if (type(initial_ls) is np.ndarray):
                initial_ls = torch.Tensor(initial_ls.flatten())
            elif (type(initial_ls) is list):
                initial_ls = torch.Tensor(initial_ls)
            else:
                initial_ls = torch.Tensor([initial_ls])
        else:
            initial_ls = torch.Tensor(
                np.clip((np.abs(observation_space.high - observation_space.low)), a_min=1e-4, a_max=1e4) / 10.)

        ssp_space = HexagonalSSPSpace(self.input_dim, features_dim, rng=rng, **kwargs)
        self.phase_base = ssp_space.phases_hex
        
        self._features_dim = ssp_space.ssp_dim
        self.n_params = int(np.sqrt((self._features_dim-1)/(2*(self.input_dim+1))))

        self.length_scale = nn.Parameter(torch.log(initial_ls)*torch.ones(self.input_dim),
                                         requires_grad=learn_ls)
        self.scales = nn.Parameter(torch.Tensor(ssp_space.scales),
                                   requires_grad=True)
        self.rotation_learners = torch.nn.ModuleList([RotationLearner(self.input_dim) for _ in range(self.n_params)])

        
        
    def forward(self, obs) -> torch.Tensor:
        x = obs
        ls_mat = torch.atleast_2d(torch.diag(torch.exp(self.length_scale))).to(x.device)

        phases_scaled=np.vstack([self.phase_base*i for i in self.scales])
        rotated_matrices = []
        for i in range(self.nparams):
            R = self.rotation_learners[i]()
            rotated_matrices.append(torch.matmul(R, phases_scaled))
        phases = torch.stack(rotated_matrices, dim=0)
        F = torch.zeros((self._features_dim, self.input_dim)).to(x.device)
        F[1:(self.nparams+1),:] = phases
        F[(self.nparams+1):,:] = -torch.flip(phases, dims=(0,))
            
        x = (F @ (x @ ls_mat).T).type(torch.complex64) # fix .to(x.device)
        x = torch.fft.ifft(torch.exp(1.j * x), axis=0).real.T
        return x