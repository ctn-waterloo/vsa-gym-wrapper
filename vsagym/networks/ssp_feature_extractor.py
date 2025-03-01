import numpy as np
import gymnasium as gym
import torch
from torch import nn as nn
from gymnasium.spaces import Space
from typing import Any, Optional, Union

from vsagym.spaces import HexagonalSSPSpace, RandomSSPSpace

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SSPFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: Space[Any],
                 features_dim: int,
                 basis_type: str = 'hex',
                 rng: Optional[Union[int, np.random.Generator]] = None,
                 input_dim: Optional[int] = None,
                 learn_phase_matrix: bool = True,
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
            initial_ls = torch.Tensor(np.clip( ( np.abs( observation_space.high - observation_space.low ) ),
                                               a_min = 1e-4, a_max = 1e4) / 10.)
            
        if basis_type == 'hex':
            ssp_space = HexagonalSSPSpace(self.input_dim, features_dim,rng=rng, **kwargs)
        elif basis_type == 'rand':
            ssp_space = RandomSSPSpace(self.input_dim, features_dim, rng=rng, **kwargs)
        else:
            raise ValueError('Invalid basis type {basis_type}'.format(basis_type=basis_type))
            
        self._features_dim = ssp_space.ssp_dim
        self.n_params = (self._features_dim-1)//2
        self.phase_matrix = nn.Parameter(torch.Tensor(ssp_space.phase_matrix[1:(self.n_params+1),:]),
                                         requires_grad=learn_phase_matrix)
        self.length_scale = nn.Parameter(torch.log(1/initial_ls)*torch.ones(self.input_dim),
                                         requires_grad=learn_ls)

    def forward(self, obs) -> torch.Tensor:
        x = obs
        ls_mat = torch.atleast_2d(torch.diag(torch.exp(self.length_scale))).to(x.device) # exp for positive val
        F = torch.zeros((self._features_dim, self.input_dim)).to(x.device)
        F[1:(self.n_params+1),:] = self.phase_matrix
        F[(self.n_params+1):,:] = -torch.flip(self.phase_matrix, dims=(0,)) # conjugate sym for realness
        x = (F @ (x @ ls_mat).T).type(torch.complex64)
        x = torch.fft.ifft(torch.exp(1.j * x), axis=0).real.T
        return x

