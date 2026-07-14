import numpy as np
import gymnasium as gym
import torch
from torch import nn as nn
from gymnasium.spaces import Space
from typing import Any, Optional, Union

from vsagym.spaces import HexagonalSSPSpace, RandomSSPSpace

# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ModulationMLP(nn.Module):
    """Small MLP used to modulate SSP encoding parameters as a function of x.

    The final layer's weights are zero-initialized and its bias is set to
    ``init_output``, so at initialization the network outputs ``init_output``
    for every input (i.e. it starts out identical to the static parameter it
    replaces) and learns input-dependent deviations during training.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 32,
                 n_layers: int = 1,
                 init_output: Optional[torch.Tensor] = None):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        head = nn.Linear(in_dim, output_dim)
        nn.init.zeros_(head.weight)
        if init_output is None:
            nn.init.zeros_(head.bias)
        else:
            with torch.no_grad():
                head.bias.copy_(init_output.flatten())
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SSPFeaturesExtractor(nn.Module):
    """SSP encoding as a trainable feature extractor.

    The length scale and phase matrix can each be:
      - ``'fixed'``: a constant,
      - ``'learn'``: a free ``nn.Parameter``,
      - ``'nn'``: the output of a small trainable MLP taking ``x`` as input,
        so the encoding parameters vary over the input space. The MLP's final
        layer is zero-initialized, so training starts from the static encoding.

    ``ls_mode`` / ``phase_mode`` select the behaviour; if left as ``None`` they
    default to ``'learn'``/``'fixed'`` according to the (older) boolean flags
    ``learn_ls`` / ``learn_phase_matrix``.
    """

    def __init__(self,
                 observation_space: Space[Any],
                 features_dim: int,
                 basis_type: str = 'hex',
                 rng: Optional[Union[int, np.random.Generator]] = None,
                 input_dim: Optional[int] = None,
                 learn_phase_matrix: bool = True,
                 learn_ls: bool = True,
                 ls_mode: Optional[str] = None,
                 phase_mode: Optional[str] = None,
                 nn_hidden_dim: int = 32,
                 nn_n_layers: int = 1,
                 **kwargs):
        # super().__init__(observation_space, features_dim=1)
        super().__init__()
        self._observation_space = observation_space

        if ls_mode is None:
            ls_mode = 'learn' if learn_ls else 'fixed'
        if phase_mode is None:
            phase_mode = 'learn' if learn_phase_matrix else 'fixed'
        if ls_mode not in ('fixed', 'learn', 'nn'):
            raise ValueError('Invalid ls_mode {ls_mode}'.format(ls_mode=ls_mode))
        if phase_mode not in ('fixed', 'learn', 'nn'):
            raise ValueError('Invalid phase_mode {phase_mode}'.format(phase_mode=phase_mode))
        self.ls_mode = ls_mode
        self.phase_mode = phase_mode
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_n_layers = nn_n_layers

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
        # When input_dim differs from obs_space dim (e.g. MiniGrid wrappers), trim or
        # broadcast initial_ls to match self.input_dim.
        if initial_ls.shape[0] != self.input_dim:
            if initial_ls.shape[0] > self.input_dim:
                initial_ls = initial_ls[:self.input_dim]
            else:
                initial_ls = initial_ls.mean() * torch.ones(self.input_dim)

        if basis_type == 'hex':
            ssp_space = HexagonalSSPSpace(self.input_dim, features_dim, rng=rng, **kwargs)
        elif basis_type == 'rand':
            ssp_space = RandomSSPSpace(self.input_dim, features_dim, rng=rng, **kwargs)
        else:
            raise ValueError('Invalid basis type {basis_type}'.format(basis_type=basis_type))

        self._features_dim = ssp_space.ssp_dim
        self.ssp_dim = ssp_space.ssp_dim
        self.n_params = (self.ssp_dim-1)//2

        init_phase_matrix = torch.Tensor(ssp_space.phase_matrix[1:(self.n_params+1), :])
        if self.phase_mode == 'nn':
            self.phase_matrix_net = ModulationMLP(self.input_dim,
                                                  self.n_params * self.input_dim,
                                                  hidden_dim=nn_hidden_dim,
                                                  n_layers=nn_n_layers,
                                                  init_output=init_phase_matrix)
        else:
            self.phase_matrix = nn.Parameter(init_phase_matrix,
                                             requires_grad=(self.phase_mode == 'learn'))

        init_log_inv_ls = torch.log(1/initial_ls)*torch.ones(self.input_dim)
        if self.ls_mode == 'nn':
            self.length_scale_net = ModulationMLP(self.input_dim,
                                                  self.input_dim,
                                                  hidden_dim=nn_hidden_dim,
                                                  n_layers=nn_n_layers,
                                                  init_output=init_log_inv_ls)
        else:
            self.length_scale = nn.Parameter(init_log_inv_ls,
                                             requires_grad=(self.ls_mode == 'learn'))

    @property
    def features_dim(self) -> int:
        """The number of features that the extractor outputs."""
        return self._features_dim

    def _conjsym(self, half_phase_matrix):
        """Build the full phase matrix (with conjugate symmetry) from its
        free half. Works on ``(n_params, d)`` or batched ``(batch, n_params, d)``."""
        zeros = torch.zeros(half_phase_matrix.shape[:-2] + (1, self.input_dim),
                            device=half_phase_matrix.device,
                            dtype=half_phase_matrix.dtype)
        return torch.cat([zeros, half_phase_matrix,
                          -torch.flip(half_phase_matrix, dims=(-2,))], dim=-2)

    def _compute_ls(self, x):
        """Log inverse length scale; ``(d,)`` or per-sample ``(batch, d)`` for ls_mode='nn'."""
        if self.ls_mode == 'nn':
            return self.length_scale_net(x)
        return self.length_scale

    def _compute_phase_matrix(self, x):
        """Full phase matrix; ``(ssp_dim, d)`` or per-sample ``(batch, ssp_dim, d)``
        for phase_mode='nn'."""
        if self.phase_mode == 'nn':
            half = self.phase_matrix_net(x).reshape(-1, self.n_params, self.input_dim)
            return self._conjsym(half)
        return self._conjsym(self.phase_matrix)

    def _encode_flexible(self, x, log_inv_ls, F):
        """Encode x given (possibly per-sample) log inverse length scales and
        (possibly per-sample) phase matrices."""
        scaled_x = x * torch.exp(log_inv_ls)
        if F.dim() == 3:
            phases = torch.einsum('bkd,bd->bk', F, scaled_x)
        else:
            phases = scaled_x @ F.T
        return torch.fft.ifft(torch.exp(1.j * phases.type(torch.complex64)), dim=-1).real

    def _get_ls_matrix(self, device, ls):
        return torch.atleast_2d(torch.diag(torch.exp(ls))).to(device)

    def _get_phase_matrix(self, device):
        F = torch.zeros((self.ssp_dim, self.input_dim)).to(device)
        F[1:(self.n_params + 1), :] = self.phase_matrix
        F[(self.n_params + 1):, :] = -torch.flip(self.phase_matrix, dims=(0,))
        return F
    def _encode(self, x, ls_mat, F):
        x = (F @ (x @ ls_mat).T).type(torch.complex64)
        x = torch.fft.ifft(torch.exp(1.j * x), axis=0).real.T
        return x
    def _bind(self, a, b):
        return torch.fft.ifft(torch.fft.fft(a, axis=-1) * torch.fft.fft(b, axis=-1), axis=-1).real

    def forward(self, obs) -> torch.Tensor:
        log_inv_ls = self._compute_ls(obs)
        F = self._compute_phase_matrix(obs)
        return self._encode_flexible(obs, log_inv_ls, F)
