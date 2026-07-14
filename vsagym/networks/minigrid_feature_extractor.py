import copy

import numpy as np
import gymnasium as gym
import torch
from torch import nn as nn
from gymnasium.spaces import Space
from typing import Any, Optional, Union

# from vsagym.spaces import HexagonalSSPSpace, RandomSSPSpace
from .ssp_feature_extractor import SSPFeaturesExtractor
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SSPMiniGridViewFeatures(SSPFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 features_dim: int,
                 basis_type: str = 'hex',
                 rng: Optional[Union[int, np.random.Generator]] = None,
                 learn_phase_matrix: bool = True,
                 learn_ls: bool = True,
                 diff_ls_for_view: bool = False,
                 view_width: int = 7, view_height: int = 7,
                 ls_mode: Optional[str] = None,
                 phase_mode: Optional[str] = None,
                 nn_hidden_dim: int = 32,
                 nn_n_layers: int = 1,
                 **kwargs):
        input_dim = 3
        super().__init__(observation_space,
                        features_dim, basis_type, rng, input_dim,
                        learn_phase_matrix, learn_ls,
                        ls_mode=ls_mode, phase_mode=phase_mode,
                        nn_hidden_dim=nn_hidden_dim, nn_n_layers=nn_n_layers,
                        **kwargs)

        self.diff_ls_for_view = diff_ls_for_view
        if self.ls_mode == 'nn':
            if diff_ls_for_view:
                self.length_scale_view_net = copy.deepcopy(self.length_scale_net)
            else:
                self.length_scale_view_net = self.length_scale_net
        elif diff_ls_for_view:
            self.length_scale_view = nn.Parameter(self.length_scale.clone().detach(),
                                                  requires_grad=learn_ls)
        else:
            self.length_scale_view = self.length_scale
        self.view_width = view_width
        self.view_height = view_height
        domain_bounds = np.array([[0, self.view_width - 1],
                                  [-(self.view_height - 1) // 2, (self.view_height - 1) // 2],
                                  [0, 3]])
        xs = [np.arange(domain_bounds[i, 1], domain_bounds[i, 0] - 1, -1) for i in range(2)]
        xx = np.meshgrid(*xs)
        self.grid_pts = torch.Tensor(np.array(xx))
        pts = np.vstack([xx[i].reshape(-1) for i in range(2)]).T
        self.unroll_pts = torch.Tensor(np.hstack([pts, -1 * np.ones((pts.shape[0], 1))]))
        self.n_pts = self.unroll_pts.shape[0]

    def _compute_ls_view(self, pts):
        """Log inverse length scale for the ego-view grid points."""
        if self.ls_mode == 'nn':
            return self.length_scale_view_net(pts)
        return self.length_scale_view

    def _encode_pose_and_grid(self, x):
        """Encode the agent pose and the ego-view grid points."""
        log_inv_ls = self._compute_ls(x)
        F = self._compute_phase_matrix(x)
        pose_ssp = self._encode_flexible(x, log_inv_ls, F)

        unroll_pts = self.unroll_pts.to(x.device)
        log_inv_ls_view = self._compute_ls_view(unroll_pts)  # different ls for ego-vec
        F_view = self._compute_phase_matrix(unroll_pts)
        ssp_grid_pts = self._encode_flexible(unroll_pts, log_inv_ls_view,
                                             F_view).reshape(1, self.n_pts, self.ssp_dim)
        return pose_ssp, ssp_grid_pts

    def forward(self, obs) -> torch.Tensor:
        x = obs[:, :self.input_dim]
        vsa_output, ssp_grid_pts = self._encode_pose_and_grid(x)
        obj_sps = obs[:,self.input_dim:-self.ssp_dim].reshape(obs.shape[0], -1, self.ssp_dim)
        vsa_output += torch.sum(self._bind(obj_sps, ssp_grid_pts), axis=1)
        vsa_output += obs[:,-self.ssp_dim:] # the 'has'/carrying vector
        vsa_output = vsa_output/torch.linalg.vector_norm(vsa_output, dim=-1, keepdim=True)
        return vsa_output

class SSPMiniGridMissionFeatures(SSPMiniGridViewFeatures):
    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 features_dim: int,
                 basis_type: str = 'hex',
                 rng: Optional[Union[int, np.random.Generator]] = None,
                 learn_phase_matrix: bool = True,
                 learn_ls: bool = True,
                 diff_ls_for_view: bool = False,
                 **kwargs):
        super().__init__(observation_space,
                         features_dim, basis_type, rng,
                         learn_phase_matrix, learn_ls, diff_ls_for_view, **kwargs)
        self._features_dim = 2*self._features_dim

    def forward(self, obs) -> torch.Tensor:
        x = obs[:, :self.input_dim]
        vsa_output, ssp_grid_pts = self._encode_pose_and_grid(x)
        obj_sps = obs[:,self.input_dim:-2*self.ssp_dim].reshape(obs.shape[0], -1, self.ssp_dim)
        vsa_output += torch.sum(self._bind(obj_sps, ssp_grid_pts), axis=1)
        vsa_output += obs[:,-2*self.ssp_dim:-self.ssp_dim] # the 'has'/carrying vector
        vsa_output = vsa_output/torch.linalg.vector_norm(vsa_output, dim=-1, keepdim=True)

        mission_vec = obs[:, -self.ssp_dim:]
        return torch.hstack([vsa_output, mission_vec])
