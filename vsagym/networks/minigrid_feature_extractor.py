import numpy as np
import gymnasium as gym
import torch
from torch import nn as nn
from gymnasium.spaces import Space
from typing import Any, Optional, Union

from vsagym.spaces import HexagonalSSPSpace, RandomSSPSpace

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SSPMiniGridViewFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, 
                 features_dim,
                 basis_type='hex',
                 rng=np.random.default_rng(0),**kwargs):
        super().__init__(observation_space, features_dim=1)
        
        self.input_dim = 3
        features_dim = features_dim
        if 'ssp_h' in kwargs:
            initial_ls = kwargs['ssp_h']
            if (type(initial_ls) is np.ndarray):
                initial_ls = torch.Tensor(initial_ls.flatten())
            elif (type(initial_ls) is list):
                initial_ls = torch.Tensor(initial_ls)
        else:
            initial_ls = 1.0
            
        if basis_type=='hex':
            ssp_space = HexagonalSSPSpace(self.input_dim, features_dim,rng=rng,**kwargs)
        elif basis_type=='rand':
            ssp_space = RandomSSPSpace(self.input_dim, features_dim,rng=rng,**kwargs)
            
        self._features_dim = ssp_space.ssp_dim
        self.nparams = (self._features_dim-1)//2
        self.phase_matrix = torch.Tensor(ssp_space.phase_matrix)
        #self.phase_matrix = nn.Parameter(torch.Tensor(ssp_space.phase_matrix[1:(self.nparams+1),:]),requires_grad=True)
        self.length_scale = nn.Parameter(initial_ls*torch.ones(self.input_dim), requires_grad=True)
        self.length_scale_view = nn.Parameter(initial_ls*torch.ones(self.input_dim), requires_grad=True)
        
        self.view_width = 7
        self.view_height = 7
        domain_bounds = np.array([ [0, self.view_width-1],
                                  [-(self.view_height-1)//2, (self.view_height-1)//2 ],
                                  [0,3]])
        xs = [np.arange(domain_bounds[i,1],domain_bounds[i,0]-1,-1) for i in range(2)]
        xx = np.meshgrid(*xs)
        xx[0] = 3 - xx[0]
        xx[1] = 6 - xx[0]
        self.grid_pts = torch.Tensor(np.array(xx))
        self.n_pts = self.grid_pts.shape[1] * self.grid_pts.shape[2]
        self.unroll_grid_pts  = torch.vstack([self.grid_pts[0,:].flatten(), self.grid_pts[1,:].flatten(), -torch.ones(self.n_pts)]).T
        

    def _encode(self, x, ls):
        ls_mat = torch.atleast_2d(torch.diag(ls)).to(x.device)
        # F = torch.zeros((self._features_dim, self.input_dim)).to(x.device)
        # F[1:(self.nparams+1),:] = self.phase_matrix
        # F[(self.nparams+1):,:] = -torch.flip(self.phase_matrix, dims=(0,))
        F = self.phase_matrix
        x = (F @ (x @ ls_mat).T).type(torch.complex64) # fix .to(x.device)
        x = torch.fft.ifft( torch.exp( 1.j * x), axis=0 ).real.T
        return x
    
    def _bind(self,a,b):
        return torch.fft.ifft(torch.fft.fft(a, axis=-1) * torch.fft.fft(b, axis=-1), axis=-1).real
        
    def forward(self, obs) -> torch.Tensor:
        x = obs[:,:self.input_dim]
        M = self._encode(x, self.length_scale)
        
        obj_sps = obs[:,self.input_dim:-self._features_dim].reshape(obs.shape[0],-1,self._features_dim)
        ssp_grid_pts = self._encode(self.unroll_grid_pts.to(x.device), self.length_scale_view).reshape(1, self.n_pts, self._features_dim)
        M += torch.sum(self._bind(obj_sps, ssp_grid_pts), axis=1)     
        M += obs[:,-self._features_dim:]
        M = M/torch.linalg.vector_norm(M, dim=-1, keepdim=True)
        return M

class SSPBabyAIViewProcesser(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim, basis_type='hex',
                 rng=np.random.default_rng(0),**kwargs):
        # We do not know features-dim here before creating ssp_space,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)
        
        self.input_dim = 3
        features_dim = features_dim
        if 'ssp_h' in kwargs:
            initial_ls = kwargs['ssp_h']
            if (type(initial_ls) is np.ndarray):
                initial_ls = torch.Tensor(initial_ls.flatten())
            elif (type(initial_ls) is list):
                initial_ls = torch.Tensor(initial_ls)
        else:
            initial_ls = 1.0
            
        if basis_type=='hex':
            ssp_space = HexagonalSSPSpace(self.input_dim, features_dim,rng=rng,**kwargs)
        elif basis_type=='rand':
            ssp_space = RandomSSPSpace(self.input_dim, features_dim,rng=rng,**kwargs)
            
        self._features_dim = ssp_space.ssp_dim
        self.nparams = (self._features_dim-1)//2
        self.phase_matrix = nn.Parameter(torch.Tensor(ssp_space.phase_matrix[1:(self.nparams+1),:]),requires_grad=True)
        self.length_scale = nn.Parameter(initial_ls*torch.ones(self.input_dim), requires_grad=True)
        self.view_width = 7
        self.view_height = 7
        domain_bounds = np.array([ [0, self.view_width-1],
                                  [-(self.view_height-1)//2, (self.view_height-1)//2 ],
                                  [0,3]])
        xs = [np.arange(domain_bounds[i,1],domain_bounds[i,0]-1,-1) for i in range(2)]
        xx = np.meshgrid(*xs)
        xx[0] = 3 - xx[0]
        xx[1] = 6 - xx[0]
        self.grid_pts = torch.Tensor(np.array(xx))
        self.n_pts = self.grid_pts.shape[1] * self.grid_pts.shape[2]
        self.unroll_grid_pts  = torch.vstack([self.grid_pts[0,:].flatten(), self.grid_pts[1,:].flatten(), -torch.ones(self.n_pts)]).T
        

    def _encode(self, x):
        ls_mat = torch.atleast_2d(torch.diag(1/self.length_scale)).to(x.device)
        F = torch.zeros((self._features_dim, self.input_dim)).to(x.device)
        F[1:(self.nparams+1),:] = self.phase_matrix
        F[(self.nparams+1):,:] = -torch.flip(self.phase_matrix, dims=(0,))
        x = (F @ (x @ ls_mat).T).type(torch.complex64) # fix .to(x.device)
        x = torch.fft.ifft( torch.exp( 1.j * x), axis=0 ).real.T
        return x
    
    def _bind(self,a,b):
        return torch.fft.ifft(torch.fft.fft(a, axis=-1) * torch.fft.fft(b, axis=-1), axis=-1).real
        
    def forward(self, obs) -> torch.Tensor:
        x = obs[:,:self.input_dim]
        M = self._encode(x)
        
        obj_sps = obs[:,self.input_dim:-2*self._features_dim].reshape(obs.shape[0],-1,self._features_dim)
        ssp_grid_pts = self._encode(self.unroll_grid_pts.to(x.device)).reshape(1, self.n_pts, self._features_dim)
        M += torch.sum(self._bind(obj_sps, ssp_grid_pts), axis=1)     
        M += obs[:,-2*self._features_dim:-self._features_dim]
        O = obs[:,-self._features_dim:]
        return torch.concatenate([M, O])
    
# policy_kwargs = dict(
#     features_extractor_class=SSPBabyAIViewProcesser,
#     features_extractor_kwargs=dict(features_dim=128),
# )