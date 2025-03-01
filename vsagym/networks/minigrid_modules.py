## Construct SSP features from input with trainable ls
import numpy as np
import typing as tp
import torch
from torch import nn
import torch.nn.functional as F

from vsagym.spaces import HexagonalSSPSpace, RandomSSPSpace

# These have not yet been tested with the newest versions of SSPSpace
class ssp_param(nn.Module):
    def __init__(self, input_dim, input_embedding_size, initial_ls=None, basis_type='hex',
                 stay_hex=False, learn_phase=True, learn_ls=True, rng=np.random.default_rng(0)):
        super().__init__()
        self.input_dim = input_dim
        self.stay_hex = stay_hex
        if basis_type == 'hex':
            ssp_space = HexagonalSSPSpace(self.input_dim, input_embedding_size, rng=rng)
        elif basis_type == 'rand':
            ssp_space = RandomSSPSpace(self.input_dim, input_embedding_size, rng=rng, sampler='norm')
        # elif basis_type=='mixed':
        #     ssp_space = HexagonalSSPSpace(2, input_embedding_size, rng=rng)
        #     ssp_space2 = RandomSSPSpace(1, ssp_space1.ssp_dim, rng=rng)

        self._features_dim = ssp_space.ssp_dim
        self.nparams = (self._features_dim - 1) // 2
        if (stay_hex) and (basis_type == 'hex'):
            self.phases_hex = nn.Parameter(torch.Tensor(ssp_space.phases_hex), requires_grad=False)

            self.gen_scales = nn.Parameter(torch.Tensor(ssp_space.scales[:, None, None]))
            self.gen_scales.requires_grad = learn_phase
            self.gen_rot_mats = nn.Parameter(torch.Tensor(ssp_space.rot_mats), requires_grad=False)
            self.generate_matrix = self._gen_force_hex
        else:
            self.phase_matrix = nn.Parameter(torch.Tensor(ssp_space.phase_matrix[1:(self.nparams + 1), :]))
            self.phase_matrix.requires_grad = learn_phase
            self.generate_matrix = self._gen_no_force

        if initial_ls is None:
            if (type(initial_ls) is np.ndarray):
                initial_ls = torch.Tensor(initial_ls.flatten())
            elif (type(initial_ls) is list):
                initial_ls = torch.Tensor(initial_ls)
            else:
                initial_ls = torch.Tensor([initial_ls])
        else:
            initial_ls = 0.1
        self.length_scale = nn.Parameter(torch.log(1 / initial_ls) * torch.ones(self.input_dim),
                                         requires_grad=learn_ls)

    def get_ls_mat(self):
        return torch.atleast_2d(torch.diag(torch.exp(self.length_scale)))

    def _gen_force_hex(self):
        phases_scaled = self.phases_hex[None, :, :] * self.gen_scales
        phases_scaled = phases_scaled.reshape(-1, self.input_dim)  # shape: (p * m, n)
        rotated_phases = torch.matmul(self.gen_rot_mats, phases_scaled.T)  # shape: (k, n, p * m)
        rotated_phases = rotated_phases.permute(0, 2, 1).reshape(-1, self.input_dim)  # shape: (k * p * m, n)

        F = torch.zeros((self._features_dim, self.input_dim))
        F[1:(self.nparams + 1), :] = rotated_phases
        F[(self.nparams + 1):, :] = -torch.flip(rotated_phases, dims=(0,))
        return F

    def _gen_no_force(self):
        F = torch.zeros((self._features_dim, self.input_dim))
        F[1:(self.nparams + 1), :] = self.phase_matrix
        F[(self.nparams + 1):, :] = -torch.flip(self.phase_matrix, dims=(0,))
        return F
class SSPInput(nn.Module):
    def __init__(self, obs_space,use_memory,  use_text, normalize,
                 input_embedding_size=151, hidden_size=0, activation_fun='relu', basis_type='hex',
                 rng=np.random.default_rng(0), stay_hex=False,
                 learn_phase=True, learn_ls=True, initial_ls=None,**kwargs):
        super().__init__()
        if type(obs_space["image"]) is int:
            self.input_dim = obs_space["image"]
        else:
            self.input_dim = obs_space["image"][0]

        if 'ssp_dim' in kwargs:
            input_embedding_size = kwargs['ssp_dim']
        else:
            input_embedding_size = input_embedding_size


        if 'length_scale' in kwargs:
            initial_ls = kwargs['ssp_h']
            if (type(initial_ls) is np.ndarray):
                initial_ls = torch.Tensor(initial_ls.flatten())
            elif (type(initial_ls) is list):
                initial_ls = torch.Tensor(initial_ls)
            elif type(initial_ls) is str:
                initial_ls = torch.Tensor(ast.literal_eval(initial_ls))
        else:
            initial_ls = 1.0


        self.ssp_parameters = ssp_param(self.input_dim, input_embedding_size,initial_ls,
                                        basis_type, stay_hex,
                                        learn_phase,learn_ls,rng)

        self._features_dim = self.ssp_parameters._features_dim

        if hidden_size>0:
            self.layers =  mlp(self._features_dim, hidden_size, activation_fun, input_embedding_size)
            self.input_embedding_size = input_embedding_size
        else:
            self.layers = torch.nn.Identity()
            self.input_embedding_size = self._features_dim


        self.other = FeatureProcesser(obs_space, self.input_embedding_size,
                                 use_text=use_text, use_memory=use_memory, normalize=normalize)
        self.embedding_size = self.other.embedding_size

    def _encode(self, x):
        ls_mat = self.ssp_parameters.get_ls_mat().to(x.device)
        F = self.ssp_parameters.generate_matrix().to(x.device)
        x = (F @ (x @ ls_mat).T).type(torch.complex64)
        x = torch.fft.ifft( torch.exp( 1.j * x), axis=0 ).real.T
        return x

    def forward(self, obs, memory):
        x = self._encode(obs.image)
        x = self.layers(x)
        embedding, memory, embed_txt = self.other(obs, x, memory)
        return embedding, memory, embed_txt



class SSPViewInput(nn.Module): # only for minigrid
    def __init__(self, obs_space,use_memory,  use_text, normalize,
                 input_embedding_size=151, hidden_size=0, activation_fun='relu', basis_type='hex', rng=np.random.default_rng(0),
                 stay_hex=False,learn_phase=True, learn_ls=True,initial_ls=None, **kwargs):
        super().__init__()
        self.input_dim = 3

        if 'ssp_dim' in kwargs:
            input_embedding_size = kwargs['ssp_dim']
        else:
            input_embedding_size = input_embedding_size


        if 'ssp_h' in kwargs:
            initial_ls = kwargs['ssp_h']
            if (type(initial_ls) is np.ndarray):
                initial_ls = torch.Tensor(initial_ls.flatten())
            elif (type(initial_ls) is list):
                initial_ls = torch.Tensor(initial_ls)
            elif type(initial_ls) is str:
                initial_ls = torch.Tensor(ast.literal_eval(initial_ls))
        else:
            initial_ls = 1.0


        self.ssp_parameters = ssp_param(self.input_dim, input_embedding_size,initial_ls,
                                        basis_type, stay_hex,
                                        learn_phase,learn_ls,rng)

        self._features_dim = self.ssp_parameters._features_dim
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

        if hidden_size>0:
            self.layers =  mlp(self._features_dim, hidden_size, activation_fun, input_embedding_size)
            self.input_embedding_size = input_embedding_size
        else:
            self.layers = torch.nn.Identity()
            self.input_embedding_size = self._features_dim


        self.other = FeatureProcesser(obs_space, self.input_embedding_size,
                                 use_text=use_text, use_memory=use_memory, normalize=normalize)
        self.embedding_size = self.other.embedding_size


    def _encode(self, x):
        ls_mat = self.ssp_parameters.get_ls_mat().to(x.device)
        F = self.ssp_parameters.generate_matrix().to(x.device)
        x = (F @ (x @ ls_mat).T).type(torch.complex64) # fix .to(x.device)
        x = torch.fft.ifft( torch.exp( 1.j * x), axis=0 ).real.T
        return x

    def _bind(self,a,b):
        return torch.fft.ifft(torch.fft.fft(a, axis=-1) * torch.fft.fft(b, axis=-1), axis=-1).real

    def forward(self, obs, memory):
        x = obs.image[:,:self.input_dim]
        M = self._encode(x)

        obj_sps = obs.image[:,self.input_dim:-self._features_dim].reshape(obs.image.shape[0],-1,self._features_dim)
        ssp_grid_pts = self._encode(self.unroll_grid_pts.to(x.device)).reshape(1, self.n_pts, self._features_dim)
        M += torch.sum(self._bind(obj_sps, ssp_grid_pts), axis=1)
        M += obs.image[:,-self._features_dim:]

        x = self.layers(M)
        embedding, memory, embed_txt = self.other(obs, x, memory)
        return embedding, memory, embed_txt

class SSPLangInput(nn.Module): # only for minigrid
    def __init__(self, obs_space,use_memory,  use_text, normalize,
                 input_embedding_size=151, hidden_size=0, activation_fun='relu', basis_type='hex', rng=np.random.default_rng(0),
                 stay_hex=False,learn_phase=True,learn_ls=True,initial_ls=None, **kwargs):
        super().__init__()
        self.input_dim = 3

        if 'ssp_dim' in kwargs:
            input_embedding_size = kwargs['ssp_dim']
        else:
            input_embedding_size = input_embedding_size


        if 'ssp_h' in kwargs:
            initial_ls = kwargs['ssp_h']
            if (type(initial_ls) is np.ndarray):
                initial_ls = torch.Tensor(initial_ls.flatten())
            elif (type(initial_ls) is list):
                initial_ls = torch.Tensor(initial_ls)
            elif type(initial_ls) is str:
                initial_ls = torch.Tensor(ast.literal_eval(initial_ls))
        else:
            initial_ls = 1.0


        self.ssp_parameters = ssp_param(self.input_dim, input_embedding_size,initial_ls,
                                        basis_type, stay_hex,
                                        learn_phase,learn_ls,rng)

        self._features_dim = self.ssp_parameters._features_dim

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

        if hidden_size>0:
            self.layers =  mlp(2*self._features_dim, hidden_size, activation_fun, input_embedding_size)
            self.input_embedding_size = input_embedding_size
        else:
            self.layers = torch.nn.Identity()
            self.input_embedding_size = 2*self._features_dim



    def _encode(self, x):
        ls_mat = self.ssp_parameters.get_ls_mat().to(x.device)
        F = self.ssp_parameters.generate_matrix().to(x.device)
        x = (F @ (x @ ls_mat).T).type(torch.complex64) # fix .to(x.device)
        x = torch.fft.ifft( torch.exp( 1.j * x), axis=0 ).real.T
        return x

    def _bind(self,a,b):
        return torch.fft.ifft(torch.fft.fft(a, axis=-1) * torch.fft.fft(b, axis=-1), axis=-1).real

    def forward(self, obs, memory):
        x = obs.image[:,:self.input_dim]
        M = self._encode(x)

        obj_sps = obs.image[:,self.input_dim:-2*self._features_dim].reshape(obs.image.shape[0],-1,self._features_dim)
        ssp_grid_pts = self._encode(self.unroll_grid_pts.to(x.device)).reshape(1, self.n_pts, self._features_dim)
        M += torch.sum(self._bind(obj_sps, ssp_grid_pts), axis=1)
        M += obs.image[:,-2*self._features_dim:-self._features_dim]

        O = obs.image[:,-self._features_dim:]

        x = self.layers(torch.hstack([M,O]))
        # x = self.layers(M+O)
        embedding, memory, embed_txt = self.other(obs, x, memory)
        return embedding, memory, embed_txt



def mlp(*layers: tp.Sequence[tp.Union[int, str]]) -> nn.Sequential:
    """Provides a sequence of linear layers and non-linearities
    providing a sequence of dimension for the neurons, or name of
    the non-linearities
    Eg: mlp(10, 12, "relu", 15) returns:
    Sequential(Linear(10, 12), ReLU(), Linear(12, 15))
    """
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert np.issubdtype(type(layers[0]), int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(_nl(layer, prev_dim))
        else:
            assert np.issubdtype(type(layer), int)
            sequence.append(nn.Linear(prev_dim, layer))
            prev_dim = layer
    return nn.Sequential(*sequence)

class _L2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = torch.sqrt(self.dim) * F.normalize(x, dim=1)
        return y

def _nl(name: str, dim: int) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension"""
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [nn.LayerNorm(dim), nn.Tanh()]
    if name == "layernorm":
        return [nn.LayerNorm(dim)]
    if name == "tanh":
        return [nn.Tanh()]
    if name == "L2":
        return [_L2(dim)]
    raise ValueError(f"Unknown non-linearity {name}")