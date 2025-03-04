import numpy as np
import gymnasium as gym
from typing import Optional, Union
from minigrid.core.constants import (
    COLOR_TO_IDX,
    OBJECT_TO_IDX,
    STATE_TO_IDX
)

from vsagym.wrappers.minigrid_wrappers.minigrid_pose_wrapper import SSPMiniGridPoseWrapper

from vsagym.spaces.spspace import SPSpace
from vsagym.spaces.ssp_box import SSPBox
from vsagym.spaces.ssp_discrete import SSPDiscrete
from vsagym.spaces.ssp_sequence import SSPSequence
from vsagym.spaces.ssp_dict import SSPDict
from gymnasium.spaces import Box


    
class SSPMiniGridViewWrapper(SSPMiniGridPoseWrapper):
    r"""A gymnasium observation wrapper that converts an agent's pose (orientation,x,y)
    and local view to a VSA embedding.
    For the MiniGrid envs.
    """
    notice_objs = ['DOOR', 'KEY', 'BALL', 'BOX', 'GOAL', 'LAVA'] # subset of objects that are encoded. optionally: add WALL
    notice_states = ['CLOSED', 'LOCKED']

    idx_to_state = {v: k.upper() for k, v in STATE_TO_IDX.items()}
    idx_to_object = {v: k.upper() for k, v in OBJECT_TO_IDX.items()}
    idx_to_color = {v: k.upper() for k, v in COLOR_TO_IDX.items()}

    def __init__(
        self,
        env: gym.Env,
        ssp_space: Optional[Union[SSPBox, SSPDiscrete, SSPSequence, SSPDict]] = None,
        shape_out: Optional[int] = None,
        obj_encoding: str = 'allbound',
        pose_weight: float = 1.0,
        view_type: str = 'local',
        **kwargs
    ):
        r"""Constructor of :class:`SSPMiniGridViewWrapper`.

        Args:
            env (Env): The MiniGrid environment
            ssp_space (SSPBox | SSPDiscrete | SSPSequence | SSPDict | None):
                    If None, an SSP space will be generated
            shape_out (int): Shape of the SSP space. Only used if ssp_space is None
            obj_encoding (str): Either 'allbound' (default) or 'slotfiller'. Different ways of encoding an object with
                    multiple features
        """
        super().__init__(env, ssp_space, shape_out, **kwargs)
        if shape_out is not None:
            assert (type(shape_out) is int), f"Expects `shape_out` to be an integer, actual type: {type(shape_out)}"

        self.pose_weight = pose_weight

        ## All words/discrete concepts/features to encode
        self.categories = ['OBJ','STATE','COL','LOCATION','HAS']  # upper case to be consistent with nengo_spa, mostly here for legacy reasons
        self.commands = ['GO_TO','PUT','NEXT_TO','OPEN','PICK_UP']
        self.locations = ['FRONT', 'BEHIND', 'LEFT', 'RIGHT']  # currently not used
        self.colors = [x.upper() for x in list(COLOR_TO_IDX.keys())]
        self.color_map = dict(zip(self.colors, self.colors))
        self.objects = [x.upper() for x in list(OBJECT_TO_IDX.keys())]
        self.obj_map = dict(zip(self.objects,
                                [o if o in self.notice_objs else 'I' for o in self.objects]))
        self.states = [x.upper() for x in list(STATE_TO_IDX.keys())]
        self.state_map = dict(zip(self.states,
                                  [s if s in self.notice_states else 'I' for s in self.states])) # I (identity) means ignore the state of an object, but not the object itself

        self.all_names = self.categories + self.commands + self.locations + self.colors + self.notice_objs + self.notice_states
        self.sp_space = SPSpace(len(self.all_names),
                                self.shape_out,
                                names=self.all_names)


        domain_bounds = np.array([[0, self.view_width-1],
                                  [-(self.view_height-1)//2, (self.view_height-1)//2 ],
                                  [0,3]])
        xs = [np.arange(domain_bounds[i,1],domain_bounds[i,0]-1,-1) for i in range(2)]
        xx = np.meshgrid(*xs)
        # xx[0] = 3 - xx[0]
        # xx[1] = 6 - xx[0]
        self.grid_pts = xx
        pts = np.vstack([xx[i].reshape(-1) for i in range(2)]).T
        pts = np.hstack([pts, -1*np.ones((pts.shape[0],1))])
        ssp_pos_grid = self.ssp_obs_space.encode(pts)
        ssp_pos_grid = ssp_pos_grid.reshape(len(xs[0]),len(xs[1]),self.shape_out)
        self.ssp_pos_grid = ssp_pos_grid

        # domain_bounds = np.array([[0, self.view_width - 1],
        #                           [-(self.view_height - 1) // 2, (self.view_height - 1) // 2],
        #                           [0, 3]])
        # xs = [np.arange(domain_bounds[i, 1], domain_bounds[i, 0] - 1, -1) for i in range(2)]
        # xx = np.meshgrid(*xs)
        # xx[0] = 3 - xx[0]
        # xx[1] = 6 - xx[0]
        # self.grid_pts = xx
        # ssp_pos_grid = self.ssp_obs_space.encode(self.grid_pts)
        # ssp_pos_grid = ssp_pos_grid.reshape(len(xs[0]), len(xs[1]), self.shape_out)
        # self.ssp_pos_grid = ssp_pos_grid

        if obj_encoding=='slotfiller':
            self._encode_object = self._encode_object_slotfiller
        elif obj_encoding=='allbound':
            self._encode_object = self._encode_object_allbound
        else:
            raise NotImplementedError


        self.view_type = view_type
        if view_type == 'local':
            self._encode_grid = self._encode_grid_local
            self._get_grid_pos = self._get_grid_pos_local
        elif view_type == 'global':
            self._encode_grid = self._encode_grid_global
            self._get_grid_pos = self._get_grid_pos_global
        else:
            raise NotImplementedError

    def _get_grid_pos_local(self, i, j):
        return np.array([self.grid_pts[0][i, j], self.grid_pts[1][i, j], -1])

    def _get_grid_pos_global(self, i, j):
        x=0
        y=1
        sign_x=1
        sign_y=1
        if self.env.unwrapped.agent_dir == 1:
            x=1;y=0
        elif self.env.unwrapped.agent_dir ==2:
            sign_y=-1
        elif self.env.unwrapped.agent_dir == 3:
            x = 1;y = 0
            sign_x = -1

        return np.array([sign_x*self.grid_pts[x][i, j] + self.env.unwrapped.agent_pos[0],##double check
                         sign_y*self.grid_pts[y][i, j] + self.env.unwrapped.agent_pos[1], -1])

    def _encode_grid_local(self, i, j):
        return self.ssp_pos_grid[i, j, :].copy()

    def _encode_grid_global(self, i, j):
        obj_ssp = self.ssp_obs_space.encode(self._get_grid_pos(i,j))
        return obj_ssp

    def _encode_object_allbound(self, obj_name, col_name, state_name):
        obj_sp = self.sp_space.bind(
            self.sp_space.name_to_vector[obj_name].copy(),
            self.sp_space.name_to_vector[col_name].copy(),
            self.sp_space.name_to_vector[state_name].copy())
        return obj_sp

    def _encode_object_slotfiller(self, obj_name, col_name, state_name):
        obj_sp = self.sp_space.bind(
            self.sp_space.name_to_vector['OBJ'].copy(),
            self.sp_space.name_to_vector[obj_name].copy())
        obj_sp += self.sp_space.bind(
            self.sp_space.name_to_vector['COL'].copy(),
            self.sp_space.name_to_vector[col_name].copy())
        obj_sp += self.sp_space.bind(
            self.sp_space.name_to_vector['STATE'].copy(),
            self.sp_space.name_to_vector[state_name].copy())
        obj_sp = obj_sp/np.linalg.norm(obj_sp, axis=-1, keepdims=True)
        return obj_sp

    def _encode_view(self, img):
        vsa_output = np.zeros(self.shape_out)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                obj = self.idx_to_object[img[i, j, 0]]
                color = self.idx_to_color[img[i, j, 1]]
                state = self.idx_to_state[img[i, j, 2]]
                if obj in self.notice_objs:
                    obj_ssp = self._encode_grid(i,j)
                    obj_name = self.obj_map[obj]
                    col_name = self.color_map[color]
                    state_name = self.state_map[state]
                    
                    obj_vec = self.sp_space.bind(
                        self._encode_object(obj_name, col_name, state_name),
                        obj_ssp)
                    vsa_output += obj_vec.flatten()
        return vsa_output
    
    def _encode_carry(self):
        if self.env.unwrapped.carrying is not None:
            obj_name = self.obj_map[self.env.unwrapped.carrying.type.upper()]
            col_name = self.color_map[self.env.unwrapped.carrying.color.upper()]

            has_sp = self.sp_space.bind(
                self.sp_space.name_to_vector['HAS'].copy(),
                self._encode_object(obj_name, col_name, 'I')
            )
            return has_sp
        else:
            return self.sp_space.name_to_vector['NULL']

    def observation(self, obs): # not vectorized
        img = obs['image']

        vsa_output = self.pose_weight * self._encode_agent_pos()
        vsa_output += self._encode_carry()
        vsa_output += self._encode_view(img)
        vsa_output = vsa_output / np.linalg.norm(vsa_output)

        return {
            'mission': obs['mission'],
            'image': vsa_output.flatten()
        }


class PrepMiniGridViewWrapper(SSPMiniGridViewWrapper):
    r"""Prepare observations into a form ready to encode with a VSA.
    This is meant to be used with SSPMiniGridViewProcesser, a feature extractor network that does the SSP encoding with trainable params
    """
    def __init__(
        self,
        env: gym.Env,
        ssp_space: Optional[Union[SSPBox, SSPDiscrete, SSPSequence, SSPDict]] = None,
        shape_out: Optional[int] = None,
        obj_encoding: str = 'allbound',
        **kwargs
    ):
        
        super().__init__(env, ssp_space, shape_out, obj_encoding, **kwargs)
        n_ssp_dims = (1+self.view_width*self.view_height)
        self.observation_space['image'] = Box(low=-np.ones(n_ssp_dims*self.shape_out + self.shape_in),
                                     high=np.max([env.unwrapped.width, env.unwrapped.height])*np.ones(n_ssp_dims*self.shape_out + self.shape_in),
                                     dtype=self.ssp_obs_space.dtype)

    def _encode_view(self, img):
        obj_poss = []
        obj_sps = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                obj = self.idx_to_object[img[i, j, 0]]
                color = self.idx_to_color[img[i, j, 1]]
                state = self.idx_to_state[img[i, j, 2]]
                obj_pos = self._get_grid_pos(i, j)
                if obj in self.notice_objs:
                    obj_name = self.obj_map[obj]
                    col_name = self.color_map[color]
                    state_name = self.state_map[state]

                    obj_sp = self._encode_object(obj_name, col_name, state_name)
                    obj_sps.append(obj_sp.flatten())
                else:
                    obj_sps.append(self.sp_space.name_to_vector['NULL'].flatten())
                obj_poss.append(obj_pos)
        return obj_poss, obj_sps


    def observation(self, obs):
        img = obs['image']
        agt_pt = np.array([self.env.unwrapped.agent_pos[0],
                            self.env.unwrapped.agent_pos[1],
                            self.env.unwrapped.agent_dir
                             ])
        has_vector = self._encode_carry()
        obj_poss, obj_sps = self._encode_view(img)
        # TODO: pass obj_poss so that the MiniGrid feature extractors support global view too

        return {
            'mission': obs['mission'],
            'image': np.hstack([agt_pt, np.array(obj_sps).flatten(), has_vector.flatten()])
        }
