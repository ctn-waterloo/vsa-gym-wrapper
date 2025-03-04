import numpy as np
import gymnasium as gym

from vsagym.spaces.ssp_box import SSPBox
from vsagym.spaces.ssp_discrete import SSPDiscrete
from vsagym.spaces.ssp_sequence import SSPSequence
from vsagym.spaces.ssp_dict import SSPDict
from typing import Optional, Union
from gymnasium.spaces import Box, Discrete


class SSPMiniGridPoseWrapper(gym.ObservationWrapper):
    r"""A gymnasium observation wrapper that converts an agent's pose (orientation,x,y)
     to an SSP embedding. For the MiniGrid envs.
    """
    def __init__(
            self,
            env: gym.Env,
            ssp_space: Optional[Union[SSPBox, SSPDiscrete, SSPSequence, SSPDict]] = None,
            shape_out: Optional[int] = None,
            **kwargs
    ):
        r"""Constructor of :class:`SSPMiniWorldXYWrapper`.

        Args:
            env (Env): The MiniGrid environment
            ssp_space (SSPBox | SSPDiscrete | SSPSequence | SSPDict | None):
                    If None, an SSP space will be generated
            shape_out (int): Shape of the SSP space. Only used if ssp_space is None
        """

        gym.Wrapper.__init__(self, env)

        if shape_out is not None:
            assert (type(shape_out) is int), f"Expects `shape_out` to be an integer, actual type: {type(shape_out)}"

        # Set-up observation space
        self.view_width = env.observation_space['image'].shape[0]
        self.view_height = env.observation_space['image'].shape[1]
        domain_bounds = np.array([[0, env.unwrapped.width],
                                  [0, env.unwrapped.height],
                                  [0, 3]])

        if type(ssp_space) is SSPBox:
            self.ssp_obs_space = ssp_space
        else:
            self.ssp_obs_space = SSPBox(
                low=domain_bounds[:, 0],
                high=domain_bounds[:, 1],
                shape_in=(3,),
                shape_out=shape_out,
                ssp_space=ssp_space,
                **kwargs)

        self.shape_in = 3
        self.shape_out = self.ssp_obs_space.shape_out
        self.observation_space["image"] = Box(low=-np.ones(self.shape_out), high=np.ones(self.shape_out),
                                              dtype=self.ssp_obs_space.dtype)

    def _get_agent_pos(self):
        return np.array([[
            self.env.unwrapped.agent_pos[0],
            self.env.unwrapped.agent_pos[1],
            self.env.unwrapped.agent_dir
        ]])

    def _encode_agent_pos(self):
        return self.ssp_obs_space.encode(self._get_agent_pos())

    def observation(self, obs):
        return {
            'mission': obs['mission'],
            'image': self._encode_agent_pos().flatten()
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class MiniGridPoseWrapper(gym.ObservationWrapper):
    def __init__(
            self,
            env: gym.Env,
            shape_out: Optional[int] = None,
            **kwargs
    ):
        gym.Wrapper.__init__(self, env)
        self.shape_in = 3
        self.shape_out = shape_out
        domain_bounds = np.array([[0, env.unwrapped.width],
                                  [0, env.unwrapped.height],
                                  [0, 3]])
        self.observation_space["image"] = Box(low=domain_bounds[:,0], high=domain_bounds[:,1])

    def observation(self, obs):
        return {
            'mission': obs['mission'],
            'image': np.array([[
                        self.env.unwrapped.agent_pos[0],
                        self.env.unwrapped.agent_pos[1],
                        self.env.unwrapped.agent_dir
                    ]])
        }





