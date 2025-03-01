import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from typing import Optional, Union
from vsagym.spaces.ssp_box import SSPBox
from vsagym.spaces.ssp_discrete import SSPDiscrete
from vsagym.spaces.ssp_sequence import SSPSequence
from vsagym.spaces.ssp_dict import SSPDict


class SSPMiniWorldPoseWrapper(gym.ObservationWrapper):
    r"""A gymnasium observation wrapper that converts an agent's pose (x,y,orientation)
     to an SSP embedding. For the MiniWorld envs.
    """
    def __init__(self,
                 env: gym.Env,
                 ssp_space: Optional[Union[SSPBox, SSPDiscrete, SSPSequence, SSPDict]] = None,
                 shape_out: Optional[int] = None,
                 **kwargs
                 ):
        r"""Constructor of :class:`SSPMiniWorldXYWrapper`.

        Args:
            env (Env): The MiniWorld environment
            ssp_space (SSPBox | SSPDiscrete | SSPSequence | SSPDict | None):
                    If None, an SSP space will be generated
            shape_out (int): Shape of the SSP space. Only used if ssp_space is None
        """
        gym.Wrapper.__init__(self, env)

        if shape_out is not None:
            assert (type(shape_out) is int), f"Expects `shape_out` to be an integer, actual type: {type(shape_out)}"

        # Set-up observation space
        if type(ssp_space) is SSPBox:
            self.ssp_obs_space = ssp_space
        else:
            self.ssp_space = SSPBox(
                    low=np.array([0, env.unwrapped.min_x, env.unwrapped.min_z]),
                    high=np.array([2 * np.pi, env.unwrapped.max_x, env.unwrapped.max_z]),  #order??
                    shape_in=(3,),
                    shape_out=shape_out,
                    ssp_space=ssp_space, # can instead provide a SSPSpace object
                    **kwargs)

        self.shape_out = self.ssp_obs_space.shape_out
        self.observation_space = Box(low=-np.ones(self.shape_out), high=np.ones(self.shape_out))

    def observation(self, obs):
        ssp_obs = self.ssp_obs_space.encode(np.array([[self.env.unwrapped.agent.pos[0],
                                                self.env.unwrapped.agent.pos[2],
                                                self.env.unwrapped.agent.dir]]))
        return ssp_obs.reshape(-1)
