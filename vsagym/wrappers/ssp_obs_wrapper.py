import types
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
from typing import Optional, Union
from ..spaces.ssp_box import SSPBox
from ..spaces.ssp_discrete import SSPDiscrete
from ..spaces.ssp_sequence import SSPSequence
from ..spaces.ssp_dict import SSPDict

class SSPObsWrapper(gym.ObservationWrapper):
    r"""A gymnasium observation wrapper that converts data to a VSA/SSP space.
    """
    def __init__(
            self,
            env: gym.Env,
            ssp_space: Optional[Union[SSPBox, SSPDiscrete, SSPSequence, SSPDict]] = None,
            shape_out: Optional[int] = None,
            **kwargs
    ):
        r"""Constructor of :class:`SSPObsWrapper`.

        Args:
            env (Env): The environment
            ssp_space (SSPBox | SSPDiscrete | SSPSequence | SSPDict | None):
                    If None, an SSP space will be generated automatically from the env's obs space
                    (note: only works for Box or Discrete spaces)
            shape_out (int): Shape of the SSP space. Only used if ssp_space is None
        """
        gym.Wrapper.__init__(self, env)

        if shape_out is not None:
            assert (type(shape_out) is int), f"Expects `shape_out` to be an integer, actual type: {type(shape_out)}"

        self.shape_in = env.observation_space.shape
        # Set-up observation space
        if ssp_space is not None:
            self.ssp_obs_space = ssp_space
        else:
            if type(env.observation_space) == Box:
                self.ssp_obs_space = SSPBox(
                    low=env.observation_space.low,
                    high=env.observation_space.high,
                    shape_in=env.observation_space.shape,
                    shape_out=shape_out,
                    dtype=env.observation_space.dtype,
                    **kwargs)
            elif type(env.observation_space) == Discrete:
                self.ssp_obs_space = SSPDiscrete(
                    n=env.observation_space.n,
                    shape_out=shape_out,
                    dtype=env.observation_space.dtype,
                    start=env.observation_space.start)
            else:
                NotImplementedError(
                    f'Unsupported observation space for automatic SSP conversion: {env.observation_space}')
        self.shape_out = self.ssp_obs_space.shape_out
        self.observation_space = Box(low=-np.ones(self.shape_out), high=np.ones(self.shape_out),
                                     dtype=self.ssp_obs_space.dtype)

    def observation(self, obs):
        return self.ssp_obs_space.encode(obs).squeeze()


