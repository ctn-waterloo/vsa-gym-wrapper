import warnings
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
import numpy as np
from typing import Optional, Union
from ..spaces.ssp_box import SSPBox
from ..spaces.ssp_discrete import SSPDiscrete
from ..spaces.ssp_sequence import SSPSequence
from ..spaces.ssp_dict import SSPDict

try:
    import gymnasium_robotics
except:
    warnings.warn("Can't import gymnasium_robotics. SSPGoalWrapper will not work.")


# Code in this file is experimental


class SSPGoalEnvWrapper(gym.ObservationWrapper):
    def __init__(
            self,
            env: gymnasium_robotics.GoalEnv,
            ssp_obs_space: Optional[Union[SSPBox, SSPDiscrete, SSPSequence, SSPDict]] = None,
            ssp_goal_space: Optional[Union[SSPBox, SSPDiscrete, SSPSequence, SSPDict]] = None,
            shape_out: Optional[int] = None,
            **kwargs
    ):

        gym.Wrapper.__init__(self, env)

        if shape_out is not None:
            assert (type(shape_out) is int), f"Expects `shape_out` to be an integer, actual type: {type(shape_out)}"

        self.shape_in = env.observation_space.shape
        # Set-up observation space
        if ssp_obs_space is not None:
            self.ssp_obs_space = ssp_obs_space
        else:
            if type(env.observation_space['observation']) is Box:
                self.ssp_obs_space = SSPBox(
                    low=env.observation_space['observation'].low,
                    high=env.observation_space['observation'].high,
                    shape_in=env.observation_space['observation'].shape,
                    shape_out=shape_out,
                    dtype=env.observation_space['observation'].dtype,  #seed = env.observation_space.seed()[0],
                    **kwargs)
            elif type(env.observation_space['observation']) is Discrete:
                self.ssp_obs_space = SSPDiscrete(
                    n=env.observation_space['observation'].n,
                    shape_out=shape_out,
                    dtype=env.observation_space['observation'].dtype,  #seed = env.observation_space.seed()[0],
                    start=env.observation_space['observation'].start)
            else:
                NotImplementedError(
                    f'Unsupported observation space for automatic SSP conversion: {env.observation_space["observation"]}')
        if ssp_goal_space is not None:
            self.ssp_goal_space = ssp_goal_space
        else:
            if type(env.observation_space['desired_goal']) is Box:
                self.ssp_goal_space = SSPBox(
                    low=env.observation_space['desired_goal'].low,
                    high=env.observation_space['desired_goal'].high,
                    shape_in=env.observation_space['desired_goal'].shape,
                    shape_out=shape_out,
                    dtype=env.observation_space['desired_goal'].dtype,  # seed = env.observation_space.seed()[0],
                    **kwargs)
            elif type(env.observation_space['desired_goal']) is Discrete:
                self.ssp_goal_space = SSPDiscrete(
                    n=env.observation_space['desired_goal'].n,
                    shape_out=shape_out,
                    dtype=env.observation_space['desired_goal'].dtype,  # seed = env.observation_space.seed()[0],
                    start=env.observation_space['desired_goal'].start)
            else:
                NotImplementedError(
                    f'Unsupported observation space for automatic SSP conversion: {env.observation_space["desired_goal"]}')
        self.shape_out = self.ssp_obs_space.shape_out + self.ssp_goal_space.shape_out
        self.observation_space = Dict(
            {
                "observation": Box(low=-np.ones(self.shape_out), high=np.ones(self.shape_out),dtype=self.ssp_obs_space.dtype),
                "desired_goal": Box(low=-np.ones(self.shape_out), high=np.ones(self.shape_out),dtype=self.ssp_goal_space.dtype),
                "achieved_goal": Box(low=-np.ones(self.shape_out), high=np.ones(self.shape_out),dtype=self.ssp_goal_space.dtype)
            }
        )

    def observation(self, obs):
        obs_ssp = self.ssp_obs_space.encode(obs['observation']).squeeze()
        desired_goal_ssp = self.ssp_goal_space.encode(obs['desired_goal']).squeeze()
        achieved_goal_ssp = self.ssp_goal_space.encode(obs['achieved_goal']).squeeze()
        return dict({'observation': obs_ssp,
                     'desired_goal': desired_goal_ssp,
                     'achieved_goal': achieved_goal_ssp})
        # return np.concatenate([obs_ssp, desired_goal_ssp, achieved_goal_ssp])

# class SSPRewardWrapper(gym.Wrapper):
#     def __init__(
#             self,
#             env: gym.Env,
#             ssp_space=None,
#             shape_out=None,
#             dissim_coef=0.0,
#             **kwargs
#     ):
#
#         gym.Wrapper.__init__(self, env)
#
#         if shape_out is not None:
#             assert (type(shape_out) is int), f"Expects `shape_out` to be an integer, actual type: {type(shape_out)}"
#
#         self.shape_in = env.observation_space.shape
#         # Set-up observation space
#         if ssp_space is not None:
#             self.ssp_obs_space = ssp_space
#         else:
#             if type(env.observation_space) == Box:
#                 self.ssp_obs_space = SSPBox(
#                     low=env.observation_space.low,
#                     high=env.observation_space.high,
#                     shape_in=env.observation_space.shape,
#                     shape_out=shape_out,
#                     dtype=env.observation_space.dtype,  #seed = env.observation_space.seed()[0],
#                     **kwargs)
#             elif type(env.observation_space) == Discrete:
#                 self.ssp_obs_space = SSPDiscrete(
#                     n=env.observation_space.n,
#                     shape_out=shape_out,
#                     dtype=env.observation_space.dtype,  #seed = env.observation_space.seed()[0],
#                     start=env.observation_space.start)
#             else:
#                 NotImplementedError(
#                     f'Unsupported observation space for automatic SSP conversion: {env.observation_space}')
#         self.shape_out = self.ssp_obs_space.shape_out
#         self.observation_space = Box(low=-np.ones(self.shape_out), high=np.ones(self.shape_out))
#         self.mu = np.zeros(self.shape_out)
#         self.dissim_coef = dissim_coef
#
#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         ssp_obs = self.ssp_obs_space.encode(obs).squeeze()
#
#         intrinsic_reward = np.clip(1 - np.dot(ssp_obs, self.mu), 0, 1)
#         reward = reward + self.dissim_coef * intrinsic_reward
#
#         self.mu += ssp_obs
#         self.mu /= np.linalg.norm(self.mu)
#         return ssp_obs, reward, terminated, truncated, info
#
#     def reset(self, **kwargs):
#         obs, reset_info = self.env.reset(**kwargs)
#         ssp_obs = self.ssp_obs_space.encode(obs).squeeze()
#         self.mu = ssp_obs
#         return ssp_obs, reset_info
