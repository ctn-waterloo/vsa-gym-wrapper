import numpy as np
import gymnasium as gym
from vsagym import wrappers

def test_sspenv_wrapper():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = wrappers.SSPEnvWrapper(env,
                                 auto_convert_obs_space=True,
                                 auto_convert_action_space=True,
                                 shape_out=251, decoder_method='from-set',
                                 length_scale=0.1)
    observation, _ = env.reset()
    assert observation.shape == (251,)
    for t in range(5):
        action = env.action_space.sample()
        assert action.shape == (251,)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated or t==4:
            observation, _ = env.reset()
    env.close();

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = wrappers.SSPEnvWrapper(env,
                                 auto_convert_obs_space=True,
                                 auto_convert_action_space=False,
                                 shape_out=251, decoder_method='from-set')
    observation, _ = env.reset()
    for t in range(5):
        action = env.action_space.sample()
        assert type(action) is np.int64
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated or t == 4:
            observation, _ = env.reset()
    env.close();

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = wrappers.SSPEnvWrapper(env,
                                 auto_convert_obs_space=False,
                                 auto_convert_action_space=True,
                                 shape_out=251, decoder_method='from-set')
    observation, _ = env.reset()
    assert observation.shape == (4,)
    for t in range(5):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated or t == 4:
            observation, _ = env.reset()
    env.close();

def test_sspobs_wrapper():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = wrappers.SSPObsWrapper(env,
                                 shape_out=251,
                                 decoder_method='from-set')
    observation, _ = env.reset()
    assert observation.shape == (251,)
    for t in range(5):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated or t==4:
            observation, _ = env.reset()
    env.close();


