import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import DQN
from vsagym import networks

def test_sspnet():
    env = gym.make('CartPole-v1')
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=dict(features_extractor_class=networks.SSPFeaturesExtractor,
                           features_extractor_kwargs={'features_dim': 251,
                                                      'length_scale': 0.1}),
    )
    model.learn(total_timesteps=1000)

test_sspnet()