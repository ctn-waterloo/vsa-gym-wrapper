import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
import minigrid
from vsagym import networks
from vsagym.wrappers import minigrid_wrappers

def test_pose_network():
    env = gym.make('MiniGrid-Empty-5x5-v0')
    env = minigrid_wrappers.MiniGridPoseWrapper(env, shape_out=251)
    env = minigrid_wrappers.FlatWrapper(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=dict(features_extractor_class=networks.SSPFeaturesExtractor,
                           features_extractor_kwargs={'features_dim': 251,
                                                      'length_scale': [1.,1.,0.1],
                                                      'input_dim': 3}),
    )
    model.learn(total_timesteps=500)

def test_view_network():
    env = gym.make('MiniGrid-KeyCorridorS3R1-v0')
    env = minigrid_wrappers.PrepMiniGridViewWrapper(env, shape_out=201)
    env = minigrid_wrappers.FlatWrapper(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=dict(features_extractor_class=networks.SSPMiniGridViewFeatures,
                           features_extractor_kwargs={'features_dim': 251,
                                                      'length_scale': [1.,1.,0.1],
                                                      'basis_type': 'hex'}),
    )
    model.learn(total_timesteps=500)

def test_mission_network():
    env = gym.make('MiniGrid-KeyCorridorS3R1-v0')
    env = minigrid_wrappers.PrepMiniGridMissionWrapper(env, shape_out=201)
    env = minigrid_wrappers.FlatWrapper(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=dict(features_extractor_class=networks.SSPMiniGridMissionFeatures,
                           features_extractor_kwargs={'features_dim': 251,
                                                      'length_scale': [1.,1.,0.1],
                                                      'basis_type': 'hex'}),
    )
    model.learn(total_timesteps=500)
