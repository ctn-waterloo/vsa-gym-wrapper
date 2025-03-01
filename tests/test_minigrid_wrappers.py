import numpy as np
import gymnasium as gym
from vsagym.wrappers import minigrid_wrappers

def test_mg_wrappers():
    env = gym.make('MiniGrid-Empty-5x5-v0')
    env = minigrid_wrappers.SSPMiniGridPoseWrapper(env,
                                 shape_out=251,
                                 decoder_method='from-set')
    observation, _ = env.reset()
    for t in range(5):
        action = env.action_space.sample()
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated or t == 4:
            observation, _ = env.reset()
    env.close()

    env = gym.make('MiniGrid-KeyCorridorS3R1-v0')
    env = minigrid_wrappers.SSPMiniGridViewWrapper(env,
                                                   obj_encoding='allbound',
                                                   view_type='local',
                                                   shape_out=251,
                                                   decoder_method='from-set')
    observation, _ = env.reset()
    for t in range(5):
        action = env.action_space.sample()
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated or t == 4:
            observation, _ = env.reset()
    env.close()

    env = gym.make('MiniGrid-KeyCorridorS3R1-v0', render_mode='rgb_array')
    env = minigrid_wrappers.SSPMiniGridViewWrapper(env,
                                                   obj_encoding='allbound',
                                                   view_type='global',
                                                   shape_out=251,
                                                   decoder_method='from-set')
    observation, _ = env.reset()
    for t in range(5):
        action = env.action_space.sample()
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated or t == 4:
            observation, _ = env.reset()
    env.close()

    env = gym.make('MiniGrid-KeyCorridorS3R1-v0')
    env = minigrid_wrappers.SSPMiniGridViewWrapper(env,
                                                   obj_encoding='slotfiller',
                                                   view_type='local',
                                                   shape_out=251,
                                                   decoder_method='from-set')
    observation, _ = env.reset()
    for t in range(5):
        action = env.action_space.sample()
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated or t == 4:
            observation, _ = env.reset()
    env.close()


    env = gym.make('MiniGrid-Dynamic-Obstacles-5x5-v0')
    env = minigrid_wrappers.SSPMiniGridMissionWrapper(env,
                                                   shape_out=251,
                                                   decoder_method='from-set')
    observation, _ = env.reset()
    for t in range(5):
        action = env.action_space.sample()
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated or t == 4:
            observation, _ = env.reset()
    env.close()

    env = gym.make('MiniGrid-Dynamic-Obstacles-5x5-v0')
    env = minigrid_wrappers.SSPMiniGridWrapper(env,shape_out=251,
                    encode_pose=False,encode_view=True,encode_mission=False)
    observation, _ = env.reset()
    for t in range(5):
        action = env.action_space.sample()
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated or t == 4:
            observation, _ = env.reset()
    env.close()





