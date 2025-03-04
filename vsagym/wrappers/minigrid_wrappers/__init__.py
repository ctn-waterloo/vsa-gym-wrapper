from .minigrid_flat_wrapper import FlatWrapper
from .minigrid_pose_wrapper import SSPMiniGridPoseWrapper, MiniGridPoseWrapper
from .minigrid_view_wrapper import SSPMiniGridViewWrapper, PrepMiniGridViewWrapper
from .minigrid_mission_wrapper import SSPMiniGridMissionWrapper, PrepMiniGridMissionWrapper

import gymnasium as gym
def SSPMiniGridWrapper(env: gym.Env,
                       encode_pose: int = True,
                       encode_view: int = True,
                       encode_mission: int = True,
                       **kwargs):
    if encode_pose & encode_view & encode_mission:
        return SSPMiniGridMissionWrapper(env, **kwargs)
    elif encode_pose & encode_view:
        return SSPMiniGridViewWrapper(env, **kwargs)
    elif encode_pose:
        return SSPMiniGridPoseWrapper(env, **kwargs)
    elif encode_view:
        kwargs['pose_weight'] = 0.0
        return SSPMiniGridViewWrapper(env, **kwargs)
    else:
        raise NotImplementedError
