"""
Tests for all gymnasium wrappers:
    SSPObsWrapper, SSPEnvWrapper, SSPGoalEnvWrapper,
    SSPMiniGridPoseWrapper, SSPMiniGridViewWrapper,
    SSPMiniGridMissionWrapper, SSPMiniGridWrapper
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from vsagym import wrappers
from vsagym.spaces.ssp_box import SSPBox
from vsagym.spaces.ssp_discrete import SSPDiscrete
from vsagym.spaces.ssp_sequence import SSPSequence

minigrid = pytest.importorskip("minigrid", reason="minigrid not installed")

# ---------------------------------------------------------------------------
# SSPObsWrapper
# ---------------------------------------------------------------------------


class TestSSPObsWrapper:
    """Tests for the observation-only wrapper (SSPObsWrapper)."""

    def _cartpole(self):
        return gym.make("CartPole-v1", render_mode=None)

    def _frozenlake(self):
        return gym.make("FrozenLake-v1", render_mode=None)

    def test_box_obs_is_1d(self):
        env = wrappers.SSPObsWrapper(self._cartpole(), shape_out=101)
        obs, _ = env.reset(seed=0)
        # HexSSP may adjust shape_out to nearest admissible value;
        # check that obs is 1-D and larger than raw 4-D CartPole obs.
        assert obs.ndim == 1
        assert obs.shape[0] > 4
        env.close()

    def test_box_obs_matches_obs_space(self):
        env = wrappers.SSPObsWrapper(self._cartpole(), shape_out=101)
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_box_obs_dtype(self):
        env = wrappers.SSPObsWrapper(self._cartpole(), shape_out=51)
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32
        env.close()

    def test_discrete_obs_shape(self):
        env = wrappers.SSPObsWrapper(self._frozenlake(), shape_out=51)
        obs, _ = env.reset(seed=0)
        # SSPDiscrete gives exactly shape_out (SPSpace has no adjustment)
        assert obs.shape == (51,)
        env.close()

    def test_observation_space_matches(self):
        env = wrappers.SSPObsWrapper(self._cartpole(), shape_out=51)
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_step_obs_matches_obs_space(self):
        env = wrappers.SSPObsWrapper(self._cartpole(), shape_out=51)
        env.reset(seed=0)
        obs, *_ = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_custom_ssp_space(self):
        base_env = self._cartpole()
        ssp = SSPBox(
            low=base_env.observation_space.low,
            high=base_env.observation_space.high,
            shape_out=101, seed=0
        )
        env = wrappers.SSPObsWrapper(base_env, ssp_space=ssp)
        obs, _ = env.reset(seed=0)
        assert obs.ndim == 1
        assert obs.shape == env.observation_space.shape
        env.close()


# ---------------------------------------------------------------------------
# SSPEnvWrapper (obs + action)
# ---------------------------------------------------------------------------


class TestSSPEnvWrapper:
    """Tests for the full env wrapper (observations and/or actions)."""

    def _cartpole(self):
        return gym.make("CartPole-v1", render_mode=None)

    # ---------- observation only ----------

    def test_obs_only_shape(self):
        env = wrappers.SSPEnvWrapper(
            self._cartpole(),
            auto_convert_obs_space=True,
            auto_convert_action_space=False,
            shape_out=101,
        )
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert obs.ndim == 1 and obs.shape[0] > 4
        env.close()

    def test_obs_only_action_passthrough(self):
        env = wrappers.SSPEnvWrapper(
            self._cartpole(),
            auto_convert_obs_space=True,
            auto_convert_action_space=False,
            shape_out=51,
        )
        env.reset(seed=0)
        action = env.action_space.sample()
        assert isinstance(action, (int, np.integer))
        env.close()

    # ---------- action only ----------

    def test_action_only_obs_passthrough(self):
        env = wrappers.SSPEnvWrapper(
            self._cartpole(),
            auto_convert_obs_space=False,
            auto_convert_action_space=True,
            shape_out=101,
        )
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4,)  # raw CartPole obs
        env.close()

    def test_action_only_action_shape(self):
        env = wrappers.SSPEnvWrapper(
            self._cartpole(),
            auto_convert_obs_space=False,
            auto_convert_action_space=True,
            shape_out=101,
        )
        env.reset(seed=0)
        action = env.action_space.sample()
        assert action.shape == (101,)
        env.close()

    # ---------- both obs + action ----------

    def test_both_obs_shape(self):
        env = wrappers.SSPEnvWrapper(
            self._cartpole(),
            auto_convert_obs_space=True,
            auto_convert_action_space=True,
            shape_out=51,
            decoder_method="from-set",
        )
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_both_step_obs_shape(self):
        env = wrappers.SSPEnvWrapper(
            self._cartpole(),
            auto_convert_obs_space=True,
            auto_convert_action_space=True,
            shape_out=51,
            decoder_method="from-set",
        )
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert "obs" in info
        assert "action" in info
        env.close()

    def test_multiple_steps(self):
        env = wrappers.SSPEnvWrapper(
            self._cartpole(),
            auto_convert_obs_space=True,
            auto_convert_action_space=False,
            shape_out=51,
        )
        obs, _ = env.reset(seed=0)
        obs_shape = env.observation_space.shape
        for _ in range(10):
            obs, reward, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                obs, _ = env.reset()
        assert obs.shape == obs_shape
        env.close()

    def test_observation_space_contains(self):
        env = wrappers.SSPEnvWrapper(
            self._cartpole(),
            auto_convert_obs_space=True,
            auto_convert_action_space=False,
            shape_out=51,
        )
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        env.close()


# ---------------------------------------------------------------------------
# MiniGrid wrappers
# ---------------------------------------------------------------------------

MINIGRID_ENV = "MiniGrid-Empty-5x5-v0"
SSP_DIM = 51   # small dim for speed


@pytest.fixture(scope="module")
def minigrid_env():
    env = gym.make(MINIGRID_ENV, render_mode=None)
    yield env
    env.close()


def _actual_ssp_dim(env_wrapper):
    """Return the actual SSP output dim the env wrapper uses."""
    return env_wrapper.ssp_obs_space.shape_out


class TestSSPMiniGridPoseWrapper:
    def test_obs_image_is_1d(self):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        env = wrappers.SSPMiniGridPoseWrapper(base, shape_out=SSP_DIM)
        obs, _ = env.reset(seed=0)
        assert obs["image"].ndim == 1
        env.close()

    def test_obs_image_shape_matches_space(self):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        env = wrappers.SSPMiniGridPoseWrapper(base, shape_out=SSP_DIM)
        dim = env.shape_out
        obs, _ = env.reset(seed=0)
        assert obs["image"].shape == (dim,)
        base.close()

    def test_obs_mission_preserved(self):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        env = wrappers.SSPMiniGridPoseWrapper(base, shape_out=SSP_DIM)
        obs, _ = env.reset(seed=0)
        assert "mission" in obs
        base.close()

    def test_step(self):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        env = wrappers.SSPMiniGridPoseWrapper(base, shape_out=SSP_DIM)
        dim = env.shape_out
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs["image"].shape == (dim,)
        base.close()

    def test_custom_ssp_space(self):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        bounds = np.array([[0, 5.0], [0, 5.0], [0, 3.0]])
        ssp = SSPBox(
            low=bounds[:, 0], high=bounds[:, 1],
            shape_in=(3,), shape_out=SSP_DIM, seed=0
        )
        env = wrappers.SSPMiniGridPoseWrapper(base, ssp_space=ssp)
        dim = env.shape_out
        obs, _ = env.reset(seed=0)
        assert obs["image"].shape == (dim,)
        base.close()


class TestSSPMiniGridViewWrapper:
    def _make(self, obj_encoding="allbound", view_type="local"):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        return wrappers.SSPMiniGridViewWrapper(
            base, shape_out=SSP_DIM,
            obj_encoding=obj_encoding, view_type=view_type
        )

    def test_allbound_obs_shape(self):
        env = self._make("allbound")
        dim = env.shape_out
        obs, _ = env.reset(seed=0)
        assert obs["image"].shape == (dim,)
        env.close()

    def test_slotfiller_obs_shape(self):
        env = self._make("slotfiller")
        dim = env.shape_out
        obs, _ = env.reset(seed=0)
        assert obs["image"].shape == (dim,)
        env.close()

    def test_global_view(self):
        env = self._make(view_type="global")
        dim = env.shape_out
        obs, _ = env.reset(seed=0)
        assert obs["image"].shape == (dim,)
        env.close()

    def test_step_allbound(self):
        env = self._make()
        dim = env.shape_out
        env.reset(seed=0)
        obs, *_ = env.step(0)
        assert obs["image"].shape == (dim,)
        env.close()

    def test_invalid_encoding(self):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        with pytest.raises(NotImplementedError):
            wrappers.SSPMiniGridViewWrapper(base, shape_out=SSP_DIM,
                                           obj_encoding="unknown")
        base.close()

    def test_invalid_view_type(self):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        with pytest.raises(NotImplementedError):
            wrappers.SSPMiniGridViewWrapper(base, shape_out=SSP_DIM,
                                           view_type="bad")
        base.close()


class TestSSPMiniGridMissionWrapper:
    def _make(self):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        return wrappers.SSPMiniGridMissionWrapper(base, shape_out=SSP_DIM)

    def test_obs_image_shape(self):
        env = self._make()
        dim = env.shape_out
        obs, _ = env.reset(seed=0)
        assert obs["image"].shape == (2 * dim,)
        env.close()

    def test_step(self):
        env = self._make()
        dim = env.shape_out
        env.reset(seed=0)
        obs, *_ = env.step(0)
        assert obs["image"].shape == (2 * dim,)
        env.close()


class TestSSPMiniGridWrapper:
    """Tests for the factory function SSPMiniGridWrapper."""

    def test_pose_only(self):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        env = wrappers.SSPMiniGridWrapper(
            base, encode_pose=True, encode_view=False,
            encode_mission=False, shape_out=SSP_DIM
        )
        dim = env.shape_out
        obs, _ = env.reset(seed=0)
        assert obs["image"].shape == (dim,)
        env.close()

    def test_pose_and_view(self):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        env = wrappers.SSPMiniGridWrapper(
            base, encode_pose=True, encode_view=True,
            encode_mission=False, shape_out=SSP_DIM
        )
        dim = env.shape_out
        obs, _ = env.reset(seed=0)
        assert obs["image"].shape == (dim,)
        env.close()

    def test_full(self):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        env = wrappers.SSPMiniGridWrapper(
            base, encode_pose=True, encode_view=True,
            encode_mission=True, shape_out=SSP_DIM
        )
        dim = env.shape_out
        obs, _ = env.reset(seed=0)
        assert obs["image"].shape == (2 * dim,)
        env.close()

    def test_view_only_no_pose(self):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        env = wrappers.SSPMiniGridWrapper(
            base, encode_pose=False, encode_view=True,
            encode_mission=False, shape_out=SSP_DIM
        )
        dim = env.shape_out
        obs, _ = env.reset(seed=0)
        assert obs["image"].shape == (dim,)
        env.close()


class TestFlatWrapper:
    def test_flat_obs_shape(self):
        base = gym.make(MINIGRID_ENV, render_mode=None)
        pose_env = wrappers.SSPMiniGridPoseWrapper(base, shape_out=SSP_DIM)
        dim = pose_env.shape_out
        env = wrappers.FlatWrapper(pose_env)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (dim,)
        env.close()
