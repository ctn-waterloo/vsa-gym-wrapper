"""
Tests for PyTorch-based SSP feature extractors.
"""

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from vsagym.networks.ssp_feature_extractor import SSPFeaturesExtractor
from vsagym.networks.hex_feature_extractor import SSPHexFeaturesExtractor
from vsagym.networks.minigrid_feature_extractor import (
    SSPMiniGridViewFeatures,
    SSPMiniGridMissionFeatures,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box(low, high, shape):
    return Box(low=np.full(shape, low), high=np.full(shape, high))


# ---------------------------------------------------------------------------
# SSPFeaturesExtractor
# ---------------------------------------------------------------------------


class TestSSPFeaturesExtractor:
    @pytest.fixture(scope="class")
    def extractor(self):
        obs_space = _box(-1.0, 1.0, (4,))
        return SSPFeaturesExtractor(obs_space, features_dim=51, basis_type="hex",
                                    rng=0, learn_phase_matrix=True, learn_ls=True)

    def test_output_shape(self, extractor):
        x = torch.randn(8, 4)
        out = extractor(x)
        assert out.shape == (8, extractor.features_dim)

    def test_features_dim_property(self, extractor):
        assert extractor.features_dim == extractor.ssp_dim

    def test_forward_is_differentiable(self, extractor):
        x = torch.randn(4, 4, requires_grad=False)
        out = extractor(x)
        loss = out.sum()
        loss.backward()
        assert extractor.phase_matrix.grad is not None

    def test_rand_basis_type(self):
        obs_space = _box(-1.0, 1.0, (3,))
        ext = SSPFeaturesExtractor(obs_space, features_dim=51, basis_type="rand",
                                   rng=0)
        x = torch.randn(5, 3)
        out = ext(x)
        assert out.shape == (5, ext.features_dim)

    def test_invalid_basis_type(self):
        obs_space = _box(-1.0, 1.0, (3,))
        with pytest.raises(ValueError):
            SSPFeaturesExtractor(obs_space, features_dim=51, basis_type="bad")

    def test_custom_length_scale(self):
        obs_space = _box(-1.0, 1.0, (2,))
        ext = SSPFeaturesExtractor(obs_space, features_dim=51, basis_type="hex",
                                   length_scale=0.5, rng=0)
        x = torch.randn(4, 2)
        out = ext(x)
        assert out.shape[0] == 4

    def test_fixed_phase_matrix(self):
        obs_space = _box(-1.0, 1.0, (2,))
        ext = SSPFeaturesExtractor(obs_space, features_dim=51, basis_type="hex",
                                   learn_phase_matrix=False, rng=0)
        assert not ext.phase_matrix.requires_grad

    def test_fixed_ls(self):
        obs_space = _box(-1.0, 1.0, (2,))
        ext = SSPFeaturesExtractor(obs_space, features_dim=51, basis_type="hex",
                                   learn_ls=False, rng=0)
        assert not ext.length_scale.requires_grad

    def test_batch_size_one(self, extractor):
        x = torch.randn(1, 4)
        out = extractor(x)
        assert out.shape == (1, extractor.features_dim)


# ---------------------------------------------------------------------------
# SSPHexFeaturesExtractor
# ---------------------------------------------------------------------------


class TestSSPHexFeaturesExtractor:
    @pytest.fixture(scope="class")
    def extractor(self):
        obs_space = _box(-1.0, 1.0, (2,))
        return SSPHexFeaturesExtractor(obs_space, features_dim=51, rng=0)

    def test_output_shape(self, extractor):
        x = torch.randn(6, 2)
        out = extractor(x)
        assert out.shape == (6, extractor.features_dim)

    def test_has_scales_param(self, extractor):
        assert hasattr(extractor, "scales")
        assert extractor.scales.requires_grad

    def test_has_rotation_learners(self, extractor):
        assert hasattr(extractor, "rotation_learners")

    def test_differentiable(self, extractor):
        x = torch.randn(4, 2)
        out = extractor(x)
        loss = out.sum()
        loss.backward()
        assert extractor.scales.grad is not None

    def test_input_dim_override(self):
        obs_space = _box(-1.0, 1.0, (10,))  # larger obs space
        ext = SSPHexFeaturesExtractor(obs_space, features_dim=51,
                                      input_dim=2, rng=0)
        x = torch.randn(3, 2)  # only 2-D input despite obs space being 10-D
        out = ext(x)
        assert out.shape[0] == 3


# ---------------------------------------------------------------------------
# SSPMiniGridViewFeatures
# ---------------------------------------------------------------------------


def _make_minigrid_view_extractor(view_w=7, view_h=7, features_dim=51,
                                   basis_type="hex", rng=0):
    """Create a SSPMiniGridViewFeatures extractor.
    The actual ssp_dim may differ from features_dim (HexSSP adjusts).
    Build a large dummy obs_space first, then query the real ssp_dim."""
    INPUT_DIM = 3
    n_pts = view_w * view_h
    # Use a safe over-size dummy obs space
    dummy_dim = INPUT_DIM + n_pts * features_dim + features_dim
    obs_space = _box(-1.0, float(max(view_w, view_h)), (dummy_dim,))
    # Create extractor — it will store the real ssp_dim
    ext = SSPMiniGridViewFeatures(
        obs_space, features_dim=features_dim,
        basis_type=basis_type, rng=rng,
        view_width=view_w, view_height=view_h,
    )
    # Re-create obs_space with the correct ssp_dim
    real_dim = INPUT_DIM + n_pts * ext.ssp_dim + ext.ssp_dim
    real_obs_space = _box(-1.0, float(max(view_w, view_h)), (real_dim,))
    return SSPMiniGridViewFeatures(
        real_obs_space, features_dim=features_dim,
        basis_type=basis_type, rng=rng,
        view_width=view_w, view_height=view_h,
    )


class TestSSPMiniGridViewFeatures:
    VIEW_W = 7
    VIEW_H = 7
    INPUT_DIM = 3

    @pytest.fixture(scope="class")
    def extractor(self):
        return _make_minigrid_view_extractor()

    def _make_fake_obs(self, extractor, batch_size=4):
        n_pts = self.VIEW_W * self.VIEW_H
        obs_dim = self.INPUT_DIM + n_pts * extractor.ssp_dim + extractor.ssp_dim
        return torch.randn(batch_size, obs_dim)

    def test_output_shape(self, extractor):
        obs = self._make_fake_obs(extractor)
        out = extractor(obs)
        assert out.shape == (4, extractor.ssp_dim)

    def test_output_normalized(self, extractor):
        obs = self._make_fake_obs(extractor, 2)
        out = extractor(obs)
        norms = torch.linalg.vector_norm(out, dim=-1)
        # Output should be approximately unit norm
        for norm in norms:
            assert float(norm) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# SSPMiniGridMissionFeatures
# ---------------------------------------------------------------------------


class TestSSPMiniGridMissionFeatures:
    VIEW_W = 7
    VIEW_H = 7
    INPUT_DIM = 3

    @pytest.fixture(scope="class")
    def extractor(self):
        # Build with dummy obs_space, query actual ssp_dim, rebuild with correct obs_space
        n_pts = self.VIEW_W * self.VIEW_H
        features_dim = 51
        dummy_dim = self.INPUT_DIM + n_pts * features_dim + 2 * features_dim
        obs_space = _box(-1.0, float(max(self.VIEW_W, self.VIEW_H)), (dummy_dim,))
        ext_tmp = SSPMiniGridMissionFeatures(obs_space, features_dim=features_dim,
                                             basis_type="hex", rng=0)
        real_dim = self.INPUT_DIM + n_pts * ext_tmp.ssp_dim + 2 * ext_tmp.ssp_dim
        real_obs_space = _box(-1.0, float(max(self.VIEW_W, self.VIEW_H)), (real_dim,))
        return SSPMiniGridMissionFeatures(real_obs_space, features_dim=features_dim,
                                          basis_type="hex", rng=0)

    def _make_fake_obs(self, extractor, batch_size=4):
        n_pts = self.VIEW_W * self.VIEW_H
        obs_dim = self.INPUT_DIM + n_pts * extractor.ssp_dim + 2 * extractor.ssp_dim
        return torch.randn(batch_size, obs_dim)

    def test_output_shape(self, extractor):
        obs = self._make_fake_obs(extractor)
        out = extractor(obs)
        # Mission features double the output dim
        assert out.shape == (4, 2 * extractor.ssp_dim)
