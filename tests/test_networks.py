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
# NN-modulated encoding parameters (ls_mode / phase_mode = 'nn')
# ---------------------------------------------------------------------------


class TestNNModulatedExtractor:
    @pytest.fixture(scope="class")
    def obs_space(self):
        return _box(-1.0, 1.0, (4,))

    @pytest.fixture(scope="class")
    def static_extractor(self, obs_space):
        return SSPFeaturesExtractor(obs_space, features_dim=51, basis_type="hex",
                                    rng=0)

    @pytest.mark.parametrize("modes", [
        {"ls_mode": "nn"},
        {"phase_mode": "nn"},
        {"ls_mode": "nn", "phase_mode": "nn"},
    ])
    def test_output_shape(self, obs_space, modes):
        ext = SSPFeaturesExtractor(obs_space, features_dim=51, basis_type="hex",
                                   rng=0, **modes)
        x = torch.randn(8, 4)
        out = ext(x)
        assert out.shape == (8, ext.features_dim)

    @pytest.mark.parametrize("modes", [
        {"ls_mode": "nn"},
        {"phase_mode": "nn"},
        {"ls_mode": "nn", "phase_mode": "nn"},
    ])
    def test_init_matches_static_encoding(self, obs_space, static_extractor, modes):
        # The modulation MLPs are zero-initialized, so at init the encoding
        # must equal the static one.
        ext = SSPFeaturesExtractor(obs_space, features_dim=51, basis_type="hex",
                                   rng=0, **modes)
        x = torch.randn(8, 4)
        with torch.no_grad():
            err = (ext(x) - static_extractor(x)).abs().max()
        assert float(err) < 1e-5

    def test_nn_modules_created(self, obs_space):
        ext = SSPFeaturesExtractor(obs_space, features_dim=51, rng=0,
                                   ls_mode="nn", phase_mode="nn")
        assert hasattr(ext, "length_scale_net")
        assert hasattr(ext, "phase_matrix_net")
        assert not hasattr(ext, "length_scale")
        assert not hasattr(ext, "phase_matrix")

    def test_differentiable(self, obs_space):
        ext = SSPFeaturesExtractor(obs_space, features_dim=51, rng=0,
                                   ls_mode="nn", phase_mode="nn")
        x = torch.randn(8, 4)
        (ext(x) ** 2).sum().backward()
        ls_head = ext.length_scale_net.net[-1]
        phase_head = ext.phase_matrix_net.net[-1]
        assert float(ls_head.weight.grad.abs().sum()) > 0
        assert float(phase_head.bias.grad.abs().sum()) > 0

    def test_nn_ls_becomes_input_dependent(self, obs_space):
        ext = SSPFeaturesExtractor(obs_space, features_dim=51, rng=0,
                                   ls_mode="nn")
        x = torch.randn(16, 4)
        opt = torch.optim.SGD(ext.parameters(), lr=1.0)
        for _ in range(3):
            opt.zero_grad()
            (ext(x) ** 2).sum().backward()
            opt.step()
        with torch.no_grad():
            ls_out = ext.length_scale_net(x)
        assert float(ls_out.std(dim=0).sum()) > 0

    def test_invalid_modes(self, obs_space):
        with pytest.raises(ValueError):
            SSPFeaturesExtractor(obs_space, features_dim=51, ls_mode="bad")
        with pytest.raises(ValueError):
            SSPFeaturesExtractor(obs_space, features_dim=51, phase_mode="bad")

    def test_legacy_bool_flags_still_work(self, obs_space):
        ext = SSPFeaturesExtractor(obs_space, features_dim=51, rng=0,
                                   learn_phase_matrix=False, learn_ls=False)
        assert ext.phase_mode == "fixed" and not ext.phase_matrix.requires_grad
        assert ext.ls_mode == "fixed" and not ext.length_scale.requires_grad


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

    def test_nn_scales_mode(self):
        obs_space = _box(-1.0, 1.0, (2,))
        ext = SSPHexFeaturesExtractor(obs_space, features_dim=51, rng=0,
                                      phase_mode="nn", ls_mode="nn")
        assert hasattr(ext, "scale_net")
        assert not hasattr(ext, "scales")
        x = torch.randn(6, 2)
        out = ext(x)
        assert out.shape == (6, ext.features_dim)
        (out ** 2).sum().backward()
        assert float(ext.scale_net.net[-1].weight.grad.abs().sum()) > 0
        assert float(ext.rotation_learners[0].params.grad.abs().sum()) > 0

    def test_nn_scales_init_matches_static(self):
        obs_space = _box(-1.0, 1.0, (2,))
        static = SSPHexFeaturesExtractor(obs_space, features_dim=51, rng=0)
        ext = SSPHexFeaturesExtractor(obs_space, features_dim=51, rng=0,
                                      phase_mode="nn")
        # Rotations are randomly initialized; align them so only the
        # scale parametrization differs.
        with torch.no_grad():
            for rl_static, rl_nn in zip(static.rotation_learners,
                                        ext.rotation_learners):
                rl_nn.params.copy_(rl_static.params)
        x = torch.randn(6, 2)
        with torch.no_grad():
            err = (ext(x) - static(x)).abs().max()
        assert float(err) < 1e-5

    def test_fixed_phase_mode(self):
        obs_space = _box(-1.0, 1.0, (2,))
        ext = SSPHexFeaturesExtractor(obs_space, features_dim=51, rng=0,
                                      phase_mode="fixed")
        assert not ext.scales.requires_grad
        assert not ext.rotation_learners[0].params.requires_grad


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

    def test_nn_modes(self):
        # NN-modulated ls/phase also work through the MiniGrid extractor
        n_pts = self.VIEW_W * self.VIEW_H
        features_dim = 51
        dummy_dim = self.INPUT_DIM + n_pts * features_dim + features_dim
        obs_space = _box(-1.0, 7.0, (dummy_dim,))
        ext_tmp = SSPMiniGridViewFeatures(obs_space, features_dim=features_dim, rng=0)
        real_dim = self.INPUT_DIM + n_pts * ext_tmp.ssp_dim + ext_tmp.ssp_dim
        ext = SSPMiniGridViewFeatures(
            _box(-1.0, 7.0, (real_dim,)), features_dim=features_dim, rng=0,
            ls_mode="nn", phase_mode="nn", diff_ls_for_view=True,
        )
        obs = torch.randn(4, real_dim)
        out = ext(obs)
        assert out.shape == (4, ext.ssp_dim)
        assert hasattr(ext, "length_scale_view_net")


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
