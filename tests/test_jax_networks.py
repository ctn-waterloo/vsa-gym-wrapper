"""
Tests for the flax (Linen) SSP feature-extractor networks in vsagym.jax.networks.

All tests are skipped gracefully if JAX or flax is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

from vsagym.jax.networks import (
    ModulationMLP,
    SSPFeaturesExtractor,
    SSPHexFeaturesExtractor,
    SSPMiniGridMissionFeatures,
    SSPMiniGridViewFeatures,
    get_ssp_dim,
)
from vsagym.spaces.sspspace import HexagonalSSPSpace, RandomSSPSpace

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 2
FEATURES_DIM = 51
LS = 0.5


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def x(key):
    return jax.random.normal(key, (8, INPUT_DIM))


def _init_and_apply(model, key, x):
    params = model.init(key, x)
    return params, model.apply(params, x)


# ---------------------------------------------------------------------------
# ModulationMLP
# ---------------------------------------------------------------------------


class TestModulationMLP:
    def test_zero_init_outputs_init_output(self, key, x):
        target = np.array([1.5, -0.5, 2.0])
        mlp = ModulationMLP(output_dim=3, init_output=target)
        params, out = _init_and_apply(mlp, key, x)
        assert out.shape == (8, 3)
        assert np.allclose(np.asarray(out), target[None, :], atol=1e-6)


# ---------------------------------------------------------------------------
# SSPFeaturesExtractor
# ---------------------------------------------------------------------------


class TestJaxSSPFeaturesExtractor:
    @pytest.mark.parametrize("basis", ["hex", "rand"])
    def test_output_shape(self, key, x, basis):
        model = SSPFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                     basis_type=basis, length_scale=LS, rng_seed=0)
        params, out = _init_and_apply(model, key, x)
        assert out.shape == (8, get_ssp_dim(INPUT_DIM, FEATURES_DIM, basis))

    @pytest.mark.parametrize("basis,np_cls", [("hex", HexagonalSSPSpace),
                                              ("rand", RandomSSPSpace)])
    def test_matches_numpy_space_at_init(self, key, x, basis, np_cls):
        model = SSPFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                     basis_type=basis, length_scale=LS, rng_seed=0)
        params, out = _init_and_apply(model, key, x)
        np_space = np_cls(INPUT_DIM, FEATURES_DIM, length_scale=LS, rng=0)
        expected = np_space.encode(np.asarray(x))
        # float32 phase precision limits agreement with the float64 numpy space
        assert np.allclose(np.asarray(out), expected, atol=2e-3)

    @pytest.mark.parametrize("modes", [
        {"ls_mode": "nn"},
        {"phase_mode": "nn"},
        {"ls_mode": "nn", "phase_mode": "nn"},
    ])
    def test_nn_modes_init_match_static(self, key, x, modes):
        static = SSPFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                      length_scale=LS, rng_seed=0)
        _, out_static = _init_and_apply(static, key, x)
        model = SSPFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                     length_scale=LS, rng_seed=0, **modes)
        _, out = _init_and_apply(model, key, x)
        assert np.allclose(np.asarray(out), np.asarray(out_static), atol=1e-5)

    def test_fixed_modes_have_no_params(self, key, x):
        model = SSPFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                     ls_mode="fixed", phase_mode="fixed", rng_seed=0)
        params = model.init(key, x)
        assert params == {} or "params" not in params or params["params"] == {}

    def test_differentiable(self, key, x):
        model = SSPFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                     length_scale=LS, rng_seed=0,
                                     ls_mode="nn", phase_mode="nn")
        params = model.init(key, x)
        grads = jax.grad(lambda p: (model.apply(p, x) ** 2).sum())(params)
        # The zero-initialized heads still receive nonzero gradients
        ls_head = grads["params"]["length_scale_net"]["Dense_1"]
        assert float(jnp.abs(ls_head["kernel"]).sum()) > 0

    def test_jit(self, key, x):
        model = SSPFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                     ls_mode="nn", rng_seed=0)
        params = model.init(key, x)
        out = jax.jit(model.apply)(params, x)
        assert out.shape[0] == 8

    def test_invalid_modes(self, key, x):
        with pytest.raises(ValueError):
            SSPFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                 ls_mode="bad").init(key, x)
        with pytest.raises(ValueError):
            SSPFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                 phase_mode="bad").init(key, x)


# ---------------------------------------------------------------------------
# SSPHexFeaturesExtractor
# ---------------------------------------------------------------------------


class TestJaxSSPHexFeaturesExtractor:
    @pytest.mark.parametrize("phase_mode", ["fixed", "learn", "nn"])
    def test_output_shape(self, key, x, phase_mode):
        model = SSPHexFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                        length_scale=LS, rng_seed=0,
                                        phase_mode=phase_mode)
        params, out = _init_and_apply(model, key, x)
        assert out.shape == (8, get_ssp_dim(INPUT_DIM, FEATURES_DIM))

    def test_learn_mode_params(self, key, x):
        model = SSPHexFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                        rng_seed=0)
        params = model.init(key, x)
        assert "scales" in params["params"]
        assert "rotation_params" in params["params"]

    def test_nn_mode_params_and_grads(self, key, x):
        model = SSPHexFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                        rng_seed=0, phase_mode="nn")
        params = model.init(key, x)
        assert "scale_net" in params["params"]
        assert "scales" not in params["params"]
        grads = jax.grad(lambda p: (model.apply(p, x) ** 2).sum())(params)
        assert float(jnp.abs(grads["params"]["rotation_params"]).sum()) > 0

    def test_rotations_are_orthogonal(self, key, x):
        model = SSPHexFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                        rng_seed=0)
        params = model.init(key, x)

        def get_rots(p):
            return model.apply(p, method=model._rotation_matrices)

        R = get_rots(params)
        eye = jnp.broadcast_to(jnp.eye(INPUT_DIM), R.shape)
        assert np.allclose(np.asarray(R @ jnp.swapaxes(R, -1, -2)), np.asarray(eye),
                           atol=1e-4)


# ---------------------------------------------------------------------------
# MiniGrid feature extractors
# ---------------------------------------------------------------------------


class TestJaxMiniGridFeatures:
    VIEW_W = 7
    VIEW_H = 7
    FEATURES_DIM = 51

    @pytest.fixture(scope="class")
    def ssp_dim(self):
        return get_ssp_dim(3, self.FEATURES_DIM)

    def test_view_features(self, key, ssp_dim):
        model = SSPMiniGridViewFeatures(input_dim=3, features_dim=self.FEATURES_DIM,
                                        length_scale=LS, rng_seed=0)
        n_pts = self.VIEW_W * self.VIEW_H
        obs = jax.random.normal(key, (4, 3 + n_pts * ssp_dim + ssp_dim))
        params, out = _init_and_apply(model, key, obs)
        assert out.shape == (4, ssp_dim)
        norms = np.asarray(jnp.linalg.norm(out, axis=-1))
        assert np.allclose(norms, 1.0, atol=0.01)

    def test_view_features_nn_modes(self, key, ssp_dim):
        model = SSPMiniGridViewFeatures(input_dim=3, features_dim=self.FEATURES_DIM,
                                        length_scale=LS, rng_seed=0,
                                        ls_mode="nn", phase_mode="nn",
                                        diff_ls_for_view=True)
        n_pts = self.VIEW_W * self.VIEW_H
        obs = jax.random.normal(key, (4, 3 + n_pts * ssp_dim + ssp_dim))
        params, out = _init_and_apply(model, key, obs)
        assert out.shape == (4, ssp_dim)
        assert "length_scale_view_net" in params["params"]
        out_jit = jax.jit(model.apply)(params, obs)
        assert np.allclose(np.asarray(out), np.asarray(out_jit), atol=1e-5)

    def test_mission_features(self, key, ssp_dim):
        model = SSPMiniGridMissionFeatures(input_dim=3, features_dim=self.FEATURES_DIM,
                                           length_scale=LS, rng_seed=0)
        n_pts = self.VIEW_W * self.VIEW_H
        obs = jax.random.normal(key, (4, 3 + n_pts * ssp_dim + 2 * ssp_dim))
        params, out = _init_and_apply(model, key, obs)
        assert out.shape == (4, 2 * ssp_dim)


# ---------------------------------------------------------------------------
# Torch parity
# ---------------------------------------------------------------------------


class TestTorchParity:
    def test_matches_torch_at_init(self, key, x):
        torch = pytest.importorskip("torch")
        from gymnasium.spaces import Box
        from vsagym.networks import SSPFeaturesExtractor as TorchSSPFeaturesExtractor

        obs_space = Box(low=np.full(INPUT_DIM, -1.0), high=np.full(INPUT_DIM, 1.0))
        torch_model = TorchSSPFeaturesExtractor(obs_space, features_dim=FEATURES_DIM,
                                                rng=0, length_scale=LS)
        with torch.no_grad():
            out_torch = torch_model(
                torch.tensor(np.asarray(x, dtype=np.float32))).numpy()

        model = SSPFeaturesExtractor(input_dim=INPUT_DIM, features_dim=FEATURES_DIM,
                                     length_scale=LS, rng_seed=0)
        _, out_jax = _init_and_apply(model, key, x)
        # float32 phase precision limits agreement between the two backends
        assert np.allclose(np.asarray(out_jax), out_torch, atol=2e-3)
