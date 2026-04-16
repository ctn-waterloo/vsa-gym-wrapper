"""
Tests for JAX-backed SSP spaces and environment wrappers.

All tests are skipped gracefully if JAX is not installed.
gymnax / xminigrid tests are additionally skipped if those packages
are missing.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# Enable 64-bit for accurate encode/decode tests
jax.config.update("jax_enable_x64", True)

from vsagym.jax.sspspace import (
    JaxHexagonalSSPSpace,
    JaxRandomSSPSpace,
    JaxSSPSpace,
    ssp_bind,
    ssp_encode,
    ssp_invert,
    ssp_make_unitary,
    ssp_normalize,
)
from vsagym.jax.spspace import JaxSPSpace, make_jax_sp_space
from vsagym.spaces.sspspace import HexagonalSSPSpace, RandomSSPSpace
from vsagym.spaces.spspace import SPSpace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SSP_DIM = 51
DOMAIN_DIM = 2
BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]])


@pytest.fixture(scope="module")
def jax_rand_space():
    return JaxRandomSSPSpace(DOMAIN_DIM, SSP_DIM, domain_bounds=BOUNDS, rng=0)


@pytest.fixture(scope="module")
def jax_hex_space():
    return JaxHexagonalSSPSpace(DOMAIN_DIM, SSP_DIM, domain_bounds=BOUNDS, rng=0)


@pytest.fixture(scope="module")
def jax_sp_space():
    return make_jax_sp_space(5, 64, rng=0)


@pytest.fixture(scope="module")
def numpy_rand_space():
    return RandomSSPSpace(DOMAIN_DIM, SSP_DIM, domain_bounds=BOUNDS, rng=0)


# ---------------------------------------------------------------------------
# JaxSSPSpace — encode
# ---------------------------------------------------------------------------


class TestJaxSSPEncode:
    def test_encode_shape_batch(self, jax_rand_space):
        x = jnp.ones((10, DOMAIN_DIM)) * 0.5
        phi = jax_rand_space.encode(x)
        assert phi.shape == (10, jax_rand_space.ssp_dim)

    def test_encode_shape_single_1d(self, jax_rand_space):
        x = jnp.array([0.3, 0.7])
        phi = jax_rand_space.encode(x)
        assert phi.shape[1] == jax_rand_space.ssp_dim

    def test_encode_matches_numpy(self, jax_rand_space, numpy_rand_space):
        x_np = np.array([[0.3, 0.7]])
        phi_np = numpy_rand_space.encode(x_np)
        phi_jax = np.array(jax_rand_space.encode(jnp.array(x_np)))
        np.testing.assert_allclose(phi_jax, phi_np, atol=1e-5)

    def test_encode_similar_points_are_similar(self, jax_rand_space):
        phi1 = jax_rand_space.encode(jnp.array([[0.5, 0.5]]))
        phi2 = jax_rand_space.encode(jnp.array([[0.51, 0.51]]))
        phi3 = jax_rand_space.encode(jnp.array([[0.9, 0.9]]))
        sim_close = float((phi1 @ phi2.T)[0, 0])
        sim_far = float((phi1 @ phi3.T)[0, 0])
        assert sim_close > sim_far

    def test_encode_hex_shape(self, jax_hex_space):
        x = jnp.ones((5, DOMAIN_DIM)) * 0.5
        phi = jax_hex_space.encode(x)
        assert phi.shape == (5, jax_hex_space.ssp_dim)


# ---------------------------------------------------------------------------
# JaxSSPSpace — bind / invert
# ---------------------------------------------------------------------------


class TestJaxSSPBind:
    def test_bind_shape(self, jax_rand_space):
        a = jax_rand_space.encode(jnp.array([[0.2, 0.8]]))
        b = jax_rand_space.encode(jnp.array([[0.6, 0.4]]))
        result = jax_rand_space.bind(a, b)
        assert result.shape == (1, jax_rand_space.ssp_dim)

    def test_bind_with_identity(self, jax_rand_space):
        a = jax_rand_space.encode(jnp.array([[0.3, 0.7]]))
        I = jax_rand_space.identity()
        result = jax_rand_space.bind(a, I)
        np.testing.assert_allclose(np.array(result), np.array(a), atol=1e-5)

    def test_bind_invert_recovers(self, jax_rand_space):
        a = jax_rand_space.encode(jnp.array([[0.2, 0.8]]))
        b = jax_rand_space.encode(jnp.array([[0.6, 0.4]]))
        bound = jax_rand_space.bind(a, b)
        inv_b = jax_rand_space.invert(b)
        recovered = jax_rand_space.bind(bound, inv_b)
        sim = float((recovered @ a.T)[0, 0])
        assert sim > 0.4

    def test_invert_shape(self, jax_rand_space):
        a = jax_rand_space.encode(jnp.array([[0.3, 0.7]]))
        inv_a = jax_rand_space.invert(a)
        assert inv_a.shape == a.shape

    def test_invert_matches_numpy(self, jax_rand_space, numpy_rand_space):
        x_np = np.array([[0.4, 0.6]])
        phi_np = numpy_rand_space.encode(x_np)
        inv_np = numpy_rand_space.invert(phi_np)

        phi_jax = jax_rand_space.encode(jnp.array(x_np))
        inv_jax = jax_rand_space.invert(phi_jax)

        np.testing.assert_allclose(np.array(inv_jax), inv_np, atol=1e-5)


# ---------------------------------------------------------------------------
# JaxSSPSpace — normalize / make_unitary
# ---------------------------------------------------------------------------


class TestJaxSSPNormalize:
    def test_normalize_unit_norm(self, jax_rand_space):
        a = jax_rand_space.encode(jnp.array([[0.3, 0.7]]))
        normed = jax_rand_space.normalize(a)
        norm = float(jnp.linalg.norm(normed))
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_make_unitary_fourier_mags(self, jax_rand_space):
        rng = np.random.default_rng(0)
        v = jnp.array(rng.standard_normal((1, SSP_DIM)))
        u = jax_rand_space.make_unitary(v)
        fu = jnp.fft.fft(u, axis=-1)
        mags = jnp.sqrt(fu.real ** 2 + fu.imag ** 2)
        np.testing.assert_allclose(np.array(mags), np.ones_like(mags), atol=1e-5)


# ---------------------------------------------------------------------------
# JIT compilation
# ---------------------------------------------------------------------------


class TestJaxJIT:
    def test_jit_encode(self, jax_rand_space):
        encode_jit = jax.jit(jax_rand_space.encode)
        x = jnp.ones((3, DOMAIN_DIM)) * 0.5
        phi = encode_jit(x)
        assert phi.shape == (3, jax_rand_space.ssp_dim)

    def test_jit_bind(self, jax_rand_space):
        bind_jit = jax.jit(jax_rand_space.bind)
        a = jax_rand_space.encode(jnp.array([[0.3, 0.7]]))
        b = jax_rand_space.encode(jnp.array([[0.6, 0.4]]))
        result = bind_jit(a, b)
        assert result.shape == a.shape

    def test_jit_invert(self, jax_rand_space):
        invert_jit = jax.jit(jax_rand_space.invert)
        a = jax_rand_space.encode(jnp.array([[0.3, 0.7]]))
        result = invert_jit(a)
        assert result.shape == a.shape

    def test_pure_fn_jit(self):
        """Module-level functions should be JIT-able with explicit arguments."""
        np_space = RandomSSPSpace(2, SSP_DIM, rng=0)
        pm = jnp.array(np_space.phase_matrix)
        ls = jnp.array(np_space.length_scale.flatten())

        @jax.jit
        def encode_fn(x):
            return ssp_encode(pm, ls, x)

        x = jnp.ones((4, 2)) * 0.5
        phi = encode_fn(x)
        assert phi.shape == (4, SSP_DIM)


# ---------------------------------------------------------------------------
# vmap compatibility
# ---------------------------------------------------------------------------


class TestJaxVmap:
    def test_vmap_encode(self, jax_rand_space):
        """Encode a batch of single vectors via vmap."""
        x_batch = jnp.ones((6, DOMAIN_DIM)) * 0.5

        def encode_single(x):
            return ssp_encode(
                jax_rand_space.phase_matrix,
                jax_rand_space.length_scale,
                x[None, :],
            )[0]

        encoded = jax.vmap(encode_single)(x_batch)
        assert encoded.shape == (6, jax_rand_space.ssp_dim)

    def test_vmap_bind(self, jax_rand_space):
        a = jax_rand_space.encode(jnp.ones((4, DOMAIN_DIM)) * 0.3)
        b = jax_rand_space.encode(jnp.ones((4, DOMAIN_DIM)) * 0.7)
        result = jax.vmap(ssp_bind)(a, b)
        assert result.shape == a.shape


# ---------------------------------------------------------------------------
# JaxSPSpace
# ---------------------------------------------------------------------------


class TestJaxSPSpace:
    def test_encode_shape(self, jax_sp_space):
        idx = jnp.array([0, 1, 2])
        vecs = jax_sp_space.encode(idx)
        assert vecs.shape == (3, 64)

    def test_decode_roundtrip(self, jax_sp_space):
        for i in range(5):
            v = jax_sp_space.encode(jnp.array([i]))
            decoded = jax_sp_space.decode(v)
            assert int(decoded[0]) == i

    def test_encode_name(self, jax_sp_space):
        name = jax_sp_space.names[0]
        vec = jax_sp_space.encode_name(name)
        assert vec.shape == (1, 64)

    def test_jit_encode(self, jax_sp_space):
        vecs = jax_sp_space.vectors

        @jax.jit
        def enc(idx):
            return vecs[idx]

        result = enc(jnp.array([0, 1]))
        assert result.shape == (2, 64)

    def test_matches_numpy_sp_space(self):
        np_sp = SPSpace(4, 32, rng=42)
        jax_sp = JaxSPSpace(np_sp)
        for i in range(4):
            np_vec = np_sp.encode(np.array([i]))
            jax_vec = np.array(jax_sp.encode(jnp.array([i])))
            np.testing.assert_allclose(jax_vec, np_vec, atol=1e-5)


# ---------------------------------------------------------------------------
# JaxSSPSpace — decode
# ---------------------------------------------------------------------------


class TestJaxSSPDecode:
    def test_decode_shape(self, jax_rand_space):
        phi = jax_rand_space.encode(jnp.array([[0.5, 0.5]]))
        x_hat = jax_rand_space.decode(phi, num_samples=20)
        assert x_hat.shape == (1, DOMAIN_DIM)

    def test_decode_roundtrip(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        space = JaxRandomSSPSpace(2, 101, domain_bounds=bounds, rng=0)
        x_true = jnp.array([[0.5, 0.5]])
        phi = space.encode(x_true)
        x_hat = space.decode(phi, num_samples=30)
        np.testing.assert_allclose(np.array(x_hat), np.array(x_true), atol=0.15)


# ---------------------------------------------------------------------------
# gymnax wrapper tests  (skipped if gymnax is not installed)
# ---------------------------------------------------------------------------

try:
    import gymnax as _gymnax
    _HAS_GYMNAX = True
except ImportError:
    _gymnax = None
    _HAS_GYMNAX = False

from vsagym.jax.gymnax_wrappers import GymnaxSSPEnvWrapper, GymnaxSSPObsWrapper


@pytest.mark.skipif(not _HAS_GYMNAX, reason="gymnax not installed")
class TestGymnaxSSPObsWrapper:
    @pytest.fixture(scope="class")
    def wrapper(self):
        env, env_params = _gymnax.make("CartPole-v1")
        return GymnaxSSPObsWrapper(env, env_params, shape_out=SSP_DIM, rng=0)

    def test_ssp_dim(self, wrapper):
        assert wrapper.ssp_dim > 0

    def test_reset_obs_shape(self, wrapper):
        key = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(key)
        assert obs.shape == (wrapper.ssp_dim,)

    def test_reset_obs_is_jax_array(self, wrapper):
        key = jax.random.PRNGKey(1)
        obs, state = wrapper.reset(key)
        assert isinstance(obs, jnp.ndarray)

    def test_step_obs_shape(self, wrapper):
        key = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(key)
        step_key = jax.random.PRNGKey(1)
        action = jnp.array(0)
        obs2, state2, reward, done, info = wrapper.step(step_key, state, action)
        assert obs2.shape == (wrapper.ssp_dim,)

    def test_jit_reset(self, wrapper):
        reset_jit = jax.jit(wrapper.reset)
        key = jax.random.PRNGKey(42)
        obs, state = reset_jit(key)
        assert obs.shape == (wrapper.ssp_dim,)

    def test_jit_step(self, wrapper):
        step_jit = jax.jit(wrapper.step)
        key = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(key)
        obs2, *_ = step_jit(jax.random.PRNGKey(1), state, jnp.array(1))
        assert obs2.shape == (wrapper.ssp_dim,)

    def test_custom_ssp_space(self):
        env, env_params = _gymnax.make("CartPole-v1")
        space = JaxRandomSSPSpace(4, SSP_DIM, rng=0)
        wrapper = GymnaxSSPObsWrapper(env, env_params, ssp_space=space)
        key = jax.random.PRNGKey(0)
        obs, _ = wrapper.reset(key)
        assert obs.shape == (SSP_DIM,)

    def test_rand_ssp_type(self):
        env, env_params = _gymnax.make("CartPole-v1")
        wrapper = GymnaxSSPObsWrapper(
            env, env_params, shape_out=SSP_DIM, ssp_type="rand", rng=0
        )
        obs, _ = wrapper.reset(jax.random.PRNGKey(0))
        assert obs.shape == (wrapper.ssp_dim,)


@pytest.mark.skipif(not _HAS_GYMNAX, reason="gymnax not installed")
class TestGymnaxSSPEnvWrapper:
    @pytest.fixture(scope="class")
    def wrapper(self):
        env, env_params = _gymnax.make("CartPole-v1")
        return GymnaxSSPEnvWrapper(env, env_params, shape_out=SSP_DIM, rng=0)

    def test_reset_obs_shape(self, wrapper):
        key = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(key)
        assert obs.shape == (wrapper.ssp_dim,)

    def test_step_without_action_ssp(self, wrapper):
        """Without action SSP, actions pass through unchanged."""
        key = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(key)
        action = jnp.array(0)
        obs2, state2, reward, done, info = wrapper.step(
            jax.random.PRNGKey(1), state, action
        )
        assert obs2.shape == (wrapper.ssp_dim,)


# ---------------------------------------------------------------------------
# xland-minigrid wrapper tests  (skipped if xminigrid is not installed)
# ---------------------------------------------------------------------------

try:
    import xminigrid as _xminigrid
    _HAS_XMINIGRID = True
except ImportError:
    _xminigrid = None
    _HAS_XMINIGRID = False

from vsagym.jax.xland_wrappers import XLandSSPPoseWrapper, XLandSSPViewWrapper


@pytest.mark.skipif(not _HAS_XMINIGRID, reason="xminigrid not installed")
class TestXLandSSPPoseWrapper:
    ENV_ID = "MiniGrid-Empty-5x5"

    @pytest.fixture(scope="class")
    def wrapper(self):
        env, env_params = _xminigrid.make(self.ENV_ID)
        return XLandSSPPoseWrapper(env, env_params, shape_out=SSP_DIM)

    def test_ssp_dim(self, wrapper):
        assert wrapper.ssp_dim > 0

    def test_reset_obs_shape(self, wrapper):
        key = jax.random.PRNGKey(0)
        env, env_params = _xminigrid.make(self.ENV_ID)
        ts = wrapper.reset(env_params, key)
        assert ts.observation.shape == (wrapper.ssp_dim,)

    def test_reset_obs_is_jax_array(self, wrapper):
        key = jax.random.PRNGKey(1)
        env, env_params = _xminigrid.make(self.ENV_ID)
        ts = wrapper.reset(env_params, key)
        assert isinstance(ts.observation, jnp.ndarray)

    def test_step_obs_shape(self, wrapper):
        key = jax.random.PRNGKey(0)
        env, env_params = _xminigrid.make(self.ENV_ID)
        ts = wrapper.reset(env_params, key)
        ts2 = wrapper.step(env_params, ts, jnp.array(0))
        assert ts2.observation.shape == (wrapper.ssp_dim,)

    def test_jit_reset(self, wrapper):
        key = jax.random.PRNGKey(42)
        env, env_params = _xminigrid.make(self.ENV_ID)
        ts = jax.jit(wrapper.reset)(env_params, key)
        assert ts.observation.shape == (wrapper.ssp_dim,)

    def test_jit_step(self, wrapper):
        key = jax.random.PRNGKey(0)
        env, env_params = _xminigrid.make(self.ENV_ID)
        ts = wrapper.reset(env_params, key)
        ts2 = jax.jit(wrapper.step)(env_params, ts, jnp.array(0))
        assert ts2.observation.shape == (wrapper.ssp_dim,)

    def test_rand_ssp_type(self):
        env, env_params = _xminigrid.make(self.ENV_ID)
        w = XLandSSPPoseWrapper(env, env_params, shape_out=SSP_DIM, ssp_type="rand")
        ts = w.reset(env_params, jax.random.PRNGKey(0))
        assert ts.observation.shape == (w.ssp_dim,)


@pytest.mark.skipif(not _HAS_XMINIGRID, reason="xminigrid not installed")
class TestXLandSSPViewWrapper:
    ENV_ID = "MiniGrid-Empty-5x5"

    @pytest.fixture(scope="class")
    def wrapper(self):
        env, env_params = _xminigrid.make(self.ENV_ID)
        # view_height/width = env view_size = 7
        return XLandSSPViewWrapper(env, env_params, shape_out=SSP_DIM,
                                    view_height=7, view_width=7)

    def test_reset_obs_shape(self, wrapper):
        key = jax.random.PRNGKey(0)
        env, env_params = _xminigrid.make(self.ENV_ID)
        ts = wrapper.reset(env_params, key)
        assert ts.observation.shape == (wrapper.ssp_dim,)

    def test_step_obs_shape(self, wrapper):
        key = jax.random.PRNGKey(0)
        env, env_params = _xminigrid.make(self.ENV_ID)
        ts = wrapper.reset(env_params, key)
        ts2 = wrapper.step(env_params, ts, jnp.array(0))
        assert ts2.observation.shape == (wrapper.ssp_dim,)

    def test_jit_reset(self, wrapper):
        key = jax.random.PRNGKey(42)
        env, env_params = _xminigrid.make(self.ENV_ID)
        ts = jax.jit(wrapper.reset)(env_params, key)
        assert ts.observation.shape == (wrapper.ssp_dim,)
