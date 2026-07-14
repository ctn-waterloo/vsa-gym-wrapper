"""
Tests for core SSP / SP classes and space constructors:
    SSP, SSPSpace, RandomSSPSpace, HexagonalSSPSpace, SPSpace
"""

import numpy as np
import pytest

from vsagym.spaces.ssp import SSP
from vsagym.spaces.spspace import SPSpace
from vsagym.spaces.sspspace import (
    HexagonalSSPSpace,
    RandomSSPSpace,
    SSPSpace,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SSP_DIM = 51
DOMAIN_DIM = 2
BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]])


@pytest.fixture(scope="module")
def rand_space():
    return RandomSSPSpace(
        DOMAIN_DIM, SSP_DIM, domain_bounds=BOUNDS, rng=0
    )


@pytest.fixture(scope="module")
def hex_space():
    return HexagonalSSPSpace(
        DOMAIN_DIM, ssp_dim=SSP_DIM, domain_bounds=BOUNDS, rng=0
    )


@pytest.fixture(scope="module")
def sp_space():
    return SPSpace(5, 64, rng=0)


# ---------------------------------------------------------------------------
# SSP class (HRR vector with operator overloads)
# ---------------------------------------------------------------------------


class TestSSP:
    def _rand_vec(self, d=SSP_DIM):
        rng = np.random.default_rng(42)
        return SSP(rng.standard_normal(d))

    def test_add(self):
        a = self._rand_vec()
        b = self._rand_vec()
        result = a + b
        assert isinstance(result, SSP)
        np.testing.assert_allclose(result.v, a.v + b.v)

    def test_sub(self):
        a = self._rand_vec()
        b = self._rand_vec()
        result = a - b
        np.testing.assert_allclose(result.v, a.v - b.v)

    def test_neg(self):
        a = self._rand_vec()
        result = -a
        np.testing.assert_allclose(result.v, -a.v)

    def test_mul_circular_convolution(self):
        a = self._rand_vec()
        b = self._rand_vec()
        result = a * b
        assert isinstance(result, SSP)
        # Verify via FFT-convolution formula
        expected = np.fft.ifft(
            np.fft.fft(a.v, axis=1) * np.fft.fft(b.v, axis=1), axis=1
        ).real
        np.testing.assert_allclose(result.v, expected, atol=1e-6)

    def test_mul_scalar(self):
        a = self._rand_vec()
        result = a * 2.0
        np.testing.assert_allclose(result.v, a.v * 2.0)

    def test_invert(self):
        # Invert is defined as the approximate inverse under binding for
        # *unitary* vectors (Fourier components of magnitude 1).
        # Test that (a * b) * ~b ≈ a for two unitary SSPs.
        from vsagym.spaces.sspspace import RandomSSPSpace
        rng_space = RandomSSPSpace(1, SSP_DIM, rng=0)
        a = SSP(rng_space.encode(np.array([[0.3]])))
        b = SSP(rng_space.encode(np.array([[0.7]])))
        bound = a * b
        recovered = bound * (~b)
        sim = float((recovered.v @ a.v.T).flatten()[0])
        assert sim > 0.5   # should be positively correlated with a

    def test_pow(self):
        a = self._rand_vec()
        # a**1 should be close to a
        result = a ** 1
        np.testing.assert_allclose(result.v.real, a.v, atol=1e-5)

    def test_radd_int_zero(self):
        """sum([ssp_a, ssp_b]) uses __radd__ with 0 as initial value."""
        rng = np.random.default_rng(1)
        ssps = [SSP(rng.standard_normal(SSP_DIM)) for _ in range(3)]
        result = sum(ssps)
        expected = ssps[0].v + ssps[1].v + ssps[2].v
        np.testing.assert_allclose(result.v, expected)


# ---------------------------------------------------------------------------
# SSPSpace operations
# ---------------------------------------------------------------------------


class TestSSPSpaceEncode:
    def test_encode_shape_batch(self, rand_space):
        x = np.random.default_rng(0).uniform(0, 1, (10, DOMAIN_DIM))
        phi = rand_space.encode(x)
        assert phi.shape == (10, rand_space.ssp_dim)

    def test_encode_shape_single(self, rand_space):
        x = np.array([[0.5, 0.5]])
        phi = rand_space.encode(x)
        assert phi.shape == (1, rand_space.ssp_dim)

    def test_encode_1d_input_promoted(self, rand_space):
        x = np.array([0.5, 0.5])
        phi = rand_space.encode(x)
        assert phi.shape[1] == rand_space.ssp_dim

    def test_encode_similar_points_are_similar(self, rand_space):
        x1 = np.array([[0.5, 0.5]])
        x2 = np.array([[0.51, 0.51]])
        x3 = np.array([[0.9, 0.9]])
        sim_close = (rand_space.encode(x1) @ rand_space.encode(x2).T).item()
        sim_far = (rand_space.encode(x1) @ rand_space.encode(x3).T).item()
        assert sim_close > sim_far

    def test_encode_self_similarity_high(self, rand_space):
        x = np.array([[0.3, 0.7]])
        phi = rand_space.encode(x)
        sim = (phi @ phi.T).item()
        assert sim > 0.9


class TestSSPSpaceBind:
    def test_bind_shape(self, rand_space):
        a = rand_space.encode(np.array([[0.2, 0.8]]))
        b = rand_space.encode(np.array([[0.6, 0.4]]))
        result = rand_space.bind(a, b)
        assert result.shape == (1, rand_space.ssp_dim)

    def test_bind_with_identity_is_identity(self, rand_space):
        a = rand_space.encode(np.array([[0.2, 0.8]]))
        identity = rand_space.identity()
        result = rand_space.bind(a, identity)
        np.testing.assert_allclose(result, a, atol=1e-5)

    def test_bind_invert_recovers_original(self, rand_space):
        a = rand_space.encode(np.array([[0.2, 0.8]]))
        b = rand_space.encode(np.array([[0.6, 0.4]]))
        bound = rand_space.bind(a, b)
        inv_b = rand_space.invert(b)
        recovered = rand_space.bind(bound, inv_b)
        sim = (recovered @ a.T).item()
        assert sim > 0.5  # should be positively similar


class TestSSPSpaceDecode:
    def test_decode_from_set(self, rand_space):
        x_true = np.array([[0.5, 0.5]])
        phi = rand_space.encode(x_true)
        x_hat = rand_space.decode(phi, method="from-set", num_samples=20)
        np.testing.assert_allclose(x_hat, x_true, atol=0.15)

    def test_decode_direct_optim(self, rand_space):
        x_true = np.array([[0.5, 0.5]])
        phi = rand_space.encode(x_true)
        x_hat = rand_space.decode(phi, method="direct-optim", num_samples=20)
        np.testing.assert_allclose(x_hat, x_true, atol=0.05)


# ---------------------------------------------------------------------------
# RandomSSPSpace
# ---------------------------------------------------------------------------


class TestRandomSSPSpace:
    def test_creation_default_sampler(self):
        space = RandomSSPSpace(2, 51, rng=0)
        assert space.ssp_dim == 51
        assert space.domain_dim == 2
        assert space.phase_matrix.shape == (51, 2)

    def test_creation_unif_sampler(self):
        space = RandomSSPSpace(2, 51, sampler="unif", rng=0)
        assert space.ssp_dim == 51

    def test_encode_decode_roundtrip(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        space = RandomSSPSpace(2, 101, domain_bounds=bounds, rng=0)
        x = np.array([[0.3, 0.7]])
        phi = space.encode(x)
        x_hat = space.decode(phi, method="from-set", num_samples=30)
        np.testing.assert_allclose(x_hat, x, atol=0.1)

    def test_seeded_reproducibility(self):
        s1 = RandomSSPSpace(2, 51, rng=99)
        s2 = RandomSSPSpace(2, 51, rng=99)
        np.testing.assert_array_equal(s1.phase_matrix, s2.phase_matrix)

    def test_different_seeds_differ(self):
        s1 = RandomSSPSpace(2, 51, rng=1)
        s2 = RandomSSPSpace(2, 51, rng=2)
        assert not np.allclose(s1.phase_matrix, s2.phase_matrix)


# ---------------------------------------------------------------------------
# HexagonalSSPSpace
# ---------------------------------------------------------------------------


class TestHexagonalSSPSpace:
    def test_creation(self):
        space = HexagonalSSPSpace(2, ssp_dim=151, rng=0)
        assert space.domain_dim == 2
        # ssp_dim may be adjusted to the nearest admissible value
        assert space.ssp_dim > 0
        assert space.phase_matrix.shape[1] == 2

    def test_creation_1d(self):
        space = HexagonalSSPSpace(1, ssp_dim=51, rng=0)
        assert space.domain_dim == 1

    def test_creation_3d(self):
        space = HexagonalSSPSpace(3, ssp_dim=151, rng=0)
        assert space.domain_dim == 3

    def test_encode_shape(self, hex_space):
        x = np.random.default_rng(0).uniform(0, 1, (5, DOMAIN_DIM))
        phi = hex_space.encode(x)
        assert phi.shape == (5, hex_space.ssp_dim)

    def test_encode_decode_roundtrip(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        space = HexagonalSSPSpace(2, ssp_dim=151, domain_bounds=bounds, rng=0)
        x = np.array([[0.5, 0.5]])
        phi = space.encode(x)
        x_hat = space.decode(phi, method="from-set", num_samples=30)
        np.testing.assert_allclose(x_hat, x, atol=0.1)

    def test_scale_sampling_log(self):
        space = HexagonalSSPSpace(2, ssp_dim=151, scale_sampling="log", rng=0)
        assert space.ssp_dim > 0

    def test_scale_sampling_rand(self):
        space = HexagonalSSPSpace(2, ssp_dim=151, scale_sampling="rand", rng=0)
        assert space.ssp_dim > 0

    def test_seeded_reproducibility(self):
        s1 = HexagonalSSPSpace(2, ssp_dim=151, rng=42)
        s2 = HexagonalSSPSpace(2, ssp_dim=151, rng=42)
        np.testing.assert_array_equal(s1.phase_matrix, s2.phase_matrix)


# ---------------------------------------------------------------------------
# SPSpace (discrete symbol encoding)
# ---------------------------------------------------------------------------


class TestSPSpace:
    def test_creation(self, sp_space):
        assert sp_space.domain_size == 5
        assert sp_space.dim == 64
        assert sp_space.vectors.shape == (5, 64)

    def test_encode_shape(self, sp_space):
        result = sp_space.encode(np.array([0, 1, 2, 3, 4]))
        assert result.shape == (5, 64)

    def test_decode_roundtrip(self, sp_space):
        for i in range(5):
            v = sp_space.encode(np.array([i]))
            idx = sp_space.decode(v)
            assert idx.item() == i

    def test_encode_by_name(self, sp_space):
        name = sp_space.names[0]
        vec_by_name = sp_space.name_to_vector[name]
        vec_by_idx = sp_space.encode(np.array([0]))
        np.testing.assert_allclose(vec_by_name, vec_by_idx, atol=1e-6)

    def test_vectors_are_unitary(self, sp_space):
        """Vectors should be close to unitary (magnitude 1 Fourier components)."""
        for v in sp_space.vectors:
            fv = np.fft.fft(v)
            magnitudes = np.abs(fv)
            np.testing.assert_allclose(magnitudes, np.ones_like(magnitudes), atol=1e-5)

    def test_identity_vector(self, sp_space):
        I = sp_space.identity()
        v = sp_space.encode(np.array([2]))
        bound = sp_space.bind(v, I)
        np.testing.assert_allclose(bound, v, atol=1e-5)

    def test_bind_invert(self, sp_space):
        v = sp_space.encode(np.array([1]))
        inv_v = sp_space.invert(v)
        result = sp_space.bind(v, inv_v)
        I = sp_space.identity()
        np.testing.assert_allclose(result, I, atol=1e-5)

    def test_single_symbol_is_identity(self):
        """SPSpace with domain_size=1 should give the identity vector."""
        sp = SPSpace(1, 32, rng=0)
        I = sp.identity()
        np.testing.assert_allclose(sp.vectors[0], I.flatten(), atol=1e-6)

    def test_seeded_reproducibility(self):
        s1 = SPSpace(4, 32, rng=7)
        s2 = SPSpace(4, 32, rng=7)
        np.testing.assert_array_equal(s1.vectors, s2.vectors)
