"""
Tests for gymnasium-compatible SSP spaces:
    SSPBox, SSPDiscrete, SSPSequence, SSPDict
"""

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from vsagym.spaces.ssp_box import SSPBox
from vsagym.spaces.ssp_dict import SSPDict
from vsagym.spaces.ssp_discrete import SSPDiscrete
from vsagym.spaces.ssp_sequence import SSPSequence
from vsagym.spaces.spspace import SPSpace

# ---------------------------------------------------------------------------
# SSPBox
# ---------------------------------------------------------------------------


class TestSSPBox:
    @pytest.fixture(scope="class")
    def box_1d(self):
        return SSPBox(-1.0, 1.0, shape_in=(1,), shape_out=51, seed=0)

    @pytest.fixture(scope="class")
    def box_2d(self):
        return SSPBox(
            np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
            shape_out=51, seed=0
        )

    @pytest.fixture(scope="class")
    def box_4d(self):
        # CartPole-like observation space
        low = np.array([-4.8, -np.inf, -0.42, -np.inf])
        high = np.array([4.8, np.inf, 0.42, np.inf])
        return SSPBox(low, high, shape_out=101, seed=0,
                      decoder_method="from-set")

    # --- creation ---

    def test_shape_in_from_arrays(self, box_2d):
        assert box_2d.shape_in == (2,)

    def test_shape_in_from_scalar(self, box_1d):
        assert box_1d.shape_in == (1,)

    def test_shape_out(self, box_1d):
        assert box_1d.shape_out == box_1d.ssp_space.ssp_dim

    def test_repr(self, box_2d):
        r = repr(box_2d)
        assert "SSPBox" in r

    def test_eq(self):
        b1 = SSPBox(-1.0, 1.0, shape_in=(2,), shape_out=51, seed=0)
        b2 = SSPBox(-1.0, 1.0, shape_in=(2,), shape_out=51, seed=0)
        assert b1 == b2

    def test_ssp_type_rand(self):
        box = SSPBox(-1.0, 1.0, shape_in=(2,), shape_out=51,
                     ssp_space="rand", seed=0)
        assert box.ssp_space is not None

    # --- encode ---

    def test_encode_shape_single(self, box_2d):
        x = np.array([[0.3, -0.3]])
        phi = box_2d.encode(x)
        assert phi.shape == (1, box_2d.shape_out)

    def test_encode_shape_batch(self, box_2d):
        x = np.random.default_rng(0).uniform(-1, 1, (8, 2))
        phi = box_2d.encode(x)
        assert phi.shape == (8, box_2d.shape_out)

    def test_encode_returns_float(self, box_2d):
        x = np.array([[0.0, 0.0]])
        phi = box_2d.encode(x)
        assert phi.dtype == np.float32

    # --- decode ---

    def test_decode_roundtrip_2d(self):
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        box = SSPBox(bounds[:, 0], bounds[:, 1], shape_out=101, seed=0,
                     decoder_method="from-set")
        x = np.array([[0.5, -0.5]])
        phi = box.encode(x)
        x_hat = box.decode(phi)
        np.testing.assert_allclose(x_hat, x, atol=0.15)

    # --- sample ---

    def test_sample_shape(self, box_2d):
        s = box_2d.sample()
        assert s.shape == (box_2d.shape_out,)

    def test_sample_with_return_x(self, box_2d):
        s, x = box_2d.sample(return_x=True)
        assert s.shape == (box_2d.shape_out,)
        assert x.shape == (2,)

    def test_samples_shape(self, box_2d):
        ssps = box_2d.samples(5)
        assert ssps.shape == (5, box_2d.shape_out)

    def test_samples_with_return_x(self, box_2d):
        ssps, xs = box_2d.samples(5, return_x=True)
        assert ssps.shape == (5, box_2d.shape_out)
        assert xs.shape == (5, 2)

    # --- bounded / unbounded ---

    def test_bounded_box(self, box_2d):
        assert box_2d.is_bounded(manner="both")

    def test_unbounded_box_sample(self, box_4d):
        s = box_4d.sample()
        assert s.shape == (box_4d.shape_out,)

    # --- length scale ---

    def test_custom_length_scale_scalar(self):
        box = SSPBox(-1.0, 1.0, shape_in=(2,), shape_out=51,
                     length_scale=0.1, seed=0)
        x = np.array([[0.3, 0.3]])
        phi = box.encode(x)
        assert phi.shape[1] == box.shape_out

    def test_custom_length_scale_array(self):
        box = SSPBox(-1.0, 1.0, shape_in=(2,), shape_out=51,
                     length_scale=np.array([0.1, 0.2]), seed=0)
        x = np.array([[0.3, 0.3]])
        phi = box.encode(x)
        assert phi.shape[1] == box.shape_out


# ---------------------------------------------------------------------------
# SSPDiscrete
# ---------------------------------------------------------------------------


class TestSSPDiscrete:
    @pytest.fixture(scope="class")
    def disc4(self):
        return SSPDiscrete(4, shape_out=64, seed=0)

    @pytest.fixture(scope="class")
    def disc3_start1(self):
        return SSPDiscrete(3, shape_out=64, start=1, seed=0)

    # --- creation ---

    def test_n(self, disc4):
        assert disc4.n == 4

    def test_repr(self, disc4):
        assert "SSPDiscrete" in repr(disc4)

    def test_invalid_n(self):
        with pytest.raises(AssertionError):
            SSPDiscrete(0, shape_out=32)

    # --- encode / decode ---

    def test_encode_shape(self, disc4):
        phi = disc4.encode(np.array([0]))
        assert phi.shape[-1] == disc4.shape_out

    def test_encode_all_symbols(self, disc4):
        for i in range(disc4.n):
            phi = disc4.encode(np.array([i]))
            assert phi.shape[-1] == disc4.shape_out

    def test_decode_roundtrip(self, disc4):
        for i in range(disc4.n):
            phi = disc4.encode(np.array([i]))
            decoded = disc4.decode(np.atleast_2d(phi))
            assert int(decoded) == i

    def test_encode_with_start(self, disc3_start1):
        phi = disc3_start1.encode(np.array([1]))
        assert phi.shape[-1] == disc3_start1.shape_out

    def test_decode_with_start(self, disc3_start1):
        for i in range(1, 4):
            phi = disc3_start1.encode(np.array([i]))
            decoded = disc3_start1.decode(np.atleast_2d(phi))
            assert int(decoded) == i

    # --- sample ---

    def test_sample_shape(self, disc4):
        s = disc4.sample()
        assert s.shape == (disc4.shape_out,)

    def test_samples_shape(self, disc4):
        ssps = disc4.samples(6)
        assert ssps.shape == (6, disc4.shape_out)

    def test_sample_with_mask(self, disc4):
        mask = np.array([1, 0, 0, 0], dtype=np.int8)
        s = disc4.sample(mask=mask)
        # Result should be the SSP of index 0
        expected = disc4.encode(np.array([0]))
        np.testing.assert_allclose(s.flatten(), expected.flatten(), atol=1e-6)


# ---------------------------------------------------------------------------
# SSPSequence
# ---------------------------------------------------------------------------


class TestSSPSequence:
    # Use ssp_space="rand" so shape_out=51 is exact (hex adjusts to admissible values)
    @pytest.fixture(scope="class")
    def seq_space(self):
        feat = SSPBox(-1.0, 1.0, shape_in=(1,), shape_out=51,
                      ssp_space="rand", seed=0)
        return SSPSequence(feat, length=3, seed=0)

    # --- creation ---

    def test_shape_out(self, seq_space):
        assert seq_space.shape_out == 51

    def test_length(self, seq_space):
        assert seq_space.length == 3

    def test_repr(self, seq_space):
        assert "SSPSequence" in repr(seq_space)

    # --- encode ---

    def test_encode_shape(self, seq_space):
        x = np.array([[0.1, 0.5, -0.3]])   # 1 sample, length=3
        phi = seq_space.encode(x)
        assert phi.shape == (1, 51)

    def test_encode_batch_shape(self, seq_space):
        x = np.random.default_rng(0).uniform(-1, 1, (4, 3))
        phi = seq_space.encode(x)
        assert phi.shape == (4, 51)

    # --- decode ---

    def test_decode_shape(self, seq_space):
        x = np.array([[0.2, -0.1, 0.8]])
        phi = seq_space.encode(x)
        x_hat = seq_space.decode(phi)
        assert x_hat.shape == (3, 1)

    def test_decode_roundtrip(self, seq_space):
        x = np.array([[0.2, -0.1, 0.8]])
        phi = seq_space.encode(x)
        x_hat = seq_space.decode(phi).flatten()
        np.testing.assert_allclose(x_hat, x.flatten(), atol=0.2)

    # --- sample ---

    def test_sample_shape(self, seq_space):
        s = seq_space.sample()
        assert s.shape[1] == 51

    def test_samples_shape(self, seq_space):
        ss = seq_space.samples(4)
        assert ss.shape == (4, 51)

    def test_sequence_with_discrete_feature(self):
        # Use odd dim so RandomSSPSpace (used for order SSPs) gives exact dim
        feat = SSPDiscrete(5, shape_out=51, seed=0)
        seq = SSPSequence(feat, length=4, seed=0)
        x = np.array([0, 1, 2, 3])
        phi = seq.encode(x.reshape(1, -1))
        assert phi.shape == (1, 51)


# ---------------------------------------------------------------------------
# SSPDict
# ---------------------------------------------------------------------------


class TestSSPDict:
    # Use ssp_space="rand" with an ODD dim so RandomSSPSpace gives exactly SSP_DIM
    # (even dims get rounded to ssp_dim-1 by the conjugate-symmetry constraint)
    SSP_DIM = 51

    @pytest.fixture(scope="class")
    def dict_space(self):
        spaces = {
            "position": SSPBox(-1.0, 1.0, shape_in=(2,), shape_out=self.SSP_DIM,
                               ssp_space="rand", seed=0),
            "color": SSPDiscrete(3, shape_out=self.SSP_DIM, seed=0),
        }
        return SSPDict(spaces, seed=0)

    def test_keys(self, dict_space):
        assert set(dict_space.spaces.keys()) == {"color", "position"}

    def test_len(self, dict_space):
        assert len(dict_space) == 2

    def test_getitem(self, dict_space):
        assert isinstance(dict_space["position"], SSPBox)
        assert isinstance(dict_space["color"], SSPDiscrete)

    def test_repr(self, dict_space):
        assert "Dict" in repr(dict_space)

    def test_sample_without_encode(self, dict_space):
        """Without set_encode, sample returns a dict of SSP vectors."""
        s = dict_space.sample()
        assert isinstance(s, dict)

    def test_set_encode(self):
        """set_encode wraps the user-supplied function."""
        from vsagym.spaces.ssp import SSP
        d = self.SSP_DIM  # odd, so RandomSSPSpace gives exactly d dimensions
        # Both spaces must have identical dim so bundling (SSP +) works
        spaces = {
            "position": SSPBox(-1.0, 1.0, shape_in=(2,), shape_out=d,
                               ssp_space="rand", seed=0),
            "color": SSPDiscrete(3, shape_out=d, seed=0),
        }
        ds = SSPDict(spaces)

        def my_encode(ssp_dict, static_spaces):
            result = SSP(ssp_dict["position"].v)
            result = result + ssp_dict["color"]
            return result.v.flatten()

        ds.set_encode(my_encode)
        x = {"position": np.array([[0.3, -0.3]]), "color": np.array([1])}
        phi = ds.encode(x)
        assert phi.shape == (d,)

    def test_invalid_space_type(self):
        with pytest.raises(AssertionError):
            SSPDict({"bad": Box(-1.0, 1.0, shape=(2,))})

    def test_seed_int(self, dict_space):
        seeds = dict_space.seed(42)
        assert isinstance(seeds, list)

    def test_seed_none(self, dict_space):
        seeds = dict_space.seed(None)
        assert isinstance(seeds, list)
