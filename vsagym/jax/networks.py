"""
Flax (Linen) versions of the SSP feature-extractor networks in
:mod:`vsagym.networks`.

Requires ``flax`` (install with ``pip install vsagym[jax]``).

Provided modules mirror the PyTorch classes:

- :class:`SSPFeaturesExtractor`      — trainable SSP encoding (hex or rand basis)
- :class:`SSPHexFeaturesExtractor`   — structured hex parametrization (scales + rotations)
- :class:`SSPMiniGridViewFeatures`   — MiniGrid ego-view features
- :class:`SSPMiniGridMissionFeatures`— MiniGrid ego-view + mission features

As in the PyTorch versions, the length scale and phase matrix can each be
``'fixed'`` (constant), ``'learn'`` (a free parameter), or ``'nn'`` (the output
of a small trainable MLP taking ``x`` as input, zero-initialized so training
starts from the static encoding).

Quick start
-----------
>>> import jax, jax.numpy as jnp
>>> from vsagym.jax.networks import SSPFeaturesExtractor, get_ssp_dim
>>>
>>> model = SSPFeaturesExtractor(input_dim=2, features_dim=151,
...                              ls_mode='nn', phase_mode='learn')
>>> params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
>>> phi = model.apply(params, jnp.zeros((8, 2)))   # (8, get_ssp_dim(2, 151))

Note that, as with the numpy/PyTorch hexagonal spaces, the actual SSP
dimensionality may differ from the requested ``features_dim``; use
:func:`get_ssp_dim` to obtain it.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from vsagym.spaces.sspspace import HexagonalSSPSpace, RandomSSPSpace
from vsagym.jax.sspspace import ssp_bind

_MODES = ('fixed', 'learn', 'nn')


def _make_np_space(input_dim: int, features_dim: int, basis_type: str = 'hex',
                   rng: Optional[Union[int, np.random.Generator]] = 0):
    if basis_type == 'hex':
        return HexagonalSSPSpace(input_dim, features_dim, rng=rng)
    elif basis_type == 'rand':
        return RandomSSPSpace(input_dim, features_dim, rng=rng)
    raise ValueError('Invalid basis type {basis_type}'.format(basis_type=basis_type))


def get_ssp_dim(input_dim: int, features_dim: int, basis_type: str = 'hex',
                rng: Optional[Union[int, np.random.Generator]] = 0) -> int:
    """Actual SSP dimensionality produced for a requested ``features_dim``."""
    return _make_np_space(input_dim, features_dim, basis_type, rng).ssp_dim


def _conjsym(half: jnp.ndarray) -> jnp.ndarray:
    """Build the full phase matrix (with conjugate symmetry) from its free half.

    ``half``: ``(n_params, d)`` or batched ``(batch, n_params, d)``.
    Returns ``(2*n_params + 1, d)`` (or batched).
    """
    zeros = jnp.zeros(half.shape[:-2] + (1, half.shape[-1]), dtype=half.dtype)
    return jnp.concatenate([zeros, half, -jnp.flip(half, axis=-2)], axis=-2)


def _encode(x: jnp.ndarray, log_inv_ls: jnp.ndarray, F: jnp.ndarray) -> jnp.ndarray:
    """SSP-encode ``x`` given (possibly per-sample) log inverse length scales
    ``(d,)``/``(batch, d)`` and phase matrices ``(D, d)``/``(batch, D, d)``."""
    scaled_x = x * jnp.exp(log_inv_ls)
    if F.ndim == 3:
        phases = jnp.einsum('bkd,bd->bk', F, scaled_x)
    else:
        phases = scaled_x @ F.T
    return jnp.fft.ifft(jnp.exp(1.j * phases), axis=-1).real


class ModulationMLP(nn.Module):
    """Small MLP used to modulate SSP encoding parameters as a function of x.

    The final layer is zero-initialized with its bias set to ``init_output``,
    so at initialization the network outputs ``init_output`` for every input.
    """
    output_dim: int
    hidden_dim: int = 32
    n_layers: int = 1
    init_output: Optional[np.ndarray] = None

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        if self.init_output is None:
            bias_init = nn.initializers.zeros
        else:
            init_output = np.asarray(self.init_output).flatten()

            def bias_init(key, shape, dtype=jnp.float32):
                return jnp.asarray(init_output, dtype).reshape(shape)
        return nn.Dense(self.output_dim, kernel_init=nn.initializers.zeros,
                        bias_init=bias_init)(x)


class SSPFeaturesExtractor(nn.Module):
    """SSP encoding as a trainable flax feature extractor.

    Mirrors :class:`vsagym.networks.SSPFeaturesExtractor`. The length scale
    and phase matrix can each be ``'fixed'``, ``'learn'``, or ``'nn'`` (a
    small MLP taking ``x`` as input; see module docstring).

    Attributes
    ----------
    input_dim:      dimensionality of the encoded input
    features_dim:   requested SSP dimensionality (see :func:`get_ssp_dim`)
    basis_type:     'hex' or 'rand'
    length_scale:   initial length scale (scalar or per-dimension sequence)
    ls_mode:        'fixed' | 'learn' | 'nn'
    phase_mode:     'fixed' | 'learn' | 'nn'
    nn_hidden_dim:  hidden width of the modulation MLPs
    nn_n_layers:    number of hidden layers of the modulation MLPs
    rng_seed:       seed for the (numpy) phase-matrix construction
    """
    input_dim: int
    features_dim: int
    basis_type: str = 'hex'
    length_scale: Union[float, Sequence[float]] = 1.0
    ls_mode: str = 'learn'
    phase_mode: str = 'learn'
    nn_hidden_dim: int = 32
    nn_n_layers: int = 1
    rng_seed: int = 0

    def setup(self):
        if self.ls_mode not in _MODES:
            raise ValueError('Invalid ls_mode {ls_mode}'.format(ls_mode=self.ls_mode))
        if self.phase_mode not in _MODES:
            raise ValueError('Invalid phase_mode {phase_mode}'.format(phase_mode=self.phase_mode))
        space = _make_np_space(self.input_dim, self.features_dim,
                               self.basis_type, self.rng_seed)
        self.ssp_dim = space.ssp_dim
        self.n_params = (space.ssp_dim - 1) // 2
        self._setup_ls()
        self._setup_phase(space)

    def _setup_ls(self):
        initial_ls = np.asarray(self.length_scale, dtype=np.float64).flatten()
        if initial_ls.size == 1:
            initial_ls = np.full(self.input_dim, initial_ls[0])
        elif initial_ls.size > self.input_dim:
            initial_ls = initial_ls[:self.input_dim]
        elif initial_ls.size < self.input_dim:
            initial_ls = np.full(self.input_dim, initial_ls.mean())
        self._init_log_inv_ls = np.log(1.0 / initial_ls)
        if self.ls_mode == 'nn':
            self.length_scale_net = ModulationMLP(self.input_dim,
                                                  self.nn_hidden_dim,
                                                  self.nn_n_layers,
                                                  init_output=self._init_log_inv_ls)
        elif self.ls_mode == 'learn':
            self.log_inv_ls = self.param(
                'length_scale',
                lambda key: jnp.asarray(self._init_log_inv_ls, jnp.float32))
        else:
            self.log_inv_ls = jnp.asarray(self._init_log_inv_ls, jnp.float32)

    def _setup_phase(self, space):
        init_half = np.asarray(space.phase_matrix[1:(self.n_params + 1), :])
        if self.phase_mode == 'nn':
            self.phase_matrix_net = ModulationMLP(self.n_params * self.input_dim,
                                                  self.nn_hidden_dim,
                                                  self.nn_n_layers,
                                                  init_output=init_half)
        elif self.phase_mode == 'learn':
            self.phase_matrix = self.param(
                'phase_matrix', lambda key: jnp.asarray(init_half, jnp.float32))
        else:
            self.phase_matrix = jnp.asarray(init_half, jnp.float32)

    def _compute_ls(self, x):
        """Log inverse length scale; ``(d,)`` or per-sample ``(batch, d)``."""
        if self.ls_mode == 'nn':
            return self.length_scale_net(x)
        return self.log_inv_ls

    def _compute_phase_matrix(self, x):
        """Full phase matrix; ``(ssp_dim, d)`` or per-sample ``(batch, ssp_dim, d)``."""
        if self.phase_mode == 'nn':
            half = self.phase_matrix_net(x).reshape(x.shape[0], self.n_params,
                                                    self.input_dim)
            return _conjsym(half)
        return _conjsym(self.phase_matrix)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        return _encode(x, self._compute_ls(x), self._compute_phase_matrix(x))


class SSPHexFeaturesExtractor(SSPFeaturesExtractor):
    """Hexagonal SSP feature extractor with a structured phase matrix.

    Mirrors :class:`vsagym.networks.SSPHexFeaturesExtractor`. ``phase_mode``:
      - ``'fixed'``: the numpy space's phase matrix is used as a constant,
      - ``'learn'`` (default): the scales and rotations are free parameters,
      - ``'nn'``: the scales are output by a small MLP taking ``x`` as input
        (rotations remain free parameters), so the phase matrix is an
        input-dependent function while preserving the hexagonal structure.
    """

    def setup(self):
        if self.ls_mode not in _MODES:
            raise ValueError('Invalid ls_mode {ls_mode}'.format(ls_mode=self.ls_mode))
        if self.phase_mode not in _MODES:
            raise ValueError('Invalid phase_mode {phase_mode}'.format(phase_mode=self.phase_mode))
        space = HexagonalSSPSpace(self.input_dim, self.features_dim,
                                  rng=self.rng_seed)
        self.ssp_dim = space.ssp_dim
        self.n_params = (space.ssp_dim - 1) // 2
        self._setup_ls()
        self._setup_phase(space)

    def _setup_phase(self, space):
        self.n_rotations = int(np.sqrt((self.ssp_dim - 1) / (2 * (self.input_dim + 1))))
        if self.phase_mode == 'fixed':
            super()._setup_phase(space)
            return
        self.phase_base = jnp.asarray(space.phases_hex, jnp.float32)
        init_scales = np.asarray(space.scales)
        n_skew = (self.input_dim * (self.input_dim - 1)) // 2
        if self.phase_mode == 'nn':
            self.scale_net = ModulationMLP(init_scales.size,
                                           self.nn_hidden_dim,
                                           self.nn_n_layers,
                                           init_output=init_scales)
        else:
            self.scales = self.param(
                'scales', lambda key: jnp.asarray(init_scales, jnp.float32))
        self.rotation_params = self.param(
            'rotation_params', nn.initializers.normal(stddev=1.0),
            (self.n_rotations, n_skew))

    def _rotation_matrices(self):
        d = self.input_dim
        iu = np.triu_indices(d, k=1)
        A = jnp.zeros((self.n_rotations, d, d),
                      dtype=self.rotation_params.dtype)
        A = A.at[:, iu[0], iu[1]].set(self.rotation_params)
        A = A - jnp.swapaxes(A, -1, -2)
        return jax.vmap(jax.scipy.linalg.expm)(A)

    def _compute_phase_matrix(self, x):
        if self.phase_mode == 'fixed':
            return _conjsym(self.phase_matrix)
        if self.phase_mode == 'nn':
            scales = self.scale_net(x)                      # (batch, n_scales)
        else:
            scales = self.scales                            # (n_scales,)
        phases_scaled = scales[..., :, None, None] * self.phase_base
        phases_scaled = phases_scaled.reshape(scales.shape[:-1] + (-1, self.input_dim))
        R = self._rotation_matrices()                       # (n_rotations, d, d)
        rotated = jnp.einsum('...kd,rde->...rke', phases_scaled, R)
        half = rotated.reshape(scales.shape[:-1] + (self.n_params, self.input_dim))
        return _conjsym(half)


class SSPMiniGridViewFeatures(SSPFeaturesExtractor):
    """MiniGrid ego-view SSP features (flax version).

    Mirrors :class:`vsagym.networks.SSPMiniGridViewFeatures`. ``input_dim``
    must be 3 (agent x, y, orientation). The observation vector is expected to
    be ``[pose (3), flattened object SSPs (n_pts * ssp_dim), carrying vector
    (ssp_dim)]``.
    """
    diff_ls_for_view: bool = False
    view_width: int = 7
    view_height: int = 7

    def setup(self):
        assert self.input_dim == 3, 'SSPMiniGridViewFeatures requires input_dim=3'
        super().setup()

        if self.ls_mode == 'nn':
            if self.diff_ls_for_view:
                self.length_scale_view_net = ModulationMLP(
                    self.input_dim, self.nn_hidden_dim, self.nn_n_layers,
                    init_output=self._init_log_inv_ls)
        elif self.diff_ls_for_view:
            self.log_inv_ls_view = self.param(
                'length_scale_view',
                lambda key: jnp.asarray(self._init_log_inv_ls, jnp.float32))

        domain_bounds = np.array([[0, self.view_width - 1],
                                  [-(self.view_height - 1) // 2, (self.view_height - 1) // 2],
                                  [0, 3]])
        xs = [np.arange(domain_bounds[i, 1], domain_bounds[i, 0] - 1, -1) for i in range(2)]
        xx = np.meshgrid(*xs)
        pts = np.vstack([xx[i].reshape(-1) for i in range(2)]).T
        self.unroll_pts = jnp.asarray(
            np.hstack([pts, -1 * np.ones((pts.shape[0], 1))]), jnp.float32)
        self.n_pts = self.unroll_pts.shape[0]

    def _compute_ls_view(self, pts):
        """Log inverse length scale for the ego-view grid points."""
        if self.ls_mode == 'nn':
            if self.diff_ls_for_view:
                return self.length_scale_view_net(pts)
            return self.length_scale_net(pts)
        if self.diff_ls_for_view:
            return self.log_inv_ls_view
        return self.log_inv_ls

    def _encode_pose_and_grid(self, x):
        """Encode the agent pose and the ego-view grid points."""
        pose_ssp = _encode(x, self._compute_ls(x), self._compute_phase_matrix(x))
        log_inv_ls_view = self._compute_ls_view(self.unroll_pts)  # different ls for ego-vec
        F_view = self._compute_phase_matrix(self.unroll_pts)
        ssp_grid_pts = _encode(self.unroll_pts, log_inv_ls_view,
                               F_view).reshape(1, self.n_pts, self.ssp_dim)
        return pose_ssp, ssp_grid_pts

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        obs = jnp.atleast_2d(obs)
        x = obs[:, :self.input_dim]
        vsa_output, ssp_grid_pts = self._encode_pose_and_grid(x)
        obj_sps = obs[:, self.input_dim:-self.ssp_dim].reshape(obs.shape[0], -1, self.ssp_dim)
        vsa_output = vsa_output + jnp.sum(ssp_bind(obj_sps, ssp_grid_pts), axis=1)
        vsa_output = vsa_output + obs[:, -self.ssp_dim:]  # the 'has'/carrying vector
        return vsa_output / jnp.linalg.norm(vsa_output, axis=-1, keepdims=True)


class SSPMiniGridMissionFeatures(SSPMiniGridViewFeatures):
    """MiniGrid ego-view + mission SSP features (flax version).

    The observation vector is expected to be ``[pose (3), flattened object
    SSPs, carrying vector (ssp_dim), mission vector (ssp_dim)]``; the output
    dimensionality is ``2 * ssp_dim``.
    """

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        obs = jnp.atleast_2d(obs)
        x = obs[:, :self.input_dim]
        vsa_output, ssp_grid_pts = self._encode_pose_and_grid(x)
        obj_sps = obs[:, self.input_dim:-2 * self.ssp_dim].reshape(obs.shape[0], -1, self.ssp_dim)
        vsa_output = vsa_output + jnp.sum(ssp_bind(obj_sps, ssp_grid_pts), axis=1)
        vsa_output = vsa_output + obs[:, -2 * self.ssp_dim:-self.ssp_dim]  # 'has'/carrying vector
        vsa_output = vsa_output / jnp.linalg.norm(vsa_output, axis=-1, keepdims=True)

        mission_vec = obs[:, -self.ssp_dim:]
        return jnp.hstack([vsa_output, mission_vec])
