"""
JAX-compatible SSP (Spatial Semantic Pointer) spaces.

All operations are JIT-compilable and vmap-compatible. The module provides both
pure functions (usable directly with jax.jit) and class wrappers (JaxSSPSpace,
JaxRandomSSPSpace, JaxHexagonalSSPSpace) that mirror the numpy API.

Quick start
-----------
>>> import jax
>>> import jax.numpy as jnp
>>> from vsagym.jax.sspspace import JaxHexagonalSSPSpace
>>>
>>> jax.config.update("jax_enable_x64", True)  # recommended for accuracy
>>> space = JaxHexagonalSSPSpace(domain_dim=2, ssp_dim=151)
>>> x = jnp.array([[0.5, 0.3]])
>>> phi = space.encode(x)          # (1, 151)
>>> x_hat = space.decode(phi)      # (1, 2)

Enabling x64
------------
JAX defaults to 32-bit floats. For SSP accuracy (especially decoding), enable
64-bit with ``jax.config.update("jax_enable_x64", True)`` before any JAX calls.
"""

from __future__ import annotations

from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from vsagym.spaces.sspspace import (
    HexagonalSSPSpace,
    RandomSSPSpace,
    SSPSpace,
)

# ---------------------------------------------------------------------------
# Pure JAX functions — JIT-compiled once at module import.
# These accept plain jnp arrays and are safe to use inside jax.jit / jax.vmap.
# ---------------------------------------------------------------------------


@jax.jit
def ssp_encode(phase_matrix: jnp.ndarray, length_scale: jnp.ndarray,
               x: jnp.ndarray) -> jnp.ndarray:
    """Encode points ``x`` into SSP vectors.

    Parameters
    ----------
    phase_matrix:
        Shape ``(ssp_dim, domain_dim)``.
    length_scale:
        Shape ``(domain_dim,)``.
    x:
        Shape ``(n_samples, domain_dim)`` or ``(domain_dim,)`` for a single point.

    Returns
    -------
    jnp.ndarray
        Shape ``(n_samples, ssp_dim)``.
    """
    x = jnp.atleast_2d(x)
    ls_mat = jnp.diag(1.0 / length_scale)
    scaled_x = x @ ls_mat                                          # (n, d)
    exponents = 1j * (phase_matrix @ scaled_x.T)                  # (D, n)
    data = jnp.fft.ifft(jnp.exp(exponents), axis=0).real          # (D, n)
    return data.T                                                   # (n, D)


@jax.jit
def ssp_encode_fourier(phase_matrix: jnp.ndarray, length_scale: jnp.ndarray,
                       x: jnp.ndarray) -> jnp.ndarray:
    """Return the Fourier-domain representation (complex-valued)."""
    x = jnp.atleast_2d(x)
    ls_mat = jnp.diag(1.0 / length_scale)
    scaled_x = x @ ls_mat
    data = jnp.exp(1j * (phase_matrix @ scaled_x.T))
    return data.T


@jax.jit
def ssp_bind(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Circular-convolution binding.

    Supports broadcasting: ``a`` and ``b`` may be a single vector or a batch.
    """
    return jnp.fft.ifft(
        jnp.fft.fft(a, axis=-1) * jnp.fft.fft(b, axis=-1), axis=-1
    ).real


@jax.jit
def ssp_invert(a: jnp.ndarray) -> jnp.ndarray:
    """Approximate inverse of ``a`` under circular-convolution binding.

    Equivalent to the numpy ``a[:, -np.arange(D)]`` trick.
    """
    n = a.shape[-1]
    idx = jnp.concatenate([jnp.array([0]), jnp.arange(n - 1, 0, -1)])
    return a[..., idx]


@jax.jit
def ssp_normalize(ssp: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Normalise SSP vectors to unit length."""
    return ssp / jnp.maximum(jnp.linalg.norm(ssp, axis=-1, keepdims=True), eps)


@jax.jit
def ssp_make_unitary(ssp: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Make SSP vectors unitary (Fourier components have magnitude 1)."""
    fssp = jnp.fft.fft(ssp, axis=-1)
    mag = jnp.sqrt(fssp.real ** 2 + fssp.imag ** 2)
    fssp = fssp / jnp.maximum(mag, eps)
    return jnp.fft.ifft(fssp, axis=-1).real


# ---------------------------------------------------------------------------
# Class wrapper
# ---------------------------------------------------------------------------


class JaxSSPSpace:
    """JAX-backed SSP space.

    Wraps a numpy :class:`~vsagym.spaces.sspspace.SSPSpace` by converting its
    phase matrix and length scale to JAX arrays. All methods dispatch to the
    JIT-compiled module-level functions above, making them safe to use inside
    ``jax.jit``-compiled functions when captured via closure or passed
    explicitly.

    Parameters
    ----------
    ssp_space:
        A numpy ``SSPSpace`` (or subclass) whose parameters are copied.
    """

    def __init__(self, ssp_space: SSPSpace):
        self.domain_dim: int = ssp_space.domain_dim
        self.ssp_dim: int = ssp_space.ssp_dim
        self.phase_matrix: jnp.ndarray = jnp.array(ssp_space.phase_matrix)
        self.length_scale: jnp.ndarray = jnp.array(
            ssp_space.length_scale.flatten()
        )
        self.domain_bounds: Optional[jnp.ndarray] = (
            jnp.array(ssp_space.domain_bounds)
            if ssp_space.domain_bounds is not None
            else None
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode ``x`` → SSP vector(s).  Shape: ``(n, ssp_dim)``."""
        return ssp_encode(self.phase_matrix, self.length_scale, x)

    def encode_fourier(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the Fourier-domain (complex) SSP representation."""
        return ssp_encode_fourier(self.phase_matrix, self.length_scale, x)

    def bind(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Circular-convolution binding."""
        return ssp_bind(a, b)

    def invert(self, a: jnp.ndarray) -> jnp.ndarray:
        """Approximate inverse under binding."""
        return ssp_invert(a)

    def normalize(self, ssp: jnp.ndarray) -> jnp.ndarray:
        """Normalize to unit length."""
        return ssp_normalize(ssp)

    def make_unitary(self, ssp: jnp.ndarray) -> jnp.ndarray:
        """Make Fourier components unit magnitude."""
        return ssp_make_unitary(ssp)

    def identity(self) -> jnp.ndarray:
        """Identity vector under binding. Shape: ``(1, ssp_dim)``."""
        return jnp.zeros((1, self.ssp_dim)).at[:, 0].set(1.0)

    # ------------------------------------------------------------------
    # Similarity / decoding
    # ------------------------------------------------------------------

    def similarity(self, ssp: jnp.ndarray, other: jnp.ndarray) -> jnp.ndarray:
        """Dot-product similarity between ``ssp`` and ``other``."""
        a = ssp_normalize(ssp)
        b = ssp_normalize(other)
        return (a * b).sum(axis=-1)

    def decode(self, ssp: jnp.ndarray, num_samples: int = 300) -> jnp.ndarray:
        """Decode SSP to domain point using exhaustive grid search.

        Parameters
        ----------
        ssp:
            Shape ``(n_samples, ssp_dim)`` or ``(ssp_dim,)``.
        num_samples:
            Grid points per dimension.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_samples, domain_dim)``.

        Note
        ----
        This method is not JIT-compiled because it requires Python-level
        ``meshgrid`` construction with a variable ``num_samples`` parameter.
        For JIT-able decoding, pre-compute the grid and pass it to
        :meth:`decode_from_grid`.
        """
        bounds = (
            self.domain_bounds
            if self.domain_bounds is not None
            else jnp.column_stack(
                [-10.0 * jnp.ones(self.domain_dim),
                  10.0 * jnp.ones(self.domain_dim)]
            )
        )
        xs = [
            jnp.linspace(float(bounds[i, 0]), float(bounds[i, 1]), num_samples)
            for i in range(self.domain_dim)
        ]
        grids = jnp.meshgrid(*xs, indexing="ij")
        pts = jnp.stack([g.reshape(-1) for g in grids], axis=1)
        return self.decode_from_grid(ssp, pts)

    def decode_from_grid(self, ssp: jnp.ndarray,
                         grid_pts: jnp.ndarray) -> jnp.ndarray:
        """Decode SSP using a pre-computed set of candidate points.

        Parameters
        ----------
        ssp:
            Shape ``(n_samples, ssp_dim)`` or ``(ssp_dim,)``.
        grid_pts:
            Shape ``(n_grid, domain_dim)`` candidate domain points.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_samples, domain_dim)``.
        """
        ssp = jnp.atleast_2d(ssp)
        sample_ssps = self.encode(grid_pts)
        unit_ssp = ssp_normalize(ssp)
        sims = sample_ssps @ unit_ssp.T                     # (n_grid, n_samples)
        best = jnp.argmax(sims, axis=0)                    # (n_samples,)
        return grid_pts[best, :]


# ---------------------------------------------------------------------------
# Convenience subclasses
# ---------------------------------------------------------------------------


class JaxRandomSSPSpace(JaxSSPSpace):
    """JAX SSP space with randomly generated frequency components.

    Parameters mirror :class:`~vsagym.spaces.sspspace.RandomSSPSpace`.
    """

    def __init__(self, domain_dim: int, ssp_dim: int,
                 domain_bounds: Optional[np.ndarray] = None,
                 length_scale: Union[float, list, np.ndarray] = 1,
                 rng: Optional[Union[int, np.random.Generator]] = None,
                 **kwargs):
        np_space = RandomSSPSpace(
            domain_dim, ssp_dim,
            domain_bounds=domain_bounds,
            length_scale=length_scale,
            rng=rng, **kwargs
        )
        super().__init__(np_space)


class JaxHexagonalSSPSpace(JaxSSPSpace):
    """JAX SSP space using the Hexagonal Tiling construction.

    Parameters mirror :class:`~vsagym.spaces.sspspace.HexagonalSSPSpace`.
    """

    def __init__(self, domain_dim: int, ssp_dim: Optional[int] = 151,
                 domain_bounds: Optional[np.ndarray] = None,
                 length_scale: Union[float, list, np.ndarray] = 1,
                 rng: Optional[Union[int, np.random.Generator]] = None,
                 **kwargs):
        np_space = HexagonalSSPSpace(
            domain_dim, ssp_dim=ssp_dim,
            domain_bounds=domain_bounds,
            length_scale=length_scale,
            rng=rng, **kwargs
        )
        super().__init__(np_space)
