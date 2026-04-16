"""
JAX-compatible Semantic Pointer (SP) space for discrete symbol encoding.

Provides :class:`JaxSPSpace` which mirrors :class:`~vsagym.spaces.spspace.SPSpace`
but stores its lookup vectors as JAX arrays and exposes JIT-compatible methods.
"""

from __future__ import annotations

from typing import List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from vsagym.spaces.spspace import SPSpace
from vsagym.jax.sspspace import ssp_bind, ssp_invert


class JaxSPSpace:
    """JAX-backed Semantic Pointer space for discrete symbols.

    Parameters
    ----------
    sp_space:
        A numpy :class:`~vsagym.spaces.spspace.SPSpace` whose vectors are copied.
    """

    def __init__(self, sp_space: SPSpace):
        self.domain_size: int = sp_space.domain_size
        self.dim: int = sp_space.dim
        self.vectors: jnp.ndarray = jnp.array(sp_space.vectors)
        self.inverse_vectors: jnp.ndarray = jnp.array(sp_space.inverse_vectors)
        self.names: List[str] = sp_space.names

        # Build name → vector look-up tables as JAX arrays.
        # The identity and NULL vectors are appended.
        _identity = jnp.zeros((1, self.dim)).at[:, 0].set(1.0)
        _null = jnp.zeros((1, self.dim))
        _all_vecs = jnp.vstack([self.vectors, _identity, _null])
        self._name_list: List[str] = list(sp_space.names) + ["I", "NULL"]
        self._name_to_idx = {n: i for i, n in enumerate(self._name_list)}
        self._all_vectors: jnp.ndarray = _all_vecs
        _all_inv = jnp.vstack([self.inverse_vectors, _identity, _null])
        self._all_inverse_vectors: jnp.ndarray = _all_inv

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, i: Union[jnp.ndarray, np.ndarray, int]) -> jnp.ndarray:
        """Map integer index → SP vector.

        Parameters
        ----------
        i:
            Integer index or array of indices, each in ``[0, domain_size)``.

        Returns
        -------
        jnp.ndarray
            Shape ``(n, dim)`` where ``n = i.size``.
        """
        i = jnp.asarray(i).reshape(-1).astype(jnp.int32)
        return self.vectors[i]

    def encode_name(self, name: str) -> jnp.ndarray:
        """Map a symbolic name → SP vector. Shape ``(1, dim)``."""
        idx = self._name_to_idx[name]
        return self._all_vectors[idx : idx + 1]

    def decode(self, v: jnp.ndarray) -> jnp.ndarray:
        """Map SP vector(s) → integer index via nearest-neighbour.

        Parameters
        ----------
        v:
            Shape ``(n, dim)`` or ``(dim,)``.

        Returns
        -------
        jnp.ndarray
            Shape ``(n,)`` integer indices.
        """
        v = jnp.atleast_2d(v)
        sims = self.vectors @ v.T              # (domain_size, n)
        return jnp.argmax(sims, axis=0)       # (n,)

    def clean_up(self, v: jnp.ndarray) -> jnp.ndarray:
        """Return the nearest SP vector for each row of ``v``.

        Shape: same as input.
        """
        v = jnp.atleast_2d(v)
        sims = self.vectors @ v.T
        return self.vectors[jnp.argmax(sims, axis=0)]

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def bind(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Circular-convolution binding."""
        return ssp_bind(a, b)

    def invert(self, a: jnp.ndarray) -> jnp.ndarray:
        """Approximate inverse under binding."""
        return ssp_invert(a)

    def normalize(self, v: jnp.ndarray) -> jnp.ndarray:
        """Normalise to unit length."""
        return v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), 1e-8)

    def make_unitary(self, v: jnp.ndarray) -> jnp.ndarray:
        """Make Fourier components unit magnitude."""
        fv = jnp.fft.fft(v, axis=-1)
        fv = fv / jnp.maximum(jnp.sqrt(fv.real ** 2 + fv.imag ** 2), 1e-8)
        return jnp.fft.ifft(fv, axis=-1).real

    def identity(self) -> jnp.ndarray:
        """Identity vector under binding. Shape ``(1, dim)``."""
        return jnp.zeros((1, self.dim)).at[:, 0].set(1.0)


def make_jax_sp_space(
    domain_size: int,
    dim: int,
    rng: Optional[Union[int, np.random.Generator]] = None,
    names: Optional[List[str]] = None,
) -> JaxSPSpace:
    """Create a :class:`JaxSPSpace` from scratch.

    Parameters
    ----------
    domain_size:
        Number of discrete symbols.
    dim:
        Dimensionality of SP vectors (should be >= domain_size for low overlap).
    rng:
        Optional random seed or :class:`numpy.random.Generator`.
    names:
        Optional list of symbolic names, length ``domain_size``.
    """
    np_space = SPSpace(domain_size, dim, rng=rng, names=names)
    return JaxSPSpace(np_space)
