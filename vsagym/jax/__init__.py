"""
vsagym.jax — JAX-accelerated SSP spaces and environment wrappers.

All SSP operations are JIT-compilable and vmap-compatible.

Quick start
-----------

**Core SSP operations:**

>>> import jax
>>> import jax.numpy as jnp
>>> from vsagym.jax import JaxHexagonalSSPSpace, JaxRandomSSPSpace
>>>
>>> jax.config.update("jax_enable_x64", True)   # recommended
>>> space = JaxHexagonalSSPSpace(domain_dim=2, ssp_dim=151)
>>> phi = space.encode(jnp.array([[0.5, 0.3]]))  # (1, 151)
>>> x_hat = space.decode(phi)                    # (1, 2)

**gymnax wrapper:**

>>> from vsagym.jax import GymnaxSSPObsWrapper
>>> import gymnax
>>> env, env_params = gymnax.make("CartPole-v1")
>>> wrapper = GymnaxSSPObsWrapper(env, env_params, shape_out=151)
>>> obs, state = jax.jit(wrapper.reset)(jax.random.PRNGKey(0))

**xland-minigrid wrapper:**

>>> from vsagym.jax import XLandSSPPoseWrapper
>>> import xminigrid
>>> env, env_params = xminigrid.make("MiniGrid-Empty-5x5")
>>> wrapper = XLandSSPPoseWrapper(env, env_params, shape_out=151)
>>> timestep = jax.jit(wrapper.reset)(env_params, jax.random.PRNGKey(0))

Modules
-------
sspspace     — JAX SSPSpace, RandomSSPSpace, HexagonalSSPSpace + pure functions
spspace      — JAX SPSpace for discrete symbols
gymnax_wrappers  — gymnax environment wrappers
xland_wrappers   — xland-minigrid environment wrappers
"""

from vsagym.jax.sspspace import (
    JaxHexagonalSSPSpace,
    JaxRandomSSPSpace,
    JaxSSPSpace,
    ssp_bind,
    ssp_encode,
    ssp_encode_fourier,
    ssp_invert,
    ssp_make_unitary,
    ssp_normalize,
)
from vsagym.jax.spspace import JaxSPSpace, make_jax_sp_space
from vsagym.jax.gymnax_wrappers import (
    GymnaxSSPEnvWrapper,
    GymnaxSSPObsWrapper,
)
from vsagym.jax.xland_wrappers import (
    XLandSSPPoseWrapper,
    XLandSSPViewWrapper,
)

__all__ = [
    # SSP spaces
    "JaxSSPSpace",
    "JaxRandomSSPSpace",
    "JaxHexagonalSSPSpace",
    # SP space
    "JaxSPSpace",
    "make_jax_sp_space",
    # Pure functions
    "ssp_encode",
    "ssp_encode_fourier",
    "ssp_bind",
    "ssp_invert",
    "ssp_normalize",
    "ssp_make_unitary",
    # gymnax wrappers
    "GymnaxSSPObsWrapper",
    "GymnaxSSPEnvWrapper",
    # xland-minigrid wrappers
    "XLandSSPPoseWrapper",
    "XLandSSPViewWrapper",
]
