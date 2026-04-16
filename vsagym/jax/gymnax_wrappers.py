"""
gymnax-compatible SSP observation and environment wrappers.

These wrappers augment `gymnax <https://github.com/RobertTLange/gymnax>`_
environments so that observations (and optionally actions) are encoded as
Spatial Semantic Pointer (SSP) vectors.

Because gymnax environments are purely functional, the wrappers expose
``reset`` / ``step`` methods that match the gymnax calling convention::

    obs, state = env.reset(key, params)
    obs, state, reward, done, info = env.step(key, state, action, params)

The wrappers store the SSP space parameters as JAX arrays, so the
``encode_obs`` method (and therefore ``reset`` / ``step``) can be
wrapped with ``jax.jit``::

    wrapper = GymnaxSSPObsWrapper(env, env_params, shape_out=151)
    jit_reset = jax.jit(wrapper.reset)
    jit_step  = jax.jit(wrapper.step)

Install gymnax with:  ``pip install gymnax``
"""

from __future__ import annotations

from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from vsagym.jax.sspspace import (
    JaxHexagonalSSPSpace,
    JaxRandomSSPSpace,
    JaxSSPSpace,
    ssp_encode,
)
from vsagym.spaces.sspspace import HexagonalSSPSpace, RandomSSPSpace, SSPSpace


def _build_jax_ssp_space(obs_low: np.ndarray, obs_high: np.ndarray,
                          domain_dim: int, shape_out: Optional[int],
                          ssp_type: str, **kwargs) -> JaxSSPSpace:
    """Helper: build a JaxSSPSpace from obs-space bounds."""
    # Clip ±inf to large finite values for domain bounds
    low_clipped = np.clip(obs_low, -1e6, 1e6)
    high_clipped = np.clip(obs_high, -1e6, 1e6)
    domain_bounds = np.column_stack([low_clipped.flatten(), high_clipped.flatten()])
    if ssp_type == "hex":
        return JaxHexagonalSSPSpace(
            domain_dim, ssp_dim=shape_out,
            domain_bounds=domain_bounds, **kwargs
        )
    elif ssp_type == "rand":
        return JaxRandomSSPSpace(
            domain_dim, ssp_dim=shape_out,
            domain_bounds=domain_bounds, **kwargs
        )
    else:
        raise ValueError(f"Unknown ssp_type '{ssp_type}'. Use 'hex' or 'rand'.")


class GymnaxSSPObsWrapper:
    """Wraps a gymnax env to encode observations as SSP vectors.

    Parameters
    ----------
    env:
        A gymnax environment instance.
    env_params:
        Default environment parameters (used when none are passed to
        ``reset`` / ``step``).
    ssp_space:
        Pre-built :class:`~vsagym.jax.sspspace.JaxSSPSpace` (or numpy
        :class:`~vsagym.spaces.sspspace.SSPSpace`).  When ``None``, a space
        is constructed automatically from the env's observation bounds.
    shape_out:
        Desired SSP dimensionality.  Ignored when ``ssp_space`` is provided.
    ssp_type:
        ``'hex'`` (default) or ``'rand'``.
    length_scale:
        Passed to the SSP constructor when building automatically.
    **kwargs:
        Additional keyword arguments forwarded to the SSP constructor.

    Examples
    --------
    >>> import jax
    >>> import gymnax
    >>> from vsagym.jax.gymnax_wrappers import GymnaxSSPObsWrapper
    >>>
    >>> env, env_params = gymnax.make("CartPole-v1")
    >>> wrapper = GymnaxSSPObsWrapper(env, env_params, shape_out=151)
    >>>
    >>> key = jax.random.PRNGKey(0)
    >>> obs, state = wrapper.reset(key)
    >>> obs.shape  # (151,)
    """

    def __init__(
        self,
        env,
        env_params,
        ssp_space: Optional[Union[JaxSSPSpace, SSPSpace]] = None,
        shape_out: Optional[int] = None,
        ssp_type: str = "hex",
        **kwargs,
    ):
        self.env = env
        self.env_params = env_params

        if ssp_space is not None:
            if isinstance(ssp_space, JaxSSPSpace):
                self.jax_ssp_space = ssp_space
            else:
                self.jax_ssp_space = JaxSSPSpace(ssp_space)
        else:
            obs_space = env.observation_space(env_params)
            obs_low = np.array(obs_space.low)
            obs_high = np.array(obs_space.high)
            domain_dim = int(np.prod(obs_space.shape))
            self.jax_ssp_space = _build_jax_ssp_space(
                obs_low, obs_high, domain_dim, shape_out, ssp_type, **kwargs
            )

        self.ssp_dim: int = self.jax_ssp_space.ssp_dim

    # ------------------------------------------------------------------
    # SSP encode helper (can be jit-wrapped independently)
    # ------------------------------------------------------------------

    def encode_obs(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode a single observation vector to an SSP. Shape: ``(ssp_dim,)``."""
        return ssp_encode(
            self.jax_ssp_space.phase_matrix,
            self.jax_ssp_space.length_scale,
            jnp.atleast_2d(obs.flatten()),
        )[0]

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self, key: jnp.ndarray,
              params=None):
        """Reset the environment and return an SSP-encoded observation.

        Returns
        -------
        ssp_obs : jnp.ndarray  shape ``(ssp_dim,)``
        state   : gymnax EnvState
        """
        if params is None:
            params = self.env_params
        obs, state = self.env.reset(key, params)
        return self.encode_obs(obs), state

    def step(self, key: jnp.ndarray, state, action: jnp.ndarray,
             params=None):
        """Step the environment and return an SSP-encoded observation.

        Returns
        -------
        ssp_obs  : jnp.ndarray  shape ``(ssp_dim,)``
        state    : gymnax EnvState
        reward   : jnp.ndarray  scalar
        done     : jnp.ndarray  bool scalar
        info     : dict
        """
        if params is None:
            params = self.env_params
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        return self.encode_obs(obs), state, reward, done, info


class GymnaxSSPEnvWrapper:
    """Wraps a gymnax env with SSP-encoded observations *and* decoded actions.

    The action space wrapper decodes SSP action vectors back to the original
    discrete/continuous action space before calling the underlying env.

    Parameters
    ----------
    env, env_params:
        gymnax environment and parameters.
    obs_ssp_space:
        Optional pre-built SSP space for observations.
    action_ssp_space:
        Optional pre-built SSP space for actions. If ``None``, actions are
        passed through unmodified.
    shape_out:
        SSP dimension (used for both obs and action if not specified separately).
    ssp_type:
        ``'hex'`` or ``'rand'``.
    **kwargs:
        Forwarded to the SSP constructors.

    Examples
    --------
    >>> wrapper = GymnaxSSPEnvWrapper(env, env_params, shape_out=151)
    >>> obs, state = wrapper.reset(key)
    >>> # action is an SSP vector; wrapper decodes before stepping the env
    >>> obs, state, reward, done, info = wrapper.step(key, state, ssp_action)
    """

    def __init__(
        self,
        env,
        env_params,
        obs_ssp_space: Optional[Union[JaxSSPSpace, SSPSpace]] = None,
        action_ssp_space: Optional[Union[JaxSSPSpace, SSPSpace]] = None,
        shape_out: Optional[int] = None,
        ssp_type: str = "hex",
        **kwargs,
    ):
        self._obs_wrapper = GymnaxSSPObsWrapper(
            env, env_params, obs_ssp_space, shape_out, ssp_type, **kwargs
        )
        self.env = env
        self.env_params = env_params
        self.ssp_dim: int = self._obs_wrapper.ssp_dim

        if action_ssp_space is not None:
            if isinstance(action_ssp_space, JaxSSPSpace):
                self._action_ssp = action_ssp_space
            else:
                self._action_ssp = JaxSSPSpace(action_ssp_space)
        else:
            self._action_ssp = None

    @property
    def jax_ssp_space(self) -> JaxSSPSpace:
        return self._obs_wrapper.jax_ssp_space

    def encode_obs(self, obs: jnp.ndarray) -> jnp.ndarray:
        return self._obs_wrapper.encode_obs(obs)

    def decode_action(self, ssp_action: jnp.ndarray) -> jnp.ndarray:
        """Decode an SSP action vector to domain action."""
        if self._action_ssp is None:
            return ssp_action
        return self._action_ssp.decode(jnp.atleast_2d(ssp_action))[0]

    def reset(self, key: jnp.ndarray, params=None):
        return self._obs_wrapper.reset(key, params)

    def step(self, key: jnp.ndarray, state, ssp_action: jnp.ndarray,
             params=None):
        """Step with an SSP action. Decodes before passing to the env."""
        if params is None:
            params = self.env_params
        action = self.decode_action(ssp_action)
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        return self.encode_obs(obs), state, reward, done, info
