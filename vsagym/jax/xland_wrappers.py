"""
xland-minigrid-compatible SSP observation wrappers.

These wrappers augment `xland-minigrid
<https://github.com/corl-team/xland-minigrid>`_ environments so that the
agent's pose (grid position + direction) is encoded as an SSP vector.

xland-minigrid uses a functional API::

    timestep = env.reset(env_params, key)
    timestep = env.step(env_params, timestep, action)

A ``Timestep`` is a pytree with fields:

* ``observation`` — the raw tile view (HxWx3 integer array)
* ``state``       — the full environment state (includes agent position)
* ``reward``
* ``discount``
* ``step_type``   — ``StepType.FIRST`` / ``MID`` / ``LAST``

The wrappers operate on the ``state`` fields to extract agent pose, which is
then SSP-encoded.  The resulting ``observation`` replaces the raw view.

Install xland-minigrid with:  ``pip install xminigrid``
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


def _replace_obs(timestep, new_obs: jnp.ndarray):
    """Return a new Timestep with ``observation`` replaced by ``new_obs``.

    xminigrid's TimeStep exposes ``.replace(**kwargs)`` (chex/flax dataclass).
    Falls back to ``._replace()`` for plain NamedTuples.
    """
    if hasattr(timestep, "replace"):
        return timestep.replace(observation=new_obs)
    if hasattr(timestep, "_replace"):
        return timestep._replace(observation=new_obs)
    raise TypeError(
        f"Cannot replace 'observation' on Timestep of type {type(timestep)}. "
        "It has neither a .replace() nor ._replace() method."
    )


def _get_agent_pose(state) -> jnp.ndarray:
    """Extract (x, y, direction) from an xminigrid State. Shape: ``(3,)``."""
    # xminigrid stores agent info in state.agent (AgentState dataclass)
    if hasattr(state, "agent"):
        pos = state.agent.position.astype(jnp.float32)        # (2,)
        direction = jnp.asarray(state.agent.direction, dtype=jnp.float32).reshape(())
    else:
        # Fallback: older API with flat fields
        pos = state.agent_position.astype(jnp.float32)
        direction = jnp.asarray(state.agent_direction, dtype=jnp.float32)
    return jnp.concatenate([pos, direction.reshape(1)])


class XLandSSPPoseWrapper:
    """Wraps an xland-minigrid env to encode agent pose as an SSP observation.

    The raw tile view is discarded; instead the observation becomes the SSP
    encoding of the agent's ``(x, y, direction)`` extracted from
    ``timestep.state.agent_position`` and ``timestep.state.agent_direction``.

    Parameters
    ----------
    env:
        An xland-minigrid environment instance.
    env_params:
        Default environment parameters.
    ssp_space:
        Pre-built SSP space.  When ``None``, a space is created automatically
        using the grid width/height from ``env_params``.
    shape_out:
        Desired SSP dimension.  Ignored when ``ssp_space`` is provided.
    ssp_type:
        ``'hex'`` (default) or ``'rand'``.
    n_directions:
        Number of discrete directions (default 4 for MiniGrid-style envs).
    **kwargs:
        Forwarded to the SSP constructor.

    Examples
    --------
    >>> import jax
    >>> import xminigrid
    >>> from vsagym.jax.xland_wrappers import XLandSSPPoseWrapper
    >>>
    >>> env, env_params = xminigrid.make("MiniGrid-Empty-5x5")
    >>> wrapper = XLandSSPPoseWrapper(env, env_params, shape_out=151)
    >>>
    >>> key = jax.random.PRNGKey(0)
    >>> timestep = wrapper.reset(env_params, key)
    >>> timestep.observation.shape  # (151,)
    """

    def __init__(
        self,
        env,
        env_params,
        ssp_space: Optional[Union[JaxSSPSpace, SSPSpace]] = None,
        shape_out: Optional[int] = None,
        ssp_type: str = "hex",
        n_directions: int = 4,
        **kwargs,
    ):
        self.env = env
        self.env_params = env_params
        self.n_directions = n_directions

        if ssp_space is not None:
            self.jax_ssp_space = (
                ssp_space
                if isinstance(ssp_space, JaxSSPSpace)
                else JaxSSPSpace(ssp_space)
            )
        else:
            # Infer grid bounds from env_params
            try:
                height = int(env_params.height)
                width = int(env_params.width)
            except AttributeError:
                # Fall back to generic large bounds
                height = 20
                width = 20

            domain_bounds = np.array([
                [0, width],
                [0, height],
                [0, n_directions],
            ], dtype=np.float32)

            if ssp_type == "hex":
                np_space = HexagonalSSPSpace(
                    3, ssp_dim=shape_out,
                    domain_bounds=domain_bounds, **kwargs
                )
            elif ssp_type == "rand":
                np_space = RandomSSPSpace(
                    3, ssp_dim=shape_out,
                    domain_bounds=domain_bounds, **kwargs
                )
            else:
                raise ValueError(f"Unknown ssp_type '{ssp_type}'.")

            self.jax_ssp_space = JaxSSPSpace(np_space)

        self.ssp_dim: int = self.jax_ssp_space.ssp_dim

    # ------------------------------------------------------------------

    def _encode_pose(self, state) -> jnp.ndarray:
        """Extract agent pose from state and SSP-encode it. Shape: ``(ssp_dim,)``."""
        pose = _get_agent_pose(state).reshape(1, 3)
        return ssp_encode(
            self.jax_ssp_space.phase_matrix,
            self.jax_ssp_space.length_scale,
            pose,
        )[0]

    def reset(self, env_params, key: jnp.ndarray):
        """Reset env and return a Timestep with SSP-encoded observation."""
        timestep = self.env.reset(env_params, key)
        ssp_obs = self._encode_pose(timestep.state)
        return _replace_obs(timestep, ssp_obs)

    def step(self, env_params, timestep, action: jnp.ndarray):
        """Step env and return a Timestep with SSP-encoded observation."""
        timestep = self.env.step(env_params, timestep, action)
        ssp_obs = self._encode_pose(timestep.state)
        return _replace_obs(timestep, ssp_obs)


class XLandSSPViewWrapper(XLandSSPPoseWrapper):
    """Wraps an xland-minigrid env to encode the view *and* pose as an SSP.

    Each visible cell is SSP-encoded at its local position, and the results
    are bundled (summed) with the agent pose SSP, producing a single vector
    that captures both *what* is visible and *where* the agent is.

    The tile view has shape ``(view_height, view_width, 3)`` where the three
    channels are (object_type, colour, state).  Objects that are not "empty"
    are bound to their local position SSP and bundled together.

    Parameters
    ----------
    All parameters from :class:`XLandSSPPoseWrapper`.
    view_height, view_width:
        Dimensions of the local view grid (default 5×5 for xland-minigrid).
    pose_weight:
        Scalar weight applied to the pose component before summing. Default 1.

    Examples
    --------
    >>> wrapper = XLandSSPViewWrapper(env, env_params, shape_out=151)
    >>> timestep = wrapper.reset(env_params, key)
    >>> timestep.observation.shape  # (151,)
    """

    EMPTY_OBJECT_IDX: int = 0   # tile type index for empty floor in xminigrid

    def __init__(
        self,
        env,
        env_params,
        ssp_space: Optional[Union[JaxSSPSpace, SSPSpace]] = None,
        shape_out: Optional[int] = None,
        ssp_type: str = "hex",
        n_directions: int = 4,
        view_height: int = 5,
        view_width: int = 5,
        pose_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(env, env_params, ssp_space, shape_out, ssp_type,
                         n_directions, **kwargs)
        self.view_height = view_height
        self.view_width = view_width
        self.pose_weight = pose_weight

        # Pre-compute local position SSPs for the view grid.
        # Local coords: row ∈ [0, view_height), col ∈ [0, view_width).
        rows = jnp.arange(view_height, dtype=jnp.float32)
        cols = jnp.arange(view_width, dtype=jnp.float32)
        rr, cc = jnp.meshgrid(rows, cols, indexing="ij")
        pts = jnp.stack([rr.reshape(-1), cc.reshape(-1)], axis=1)
        # Use RandomSSPSpace for local positions so the dim exactly matches ssp_dim.
        _local_bounds = np.array([[0, view_height], [0, view_width]], dtype=np.float32)
        _local_np = RandomSSPSpace(2, ssp_dim=self.ssp_dim,
                                   domain_bounds=_local_bounds)
        self._local_ssp = JaxSSPSpace(_local_np)
        # Pre-compute grid of local SSPs: (view_height*view_width, ssp_dim)
        self._grid_ssps = self._local_ssp.encode(pts)

    def _encode_view(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Bundle all non-empty objects in the view. Shape: ``(ssp_dim,)``."""
        # observation: (view_height, view_width, C) where C >= 1
        vsa_out = jnp.zeros(self.ssp_dim)
        n_channels = observation.shape[-1]
        flat_obs = observation.reshape(-1, n_channels)  # (H*W, C)
        obj_types = flat_obs[:, 0]                      # (H*W,) — tile type

        # For each cell with a non-empty object, bind the object type SP
        # (here we use a simple one-hot-like encoding) with its position SSP.
        # We superimpose all contributions.
        def _add_cell(carry, inputs):
            vsa, grid_ssps = carry
            obj_type, pos_ssp = inputs
            is_nonempty = (obj_type != self.EMPTY_OBJECT_IDX).astype(jnp.float32)
            # Simple approach: use obj_type as a scalar multiplier on pos_ssp
            contribution = is_nonempty * pos_ssp * obj_type.astype(jnp.float32)
            return (vsa + contribution, grid_ssps), None

        (vsa_out, _), _ = jax.lax.scan(
            _add_cell,
            (vsa_out, self._grid_ssps),
            (obj_types, self._grid_ssps),
        )
        return vsa_out

    def _encode_pose(self, state) -> jnp.ndarray:  # type: ignore[override]
        # Reuse parent pose encoding
        return super()._encode_pose(state)

    def _wrap_timestep(self, timestep) -> jnp.ndarray:
        pose_ssp = self._encode_pose(timestep.state)
        view_ssp = self._encode_view(timestep.observation)
        combined = self.pose_weight * pose_ssp + view_ssp
        norm = jnp.linalg.norm(combined)
        ssp_obs = jnp.where(norm > 1e-8, combined / norm, combined)
        return _replace_obs(timestep, ssp_obs)

    def reset(self, env_params, key: jnp.ndarray):
        timestep = self.env.reset(env_params, key)
        return self._wrap_timestep(timestep)

    def step(self, env_params, timestep, action: jnp.ndarray):
        timestep = self.env.step(env_params, timestep, action)
        return self._wrap_timestep(timestep)
