"""
PPO + SSP View Encoding for xland-minigrid
==========================================

Trains a PPO agent on xland-minigrid environments (MiniGrid or XLand variants),
replacing the standard CNN feature extractor with a learned SSP scene encoder.

SSP encoding design
-------------------
The raw observation is a (view_size, view_size, 2) uint8 array where:
  channel 0 = tile type id (0..NUM_TILES-1)
  channel 1 = color id     (0..NUM_COLORS-1)

For every cell (i, j) we compute:

  cell_vec(i,j) = pos_ssp(i,j) ⊛ tile_emb[tile_id] ⊛ color_emb[color_id]

where:
  - pos_ssp(i,j) = IFFT{ exp(i A [i/H, j/W]^T / l) } is a learned SSP of
    position, parameterized by phase matrix A and length scale l.
  - tile_emb, color_emb  are learned HRR embedding lookup tables.
  - ⊛ denotes circular convolution (binding).

The scene vector is the normalised sum (bundling) over all cells:
  φ = normalize( Σ_{i,j} cell_vec(i,j) )

Both A and l are trained end-to-end with PPO.

Baseline comparison
-------------------
The CNN baseline uses a Conv2D feature extractor on the same observation,
which can be selected with --encoder cnn.

Usage
-----
# Quick sanity check (MiniGrid, 100 updates):
python xland_example.py --env MiniGrid-FourRooms --n_updates 100 --n_envs 8

# Longer run on MiniGrid:
python xland_example.py --env MiniGrid-FourRooms --n_updates 500 \
       --n_envs 32 --ssp_dim 128 --lr 3e-4

# XLand (R1, small grid):
python xland_example.py --env XLand-MiniGrid-R1-9x9 --n_updates 1000 \
       --n_envs 32 --ssp_dim 127

# CNN baseline comparison (same command, different --encoder):
python xland_example.py --env MiniGrid-FourRooms --encoder cnn --n_updates 500

# Compare SSP vs CNN across 3 seeds:
python xland_example.py --env MiniGrid-FourRooms --n_updates 300 \
       --n_envs 32 --ssp_dim 127 --n_seeds 3
python xland_example.py --env MiniGrid-FourRooms --encoder cnn \
       --n_updates 300 --n_envs 32 --n_seeds 3

Tips
----
- Use an odd ssp_dim (e.g. 127, 255) to get exactly that many SSP dimensions.
  Even values are silently reduced by 1 due to conjugate-symmetry constraints.
- The SSP phase matrix and length scale are initialised randomly and trained
  end-to-end.  With --lr_decay the learning rate anneals for the final 10 %.
- Compilation is slow on first call (~30s for ssp_dim=127); subsequent jit
  calls are fast.  Run with a small --n_updates first to check things work.
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import xminigrid
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from xminigrid.core.constants import NUM_COLORS, NUM_TILES
from xminigrid.wrappers import GymAutoResetWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bind(a: jax.Array, b: jax.Array) -> jax.Array:
    """Circular-convolution binding (HRR product). Broadcasts over batch."""
    return jnp.fft.ifft(
        jnp.fft.fft(a, axis=-1) * jnp.fft.fft(b, axis=-1), axis=-1
    ).real


def _ssp_encode(phase_matrix: jax.Array, length_scale: jax.Array,
                x: jax.Array) -> jax.Array:
    """Encode points x ∈ R^d as SSP vectors.

    Args:
        phase_matrix: (ssp_dim, domain_dim)
        length_scale: (domain_dim,)
        x: (n, domain_dim)
    Returns:
        (n, ssp_dim)
    """
    ls_mat = jnp.diag(1.0 / length_scale)
    scaled = x @ ls_mat                                     # (n, domain_dim)
    exponents = 1j * (phase_matrix @ scaled.T)              # (ssp_dim, n)
    return jnp.fft.ifft(jnp.exp(exponents), axis=0).real.T  # (n, ssp_dim)


def _build_phase_matrix(phase_raw: jax.Array, ssp_dim: int) -> jax.Array:
    """Reconstruct the full conjugate-symmetric phase matrix from its upper half."""
    n = phase_raw.shape[0]   # = (ssp_dim - 1) // 2
    zeros = jnp.zeros((1, phase_raw.shape[1]))
    upper = phase_raw
    lower = -jnp.flip(phase_raw, axis=0)
    return jnp.concatenate([zeros, upper, lower], axis=0)  # (ssp_dim, domain_dim)


# ---------------------------------------------------------------------------
# SSP Scene Encoder  (Flax module)
# ---------------------------------------------------------------------------


class SSPViewEncoder(nn.Module):
    """Encode a (H, W, 2) xminigrid tile grid as a single SSP scene vector.

    Parameters
    ----------
    ssp_dim:
        Dimensionality of SSP vectors.  Actual dimension = 2*n_phases + 1 where
        n_phases = (ssp_dim - 1) // 2.
    view_height, view_width:
        View grid dimensions (typically env_params.view_size × view_size).
    init_phase_scale:
        Std of the random normal initialiser for the phase matrix.
    """

    ssp_dim: int = 128
    view_height: int = 5
    view_width: int = 5
    init_phase_scale: float = 0.5

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        """
        Args:
            obs: (H, W, 2) uint8 — tile grid from xminigrid
        Returns:
            (actual_ssp_dim,) float32 — normalised scene SSP

        Note: actual_ssp_dim = 2 * n_phases + 1 where n_phases = (ssp_dim-1)//2.
        For even ssp_dim this is ssp_dim-1; for odd ssp_dim it equals ssp_dim.
        Use an odd ssp_dim to get the exact requested dimension.
        """
        H, W, _ = obs.shape
        n_phases = (self.ssp_dim - 1) // 2
        # Actual SSP dimension after enforcing conjugate symmetry
        actual_dim = 2 * n_phases + 1

        # ---- Learned SSP parameters ----------------------------------------
        # Phase matrix upper half: (n_phases, 2) — 2-D position domain
        phase_raw = self.param(
            "phase_matrix",
            nn.initializers.normal(self.init_phase_scale),
            (n_phases, 2),
        )
        # Length scale in log space: (2,) — one per spatial dimension
        log_ls = self.param(
            "log_length_scale",
            nn.initializers.zeros,
            (2,),
        )

        # ---- Learned symbol embeddings -------------------------------------
        # Must use actual_dim, not ssp_dim, to match the phase matrix output
        scale = 1.0 / jnp.sqrt(actual_dim)
        tile_emb = self.param(
            "tile_embeddings",
            nn.initializers.normal(scale),
            (NUM_TILES, actual_dim),
        )
        color_emb = self.param(
            "color_embeddings",
            nn.initializers.normal(scale),
            (NUM_COLORS, actual_dim),
        )

        # ---- Construct phase matrix and length scale -----------------------
        phase_matrix = _build_phase_matrix(phase_raw, self.ssp_dim)  # (ssp_dim, 2)
        length_scale = jnp.exp(log_ls)                                # (2,)

        # ---- Grid positions ------------------------------------------------
        rows = jnp.arange(H, dtype=jnp.float32) / H
        cols = jnp.arange(W, dtype=jnp.float32) / W
        rr, cc = jnp.meshgrid(rows, cols, indexing="ij")
        grid_pts = jnp.stack([rr.reshape(-1), cc.reshape(-1)], axis=1)  # (H*W, 2)

        # ---- Position SSPs -------------------------------------------------
        pos_ssps = _ssp_encode(phase_matrix, length_scale, grid_pts)  # (H*W, ssp_dim)

        # ---- Symbol vectors ------------------------------------------------
        tile_ids = obs[:, :, 0].reshape(-1).astype(jnp.int32)    # (H*W,)
        color_ids = obs[:, :, 1].reshape(-1).astype(jnp.int32)   # (H*W,)

        tile_vecs = tile_emb[tile_ids]    # (H*W, ssp_dim)
        color_vecs = color_emb[color_ids] # (H*W, ssp_dim)

        # ---- Binding: pos ⊛ tile ⊛ color -----------------------------------
        cell_vecs = _bind(pos_ssps, _bind(tile_vecs, color_vecs))  # (H*W, ssp_dim)

        # ---- Bundling: sum over all cells ----------------------------------
        scene_vec = jnp.sum(cell_vecs, axis=0)                        # (ssp_dim,)
        scene_vec = scene_vec / jnp.maximum(jnp.linalg.norm(scene_vec), 1e-8)

        return scene_vec


# ---------------------------------------------------------------------------
# CNN Baseline Encoder  (Flax module)
# ---------------------------------------------------------------------------


class CNNViewEncoder(nn.Module):
    """Flat CNN over the (H, W, 2) tile grid — baseline comparison."""

    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        # Cast to float and normalise tile ids to [0, 1]
        x = obs.astype(jnp.float32) / max(NUM_TILES, NUM_COLORS)  # (H, W, 2)
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape(-1)  # flatten
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return x


# ---------------------------------------------------------------------------
# Actor-Critic  (Flax module)
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Actor-critic network with pluggable feature extractor.

    Parameters
    ----------
    n_actions:
        Number of discrete actions.
    encoder:
        ``'ssp'`` (default) or ``'cnn'``.
    ssp_dim:
        SSP dimension (used when encoder='ssp').
    hidden_dim:
        Width of the MLP trunk.
    view_height, view_width:
        View grid shape (must match env_params.view_size).
    """

    n_actions: int = 6
    encoder: str = "ssp"
    ssp_dim: int = 128
    hidden_dim: int = 256
    view_height: int = 5
    view_width: int = 5

    @nn.compact
    def __call__(self, obs: jax.Array):
        """
        Args:
            obs: (H, W, 2) uint8 observation
        Returns:
            logits: (n_actions,)  raw policy logits
            value:  ()             state value
        """
        # Feature extraction
        if self.encoder == "ssp":
            feat = SSPViewEncoder(
                ssp_dim=self.ssp_dim,
                view_height=self.view_height,
                view_width=self.view_width,
            )(obs)
        else:  # cnn
            feat = CNNViewEncoder(hidden_dim=self.ssp_dim)(obs)

        # Shared MLP trunk
        x = nn.Dense(self.hidden_dim)(feat)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)

        # Policy and value heads
        logits = nn.Dense(self.n_actions)(x)
        value = nn.Dense(1)(x).squeeze(-1)

        return logits, value


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


class Transition(NamedTuple):
    obs: jax.Array        # (n_envs, H, W, 2)
    action: jax.Array     # (n_envs,)
    log_prob: jax.Array   # (n_envs,)
    value: jax.Array      # (n_envs,)
    reward: jax.Array     # (n_envs,)
    done: jax.Array       # (n_envs,)  True on terminal step


# ---------------------------------------------------------------------------
# PPO utilities
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: jax.Array,   # (T, n_envs)
    values: jax.Array,    # (T, n_envs)
    dones: jax.Array,     # (T, n_envs)
    last_value: jax.Array,  # (n_envs,)
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute GAE(λ) advantages and returns.

    Returns
    -------
    advantages: (T, n_envs)
    returns:    (T, n_envs)
    """

    def _body(carry, inputs):
        gae, next_val = carry
        rew, val, done = inputs
        delta = rew + gamma * next_val * (1.0 - done) - val
        gae = delta + gamma * lam * (1.0 - done) * gae
        return (gae, val), gae

    _, advantages_rev = jax.lax.scan(
        _body,
        init=(jnp.zeros_like(last_value), last_value),
        xs=(rewards[::-1], values[::-1], dones[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


def ppo_loss(
    params: Any,
    apply_fn: Any,
    obs: jax.Array,        # (batch, H, W, 2)
    actions: jax.Array,    # (batch,)
    old_log_probs: jax.Array,  # (batch,)
    advantages: jax.Array,  # (batch,)
    returns: jax.Array,     # (batch,)
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[jax.Array, dict]:
    """PPO clipped surrogate loss."""
    logits, values = jax.vmap(partial(apply_fn, params))(obs)

    # Policy loss
    log_probs = jax.nn.log_softmax(logits)
    log_prob = log_probs[jnp.arange(actions.shape[0]), actions]
    ratio = jnp.exp(log_prob - old_log_probs)
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg_loss1 = -adv_norm * ratio
    pg_loss2 = -adv_norm * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

    # Value loss
    vf_loss = jnp.mean((values - returns) ** 2)

    # Entropy bonus
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    ent_loss = -jnp.mean(entropy)

    total_loss = pg_loss + vf_coef * vf_loss + ent_coef * ent_loss
    info = {
        "loss": total_loss,
        "pg_loss": pg_loss,
        "vf_loss": vf_loss,
        "entropy": -ent_loss,
        "approx_kl": jnp.mean((ratio - 1) - jnp.log(ratio)),
    }
    return total_loss, info


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_train(config: argparse.Namespace):
    """Build training state and a JIT-compiled single-update-step function.

    Key design: only ONE update step is JIT-compiled.  The outer loop over
    n_updates is a plain Python for-loop so JAX never has to trace 500 copies
    of the graph at once.  This keeps compilation time to ~10-30 s regardless
    of n_updates or ssp_dim.
    """

    # ---- Environment setup -----------------------------------------------
    base_env, env_params = xminigrid.make(config.env)
    env = GymAutoResetWrapper(base_env)
    n_actions = env.num_actions(env_params)
    view_size = env_params.view_size

    # ---- Network setup ---------------------------------------------------
    network = ActorCritic(
        n_actions=n_actions,
        encoder=config.encoder,
        ssp_dim=config.ssp_dim,
        hidden_dim=config.hidden_dim,
        view_height=view_size,
        view_width=view_size,
    )

    # ---- Build initial training state ------------------------------------
    def init(rng: jax.Array):
        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((view_size, view_size, 2), dtype=jnp.uint8)
        params = network.init(init_rng, dummy_obs)

        if config.lr_decay:
            lr_schedule = optax.linear_schedule(
                init_value=config.lr,
                end_value=config.lr * 0.1,
                transition_steps=config.n_updates,
            )
        else:
            lr_schedule = config.lr

        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(lr_schedule, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply, params=params, tx=tx
        )

        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config.n_envs)
        timestep = jax.vmap(partial(env.reset, env_params))(reset_rngs)
        return train_state, timestep, rng

    # ---- Single update step (this is the function we JIT) ----------------
    def update_step(train_state, timestep, rng):
        """Collect n_steps rollout + PPO update.  Returns updated state + metrics."""

        # -- Rollout (scan over n_steps) -----------------------------------
        def env_step(carry, _):
            ts, rng = carry
            rng, action_rng = jax.random.split(rng)

            obs = ts.observation                           # (n_envs, H, W, 2)
            logits, values = jax.vmap(
                partial(train_state.apply_fn, train_state.params)
            )(obs)

            action_rngs = jax.random.split(action_rng, config.n_envs)
            actions = jax.vmap(jax.random.categorical)(action_rngs, logits)

            log_probs_all = jax.nn.log_softmax(logits)
            log_probs = log_probs_all[jnp.arange(config.n_envs), actions]

            transition = Transition(
                obs=obs,
                action=actions,
                log_prob=log_probs,
                value=values,
                reward=ts.reward,
                done=ts.last().astype(jnp.float32),
            )

            ts_next = jax.vmap(partial(env.step, env_params))(ts, actions)
            return (ts_next, rng), transition

        (timestep_after, rng), transitions = jax.lax.scan(
            env_step,
            init=(timestep, rng),
            xs=None,
            length=config.n_steps,
        )

        # -- Bootstrap + GAE -----------------------------------------------
        _, last_values = jax.vmap(
            partial(train_state.apply_fn, train_state.params)
        )(timestep_after.observation)

        advantages, returns = compute_gae(
            rewards=transitions.reward,
            values=transitions.value,
            dones=transitions.done,
            last_value=last_values,
            gamma=config.gamma,
            lam=config.gae_lambda,
        )

        # -- Flatten batch -------------------------------------------------
        batch_size = config.n_steps * config.n_envs
        mb_size = batch_size // config.n_minibatches
        obs_flat  = transitions.obs.reshape(batch_size, view_size, view_size, 2)
        act_flat  = transitions.action.reshape(batch_size)
        logp_flat = transitions.log_prob.reshape(batch_size)
        adv_flat  = advantages.reshape(batch_size)
        ret_flat  = returns.reshape(batch_size)

        # -- PPO epochs (scan over epochs, scan over mini-batches) ---------
        def update_epoch(ts, rng_epoch):
            rng_epoch, perm_rng = jax.random.split(rng_epoch)
            perm = jax.random.permutation(perm_rng, batch_size)

            def _mb(arr, extra=()):
                return arr[perm].reshape((config.n_minibatches, mb_size) + extra)

            obs_mbs  = _mb(obs_flat,  (view_size, view_size, 2))
            act_mbs  = _mb(act_flat)
            logp_mbs = _mb(logp_flat)
            adv_mbs  = _mb(adv_flat)
            ret_mbs  = _mb(ret_flat)

            def update_mb(ts_inner, mb):
                o, a, lp, adv, ret = mb

                def loss_fn(params):
                    return ppo_loss(
                        params=params,
                        apply_fn=ts_inner.apply_fn,
                        obs=o, actions=a, old_log_probs=lp,
                        advantages=adv, returns=ret,
                        clip_eps=config.clip_eps,
                        vf_coef=config.vf_coef,
                        ent_coef=config.ent_coef,
                    )

                (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    ts_inner.params
                )
                return ts_inner.apply_gradients(grads=grads), info

            ts_out, infos = jax.lax.scan(
                update_mb,
                init=ts,
                xs=(obs_mbs, act_mbs, logp_mbs, adv_mbs, ret_mbs),
            )
            return ts_out, (rng_epoch, infos)

        rng, epoch_rng = jax.random.split(rng)
        epoch_rngs = jax.random.split(epoch_rng, config.n_epochs)
        train_state, (_, epoch_infos) = jax.lax.scan(
            update_epoch, init=train_state, xs=epoch_rngs,
        )

        metrics = {
            "mean_reward": transitions.reward.mean(),
            "loss":        epoch_infos["loss"].mean(),
            "pg_loss":     epoch_infos["pg_loss"].mean(),
            "vf_loss":     epoch_infos["vf_loss"].mean(),
            "entropy":     epoch_infos["entropy"].mean(),
            "approx_kl":   epoch_infos["approx_kl"].mean(),
        }
        return train_state, timestep_after, rng, metrics

    return init, jax.jit(update_step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="PPO + SSP view encoding on xland-minigrid"
    )
    # Environment
    p.add_argument(
        "--env",
        default="MiniGrid-FourRooms",
        help="xminigrid environment id (default: MiniGrid-FourRooms)",
    )
    # Encoder
    p.add_argument(
        "--encoder",
        choices=["ssp", "cnn"],
        default="ssp",
        help="Feature extractor: 'ssp' (default) or 'cnn' baseline",
    )
    p.add_argument("--ssp_dim", type=int, default=128,
                   help="SSP vector dimensionality (default: 128)")
    p.add_argument("--hidden_dim", type=int, default=256,
                   help="MLP hidden layer width (default: 256)")
    # PPO hyper-parameters
    p.add_argument("--n_envs", type=int, default=16,
                   help="Parallel environments (default: 16)")
    p.add_argument("--n_steps", type=int, default=128,
                   help="Rollout steps per update (default: 128)")
    p.add_argument("--n_updates", type=int, default=300,
                   help="Number of PPO update cycles (default: 300)")
    p.add_argument("--n_epochs", type=int, default=4,
                   help="PPO epochs per update (default: 4)")
    p.add_argument("--n_minibatches", type=int, default=4,
                   help="Mini-batches per epoch (default: 4)")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Adam learning rate (default: 3e-4)")
    p.add_argument("--lr_decay", action="store_true",
                   help="Linearly decay LR to lr*0.1 over n_updates")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor (default: 0.99)")
    p.add_argument("--gae_lambda", type=float, default=0.95,
                   help="GAE lambda (default: 0.95)")
    p.add_argument("--clip_eps", type=float, default=0.2,
                   help="PPO clip epsilon (default: 0.2)")
    p.add_argument("--vf_coef", type=float, default=0.5,
                   help="Value loss coefficient (default: 0.5)")
    p.add_argument("--ent_coef", type=float, default=0.01,
                   help="Entropy bonus coefficient (default: 0.01)")
    p.add_argument("--max_grad_norm", type=float, default=0.5,
                   help="Global gradient clip norm (default: 0.5)")
    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=10,
                   help="Print metrics every N updates (default: 10)")
    p.add_argument("--n_seeds", type=int, default=1,
                   help="Number of independent seeds to run (default: 1)")
    return p.parse_args()


def print_metrics(update: int, metrics: dict, elapsed: float, steps: int = 0):
    """Pretty-print a row of training metrics."""
    throughput = f"{steps/elapsed:.0f} sps" if elapsed > 0 and steps > 0 else ""
    print(
        f"[{update:4d}]  "
        f"mean_rew={float(metrics['mean_reward']):.4f}  "
        f"loss={float(metrics['loss']):.4f}  "
        f"pg={float(metrics['pg_loss']):.4f}  "
        f"vf={float(metrics['vf_loss']):.4f}  "
        f"ent={float(metrics['entropy']):.3f}  "
        f"kl={float(metrics['approx_kl']):.4f}  "
        f"{throughput}",
        flush=True,
    )


def run_training(args, seed: int):
    """Run one full training run and return (final_train_state, reward_history)."""
    init_fn, jit_step = make_train(args)
    rng = jax.random.PRNGKey(seed)

    # Initialise (fast, no JIT)
    train_state, timestep, rng = init_fn(rng)

    print("Compiling update step... (one-off, ~10-30 s for large ssp_dim)")
    t0 = time.time()

    # Force compilation by running the first step
    train_state, timestep, rng, metrics = jit_step(train_state, timestep, rng)
    jax.block_until_ready(metrics["mean_reward"])
    print(f"Compiled in {time.time() - t0:.1f}s\n")

    print(f"{'Update':>6}  {'MeanRew':>9}  {'Loss':>8}  {'PG':>8}  "
          f"{'VF':>8}  {'Ent':>7}  {'KL':>8}  {'Time/step':>10}")
    print("-" * 75)

    reward_history = [float(metrics["mean_reward"])]
    t_last = time.time()

    # Print step 0 (already computed above)
    print_metrics(0, metrics, time.time() - t_last, args.n_steps * args.n_envs)
    t_last = time.time()

    # Python for-loop — only ONE step is JIT-compiled, loop itself is Python
    for i in range(1, args.n_updates):
        train_state, timestep, rng, metrics = jit_step(train_state, timestep, rng)
        reward_history.append(float(metrics["mean_reward"]))

        if i % args.log_interval == 0 or i == args.n_updates - 1:
            jax.block_until_ready(metrics["mean_reward"])
            elapsed = time.time() - t_last
            print_metrics(i, metrics, elapsed, args.log_interval * args.n_steps * args.n_envs)
            t_last = time.time()

    return train_state, reward_history


def main():
    args = parse_args()

    total_steps = args.n_updates * args.n_steps * args.n_envs
    print(f"\n{'='*70}")
    print(f"  env        : {args.env}")
    print(f"  encoder    : {args.encoder.upper()}  (ssp_dim={args.ssp_dim})")
    print(f"  n_envs     : {args.n_envs}  |  n_steps={args.n_steps}  |  n_updates={args.n_updates}")
    print(f"  total steps: {total_steps:,}")
    print(f"  PPO        : epochs={args.n_epochs}  mb={args.n_minibatches}  "
          f"lr={args.lr}  γ={args.gamma}  λ={args.gae_lambda}")
    print(f"{'='*70}\n", flush=True)

    rewards_per_seed = []
    for s in range(args.n_seeds):
        if args.n_seeds > 1:
            print(f"\n--- seed {s} ---")
        final_state, reward_history = run_training(args, seed=args.seed + s)
        rewards_per_seed.append(reward_history[-1])
        print(f"\nFinal mean reward (seed {s}): {reward_history[-1]:.4f}")

    if args.n_seeds > 1:
        arr = np.array(rewards_per_seed)
        print(f"\n{args.n_seeds} seeds — mean±std: {arr.mean():.4f} ± {arr.std():.4f}")

    return final_state, reward_history


if __name__ == "__main__":
    main()
