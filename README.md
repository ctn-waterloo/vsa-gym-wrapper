# vsagym: Gymnasium Wrappers Using Vector Symbolic Architecture (VSA)

`vsagym` is a Python package that provides Gymnasium environment wrappers, spaces, and feature extractors for working with **Vector Symbolic Architecture (VSA)** and **Spatial Semantic Pointer (SSP)** representations in reinforcement learning.

A **JAX backend** (`vsagym.jax`) provides JIT-compiled, vmap-compatible SSP operations and wrappers for functional JAX-based RL environments ([gymnax](https://github.com/RobertTLange/gymnax), [xland-minigrid](https://github.com/corl-team/xland-minigrid)).

---

## Installation

```bash
pip install -e .
```

### Optional backends

```bash
# JAX (JIT-compiled SSP operations)
pip install -e ".[jax]"

# gymnax integration
pip install -e ".[gymnax]"

# xland-minigrid integration
pip install -e ".[xland]"
```

### Core dependencies
- `numpy`, `scipy` — core SSP math
- `gymnasium` — base environment interface
- `torch` — PyTorch feature extractors
- `minigrid` — MiniGrid environment wrappers

---

## Background: SSPs and VSAs

**Vector Symbolic Architecture (VSA)** represents structured information in high-dimensional vectors using algebraic operations. **Spatial Semantic Pointers (SSPs)** encode continuous spatial data as:

$$\phi(\mathbf{x}) = \mathcal{F}^{-1}\!\left\{ e^{i A \mathbf{x} / \ell} \right\}$$

where $A \in \mathbb{R}^{D \times n}$ is the phase matrix, $\ell \in \mathbb{R}^n$ is the length scale, and $\mathcal{F}^{-1}$ is the inverse DFT. Two key operations compose representations:

- **Binding** ($\circledast$, circular convolution): encodes relational structure, e.g. $\phi_{\text{red}} \circledast \phi_{\text{ball}}$
- **Bundling** ($+$, vector addition): encodes sets, e.g. $\phi_1 + \phi_2 + \phi_3$

SSP representations are useful in RL because they:
- Encode continuous and discrete data in a unified vector space
- Preserve metric structure (nearby states have similar SSPs)
- Support symbolic manipulations (query, compare, compose)
- Are differentiable — parameters $A$ and $\ell$ can be learned

---

## Features at a Glance

| Module | What it does |
|---|---|
| `vsagym.spaces` | Gymnasium-compatible SSP spaces (Box, Discrete, Sequence, Dict) |
| `vsagym.wrappers` | Observation/action wrappers for any gym env; MiniGrid wrappers |
| `vsagym.networks` | Trainable PyTorch SSP feature extractors |
| `vsagym.jax` | JAX: JIT-compiled SSP ops + gymnax/xland wrappers |

---

## Quick Start

### Basic observation wrapper

```python
import gymnasium as gym
from vsagym.wrappers import SSPObsWrapper

env = gym.make("CartPole-v1")
env = SSPObsWrapper(env, shape_out=256)

obs, _ = env.reset()
print(obs.shape)   # (256,)  — SSP-encoded CartPole observation

for _ in range(200):
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, _ = env.reset()
```

### Custom SSP space

```python
from vsagym.spaces import SSPBox, HexagonalSSPSpace
import numpy as np

# Build the SSP space manually for full control
bounds = np.column_stack([env.observation_space.low,
                           env.observation_space.high])
ssp_space = HexagonalSSPSpace(domain_dim=4, ssp_dim=256,
                               domain_bounds=bounds, length_scale=0.1)

box = SSPBox(low=env.observation_space.low,
             high=env.observation_space.high,
             shape_out=256, ssp_space=ssp_space)

phi = box.encode(obs)          # (1, 256)
x_hat = box.decode(phi)        # (1, 4)  approximate inverse
```

---

## Gymnasium Spaces

### `SSPBox` — continuous observations

```python
from vsagym.spaces import SSPBox
import numpy as np

box = SSPBox(low=-1.0, high=1.0, shape_in=(2,), shape_out=256)

phi = box.encode(np.array([[0.3, -0.5]]))   # (1, 256)
x   = box.decode(phi)                        # (1, 2)
s   = box.sample()                           # (256,)
```

**Constructor options:**
- `ssp_space="hex"` (default) — Hexagonal SSP tiling
- `ssp_space="rand"` — Random Fourier Features
- `ssp_space=<SSPSpace>` — provide your own phase matrix
- `length_scale` — scalar or per-dimension array

### `SSPDiscrete` — discrete/categorical observations

```python
from vsagym.spaces import SSPDiscrete

disc = SSPDiscrete(n=5, shape_out=128)

phi = disc.encode(np.array([2]))    # (1, 128)
i   = disc.decode(phi)              # 2
s   = disc.sample()                 # (128,)
```

### `SSPSequence` — fixed-length sequences

```python
from vsagym.spaces import SSPBox, SSPSequence

feat  = SSPBox(-1.0, 1.0, shape_in=(1,), shape_out=128)
seq   = SSPSequence(feat, length=5)

traj  = np.array([[0.1, 0.3, -0.2, 0.8, 0.5]])   # (1, 5)
phi   = seq.encode(traj)                             # (1, 128)
x_hat = seq.decode(phi)                              # (5, 1)
```

### `SSPDict` — multi-modal compositional encoding

```python
from vsagym.spaces import SSPBox, SSPDiscrete, SSPDict
from vsagym.spaces.ssp import SSP

obs_space = SSPDict({
    "position": SSPBox(-1.0, 1.0, shape_in=(2,), shape_out=128),
    "color":    SSPDiscrete(6, shape_out=128),
})

# Define a custom encoding: bind position and color, then normalize
def encode_fn(ssp_dict, static_spaces):
    result = ssp_dict["position"] * ssp_dict["color"]   # binding
    return result.v.flatten() / np.linalg.norm(result.v)

obs_space.set_encode(encode_fn)

obs = {"position": np.array([[0.3, -0.5]]), "color": np.array([2])}
phi = obs_space.encode(obs)   # (128,)
```

---

## Gymnasium Wrappers

### `SSPObsWrapper` — encode observations only

```python
from vsagym.wrappers import SSPObsWrapper

env = SSPObsWrapper(gym.make("CartPole-v1"), shape_out=256)
obs, _ = env.reset()   # (256,)
```

### `SSPEnvWrapper` — encode obs and/or decode actions

```python
from vsagym.wrappers import SSPEnvWrapper

env = SSPEnvWrapper(
    gym.make("CartPole-v1"),
    auto_convert_obs_space=True,     # SSP-encode observations
    auto_convert_action_space=True,  # SSP-decode actions
    shape_out=256,
    decoder_method="from-set",
)
obs, _ = env.reset()
action = env.action_space.sample()  # sample an SSP action
obs, reward, *_ = env.step(action)
```

---

## MiniGrid Wrappers

`vsagym` provides rich VSA encodings for [MiniGrid](https://github.com/Farama-Foundation/Minigrid) environments.

### Pose encoding only

```python
from vsagym.wrappers import SSPMiniGridPoseWrapper
import gymnasium as gym

env = gym.make("MiniGrid-Empty-8x8-v0")
env = SSPMiniGridPoseWrapper(env, shape_out=256)

obs, _ = env.reset()
# obs["image"] is a 256-D SSP of (x, y, direction)
```

### View + pose encoding

```python
from vsagym.wrappers import SSPMiniGridViewWrapper

env = SSPMiniGridViewWrapper(
    gym.make("MiniGrid-FourRooms-v0"),
    shape_out=256,
    obj_encoding="allbound",  # or "slotfiller"
    view_type="local",        # or "global"
)
obs, _ = env.reset()
# obs["image"] encodes agent pose + objects in view
```

The observation is:
$$\Phi = \phi(x,y,\theta) + \texttt{HAS} \circledast \sum_{\text{carried}} \texttt{OBJ}_i \circledast \texttt{COL}_i \circledast \texttt{STATE}_i + \sum_{\text{in view}} \Delta\phi_i \circledast \texttt{OBJ}_i \circledast \texttt{COL}_i \circledast \texttt{STATE}_i$$

### Mission + view + pose

```python
from vsagym.wrappers import SSPMiniGridMissionWrapper

env = SSPMiniGridMissionWrapper(
    gym.make("BabyAI-GoToObj-v0"),
    shape_out=256
)
obs, _ = env.reset()
# obs["image"] is a 512-D vector: [scene_encoding | mission_encoding]
```

### Factory function

```python
from vsagym.wrappers import SSPMiniGridWrapper

env = SSPMiniGridWrapper(
    gym.make("MiniGrid-FourRooms-v0"),
    encode_pose=True,
    encode_view=True,
    encode_mission=True,
    shape_out=256,
)
```

---

## PyTorch Feature Extractors

These modules are compatible with `stable_baselines3` and other PyTorch RL frameworks. They learn the SSP parameters ($A$ and $\ell$) end-to-end.

### `SSPFeaturesExtractor`

```python
from vsagym.networks import SSPFeaturesExtractor
from gymnasium.spaces import Box
import torch

obs_space = Box(low=-1.0, high=1.0, shape=(4,))
extractor = SSPFeaturesExtractor(
    obs_space, features_dim=256,
    basis_type="hex",   # or "rand"
    learn_phase_matrix=True,
    learn_ls=True,
)

obs = torch.randn(32, 4)
phi = extractor(obs)   # (32, 256)
```

### `SSPHexFeaturesExtractor`

Parametrizes the Hexagonal SSP space via learnable scales and rotation matrices:

```python
from vsagym.networks import SSPHexFeaturesExtractor

extractor = SSPHexFeaturesExtractor(obs_space, features_dim=256)
phi = extractor(obs)
```

### MiniGrid feature extractors

```python
from vsagym.networks import SSPMiniGridViewFeatures, SSPMiniGridMissionFeatures

# Use with PrepMiniGridViewWrapper / PrepMiniGridMissionWrapper
extractor = SSPMiniGridViewFeatures(obs_space, features_dim=256)
```

### Integration with stable_baselines3

```python
from stable_baselines3 import PPO
from vsagym.wrappers import SSPMiniGridPoseWrapper
from vsagym.networks import SSPFeaturesExtractor

env = SSPMiniGridPoseWrapper(gym.make("MiniGrid-Empty-8x8-v0"),
                              shape_out=256)
env = FlatWrapper(env)

policy_kwargs = dict(
    features_extractor_class=SSPFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=256, basis_type="hex"),
)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=100_000)
```

---

## JAX Backend (`vsagym.jax`)

The `vsagym.jax` module provides JAX-native SSP spaces and wrappers that are:
- **JIT-compiled** via `jax.jit`
- **Vectorized** via `jax.vmap`
- **Compatible** with gymnax and xland-minigrid functional environments

> **Tip:** Enable 64-bit precision for better numerical accuracy:
> ```python
> jax.config.update("jax_enable_x64", True)
> ```

### JAX SSP spaces

```python
import jax
import jax.numpy as jnp
from vsagym.jax import JaxHexagonalSSPSpace, JaxRandomSSPSpace

space = JaxHexagonalSSPSpace(domain_dim=2, ssp_dim=256)

x = jnp.array([[0.5, 0.3], [0.1, 0.9]])
phi = space.encode(x)       # (2, 256)  — JIT-compiled
x_hat = space.decode(phi)   # (2, 2)

# Binding & inversion
a = space.encode(jnp.array([[0.2, 0.8]]))
b = space.encode(jnp.array([[0.6, 0.4]]))
bound     = space.bind(a, b)
recovered = space.bind(bound, space.invert(b))
```

### Pure functions (use inside your own JIT blocks)

```python
from vsagym.jax import ssp_encode, ssp_bind, ssp_invert, ssp_normalize
import jax

# Pre-compute the space parameters (numpy)
from vsagym.spaces import HexagonalSSPSpace
np_space = HexagonalSSPSpace(2, ssp_dim=256)
pm = jnp.array(np_space.phase_matrix)
ls = jnp.array(np_space.length_scale.flatten())

@jax.jit
def my_encoding(x):
    phi = ssp_encode(pm, ls, x)
    return ssp_normalize(phi)

# vmap over a batch of observations
encode_batch = jax.vmap(lambda x: ssp_encode(pm, ls, x[None])[0])
phi_batch = encode_batch(jnp.ones((100, 2)))   # (100, 256)
```

### gymnax wrapper

```python
import jax
import gymnax
from vsagym.jax import GymnaxSSPObsWrapper

env, env_params = gymnax.make("CartPole-v1")
wrapper = GymnaxSSPObsWrapper(env, env_params, shape_out=256)

# JIT-compile reset and step
reset_fn = jax.jit(wrapper.reset)
step_fn  = jax.jit(wrapper.step)

key = jax.random.PRNGKey(0)
obs, state = reset_fn(key)          # obs: (256,)
obs, state, reward, done, info = step_fn(
    jax.random.PRNGKey(1), state, jnp.array(0)
)
```

**Vectorized rollouts** (vmap over seeds):

```python
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def rollout(wrapper, key, n_steps=200):
    def step(carry, _):
        obs, state, key = carry
        key, k1, k2 = jax.random.split(key, 3)
        action = jnp.array(0)
        obs, state, reward, done, info = wrapper.step(k1, state, action)
        return (obs, state, key), reward
    obs, state = wrapper.reset(key)
    _, rewards = jax.lax.scan(step, (obs, state, key), None, length=n_steps)
    return rewards

# Vectorize over 64 parallel environments
keys = jax.random.split(jax.random.PRNGKey(0), 64)
all_rewards = jax.vmap(partial(rollout, wrapper))(keys)
```

### xland-minigrid wrapper

```python
import jax
import xminigrid
from vsagym.jax import XLandSSPPoseWrapper

env, env_params = xminigrid.make("MiniGrid-Empty-5x5")
wrapper = XLandSSPPoseWrapper(env, env_params, shape_out=256)

key = jax.random.PRNGKey(0)
timestep = jax.jit(wrapper.reset)(env_params, key)
print(timestep.observation.shape)   # (256,)

timestep = jax.jit(wrapper.step)(
    env_params, timestep, jnp.array(0)
)
```

For view + pose encoding:

```python
from vsagym.jax import XLandSSPViewWrapper

wrapper = XLandSSPViewWrapper(
    env, env_params,
    shape_out=256,
    view_height=5, view_width=5,
    pose_weight=1.0,
)
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

To also run JAX and gymnax tests:

```bash
pip install -e ".[dev,gymnax]"
pytest tests/ -v
```

---

## Project Structure

```
vsagym/
├── spaces/
│   ├── ssp.py              # SSP class (HRR operators: +, *, ~, **)
│   ├── sspspace.py         # SSPSpace, RandomSSPSpace, HexagonalSSPSpace
│   ├── spspace.py          # SPSpace (discrete symbols)
│   ├── ssp_box.py          # SSPBox (continuous gymnasium space)
│   ├── ssp_discrete.py     # SSPDiscrete (discrete gymnasium space)
│   ├── ssp_sequence.py     # SSPSequence (sequence gymnasium space)
│   └── ssp_dict.py         # SSPDict (multi-modal gymnasium space)
├── wrappers/
│   ├── ssp_obs_wrapper.py  # SSPObsWrapper
│   ├── ssp_env_wrapper.py  # SSPEnvWrapper
│   ├── ssp_goal_wrapper.py # SSPGoalEnvWrapper (GoalEnv)
│   └── minigrid_wrappers/
│       ├── minigrid_pose_wrapper.py
│       ├── minigrid_view_wrapper.py
│       └── minigrid_mission_wrapper.py
├── networks/
│   ├── ssp_feature_extractor.py   # SSPFeaturesExtractor (PyTorch)
│   ├── hex_feature_extractor.py   # SSPHexFeaturesExtractor
│   └── minigrid_feature_extractor.py
└── jax/
    ├── sspspace.py         # JaxSSPSpace + pure JIT functions
    ├── spspace.py          # JaxSPSpace
    ├── gymnax_wrappers.py  # GymnaxSSPObsWrapper, GymnaxSSPEnvWrapper
    └── xland_wrappers.py   # XLandSSPPoseWrapper, XLandSSPViewWrapper
```

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

---

## Citation

If you use `vsagym` in your research, please cite:

```bibtex
@phdthesis{dumont2025,
    title   = {Symbols, Dynamics, and Maps: A Neurosymbolic Approach to Spatial Cognition},
    author  = {Nicole Sandra-Yaffa Dumont},
    type    = {PhD Thesis},
    school  = {University of Waterloo},
    address = {Waterloo, ON},
    year    = {2025},
    url     = {https://hdl.handle.net/10012/21501}
}
```
