# vsagym: Gymnasium Environment Wrappers Using Vector Symbolic Architecture (VSA)

`vsagym` is a Python package that provides Gymnasium environment wrappers and tools for working with Vector Symbolic Architecture (VSA) and Spatial Semantic Pointer (SSP) representations in reinforcement learning (RL) tasks.



## Installation

To install `vsagym`, run the following command:

```bash
pip install -e .
```

A PyPI release is planned for the future.

### Dependencies
- `numpy`
- `scipy`
- `gymnasium`
- `torch`
- `stable_baselines3`

## TODO
- [x] Add tests for SSPObsWrapper and SSPEnvWrapper with SSPDict spaces
- [x] Add tests for the MiniGrid wrappers
- [x] Edit the base feature extractor network to be consistent with new class names
- [x] Make base feature extractor example in notebook
- [ ] Support mission + pose (no view) in SSP minigrid wrapper
- [x] Edit the MiniGrid feature extractor network to be consistent with new class names
- [x] Make MiniGrid usage example(s) in notebook (combine with the feature extractor too) using stablebaselines
- [ ] Option for having the phase matrix (and/or length scale) as the output of a network instead of fixed parameters
- [ ] Improved mission statement parsing
- [ ] Integration with other RL frameworks (e.g., gymjax)


## Background: SSPs and VSAs in Reinforcement Learning

Vector Symbolic Architecture (VSA) is a framework for representing structured data in high-dimensional spaces. Spatial Semantic Pointers (SSPs) are a specific type of VSA representation designed to encode continuous spatial information efficiently.

A Spatial Semantic Pointer (SSP) represents a value $ \mathbf{x} \in \mathbb{R}^n $ in the Holographic Reduced Representation (HRR) VSA as:

$$ \phi(\mathbf{x}) = W^{-1} e^{ i A  \mathbf{x}/ \ell }  $$

where:
- $W^{-1}$ is the inverse Discrete Fourier Transform (DFT) matrix.
- $A \in \mathbb{R}^{d \times n}$ is the **phase matrix**.
- $\ell \in \mathbb{R}^{n}$ is the **length scale**.
- The exponential function is applied element-wise.

Both $A$ and $\ell$ are free parameters. When $A$ is randomly initialized, the representation behaves similarly to a Random Fourier Feature.


SSPs and VSAs are useful in reinforcement learning because they:
- Provide a compact, compositional representation of continuous and discrete data.
- Enable symbolic-ish manipulations of state representations.
- Allow for structured embeddings that retain similarity relationships in state space.

## Features

### Spaces: Gymnasium Spaces
`vsagym` defines custom Gymnasium spaces for encoding observations and actions in an SSP/VSA format.

- **SSPBox:** Encodes continuous data from a `gym.spaces.Box` space using SSPs. If not provided, the underlying mapping (an `SSPSpace` object) is automatically generated.
- **SSPDiscrete:** Encodes discrete data from a `gym.spaces.Discrete` space using SSPs. The underlying mapping (an `SPSpace` object) is automatically generated. (Named `SSPDiscrete` instead of `SPDiscrete` for consistency.)
- **SSPSequences:** Encodes sequences of continuous or discrete data using SSP representations. Requires an `SSPBox` or `SSPDiscrete` space.
- **SSPDict:** A flexible space for defining SSP encodings over multiple data types. Enables binding and bundling different data types into a single compressed vector encoding (see examples in the provided notebooks).

### Wrappers
`vsagym` includes a set of wrappers to transform observations and/or actions into SSP/VSA embeddings. 

- **SSPEnvWrapper:** Automatically generates SSP representations for standard Gym observation and/or action spaces.
- **SSPObsWrapper:** Automatically generates SSP representations for standard Gym observation spaces.

### SSP Feature Extractors

The package provides `SSPFeaturesExtractor`, a PyTorch module (subclass of `BaseFeaturesExtractor` from `stable_baselines3`). This module integrates with `stable_baselines3` and other RL frameworks (e.g., `rlzoo`), allowing:
- Trainable SSP parameters (e.g., length scale and transformation matrix `A`).
- Customizable feature extraction for RL models using SSPs.

## MiniGrid Support

`vsagym` includes specific wrappers and feature extractors for the MiniGrid environment.

### MiniGrid Wrappers

- **SSPMiniGridPoseWrapper:** Encodes the agent’s position $(x, y, \theta) $ using SSPs:
  
  $$\phi_{\text{pose}} = \phi \left ( [x,y,\theta] \right ) $$
  
  where $x, y$ are the agent’s global coordinates and $\theta $ represents orientation (0-3 discrete values).

- **SSPMiniGridViewWrapper:** Encodes both the agent’s field of view and its pose using HRR algebra. Supports two encoding methods:
    
    1. **All-bound encoding (`obj_encoding='allbound'`):**
       
       $$\Phi_{\text{view}} = \phi([x,y,\theta]) + \mathtt{HAS} \circledast \sum_{\text{objects carried}} \mathtt{ITEM}_i \circledast \mathtt{COLOUR}_i \circledast \mathtt{STATE}_i$$

       $$+ \sum_{\text{objects in view}} \Delta\phi_i \circledast \mathtt{ITEM}_i \circledast \mathtt{COLOUR}_i \circledast \mathtt{STATE}_i$$
       
       where `ITEM`, `COLOUR`, and `STATE` vectors encode object attributes.
    
    3. **Slot-filler encoding (`obj_encoding='slotfiller'`):**
       
       $$\Phi_{\text{slot-filler}} = \phi([x,y,\theta]) + \mathtt{HAS} \circledast \sum_{\text{objects carried}} \left( \mathtt{ITEM} \circledast \mathtt{I}_i + \mathtt{COLOUR} \circledast \mathtt{C}_i + \mathtt{STATE} \circledast \mathtt{S}_i \right)$$
 
       $$+ \sum_{\text{objects in view}} \Delta\phi_i \circledast \left( \mathtt{ITEM} \circledast \mathtt{I}_i + \mathtt{COLOUR} \circledast \mathtt{C}_i + \mathtt{STATE} \circledast \mathtt{S}_i \right) $$
       
       In this encoding, object similarities are preserved more effectively.

- **Local vs. Global Views:**
  - `view_type='local'`: Uses relative object positions $\Delta\phi_i $.
  - `view_type='global'`: Uses absolute object positions $\phi_i $.

- **SSPMiniGridMissionWrapper:** Adds an SSP representation of the mission statement to the state encoding. Missions are parsed using regex to extract command structures (`GO_TO`, `PICK_UP`, etc.) and bind them to object properties.

- **SSPMiniGridWrapper:** A general-purpose interface for selecting combinations of pose, view, and mission encodings. Options:
  - `encode_pose=True/False`
  - `encode_view=True/False`
  - `encode_mission=True/False` (not supported if `encode_view=False`)

### MiniGrid Feature Extractors and Networks

The package also provides specialized feature extractors and neural network components tailored for use with MiniGrid environments and SSP representations. 

## Usage Examples

Detailed examples and notebooks demonstrating the use of `vsagym` are available in the repository.


## Contributing
Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License


## Citation

If you use `vsagym` in your research, please cite the following PhD thesis:

```
@phdthesis{dumont2025,
  author  = {Nicole Sandra-Yaffa Dumont},
  title   = {Symbols, Dynamics, and Maps: A Neurosymbolic Approach to Spatial Cognition},
  school  = {Univeristy of Waterloo},
  year    = {2025}
}
```



