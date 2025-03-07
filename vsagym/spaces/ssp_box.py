import gymnasium as gym
from gymnasium.spaces import Space
from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union
from typing import Any, Iterable, Mapping
from .sspspace import *
from numpy.typing import NDArray
import numpy as np
import warnings

class SSPBox(Space[np.ndarray]):
    r"""SSP representation of a (possibly unbounded) box in :math:`\mathbb{R}^n`.
    """

    def __init__(
        self,
        low: Union[SupportsFloat, np.ndarray],
        high: Union[SupportsFloat, np.ndarray],
        shape_in: Optional[Sequence[int]] = None,
        shape_out: Optional[int] = None,
        dtype: Type = np.float32,
        seed: Optional[Union[int, np.random.Generator]] = None,
        ssp_space: Optional[SSPSpace] = None,
        decoder_method: Optional[str] = 'direct-optim',
        length_scale: Optional[Union[float, list, np.ndarray]] = None,
        **kwargs
    ):
        r"""Constructor of :class:`SSPBox`, an SSP space for embedding data from a Box space

        The low and high are parameters defining the Box data. The argument ``low`` specifies the lower bound of each dimension and ``high`` specifies the upper bounds.
        I.e., the underlying Box space is the product of the intervals :math:`[\text{low}[i], \text{high}[i]]`.

        If ``low`` (or ``high``) is a scalar, the lower bound (or upper bound, respectively) will be assumed to be
        this value across all dimensions.

        Args:
            low (Union[SupportsFloat, np.ndarray]): Lower bounds of the intervals defining the data space
            high (Union[SupportsFloat, np.ndarray]): Upper bounds of the intervals defining the data space
            shape_in (Optional[Sequence[int]]): The shape of the underlying data -- i.e., shape of input to the encode method. The shape is inferred from the shape of `low` or `high` `np.ndarray`s with
                `low` and `high` scalars defaulting to a shape of (1,)
            shape_out: The size of the SSP vectors -- i.e., size of output from the encode method. This is optional as other kwargs can be used instead to set the shape under some methods
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.
            ssp_space: Optionally, you can pass a SSPSpace object to use for the encoding or pass a string to define the type of SPSpace. If ssp_space is "rand" then a RandomSSPSpace is created. If ssp_space is "hex" or None, a HexagonalSSPSpace is created (default). Note that HexagonalSSPSpace only allow embeddings of particular sizes (see n_scales and n_rotates under kwargs for more details). If needed, the shape_out will be changed to the closest admissible size.
            decoder_method: A string defining the method used to decode SSPs. Options include 'direct-optim' (default), 'from-set','network', and 'network-optim'. For most applications, decoding will not be required. The methods 'direct-optim' and 'from-set' are the most accurate and robust but scale poorly with shape_in and can be slow. The methods 'network' and 'network-optim' are fast but can be brittle and require tensorflow. 
            length_scale: The length_scale of the SSP encoding. If a float, the same value will be used for all data dimensions. If not provided, a simple heuristic will be used to set the length_scale
        """
        assert (
            dtype is not None
        ), "Box dtype must be explicitly provided, cannot be None."
        
        self.dtype = np.dtype(dtype)
        self.shape_out = shape_out

        # determine shape if it isn't provided directly
        if shape_in is not None:
            if type(shape_in) is int:
                shape_in = (shape_in,)
            else:
                shape_in = shape_in
        elif isinstance(low, np.ndarray):
            shape_in = low.shape
        elif isinstance(high, np.ndarray):
            shape_in = high.shape
        elif is_float_integer(low) and is_float_integer(high):
            shape_in = (1,)
        else:
            raise ValueError(
                f"Box shape_in is inferred from low and high, expect their types to be np.ndarray, an integer or a float, actual type low: {type(low)}, high: {type(high)}"
            )
        self.shape_in = shape_in
        self._shape = shape_in
        
        # Capture the boundedness information before replacing np.inf with get_inf
        _low = np.full(shape_in, low, dtype=float) if is_float_integer(low) else low
        self.bounded_below = -np.inf < _low
        _high = np.full(shape_in, high, dtype=float) if is_float_integer(high) else high
        self.bounded_above = np.inf > _high

        # Capture the boundedness information before replacing np.inf with get_inf
        _low = np.full(shape_in, low, dtype=float) if is_float_integer(low) else low
        self.bounded_below = -np.inf < _low
        _high = np.full(shape_in, high, dtype=float) if is_float_integer(high) else high
        self.bounded_above = np.inf > _high

        low = _broadcast(low, dtype, shape_in, inf_sign="-")  # type: ignore
        high = _broadcast(high, dtype, shape_in, inf_sign="+")  # type: ignore

        self.low = low.astype(self.dtype)
        self.high = high.astype(self.dtype)
        self.bounds = np.vstack([self.low, self.high]).T

        # recheck shape for case where shape and (low or high) are provided
        if self.low.shape != shape_in:
            raise ValueError(
                f"Box low.shape doesn't match provided shape, low.shape={self.low.shape}, shape={self.shape}"
            )
        if self.high.shape != shape_in:
            raise ValueError(
                f"Box high.shape doesn't match provided shape, high.shape={self.high.shape}, shape={self.shape}"
            )

        self.shape_out = shape_out

        low_precision = get_precision(low.dtype)
        high_precision = get_precision(high.dtype)
        dtype_precision = get_precision(self.dtype)
        if min(low_precision, high_precision) > dtype_precision:  # type: ignore
            warnings.warn(f"Box bound precision lowered by casting to {self.dtype}")
        self.low = low.astype(self.dtype)
        self.high = high.astype(self.dtype)
        self.bounds = np.vstack([self.low, self.high]).T

        self.low_repr = _short_repr(self.low)
        self.high_repr = _short_repr(self.high)
        
        # From Kathyrn
        if length_scale is None:
            length_scale = np.clip( ( np.abs( high - low ) ),a_min = 1e-4, a_max = 1e4) / 10.
        
        
        if (ssp_space is None) or (ssp_space=='hex'):
            self.ssp_space = HexagonalSSPSpace(
                                self.shape_in[0],
                                ssp_dim=self.shape_out, 
                                domain_bounds=self.bounds, seed=seed,
                                length_scale=length_scale,
                                **kwargs)
        elif ssp_space=='rand':
            self.ssp_space = RandomSSPSpace(
                                self.shape_in[0],
                                ssp_dim=self.shape_out, 
                                domain_bounds=self.bounds, seed=seed,
                                length_scale=length_scale,
                                **kwargs)
        else:
            assert issubclass(type(ssp_space), SSPSpace)
            assert ssp_space.domain_dim == shape_in[0]
            self.ssp_space = ssp_space
        self.shape_out = self.ssp_space.ssp_dim
        
    
        super().__init__((self.shape_out, ), self.dtype, seed)
        
        self.init_samples = None
        if (decoder_method == 'network') | (decoder_method == 'network-optim'):
            self.ssp_space.train_decoder_net();
        elif decoder_method == "from-set":
            self.init_samples = self.samples(np.min([10000,20**self.shape_in[0]]), return_x=True)
        self.decoder_method = decoder_method

    @property
    def shape(self) -> Tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self._shape

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True


    def is_bounded(self, manner: str = "both") -> bool:
        """Checks whether the box is bounded in some sense.

        Args:
            manner (str): One of ``"both"``, ``"below"``, ``"above"``.

        Returns:
            If the space is bounded

        Raises:
            ValueError: If `manner` is neither ``"both"`` nor ``"below"`` or ``"above"``
        """
        below = bool(np.all(self.bounded_below))
        above = bool(np.all(self.bounded_above))
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError(
                f"manner is not in {{'below', 'above', 'both'}}, actual value: {manner}"
            )

    def encode(self, x):
        return self.ssp_space.encode(np.atleast_2d(x)).astype(self.dtype)
    
    def decode(self, ssp):
        ssp = np.atleast_2d(ssp)
        return self.ssp_space.decode(ssp,method=self.decoder_method,samples=self.init_samples).astype(self.dtype)

    def sample(self, mask: None = None, return_x=False):
        r"""Generates a single random sample inside the SSPBox.

        In creating a sample of the box, each coordinate is sampled (independently) from a distribution
        that is chosen according to the form of the interval:

        * :math:`[a, b]` : uniform distribution
        * :math:`[a, \infty)` : shifted exponential distribution
        * :math:`(-\infty, b]` : shifted negative exponential distribution
        * :math:`(-\infty, \infty)` : normal distribution

        Args:
            mask: A mask for sampling values from the Box space, currently unsupported.

        Returns:
            A sampled value from the Box
        """
        if mask is not None:
            raise gym.error.Error(
                f"Box.sample cannot be provided a mask, actual value: {mask}"
            )

        high = self.high if self.dtype.kind == "f" else self.high.astype("int64") + 1
        sample = np.empty(self.shape_in[0])

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = self.np_random.normal(size=unbounded[unbounded].shape)

        sample[low_bounded] = (
            self.np_random.exponential(size=low_bounded[low_bounded].shape)
            + self.low[low_bounded]
        )

        sample[upp_bounded] = (
            -self.np_random.exponential(size=upp_bounded[upp_bounded].shape)
            + self.high[upp_bounded]
        )

        sample[bounded] = self.np_random.uniform(
            low=self.low[bounded], high=high[bounded], size=bounded[bounded].shape
        )
        if self.dtype.kind == "i":
            sample = np.floor(sample)

        ssp_sample = self.encode(sample).reshape(-1)

        if return_x:
            return ssp_sample.astype(self.dtype), sample.astype(self.dtype)
        return ssp_sample.astype(self.dtype)
    
    def samples(self, n_samples, mask: None = None, return_x = False):
        r"""Generates many random samples inside the SSPBox.
        """
        if mask is not None:
            raise gym.error.Error(
                f"Box.sample cannot be provided a mask, actual value: {mask}"
            )

        high = self.high if self.dtype.kind == "f" else self.high.astype("int64") + 1
        sample = np.empty((n_samples,self.shape_in[0]))

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[:,unbounded] = self.np_random.normal(size=(n_samples,) + unbounded[unbounded].shape)

        sample[:,low_bounded] = (
            self.np_random.exponential(size=(n_samples,) + low_bounded[low_bounded].shape)
            + self.low[low_bounded]
        )

        sample[:,upp_bounded] = (
            -self.np_random.exponential(size=(n_samples,) + upp_bounded[upp_bounded].shape)
            + self.high[upp_bounded]
        )

        sample[:,bounded] = self.np_random.uniform(
            low=self.low[bounded], high=high[bounded], size=(n_samples,) + bounded[bounded].shape
        )
        if self.dtype.kind == "i":
            sample = np.floor(sample)

        ssp_sample = self.encode(sample)

        if return_x:
            return ssp_sample.astype(self.dtype), sample.astype(self.dtype)
        return ssp_sample.astype(self.dtype)

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not isinstance(x, np.ndarray):
            warnings.warn("Casting input x to numpy array.")
            try:
                x = np.asarray(x, dtype=self.dtype)
            except (ValueError, TypeError):
                return False

        return bool(
            np.can_cast(x.dtype, self.dtype)
            and x.shape == self.shape_out
        )

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n: Sequence[Union[float, int]]) -> List[np.ndarray]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self) -> str:
        """A string representation of this space.

        The representation will include bounds, shape and dtype.
        If a bound is uniform, only the corresponding scalar will be given to avoid redundant and ugly strings.

        Returns:
            A representation of the space
        """
        return f"SSPBox({self.low_repr}, {self.high_repr}, {self.shape_in}, {self.shape_out}, {self.dtype})"

    def __eq__(self, other) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype or ssp_space equivalence."""
        return (
            isinstance(other, SSPBox)
            and (self.shape_in == other.shape_in)
            and (self.shape_out == other.shape_out)
            and np.allclose(self.low, other.low)
            and np.allclose(self.high, other.high)
        )

    def __setstate__(self, state: Dict):
        """Sets the state of the box for unpickling a box with legacy support."""
        super().__setstate__(state)

        # legacy support through re-adding "low_repr" and "high_repr" if missing from pickled state
        if not hasattr(self, "low_repr"):
            self.low_repr = _short_repr(self.low)

        if not hasattr(self, "high_repr"):
            self.high_repr = _short_repr(self.high)
            
def get_precision(dtype) -> SupportsFloat:
    """Get precision of a data type."""
    if np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).precision
    else:
        return np.inf          


def _short_repr(arr: NDArray[Any]) -> str:
    """Create a shortened string representation of a numpy array.

    If arr is a multiple of the all-ones vector, return a string representation of the multiplier.
    Otherwise, return a string representation of the entire array.

    Args:
        arr: The array to represent

    Returns:
        A short representation of the array
    """
    if arr.size != 0 and np.min(arr) == np.max(arr):
        return str(np.min(arr))
    return str(arr)


def is_float_integer(var: Any) -> bool:
    """Checks if a variable is an integer or float."""
    return np.issubdtype(type(var), np.integer) or np.issubdtype(type(var), np.floating)


def _broadcast(
    value: Union[SupportsFloat, np.ndarray],
    dtype,
    shape: Tuple[int, ...],
    inf_sign: str,
) -> np.ndarray:
    """Handle infinite bounds and broadcast at the same time if needed."""
    if is_float_integer(value):
        value = get_inf(dtype, inf_sign) if np.isinf(value) else value  # type: ignore
        value = np.full(shape, value, dtype=dtype)
    else:
        assert isinstance(value, np.ndarray)
        if np.any(np.isinf(value)):
            # create new array with dtype, but maintain old one to preserve np.inf
            temp = value.astype(dtype)
            temp[np.isinf(value)] = get_inf(dtype, inf_sign)
            value = temp
    return value

