from gymnasium.spaces import Space
from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union
from typing import Any, Iterable, Mapping
import warnings

from .sspspace import *
from .spspace import *

from numpy.typing import NDArray
import numpy as np
from .ssp_box import SSPBox
from .ssp_discrete import SSPDiscrete
from .ssp_box import _short_repr, is_float_integer

class SSPSequence(Space[np.ndarray]):
    r"""SSP encodings of fixed-finite-length sequences.

   This space represents the set of tuples of the form :math:`(a_0, \dots, a_n)` where the :math:`a_i` belong
   to some space that is specified during initialization and the integer :math:`n` is not fixed

   Example:
       >>> observation_space = SSPSequence(SSPBox(0, 1), seed=2)
   """

    def __init__(
        self,
        space: Space[Any],
        length: int,
        seed: Optional[Union[int, np.random.Generator]] = None, #: int | np.random.Generator | None
        ssp_order_space: Optional[SPSpace] = None
    ):
        r"""Constructor of :class:`SSPSequence`.
        """
        assert (isinstance( #todo: allow SSPDict with encodings defined?
            space, SSPBox
        ) or isinstance(
            space, SSPDiscrete
        )), f"Expects the feature space to be instance of a ssp gym Space, actual type: {type(space)}"
        self.feature_space = space
        self.shape_out = space.shape_out
        self.shape_in = (space.shape_in[0]*length,)
        self.dtype = space.dtype
        self.length = length
        
        if ssp_order_space is None:
            ssp_order_space = RandomSSPSpace(1, ssp_dim=space.ssp_space.ssp_dim,
                 domain_bounds=np.array([[0,self.length]]), length_scale=1, rng=seed)
             
        self.ssp_order_space = ssp_order_space
        self.order_ssps = self.ssp_order_space.encode(np.arange(self.length).reshape(-1,1) )
        self.inverse_order_ssps = self.ssp_order_space.invert(self.order_ssps)
        
        super().__init__((self.shape_out,), self.dtype, seed)
        
    @property
    def shape(self) -> Tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self.shape_out

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def seed(self, seed = None):
        """Seed the PRNG of this space and the feature space."""
        seeds = super().seed(seed)
        seeds += self.feature_space.seed(seed)
        return seeds

    def encode(self, x):
        enc_x = np.atleast_2d(x)
        S = np.zeros((enc_x.shape[0], self.shape_out))
        enc_x = enc_x.reshape(-1,self.length,self.feature_space.shape_in[0])
        for j in range(self.length):
            S += self.ssp_order_space.bind(self.order_ssps[j,:], 
                                       self.feature_space.encode(enc_x[:,j,:]))
        return S
   
    def decode(self,ssp):
        decoded_traj = np.zeros((self.length,self.feature_space.shape_in[0]))
        queries = self.ssp_order_space.bind(self.inverse_order_ssps, ssp)
        decoded_traj = self.feature_space.decode(queries)
        return decoded_traj #.reshape(-1)

    def sample(self): # TODO: add masks
        """Generates a single random sample from this space.
        """
        return self.samples(1)
        # return self.encode(self.feature_space.samples(self.length, return_x=True))
        # return np.sum(self.ssp_order_space.bind(self.feature_space.samples(self.length),
        #                                  self.order_ssps), axis=0).reshape(-1)
        
    def samples(self, n_samples):
        r"""Generates many random samples inside this space.
        """
        _, sample = self.feature_space.samples(self.length*n_samples, return_x=True)
        return self.encode(sample.reshape(n_samples,-1))

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
        return f"SSPSequence({self.feature_space}, {self.length})"

    def __eq__(self, other) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype or ssp_space equivalence."""
        return (
            isinstance(other, SSPSequence)
            and self.feature_space == other.feature_space
        )


