import numpy as np
from typing import Optional, Union, List
# from functools import singledispatchmethod
from .sspspace import _get_rng

class SPSpace:
    r"""  Class for Semantic Pointer (SP) representation mapping

    Parameters
    ----------
        domain_size : int
            The number of discrete symbols that will be encoded in this space.
            
        dim : int
            The dimensionality of the SPs, should be >= domain_size.
            
        rng : int
            The random state for generating the SPs.
            
    Attributes
    ----------
        domain_size, dim : int
        
        vectors : np.ndarray
           A (domain_size x dim) array of all SPs
           
        inverse_vectors : Node
            Inverse (under binding) SPs
            
    Examples
    --------
       from sspslam import SPSpace
       sp_space = SPSpace(5, 100)

    """

    def __init__(self, domain_size: int,
                 dim: int,
                 vectors: Optional[np.ndarray] = None,
                 rng: Optional[Union[int, np.random.Generator]] = None,
                 names: Optional[List[str]] = None,
                 **kwargs):
        self.domain_size = int(domain_size)
        self.dim = int(dim)

        self.rng = _get_rng(rng)

        if self.domain_size == 1:  # only one is special case, vectors only contains identity
            self.vectors = np.zeros((self.domain_size, self.dim))
            self.vectors[:, 0] = 1
        elif vectors is not None:
            assert vectors.shape[0] == self.domain_size
            assert vectors.shape[1] == self.dim
            self.vectors = vectors
        else:
            _vectors = self.rng.normal(size=(self.domain_size, self.dim))
            _vectors /= np.linalg.norm(_vectors, axis=0, keepdims=True)
            self.vectors = self.make_unitary(_vectors)

            for j in range(self.domain_size):
                q = self.vectors[j, :] / np.linalg.norm(self.vectors[j, :])
                for k in range(j + 1, self.domain_size):
                    self.vectors[k, :] = self.vectors[k, :] - (q.T @ self.vectors[k, :]) * q
                    self.vectors[k, :] = self.make_unitary(self.vectors[k, :])
        self.inverse_vectors = self.invert(self.vectors)

        if names is None:
            names = [f'SP{i}' for i in range(self.domain_size)]
        self.names = names

        self.idx_to_name = dict(zip(np.arange(self.domain_size), names))
        self.name_to_vector = dict(zip(names, list(self.vectors)))
        self.name_to_vector['I'] = self.identity()
        self.name_to_vector['NULL'] = np.zeros(self.dim)
        self.name_to_inv_vector = dict(zip(names, list(self.inverse_vectors)))
        self.name_to_inv_vector['I'] = self.identity()
        self.name_to_inv_vector['NULL'] = np.zeros(self.dim)
        self.name_to_idx = dict(zip(names, np.arange(domain_size)))

        # @singledispatchmethod
    def encode(self, i: Union[list, np.ndarray]) -> np.ndarray:
        """
        Maps indexes or names to SPs

        Parameters
        ----------
        i : np.array of int or str
            An array of ints, each in [0, domain_size) or an array of strings

        Returns
        -------
        np.array
            Semantic Pointers.

        """
        if i.dtype == str:
            return np.array([self.name_to_vector[_i] for _i in list(i)])
        else:
            i = np.array(i)
            return self.vectors[i.reshape(-1).astype(int)]

    def decode(self,
               v: np.ndarray,
               **kwargs):
        """
        Maps SP vectors to indexes
        
        Parameters
        ----------
        v : np.array
            A (n_samples x ssp_dim) vector

        Returns
        -------
        np.array
            A n_samples length vector of indexes 

        """
        sims = self.vectors @ v.T
        return np.argmax(sims, axis=0)

    def clean_up(self, v, **kwargs):
        """
        Maps dim-D vector to SP
        
        Parameters
        ----------
        v : np.array
            A (n_samples x ssp_dim) vector

        Returns
        -------
        np.array
            A (n_samples x ssp_dim) vector, each row a Semantic Pointer.

        """
        sims = self.vectors @ v.T
        return self.vectors[np.argmax(sims, axis=0)]

    def normalize(self, v):
        """
        Normalizes input
        """
        return v / np.sqrt(np.sum(v ** 2))

    def make_unitary(self, v):
        """
        Makes input unitary (Fourier components have magnitude of 1)
        """
        fv = np.fft.fft(v, axis=-1)
        fv = fv / np.sqrt(fv.real ** 2 + fv.imag ** 2)
        return np.fft.ifft(fv, axis=-1).real

    def identity(self):
        """
        Returns
        -------
        np.array
            dim-D identity vector under binding

        """
        s = np.zeros((1, self.dim))
        s[:, 0] = 1
        return s

    def bind(self, *arrays):
        # Binds together input with circular convolution
        arrays = [np.atleast_2d(arr) for arr in arrays]
        fft_result = np.fft.fft(arrays[0], axis=-1)
        for arr in arrays[1:]:
            fft_result = fft_result * np.fft.fft(arr, axis=-1)  # loop for broadcasting
        return np.fft.ifft(fft_result, axis=-1).real

    def invert(self, a):
        """
        Inverts input under binding
        """
        a = np.atleast_2d(a)
        return a[:, -np.arange(self.dim)]

    def get_binding_matrix(self, v):
        """
        Maps input vector to a matrix that, when multiplied with another vecotr, will bind vectors
        """
        C = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                C[i, j] = v[:, (i - j) % self.dim]
        return C

