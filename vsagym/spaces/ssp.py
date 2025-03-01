import numpy as np


class SSP():
    """
    A Semantic Pointer, based on Holographic Reduced Representations.

    Operators are overloaded so that ``+`` and ``-`` are addition,
    ``*`` is circular convolution, and ``~`` is the two-sided inversion operator.
    """

    def __init__(self, data):
        self.v = np.atleast_2d(data)

    def __add__(self, other):
        return self._add(other, swap=False)

    def __radd__(self, other):
        return self._add(other, swap=True)

    def _add(self, other, swap=False):
        if isinstance(other, SSP):
            other = other.v
        return SSP(data=self.v + other)

    def __neg__(self):
        return SSP(data=-self.v)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        """
        Multiplication of two SemanticPointers is circular convolution.

        If multiplied by a scalar, we do normal multiplication.
        """
        return self._mul(other, swap=False)

    def __rmul__(self, other):
        """
        Multiplication of two SemanticPointers is circular convolution.

        If multiplied by a scalar, we do normal multiplication.
        """
        return self._mul(other, swap=True)

    def _mul(self, other, swap=False):
        if type(other) is np.ndarray:
            a, b = np.atleast_2d(self.v), np.atleast_2d(other)
            return SSP(data=np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(b, axis=1), axis=1).real)
        elif isinstance(other, SSP):
            a, b = np.atleast_2d(self.v), np.atleast_2d(other.v)
            return SSP(data=np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(b, axis=1), axis=1).real)
        else:
            return SSP(data=self.v * other)

    def __pow__(self, other):
        v = np.atleast_2d(self.v)
        return SSP(data=np.fft.ifft(np.fft.fft(v ** other, axis=1), axis=1))

    def __invert__(self):
        v = np.atleast_2d(self.v)
        return SSP(data=v[:, -np.arange(self.v.shape[1])])


