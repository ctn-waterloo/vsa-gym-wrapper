import numpy as np
from scipy.stats import qmc
from scipy.stats import special_ortho_group
from scipy.optimize import minimize
from scipy.special import gammainc
from typing import Optional, Union

import warnings

def _get_rng(rng: Optional[Union[int, np.random.Generator]] = None):
    if rng is None:
        rng = np.random.default_rng()  # New generator for each call
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    return rng

class SSPSpace:
    r"""  Class for Spatial Semantic Pointer (SSP) representation mapping
    
    Parameters
    ----------
        domain_size : int
            The dimensionality of the domain being represented by SSPs.
        ssp_dim : int
            The dimensionality of the spatial semantic pointer vector.
            
        phase_matrix :  np.ndarray
            A ssp_dim x domain_dim ndarray representing the frequency 
            components of the SSP representation.

        domain_bounds : np.ndarray
            A domain_dim X 2 ndarray giving the lower and upper bounds of the 
            domain, used in decoding from an ssp to the point it represents.

        length_scale : float or np.ndarray
            Scales values before encoding.
            

    """

    def __init__(self, domain_dim: int,
                 ssp_dim: int,
                 phase_matrix: np.ndarray,
                 domain_bounds: Optional[np.ndarray] = None,
                 length_scale: Optional[Union[float, list, np.ndarray]] = 1,
                 rng: Optional[Union[int, np.random.Generator]] = None,
                 ):

        self.domain_dim = domain_dim
        self.ssp_dim = ssp_dim

        if (type(length_scale) is np.ndarray) or (type(length_scale) is list):
            length_scale = np.array(length_scale).reshape(self.domain_dim, 1)
        self.length_scale = np.array(length_scale) * np.ones((self.domain_dim, 1))

        self.rng = _get_rng(rng)

        if domain_bounds is not None:
            assert domain_bounds.shape[0] == domain_dim

        self.domain_bounds = domain_bounds
        self.decoder_model = None

        assert phase_matrix.shape[0] == ssp_dim
        assert phase_matrix.shape[1] == domain_dim
        self.phase_matrix = phase_matrix

    def update_lengthscale(self, scale):
        '''
        Changes the lengthscale being used in the encoding.
        '''
        if not isinstance(scale, np.ndarray) or scale.size == 1:
            self.length_scale = scale * np.ones((self.domain_dim,))
        else:
            assert scale.size == self.domain_dim
            self.length_scale = scale
        assert self.length_scale.size == self.domain_dim

    def encode(self, x):
        '''
        Transforms input data into an SSP representation.

        Parameters:
        -----------
        x : np.ndarray
            A (num_samples, domain_dim) array representing data to be encoded.

        Returns:
        --------
        data : np.ndarray
            A (num_samples, ssp_dim) array of the ssp representation of the data
            
        '''

        x = np.atleast_2d(x)
        ls_mat = np.atleast_2d(np.diag(1 / self.length_scale.flatten()))
        assert ls_mat.shape == (self.domain_dim, self.domain_dim), \
            f'Expected Len Scale mat with dimensions {(self.domain_dim, self.domain_dim)}, got {ls_mat.shape}'
        scaled_x = x @ ls_mat
        data = np.fft.ifft(np.exp(1.j * self.phase_matrix @ scaled_x.T), axis=0).real
        return data.T

    def encode_and_deriv(self, x):
        '''
        Returns the ssp representation of the data and the derivative of
        the encoding.

        Parameters:
        -----------
        x : np.ndarray
            A (num_samples, domain_dim) array representing data to be encoded.

        Returns:
        --------
        data : np.ndarray
            A (num_samples, ssp_dim) array of the ssp representation of the 
            data

        grad : np.ndarray
            A (num_samples, ssp_dim, domain_dim) array of the ssp representation of the data

        '''
        x = np.atleast_2d(x)
        ls_mat = np.atleast_2d(np.diag(1 / self.length_scale))
        assert ls_mat.shape == (self.domain_dim, self.domain_dim), \
            f'Expected Len Scale mat with dimensions {(self.domain_dim, self.domain_dim)}, got {ls_mat.shape}'
        scaled_x = x @ ls_mat
        data = np.fft.ifft(np.exp(1.j * self.phase_matrix @ scaled_x.T), axis=0).real
        ddata = np.fft.ifft(1.j * (self.phase_matrix @ ls_mat) @
                            np.exp(1.j * self.phase_matrix @ scaled_x.T), axis=0).real
        return data.T, ddata.T

    def encode_fourier(self, x):
        x = np.atleast_2d(x)
        ls_mat = np.atleast_2d(np.diag(1 / self.length_scale.flatten()))
        assert ls_mat.shape == (self.domain_dim, self.domain_dim), \
            f'Expected Len Scale mat with dimensions {(self.domain_dim, self.domain_dim)}, got {ls_mat.shape}'
        scaled_x = x @ ls_mat
        data = np.exp(1.j * self.phase_matrix @ scaled_x.T)
        return data.T

    def decode(self, ssp, method='from-set', sampling_method='grid',
               num_samples=300, samples=None):  # other args for specfic methods
        '''
        Transforms ssp representation back into domain representation.

        Parameters:
        -----------
        ssp : np.ndarray
            SSP representation of a data point.

        method : {'from-set', 'direct-optim'}
            The technique for decoding the ssp.  from-set samples the domain
            and finds the closest match under the dot product. direct-optim
            does an initial coarse sampling and then optimizes the decoded
            value starting from the initial best match in the coarse sampling.

        sampling_method : {'grid'|'length-scale'|'sobol'}
            Evenly distributes samples along the domain axes

        num_samples : int
            The number of samples along each axis.

        Returns:
        --------
        x : np.ndarray
            The decoded point
        '''
        if (method == 'direct-optim') | (method == 'from-set'):
            if samples is None:
                sample_ssps, sample_points = self.get_sample_pts_and_ssps(method=sampling_method,
                                                                          num_points_per_dim=num_samples)
            else:
                sample_ssps, sample_points = samples
                assert sample_ssps.shape[1] == ssp.shape[1], f'Expected {sample_ssps.shape} dim, got {ssp.shape}'

        unit_ssp = ssp / np.maximum(np.linalg.norm(ssp, axis=-1, keepdims=True), 1e-6)

        if method == 'from-set':
            sims = sample_ssps @ unit_ssp.T
            return sample_points[np.argmax(sims, axis=0), :]

        elif method == 'direct-optim':
            def min_func(x, target):
                x_ssp = self.encode(np.atleast_2d(x))
                return -np.inner(x_ssp, target).flatten()

            retvals = np.zeros((ssp.shape[0], self.domain_dim))
            for s_idx, u_ssp in enumerate(unit_ssp):
                x0 = self.decode(np.atleast_2d(u_ssp),
                                 method='from-set',
                                 sampling_method='length-scale',
                                 num_samples=num_samples, samples=samples)

                soln = minimize(min_func, x0.flatten(),
                                args=(np.atleast_2d(u_ssp),),
                                method='L-BFGS-B',
                                bounds=self.domain_bounds)
                retvals[s_idx, :] = soln.x
            return retvals

        else:
            raise NotImplementedError(f'Unrecognized decoding method: {method}')

    def clean_up(self, ssp, method='from-set', sampling_method='grid', num_samples=300):
        x = self.decode(ssp, method, sampling_method, num_samples)
        return self.encode(x)

    def get_sample_points(self, samples_per_dim=100, method='length-scale'):
        '''
        Identifies points in the domain of the SSP encoding that 
        will be used to determine optimal decoding.

        Parameters
        ----------

        method: {'grid'|'length-scale'|'sobol'}
            The way to select samples from the domain. 
            'grid' uniformly spaces samples_per_dim points on the domain
            'sobol' decodes using samples_per_dim**data_dim points generated 
                using a sobol sampling
            'length-scale' uses the selected lengthscale to determine the number
                of sample points generated per dimension.

        Returns
        -------

        sample_pts : np.ndarray 
            A (num_samples, domain_dim) array of candiate decoding points.
        '''

        if self.domain_bounds is None:
            bounds = np.vstack([-10 * np.ones(self.domain_dim), 10 * np.ones(self.domain_dim)]).T
        else:
            bounds = self.domain_bounds

        if method == 'grid':
            num_pts_per_dim = [samples_per_dim for _ in range(bounds.shape[0])]
        elif method == 'length-scale':
            num_pts_per_dim = [2 * int(np.ceil((b[1] - b[0]) / self.length_scale[b_idx])) for b_idx, b in
                               enumerate(bounds)]
        else:
            num_pts_per_dim = samples_per_dim

        if method == 'grid' or method == 'length-scale':
            xxs = np.meshgrid(*[np.linspace(bounds[i, 0],
                                            bounds[i, 1],
                                            num_pts_per_dim[i]
                                            ) for i in range(self.domain_dim)])
            retval = np.array([x.reshape(-1) for x in xxs]).T
            assert retval.shape[1] == self.domain_dim, f'Expected {self.domain_dim}d data, got {retval.shape[1]}d data'
            return retval

        elif method == 'sobol': #quasi-random
            num_points = np.prod(num_pts_per_dim)

            sampler = qmc.Sobol(d=self.domain_dim, seed=self.rng)
            lbounds = bounds[:, 0]
            ubounds = bounds[:, 1]
            u_sample_points = sampler.random(num_points)
            sample_points = qmc.scale(u_sample_points, lbounds, ubounds).T

        elif method == 'Rd': #different quasi-random
            num_points = np.prod(samples_per_dim)
            u_sample_points = _Rd_sampling(num_points, self.domain_dim)
            lbounds = bounds[:, 0]
            ubounds = bounds[:, 1]
            sample_points = qmc.scale(u_sample_points, lbounds, ubounds).T
        else:
            raise NotImplementedError(f'Sampling method {method} is not implemented')
        return sample_points.T

    def get_sample_ssps(self, num_points, **kwargs):
        sample_points = self.get_sample_points(num_points, **kwargs)
        sample_ssps = self.encode(sample_points)
        return sample_ssps

    def get_sample_pts_and_ssps(self, num_points_per_dim=100, method='grid'):
        sample_points = self.get_sample_points(
            method=method,
            samples_per_dim=num_points_per_dim
        )
        if method == 'grid':
            expected_points = int(num_points_per_dim ** (self.domain_dim))
            assert sample_points.shape[
                       0] == expected_points, f'Expected {expected_points} samples, got {sample_points.shape[0]}.'

        sample_ssps = self.encode(sample_points)

        if method == 'grid':
            assert sample_ssps.shape[0] == expected_points

        return sample_ssps, sample_points

    def normalize(self, ssp):
        return ssp / np.maximum(np.linalg.norm(ssp, axis=-1, keepdims=True), 1e-8)

    def make_unitary(self, ssp):
        fssp = np.fft.fft(ssp, axis=-1)
        fssp = fssp / np.maximum(np.sqrt(fssp.real ** 2 + fssp.imag ** 2), 1e-8)
        return np.fft.ifft(fssp, axis=-1).real

    def identity(self):
        s = np.zeros((1,self.ssp_dim))
        s[:,0] = 1
        return s

    def bind(self, *arrays):
        # Binds together input with circular convolution
        arrays = [np.atleast_2d(arr) for arr in arrays]
        fft_result = np.fft.fft(arrays[0], axis=-1)
        for arr in arrays[1:]:
            fft_result = fft_result * np.fft.fft(arr, axis=-1)  # loop for broadcasting
        return np.fft.ifft(fft_result, axis=-1).real
    def invert(self, a):
        a = np.atleast_2d(a)
        return a[:, -np.arange(self.ssp_dim)]

    def similarity_plot(self, ssp, n_grid=100, plot_type='heatmap', ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if self.domain_dim == 1:
            xs = np.linspace(self.domain_bounds[0, 0], self.domain_bounds[0, 1], n_grid)
            sims = ssp @ self.encode(np.atleast_2d(xs).T).T
            im = ax.plot(xs, sims.reshape(-1), **kwargs)
            ax.set_xlim(self.domain_bounds[0, 0], self.domain_bounds[0, 1])

        elif self.domain_dim == 2:
            xs = np.linspace(self.domain_bounds[0, 0], self.domain_bounds[0, 1], n_grid)
            ys = np.linspace(self.domain_bounds[1, 0], self.domain_bounds[1, 1], n_grid)
            X, Y = np.meshgrid(xs, ys)
            sims = ssp @ self.encode(np.vstack([X.reshape(-1), Y.reshape(-1)]).T).T
            if plot_type == 'heatmap':
                im = ax.pcolormesh(X, Y, sims.reshape(X.shape), **kwargs)
            elif plot_type == 'contour':
                im = ax.contour(X, Y, sims.reshape(X.shape), **kwargs)
            elif plot_type == 'contourf':
                im = ax.contourf(X, Y, sims.reshape(X.shape), **kwargs)
            ax.set_xlim(self.domain_bounds[0, 0], self.domain_bounds[0, 1])
            ax.set_ylim(self.domain_bounds[1, 0], self.domain_bounds[1, 1])

        else:
            raise NotImplementedError()
        return im


class RandomSSPSpace(SSPSpace):
    '''
    Creates an SSP space using randomly generated frequency components.

    Parameters:
    -----------
    scale_min : flaot
        The min phase matrix component
    scale_max : flaot
        The max phase matrix component
    sampler : 'unif' (default) or 'norm'
        Method for random sampling, uniform or normal

    Other params are described in SPSpace
    '''
    def __init__(self, domain_dim: int,
                 ssp_dim: int,
                 domain_bounds: Optional[np.ndarray] = None,
                 scale_min: Optional[float] = 0.25,
                 scale_max: Optional[float] = 2.0,
                 length_scale: Optional[Union[float, list, np.ndarray]] = 1,
                 sampler: Optional[str] = 'norm', # unif or 'nomr'
                 norm_scale: Optional[float] = None,
                 rng: Optional[Union[int, np.random.Generator]] = None,
                 **kwargs):
        self.rng = _get_rng(rng)
        if sampler == 'unif':
            n_samples = (ssp_dim - 1) // 2
            samples = self.rng.normal(size=(n_samples, domain_dim))
            ssq = np.sum(samples ** 2, axis=1)
            fr = scale_max * gammainc(domain_dim / 2, ssq / 2) ** (1 / domain_dim) / np.sqrt(ssq)
            frtiled = np.tile(fr.reshape(n_samples, 1), (1, domain_dim))
            phases = np.multiply(samples, frtiled)

        elif sampler == 'norm':
            if norm_scale is None:
                norm_scale = np.sqrt(np.pi / 2) * (
                            (scale_max - scale_min) / 2 + scale_min)  #expected absolute value is norm_scale*sqrt(2/pi)
            phases = self.rng.normal(loc=0., scale=norm_scale, size=((ssp_dim - 1) // 2, domain_dim))

        else:
            raise NotImplementedError()

        phase_matrix = conjsym(phases, ssp_dim % 2 == 0)

        super().__init__(domain_dim,
                         phase_matrix.shape[0],
                         phase_matrix=phase_matrix,
                         domain_bounds=domain_bounds,
                         length_scale=length_scale, rng=self.rng,
                         )

    def sample_wave_encoders(self, n_neurons):
        d = self.ssp_dim
        n = self.domain_dim
        A = self.phase_matrix
        N = (d - 2) // 2

        sample_pts = self.get_sample_points(n_neurons, method='sobol')[:n_neurons, :]
        n_per_pattern = int(np.floor(n_neurons / N))
        sorts = np.concatenate(
            [np.repeat(np.arange(0, N), n_per_pattern), self.rng.integers(0, N, size=n_neurons - N * n_per_pattern)])

        encoders = np.zeros((n_neurons, d))
        for i in range(n_neurons):
            res = np.zeros(d, dtype=complex)
            res[sorts[i] + 1] = np.exp(1.j * A[sorts[i] + 1] @ sample_pts[i, :])
            res[(N + 1):] = np.conjugate(np.flip(res[1:(N + 1)]))
            res[0] = 1
            if d % 2 == 0:
                res[d // 2] = 1
            encoders[i, :] = np.fft.ifft(res).real
        encoders = encoders / np.linalg.norm(encoders, axis=-1, keepdims=True)
        return encoders


class HexagonalSSPSpace(SSPSpace):
    '''
    Creates an SSP space using the Hexagonal Tiling developed by NS Dumont  (2020).
    The (domain_dim+1) vertices of a domain_dim simplex are generated.
    All vectors are scaled by n_scales different values to obtain n_scales * (domain_dim+1) vectors (each of length domain_dim)
    All vectors are rotated by n_rotates different rotation matrices
        to obtain n_rotates * n_scales * (domain_dim+1) vectors
    These vectors are used to construct the HexSSP phase matrix.
    After enforcing conjugate symmetry, the final ssp_dim is 2 * n_rotates * n_scales * (domain_dim+1) + 1

    Note that if both n_scales or n_rotates are positive, they will be used to construct the HexSSP phase matrix
    and so the ssp_dim input will be ignored.
    If instead, either n_scales or n_rotates is non-positive, the input ssp_dim will be used by solving for the
    n_scales=n_rotates such that  2 * n_rotates * n_scales * (domain_dim+1) + 1 is close to ssp_dim

    Parameters:
    -----------
    n_scales : int
        Number of scales to use
    n_rotates : int
        Number of rotations to use. Note that rotations are ignored when domain_dim=1,
         are generated uniformly when domain_dim=2, and are randomly sampled with
         scipy.stats.special_ortho_group when domain_dim>2
    scale_min : flaot
        The min scaling of simplex vectors. If None, one will be generated with a heuristic
    scale_max : flaot
        The max scaling of simplex vectors
    scale_sampling : 'lin' (default), 'log', 'rand'
        Method for obtaining the scales, 'lin' uses linspace(scale_min,scale_max),
         'log' uses logspace, and 'rand' uses uniform random sampling

    Other params are described in SPSpace
    '''
    def __init__(self, domain_dim: int,
                 ssp_dim: Optional[int] = 151,
                 domain_bounds: Optional[np.ndarray] = None,
                 n_scales: Optional[int] = -1,
                 n_rotates: Optional[int] = -1,
                 scale_min: Optional[float] = 1,
                 scale_max: Optional[float] = np.pi,
                 length_scale: Optional[Union[float, list, np.ndarray]] = 1,
                 scale_sampling: Optional[str] = 'lin',
                 rng: Optional[Union[int, np.random.Generator]] = None,
                 **kwargs):
        assert ssp_dim > 0, 'ssp_dim must be positive'
        assert ssp_dim > 2*(domain_dim+1) + 1

        self.rng = _get_rng(rng)
        # user wants to define ssp with total dim, not number of simplex rotates and scales
        if (n_rotates <= 0) or (n_scales <= 0):
            n_rotates = int(np.sqrt((ssp_dim - 1) / (2 * (domain_dim + 1))))
            n_scales = n_rotates
            # ssp_dim = n_rotates * n_scales * (domain_dim + 1) * 2 + 1

        phases_hex = np.hstack([np.sqrt(1 + 1 / domain_dim) * np.identity(domain_dim) - (domain_dim ** (-3 / 2)) * (
                    np.sqrt(domain_dim + 1) + 1),
                    (domain_dim ** (-1 / 2)) * np.ones((domain_dim, 1))]).T
        self.phases_hex = phases_hex
        self.grid_basis_dim = domain_dim + 1  # number of simplex vertices
        self.num_grids = n_rotates * n_scales
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.n_scales = n_scales
        self.n_rotates = n_rotates

        if domain_dim == 1:
            n_scales = n_scales * n_rotates
        irrational_base = (1 + np.sqrt(5)) / 2
        if domain_dim == 1:
            n_scales = n_scales * n_rotates
        if scale_sampling == 'lin':
            if scale_min is None:
                scale_min = scale_max / (n_scales * (irrational_base - 1) + 1)
            scales = np.linspace(scale_min, scale_max, n_scales)
        elif scale_sampling == 'log':
            if scale_min is None:
                scale_min = scale_max / (irrational_base ** (n_scales - 1))
            scales = np.geomspace(scale_min, scale_max, n_scales)
        elif scale_sampling == 'rand':
            if scale_min is None:
                scale_min = 0
            scales = self.rng.uniform(scale_min, scale_max, n_scales)
        phases_scaled = np.vstack([phases_hex * i for i in scales])

        if (n_rotates == 1) or (domain_dim == 1):
            phases_scaled_rotated = phases_scaled
            R_mats=None

        elif domain_dim == 2:
            angles = np.linspace(0, 2 * np.pi / 3, n_rotates, endpoint=False)
            R_mats = np.stack([np.stack([np.cos(angles), -np.sin(angles)], axis=1),
                               np.stack([np.sin(angles), np.cos(angles)], axis=1)], axis=1)
            phases_scaled_rotated = (R_mats @ phases_scaled.T).transpose(0, 2, 1).reshape(-1, domain_dim)
        else:
            R_mats = special_ortho_group.rvs(domain_dim, size=n_rotates, random_state=self.rng)
            phases_scaled_rotated = (R_mats @ phases_scaled.T).transpose(0, 2, 1).reshape(-1, domain_dim)

        self.scales = scales
        self.rot_mats = R_mats
        phase_matrix = conjsym(phases_scaled_rotated)
        ssp_dim = phase_matrix.shape[0]

        super().__init__(domain_dim, ssp_dim, phase_matrix=phase_matrix,
                         domain_bounds=domain_bounds, length_scale=length_scale, rng=self.rng)

    def sample_grid_encoders(self, n_neurons, method='sobol'):
        d = self.ssp_dim
        n = self.domain_dim
        A = self.phase_matrix
        k = (d - 1) // 2
        if d % 2 == 0:
            N = ((d - 2) // 2) // (n + 1)
        else:
            N = ((d - 1) // 2) // (n + 1)

        if method == 'grid':
            num_pts = int(np.ceil(n_neurons ** (1 / self.domain_dim)))
        else:
            num_pts = n_neurons

        sample_pts = self.get_sample_points(num_pts, method=method)[:n_neurons, :]
        n_per_pattern = int(np.floor(n_neurons / N))
        sorts = np.concatenate(
            [np.repeat(np.arange(0, N), n_per_pattern), self.rng.integers(0, N, size=n_neurons - N * n_per_pattern)])

        encoders = np.zeros((n_neurons, d))
        for i in range(n_neurons):
            res = np.zeros(d, dtype=complex)
            res[(1 + sorts[i] * (n + 1)):(n + 2 + sorts[i] * (n + 1))] = np.exp(
                1.j * A[(1 + sorts[i] * (n + 1)):(n + 2 + sorts[i] * (n + 1))] @ sample_pts[i, :])
            res[(k + 1):] = np.conjugate(np.flip(res[1:(k + 1)]))
            res[0] = 1
            if d % 2 == 0:
                res[d // 2] = 1
            encoders[i, :] = np.fft.ifft(res).real
        encoders = encoders / np.linalg.norm(encoders, axis=-1, keepdims=True)
        return encoders


def conjsym(K, even=False):
    d = K.shape[0]
    n = d * 2 + 1  #+ 1*even
    F = np.zeros((n, K.shape[1]), dtype="complex")
    F[1:(d + 1), :] = K
    F[(d + 1):, :] = -np.flip(K, axis=0)
    return F.real


def _get_sub_FourierSSP(n, N, sublen=3):
    tot_len = 2 * sublen * N + 1
    FA = np.zeros((2 * sublen + 1, tot_len))
    FA[0:sublen, sublen * n:sublen * (n + 1)] = np.eye(sublen)
    FA[sublen, sublen * N] = 1
    FA[sublen + 1:, tot_len - np.arange(sublen * (n + 1), sublen * n, -1)] = np.eye(sublen)
    return FA


def _get_sub_SSP(n, N, sublen=3):
    tot_len = 2 * sublen * N + 1
    FA = _get_sub_FourierSSP(n, N, sublen=sublen)
    W = np.fft.fft(np.eye(tot_len))
    invW = np.fft.ifft(np.eye(2 * sublen + 1))
    A = invW @ np.fft.ifftshift(FA) @ W
    return A.real


def _proj_sub_FourierSSP(n, N, sublen=3):
    tot_len = 2 * sublen * N + 1
    FB = np.zeros((2 * sublen + 1, tot_len))
    FB[0:sublen, sublen * n:sublen * (n + 1)] = np.eye(sublen)
    FB[sublen, sublen * N] = 1 / N  # all sub vectors have a "1" zero freq term so scale it so full vector will have 1
    FB[sublen + 1:, tot_len - np.arange(sublen * (n + 1), sublen * n, -1)] = np.eye(sublen)
    return FB.T


def _proj_sub_SSP(n, N, sublen=3):
    tot_len = 2 * sublen * N + 1
    FB = _proj_sub_FourierSSP(n, N, sublen=sublen)
    invW = np.fft.ifft(np.eye(tot_len))
    W = np.fft.fft(np.eye(2 * sublen + 1))
    B = invW @ np.fft.ifftshift(FB) @ W
    return B.real


def _Rd_sampling(n, d, seed=0.5):
    def phi(d):
        x = 2.0000
        for i in range(10):
            x = pow(1 + x, 1 / (d + 1))
        return x

    g = phi(d)
    alpha = np.zeros(d)
    for j in range(d):
        alpha[j] = pow(1 / g, j + 1) % 1
    z = np.zeros((n, d))
    for i in range(n):
        z[i] = seed + alpha * (i + 1)
    z = z % 1
    return z
