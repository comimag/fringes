import logging
import os
import itertools as it
from collections import namedtuple
import time
import json

import numpy as np
import scipy as sp
import sympy
import skimage as ski
import cv2
import toml
import yaml

from .util import vshape, bilateral, _remap
from . import grid
from .decoder import decode

logger = logging.getLogger(__name__)


class Fringes:
    """Easy-to-use class to configure, encode and decode customized fringe patterns using phase shifting algorithms."""

    # note: the class docstring is continuated at the end of the class

    # value limits
    _Hmax = 101  # this is arbitrary
    _Dmax = 2  # max 2 dimensions
    _Kmax = 101  # this is arbitrary, but must be < 128 when deploying spatial or frequency multiplexing @ uint8
    _Nmax = 1001  # this is arbitrary; more is better but the improvement scales with sqrt(N); @FDM: > 2 * _fmax + 1
    _Mmax = 101  # this is arbitrary; more is better but the improvement scales with sqrt(M)
    # _Pmax: int = 35651584  # ~8K i.e. max luma picture size of h264, h265, h266 video codecs as of 2022; todo: update
    _Pmax = 2**30  # 2^30 = 1,073,741,824 i.e. default size   limit of imread() in OpenCV
    _Xmax = 2**20  # 2^20 = 1,048,576     i.e. default width  limit of imread() in OpenCV
    _Ymax = 2**20  # 2^20 = 1,048,576     i.e. default height limit of imread() in OpenCV
    _Lmax = 2**20  # 2^20 = 1,048,576     i.e. default height limit of imread() in OpenCV
    _Tmax = _Hmax * _Dmax * _Kmax * _Nmax
    _alphamax = 2
    _gammamax = 3  # most screens have a gamma of ~2.2
    # _lminmin = 2  # l == 2 only if p0 != pi / 2 + 2pi*k, best if p0 == pi + 2pi*k with k is positive integer
    #            also l <= 2 yields errors in SPU: phase jumps = 2PI / lmin >= np.pi
    _lminmin = 3  # l >= 3 yields sufficient modulation theoretically
    # _lminmin = 8  # l >= 8 yields sufficient modulation practically [Liu2014]

    # allowed values; take care to only use immutable types!  # todo: is that so?
    values = {
        "grid": ("image"),  # todo: ("image", "Cartesian", "polar", "log-polar")
        "indexing": ("xy", "ij"),
        "dtype": (
            # "bool",  # results are too unprecise
            "uint8",
            "uint16",
            # "uint32",  # integer overflow in pyqtgraph -> replace line 528 of ImageItem.py with:
            # "uint64",  # bins = self._xp.arange(mn, mx + 1.01 * step, step, dtype="uint64")
            # "float16",  # numba doesn't handle float16, also most algorithms convert float16 to float32 anyway
            "float32",
            "float64",
        ),
        "mode": ("fast", "precise"),
    }

    # allowed values; take care to only use immutable types!
    _grids = (("image"),)  # todo: ("image", "Cartesian", "polar", "log-polar")
    _indexings = ("xy", "ij")
    _dtypes = (
        # "bool",  # results are too unprecise
        "uint8",
        "uint16",
        # "uint32",  # integer overflow in pyqtgraph -> replace line 528 of ImageItem.py with:
        # "uint64",  # bins = self._xp.arange(mn, mx + 1.01 * step, step, dtype="uint64")
        # "float16",  # numba doesn't handle float16, also most algorithms convert float16 to float32 anyway
        "float32",
        "float64",
    )
    _modes = ("fast", "precise")

    _loader = {
        ".json": json.load,
        ".yaml": yaml.safe_load,
        ".toml": toml.load,
    }

    _verbose_output = (
        "brightness",
        "modulation",
        "registration",
        "phase",
        "order",
        "residuals",
        "uncertainty",
        "exposure",
        "visibility",
    )

    # default values are defined here; take care to only use immutable types!
    def __init__(
        self,
        *args,  # bundels all args, what follows are only kwargs
        Y: int = 1200,
        X: int = 1920,
        H: int = 1,  # inferred from h
        M: float = 1.0,  # inferred from h
        D: int = 2,
        K: int = 3,
        T: int = 24,  # T is inferred
        N: tuple | np.ndarray = np.array([[4, 4, 4], [4, 4, 4]], int),
        l: tuple | np.ndarray = 1920 / np.array([[13, 7, 89], [13, 7, 89]], float),  # inferred from v
        v: tuple | np.ndarray = np.array([[13, 7, 89], [13, 7, 89]], float),
        f: tuple | np.ndarray = np.array([[1, 1, 1], [1, 1, 1]], float),
        h: tuple | np.ndarray = np.array([[255, 255, 255]], int),
        p0: float = np.pi,
        gamma: float = 1.0,
        A: float = 255 / 2,  # i.e. Imax / 2 @ uint8; inferred from Imax and beta
        B: float = 255 / 2,  # i.e. Imax / 2 @ uint8; inferred from Imax and beta and V
        beta: float = 0.5,
        V: float = 1.0,  # V is inferred from A and B
        Vmin: float = 0.0,
        umax: float = 0.5,
        alpha: float = 1.0,
        dtype: str | np.dtype = "uint8",
        grid: str = "image",
        angle: float = 0.0,
        axis: int = 0,
        SDM: bool = False,
        WDM: bool = False,
        FDM: bool = False,
        static: bool = False,
        lmin: float = 8.0,
        indexing: str = "xy",
        reverse: bool = False,
        verbose: bool = False,
        Bv: tuple | np.ndarray = None,
        PSF: float = 0.0,
        dark: float = 0.0,
        gain: float = 0.0,
        y0: float = 0.0,
        mode: str = "fast",
        #  **kwargs,  # bundles all undefined kwargs, else error: __init__() got an unexpected keyword argument
    ) -> None:
        # given values which are in defaults but are not identical to them
        given = {
            k: v for k, v in sorted(locals().items()) if k in self.defaults and not np.array_equal(v, self.defaults[k])
        }

        # set default values
        self._UMR = None  # used for caching
        for k, v in self.defaults.items():
            if k not in "HMTlAB":  # these properties are inferred from others
                setattr(self, f"_{k}", v)  # define private variables from where the properties get their value from

        # set given values
        self.params = given

        # _ = self.UMR  # property 'UMR' logs warning if necessary

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.encode(*args, **kwargs)

    def __getitem__(self, frames: int | tuple | slice) -> np.ndarray:
        if isinstance(frames, slice):
            frames = np.arange(self.T)[frames]

        return self.encode(frames=frames)

    def __iter__(self):
        self._t = 0
        return self

    def __next__(self) -> np.ndarray:
        if self._t < self.T:
            I = self.encode(frames=self._t)
            self._t += 1
            return I
        else:
            del self._t
            raise StopIteration()

    def __len__(self) -> int:
        """Number of frames."""
        return self.T

    def __eq__(self, other) -> bool:
        return hasattr(other, "params") and self.params == other.params

    def __contains__(self, item):
        return item in self.properties

    def __str__(self) -> str:
        return self.__name__

    def __repr__(self) -> str:
        return f"{self.params}"

    def load(self, fname: str = None) -> dict:
        """Load parameters from a config file to the `Fringes` instance.

        .. warning:: The parameters are only loaded if the config file provides the section `fringes`.

        Parameters
        ----------
        fname : str, optional
            File name of the file to load.
            Supported file formats are: *.json, *.yaml, *.toml.
            If `fname` is not provided, the file `.fringes.yaml` within the user home directory is loaded.

        Returns
        -------
        params : dict
            The loaded parameters as a dictionary.
            `params` may be empty.

        Examples
        --------
        >>> import os
        >>> fname = os.path.join(os.path.expanduser("~"), ".fringes.yaml")

        >>> import fringes as frng
        >>> f = frng.Fringes()

        >>> f.load(fname)
        """

        if fname is None:
            fname = os.path.join(os.path.expanduser("~"), ".fringes.yaml")

        if not os.path.isfile(fname):
            logger.error(f"File '{fname}' does not exist.")
            return

        with open(fname, "r") as f:
            ext = os.path.splitext(fname)[-1]

            if ext == ".json":
                p = json.load(f)
            elif ext == ".yaml":
                p = yaml.safe_load(f)
            elif ext == ".toml":
                p = toml.load(f)
            else:
                logger.error(f"Unknown file type '{ext}'.")
                return {}

        if "fringes" in p:
            params = p["fringes"]
            self.params = params

            logger.info(f"Loaded parameters from '{fname}'.")

            return params
        else:
            logger.error(f"No 'fringes' section in file '{fname}'.")
            return {}

    def save(self, fname: str = None) -> None:
        """Save the parameters of the `Fringes` instance to a config file.

        Within the file, the parameters are written to the section `fringes`.

        Parameters
        ----------
        fname : str, optional
            File name of the file to save.
            Supported file formats are: *.json, *.yaml, *.toml.
            If `fname` is not provided, the parameters are saved to
            the file `.fringes.yaml` within the user home directory.

        Examples
        --------
        >>> import os
        >>> fname = os.path.join(os.path.expanduser("~"), ".fringes.yaml")

        >>> import fringes as frng
        >>> f = frng.Fringes()

        >>> f.save(fname)
        """

        if fname is None:
            fname = os.path.join(os.path.expanduser("~"), ".fringes.yaml")

        if not os.path.isdir(os.path.dirname(fname)):
            logger.warning(f"File directory does not exist.")
            return

        name, ext = os.path.splitext(fname)
        if not ext:
            name, ext = ext, name

        if ext not in self._loader.keys():
            logger.warning(f"File extension is unknown. Must be one of {self._loaders.keys()}")
            return

        with open(fname, "w") as f:
            if ext == ".json":
                json.dump({"fringes": self.params}, f, indent=4)
            elif ext == ".yaml":
                yaml.dump({"fringes": self.params}, f)
            elif ext == ".toml":
                toml.dump({"fringes": self.params}, f)

        logger.debug(f"Saved parameters to {fname}.")  # todo: info?

    def reset(self) -> None:
        """Reset parameters of the `Fringes` instance to default values."""

        self.params = self.defaults
        logger.info("Reset parameters to defaults.")

    def optimize(self, T: int = None, umax: float = None) -> None:  # todo: self.umax
        """Optimize the parameters of the `Fringes` instance.

        Parameters
        ----------
         T : int, optional
            Number of frames.
            If `T` is not provided, the number of frames from the `Fringes` instance is used.
            Then, the `Fringes` instance's number of shifts `N` is distributed optimally over the directions and sets.

         umax : float, optional
            Maximum allowable uncertainty.
            Must be greater than zero.

        Notes
        -----
        If `umax` is specified, the parameters are determined
        that allow a maximal uncertainty of `umax`
        with a minimum number of frames.

        Else, the parameters of the `Fringes` instance are optimized to yield the minimal uncertainty
        using the given number of frames `T`.
        """

        K = np.log(self.L) / np.log(self.lopt)  # lopt ** K = L
        K = np.ceil(max(2, K))
        self.K = K
        self.v = "optimal"

        if umax is not None:  # umax -> T
            self.N = int(np.median(self.N))  # make N const.
            a = self.u.max() / umax
            N = self.N * a**2
            self.N = np.maximum(3, np.ceil(N))

            if self.u > umax:
                ...  # todo: check if umax is reached
        else:  # T -> u
            if T is None:
                T = self.T

            self.T = T  # distribute frames optimally (evenly) on shifts

        logger.info("Optimized parameters.")

    @staticmethod
    def gamma_auto_correct(I: np.ndarray) -> np.ndarray:
        """Automatically estimate and apply the gamma correction factor
        to linearize the display/camera response curve.

        Parameters
        ----------
        I : np.ndarray
            Recorded data.

        Returns
        -------
        J : np.ndarray
            Linearized data.
        """

        # normalize to [0, 1]
        Imax = np.iinfo(I.dtype).max if I.dtype.kind in "ui" else 1
        J = I / Imax

        # estimate gamma correction factor
        med = np.nanmedian(J)  # Median is a robust estimator for the mean.
        gamma = np.log(med) / np.log(0.5)
        inv_gamma = 1 / gamma

        # apply inverse gamma
        # table = np.array([((g / self.Imax) ** invGamma) * self.Imax for g in range(self.Imax + 1)], self.dtype)
        # J = cv2.LUT(J, table)
        J **= inv_gamma
        J *= Imax

        return J

    def deinterlace(self, I: np.ndarray) -> np.ndarray:
        """Deinterlace fringe patterns.

        This for fringe patterns
        which were acquired with a line scan camera
        while each frame has been displayed and captured
        while the object was moving by one pixel.

        Parameters
        ----------
        I : np.ndarray
            Fringe pattern sequence.
            It is reshaped to videoshape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.

        Returns
        -------
        I : np.ndarray
            Deinterlaced fringe pattern sequence.

        Raises
        ------
        AssertionError
            If the number of frames of `I` and the attribute `T` of the `Fringes` instance don't match.

        Examples
        --------
        >>> import fringes as frng
        >>> f = frng.Fringes()
        >>> I = f.deinterlace(I)
        """

        t0 = time.perf_counter()

        T, Y, X, C = vshape(I).shape
        assert T * Y % self.T == 0, "Number of frames of parameters and data don't match."

        # I = I.reshape((T * Y, X, C))  # concatenate
        I = I.reshape((-1, self.T, X, C)).swapaxes(0, 1)

        logger.info(f"{int(1000 * (time.perf_counter() - t0))}ms")

        return I

    def coordinates(self) -> np.ndarray:
        """Generate the coordinate matrices of the coordinate system defined in `grid`.

        Returns
        -------
        xi : np.ndarray
            Coordinate matrices.
        """

        t0 = time.perf_counter()

        sys = (
            "img"
            if self.grid == "image"
            else "cart" if self.grid == "Cartesian" else "pol" if self.grid == "polar" else "logpol"
        )

        xi = np.array(getattr(grid, sys)(self.Y, self.X, self.angle))

        if self.indexing == "ij":
            xi = xi[::-1]  # returns a view

        if self.D == 1:
            xi = xi[self.axis][None, :, :]  # returns a view

        if self.grid in ["polar", "log-polar"]:
            xi *= self.L

        logger.info(f"{int(1000 * (time.perf_counter() - t0))}ms")

        return xi

    def _orders(self) -> np.ndarray:
        """Generate fringe orders.

        Returns
        -------
        k : np.ndarray
            Fringe orders of the encoded fringe pattern sequence.
        """

        k = self.coordinates()[:, None, :, :, None] // self._l[:, :, None, None, None]

        return k.reshape(self.D * self.K, self.Y, self.X, self.C)

    def _modulate(self, xi: np.ndarray, frames: np.ndarray, rint: bool = True) -> np.ndarray:
        """Encode base fringe patterns by spatio-temporal modulation.

        Parameters
        ----------
        xi : None or list of arrays or one array of grid indices, optional
            List of coordinate matrices from coordinate vectors in Cartesian ('xy') indexing
            (e.g. from `numpy.meshgrid <https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html>`_).
            The default is equal to 'numpy.indices((self.Y, self.X))'.

        frames : np.ndarray
            Indices of the frames to be encoded.

        rint : bool, optional
            If this is set to True (the default)
            and the used dtype (attribute `dtype` of the Fringes instance) is of type interger,
            the encoded patterns will be rounded to the nearest integer.
            If this is set False and the used dtype is of type interger,
            the fractional part of the encoded patterns will be discarded.

        Returns
        -------
        I : np.ndarray
            Base fringe patterns.
        """

        t0 = time.perf_counter()

        frames = np.array(list(set(t % np.sum(self._N) for t in frames)))  # numpy.unique returns sorted
        T = len(frames)
        Y = self.Y if xi is None else xi.shape[1]
        X = self.X if xi is None else xi.shape[2]

        is_mixed_color = np.any((self.h != 0) * (self.h != 255))
        dtype = np.dtype("float64") if self.SDM or self.FDM or is_mixed_color else self.dtype

        # coordinates
        if xi is None:
            if self.grid != "image" or self.angle != 0:
                xi = self.coordinates()
            else:
                B = int(np.ceil(np.log2(self.R.max() - 1)))  # next power of two
                B += -B % 8  # next power of two divisible by 8
                xi = np.indices((self.Y, self.X), dtype=f"uint{B}", sparse=True)
                if self.indexing == "xy":
                    xi = xi[::-1]
                if self.D == 1:
                    xi = [xi[self.axis]]

        I = np.empty([T, Y, X], dtype)

        # Ncum = np.cumsum(self._N).reshape(self.D, self.K)
        # for t in frames:
        #     d, i = np.argwhere(t < Ncum)[0]
        #     n = t - Ncum[d, i] + self._N[0, 0]
        #     ...

        idx = 0
        frame = 0
        for d in range(self.D):
            x = (xi[d] + self.x0) / self.L

            for i in range(self.K):
                k = 2 * np.pi * self._v[d, i]
                w = 2 * np.pi * self._f[d, i]

                if self.reverse:
                    w *= -1

                for n in range(self._N[d, i]):
                    if frame in frames:
                        t = n / 4 if self._N[d, i] == 2 else n / self._N[d, i]

                        val = self.Imax * (self.beta * (1 + self.V * np.cos(k * x - w * t - self.p0))) ** self.gamma

                        if dtype.kind in "ui" and rint:
                            np.rint(val, out=val)
                        elif dtype.kind in "b":
                            val = val >= 0.5

                        I[idx] = val

                        idx += 1
                    frame += 1

        logger.debug(f"{int(1000 * (time.perf_counter() - t0))}ms")

        return I.reshape(-1, Y, X, 1)

    def _demodulate(
        self, I: np.ndarray, verbose: bool = False, func: str = "ski"
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """Decode base fringe patterns by spatio-temporal demodulation.

        Parameters
        ----------
        I : np.ndarray
            Fringe pattern sequence.
            It is reshaped to videoshape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.

        verbose : bool, optional
            If this or the argument `verbose` of the Fringes instance is set to True,
            additional infomation is computed and retuned.
            This includes: phase maps, residuals, fringe orders, visibility and relative exposure.

        func : str, optional
            Unwrapping function to use. The default is 'ski'.

            - 'ski': `Scikit-image <https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html>`_

            - else: `OpenCV https://docs.opencv.org/4.7.0/df/d3a/group__phase__unwrapping.html>`_

        Returns
        -------
        brightness : np.ndarray
            Local background signal.

        modulation : np.ndarray
            Local amplitude of the cosine signal.

        registration : np.ndarray
            Decoded coordinates.

            .. note:: The registration is a mapping in the same pixel grid as the camera sensor
              and contains the information where each camera pixel, i.e. each camera sightray,
              was looking at during the fringe pattern acquisition.

        residuals : np.ndarray, optional
            Residuals from the optimization-based unwrapping process.

        phase : np.ndarray, optional
            Local phase.
        """

        t0 = time.perf_counter()

        # parse
        T, Y, X, C = vshape(I).shape  # extract Y, X, C from data as these parameters depend on used camera
        I = I.reshape((T, Y, X, C))

        # if self.FDM:
        #    c = np.fft.rfft(I, axis=0) / T  # todo: hfft
        #
        #    # if np.any(self._N > 2 * self.D * self.K):  # 2 * np.abs(_f).max() + 1:
        #    #     i = np.append(np.zeros(1, int), self._f.flatten().astype(int, copy=False))  # add p0
        #    #     c = c[i]
        #
        #    a = abs(c)
        #    # i = np.argsort(a[1:], axis=0)[::-1]  # indices of frequencies, sorted by their magnitude
        #    phi = -np.angle(c * np.exp(-1j * (self.p0 - np.pi)))[_f.flatten().astype(int, copy=False)]  # todo: why p0 - PI???

        if self.uwr == "FTM":
            # todo: make passband symmetrical around carrier frequency?
            if self.D == 2:
                fx = np.fft.fftshift(np.fft.fftfreq(X))  # todo: hfft
                fy = np.fft.fftshift(np.fft.fftfreq(Y))
                fxx, fyy = np.meshgrid(fx, fy)
                mx = np.abs(fxx) > np.abs(
                    fyy
                )  # mask for x-frequencies  # todo: make left and right borders round (symmetrical around base band)
                my = np.abs(fxx) < np.abs(
                    fyy
                )  # mask for y-frequencies  # todo: make lower and upper borders round (symmetrical around base band)

                W = 100  # assume window width for filtering out baseband
                W = min(max(3, W), min(X, Y) / 20)  # clip to ensure plausible value
                a = int(min(max(0, W), X / 4) + 0.5)  # todo: find good upper cut off frequency
                # a = X // 4
                mx[:, :a] = 0  # remove high frequencies
                b = int(X / 2 - W / 2 + 0.5)
                mx[:, b:] = 0  # remove baseband and positive frequencies

                H = 100  # assume window height for filtering out baseband
                H = min(max(3, H), min(X, Y) / 20)  # clip to ensure plausible value
                c = int(min(max(0, H), Y / 4) + 0.5)  # todo: find good upper cut off frequency
                # c = Y // 4
                my[:c, :] = 0  # remove high frequencies
                d = int(Y / 2 - H / 2 + 0.5)
                my[d:, :] = 0  # remove baseband and positive frequencies

                # todo: smooth edges of filter masks, i.e. make them Hann Windows

                if C > 1:
                    I = I[..., 0]
                    C = 1
                    I = I.reshape((T, Y, X, C))

                phi = np.empty([self.D, Y, X, C], np.float32)
                res = np.empty([self.D, Y, X, C], np.float32)
                # if self.verbose:
                #     phi = np.empty([self.D, Y, X, C], np.float32)
                #     res = np.empty([self.D, Y, X, C], np.float32)
                reg = np.empty([self.D, Y, X, C], np.float32)
                bri = np.empty([self.D, Y, X, C], np.float32)
                mod = np.empty([self.D, Y, X, C], np.float32)
                fid = np.full([self.D, Y, X, C], np.nan, np.float32)
                for c in range(C):
                    # todo: hfft
                    I_FFT = np.fft.fftshift(np.fft.fft2(I[0, ..., c]))

                    I_FFT_x = I_FFT * mx
                    ixy, ixx = np.unravel_index(I_FFT_x.argmax(), I_FFT_x.shape)  # get indices of carrier frequency
                    I_FFT_x = np.roll(I_FFT_x, X // 2 - ixx, 1)  # move to center

                    I_FFT_y = I_FFT * my
                    iyy, iyx = np.unravel_index(I_FFT_y.argmax(), I_FFT_y.shape)  # get indices of carrier frequency
                    I_FFT_y = np.roll(I_FFT_y, Y // 2 - iyy, 0)  # move to center

                    Jx = np.fft.ifft2(np.fft.ifftshift(I_FFT_x))
                    Jy = np.fft.ifft2(np.fft.ifftshift(I_FFT_y))

                    reg[0, ..., c] = np.angle(Jx)
                    reg[1, ..., c] = np.angle(Jy)
                    # todo: bri
                    mod[0, ..., c] = np.abs(Jx) * 2  # factor 2 because one sideband is filtered out
                    mod[1, ..., c] = np.abs(Jy) * 2  # factor 2 because one sideband is filtered out
                    if self.verbose or verbose:
                        phi[0, ..., c] = reg[0, ..., c]
                        phi[1, ..., c] = reg[1, ..., c]
                        res[0, ..., c] = np.log(np.abs(I_FFT))  # J  # todo: hfft
                        res[1, ..., c] = np.log(np.abs(I_FFT))  # J
                        # todo: I - J
            elif self.D == 1:
                L = max(X, Y)
                fx = np.fft.fftshift(np.fft.fftfreq(X)) * X * Y / L  # todo: hfft
                fy = np.fft.fftshift(np.fft.fftfreq(Y)) * Y * X / L
                fxx, fyy = np.meshgrid(fx, fy)
                frr = np.sqrt(fxx**2 + fyy**2)  # todo: normalization of both directions

                mr = frr <= L / 2  # ensure same sampling in all directions
                W = 10
                W = min(max(1, W / 2), L / 20)
                mr[frr < W] = 0  # remove baseband
                mr[frr > L / 4] = 0  # remove too high frequencies

                mh = np.empty([Y, X])
                mh[:, : X // 2] = 1
                mh[:, X // 2 :] = 0

                if C > 1:
                    I = I[..., 0]
                    C = 1
                    I = I.reshape((T, Y, X, C))

                phi = np.empty([self.D, Y, X, C], np.float32)
                res = np.empty([self.D, Y, X, C], np.float32)
                fid = np.full([self.D, Y, X, C], np.nan, np.float32)
                reg = np.empty([self.D, Y, X, C], np.float32)
                bri = np.empty([self.D, Y, X, C], np.float32)
                mod = np.empty([self.D, Y, X, C], np.float32)
                for c in range(C):
                    # todo: hfft
                    I_FFT = np.fft.fftshift(np.fft.fft2(I[0, ..., c]))

                    I_FFT_r = I_FFT * mr
                    iy, ix = np.unravel_index(I_FFT_r.nanargmax(), I_FFT_r.shape)  # get indices of carrier frequency
                    y, x = Y / 2 - iy, X / 2 - ix
                    a = np.degrees(np.arctan2(y, x))
                    mhr = sp.ndimage.rotate(mh, a, reshape=False, order=0, mode="nearest")

                    I_FFT_r *= mhr  # remove one sideband
                    I_FFT_r = np.roll(I_FFT_r, X // 2 - ix, 1)  # move to center
                    I_FFT_r = np.roll(I_FFT_r, Y // 2 - iy, 0)  # move to center

                    J = np.fft.ifft2(np.fft.ifftshift(I_FFT_r))

                    reg[0, ..., c] = np.angle(J)
                    # todo: bri
                    mod[0, ..., c] = np.abs(J) * 2  # factor 2 because one sideband is filtered out
                    if self.verbose or verbose:
                        phi[0, ..., c] = reg[0, ..., c]
                        res[0, ..., c] = np.log(np.abs(I_FFT))  # J
                        # todo: I - J
        else:
            bri, mod, phi, reg, res = decode(
                I,
                self._N,
                self._v,
                self._f * (-1 if self.reverse else 1),
                self.R,
                self.UMR,
                self.x0,
                self.p0,
                self.Vmin,
                self.verbose or verbose,
            )

        logger.debug(f"{int(1000 * (time.perf_counter() - t0))}ms")

        return bri, mod, phi, reg, res

    def _multiplex(self, I: np.ndarray, rint: bool = True) -> np.ndarray:
        """Multiplex fringe patterns.

        Parameters
        ----------
        I : np.ndarray
            Base fringe patterns.
        rint : bool, optional
            If this is set to True (the default)
            and the used dtype (attribute `dtype` of the Fringes instance) is of type interger,
            the encoded patterns will be rounded to the nearest integer.
            If this is set False and the used dtype is of type interger,
            the fractional part of the encoded patterns will be discarded.
        Returns
        -------
        I : np.ndarray
            Multiplexed fringe patterns.
        """

        t0 = time.perf_counter()

        if self.WDM:
            assert not self.FDM
            assert self._monochrome
            assert np.all(self.N == 3)

            I = I.reshape((-1, 3, self.Y, self.X, 1))  # returns a view
            I = I.swapaxes(1, -1)  # returns a view
            I = I.reshape((-1, self.Y, self.X, self.C))  # returns a view

        if self.SDM:
            assert not self.FDM
            assert self.grid in self._grids[:2]
            assert self.D == 2
            assert I.dtype.kind == "f"

            I = I.reshape((self.D, -1))  # returns a view
            I -= self.A
            I = np.sum(I, axis=0)
            I += self.A
            # I *= 1 / self.D
            I = I.reshape((-1, self.Y, self.X, self.C if self.WDM else 1))  # returns a view
            if self.dtype.kind in "uib":
                if rint:
                    np.rint(I, out=I)
                I = I.astype(self.dtype, copy=False)  # returns a view

        if self.FDM:
            assert not self.WDM
            assert not self.SDM
            assert len(np.unique(self.N)) == 1
            assert I.dtype.kind == "f"

            if np.any(self._N < 2 * np.abs(self._f).max() + 1):  # todo: fractional periods
                logger.warning("Decoding might be disturbed.")

            I = I.reshape((self.D * self.K, -1))  # returns a view
            I -= self.A
            I = np.sum(I, axis=0)
            I += self.A
            # I *= 1 / (self.D * self.K)
            I = I.reshape((-1, self.Y, self.X, 1))  # returns a view
            if self.dtype.kind in "uib":
                if rint:
                    np.rint(I, out=I)
                I = I.astype(self.dtype, copy=False)  # returns a view

        logger.debug(f"{int(1000 * (time.perf_counter() - t0))}ms")

        return I

    def _demultiplex(self, I: np.ndarray) -> np.ndarray:
        """Demultiplex fringe patterns.

        Parameters
        ----------
        I : np.ndarray
            Multiplexed fringe patterns.

        Returns
        -------
        I : np.ndarray
            Demultiplexed fringe patterns.
        """

        t0 = time.perf_counter()

        T, Y, X, C = vshape(I).shape

        if self.SDM:
            assert not self.FDM
            assert self.grid in self._grids[:2]
            assert self.D == 2

            if X % 2 == 0:
                fx = np.fft.fftshift(np.fft.rfftfreq(X))
                fy = np.fft.fftshift(np.fft.fftfreq(Y))
                fxx, fyy = np.meshgrid(fx, fy)
                mx = np.abs(fxx) >= np.abs(fyy)
                my = np.abs(fxx) <= np.abs(fyy)
                J = I
                I = np.empty((2 * T, Y, X, C))
                for t in range(T):
                    for c in range(C):
                        I_FFT = np.fft.fftshift(np.fft.rfft2(J[t, ..., c]))
                        I[t, ..., c] = np.fft.irfft2(np.fft.ifftshift(I_FFT * mx))
                        I[T + t, ..., c] = np.fft.irfft2(np.fft.ifftshift(I_FFT * my))
            elif Y % 2 == 0:
                fx = np.fft.fftshift(np.fft.rfftfreq(Y))
                fy = np.fft.fftshift(np.fft.fftfreq(X))
                fxx, fyy = np.meshgrid(fx, fy)
                mx = np.abs(fxx) >= np.abs(fyy)
                my = np.abs(fxx) <= np.abs(fyy)
                J = I.transpose(0, 2, 1, 3)
                I = np.empty((2 * T, X, Y, C))
                for t in range(T):
                    for c in range(C):
                        I_FFT = np.fft.fftshift(np.fft.rfft2(J[t, ..., c]))
                        I[t, ..., c] = np.fft.irfft2(np.fft.ifftshift(I_FFT * my))
                        I[T + t, ..., c] = np.fft.irfft2(np.fft.ifftshift(I_FFT * mx))
                I = I.transpose(0, 2, 1, 3)
            else:
                fx = np.fft.fftshift(np.fft.fftfreq(X))
                fy = np.fft.fftshift(np.fft.fftfreq(Y))
                fxx, fyy = np.meshgrid(fx, fy)
                mx = np.abs(fxx) >= np.abs(fyy)
                my = np.abs(fxx) <= np.abs(fyy)
                J = I
                I = np.empty((2 * T, Y, X, C))
                for t in range(T):
                    for c in range(C):
                        I_FFT = np.fft.fftshift(np.fft.fft2(J[t, ..., c]))
                        I[t, ..., c] = np.abs(np.fft.ifft2(np.fft.ifftshift(I_FFT * mx)))
                        I[T + t, ..., c] = np.abs(np.fft.ifft2(np.fft.ifftshift(I_FFT * my)))

        if self.WDM:
            assert not self.FDM
            assert C == 3
            I = I.reshape((-1, 1, Y, X, C))  # returns a view
            I = I.swapaxes(-1, 1)  # returns a view
            I = I.reshape((-1, Y, X, 1))  # returns a view

        if self.FDM:
            assert not self.WDM
            assert not self.SDM  # todo: allow self.SDM?
            assert len(np.unique(self.N)) == 1
            I = np.tile(I, (self.D * self.K, 1, 1, 1))

        logger.debug(f"{int(1000 * (time.perf_counter() - t0))}ms")

        return I

    def _colorize(self, I: np.ndarray, frames: np.ndarray) -> np.ndarray:
        """Colorize fringe patterns.

        Parameters
        ----------
        I : np.ndarray
            Base fringe patterns,
            possibly multiplexed.
        frames : None or int or tuple of ints, optional
            Indices of the frames to be encoded.
            The default, frames=None, will encode all frames at once.
            If frames is negative, it counts from the last to the first frame.
            If frames contains numbers whose magnitude is larger than the total number of frames
            (as specified by the attribute `T` of the Fringes instance), it is wrapped around.
            If frames is a tuple of ints, only the frames specified in the tuple are encoded.

        Returns
        -------
        I : np.ndarray
            Colorized fringe patterns.
        """

        t0 = time.perf_counter()

        T_ = I.shape[0]  # number of frames for each hue
        T = len(frames)
        J = np.empty((T, self.Y, self.X, self.C), self.dtype)

        for t in frames:
            tb = t % I.shape[0]  # indices from base fringe pattern I

            for c in range(self.C):
                h = int(t // T_)  # hue index
                cb = c if self.WDM else 0  # color index of base fringe pattern I

                if self.h[h, c] == 0:  # uibf -> uibf
                    J[t, ..., c] = 0
                elif self.h[h, c] == 255 and J.dtype == self.dtype:  # uibf -> uibf
                    J[t, ..., c] = I[tb, ..., cb]
                elif self.dtype.kind in "uib":  # f -> uib
                    J[t, ..., c] = np.rint(I[tb, ..., cb] * (self.h[h, c] / 255))  # .astype(self.dtype, copy=False)
                else:  # f -> f
                    J[t, ..., c] = I[tb, ..., cb] * (self.h[h, c] / 255)

        # for h in range(self.H):
        #     if frames is None:
        #         for c in range(self.C):
        #             cj = c if self.WDM else 0  # todo: ???
        #             if self.h[h, c] == 0:  # uib -> uib, f -> f
        #                 J[h * Th : (h + 1) * Th, ..., c] = 0
        #             elif self.h[h, c] == 255 and J.dtype == self.dtype:  # uib -> uib, f -> f
        #                 J[h * Th : (h + 1) * Th, ..., c] = I[..., cj]
        #             elif self.dtype.kind in "uib":  # f -> uib
        #                 J[h * Th : (h + 1) * Th, ..., c] = np.rint(I[..., cj] * (self.h[h, c] / 255)).astype(
        #                     self.dtype, copy=False
        #                 )
        #             else:  # f -> f
        #                 J[h * Th : (h + 1) * Th, ..., c] = I[..., cj] * (self.h[h, c] / 255)
        #     elif h in hues:  # i.e. frames is not None and h in hues
        #         for c in range(self.C):
        #             cj = c if self.WDM else 0  # todo: ???
        #             if self.h[h, c] == 0:  # uib -> uib, f -> f
        #                 J[i, ..., c] = 0
        #             elif self.h[h, c] == 255 and J.dtype == self.dtype:  # uib -> uib, f -> f
        #                 J[i, ..., c] = I[i, ..., cj]
        #             elif self.dtype.kind in "uib":  # f -> uib
        #                 J[i, ..., c] = np.rint(I[i, ..., cj] * (self.h[h, c] / 255)).astype(self.dtype, copy=False)
        #             else:  # f -> f
        #                 J[i, ..., c] = I[i, ..., cj] * (self.h[h, c] / 255)
        #         i += 1

        logger.debug(f"{int(1000 * (time.perf_counter() - t0))}ms")

        return J

    def _decolorize(self, I: np.ndarray) -> np.ndarray:
        """Decolorize fringe patterns by weighted averaging of hues.

        Parameters
        ----------
        I : np.ndarray
            Colorized fringe patterns.

        Returns
        -------
        I : np.ndarray
            Decolorized fringe patterns."""

        t0 = time.perf_counter()

        T, Y, X, C = vshape(I).shape
        I = I.reshape((self.H, T // self.H, Y, X, C))  # returns a view

        base = np.all(np.count_nonzero(self.h, axis=1) == 1)  # each hue consists of only one RGB base color
        solo = np.all(np.count_nonzero(self.h, axis=0) == 1)  # each RGB component exists exactly once
        mono = len(set(self.h[self.h != 0])) == 1  # all colors are monochromatic, i.e. are the same or zero
        # if base and solo: no averaging necessary
        # if mono: all weights are the same

        if self.H == 3 and C in [1, 3] and base and solo and mono:
            I = np.moveaxis(I, 0, -2)  # returns a view

            # basic slicing returns a view
            idx = np.argmax(self.h, axis=1)
            if np.array_equal(idx, [0, 2, 1]):  # RBG
                I = I[..., 0::-1, :]
            elif np.array_equal(idx, [1, 2, 0]):  # GBR
                I = I[..., 1:1:, :]
            elif np.array_equal(idx, [1, 0, 2]):  # GRB
                I = I[..., 1:1:-1, :]
            elif np.array_equal(idx, [2, 1, 0]):  # BGR
                I = I[..., 2::-1, :]
            elif np.array_equal(idx, [2, 0, 1]):  # BRG
                I = I[..., 2:2:-1, :]
            # elif np.array_equal(idx, [0, 1, 2]):  # RGB
            #     I = I[..., :, :]

            if C == 1:
                I = I[..., 0]  # returns a view
            elif C == 3:
                I = np.diagonal(I, axis1=-2, axis2=-1)  # returns a view
        elif self.H == 2 and C in [1, 3] and solo and mono:
            # advanced indexing returns a copy, not a view
            if C == 1:
                I = np.squeeze(
                    I,
                )  # returns a view
                I = np.moveaxis(I, 0, -1)  # returns a view
                idx = np.argmax(self.h, axis=0)
                I = I[..., idx]
            elif C == 3:
                idx = self.h != 0
                I = np.moveaxis(I, 0, -2)  # returns a view
                I = I[..., idx]
        else:
            # fuse colors by weighted averaging

            w = self.h / np.sum(self.h, axis=0)  # normalized weights
            # w[np.isnan(w)] = 0
            if C == 1 and mono:
                w = w[:, 0][:, None]  # ensures that the result has only one color channel

            # if np.all((w == 0) | (w == 1)):  # todo: fuse hues with WAVG
            #     w = w.astype(bool, copy=False)  # multiplying with bool preserves dtype
            #     dtype = I.dtype  # without this, np.sum chooses a dtype which can hold the theoretical maximal sum
            # else:

            dtype = float  # without this, np.sum chooses a dtype which can hold the theoretical maximal sum
            I = np.sum(I * w[:, None, None, None, :], axis=0, dtype=dtype)
            # todo: numpy.tensordot ?

        logger.debug(f"{int(1000 * (time.perf_counter() - t0))}ms")

        return I

    def encode(
        self,
        xi: list | np.ndarray = None,
        frames: int | tuple = None,
        rint: bool = True,
        simulate: bool = False,
    ) -> np.ndarray:
        """Encode fringe patterns.

        Parameters
        ----------
        xi : None or list of arrays or one array of grid indices, optional
            List of coordinate matrices from coordinate vectors in Cartesian ('xy') indexing
            (e.g. from `numpy.meshgrid <https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html>`_).
            The default is equal to 'numpy.indices((self.Y, self.X))'.

        frames : None or int or tuple of ints, optional
            Indices of the frames to be encoded.
            The default, frames=None, will encode all frames at once.
            If frames is negative, it counts from the last to the first frame.
            If frames contains numbers whose magnitude is larger than the total number of frames
            (as specified by the attribute `T` of the Fringes instance), it is wrapped around.
            If frames is a tuple of ints, only the frames specified in the tuple are encoded.
            If indices occur more than once, only the first occurence is encoded.

        rint : bool, optional
            If this is set to True (the default)
            and the used dtype (attribute `dtype` of the Fringes instance) is of type interger,
            the encoded patterns will be rounded to the nearest integer.
            If this is set False and the used dtype is of type interger,
            the fractional part of the encoded patterns will be discarded.

        simulate : bool, optional
            If this is set to True, the acquisition, i.e. the transmission channel, will be simulated.
            This includes the modulation transfer function
            (computed from the imaging system's point spread function)
            and intensity noise added by the camera.
            The required parameters for this are the instance's attributes
            `PSF`, `gain`, `dark_current`, `dark`, `quant` and 'shot' .
            Default is False.

        Returns
        -------
        I : np.ndarray
            Fringe pattern sequence.

        Raises
        ------
        Value Error
            The length of the list/number of coordinate matrices doesn't match the Fringes instance's parameter `D`,
            the smallest contained coodrinate is smaller than zero
            or the largest contained coodinate is equal to or exceeds the Fringes instance's width `X` resp. height `Y`.

        Notes
        -----
        To receive the frames iteratively (i.e. in a lazy manner),
        simply iterate over the Fringes instance.
        Alternatively, to receive arbitrary frames,
        index the Fringes instance directly,
        either with an integer, a tuple or a slice.

        Examples
        --------
        >>> import fringes as frng
        >>> f = frng.Fringes()

        Encode the complete fringe pattern sequence.

        >>> I = f.encode()

        Encode the first frame of the fringe pattern sequence.

        >>> I = f.encode(frames=0)
        >>> I = f[0]
        >>> I = next(iter(f))

        Encode the last frame of the fringe pattern sequence.

        >>> I = f.encode(frames=-1)
        >>> I = f[-1]

        Encode the first two frames of the fringe pattern sequence.

        >>> I = f.encode(frames=(0, 1))
        >>> I = f[0, 1]
        >>> I = f[:2]

        Create a generator to receive the frames iteratively, i.e. in a lazy manner.

        >>> I = (frame for frame in f)
        """

        t0 = time.perf_counter()

        # check coordinates
        if xi is not None:
            xi = np.array(xi)

            if len(xi) != self.D:
                raise ValueError(f"Number of coordinate matrices != {self.D}.")

            if self.indexing == "ij":
                xi = xi[::-1]

            for d in range(self.D):
                if np.min(xi[d]) < 0:
                    raise ValueError(f"Coordinates contain values < 0.")
                elif np.max(xi[d]) >= self.R[d] + self.x0:
                    raise ValueError(f"Direction {d} contains coordinates > {self.R[d]}.")

        # frames
        if frames is None:
            # frames = np.arange(np.sum(self._N))
            frames = np.arange(self.H * np.sum(self._N))
        else:  # lazy encoding
            try:  # ensure frames is iterable
                iter(frames)
            except TypeError:
                frames = [frames]

            frames = np.array(frames, int).ravel() % self.T

            if self.FDM:
                frames = np.array([np.arange(i * self.D * self.K, (i + 1) * self.D * self.K) for i in frames]).ravel()

            if self.WDM:  # WDM before SDM
                N = 3
                frames = np.array([np.arange(i * N, (i + 1) * N) for i in frames]).ravel()

            if self.SDM:  # WDM before SDM
                len_D0 = np.sum(self._N[0])
                frames = np.array([np.arange(i, i + len_D0 + 1, len_D0) for i in frames]).ravel()

        # modulate
        I = self._modulate(xi, frames, rint)

        # multiplex (reduce number of frames)
        if self.SDM or self.WDM or self.FDM:
            I = self._multiplex(I, rint)

        # apply inscribed circle
        if self.grid in ["polar", "log-polar"]:
            I *= grid.innercirc(self.Y, self.X)[None, :, :, None]

        # colorize (extended averaging)
        if self.H > 1 or np.any(self.h != 255):  # can be used for extended averaging
            I = self._colorize(I, frames)

        logger.info(f"{int(1000 * (time.perf_counter() - t0))}ms")

        return (
            self._simulate(I, PSF=0, system_gain=self.gain, dark_current=self.y0 / self.gain, dark_noise=self.dark)
            if simulate
            else I
        )

    def decode(
        self,
        I: np.ndarray,
        verbose: bool = False,
        despike: bool = False,
        denoise: bool = False,
    ) -> namedtuple:
        r"""Decode fringe patterns.

        Parameters
        ----------
        I : np.ndarray
            Fringe pattern sequence.
            It is reshaped to videoshape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.

            .. note:: It must have been encoded with the same parameters set to the Fringes instance as the encoded one.

        verbose : bool, optional
            If this or the argument `verbose` of the Fringes instance is set to True,
            additional infomation is computed and retuned.
            This includes: phase, residuals, orders, uncertainty, visibility and exposure.

        despike: bool, optional
            If this is set to true, single pixel outliers in the unwrapped phase map are replaced
            by their local neighborhood using a median filter.

        denoise: bool, optional
            If this is set to True, the unwrapped phase map is smoothened
            by a bilateral filter which is edge-preserving.

        Returns
        -------
        brightness : np.ndarray
            Local background signal.

        modulation : np.ndarray
            Local amplitude of the cosine signal.

        registration : np.ndarray
            Decoded coordinates.

            .. note:: The registration is a mapping in the same pixel grid as the camera sensor
              and contains the information where each camera pixel, i.e. each camera sightray,
              was looking at during the fringe pattern acquisition.

        phase : np.ndarray, optional
            Local phase.

        orders : np.ndarray, optional
            Fringe orders.

        residuals : np.ndarray, optional
            Residuals from the optimization-based unwrapping process.

        uncertainty : np.ndarray, optional
            uncertainty of positional decoding in pixel units

        visibility : np.ndarray, optional
            Local visibility (fringe contrast).

        exposure : np.ndarray, optional
            Local exposure (relative average intensity).

        Raises
        ------
        AssertionError
            If the number of frames of `I` and the attribute `T` of the `Fringes` instance don't match.

        Examples
        --------
        >>> import fringes as frng
        >>> f = frng.Fringes()
        >>> I = f.encode()

        >>> A, B, x = f.decode(I)

        >>> A, B, x, p, k, r, u, V, H = f.decode(I, verbose=True)
        """

        t0 = time.perf_counter()

        # get and apply videoshape
        T, Y, X, C = vshape(I).shape  # extract Y, X, C from data as these parameters depend on the used camera
        I = I.reshape((T, Y, X, C))

        # subtract dark signal
        if self.y0 > 0:
            I[I >= self.y0] = I[I >= self.y0] - self.y0
            I[I < self.y0] = 0

        # decolorize (fuse hues/colors) [for gray fringes, color fusion is not performed, but extended averaging is]
        if self.H > 1 or not self._monochrome:
            I = self._decolorize(I)

        # demultiplex
        if self.SDM and 1 not in self.N or self.WDM or self.FDM:
            # todo: if self.SDM and 1 in self.N: Fourier-transform method
            I = self._demultiplex(I)

        # demodulate
        bri, mod, phi, reg, res = self._demodulate(I, verbose)

        # verbose
        if self.verbose or verbose:
            unc, fid, vis, exp = self._verbose_(I, bri, mod, reg)

        # blacken where color value of hue was black
        if self.H > 1 and C == 3:
            idx = np.sum(self.h, axis=0) == 0
            if np.any(idx):  # blacken where color value of hue was black
                bri[..., idx] = 0
                mod[..., idx] = 0
                reg[..., idx] = np.nan
                if self.verbose:
                    res[..., idx] = np.nan
                    unc[..., idx] = np.nan  # self.R / np.sqrt(12)  # todo: circular distribution
                    phi[..., idx] = np.nan
                    fid[..., idx] = np.nan
                    vis[..., idx] = 0
                    exp[..., idx] = 0

        # spatial unwrapping
        if self._ambiguous:
            logger.warning("Unwrapping is not spatially independent and only yields a relative phase map.")
            reg = self._unwrap(reg, bri)  # todo: res if verbose
        else:  # coordiante retransformation
            # todo: tests

            if self.D == 2:
                # todo: swapaxes
                if self.grid == "Cartesian":
                    if self.X >= self.Y:
                        reg[0] += self.X / 2 - 0.5
                        reg[0] %= self.X
                        reg[1] *= -1
                        reg[1] += self.Y / 2 - 0.5
                        reg[1] %= self.X
                    else:
                        reg[0] += self.X / 2 - 0.5
                        reg[0] %= self.Y
                        reg[1] *= -1
                        reg[1] += self.Y / 2 - 0.5
                        reg[1] %= self.Y

                # todo: polar, logpolar

                if self.angle != 0:
                    t = np.deg2rad(-self.angle)

                    if self.angle % 90 == 0:
                        c = np.cos(t)
                        s = np.sin(t)
                        R = np.array([[c, -s], [s, c]])
                        # R = np.matrix([[c, -s], [s, c]])
                        ur = R[0, 0] * reg[0] + R[0, 1] * reg[1]
                        vr = R[1, 0] * reg[0] + R[1, 1] * reg[2]
                        # u = np.dot(uu, R)  # todo: matrix multiplication
                        # v = np.dot(vv, R)
                    else:
                        tan = np.tan(t)
                        ur = reg[0] - reg[1] * np.tan(t)
                        vr = reg[0] + reg[1] / np.tan(t)

                    vv = (reg[1] - reg[0]) / (1 / tan - tan)
                    uu = reg[0] + vv * tan
                    reg = np.stack((uu, vv), axis=0)
                    reg = np.stack((ur, vr), axis=0)

        if despike:
            reg = sp.ndimage.median_filter(reg, size=3, mode="nearest", axes=(1, 2))
            # todo: despike all channels

            # reg[:, -1, -1, ...] = 0
            # reg[:, -10, -10, ...] = 0
            # points = np.arange(Y), np.arange(X)
            # for d in range(self.D):
            #     for c in range(C):
            #         spikes = np.abs(reg[d, :, :, c] - median(reg[d, :, :, c])[0, :, :, 0]) > np.std(reg[d, :, :, c])  # bilateral
            #         values = reg[d, :, :, c]
            #         xi = np.nonzero(spikes)
            #         values[xi] = 0
            #         xi = np.argwhere(spikes)
            #         a = sp.interpolate.interpn(points, values, xi, method="cubic")
            #         reg[d] = sp.interpolate.interpn(points, values, xi, method="cubic")

        if denoise:
            # # blurring due to uncertainty and PSF
            # u = self.u if self.indexing == "ij" else self.u[::-1]  # todo: D = 1, i.e. shape of sigma equal to axes?
            # sigma = np.sqrt(u ** 2 + self.PSF ** 2)
            # reg = sp.ndimage.gaussian_filter(reg, sigma, mode='nearest', axes=(1, 2))
            reg = bilateral(reg, k=3)
            # todo: denoise all channels

        # create named tuple to return
        if self.verbose or verbose:
            dec = namedtuple(
                "decoded",
                "brightness modulation registration residuals uncertainty phase orders visibility exposure",
            )(bri, mod, reg, res, unc, phi, fid, vis, exp)
        else:
            dec = namedtuple("decoded", "brightness modulation registration")(bri, mod, reg)

        logger.info(f"{int(1000 * (time.perf_counter() - t0))}ms")

        return dec

    def _verbose_(self, I: np.ndarray, A: np.ndarray, B: np.ndarray, xi: np.ndarray, lessbits: bool = False):
        """Compute verbose output.

        Parameters
        ----------
        I : np.ndarray
            Fringe pattern sequence.

        A : np.ndarray
            Brightness.

        B : np.ndarray
            Modulation.

        xi : np.ndarray
            Registration.

        lessbits : bool, optional
            The fringe pattern sequence 'I' may contain fewer bits of information than its corresponding dtype.
            This occurs if e.g. a 10 or 12 bit camera is used, for which the corresponding dtype would be 'uint16'.
            If 'lessbits' is True, the number of bits is estimated based on the maximal value of 'I'.
            This affects the value of the exposure 'e'.

        Returns
        -------
        u : np.ndarray
            Uncertainty of positional decoding, in pixel units.

        k : np.ndarray
            Fringe orders.

        V : np.ndarray
            Visibility (fringe contrast).

        e : np.ndarray
            Exposure (relative average intensity).
        """

        Y, X, C = A.shape[1:]

        dark = self.gain * self.dark
        quant = 0 if self.dark > 0 else self.quant
        shot = (
            np.sqrt(self.gain * np.maximum(0, A - self.y0)) if self.gain != 0 else np.zeros_like(A)
        )  # average intensity = brightness
        ui = np.sqrt(dark**2 + quant**2 + shot**2)  # intensity noise
        upi = (
            np.sqrt(2)
            / np.sqrt(self.M)
            / np.sqrt(self._N[:, :, None, None, None])
            / B.reshape(self.D, self.K, Y, X, C)
            * ui[:, None, :, :]
        )  # local phase uncertainties  # todo: M
        uxi = upi / (2 * np.pi) * self._l[:, :, None, None, None]  # local positional uncertainties
        u = np.sqrt(1 / np.sum(1 / uxi**2, axis=1))  # global positional uncertainty
        u = u.astype(np.float32, copy=False)

        k = xi[:, None, :, :, :] // self._l[:, :, None, None, None]
        # p = (xi[:, None, :, :, :] / self._l[:, :, None, None, None] - k) * 2 * np.pi - self.p0
        k = k.astype(int, copy=False).reshape(self.D * self.K, Y, X, C)
        # p = p.astype(np.float32, copy=False).reshape(self.D * self.K, Y, X, C)

        V = B.reshape(self.D, self.K, Y, X, C) / np.maximum(
            A[:, None, :, :, :], np.finfo(np.float_).eps
        )  # avoid division by zero
        V = V.astype(np.float32, copy=False).reshape(self.D * self.K, Y, X, C)

        if I.dtype.kind in "ui":
            if lessbits and np.iinfo(I.dtype).bits > 8:  # data may contain fewer bits of information
                Imax = int(np.ceil(np.log2(I.max())))  # same or next power of two
                Imax += -Imax % 2  # same or next power of two divisible by two
            else:
                Imax = np.iinfo(I.dtype).max
        else:  # float
            Imax = 1
        e = A / Imax
        e = e.astype(np.float32, copy=False)

        return u, k, V, e

    def _unwrap(
        self, phi: np.ndarray, B: np.ndarray, func: str = "ski"
    ) -> (np.ndarray, np.ndarray):  # todo: use B for quality guidance
        """Unwrap phase maps spacially.

        Parameters
        ----------
        phi : np.ndarray
            Phase maps to unwrap spatially, stacked along the first dimension.
            It is reshaped to videoshape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.
            The frames (first dimension) as well the color channels (last dimension)
            are unwrapped separately.

        B : np.ndarray, optional
            Modulation of the decoded phase.
            It is reshaped to videoshape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.

        func : str, optional
            Unwrapping function to use. The default is 'ski'.

            - 'ski': `Scikit-image[1]_ <https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html>`_

            - else: `OpenCV[2]_ <https://docs.opencv.org/4.7.0/df/d3a/group__phase__unwrapping.html>`_

        Returns
        -------
        unwrapped : np.ndarray
            Unwrapped phase maps.

        References
        ----------
        .. [1] `Herráez et al.,
        "Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path",
        Applied Optics,
        2002.
        <https://doi.org/10.1364/AO.41.007437>`_

        .. [2] `Lei et al.,
        "A novel algorithm based on histogram processing of reliability for two-dimensional phase unwrapping",
        Optik - International Journal for Light and Electron Optics,
        2015.
        <https://doi.org/10.1016/j.ijleo.2015.04.070>`_
        """

        t0 = time.perf_counter()

        T, Y, X, C = vshape(phi).shape
        assert T % self.D == 0, "Number of frames of parameters and data don't match."

        func = "ski"  # todo: enable cv2 unwrapping

        if func in "cv2":  # OpenCV unwrapping
            # params = cv2.phase_unwrapping_HistogramPhaseUnwrapping_Params()
            params = cv2.phase_unwrapping.HistogramPhaseUnwrapping.Params()
            params.height = Y
            params.width = X
            # unwrapping_instance = cv2.phase_unwrapping.HistogramPhaseUnwrapping_create(params)
            unwrapping_instance = cv2.phase_unwrapping.HistogramPhaseUnwrapping.create(params)

        reg = np.empty((self.D, Y, X, C), np.float32)
        if self.verbose:
            res = np.empty((self.D, Y, X, C), np.float32)

        for d in range(self.D):
            if self.K == 1:  # todo: self.K[d] == 1
                logger.info(f"Spatial phase unwrapping in 2D{' for each color indepently' if C > 1 else ''}.")
            else:
                logger.info(f"Spatial phase unwrapping in 3D{' for each color indepently' if C > 1 else ''}.")
                func = "ski"  # only ski can unwrap in 3D

            for c in range(C):
                if func in "cv2":  # OpenCV algorithm is usually faster, but can be much slower in noisy images
                    # dtype must be np.float32  # todo: test this
                    if False:  # todo: isinstance(B, np.ndarray) and vshape(B).shape == phi.shape:
                        # todo: unwrap with mask
                        # todo: swapaxes ???
                        SNR = self.B[d, :, :, c] / self.ui
                        upi = np.sqrt(2) / np.sqrt(self.M) / np.sqrt(self._N) / SNR  # local phase uncertainties
                        upin = upi / (2 * np.pi)  # normalized local phase uncertainties
                        uxi = upin * self._l[d]  # local positional uncertainties
                        ux = np.sqrt(
                            1 / np.sum(1 / uxi**2)
                        )  # global phase uncertainty (by inverse variance weighting of uxi)
                        mask = np.astype(ux < 0.5, copy=False)  # todo: which limit?

                        reg[d, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[d, :, :, c], mask)  # todo: test this
                    else:
                        reg[d, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[d, :, :, c])

                    if self.verbose:
                        res[d, :, :, c] = unwrapping_instance.getInverseReliabilityMap()  # todo: test this
                        # todo: res vs. rel
                else:  # Scikit-image algorithm is slower but delivers better results on edges
                    reg[d, :, :, c] = ski.restoration.unwrap_phase(phi[d, :, :, c])

                    if self.verbose:
                        res[d, :, :, c] = np.nan

            regmin = np.min(reg[d])
            if regmin < 0:
                reg[d] -= regmin

        reg *= self._l[:, 0, None, None, None] / (2 * np.pi)

        logger.debug(f"{int(1000 * (time.perf_counter() - t0))}ms")

        return reg  # todo: res if verbose

    # @staticmethod
    # def unwrap(phi: np.ndarray, mask: np.ndarray = None, func: str = "ski") -> np.array:
    #     """Unwrap phase maps spacially.
    #
    #     Parameters
    #     ----------
    #     phi : np.ndarray
    #         Phase maps to unwrap spatially, stacked along the first dimension.
    #         It is reshaped to videoshape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.
    #         The frames (first dimension) as well the color channels (last dimension)
    #         are unwrapped separately.
    #
    #     mask : np.ndarray, optional
    #         Mask image with dtype 'np.uint8' used when some pixels do not hold any phase information.
    #
    #     func : str, optional
    #         Unwrapping function to use. The default is 'ski'.
    #
    #         - 'ski': `Scikit-image <https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html>`_ [1]_
    #
    #         - else: `OpenCV <https://docs.opencv.org/4.7.0/df/d3a/group__phase__unwrapping.html>`_ [2]_
    #
    #     Returns
    #     -------
    #     unwrapped : np.ndarray
    #         Unwrapped phase maps.
    #
    #     References
    #     ----------
    #     .. [1] `Herráez et al.,
    #     "Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path",
    #     Applied Optics,
    #     2002.
    #     <https://doi.org/10.1364/AO.41.007437>`_
    #
    #     .. [2] `Lei et al.,
    #     "A novel algorithm based on histogram processing of reliability for two-dimensional phase unwrapping",
    #     Optik - International Journal for Light and Electron Optics,
    #     2015.
    #     <https://doi.org/10.1016/j.ijleo.2015.04.070>`_
    #     """
    #
    #     # todo: counter + log
    #
    #     T, Y, X, C = vshape(phi).shape
    #     phi = phi.reshape((T, Y, X, C))
    #
    #     if func in "cv2":  # OpenCV unwrapping
    #         # params = cv2.phase_unwrapping_HistogramPhaseUnwrapping_Params()
    #         params = cv2.phase_unwrapping.HistogramPhaseUnwrapping.Params()
    #         params.height = Y
    #         params.width = X
    #         # unwrapping_instance = cv2.phase_unwrapping.HistogramPhaseUnwrapping_create(params)
    #         unwrapping_instance = cv2.phase_unwrapping.HistogramPhaseUnwrapping.create(params)
    #
    #     uwr = np.empty_like(phi)
    #
    #     for t in range(T):
    #         for c in range(C):
    #             if func in "cv2":  # OpenCV algorithm is usually faster, but can be much slower in noisy images
    #                 # dtype must be np.float32  # todo: test this
    #                 if False:  # isinstance(mask, np.ndarray) and vshape(mask).shape == phi.shape:
    #                     # todo: unwrap with mask
    #                     mask = vshape(mask)
    #                     mask = np.astype(mask[t, :, :, c], copy=False)
    #                     uwr[t, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[t, :, :, c], mask)
    #                 else:
    #                     uwr[t, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[t, :, :, c])
    #
    #                 # # dtype of phase must be np.uint8, dtype of shadow_mask np.uint8  # todo: test this
    #                 # reg[d, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[d, :, :, c], shadowMask=shadow_mask)
    #             else:  # Scikit-image algorithm is slower but delivers better results on edges
    #                 uwr[t, :, :, c] = ski.restoration.unwrap_phase(phi[t, :, :, c])
    #
    #     return uwr

    def source(
        self,
        xi: np.ndarray,
        B: np.ndarray = None,
        u: np.ndarray | float = 0,
        dx: float = 1,
        mode: str = "fast",
    ) -> np.ndarray:
        """Source activation heatmap.

        The decoded coordinates (having sub-pixel accuracy)
        are mapped from the camera grid
        to integer positions on the screen grid
        with weights from the modulation.

        This yields the source activation heatmap:
        a grid representing the screen (light source)
        with the pixel values being a relative measure
        of how much a screen (light source) pixel contributed
        to the exposure of the camera sensor.

        The dimensions of the screen are taken from the `Fringes` instance.

        Parameters
        ----------
        xi : np.ndarray
            Registration, i.e. the decoded screen coordinates as seen by the camera.
            It is reshaped to videoshape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.

        B : np.ndarray, optional
            Modulation. Used for weighting.
            It is reshaped to videoshape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.
            If `B` is an array, it must have the same height `Y`, width `X` and color channels `C` as `xi`.
            If `B` is not given, equal weights are used.

        u : np.ndarray | float, optional
            Uncertainty.
            It is reshaped to videoshape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.
            If `u` is an array, it must have the same height `Y`, width `X` and color channels `C` as `xi`.
            Default is zero.

        dx : float, optional
            Size of one camera pixel, projected onto the screen, in units of screen pixels.
            Default is one.

        mode : str, optional
            By default, fast remapping is applied.
            Else, inverse distance weighted remapping is applied,
            which is more precise but also more time-consuming.

        Returns
        -------
        src : np.ndarray
            Source activation heatmap.

        Notes
        -----
        In fact, this is the inverse function of OpenCV’s remap() [3]_.

        References
        ----------
        .. [3] `OpenCV,
               "remap()",
               OpenCV,
               2024.
               <https://docs.opencv.org/4.9.0/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4>`_

        Examples
        --------
        >>> import fringes as frng
        >>> f = frng.Fringes()
        >>> I = f.encode()

        >>> A, B, x = f.decode(I)

        >>> src = f.remap(x, B)
        """
        # B = = cv2.remap(src, xi[0], xi[1], cv2.INTER_LINEAR)  # this is the inverse function of what we want

        t0 = time.perf_counter()

        T, Y, X, C = vshape(xi).shape

        # trim Xi
        xi = xi.reshape((-1, Y, X, C))
        if self.D == 1:
            if self.axis == 0:
                # xi = np.vstack((xi, np.zeros_like(xi)))
                xi = np.concatenate((xi, np.zeros_like(xi)), axis=0)
            else:
                # xi = np.vstack((np.zeros_like(xi), xi))
                xi = np.concatenate((np.zeros_like(xi), xi), axis=0)
        elif self.indexing == "ij":
            xi = xi[::-1]  # returns a view
            # B does not need to be changed because there is only one mean value for each (x, y)-coordinate
            # u does not need to be changed because there is only one value for each (x, y)-coordinate
        valid = (xi[0] <= self.X) * (xi[1] <= self.Y)

        # trim B
        if isinstance(B, np.ndarray):
            assert xi.shape[1:] == B.shape[1:], "'xi' and 'B' have different width, height or number of color channels"
            B = B.reshape((-1, Y, X, C))
            # B = np.max(B, axis=0)
            B = np.mean(B, axis=0)

        if mode == "fast":
            if not isinstance(B, np.ndarray):
                B = np.ones((Y, X, C), np.uint8)

            # trim u
            if isinstance(u, np.ndarray):
                assert (
                    xi.shape[1:] == u.shape[1:]
                ), "'xi' and 'u' have different width, height or number of color channels"
                u = u.reshape((-1, Y, X, C))
                u = np.maximum(u, self.u)

            # initialize
            src = np.zeros((self.Y, self.X, C), np.float32)

            # remap
            # xi = xi[::-1]  # returns a view
            # idx = np.rint(xi).astype(int, copy=False)
            # for c in range(C):  # looping through color channels reduces memory consumption
            #     src[idx[1].ravel(), idx[0].ravel(), c] += B[..., c].ravel()  # ravel() returns a view
            # todo: advanced indexing with nan?
            B[~valid] = 0
            src = _remap(src, xi, B)

            # morphology
            dr = dx / np.sqrt(np.pi)  # mapping radius of sqare (=dx/2) to radius of circle (dr) with same area
            a = 3  # number of standard deviations
            u = self.u if self.indexing == "ij" else self.u[::-1]  # todo: D = 1, i.e. shape of sigma equal to axes?
            R = np.ceil(dr + a * np.sqrt(u**2 + self.PSF**2)).astype(int)
            if any(r > 1 for r in R):
                kernel = np.ones((R[0], R[-1]), np.uint8)  # ensures kernel is 2D even if D == 1
                src = sp.ndimage.grey_dilation(src, structure=kernel)
        else:
            xi = xi[:, valid].reshape(2, -1, C).swapaxes(0, 1)  # n data points of dimension m
            if B is not None:
                B = B[valid].reshape(-1, C)  # n data points of dimension m

            dr = dx / np.sqrt(np.pi)  # mapping radius of sqare (=dx/2) to radius of circle (dr) with same area
            a = 3  # number of standard deviations
            # todo: use u if given as an array (but how?)
            u = np.prod(self.u, axis=0) ** (
                1 / self.D
            )  # geometric mean averages the semi-axes of the uncertainty ellipses
            R = dr + a * np.sqrt(u**2 + self.PSF**2)

            # todo: query tree

            src = np.empty((self.Y, self.X, C), np.float32)
            for c in range(C):
                kdtree = sp.spatial.KDTree(xi[:, :, c])
                for xs in range(self.X):
                    for ys in range(self.Y):
                        i = kdtree.query_ball_point(x=(xs, ys), r=R, p=2)  # , workers=-1)  # list of indices

                        if not i:  # empty list, i.e. no points found within distance R
                            v = 0
                        else:
                            xr = xi[i, 0, c]  # returns a view
                            yr = xi[i, 1, c]  # returns a view
                            d = np.sqrt((xs - xr) ** 2 + (ys - yr) ** 2)  # distance

                            # inverse distance weighting using modified Shepard's method:
                            # https://en.wikipedia.org/wiki/Inverse_distance_weighting
                            w = 1 / (d**2)
                            w[d == 0] = 1
                            # w[d > R] = 0
                            w /= np.sum(w)

                            # todo: use PSF as weights
                            # sum / avg / max of vals within R

                            if B is not None:
                                # v = np.sum(B[i, c] * w)
                                v = np.dot(B[i, c], w)
                            else:
                                v = np.sum(w)

                        src[ys, xs, c] = v

        mx = src.max()
        if mx > 0 and mx != 1:
            src /= mx

        logger.info(f"{int(1000 * (time.perf_counter() - t0))}ms")

        return src.reshape(-1, Y, X, C)

    def brightfield_inverse(self, src: np.ndarray, t: float = 0.1, k: int = 3) -> np.ndarray:
        """Inverse bright-field

        with radiomatric compensation
        assuming a linear response function for display/projector and camera.

        Parameters
        ----------
        src : np.ndarray
            Source activation heatmap.

        t : float
            Threshold within [0, 1].

        k : int
            Edge ength of morphological operator.

        Returns
        -------
        bfi : Inverse bright-field.
        """

        t = np.clip(t, 0, 1)
        t = 0.2
        src[src < t] = np.nan  # 0

        # radiometric compensation
        bfi = 1 / src  #  - 1
        bfi /= np.nanmax(bfi)
        bfi *= self.Imax
        bfi[bfi == np.nan] = 0
        bfi = bfi.astype(self.dtype, copy=False)

        # blurring due to uncertainty and PSF
        # u = self.u if self.indexing == "ij" else self.u[::-1]  # todo: D = 1, i.e. shape of sigma equal to axes?
        # sigma = np.sqrt(u ** 2 + self.PSF ** 2)
        # a = 3
        # k = 1 + dx + 2 * a * sigma  # 2: both sides
        # k = np.ceil(k).astype(int)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        T, Y, X, C = vshape(bfi).shape
        for t in range(T):
            for c in range(C):
                bfi[t, :, :, c] = cv2.dilate(bfi[t, :, :, c], kernel)

        return bfi  # todo: brightfield_inverse

    def brightfield(self, src: np.ndarray, t: float = 0.1, k: int = 3) -> np.ndarray:
        """Bright-field.

        Parameters
        ----------
        src : np.ndarray
            Source activation heatmap.

        t : float
            Threshold within [0, 1].

        Returns
        -------
        bf : Bright-field.
        """

        t = np.clip(t, 0, 1)
        bf = (src > t).astype(self.dtype, copy=False) * self.Imax

        if k:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

            T, Y, X, C = vshape(bf).shape
            for t in range(T):
                for c in range(C):
                    bf[t, :, :, c] = cv2.dilate(bf[t, :, :, c], kernel)

        return bf

    def darkfield(self, src: np.ndarray, t: float = 0.1, k: int = 3) -> np.ndarray:
        """Dark-field.

        Parameters
        ----------
        src : np.ndarray
            Source activation heatmap.

        t : float
            Threshold within [0, 1].

        Returns
        -------
        df : Dark-field.
        """

        t = np.clip(t, 0, 1)
        df = (src <= t).astype(self.dtype, copy=False) * self.Imax

        if k:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

            T, Y, X, C = vshape(df).shape
            for t in range(T):
                for c in range(C):
                    df[t, :, :, c] = cv2.dilate(df[t, :, :, c], kernel)

        return df

    def _simulate(
        self,
        I: np.ndarray,
        PSF: float = 5,
        system_gain: float = 0.038,
        dark_current: float = 3.64 / 0.038,  # [electrons]  # some cameras feature a dark current compensation
        dark_noise: float = 13.7,  # [electrons]
        seed: int = 268664434431581513926327163960690138719,  # secrets.randbits(128)
    ) -> np.ndarray:
        """Simulate the acquisition, i.e. the transmission channel.

        This includes the modulation transfer function (computed from the imaging system's point spread function)
        and intensity noise added by the camera.

        Parameters
        ----------
        I : np.ndarray
            Fringe pattern sequence.

        PSF : float, optional
            Standard deviation of the Point Spread Function, in pixel units.
            The default is 3.

        system_gain : float, optional
            System gain of the digital camera.
            The default is 0.038.

        dark_current : float, optional
            Dark current of the digital camera, in unit electrons.
            The default is ~100.

        dark_noise : float, optional
            Dark noise of the digital camera, in units electrons.
            The default is 13.7.

        seed : int, optional
            A seed to initialize the Random Number Generator.
            It makes the random numbers predictable.
            See `Seeding and Entropy <https://numpy.org/doc/stable/reference/random/bit_generators/index.html#seeding-and-entropy>`_ for more information about seeding.

        Returns
        -------
        I : np.ndarray
            Simulated fringe pattern sequence.
        """

        t0 = time.perf_counter()

        I.shape = vshape(I).shape
        I = I.astype(float, copy=False)

        # # magnification
        # if magnification != 1:  # attention: magnification must be an integer
        #     I = sp.ndimage.uniform_filter(I, size=magnification, mode="reflect", axes=(1, 2))

        # PSF (e.g. defocus)
        if PSF != 0:
            I = sp.ndimage.gaussian_filter(I, sigma=PSF, order=0, mode="reflect", axes=(1, 2))

        if system_gain > 0:
            # random number generator
            rng = np.random.default_rng(seed)

            # add shot noise
            shot = (I - rng.poisson(I)) * np.sqrt(system_gain)
            I += shot
            # s_ = np.std(shot)

            if dark_current > 0 or dark_noise > 0:
                # add dark signal and dark noise
                dark_current_y0 = dark_current * system_gain
                dark_noise_y0 = dark_noise * system_gain
                dark = rng.normal(dark_current_y0, dark_noise_y0, I.shape)
                I += dark
                # d_ = np.std(dark)

        # add spatial nonuniformity
        SNU = 0  # todo: spatial nonuniformity
        I += SNU

        # clip values
        np.clip(I, 0, self.Imax, out=I)

        # quantization noise is added by converting to integer
        I = I.astype(self.dtype, copy=False)

        logger.info(f"{int(1000 * (time.perf_counter() - t0))}ms")

        return I

    def _trim(self, a: np.ndarray) -> np.ndarray:
        """Change `a`.ndim to 2 and limit `a`.shape."""

        if a.ndim == 0:
            a = np.full((self.D, self.K), a)
        elif a.ndim == 1:
            a = np.vstack([a[: self._Kmax] for d in range(self.D)])
        elif a.ndim == 2:
            a = a[: self._Dmax, : self._Kmax]
        else:
            a = a[: self._Dmax, : self._Kmax, ..., -1]

        return a

    def MTF(self, v: float | np.ndarray) -> np.ndarray:
        """Modulation Transfer Function.

        Returns the relative modulation at spatial frequencies `v`.

        Parameters
        ----------
        v: np.ndarray
            Spatial frequencies at which to determine the normalized modulation.

        Returns
        ----------
        B : np.ndarray
            Relative modulation, in the same shape as `v`.

        Notes
        -----
        - If the attribute `Bv` of the Fringes instance is not None, the MTF is interpolated from previous measurements.\n
        - Else, if the attribute `PSF` of the Fringes instance is larger than zero, the MTF is computed from the optical transfer function of the optical system, i.e. as the magnitude of the Fourier-transformed 'Point Spread Function' (PSF).\n
        - Else, it returns ones.
        """

        v = np.array(v, float, copy=False, ndmin=1)

        if self.Bv is not None:  # interpolate from measurement
            # todo: test: unique + LUT
            v_ = v.ravel()
            vu = np.unique(v)
            MTF = sp.interpolate.interp1d(
                v_, self.Bv, kind="cubic", fill_value="extrapolate"
            )  # todo: ...and extrapolated at points outside the data range?
            B = MTF(vu).clip(0, 1).reshape(v.shape)  # interpolate from measured modulation values
            idx = np.argwhere(v == vu)
            B = B[idx].reshape(v.shape)
        elif self.PSF > 0:  # determine MTF from PSF
            B = self.B * np.exp(-2 * (np.pi * self.PSF * v) ** 2)  # todo: fix
            # todo: what is smaller: dl or lv?
            B = 1 - v / (self.L / (self.lmin - 1))  # approximation of [Bothe2008]
        else:
            B = np.ones(v.shape)

        return B

    @property
    def T(self) -> int:
        """Number of frames."""

        T = self.H * np.sum(self._N)

        if self.FDM:  # todo: fractional periods
            T /= self.D * self.K

        if self.SDM:
            T /= self.D

        if self.WDM:
            if np.all(self.N == 3):  # WDM along shifts
                T /= 3
            # elif self.K > self.D:  # WDM along sets todo
            #     a = np.sum(self._N, axis=1)
            #     b = np.max(a)
            #     c = np.ceil(b / 3)
            #     d = int(c)
            #
            #     a2 = np.sum(self._N, axis=0)
            #     b2 = np.max(a2)
            #     c2 = np.ceil(b2 / 2)
            #     d2 = int(c2)
            #
            #     if d < d2:
            #         T = int(np.ceil(np.max(np.sum(self._N, axis=1)) / 3))
            #     else:  # use red and blue
            #         T = int(np.ceil(np.max(np.sum(self._N, axis=0)) / 2))
            # else:  # WDM along directions, use red and blue todo
            #     a = np.sum(self._N, axis=0)
            #     b = np.max(a)
            #     c = np.ceil(b / 2)
            #     d = int(c)
            #     T = int(np.ceil(np.max(np.sum(self._N, axis=1)) / 2))

        return int(T)  # use int() to ensure type is "int" instead of "numpy.core.multiarray.scalar"  # todo: necessary?

    @T.setter
    def T(self, T: int):
        # attention: params may change even if Tnew == Told

        _T = int(min(max(1, T), self._Tmax))

        if _T == 1:  # WDM + SDM todo: FTM?
            if self.grid not in self._grids[:2]:
                logger.error(f"Couldn't set 'T = 1': grid not in {self._grids[:2]}'.")
                return

            self.H = 1
            self.K = 1

            self.FDM = False  # reset FDM before setting N
            self.N = 3  # set N before WDM
            self.WDM = True

            if self.D == 2:
                self.SDM = True
        elif _T == 2:  # WDM
            self.H = 1
            self.D = 2
            self.K = 1

            self.FDM = False  # reset FDM before setting N
            self.SDM = False
            self.N = 3  # set N before WDM
            self.WDM = True
        else:
            # as long as enough shifts are there to compensate for nonlinearities,
            # it doesn't matter if we use more shifts or more sets

            # set boundaries
            Nmin = 3  # minimum number of phase shifts for first set to de demodulated/decoded
            N12 = False
            # todo: N12 = Nmin == 3  # allow N to be in [1, 2] if K >= 2
            # todo: N12 = 1 in self.N or 2 in self.N
            Ngood = 4  # minimum number of phase shifts to obtain good results i.e. reliable results in practice
            Kmax = self.K  # 2  # 3  # todo: which is better: 2 or 3?
            self.FDM = False
            self.SDM = False
            self.WDM = False
            # todo: T == 4 -> no mod
            #  T == 5 -> FDM if _T >= Nmin?

            # try D == 2  # todo: mux
            if _T < 2 * Nmin:
                self.D = 1
            else:
                self.D = 2

            # try to keep hues
            if _T < self.H * self.D * Nmin:
                self.H = _T // (self.D * Nmin)

            while _T % self.H != 0:
                self.H -= 1

            if self.H > 1:
                _T //= self.H

            # try K = Kmax
            if N12:
                K = _T // self.D - (Nmin - 1)  # Nmin - 1 = 2; i.e. 2 more for first set
            else:
                K = _T // (self.D * Nmin)
            self.K = min(K, Kmax)

            # ensure UMR >= R
            if self._ambiguous:
                imin = np.argmin(self._v, axis=0)
                self._v[imin] = 1

            # try N >= Ngood >= Nmin
            N = np.empty([self.D, self.K], int)
            Navg = _T // (self.D * self.K)
            if Navg < Nmin:  # use N12
                N[:, 0] = Nmin
                Nbase = (_T - self.D * Nmin) // (self.D * (self.K - 1))  # attention: if K == 1 -> D == 1
                N[:, 1:] = Nbase
                dT = _T - np.sum(N)
                if dT != 0:
                    k = int(dT // self.D)
                    N[:, 1 : k + 1] += 1
                    if dT % self.D != 0:
                        N[0, k + 1] += 1
            else:
                N[...] = Navg
                dT = _T - np.sum(N)
                if dT != 0:
                    k = int(dT // self.D)
                    N[:, :k] += 1
                    if dT % self.D != 0:
                        d = dT % self.D
                        N[:d, k] += 1
            self.N = N

    @property
    def Y(self) -> int:
        """Height of fringe patterns.
        [Y] = px."""
        return self._Y

    @Y.setter
    def Y(self, Y: int):
        _Y = int(min(max(1, Y), self._Ymax, self._Pmax / self.X))

        if self._Y != _Y:
            self._Y = _Y
            logger.debug(f"{self._Y = }")
            self._UMR = None

            if self._X == self._Y == 1:
                self.D = 1
                self.axis = 0
            elif self._X == 1:
                self.D = 1
                self.axis = 1
            elif self._Y == 1:
                self.D = 1
                self.axis = 0

    @property
    def X(self: int) -> int:
        """Width of fringe patterns.
        [X] = px."""
        return self._X

    @X.setter
    def X(self, X: int):
        _X = int(min(max(1, X), self._Xmax, self._Pmax / self.Y))

        if self._X != _X:
            self._X = _X
            logger.debug(f"{self._X = }")
            self._UMR = None

            if self._X == self._Y == 1:
                self.D = 1
                self.axis = 0
            elif self._X == 1:
                self.D = 1
                self.axis = 1
            elif self._Y == 1:
                self.D = 1
                self.axis = 0

    @property
    def C(self) -> int:
        """Number of color channels."""
        return 3 if self.WDM or not self._monochrome else 1

    @property
    def R(self) -> np.ndarray:
        """Lengths of fringe patterns for each direction.
        [R] = px."""

        R = np.array([self.X, self.Y])

        if self.indexing == "ij":
            R = R[::-1]

        if self.D == 1:
            R = np.atleast_1d(R[self.axis])

        return R

    @property
    def alpha(self) -> float:
        """Factor for extending the coding range `L`."""
        # alpha = float(1 + 2 * x0 / np.max(self.R))
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        _alpha = float(min(max(1, alpha), self._alphamax))

        if self._alpha != _alpha:
            self._alpha = _alpha
            logger.debug(f"{self._alpha = }")
            self._UMR = None

    @property
    def x0(self) -> float:
        """Coordinate offset."""
        # todo: x0.setter -> update alpha
        #  x0max = np.min(self.l)
        return float(np.max(self.R) * (self.alpha - 1) / 2)

    @property
    def L(self) -> float:
        """Coding range.
        [L] = px."""
        return float(np.max(self.R) * self.alpha)

    @property
    def grid(self) -> str:
        """Coordinate system of the fringe patterns.

        The following values can be set:\n
        'image':     The top left corner pixel of the grid is the origin and positive directions are right- resp. downwards.\n
        'Cartesian': The center of grid is the origin and positive directions are right- resp. upwards.\n
        'polar':     The center of grid is the origin and positive directions are clockwise resp. outwards.\n
        'log-polar': The center of grid is the origin and positive directions are clockwise resp. outwards.
        """
        return self._grid

    @grid.setter
    def grid(self, grid: str):
        _grid = str(grid)

        if (self.SDM or self.uwr == "FTM") and self.grid not in self._grids[:2]:
            logger.error(f"Couldn't set 'grid': grid not in {self._grids[:2]}'.")
            return

        if self._grid != _grid and _grid in self._grids:
            self._grid = _grid
            logger.debug(f"{self._grid = }")
            self.SDM = self.SDM

    @property
    def angle(self) -> float:
        """Angle of the coordinate system's principal axis."""
        return self._angle

    @angle.setter
    def angle(self, angle: float):
        _angle = float(np.remainder(angle, 360))  # todo: +- 45

        if self._angle != _angle:
            self._angle = _angle
            logger.debug(f"{self._angle = }")

    @property
    def D(self) -> int:
        """Number of directions."""
        return self._D

    @D.setter
    def D(self, D: int):
        _D = int(min(max(1, D), self._Dmax))

        if self._D > _D:
            self._D = _D
            logger.debug(f"{self._D = }")

            self.N = self._N[: self.D, :]
            self.v = self._v[: self.D, :]
            self.f = self._f[: self.D, :]

            if self._D == self._K == 1:
                self.FDM = False

            self.SDM = False
        elif self._D < _D:
            self._D = _D
            logger.debug(f"{self._D = }")

            self.N = np.append(self._N, np.tile(self._N[-1, :], (_D - self._N.shape[0], 1)), axis=0)
            self.v = np.append(self._v, np.tile(self._v[-1, :], (_D - self._v.shape[0], 1)), axis=0)
            self.f = np.append(self._f, np.tile(self._f[-1, :], (_D - self._f.shape[0], 1)), axis=0)

            self.B = self.B

    @property
    def axis(self) -> int:
        """Axis along which to shift if number of directions equals one.

        Either `0` or `1`."""
        return self._axis

    @axis.setter
    def axis(self, axis: int):
        _axis = int(min(max(0, axis), 1))

        if self._axis != _axis:
            self._axis = _axis
            logger.debug(f"{self._axis = }")
            self._UMR = None

    @property
    def _M(self) -> np.ndarray:
        """Number of averaged intensity samples."""
        M = np.sum(self.h, axis=0) / 255
        return np.atleast_1d(M[0]) if self._monochrome else M

    @property
    def M(self) -> float:  # todo: -> np.ndarray:
        """Number of averaged intensity samples."""
        M = max(1 / 255, np.rint(np.mean(self._M)))  # todo
        return float(M)  # convert Numpy float64 to Python float

    @M.setter
    def M(self, M: float):
        _M = min(max(1 / 255, M), self._Mmax)

        if np.any(self.M != _M):
            if _M < 1:  # fractional part only
                h = np.array([[int(255 * _M % 255) for c in range(3)]], np.uint8)
            elif _M % 1 == 0:  # integer part only
                h = np.array([[255, 255, 255]] * int(_M), np.uint8)
            else:  # integer and fractional part
                h_int = np.array([[255, 255, 255]] * int(_M), np.uint8)
                h_fract = np.array([[int(255 * _M % 255) for c in range(3)]], np.uint8)
                h = np.concatenate((h_int, h_fract))

            self.h = h

    @property
    def H(self) -> int:
        """Number of hues."""
        return self.h.shape[0]

    @H.setter
    def H(self, H: int):
        _H = int(min(max(1, H), self._Hmax))

        # if self.WDM:
        #     logger.error("Couldn't set 'H': WDM is active.")
        #     return

        if self.H != _H:
            if self.WDM:
                self.h = "w" * _H
            elif _H == 1:
                self.h = "w"
            elif _H == 2:
                self.h = "rb"  # todo
            else:
                h = "rgb" * (_H // 3 + 1)
                self.h = h[:_H]

    @property
    def h(self) -> np.ndarray:
        """Hues i.e. colors of fringe patterns.

        Possible values are any sequence of RGB color triples within the interval [0, 255].
        However, black (0, 0, 0) is not allowed.

        The hue values can also be set by assigning any combination of the following characters as a string:\n
        - 'r': red \n
        - 'g': green\n
        - 'b': blue\n
        - 'c': cyan\n
        - 'm': magenta\n
        - 'y': yellow\n
        - 'w': white\n

        Before decoding, repeating hues will be fused by averaging."""
        return self._h

    @h.setter
    def h(self, h: int | tuple[int] | list[int] | np.ndarray | str):
        if isinstance(h, str):
            LUT = {
                "r": [255, 0, 0],
                "g": [0, 255, 0],
                "b": [0, 0, 255],
                "c": [0, 255, 255],
                "m": [255, 0, 255],
                "y": [255, 255, 0],
                "w": [255, 255, 255],
            }
            if set(h.lower()).intersection(LUT.keys()):
                h = [LUT[c] for c in h.lower()]
            elif h == "default":
                h = self.defaults["h"]
            else:
                return

        # make array, clip first and then cast to dtype to avoid integer under-/overflow
        _h = np.array(h).clip(0, 255).astype("uint8", copy=False)

        if not _h.size:  # empty array
            return

        # trim: change shape to (H, 3) or limit shape
        if _h.ndim == 0:
            _h = np.full((self.H, 3), _h)
        elif _h.shape[min(_h.ndim - 1, 1)] < 3:
            _h = np.full((self.H, 3), _h[min(_h.ndim - 1, 1)])
        elif _h.ndim == 1:
            _h = np.vstack([_h[:3] for h in range(self.H)])
        elif _h.ndim == 2:
            _h = _h[: self._Hmax, :3]
        else:
            _h = _h[: self._Hmax, :3, ..., -1]

        if _h.shape[1] == 2:  # C-axis must equal 3
            logger.error("Couldn't set 'h': Only 2 instead of 3 color channels provided.")
            return

        if np.any(np.max(_h, axis=1) == 0):
            logger.error("Didn't set 'h': Black color is not allowed.")
            return

        if self.WDM and not self._monochrome:
            logger.error("Couldn't set 'h': 'WDM' is active, but not all hues are monochromatic.")
            return

        if not np.array_equal(self._h, _h):
            Hold = self.H
            self._h = _h
            logger.debug(f"self._h = {str(self._h).replace(chr(10), ',')}")
            if Hold != _h.shape[0]:
                logger.debug(f"self.H = {_h.shape[0]}")  # computed upon call
            logger.debug(f"{self.M = }")  # computed upon call

    @property
    def TDM(self) -> bool:
        """Temporal division multiplexing."""
        return self.T > 1

    @property
    def SDM(self) -> bool:
        """Spatial division multiplexing.

        The directions D are multiplexed, resulting in a crossed fringe pattern.
        The amplitude B is halved.
        It can only be activated if we have two directions, i.e. D ≡ 2.
        The number of frames T is reduced by the factor 2."""
        return self._SDM

    @SDM.setter
    def SDM(self, SDM: bool):
        _SDM = bool(SDM)

        if _SDM:
            if self.D != 2:
                _SDM = False
                logger.error("Didn't set 'SDM': Pointless as only one dimension exist.")

            if self.grid not in self._grids[:2]:
                _SDM = False
                logger.error(f"Couldn't set 'SDM': grid not in {self._grids[:2]}'.")

            if self.FDM:
                _SDM = False
                logger.error("Couldn't set 'SDM': FDM is active.")

        if self._SDM != _SDM:
            self._SDM = _SDM
            logger.debug(f"{self._SDM = }")
            logger.debug(f"{self.T = }")  # computed upon call

            if self.SDM:
                self.B /= self.D
            else:
                self.B *= self.D

    @property
    def WDM(self) -> bool:
        """Wavelength division multiplexing.

        The shifts are multiplexed into the color channel, resulting in an RGB fringe pattern.
        It can only be activated if all shifts equal 3, i.e. N ≡ 3.
        The number of frames T is reduced by the factor 3."""
        return self._WDM

    @WDM.setter
    def WDM(self, WDM: bool):
        _WDM = bool(WDM)

        if _WDM:
            if not np.all(self.N == 3):
                _WDM = False
                logger.error("Couldn't set 'WDM': At least one Shift != 3.")

            if not self._monochrome:
                _WDM = False
                logger.error("Couldn't set 'WDM': Not all hues are monochromatic.")

            if self.FDM:  # todo: remove this, already covered by N
                _WDM = False
                logger.error("Couldn't set 'WDM': FDM is active.")

        if self._WDM != _WDM:
            self._WDM = _WDM
            logger.debug(f"{self._WDM = }")
            logger.debug(f"{self.T = }")  # computed upon call

    @property
    def FDM(self) -> bool:
        """Frequency division multiplexing.

        The directions D and the sets K are multiplexed, resulting in a crossed fringe pattern if D ≡ 2.
        It can only be activated if D ∨ K > 1 i.e. D * K > 1.
        The amplitude B is reduced by the factor D * K.
        Usually f equals 1 and is essentially only changed if frequency division multiplexing (FDM) is activated:
        Each set per direction receives an individual temporal frequency f,
        which is used in temporal demodulation to distinguish the individual sets.
        A minimal number of shifts Nmin ≥ ⌈ 2 * fmax + 1 ⌉ is required
        to satisfy the sampling theorem and N is updated automatically if necessary.
        If one wants a static pattern, i.e. one that remains congruent when shifted, set static to True.
        """
        return self._FDM

    @FDM.setter
    def FDM(self, FDM: bool):
        _FDM = bool(FDM)

        if _FDM:
            if self.D == self.K == 1:
                _FDM = False
                logger.error("Didn't set 'FDM': Dimensions * Sets = 1, so nothing to multiplex.")

            if self.SDM:
                _FDM = False
                logger.error("Couldn't set 'FDM': SDM is active.")

            if self.WDM:  # todo: remove, already covered by N
                _FDM = False
                logger.error("Couldn't set 'FDM': WDM is active.")

        if self._FDM != _FDM:
            self._FDM = _FDM
            logger.debug(f"{self._FDM = }")
            # self.K = self._K
            self.N = self._N
            self.v = self._v
            if self.FDM:
                self.f = self._f
            else:
                self.f = np.ones((self.D, self.K))

            # keep maximum possible visibility constant
            if self.FDM:
                self.B /= self.D * self.K
            else:
                self.B *= self.D * self.K

    @property
    def static(self) -> bool:
        """Flag for creating static fringes (so they remain congruent when shifted)."""
        return self._static

    @static.setter
    def static(self, static: bool):
        _static = bool(static)

        if self._static != _static:
            self._static = _static
            logger.debug(f"{self._static = }")
            self.v = self._v
            self.f = self._f

    @property
    def K(self) -> int:
        """Number of sets (number of fringe patterns with different spatial frequencies)."""
        return self._K

    @K.setter
    def K(self, K: int):
        # todo: different K for each D: use array of arrays
        # a = np.ones(2)
        # b = np.ones(5)
        # c = np.array([a, b])

        Kmax = (self._Nmax - 1) / 2 / self.D if self.FDM else self._Kmax  # todo: check if necessary
        _K = int(min(max(1, K), Kmax))

        if self._K != _K:
            # V = self.V  # try to keep visibility

            if self._K > _K:  # remove elements
                self._K = _K
                logger.debug(f"{self._K = }")

                self.N = self._N[:, : self.K]
                self.v = self._v[:, : self.K]
                self.f = self._f[:, : self.K]

                if self._D == self._K == 1:
                    self.FDM = False

                # todo: keep modu as high as possible
                if self.FDM:
                    ...
            elif self._K < _K:  # add elements
                self._K = _K
                logger.debug(f"{self._K = }")

                self.N = np.append(
                    self._N, np.tile(self._N[0, 0], (self.D, _K - self._N.shape[1])), axis=1
                )  # don't append N from defaults, this might be in conflict with WDM!
                v = self.L ** (1 / np.arange(self._v.shape[1] + 1, _K + 1))
                self.v = np.append(self._v, np.tile(v, (self.D, 1)), axis=1)
                self.f = np.append(
                    self._f,
                    np.tile(self.defaults["f"][0, 0], (self.D, _K - self._f.shape[1])),
                    axis=1,
                )

            # self.V = V  # try to keep visibility
            self.V = 1  # maximize visibility

    @property
    def _Nmin(self) -> int:
        """Minimum number of shifts to (uniformly) sample temporal frequencies.

        Per direction at least one set with N ≥ 3 is necessary
        to solve for the three unknowns brightness A, modulation B and coordinate xi."""
        if self.FDM:
            Nmin = int(np.ceil(2 * self.f.max() + 1))  # sampling theorem
            # todo: 2 * D * K + 1 -> fractional periods if static
        else:
            Nmin = 3  # todo: 1 -> use old decoder
        return Nmin

    @property
    def N(self) -> np.ndarray:
        """Number of phase shifts."""
        if self.D == 1 or len(np.unique(self._N, axis=0)) == 1:  # sets in directions are identical
            N = self._N[0]  # 1D
        else:
            N = self._N  # 2D
        return N

    @N.setter
    def N(self, N: int | tuple[int] | list[int] | np.ndarray):
        _N = np.array(N, int).clip(self._Nmin, self._Nmax)  # make array, cast to dtype, clip

        if not _N.size:  # empty array
            return

        _N = self._trim(_N)

        if np.all(_N == 1) and _N.shape[1] == 1:  # any
            pass  # FTM
        elif np.any(_N <= 2):
            for d in range(self.D):
                if not any(_N[d] >= 3):
                    i = np.argmax(_N[d])  # np.argmin(_N[d])
                    _N[d, i] = 3

        if self.WDM and not np.all(_N == 3):
            logger.error("Couldn't set 'N': At least one Shift != 3.")
            return

        if self.FDM and not np.all(_N == _N[0, 0]):
            # _N = np.tile(self._Nmin, _N.shape)
            _N = np.tile(_N[0, 0], _N.shape)

        if not np.array_equal(self._N, _N):
            self._N = _N
            logger.debug(f"self._N = {str(self._N).replace(chr(10), ',')}")
            self._UMR = None
            self.D, self.K = self._N.shape
            logger.debug(f"{self.T = }")

    @property
    def lmin(self) -> float:
        """Minimum resolvable wavelength.
        [lmin] = px."""

        # don't use self._fmax, else circular loop
        if self.FDM and self.static:
            fmax = min((self._Nmin - 1) / 2, self.L / self._lmin)
        else:
            fmax = (self._Nmin - 1) / 2
        return min(self._lmin, self.L / fmax) if self.FDM and self.static else self._lmin

    @lmin.setter
    def lmin(self, lmin: float):
        _lmin = float(max(self._lminmin, lmin))

        if self._lmin != _lmin:
            self._lmin = _lmin
            logger.debug(f"{self._lmin = }")
            logger.debug(f"{self.vmax = }")  # computed upon call
            self.l = self.l  # l triggers v

    @property
    def lopt(self) -> float:
        """Optimal wavelength for minimal decoding uncertainty.
        [lopt] = px."""
        return self.L / self.vopt

    @property
    def l(self) -> np.ndarray:
        """Wavelengths of fringe periods.
        [l] = px.

        When L changes, v is kept constant and only l is changed."""
        return self.L / self.v

    @l.setter
    def l(self, l: int | float | tuple[int | float] | list[int | float] | np.ndarray | str):
        if isinstance(l, str):
            if "," in l:
                l = np.fromstring(l, sep=",")  # todo: N, v, f
            elif l == "optimal":
                lmin = int(np.ceil(self.lmin))
                lmax = int(
                    np.ceil(
                        max(
                            self.L / lmin,  # todo: only if B differ slightly
                            self.lmin,
                            min(self.lopt, self.L),
                            np.sqrt(self.L),
                        )
                    )
                )

                if lmin == lmax and lmax < self.L:
                    lmax += 1

                if lmax < self.L and not sympy.isprime(lmax):
                    lmax = sympy.ntheory.generate.nextprime(lmax, 1)  # ensures lcm(a, lmax) >= L for all a >= lmin

                n = lmax - lmin + 1

                l_ = np.array([lmin])
                l_max = lmin + 1
                lcm = l_
                while lcm < self.L:
                    lcm_new = np.lcm(lcm, l_max)
                    if lcm_new > lcm:
                        l_ = np.append(l_, l_max)
                        lcm = lcm_new
                    l_max += 1
                K = min(len(l_), self.K)

                C = sp.special.comb(n, K, exact=True, repetition=True)  # number of unique combinations
                combos = it.combinations_with_replacement(range(lmin, lmax + 1), K)

                # B = int(np.ceil(np.log2(lmax - 1) / 8)) * 8  # number of bits required to store integers up to lmax
                # combos = np.fromiter(combos, np.dtype((f"uint{B}", K)), C)

                kroot = self.L ** (1 / K)
                if self.lmin <= kroot:
                    lcombos = np.array(
                        [l for l in combos if np.any(np.array([l]) > kroot) and np.lcm.reduce(l) >= self.L]
                    )
                else:
                    lcombos = np.array([l for l in combos if np.lcm.reduce(l) >= self.L])

                # lcombos = filter(lcombos, K, self.L, lmin)

                # idx = np.argmax(np.sum(1 / lcombos**2, axis=1))
                # l = lcombos[idx]

                v = self.L / lcombos
                B = self.MTF(v)
                var = 1 / self.M / self.N * lcombos**2 / B**2  # todo: D, M
                idx = np.argmax(np.sum(1 / var, axis=1))

                l = lcombos[idx]

                if K < self.K:
                    l = np.concatenate((np.full(self.K - K, lmin), l))

                # while lmax < self.L and np.gcd(lmin, lmax) != 1:
                #     lmax += 1  # maximum number of iterations? = min(next prime after lmax - lmax, max(0, L - lmax, ))
                # l = np.array([lmin] * (self.K - 1) + [lmax])

                # vmax = int(max(1 if self.K == 1 else 2, self.vmax))  # todo: ripples from int()
                # v = np.array([vmax] * (self.K - 1) + [vmax - 1])  # two consecutive numbers are always coprime
                # lv = self.L / v
                # lv = np.maximum(self._lmin, np.minimum(lv, self.L))
                #
                # idx = np.argmax((np.sum(1 / (l ** 2), axis=0), np.sum(1 / (lv ** 2), axis=0)))
                # l = l if idx == 0 else lv
                # print("l" if idx == 0 else "v")
            elif l == "close":
                lmin = int(max(np.ceil(self.lmin), self.L ** (1 / self.K) - self.K))
                l = lmin + np.arange(self.K)
                while np.lcm.reduce(l) < self.L:
                    l += 1
            elif l == "small":
                lmin = int(np.ceil(self.lmin))
                lmax = int(np.ceil(self.L ** (1 / self.K)))  # wavelengths are around kth root of self.L

                if self.K >= 2:
                    lmax += 1

                    if self.K >= 3:
                        lmax += 1

                        if lmax % 2 == 0:  # kth root was even
                            lmax += 1

                        if self.K > 3:
                            ith = self.K - 3
                            lmax = sympy.ntheory.generate.nextprime(lmax, ith)

                if lmin > lmax or lmax - lmin + 1 <= self.K:
                    l = lmin + np.arange(self.K)
                else:
                    lmax = max(
                        lmin, min(lmax, int(np.ceil(self.L)))
                    )  # max in outer condition ensures lmax >= lmin even if L < lmin
                    if lmin == lmax and lmax < self.L:
                        lmax += 1  # ensures lmin and lmax differ so that lcm(l) >= L

                    n = lmax - lmin + 1
                    K = min(self.K, n)  # ensures K <= n
                    C = sp.special.comb(n, K, exact=True, repetition=False)  # number of unique combinations
                    combos = np.array(
                        [
                            c
                            for c in it.combinations(range(lmin, lmax + 1), K)
                            if np.any(np.array([c]) > self.L ** (1 / self.K)) and np.lcm.reduce(c) >= self.L
                        ]
                    )

                    idx = np.argmax(np.sum(1 / combos**2, axis=1))
                    l = combos[idx]

                if K < self.K:
                    l = np.concatenate((l, np.arange(l.max() + 1, l.max() + 1 + self.K - K)))
            elif l == "exponential":
                l = np.concatenate(([np.inf], np.geomspace(self.L, self.lmin, self.K - 1)))
            elif l == "linear":
                l = np.concatenate(([np.inf], np.linspace(self.L, self.lmin, self.K - 1)))
            else:
                return

        self._UMR = None  # to be safe
        _l = np.array(l, float)
        self.v = self.L / np.array(l, float)

    @property
    def _l(self) -> np.ndarray:  # kept for backwards compatibility with fringes-GUI
        """Wavelengths of fringe periods.
        [l] = px.

        When L changes, v is kept constant and only l is changed."""
        return self.L / self._v

    @property
    def vmax(self) -> float:
        """Maximum resolvable spatial frequency."""
        return self.L / self.lmin

    @property
    def vopt(self) -> float:
        """Optimal spatial frequency for minimal decoding uncertainty."""

        if self.Bv is not None:  # interpolate from measurement
            v = np.arange(1, self.vmax + 1)
            B = self.MTF(v)
            N = self._N.ravel()[np.argpartition(self._N.ravel(), int(self._N.size // 2))[int(self._N.size // 2)]]
            var = 1 / self.M / N / (v**2) / B**2  # todo: D, M
            idx = np.argmax(np.sum(1 / var, axis=1))
            vopt = v[idx]
        elif self.PSF > 0:  # determine from PSF
            vopt_ = 1 / (2 * np.pi * self.PSF)  # todo
            lopt = 1 / vopt_
            vopt = self.L / lopt
            vopt = self.vmax / 2  # approximation [Bothe2008]
        else:
            vopt = int(self.vmax)

        return vopt

    @property
    def v(self) -> np.ndarray:
        """Spatial frequencies (number of periods/fringes across maximum coding length)."""
        if self.D == 1 or len(np.unique(self._v, axis=0)) == 1:  # sets in directions are identical
            v = self._v[0]  # 1D
        else:
            v = self._v  # 2D
        return v

    @v.setter
    def v(self, v: int | float | tuple[int | float] | list[int | float] | np.ndarray | str):
        if isinstance(v, str):
            if v == "optimal":
                # |{v}| = 2
                vmax = int(max(1 if self.K == 1 else 2, self.vopt))
                v = np.array([vmax] * (self.K - 1) + [vmax - 1])  # two consecutive numbers are always coprime

                # # # |{v}| = K
                # vmax = int(max(self.K, self.vopt))
                # v = vmax - np.arange(self.K)
            elif v == "exponential":
                # K = int(np.ceil(np.log2(self.vmax))) + 1  # + 1: 2 ** 0 = 1
                v = np.concatenate(([0], np.geomspace(1, self.vmax, self.K - 1)))
            elif v == "linear":
                v = np.concatenate(([0], np.linspace(1, self.vmax, self.K - 1)))
            else:
                return

        _v = np.array(v, float).clip(0, self.vmax)  # make array, cast to dtype, clip

        if not _v.size:  # empty array
            return

        _v = self._trim(_v)

        if self.FDM:
            if self.static:
                if (
                    _v.size != self.D * self.K
                    or not np.all(_v % 1 == 0)
                    or not np.lcm.reduce(_v.astype(int, copy=False).ravel()) == np.prod(_v)  # todo: equal ggt = 1 ?
                ):  # todo: allow coprimes?!
                    n = min(10, self.vmax // 2)
                    ith = self.D * self.K
                    pmax = sympy.ntheory.generate.nextprime(n, ith + 1)
                    p = np.array(list(sympy.ntheory.generate.primerange(n, pmax + 1)))[:ith]  # primes
                    p = [p[-i // 2] if i % 2 else p[i // 2] for i in range(len(p))]  # resort primes
                    _v = np.sort(np.array(p, float).reshape((self.D, self.K)), axis=1)  # resort primes
                    logger.warning(
                        f"Periods were not coprime. " f"Changing values to {str(_v.round(3)).replace(chr(10), ',')}."
                    )
            # else:
            #     vmax = (self._Nmax - 1) / 2 > _v
            #     _v = np.minimum(_v, vmax)

        if not np.array_equal(self._v, _v):
            self._v = _v
            logger.debug(f"self.v = {str(self._v.round(3)).replace(chr(10), ',')}")
            logger.debug(f"self.l = {str(self._l.round(3)).replace(chr(10), ',')}")
            self._UMR = None
            self.D, self.K = self._v.shape
            self.f = self._f

    @property
    def _fmax(self):
        """Maximum temporal frequency (maximum number of periods to shift over)."""
        return min((self._Nmin - 1) / 2, self.vmax) if self.FDM and self.static else (self._Nmin - 1) / 2

    @property
    def f(self) -> np.ndarray:
        """Temporal frequency (number of periods to shift over)."""
        if self.D == 1 or len(np.unique(self._f, axis=0)) == 1:  # sets in directions are identical
            f = self._f[0]  # 1D
        else:
            f = self._f  # 2D
        return f

    @f.setter
    def f(self, f: int | float | tuple[int | float] | list[int | float] | np.ndarray | str):
        _f = np.array(f, float).clip(-self._fmax, self._fmax)  # make array, cast to dtype, clip

        if not _f.size:  # empty array
            return

        _f = self._trim(_f)

        D = min(_f.shape[0], self._N.shape[0])
        K = min(_f.shape[1], self._N.shape[1])
        if np.any(_f[:D, :K] % self._N[:D, :K] == 0):
            # _f = np.ones(_f.shape)
            _f[:D, :K][_f[:D, :K] % self._N[:D, :K] == 0] = 1

        if self.FDM:
            if self.static:
                _f = self._v  # periods to shift over = one full revolution
            else:
                if (
                    _f.shape != (self.D, self.K)
                    or not np.all(i % 1 == 0 for i in _f)
                    or len(np.unique(np.abs(_f))) < _f.size
                ):  # assure _f are int and absolute values of _f differ
                    _f = np.arange(1, self.D * self.K + 1, dtype=float).reshape((self.D, self.K))

        if 0 not in _f and not np.array_equal(self._f, _f):
            self._f = _f
            logger.debug(f"self._f = {str((self._f * (-1 if self.reverse else 1)).round(3)).replace(chr(10), ',')}")
            self.D, self.K = self._f.shape
            self.N = self._N  # todo: remove if fractional periods is implemented, log warning

    @property
    def p0(self) -> float:
        """Phase offset within interval (-2pi, +2pi).

        It can be used to e.g. let the fringe patterns start (at the origin) with a gray value of zero.
        """
        return self._p0

    @p0.setter
    def p0(self, p0: float):
        _p0 = float(np.abs(p0) % (2 * np.pi) * np.sign(p0))

        if self._p0 != _p0:
            self._p0 = _p0
            logger.debug(f"self._p0 = {self._p0 / np.pi} PI")

    @property
    def Bv(self) -> np.ndarray:
        """Modulation at spatial frequencies `v`.

        The modulation values are determined from a measurement."""
        return self._Bv

    @Bv.setter
    def Bv(self, B: np.ndarray):
        if B is None:
            self._Bv = None
            logger.debug(f"self.Bv = {self._Bv}")
            return

        _B = np.array(np.maximum(0, B), float)

        _B.shape = T, Y, X, C = vshape(_B).shape

        assert T == self.D * self.K

        _B = _B.reshape(self.D, self.K, self.Y, self.X, self.C)

        # filter
        _B = np.nanmedian(_B, axis=-1)  # filter along color axis
        # _B = np.nanmedian(_B, axis=(2, 3))  # filter along spatial axes
        _B = np.nanquantile(_B, 0.9, axis=(2, 3))  # filter along spatial axes

        #  normalize (only relative weights are important)
        Bmax = np.iinfo(_B.dtype).max if _B.dtype.kind in "ui" else 1
        _B /= Bmax
        _B[np.isnan(_B)] = 0

        _Bv = np.vstack(_B, self._v)

        if not np.array_equal(self._Bv, _Bv):
            self._Bv = _B
            logger.debug(f"self.Bv = {str(self.Bv.round(3)).replace(chr(10), ',')}")

    @property
    def _monochrome(self) -> bool:
        """True if all hues are monochromatic, i.e. the RGB values are identical for each hue."""
        return all(len(set(h)) == 1 for h in self.h)

    @property
    def _ambiguous(self) -> bool:
        """True if unambiguous measument range is larger than the screen length."""
        return bool(np.any(self.UMR < self.R * self.alpha))

    @property
    def indexing(self) -> str:
        """Indexing convention.

        Cartesian indexing `xy` (the default) will index the row first,
        while matrix indexing `ij` will index the colum first.
        """
        return self._indexing

    @indexing.setter
    def indexing(self, indexing):
        _indexing = str(indexing)

        if self._indexing != _indexing and _indexing in self._indexings:
            self._indexing = _indexing
            logger.debug(f"{self._indexing = }")
            self._UMR = None

    @property
    def reverse(self) -> bool:
        """Flag for shifting fringes in reverse direction."""
        return self._reverse

    @reverse.setter
    def reverse(self, reverse: bool):
        _reverse = bool(reverse)

        if self._reverse != _reverse:
            self._reverse = _reverse
            logger.debug(f"{self._reverse = }")
            logger.debug(f"self._f = {str((self._f * (-1 if self.reverse else 1)).round(3)).replace(chr(10), ',')}")

    @property
    def verbose(self) -> bool:
        """Flag for additionally returning intermediate and verbose results:\n
        - phase maps\n
        - residuals\n
        - fringe orders\n
        - visibility\n
        - exposure
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        _verbose = bool(verbose)

        if self._verbose != _verbose:
            self._verbose = _verbose
            logger.debug(f"{self._verbose = }")

    @property
    def mode(self) -> str:
        """Mode for remapping.

        The following values can be set:\n
        - 'fast'\n
        - 'precise'
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        _mode = str(mode)

        if self._mode != _mode and _mode in self._modes:
            self._mode = _mode
            logger.debug(f"{self._mode = }")

    @property
    def uwr(self) -> str:
        """Phase unwrapping method."""

        if self.K == 1 and np.all(self._N == 1) and self.grid in self._grids[:2]:
            # todo: v >> 1, i.e. l ~ 8
            uwr = "FTM"  # Fourier-transform method
        elif self.K == np.all(self.v <= 1):
            uwr = "none"
        elif self._ambiguous:
            uwr = "spatial"
        else:
            uwr = "temporal"

        return uwr

    @property
    def gamma(self) -> float:
        """Gamma correction factor used to compensate nonlinearities of the display response curve."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float):
        _gamma = float(min(max(0, gamma), self._gammamax))

        if self._gamma != _gamma and _gamma != 0:
            self._gamma = _gamma
            logger.debug(f"{self._gamma = }")

    @property
    def shape(self) -> tuple[int]:
        """Shape of fringe pattern sequence in video shape (frames, height, with, color channels)."""
        return self.T, self.Y, self.X, self.C

    # @property
    # def ndim(self) -> int:
    #     """Number of dimensions."""
    #     return len(self.shape)

    @property
    def size(self) -> np.uint64:
        """Number of pixels of fringe pattern sequence (frames * height * width * color channels)."""
        return float(np.prod(self.shape, dtype=np.uint64))  # using uint64 prevents integer overflow

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by fringe pattern sequence.

        Does not include memory consumed by non-element attributes of the array object.
        """
        # return self.size * self.dtype.itemsize
        return self.T * self.Y * self.X * self.C * self.dtype.itemsize

    @property
    def dtype(self) -> np.dtype:
        """Data type.

        The following values can be set:\n
        - 'bool'\n
        - 'uint8'\n
        - 'uint16'\n
        - 'float32'\n
        - 'float64'\n
        """
        return np.dtype(self._dtype)  # this is a hotfix for setting _dtype directly as a str in init

    @dtype.setter
    def dtype(self, dtype: np.dtype | str):
        _dtype = np.dtype(dtype)

        if self._dtype != _dtype and str(_dtype) in self._dtypes:
            self._dtype = _dtype
            logger.debug(f"{self._dtype = }")
            logger.debug(f"self.A = {self.A}")
            logger.debug(f"self.B = {self.B}")

    @property
    def Imax(self) -> int:
        """Maximum gray value."""
        return np.iinfo(self.dtype).max if self.dtype.kind in "ui" else 1

    @property
    def _Amin(self):
        """Minimum bias."""
        return self.B / self._Vmax

    @property
    def _Amax(self):
        """Maximum bias."""
        return self.Imax - self._Amin

    @property
    def A(self) -> float:
        """Bias."""
        return self.Imax * self.beta

    @A.setter
    def A(self, A: float):
        _A = float(min(max(self._Amin, A), self._Amax))

        if self.A != _A:
            self.beta = _A / self.Imax
            logger.debug(f"{self.A = }")

    @property
    def _Bmax(self):
        """Maximum amplitude."""
        return min(self.A, self.Imax - self.A) * self._Vmax

    @property
    def B(self) -> float:
        """Amplitude."""
        return self.A * self.V

    @B.setter
    def B(self, B: float):
        _B = float(min(max(0, B), self._Bmax))

        if self.B != _B:  # and _B != 0:
            self.V = _B / self.A
            logger.debug(f"{self.B = }")

    @property
    def _betamax(self):
        """Maximum relative bias (exposure)."""
        return 1 / (1 + self.V)

    @property
    def beta(self) -> float:
        """Relative bias (exposure), i.e. relative mean intensity ∈ [0, 1]."""
        return self._beta

    @beta.setter
    def beta(self, beta) -> float:
        _beta = float(min(max(0, beta), self._betamax))

        if self._beta != _beta:
            self._beta = _beta
            logger.debug(f"{self.beta = }")

    @property
    def _Vmax(self):
        """Maximum visibility."""
        if self.FDM:
            return 1 / (self.D * self.K)
        elif self.SDM:
            return 1 / self.D
        else:
            return 1

    @property
    def V(self) -> float:
        """Fringe visibility (fringe contrast) ∈ [0, 1]."""
        return self._V

    @V.setter
    def V(self, V: float):
        _V = float(min(max(0, V), self._Vmax))

        if self._V != _V:
            self._V = _V
            logger.debug(f"{self.V = }")

    @property
    def Vmin(self) -> float:
        """Minimum visibility for measurement to be valid."""
        return self._Vmin

    @Vmin.setter
    def Vmin(self, Vmin: float):
        _Vmin = float(min(max(0, Vmin), 1))

        if self._Vmin != _Vmin:
            self._Vmin = _Vmin
            logger.debug(f"{self._Vmin = }")

    @property
    def umax(self) -> float:
        """Standard deviation of maximum uncertainty for measurement to be valid.
        [umax] = px."""
        return self._umax

    @umax.setter
    def umax(self, umax: float):
        _umax = float(min(max(0, umax), self.L))  # todo: L / 2 due to circular distribution  # todo: R.max() / 2

        if self._umax != _umax:
            self._umax = _umax
            logger.debug(f"{self._umax = }")

    # @property
    # def r(self) -> int:
    #     """Number of quantization bits."""
    #     return 1 if self.dtype.kind in "b" else np.iinfo(
    #         self.dtype).bits if self.dtype.kind in "ui" else 10 ** np.finfo(self.dtype).precision

    # @property
    # def Q(self) -> float:
    #     """Number of quantization levels."""
    #     return 2 ** self.r

    @property
    def q(self) -> float:
        """Quantization step size."""
        # LSB
        return 1.0 if self.dtype.kind in "uib" else np.finfo(self.dtype).resolution

    @property
    def quant(self) -> float:
        """Quantization noise (standard deviation).
        [quant] = DN."""
        return float(self.q / np.sqrt(12))  # convert Numpy float64 to Python float

    @property
    def dark(self) -> float:
        """Dark noise of the digital camera (standard deviation).
        [dark] = electrons."""
        return self._dark

    @dark.setter
    def dark(self, dark: float):
        _dark = float(min(max(0, dark), np.sqrt(self.Imax)))

        # _dark = max(_dark, 0.49)  # todo: temporal noise is dominated by quantization noise ->

        _dark = max(0, _dark - self.quant)  # correct for quantization noise contained in dark noise measurement

        if self._dark != _dark:
            self._dark = _dark
            logger.debug(f"{self._dark = }")

    @property
    def shot(self) -> float:
        """Shot noise of digital camera (standard deviation).
        [shot] = DN."""
        A = self.A * self.MTF(0)  # todo: Paper from (!) includes B
        return np.sqrt(self.gain * max(0, A - self.y0)) if self.gain != 0 else 0  # average intensity is bias

    @property
    def gain(self) -> float:
        """Overall system gain of digital camera.
        [gain] = DN / electrons."""
        return self._gain

    @gain.setter
    def gain(self, gain: float):
        _gain = min(max(0, gain), 1)

        if self._gain != _gain:
            self._gain = _gain
            logger.debug(f"{self._gain = }")

    @property
    def PSF(self) -> float:  # todo: magnification?
        """Standard deviation of Point Spread Function for defocus.
        [PSF] = px."""
        return self._PSF

    @PSF.setter
    def PSF(self, PSF):
        _PSF = float(max(0, PSF))

        if self._PSF != _PSF:
            self._PSF = _PSF
            logger.debug(f"{self._PSF = }")

    @property
    def y0(self) -> float:
        """Dark signal.
        [y0] = DN"""
        return self._y0

    @y0.setter
    def y0(self, y0: int | float):
        _y0 = float(min(max(0, y0), self.Imax))

        if self._y0 != _y0:
            self._y0 = _y0
            logger.debug(f"{self._y0 = }")

    @property
    def UMR(self) -> np.ndarray:
        """Unambiguous measurement range.
        [UMR] = px

        The coding is only unique within the interval [0, UMR); after that it repeats itself.

        The UMR is derived from l and v:\n
        - If l ∈ ℕ, UMR = lcm(l), with lcm being the least common multiple.\n
        - Else, if v ∈ ℕ, UMR = L / gcd(v), with gcd being the greatest common divisor.\n
        - Else, if l ∨ v ∈ ℚ, lcm resp. gcd are extended to rational numbers.\n
        - Else, if l ∧ v ∈ ℝ ∖ ℚ, UMR = prod(l), with prod being the product operator.
        """

        if self._UMR is not None:  # cache
            return self._UMR  # todo: resetting cache doesn't work for some reasons...fixed? -> test!

        # precision = np.finfo("float64").precision - 1
        precision = 13
        atol = 10**-precision

        UMR = np.empty(self.D)
        for d in range(self.D):
            l = self._l[d]  # .copy()
            v = self._v[d]  # .copy()

            if 1 in self._N[d]:  # here, in TPU twice the combinations have to be tried
                # todo: test if valid
                l[self._N[d] == 1] /= 2
                v[self._N[d] == 1] *= 2

            if 0 in v:  # equivalently: np.inf in l
                l = l[v != 0]
                v = v[v != 0]

            if len(l) == 0 or len(v) == 0:
                UMR[d] = 1  # one since we can only control discrete pixels
                break

            if np.all(l % 1 == 0):  # all l are integers
                UMR[d] = np.lcm.reduce(l.astype(int, copy=False))
            elif np.all(v % 1 == 0):  # all v are integers
                UMR[d] = self.L / np.gcd.reduce(v.astype(int, copy=False))
            elif np.allclose(l, np.rint(l), rtol=0, atol=atol):  # all l are integers within tolerance
                UMR[d] = np.lcm.reduce(np.rint(l).astype(int, copy=False))
            elif np.allclose(v, np.rint(v), rtol=0, atol=atol):  # all v are integers within tolerance
                UMR[d] = self.L / np.gcd.reduce(np.rint(v).astype(int, copy=False))
            else:
                # mutual divisibility test
                for i in range(self.K - 1):
                    for j in range(i + 1, self.K):
                        if l[i] % l[j] < atol or 1 - atol < l[i] % l[j] < 1:
                            l[j] = 1
                        elif l[j] % l[i] < atol or 1 - atol < l[j] % l[i] < 1:
                            l[i] = 1
                v = v[l != 1]
                l = l[l != 1]

                # number of decimals
                Dl = max(str(i)[::-1].find(".") for i in l)
                Dv = max(str(i)[::-1].find(".") for i in v)

                # estimate whether elements are rational or irrational
                if Dl < precision or Dv < precision:  # rational numbers without integers
                    # extend lcm/gcd to rational numbers

                    if Dl <= Dv:
                        ls = l * 10**Dl  # wavelengths scaled
                        UMR[d] = np.lcm.reduce(ls.astype(int, copy=False)) / 10**Dl
                        logger.debug("Extended lcm to rational numbers.")
                    else:
                        vs = v * 10**Dv  # spatial frequencies scaled
                        UMR[d] = self.L / (np.gcd.reduce(vs.astype(int, copy=False)) / 10**Dv)
                        logger.debug("Extended gcd to rational numbers.")
                else:  # irrational numbers or rational numbers with more digits than "precision"
                    UMR[d] = np.prod(l)

        self._UMR = UMR
        logger.debug(f"self.UMR = {str(self._UMR)}")

        if self._ambiguous:
            logger.warning("UMR < R. Unwrapping will not be spatially independent and only yield a relative phase map.")

        return self._UMR

    @property
    def eta(self) -> float:
        """Coding efficiency."""
        eta = self.R / self.UMR
        eta[self.UMR < self.R] = 0
        return eta

    @property
    def ui(self) -> float:
        """Intensity noise."""
        quant = 0 if self.dark > 0 else self.quant
        quant = self.quant
        ui = np.sqrt(self.gain**2 * self.dark**2 + quant**2 + self.shot**2)
        return ui

    @property
    def upi(self) -> float:
        """Phase uncertainty.
        [upi] = rad"""
        B = self.B * self.MTF(self._v)
        SNR = B / self.ui
        upi = np.sqrt(2) / np.sqrt(self.M) / np.sqrt(self._N) / SNR  # local phase uncertainties
        return upi

    @property
    def u(self) -> np.ndarray:
        """Uncertainty of measurement (standard deviation).
        [u] = px.

        It is based on the phase noise model
        and propagated through the unwrapping process and the phase fusion."""

        upin = self.upi / (2 * np.pi)  # normalized local phase uncertainties
        uxi = upin * self._l  # local positional uncertainties
        ux = np.sqrt(1 / np.sum(1 / uxi**2, axis=1))  # global positional uncertainty (by inverse variance weighting)
        # todo: factor out L ?!
        return ux

    @property
    def SNR(self) -> np.ndarray:
        """Signal-to-noise ratio of the phase shift coding.

        It is a masure of how many points can be distinguished within the screen length [0, R)
        """
        return self.R / self.u

    @property
    def SNRdB(self) -> np.ndarray:
        """Signal-to-noise ratio.
        [SNRdB] = dB."""
        return 20 * np.log10(self.SNR)

    @property
    def DR(self) -> np.ndarray:
        """Dynamic range of the phase shift coding.

        It is a measure of how many points can be distinguished within the unambiguousmeasurement range [0, UMR).
        """
        return self.UMR / self.u

    @property
    def DRdB(self) -> np.ndarray:
        """Dynamic range. [DRdB] = dB."""
        return 20 * np.log10(self.DR)

    @property
    def efficiency(self) -> np.ndarray:
        """Coding efficiency."""
        return self.SNR / self.T  # todo: data transfer rate?

    @property
    def params(self) -> dict:
        """Base parameters required for en- & decoding fringe patterns.

        This contains all property objects of the class which have a setter method,
        i.e. are (usually) not derived from others.
        """

        params = {}
        for p in sorted(dir(self)):  # sorted() ensures setting params in the right order in the setter method
            if p != "params":
                if isinstance(getattr(type(self), p, None), property) and getattr(type(self), p, None).fset is not None:
                    if isinstance(getattr(self, p, None), np.ndarray):
                        params[p] = getattr(self, p, None).tolist()
                    elif isinstance(getattr(self, p, None), np.dtype):
                        params[p] = str(getattr(self, p, None))
                    else:
                        params[p] = getattr(self, p, None)
        return params

    @params.setter
    def params(self, params: dict = {}):
        # iterating self.params ensures that only properies with a setter method are set
        params_old = self.params.copy()
        for k, v in params_old.items():
            if k in params and k != "T":
                setattr(self, k, params[k])

        for k, v in self.params.items():
            if k in params:
                if k in "Nlvf" and np.array(v).ndim != np.array(params[k]).ndim:
                    if not np.array_equal(v, params[k][0]):
                        break
                else:
                    if not np.array_equal(v, params[k]):
                        break
        else:  # else clause executes only after the loop completes normally, i.e. did not encounter a break
            return

        logger.warning(f"'{k}' got overwritten by interdependencies. Choose a consistent parameter set.")
        self.params = params_old

    types: dict = dict(sorted(__init__.__annotations__.items()))
    """Types of 'params' values."""

    defaults: dict = dict(sorted(__init__.__kwdefaults__.items()))
    """Default values for `params`."""

    glossary: dict = {
        k: v.__doc__ for k, v in sorted(vars().items()) if not k.startswith("_") and isinstance(v, property)
    }
    """Glossary."""
    # and v.__doc__

    # restrict instance attributes to the ones listed here
    # (commend the next line out to prevent this)
    __slots__ = tuple("_" + k for k in defaults.keys() if k not in "HMTlAB") + ("logger", "_UMR", "_t")

    # continuing the class docstring following the NumPy style guide:
    # https://numpydoc.readthedocs.io/en/latest/format.html#class-docstring
    __doc__ += "\n\nParameters\n----------"
    __doc__ += "\n*args : iterable\n    Non-keyword arguments are explicitly discarded."
    __doc__ += "\n    Only keyword arguments are considered."
    for __k, __v in __init__.__kwdefaults__.items():
        # do this in for loop instead of list comprehension,
        # because for the latter, names in class scope are not accessible
        __doc__ += f"\n{__k} : {types[__k]}, default = {__v}\n    {glossary[__k].splitlines()[0]}"
    # __doc__ += "\n\nAttributes\n----------"
    # for __k, __v in sorted(vars().items()):
    #     # do this in for loop instead of list comprehension,
    #     # because for the latter, names in class scope are not accessible
    #     if __k not in defaults and not __k.startswith("_") and isinstance(__v, property):
    #         __doc__ += f"\n{__k} : {type(__v)}"
    # __doc__ += "\ntypes : dict"
    # __doc__ += "\ndefaults : dict"
    # __doc__ += "\nglossary : dict"
    # __doc__ += f"\nlogger : Logger, default = logger.getLogger({__qualname__})"  # todo
    # __doc__ += "\n    https://docs.python.org/3/library/logger.html#logger-objects"
    # __doc__ += "\n\nMethods\n-------"
    # __doc__ += f"\nencode\n    {encode.__doc__.splitlines()[0]}"
    # __doc__ += f"\ndecode\n    {decode.__doc__.splitlines()[0]}"
    del __k, __v


if __name__ == "__main__":
    import fringes as frng  # import the fringes package

    # configure logging
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s.%(funcName)s(): %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger("fringes")
    logger.setLevel("INFO")

    f = frng.Fringes()      # instantiate the Fringes class

    f.X = 1920              # set width of the fringe patterns
    f.Y = 1080              # set height of the fringe patterns
    f.K = 2                 # set number of sets
    f.N = 4                 # set number of shifts
    f.v = [9, 10]           # set spatial frequencies

    f.T                     # get number of frames

    I = f.encode()          # encode fringe patterns

    A, B, X = f.decode(I)   # decode fringe patterns
