import os
import logging as lg
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
import asdf
from si_prefix import si_format as si

from .util import vshape, bilateral, median
from . import grid
from .decoder import decode  # todo: fast_decode i.e. fast_unwrap

import inspect


class Fringes:
    """Phase shifting algorithms for encoding and decoding sinusoidal fringe patterns."""

    # value limits
    _Hmax = 101  # this is arbitrary
    _Dmax = 2  # max 2 dimensions
    _Kmax = 101  # this is arbitrary, but must be < 128 when deploying spatial or frequency multiplexing @ uint8
    _Nmax = 1001  # this is arbitrary; more is better but the improvement scales with sqrt(N); @FDM: > 2 * fmax + 1
    _Mmax = 101  # this is arbitrary; more is better but the improvement scales with sqrt(M)
    # _Pmax: int = 35651584  # ~8K i.e. max luma picture size of h264, h265, h266 video codecs as of 2022; todo: update
    _Pmax = 2 ** 30  # 2^30 = 1,073,741,824 i.e. default size   limit of imread() in OpenCV
    _Xmax = 2 ** 20  # 2^20 = 1,048,576     i.e. default width  limit of imread() in OpenCV
    _Ymax = 2 ** 20  # 2^20 = 1,048,576     i.e. default height limit of imread() in OpenCV
    _Lmax = 2 ** 20  # 2^20 = 1,048,576     i.e. default height limit of imread() in OpenCV
    _Tmax = _Hmax * _Dmax * _Kmax * _Nmax
    _alphamax = 2
    _gammamax = 3  # most screens have a gamma of ~2.2
    # _lminmin = 2  # l == 2 only if offset != pi / 2 + 2pi*k, best if offset == pi + 2pi*k with k is positive integer
    #            also l <= 2 yields errors in SPU: phase jumps = 2PI / lmin >= np.pi
    _lminmin = 3  # l >= 3 yields sufficient modulation theoretically
    # _lminmin = 8  # l >= 8 yields sufficient modulation practically [Liu2014]

    # allowed values; take care to only use immutable types!
    _grids = (
        "image",
        "Cartesian",
        "polar",
        "log-polar"
    )
    _modes = (
        "fast",
        "precise",
        "robust"
              )
    _dtypes = (
        "bool",
        "uint8",
        "uint16",
        # 'uint32',  # integer overflow in pyqtgraph -> replace line 528 of ImageItem.py with:
        # 'uint64',  # bins = self._xp.arange(mn, mx + 1.01 * step, step, dtype="uint64")
        # "float16",  # numba doesn't handle float16, also most algorithms convert float16 to float32 anyway
        "float32",
        "float64",
    )
    _loader = {
        ".json": json.load,
        ".yaml": yaml.safe_load,
        ".toml": toml.load,
        ".asdf": asdf.open,
    }

    # default values are defined here; take care to only use immutable types!
    def __init__(self,
                 Y: int = 1200,
                 X: int = 1920,
                 H: int = 1,  # H is inferred from h
                 M: float = 1.,  # M is inferred from h
                 D: int = 2,
                 K: int = 3,
                 T: int = 24,  # T is inferred
                 N: tuple | np.ndarray = np.array([[4, 4, 4], [4, 4, 4]], int),
                 l: tuple | np.ndarray = 1920 / np.array([[13, 7, 89], [13, 7, 89]], float),
                 v: tuple | np.ndarray = np.array([[13, 7, 89], [13, 7, 89]], float),
                 f: tuple | np.ndarray = np.array([[1, 1, 1], [1, 1, 1]], float),
                 h: tuple | np.ndarray = np.array([[255, 255, 255]], int),
                 o: float = np.pi,
                 gamma: float = 1.,
                 A: float = 255 / 2,  # i.e. Imax / 2 @ uint8
                 B: float = 255 / 2,  # i.e. Imax / 2 @ uint8
                 beta: float = .5,  # beta is inferred from A and Imax
                 V: float = 1.,  # V is inferred from A and B
                 alpha: float = 1.,
                 dtype: str | np.dtype = "uint8",
                 grid: str = "image",
                 angle: float = 0.,
                 axis: int = 0,
                 SDM: bool = False,
                 WDM: bool = False,
                 FDM: bool = False,
                 static: bool = False,
                 lmin: float = 8.,
                 reverse: bool = False,
                 verbose: bool = False,
                 mode: str = "fast",
                 Vmin: float = 0.,
                 esat: float = np.inf,
                 dark: float = 0.,
                 gain: float = 0,
                 ) -> None:
        # given values which are in defaults but are not identical to them
        given = {k: v for k, v in sorted(locals().items()) if
                 k in self.defaults and not np.array_equal(v, self.defaults[k])}

        # logger
        self.logger = lg.getLogger(self.__class__.__name__)  # todo: give each logger its own instance name, via id() ?
        self.logger.setLevel("INFO")
        if not self.logger.hasHandlers():
            formatter = lg.Formatter("%(asctime)s %(levelname)-8s %(name)7s.%(funcName)-11s: %(message)s")
            handler = lg.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # set default values
        self._UMR = None  # used for caching
        for k, v in self.defaults.items():
            if k not in ["H", "M", "T", "beta", "V"]:  # these properties are inferred from others
                setattr(self, f"_{k}", v)  # define private variables from where the properties get their value from

        # set given values
        self.params = given

        self.UMR  # property 'UMR' logs warning if necessary

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Encode fringe patterns."""
        return self.encode(*args, **kwargs)

    def __getitem__(self, t: int | tuple | list) -> np.ndarray:
        """Single frame(s) of fringe pattern sequence."""
        return self.encode(t=t)

    def __iter__(self):
        self._t = 0
        return self

    def __next__(self) -> np.ndarray:
        if self._t < self.T:
            I = self.encode(frame=self._t)
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

    def __str__(self) -> str:
        return "Fringes"

    def __repr__(self) -> str:
        return f"{self.params}"

    def load(self, fname: str = os.path.join(os.path.expanduser("~"), ".fringes.yaml")) -> dict:
        """Load a parameter set from a file.
        Supported file formats are: *.json, *.yaml, *.toml, *.asdf.
        If `fname` is not provided, the file `.fringes.yaml` within the user home directory is tried to load.
        The parameters are only loaded if the config file provides the section `fringes`.
        """

        if not os.path.isfile(fname):
            self.logger.error(f"File '{fname}' does not exist.")
            return

        ext = os.path.splitext(fname)[-1]
        if ext == ".asdf":
            with asdf.open(fname) as f:
                p = f.copy()
        else:
            with open(fname, "r") as f:
                if ext == ".json":
                    p = json.load(f)
                elif ext == ".yaml":
                    p = yaml.safe_load(f)
                elif ext == ".toml":
                    p = toml.load(f)
                else:
                    self.logger.error(f"Unknown file type '{ext}'.")
                    return {}

        if "fringes" in p:
            params = p["fringes"]
            self.params = params

            self.logger.info(f"Loaded parameters from '{fname}'.")

            return params
        else:
            self.logger.error(f"No 'fringes' section in file '{fname}'.")
            return {}

    def save(self, fname: str = os.path.join(os.path.expanduser("~"), ".fringes.yaml")) -> None:
        """Save the parameters to a file.
        Supported file formats are: *.json, *.yaml, *.toml, *.asdf.
        If `fname` is not provided, the parameters are saved to
        the file `.fringes.yaml` within the user home directory.
        Within the file, the parameters are written to the section `fringes`."""

        if not os.path.isdir(os.path.dirname(fname)):
            self.logger.warning(f"File directory does not exist.")
            return
        elif os.path.splitext(fname)[-1] not in self._loader.keys():
            self.logger.warning(f"File extension is unknown. Must be one of {self._loaders.keys()}")
            return

        ext = os.path.splitext(fname)[-1]
        if ext == ".asdf":
            asdf.AsdfFile({"fringes": self.params}).write_to(fname)
        else:
            with open(fname, "w") as f:
                if ext == ".json":
                    json.dump({"fringes": self.params}, f, indent=4)
                elif ext == ".yaml":
                    yaml.dump({"fringes": self.params}, f)
                elif ext == ".toml":
                    toml.dump({"fringes": self.params}, f)

        self.logger.info(f"Saved parameters to {fname}.")

    def reset(self) -> None:
        """Reset parameters to defaults."""

        self.params = self.defaults

        self.logger.info("Reset parameters to defaults.")

    def optimize(self, T: int = 24) -> None:
        """Optimize the parameters based on:
         T: Given number of frames.
         lmin: Minimum resolvable wavelength.
         L: Length of fringe patterns."""
        self.h = "w"
        self.T = T
        self.v = "auto"
        self.logger.info("Optimized parameters.")

    @staticmethod
    def gamma_auto_correct(I: np.ndarray) -> np.ndarray:
        """Automatically estimate and apply the gamma correction factor
        to linearize the display/camera response curve."""

        # normalize to [0, 1]
        Imax = np.iinfo(I.dtype).max if I.dtype.kind in "ui" else 1
        J = I / Imax

        # estimate gamma correction factor
        med = np.nanmedian(J)  # median is a robust estimator for the mean
        gamma = np.log(med) / np.log(0.5)
        inv_gamma = 1 / gamma

        # apply inverse gamma
        # table = np.array([((g / self.Imax) ** invGamma) * self.Imax for g in range(self.Imax + 1)], self.dtype)
        # J = cv2.LUT(J, table)
        J **= inv_gamma
        J *= Imax

        return J

    def mtf2vmax(self, B: np.ndarray) -> np.ndarray:
        """Compute the normalized modulation transfer function at spatial frequencies `v` from decoded modulation `B`
        and use the result to set the optimal lmin: minimum resolvable wavelength.
        """
        # filter
        # MTF = np.median(B, axis=(1, 2))
        MTF = np.quantile(B, 0.1, axis=(1, 2))

        # sort

        #  normalize (only relative weights are important)
        # MTF = MTF.reshape((self.D, -1, B.shape[-1]))  # MTF per direction
        MTF = MTF.reshape((self.D, self.K, -1))  # MTF per direction
        MTF /= np.nanmax(MTF, axis=(1, 2))[:, None, None]
        MTF[np.isnan(MTF)] = 0

        C = MTF.shape[-1]
        # v = np.arange(self.vmax)  # todo: limit range; what happens beyond endpoints?
        v = np.arange(self.v.min(), self.v.max() + 1)
        vmax = np.empty((self.D, C))
        for d in range(self.D):
            for c in range(C):
                interp = sp.interpolate.interp1d(self._v[d], MTF[d, ..., c], kind="cubic", axis=-1)
                mtf = interp(v)
                idx = np.argmin(mtf >= 0.5)  # index of last element where MTF >= 0.5, cf. [Bothe2008]
                if idx > 0:
                    idx -= 1
                vmax[d, c] = v[idx]

        self.lmin = self.L / vmax.max()

        self.logger.info(f"vmax = {self.vmax}")

        return self.vmax

    def deinterlace(self, I: np.ndarray) -> np.ndarray:
        """Deinterlace fringe patterns acquired with a line scan camera
        while each frame has been displayed and captured
        while the object has been moved by one pixel."""

        t0 = time.perf_counter()

        T, Y, X, C = vshape(I).shape
        assert T * Y % self.T == 0, "Number of frames of parameters and data don't match."
        # I = I.reshape((T * Y, X, C))  # concatenate

        self.logger.info(f"{si(time.perf_counter() - t0)}s")

        return I.reshape((-1, self.T, X, C)).swapaxes(0, 1)

    def coordinates(self) -> np.ndarray:
        """Generate the coordinate matrices of the coordinate system defined in `grid`."""

        t0 = time.perf_counter()

        sys = "img" if self.grid == "image" else "cart" if self.grid == "Cartesian" else "pol" if self.grid == "polar" else "logpol"
        xi = np.array(getattr(grid, sys)(self.Y, self.X, self.angle))[self.axis if self.D == 1 else ...]

        if self.grid in ["polar", "log-polar"]:
            xi *= self.L

        self.logger.info(f"{si(time.perf_counter() - t0)}s")

        return xi.reshape((self.D, self.Y, self.X, 1))

    def _modulate(self, frame: tuple = None, rint: bool = True) -> np.ndarray:  # todo: rint = False as default? influence on residuals?
        """Encode base fringe patterns by spatio-temporal modulation."""

        # dd = [d for d in range(self.D) for k in range(self.K) for n in range(self._N[d, k])]
        # kk = [k for d in range(self.D) for k in range(self.K) for n in range(self._N[d, k])]
        # NN = [self._N[d, k] for d in range(self.D) for k in range(self.K) for n in range(self._N[d, k])]
        # nn = [n for d in range(self.D) for k in range(self.K) for n in range(self._N[d, k])]
        # nn = np.array([n for N in self._N[dd, kk] for n in range(N)])
        # return = [(d, k) for d in range(self.D) for k in range(self.K) for n in range(self._N[d, k])][t]

        # Nacc = np.cumsum(self._N.ravel()).reshape((self.D, self.K))
        # return np.unravel_index(np.argmax(t < Nacc), Nacc.shape)  # t < Nacc  # argmax @bool: first element of True

        # Nacc = 0
        # for d in range(self.D):
        #     for k in range(self.K):
        #         Nacc += self._N[d, k]
        #         if t < Nacc:
        #             return d, k

        t0 = time.perf_counter()

        if frame is None:
            frame = np.arange(np.sum(self._N))

        try:  # ensure frame is iterable
            T = len(frame)
        except TypeError:
            T = 1
            frame = [frame]
        frames = np.array(frame)

        if self.grid != "image" or self.angle != 0:
            xi = self.coordinates()[..., 0]
            assert xi.ndim == 3, "uv-coordinates are not three-dimensional with shape (D, Y, X)"
        else:
            xi = None

        # is_pure = all(c == 0 or c == 255 for h in self.h for c in h)
        # dtype = np.dtype("float64") if self.SDM or self.FDM or not is_pure else self.dtype
        dtype = np.dtype("float64") if self.SDM or self.FDM else self.dtype

        I = np.empty([T, self.Y, self.X], dtype)
        idx = 0
        f = 0
        for d in range(self.D):
            if xi is None and self.grid == "image":
                x = np.arange(self.R[d]) / self.L  # gets broadcasted
                if self.D == 2:
                    if d == 0:
                        x = x[None, :]
                    else:
                        x = x[:, None]
                else:
                    if self.axis == 0:
                        x = x[None, :]
                    else:
                        x = x[:, None]
            else:
                x = xi[d] / self.L

            for i in range(self.K):
                k = 2 * np.pi * self._v[d, i]
                w = 2 * np.pi * self._f[d, i]

                if self.reverse:
                    w *= -1

                for n in range(self._N[d, i]):
                    if f in frames:
                        t = n / 4 if self._N[d, i] == 2 else n / self._N[d, i]

                        cos = np.cos(k * x - w * t - self.o)

                        if self.gamma == 1:
                            val = self.A + self.B * cos
                        else:
                            val = (self.A / self.Imax + self.B / self.Imax * cos) ** self.gamma * self.Imax

                        val = self.Imax * (self.beta * (1 + self.V * np.cos(k * x - w * t - self.o))) ** self.gamma

                        if dtype.kind in "ui" and rint:
                            # val += .5
                            np.rint(val, out=val)
                            I[idx] = val.astype(dtype, copy=False)  # todo: astype necessary?
                        elif dtype.kind in "b":
                            I[idx] = val >= .5
                        else:
                            I[idx] = val
                        idx += 1
                    f += 1

        # dt = np.float64 if self.SDM or self.FDM or np.any((self.h != 0) * (self.h != 255)) else self.dtype
        # I = encode(dt, np.ones(1), frames, self._N, self._v, self._f * (-1 if self.reverse else 1), self.o, self.Y, self.X, 1, self.axis, self.gamma, self.A, self.B)

        self.logger.debug(f"{si(time.perf_counter() - t0)}s")

        return I.reshape(-1, self.Y, self.X, 1)

    def _demodulate(self, I: np.ndarray, verbose: bool = False) -> tuple:
        """Decode base fringe patterns by spatio-temporal demodulation."""

        t0 = time.perf_counter()

        _f = self._f * (-1 if self.reverse else 1)

        # parse
        T, Y, X, C = vshape(I).shape  # extract Y, X, C from data as these parameters depend on used camera
        I = I.reshape((T, Y, X, C))

        # if self.FDM:
        #    c = np.fft.rfft(I, axis=0) / T  # todo: hfft
        #
        #    # if np.any(self._N > 2 * self.D * self.K):  # 2 * np.abs(_f).max() + 1:
        #    #     i = np.append(np.zeros(1, int), self._f.flatten().astype(int, copy=False))  # add offset
        #    #     c = c[i]
        #
        #    a = abs(c)
        #    # i = np.argsort(a[1:], axis=0)[::-1]  # indices of frequencies, sorted by their magnitude
        #    phi = -np.angle(c * np.exp(-1j * (self.o - np.pi)))[_f.flatten().astype(int, copy=False)]  # todo: why offset - PI???

        if self.SSB:
            # todo: make passband symmetrical around carrier frequency?
            if self.D == 2:
                fx = np.fft.fftshift(np.fft.fftfreq(X))  # todo: hfft
                fy = np.fft.fftshift(np.fft.fftfreq(Y))
                fxx, fyy = np.meshgrid(fx, fy)
                mx = np.abs(fxx) > np.abs(
                    fyy)  # mask for x-frequencies  # todo: make left and right borders round (symmetrical around base band)
                my = np.abs(fxx) < np.abs(
                    fyy)  # mask for y-frequencies  # todo: make lower and upper borders round (symmetrical around base band)

                W = 100  # assume window width for filtering out baseband
                W = min(max(3, W), min(X, Y) / 20)  # clip to ensure plausible value
                a = int(min(max(0, W), X / 4) + .5)  # todo: find good upper cut off frequency
                # a = X // 4
                mx[:, :a] = 0  # remove high frequencies
                b = int(X / 2 - W / 2 + .5)
                mx[:, b:] = 0  # remove baseband and positive frequencies

                H = 100  # assume window height for filtering out baseband
                H = min(max(3, H), min(X, Y) / 20)  # clip to ensure plausible value
                c = int(min(max(0, H), Y / 4) + .5)  # todo: find good upper cut off frequency
                # c = Y // 4
                my[:c, :] = 0  # remove high frequencies
                d = int(Y / 2 - H / 2 + .5)
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
                frr = np.sqrt(fxx ** 2 + fyy ** 2)  # todo: normalization of both directions

                mr = frr <= L / 2  # ensure same sampling in all directions
                W = 10
                W = min(max(1, W / 2), L / 20)
                mr[frr < W] = 0  # remove baseband
                mr[frr > L / 4] = 0  # remove too high frequencies

                mh = np.empty([Y, X])
                mh[:, :X // 2] = 1
                mh[:, X // 2:] = 0

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
            if I.dtype == np.float16:
                I = I.astype(np.float32, copy=False)  # numba does not support float16

            Vmin = self.Vmin
            if self.SDM:
                Vmin /= self.D
            elif self.FDM:
                Vmin /= self.D * self.K

            if self.mode == "fast":
                SQNR = self.B / self.quant
                Vmin = max(Vmin, 1 / SQNR)
                r = min(self.u, 1.)  # todo: 0.5 or self.u
            else:
                r = 0.
            r = 0.  # todo

            # """Weights for inverse variance weighting."""
            # var = 1 / self._N / self.M / np.sqrt(
            #     self._v)  # / B has to be done in decoder as it depends on captured scene
            # w = 1 / var
            # for d in range(self.D):
            #     w[d] /= np.sum(w[d])

            phi, bri, mod, reg, res, fid = decode(
                I, self._N, self._v, _f, self.R, self.alpha, self.o, r, self.mode, Vmin, self.verbose or verbose
            )

        self.logger.debug(f"{si(time.perf_counter() - t0)}s")

        return phi, bri, mod, reg, res, fid

    def _multiplex(self, I: np.ndarray, rint: bool = True) -> np.ndarray:
        """Multiplex fringe patterns."""

        t0 = time.perf_counter()

        if self.WDM:
            assert not self.FDM
            assert self._ismono
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
                self.logger.warning("Decoding might be disturbed.")

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

        self.logger.debug(f"{si(time.perf_counter() - t0)}s")

        return I

    def _demultiplex(self, I: np.ndarray) -> np.ndarray:
        """Demultiplexing fringe patterns."""

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

        self.logger.debug(f"{si(time.perf_counter() - t0)}s")

        return I

    def _colorize(self, I: np.ndarray, frame: tuple = None) -> np.ndarray:
        """Colorize fringe patterns."""

        t0 = time.perf_counter()

        T = len(frame) if frame is not None else self.T
        J = np.empty((T, self.Y, self.X, self.C), self.dtype)

        Th = I.shape[0]  # number of frames for each hue

        if frame is not None:
            hues = [int(t // np.sum(self._N)) for t in frame]
            t = 0

        for h in range(self.H):
            if frame is None:
                for c in range(self.C):
                    cj = c if self.WDM else 0
                    if self.h[h, c] == 0:  # uib -> uib, f -> f
                        J[h * Th: (h + 1) * Th, ..., c] = 0
                    elif self.h[h, c] == 255 and J.dtype == self.dtype:  # uib -> uib, f -> f
                        J[h * Th: (h + 1) * Th, ..., c] = I[..., cj]
                    elif self.dtype.kind in "uib":  # f -> uib
                        J[h * Th: (h + 1) * Th, ..., c] = np.rint(I[..., cj] * (self.h[h, c] / 255)).astype(self.dtype,
                                                                                                            copy=False)
                    else:  # f -> f
                        J[h * Th: (h + 1) * Th, ..., c] = I[..., cj] * (self.h[h, c] / 255)
            elif h in hues:  # i.e. frame is not None and h in hues
                for c in range(self.C):
                    cj = c if self.WDM else 0
                    if self.h[h, c] == 0:  # uib -> uib, f -> f
                        J[t, ..., c] = 0
                    elif self.h[h, c] == 255 and J.dtype == self.dtype:  # uib -> uib, f -> f
                        J[t, ..., c] = I[t, ..., cj]
                    elif self.dtype.kind in "uib":  # f -> uib
                        J[t, ..., c] = np.rint(I[t, ..., cj] * (self.h[h, c] / 255)).astype(self.dtype, copy=False)
                    else:  # f -> f
                        J[t, ..., c] = I[t, ..., cj] * (self.h[h, c] / 255)
                t += 1

        self.logger.debug(f"{si(time.perf_counter() - t0)}s")

        return J

    def _decolorize(self, I: np.ndarray) -> np.ndarray:
        """Decolorize fringe patterns i.e. fuse hues/colors."""

        t0 = time.perf_counter()

        T, Y, X, C = vshape(I).shape
        I = I.reshape((self.H, T // self.H, Y, X, C))  # returns a view

        is_base = all(np.sum(h != 0) == 1 for h in self.h)  # every hue consists of only one of the RGB base colors
        is_single = all(np.sum(c != 0) <= 1 for c in self.h.T)  # each RGB color exists only once
        is_single_and_value = all(
            np.sum(c != 0) == 1 for c in self.h.T)  # each RGB color exists only once and is natural
        is_equal_or_zero = len(set(self.h[self.h != 0])) == 1  # all colors are equal or zero
        # if is_base and is_single or is_single_and_value: no averaging necessary
        # if is_equal_or_zero: no weights necessary
        if self.H == 3 and C in [1, 3] and is_base and is_single and is_equal_or_zero:  # pure RGB colors
            I = np.moveaxis(I, 0, -2)  # returns a view

            idx = np.argmax(self.h, axis=1)
            if np.array_equal(idx, [0, 1, 2]):  # RGB
                # I = I[..., :, :]
                pass
            elif np.array_equal(idx, [0, 2, 1]):  # RBG
                I = I[..., 0::-1, :]
            elif np.array_equal(idx, [1, 2, 0]):  # GBR
                I = I[..., 1:1:, :]
            elif np.array_equal(idx, [1, 0, 2]):  # GRB
                I = I[..., 1:1:-1, :]
            elif np.array_equal(idx, [2, 1, 0]):  # BGR
                I = I[..., 2::-1, :]
            elif np.array_equal(idx, [2, 0, 1]):  # BRG
                I = I[..., 2:2:-1, :]

            if C == 1:
                I = I[..., 0]
            elif C == 3:
                I = np.diagonal(I, axis1=-2, axis2=-1)  # returns a view
        elif self.H == 2 and C == 3 and is_single_and_value and is_equal_or_zero:  # todo: C == 3 avoids CMY colors appearing twice as bright as RGB colors (as it is with mono cameras) assuming spectral bands don't overlap
            I = np.moveaxis(I, 0, -2)  # returns a view
            idx = self.h != 0
            I = I[..., idx]  # advanced indexing doesn't return a view
        else:  # fuse colors by weighted averaging
            w = self.h / np.sum(self.h, axis=0)  # normalized weights
            # w[np.isnan(w)] = 0

            if np.all((w == 0) | (w == 1)):  # todo: when does this happen?
                w = w.astype(bool, copy=False)  # multiplying with bool preserves dtype
                dtype = I.dtype  # without this, np.sum chooses a dtype which can hold the theoretical maximal sum
            else:
                dtype = float  # without this, np.sum chooses a dtype which can hold the theoretical maximal sum

            I = np.sum(I * w[:, None, None, None, :], axis=0, dtype=dtype)

        self.logger.debug(f"{si(time.perf_counter() - t0)}s")

        return I

    def encode(self, frame: int | tuple | list = None, rint: bool = True) -> np.ndarray:
        """Encode fringe patterns."""

        t0 = time.perf_counter()

        # check UMR
        if self._isambiguous:
            self.logger.warning(
                "UMR < R. Unwrapping will not be spatially independent and only yield a relative phase map.")

        if frame is not None:  # lazy encoding
            try:  # ensure frame is iterable
                len(frame)
            except:
                frame = [frame]

            frames = np.array(frame).ravel()

            frame = frames % np.sum(self._N)

            if self.FDM:
                frame = np.array([np.arange(t * self.D * self.K, (t + 1) * self.D * self.K) for t in frame]).ravel()

            if self.WDM:  # WDM before SDM
                N = 3
                frame = np.array([np.arange(t * N, (t + 1) * N) for t in frame]).ravel()

            if self.SDM:  # WDM before SDM
                EN0 = np.sum(self._N[0])
                frame = np.array([np.arange(t, t + EN0 + 1, EN0) for t in frame]).ravel()
        else:
            frames = None

        # modulate
        I = self._modulate(frame, rint)

        # multiplex (reduce number of frames)
        if self.SDM or self.WDM or self.FDM:
            I = self._multiplex(I, rint)

        # apply inscribed circle
        if self.grid in ["polar", "log-polar"]:
            I *= grid.innercirc(self.Y, self.X)[None, :, :, None]

        # colorize (extended averaging)
        if self.H > 1 or np.any(self.h != 255):  # can be used for extended averaging
            I = self._colorize(I, frames)

        self.logger.info(f"{si(time.perf_counter() - t0)}s")

        return I

    def decode(self, I: np.ndarray, verbose: bool = False, denoise: bool = False, despike: bool = False) -> namedtuple:
        """Decode fringe patterns."""

        t0 = time.perf_counter()

        # get and apply videoshape
        T, Y, X, C = vshape(I).shape  # extract Y, X, C from data as these parameters depend on used camera
        I = I.reshape((T, Y, X, C))

        # assertions
        if T != self.T:
            self.logger.error("Number of frames of parameters and data don't match.")
            return

        if self.FDM:
            assert len(np.unique(self.N)) == 1, "Shifts aren't equal."

        # decolorize i.e. fuse hues/colors
        if self.H > 1 or not self._ismono:  # for gray fringes, color fusion is not performed, but extended averaging is
            I = self._decolorize(I)

        # demultiplex
        if self.SDM and 1 not in self.N or self.WDM or self.FDM:
            I = self._demultiplex(I)

        # demodulate
        phi, bri, mod, reg, res, fid = self._demodulate(I, verbose)

        if self.H > 1:
            idx = np.sum(self.h, axis=0) == 0
            if True in idx:  # blacken where color value of hue was black
                reg[..., idx] = np.nan
                if self.verbose:
                    res[..., idx] = np.nan
                    fid[..., idx] = -1  # np.nan

        # spatial unwrapping
        if self._isambiguous:
            self.logger.warning("Unwrapping is not spatially independent and only yields a relative phase map.")
            uwr = self._unwrap(phi if self.verbose or verbose else reg)

            if self.verbose:
                reg, res, fid = uwr
            else:
                reg = uwr[0]
        else:  # coordiante retransformation
            if self.D == 2:
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
            reg = median(reg, k=3)

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
            reg = bilateral(reg, k=3)

        # create named tuple to return
        if self.verbose or verbose:
            dec = namedtuple("decoding", "brightness modulation phase registration residuals orders")(bri, mod, phi,
                                                                                                      reg, res, fid)
        else:
            dec = namedtuple("decoding", "brightness modulation registration")(bri, mod, reg)

        self.logger.info(f"{si(time.perf_counter() - t0)}s")

        return dec

    def _unwrap(self, phi: np.ndarray, B: np.ndarray = None, func: str = "ski") -> namedtuple:  # todo: use B for quality guidance
        """Unwrap phase maps spacially."""

        t0 = time.perf_counter()

        T, Y, X, C = vshape(phi).shape
        assert T % self.D == 0, "Number of frames of parameters and data don't match."
        if self.D != T:
            phi = phi.reshape((self.D, -1, Y, X, C))
        else:
            phi = phi.reshape((T, Y, X, C))

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
            fid = np.empty((self.D, self.K, Y, X, C), np.int_)

        for d in range(self.D):
            if self.K == 1:  # todo: self.K[d] == 1
                self.logger.info(f"Spatial phase unwrapping in 2D{' for each color indepently' if C > 1 else ''}.")
            else:
                self.logger.info(f"Spatial phase unwrapping in 3D{' for each color indepently' if C > 1 else ''}.")
                func = "ski"  # todo: 3D SPU with OpenCV?

            for c in range(C):
                if X == 1:
                    reg[d, :, :, c] = np.unwrap(phi[d, :, :, c], axis=1)
                elif Y == 1:
                    reg[d, :, :, c] = np.unwrap(phi[d, :, :, c], axis=0)
                else:
                    if func in "cv2":  # OpenCV algorithm is usually faster, but can be much slower in noisy images
                        # dtype must be np.float32  # todo: test this
                        if isinstance(mask, np.ndarray) and vshape(mask).shape == phi.shape:
                            ui = self.quant
                            SNR = self.B[d, :, :, c] / ui
                            upi = np.sqrt(2) / np.sqrt(self.M) / np.sqrt(self._N) / SNR  # local phase uncertainties
                            upin = upi / (2 * np.pi)  # normalized local phase uncertainty
                            uxi = upin * self._l  # local positional uncertainties
                            ux = np.sqrt(1 / np.sum(1 / uxi ** 2))  # global phase uncertainty (by inverse variance weighting of uxi)
                            mask = np.astype(ux < 0.5, copy=False)  # todo: which limit?

                            reg[d, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[d, :, :, c], mask)  # todo: test this
                        else:
                            reg[d, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[d, :, :, c])

                        if self.verbose:
                            res[d, :, :, c] = unwrapping_instance.getInverseReliabilityMap()  # todo: test this
                            # todo: res vs. rel
                    else:  # Scikit-image algorithm is slower but delivers better results on edges
                        reg[d, :, :, c] = ski.restoration.unwrap_phase(phi[d, :, :, c])
                        # todo: res

            regmin = np.amin(reg[d])
            if regmin < 0:
                reg[d] -= regmin

            reg[d] = reg[d] / (2 * np.pi) * self._l[d, 0]

            if self.verbose:
                fid[d, 0] = -1  # np.nan  # unknown

        if self.verbose:
            uwr = namedtuple("unwrapping", "registration residuals orders")(reg, res, fid)
        else:
            uwr = namedtuple("unwrapping", "registration")(reg)

        self.logger.debug(f"{si(time.perf_counter() - t0)}s")

        return uwr

    @staticmethod
    def unwrap(phi: np.ndarray, mask: np.ndarray = None, func: str = "ski") -> np.array:  # todo: use B for quality guidance
        """Unwrap phase maps spacially.
        Based on the flag `func`, this is either done by
        https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html or
        https://docs.opencv.org/4.7.0/df/d3a/group__phase__unwrapping.html"""

        T, Y, X, C = vshape(phi).shape
        phi = phi.reshape((T, Y, X, C))

        if func in "cv2":  # OpenCV unwrapping
            # params = cv2.phase_unwrapping_HistogramPhaseUnwrapping_Params()
            params = cv2.phase_unwrapping.HistogramPhaseUnwrapping.Params()
            params.height = Y
            params.width = X
            # unwrapping_instance = cv2.phase_unwrapping.HistogramPhaseUnwrapping_create(params)
            unwrapping_instance = cv2.phase_unwrapping.HistogramPhaseUnwrapping.create(params)

        uwr = np.empty_like(phi)
        for t in range(T):
            for c in range(C):
                if func in "cv2":  # OpenCV algorithm is usually faster, but can be much slower in noisy images
                    # dtype must be np.float32  # todo: test this
                    if isinstance(mask, np.ndarray) and vshape(mask).shape == phi.shape:
                        mask = np.astype(mask[t, :, :, c], copy=False)
                        uwr[t, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[t, :, :, c], mask)
                    else:
                        uwr[t, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[t, :, :, c])

                    # # dtype of phasemust be np.uint8, dtype of shadow_mask np.uint8  # todo: test this
                    # reg[d, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[d, :, :, c], shadowMask=shadow_mask)
                else:  # Scikit-image algorithm is slower but delivers better results on edges
                    uwr[t, :, :, c] = ski.restoration.unwrap_phase(phi[t, :, :, c])

        return uwr

    def remap(self,
              xi: np.ndarray,
              B: np.ndarray = None,
              Bmin: float = 0,
              scale: float = 1,
              normalize: bool = True) -> np.ndarray:
        """Mapping decoded coordinates (having sub-pixel accuracy)
        from camera grid to (integer) positions on pattern/screen grid
        with weights from modulation.
        This yields a grid representing the screen (light source)
        with the pixel values being a relative measure
        of how much a screen (light source) pixel contributed
        to the exposure of the camera sensor."""

        t0 = time.perf_counter()

        T, Y, X, C = vshape(xi).shape

        if B is not None and B.ndim > 1:
            assert xi.shape[1:] == B.shape[1:], "'xi' and 'B' have different width, height or color channels"

            B = B.reshape((-1, Y, X, C))

        xi = xi.reshape((-1, Y, X, C))

        assert np.all(np.round(np.max(xi[d])) <= self.R[d] - 1 for d in range(self.D)), \
            f"Coordinates contain values > {self.R - 1}, decoding might be erroneous."

        if T == 1:
            if self.axis == 0:
                xi = np.vstack((xi, np.zeros_like(xi)))
            else:  # self.axis == 1
                xi = np.vstack((np.zeros_like(xi), xi))

        src = np.zeros((int(np.round(self.Y * scale)), int(np.round(self.X * scale)), C), np.float32)
        idx = np.rint(xi.swapaxes(1, 2) * scale).astype(int, copy=False)
        if B is not None:
            val = np.mean(B.swapaxes(1, 2), axis=0)
            if Bmin > 0:
                val *= val >= Bmin
        else:
            val = np.ones((Y, X, C), np.uint8)
        for c in range(C):
            src[idx[1].ravel(), idx[0].ravel(), c] += val[..., c].ravel()

        if normalize:
            mx = src.max()
            if mx > 0:
                src /= mx

        self.logger.info(f"{si(time.perf_counter() - t0)}s")

        return src

    def _error(self):  # todo: remove
        """Error."""
        # """Mean absolute distance between decoded and true coordinates, considering only quantization noise."""

        # f = Fringes(**{k: v for k, v in self.params.items() if k in Fringes.defaults})
        I = self.encode()
        dec = self.decode(I)

        eps = np.abs(self.coordinates() - dec.registration)  # / self.L
        idxe = np.argwhere(eps.squeeze() > 0.1)

        xavg = np.nanmean(eps)
        xmed = np.nanmedian(eps)
        xmax = np.nanmax(eps)
        xstd = np.nanstd(eps)
        SNR = self.L / xstd
        SPNR = self.L / xmax
        B = max(1, np.log2(SNR))
        # reserve = int(SNR / self.L) if SNR != np.inf else SNR
        reserve = int(SNR) if SNR != np.inf else SNR
        DR = max(0, 20 * np.log10(SNR))
        DRP = max(0, 20 * np.log10(SPNR))

        if self.verbose:
            ravg = np.nanmean(dec.residuals)
            rmed = np.nanmedian(dec.residuals)
            rmax = np.nanmax(dec.residuals)

            errors = namedtuple(
                "errors",
                "dynrange dynrangepeak bits meanabsdist medabsdist maxabsdist stdabsdist reserve, SNR, SPNR, rmeanabsdist rmedabsdist rmaxabsdist"
            )(DR, DRP, B, xavg, xmed, xmax, xstd, reserve, SNR, SPNR, ravg, rmed, rmax, )
        else:
            errors = namedtuple(
                "errors",
                "dynrange dynrangepeak bits meanabsdist medabsdist maxabsdist stdabsdist reserve SNR SPNR"
            )(DR, DRP, B, xavg, xmed, xmax, xstd, reserve, SNR, SPNR, )

        return errors

    @property
    def grid(self) -> str:
        """Coordinate system of fringe patterns."""
        return self._grid

    @grid.setter
    def grid(self, grid: str):
        _grid = str(grid)

        if (self.SDM or self.SSB) and self.grid not in self._grids[:2]:
            self.logger.error(f"Couldn't set 'grid': grid not in {self._grids[:2]}'.")
            return

        if self._grid != _grid and _grid in self._grids:
            self._grid = _grid
            self.logger.debug(f"{self._grid = }")
            self.SDM = self.SDM

    @property
    def angle(self) -> float:
        """Angle of coordinate system's principal axis."""
        return self._angle

    @angle.setter
    def angle(self, angle: float):
        _angle = float(np.remainder(angle, 360))  # todo: +- 45

        if self._angle != _angle:
            self._angle = _angle
            self.logger.debug(f"{self._angle = }")
            self._l = self.L / self._v

    @property
    def D(self) -> int:
        """Number of directions."""
        return self._D

    @D.setter
    def D(self, D: int):
        _D = int(min(max(1, D), self._Dmax))

        if self._D > _D:
            self._D = _D
            self.logger.debug(f"{self._D = }")

            self.N = self._N[:self.D, :]
            self.l = self._l[:self.D, :]  # l triggers v
            self.f = self._f[:self.D, :]

            if self._D == self._K == 1:
                self.FDM = False

            self.SDM = False
        elif self._D < _D:
            self._D = _D
            self.logger.debug(f"{self._D = }")

            self.N = np.append(self._N, np.tile(self._N[-1, :], (_D - self._l.shape[0], 1)), axis=0)
            self.l = np.append(self._l, np.tile(self._l[-1, :], (_D - self._l.shape[0], 1)), axis=0)  # l triggers v
            self.f = np.append(self._f, np.tile(self._f[-1, :], (_D - self._f.shape[0], 1)), axis=0)

            self.B = self.B

    @property
    def axis(self) -> int:
        """Axis along which to shift if number of directions equals one."""
        return self._axis

    @axis.setter
    def axis(self, axis: int | str):
        if isinstance(axis, str):
            if axis.lower() in ["x", "u"]:
                axis = 0
            elif axis.lower() in ["y", "v"]:
                axis = 1
            else:
                return

        _axis = int(min(max(0, axis), 1))

        if self._axis != _axis:
            self._axis = _axis
            self.logger.debug(f"{self._axis = }")
            self.l = self.L / self.v  # l triggers v

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

        if _T == 1:  # WDM + SDM todo: SSB?
            if self.grid not in self._grids[:2]:
                self.logger.error(f"Couldn't set 'T = 1': grid not in {self._grids[:2]}'.")
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
            Kmax = 2#3  # todo: which is better: 2 or 3?
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
            if self._isambiguous:
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
                    N[:, 1:k + 1] += 1
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
        """Height of fringe patterns [px]."""
        return self._Y

    @Y.setter
    def Y(self, Y: int):
        _Y = int(min(max(1, Y), self._Ymax, self._Pmax / self.X))

        if self._Y != _Y:
            self._Y = _Y
            self.logger.debug(f"{self._Y = }")

            if self._X == self._Y == 1:
                self.D = 1
                self.axis = 0
            elif self._X == 1:
                self.D = 1
                self.axis = 1
            elif self._Y == 1:
                self.D = 1
                self.axis = 0

            self.l = self.L / self._v

    @property
    def X(self: int) -> int:
        """Width of fringe patterns [px]."""
        return self._X

    @X.setter
    def X(self, X: int):
        _X = int(min(max(1, X), self._Xmax, self._Pmax / self.Y))

        if self._X != _X:
            self._X = _X
            self.logger.debug(f"{self._X = }")

            if self._X == self._Y == 1:
                self.D = 1
                self.axis = 0
            elif self._X == 1:
                self.D = 1
                self.axis = 1
            elif self._Y == 1:
                self.D = 1
                self.axis = 0

            self.l = self.L / self._v

    @property
    def C(self) -> int:
        """Number of color channels."""
        return 3 if self.WDM or not self._ismono else 1

    @property
    def P(self) -> int:
        """Number of pixels per color channel and frame."""
        return self.Y * self.X

    @property
    def alpha(self) -> float:
        """Factor for extending the coding range 'L'."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        _alpha = float(min(max(1, alpha), self._alphamax))

        if self._alpha != _alpha:
            self._alpha = _alpha
            self.logger.debug(f"{self._alpha = }")
            self.l = self.L / self._v

    @property
    def R(self) -> np.ndarray:
        """Lengths of fringe patterns for each direction [px]."""
        if self.D == 2:
            R = np.array([self.X, self.Y])
        else:
            if self.axis == 0:
                R = np.array([self.X])
            else:
                R = np.array([self.Y])
        return R

    @property
    def L(self) -> int | float:
        """Length to be encoded [px]."""
        return float(self.R.max() * self.alpha)

    @property
    def UMR(self) -> np.ndarray:
        """Unambiguous measurement range."""
        # If neither wavelength nor periods are integers, lcm resp. gcd are extended to rational numbers.

        if self._UMR is not None:  # cache
            return self._UMR

        # precision = np.finfo("float64").precision - 1
        precision = 10

        UMR = np.empty(self.D)
        for d in range(self.D):
            l = self._l[d].copy()
            v = self._v[d].copy()

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

            if all(i % 1 == 0 for i in l):  # all l are integers
                UMR[d] = np.lcm.reduce([int(i) for i in l])
            elif all(i % 1 == 0 for i in v):  # all v are integers
                UMR[d] = self.L / np.gcd.reduce([int(i) for i in v])
            elif all(np.isclose(l, np.rint(l), rtol=0, atol=10 ** -precision)):
                UMR[d] = np.lcm.reduce([int(i) for i in np.rint(l)])  # all l are approximately integers
            elif all(np.isclose(v, np.rint(v), rtol=0, atol=10 ** -precision)):
                UMR[d] = self.L / np.gcd.reduce([int(i) for i in np.rint(v)])  # all v are approximately integers
            else:  # l and v both are not integers, not even approximately
                lcopy = l.copy()

                # mutual factorial test for integers i.e. mutual divisibility test
                for i in range(self.K - 1):
                    for j in range(i + 1, self.K):
                        if l[i] % l[j] < 10 ** - precision:
                            lcopy[j] = 1
                        elif l[j] % l[i] < 10 ** - precision:
                            lcopy[i] = 1

                l = l[lcopy != 1]

                # estimate whether elements are rational or irrational
                decimals = [len(str(i)) - len(str(int(i))) - 1 for i in l]  # -1: dot
                Dl = max(decimals)
                decimals = [len(str(i)) - len(str(int(i))) - 1 for i in v]  # -1: dot
                Dv = max(decimals)

                if min(Dl, Dv) < precision:  # rational numbers without integers (i.e. not covered by isclose(atol))
                    # extend lcm/gcd to rational numbers

                    if Dl <= Dv:
                        ls = l * 10 ** Dl  # wavelengths scaled
                        UMR[d] = np.lcm.reduce([int(i) for i in ls]) / 10 ** Dl
                        self.logger.debug("Extended lcm to rational numbers.")
                    else:
                        vs = v * (10 ** Dv)  # wavelengths scaled
                        UMR[d] = self.L / np.gcd.reduce([int(i) for i in vs]) / (10 ** Dv)
                        self.logger.debug("Extended gcd to rational numbers.")
                else:  # irrational numbers or rational numbers with more digits than "precision"
                    # round and extend lcm to rational numbers
                    ls = np.round(l * 10 ** precision, decimals=precision).astype("uint64")
                    UMR[d] = np.lcm.reduce(ls) / 10 ** precision

        # self._UMR = float(np.min(UMR))  # cast type frm "numpy.core.multiarray.scalar" to "int" or "float"

        self._UMR = UMR
        self.logger.debug(f"self.UMR = {str(self.UMR)}")

        if self._isambiguous:
            self.logger.warning(
                "UMR < R. Unwrapping will not be spatially independent and only yield a relative phase map.")

        return self._UMR

    @property
    def eta(self) -> float:
        """Coding efficiency."""
        eta = self.R / self.UMR
        eta[self.UMR < self.R] = 0
        return eta

    @property
    def M(self) -> float | np.ndarray:
        """Number of averaged intensity samples."""
        M = np.sum(self.h, axis=0) / 255
        if len(set(M)) == 1:
            M = M[0]
        else:
            # todo: M = M[None, None, :]
            pass
        return float(M.max())  # convert Numpy float64 to Python float  # todo: fix M.max()

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
        #     self.logger.error("Couldn't set 'H': WDM is active.")
        #     return

        if self.H != _H:
            if self.WDM:
                self.h = "w" * _H
            elif _H == 2:
                self.h = "rb"
            else:
                h = "rgb" * (_H // 3 + 1)
                self.h = h[:_H]

    @property
    def h(self) -> np.ndarray:
        """Hues i.e. colors of fringe patterns."""
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
            if set(h).intersection(LUT.keys()):
                h = [LUT[c] for c in h]
            elif h == "default":
                h = self.defaults["h"]
            else:
                return

        # make array, clip first and then cast to dtype to avoid integer under-/overflow
        _h = np.array(h).clip(0, 255).astype("uint8", copy=False)

        # empty array
        if not _h.size:
            return

        # change shape to (H, 3) or limit shape
        if _h.ndim == 0:
            _h = np.full((self.H, 3), _h)
        elif _h.shape[min(_h.ndim - 1, 1)] < 3:
            _h = np.full((self.H, 3), _h[min(_h.ndim - 1, 1)])
        elif _h.ndim == 1:
            _h = np.vstack([_h[:3] for h in range(self.H)])
        elif _h.ndim == 2:
            _h = _h[:self._Hmax, :3]
        else:
            _h = _h[:self._Hmax, :3, ..., -1]

        if _h.shape[1] == 2:  # C-axis must equal 3
            self.logger.error("Couldn't set 'h': Only 2 instead of 3 color channels provided.")
            return

        if np.any(np.max(_h, axis=1) == 0):
            self.logger.error("Didn't set 'h': Black color is not allowed.")
            return

        if self.WDM and not self._ismono:
            self.logger.error("Couldn't set 'h': 'WDM' is active, but not all hues are monochromatic.")
            return

        if _h.size and not np.array_equal(self._h, _h):
            Hold = self.H
            self._h = _h
            self.logger.debug(f"self._h = {str(self._h).replace(chr(10), ',')}")
            if Hold != _h.shape[0]:
                self.logger.debug(f"self.H = {_h.shape[0]}")  # computed upon call
            self.logger.debug(f"{self.M = }")  # computed upon call

    @property
    def TDM(self) -> bool:
        """Temporal division multiplexing."""
        return self.T > 1

    @property
    def SDM(self) -> bool:
        """Spatial division multiplexing."""
        return self._SDM

    @SDM.setter
    def SDM(self, SDM: bool):
        _SDM = bool(SDM)

        if _SDM:
            if self.D != 2:
                _SDM = False
                self.logger.error("Didn't set 'SDM': Pointless as only one dimension exist.")

            if self.grid not in self._grids[:2]:
                _SDM = False
                self.logger.error(f"Couldn't set 'SDM': grid not in {self._grids[:2]}'.")

            if self.FDM:
                _SDM = False
                self.logger.error("Couldn't set 'SDM': FDM is active.")

        if self._SDM != _SDM:
            self._SDM = _SDM
            self.logger.debug(f"{self._SDM = }")
            self.logger.debug(f"{self.T = }")  # computed upon call

            if self.SDM:
                self.B /= self.D
            else:
                self.B *= self.D

    @property
    def WDM(self) -> bool:
        """Wavelength division multiplexing."""
        return self._WDM

    @WDM.setter
    def WDM(self, WDM: bool):
        _WDM = bool(WDM)

        if _WDM:
            if not np.all(self.N == 3):
                _WDM = False
                self.logger.error("Couldn't set 'WDM': At least one Shift != 3.")

            if not self._ismono:
                _WDM = False
                self.logger.error("Couldn't set 'WDM': Not all hues are monochromatic.")

            if self.FDM:  # todo: remove this, already covered by N
                _WDM = False
                self.logger.error("Couldn't set 'WDM': FDM is active.")

        if self._WDM != _WDM:
            self._WDM = _WDM
            self.logger.debug(f"{self._WDM = }")
            self.logger.debug(f"{self.T = }")  # computed upon call

    @property
    def FDM(self) -> bool:
        """Frequency division multiplexing."""
        return self._FDM

    @FDM.setter
    def FDM(self, FDM: bool):
        _FDM = bool(FDM)

        if _FDM:
            if self.D == self.K == 1:
                _FDM = False
                self.logger.error("Didn't set 'FDM': Dimensions * Sets = 1, so nothing to multiplex.")

            if self.SDM:
                _FDM = False
                self.logger.error("Couldn't set 'FDM': SDM is active.")

            if self.WDM:  # todo: remove, already covered by N
                _FDM = False
                self.logger.error("Couldn't set 'FDM': WDM is active.")

        if self._FDM != _FDM:
            self._FDM = _FDM
            self.logger.debug(f"{self._FDM = }")
            # self.K = self._K
            self.N = self._N
            self.l = self._l  # l triggers v
            if self.FDM:
                self.f = self._f
            else:
                self.f = np.ones((self.D, self.K))

            # keep maximum possible visibility constant
            if self.FDM:
                self.B /= (self.D * self.K)
            else:
                self.B *= (self.D * self.K)

    @property
    def static(self) -> bool:
        """Flag for creating static fringes, i.e. they remain congruent when shifted."""
        return self._static

    @static.setter
    def static(self, static: bool):
        _static = bool(static)

        if self._static != _static:
            self._static = _static
            self.logger.debug(f"{self._static = }")
            self.l = self._l  # l triggers v
            self.f = self._f

    @property
    def K(self) -> int:
        """Number of sets."""
        return self._K

    @K.setter
    def K(self, K: int):
        # todo: different K for each D: use array of arrays
        # a = np.ones(2)
        # b = np.ones(5)
        # c = np.array([a, b])

        Kmax = (self._Nmax - 1) / 2 / self.D if self.FDM else self._Kmax  # todo: check if necessary
        _K = int(min(max(1, K), Kmax))

        if self._K > _K:  # remove elements
            self._K = _K
            self.logger.debug(f"{self._K = }")

            self.N = self._N[:, :self.K]
            self.l = self._l[:, :self.K]  # l triggers v
            self.f = self._f[:, :self.K]

            if self._D == self._K == 1:
                self.FDM = False
        elif self._K < _K:  # add elements
            self._K = _K  # set l before K
            self.logger.debug(f"{self._K = }")

            self.N = np.append(self._N, np.tile(self._N[0, 0], (self.D, _K - self._N.shape[1])),
                               axis=1)  # don't append N from defaults, this might be in conflict with WDM!
            l = self.L ** (1 / np.arange(self._l.shape[1] + 1, _K + 1))
            self.l = np.append(self._l, np.tile(l, (self.D, 1)), axis=1)  # l triggers v
            self.f = np.append(self._f, np.tile(self.defaults["f"][0, 0], (self.D, _K - self._f.shape[1])), axis=1)

            self.B = self.B

    @property
    def Nmin(self) -> int:
        """Minimum number of shifts to (uniformly) sample temporal frequencies."""
        if self.FDM:
            Nmin = int(np.ceil(2 * self.f.max() + 1))  # sampling theorem
            # todo: 2 * D * K + 1 -> fractional periods if static
        else:
            Nmin = 1
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
    def N(self, N: int | tuple[int] | list[int] | np.ndarray | str):
        if isinstance(N, str):
            if N == "auto":
                N = np.full((self.D, self.K), 4, int)
            else:
                return

        # make array, cast to dtype, clip
        _N = np.array(N, int).clip(self.Nmin, self._Nmax)

        if not _N.size:  # empty array
            return

        # change ndim to 2 and limit shape
        if _N.ndim == 0:
            _N = np.full((self.D, self.K), _N)
        elif _N.ndim == 1:
            _N = np.vstack([_N[:self._Kmax] for d in range(self.D)])
        elif _N.ndim == 2:
            _N = _N[:self._Dmax, :self._Kmax]
        else:
            _N = _N[:self._Dmax, :self._Kmax, ..., -1]

        if np.all(_N == 1) and _N.shape[1] == 1:  # any
            pass  # SSB
        elif np.any(_N <= 2):
            for d in range(self.D):
                if not any(_N[d] >= 3):
                    i = np.argmax(_N[d])  # np.argmin(_N[d])
                    _N[d, i] = 3

        if self.WDM and not np.all(_N == 3):
            self.logger.error("Couldn't set 'N': At least one Shift != 3.")
            return

        if self.FDM and not np.all(_N == _N[0, 0]):
            # _N = np.tile(self.Nmin, _N.shape)
            _N = np.tile(_N[0, 0], _N.shape)

        if _N.size and not np.array_equal(self._N, _N):
            self._N = _N  # set N before K
            self.logger.debug(f"self._N = {str(self.N).replace(chr(10), ',')}")
            self._UMR = None
            self.D, self.K = self._N.shape  # set N before D, K and D
            self.logger.debug(f"{self.T = }")  # set D and K before T

    @property
    def l(self) -> np.ndarray:
        """Wavelengths of fringe periods [px]."""
        if self.D == 1 or len(np.unique(self._l, axis=0)) == 1:  # sets in directions are identical
            v = self._l[0]  # 1D
        else:
            v = self._l  # 2D
        return v

    @l.setter
    def l(self, l: int | float | tuple[int | float] | list[int | float] | np.ndarray | str):
        if isinstance(l, str):
            if l == "auto":
                if self.K == 1:
                    l = self.L
                elif self.K == 2:
                    r = int(np.sqrt(self.L) + 0.5)  # rint(root)
                    l = np.array([r, r + 1])  # two consecutive integers are always coprime
                else:
                    pass

                lmin = int(np.ceil(self.lmin))

                lmax = int(self.L ** (1 / self.K) + 0.5)  # rint; wavelengths are around kth root of self.L

                if self.K >= 2:
                    lmax += 1

                    if self.K >= 3:
                        lmax += 1

                        if lmax % 2 == 0:  # kth root was even
                            lmax += 1

                        if self.K >= 3:
                            ith = self.K - 3
                            lmax = sympy.ntheory.generate.nextprime(lmax, ith)

                    lmax = max(lmin, lmax)

                    if self.lmin < self.L:
                        lmax = min(lmax, int(np.ceil(self.L)))

                    n = lmax - lmin + 1
                    K = min(self.K, n)
                    C = sp.special.comb(n, K, exact=True, repetition=False)  # number of unique combinations
                    combos = (c for c in it.combinations(range(lmin, lmax + 1), K) if np.lcm.reduce(c) >= self.L)
                    cnext = next(combos)
                    l = np.array(cnext)  # this removes first element of combos
                    mn2 = np.sum(l ** 2)
                    sum2min = np.sum(((lmin + np.arange(K)) ** 2))  # starting with lmin and increasing by one per set
                    if mn2 != sum2min:
                        mn = np.sum(l)
                        b = list(combos)
                        for c in combos:  # iterate over the remaining combos
                            c = np.array(c)
                            sum = np.sum(c)  # argmin of L1 norm is equal to argmin of L2 norm due to monotonicity

                            if sum < mn:
                                mn = sum
                                # mn2 = np.sum(c ** 2)
                                l = c
                            elif sum == mn:  # under the L1 norm there may be ambiguities, so also check L2 norm
                                sum2 = np.sum(c ** 2)

                                if sum2 < mn2:
                                    mn2 = sum2  # determine mn2 here, because (sum == mn) is rarer than (sum < mn) # todo: is this assumption true?
                                    l = c
                                elif sum2 == mn2:
                                    pass  # this schould never happen since there exists only one solution in the L2 norm

                                if mn2 == sum2min:
                                    break

                            # c = np.array(c)
                            # sum2 = np.sum(c ** 2)
                            #
                            # if sum2 <= sum2min:
                            #    l = c
                            #    break
                            # elif sum2 < mn2:
                            #    mn2 = sum2
                            #    l = c
                            # elif sum2 == mn:
                            #    pass  # this schould never happen since there exists only one solution in the L2 norm

                    # todo: K -> self.K
                    # if K != self.K:  # then K is smaller than self.K  # todo: is this assumption true?
                    #     c = self.K / K
                    #     if c % 1 == 0:  # c is integer
                    #         self.N += 1
                    #     else:
                    #         # l = np.repeat(l, int(np.ceil(c)))[:self.K]
                    #         l = np.tile(l, int(np.ceil(c)))[:self.K]

                    # if self.K == 1:
                    #     l = self.L
                    # else:
                    #     if self.vmax > self.lmin:
                    #         vmax = int(self.vmax)
                    #
                    #         # decreasing sequence
                    #         # v = vmax - np.arange(self.K)
                    #
                    #         # alternating largest two consecutive number of periods, starting with largest
                    #         i = np.empty(self.K, int)
                    #         i[0::2] = 0
                    #         i[1::2] = 1
                    #         v = np.array([vmax, vmax - 1])[i]  # indices which are alternating between 0 and 1
                    #
                    #         l = self.L / v
                    #     else:
                    #         lmin = int(np.ceil(self.lmin))
                    #
                    #         # increasing sequence
                    #         l = lmin + np.arange(self.K)
                    #
                    #         # # alternating smallest two consecutive wavelengths, starting with smallest
                    #         # i = np.empty(self.K, int)  # indices which are alternating between 0 and 1
                    #         # i[0::2] = 0
                    #         # i[1::2] = 1
                    #         # l = np.array([lmin, lmin + 1])[i]
            elif l == "exponential":
                # K = int(np.ceil(np.log2(self.vmax))) + 1  # + 1: 2 ** 0 = 1
                l = np.concatenate(([np.inf], np.geomspace(self.L, self.lmin, self.K)))
            elif l == "linear":
                l = np.concatenate(([np.inf], np.linspace(self.L, self.lmin, self.K - 1)))
            else:
                return

        # make array, cast to dtype, clip
        _l = np.array(l, float).clip(self.lmin, np.inf)

        # empty array
        if not _l.size:
            return

        # change ndim to 2 and limit shape
        if _l.ndim == 0:
            _l = np.full((self.D, self.K), _l)
        elif _l.ndim == 1:
            _l = np.vstack([_l[:self._Kmax] for d in range(self.D)])
        elif _l.ndim == 2:
            _l = _l[:self._Dmax, :self._Kmax]
        else:
            _l = _l[:self._Dmax, :self._Kmax, ..., -1]

        if self.FDM:
            if self.static:
                if _l.size != self.D * self.K or not np.all(_l % 1 == 0) or not np.lcm.reduce(
                        (self.L / _l).astype(int, copy=False).ravel()) == np.prod(self.L / _l):  # todo: allow coprimes?!
                    n = min(10, self.vmax // 2)
                    ith = self.D * self.K
                    pmax = sympy.ntheory.generate.nextprime(n, ith + 1)
                    p = np.array(list(sympy.ntheory.generate.primerange(n, pmax + 1)))[:ith]  # primes
                    p = [p[-i // 2] if i % 2 else p[i // 2] for i in range(len(p))]  # resort primes
                    _v = np.sort(np.array(p, float).reshape((self.D, self.K)), axis=1)  # resort primes
                    _l = self.L / _v
                    self.logger.warning(f"Wavelengths were not coprime. "
                                        f"Changing values to {str(_l.round(3)).replace(chr(10), ',')}.")
            else:
                lmin = (self._Nmax - 1) / 2 > self.L / _l
                if np.any(_l < lmin):  # clip v so that Nmax <= 2 * max(v) + 1 = 2 * max(f) + 1
                    _l = np.minimum(_l, lmin)

        if _l.size and not np.array_equal(self._l, _l):
            self._l = _l  # set l before K
            self.logger.debug(f"self._l = {str(self.l.round(3)).replace(chr(10), ',')}")
            self._v = self.L / self._l
            self.logger.debug(f"self._v = {str(self.v.round(3)).replace(chr(10), ',')}")
            self._UMR = None
            self.D, self.K = self._l.shape  # set l before K and K before f
            self.f = self._f

    @property
    def v(self) -> np.ndarray:
        """Spatial frequencies,
        i.e. number of periods/fringes across maximum coding length."""
        if self.D == 1 or len(np.unique(self._v, axis=0)) == 1:  # sets in directions are identical
            v = self._v[0]  # 1D
        else:
            v = self._v  # 2D
        return v

    @v.setter
    def v(self, v: int | float | tuple[int | float] | list[int | float] | np.ndarray | str):
        if isinstance(v, str):
            if v == "auto":
                if self.K == 1:
                    v = 1
                else:
                    # if self.vmax > self.lmin and np.lcm.reduce(int(np.ceil(self.lmin)) + np.arange(self.K)) < self.L:
                    if 1 / self.L * np.sum(self.vmax - np.arange(self.K)) > 1 / np.sum(self.lmin + np.arange(self.K)):
                        vmax = int(self.vmax)  # todo: ripples from int()
                        #vmax = self.vmax

                        # decreasing sequence
                        v = vmax - np.arange(self.K)

                        # # alternating largest two consecutive number of periods, starting with largest
                        # i = np.empty(self.K, int)
                        # i[0::2] = 0
                        # i[1::2] = 1
                        # v = np.array([vmax, vmax - 1])[i]  # indices which are alternating between 0 and 1
                    else:
                        lmin = int(np.ceil(self.lmin))
                        lmin = self.lmin

                        # increasing sequence
                        l = lmin + np.arange(self.K)

                        # # alternating smallest two consecutive wavelengths, starting with smallest
                        # i = np.empty(self.K, int)  # indices which are alternating between 0 and 1
                        # i[0::2] = 0
                        # i[1::2] = 1
                        # l = np.array([lmin, lmin + 1])[i]

                        v = self.L / l
            elif v == "exponential":
                # K = int(np.ceil(np.log2(self.vmax))) + 1  # + 1: 2 ** 0 = 1
                v = np.concatenate(([0], np.geomspace(1, self.vmax, self.K)))
            elif v == "linear":
                v = np.concatenate(([0], np.linspace(1, self.vmax, self.K - 1)))
            else:
                return

        # make array, cast to dtype, clip
        _v = np.array(v, float).clip(0, self.vmax)

        # empty array
        if not _v.size:
            return

        # change ndim to 2 and limit shape
        if _v.ndim == 0:
            _v = np.full((self.D, self.K), _v)
        elif _v.ndim == 1:
            _v = np.vstack([_v[:self._Kmax] for d in range(self.D)])
        elif _v.ndim == 2:
            _v = _v[:self._Dmax, :self._Kmax]
        else:
            _v = _v[:self._Dmax, :self._Kmax, ..., -1]

        if self.FDM:
            if self.static:
                if _v.size != self.D * self.K or not np.all(_v % 1 == 0) or \
                        not np.lcm.reduce(_v.astype(int, copy=False).ravel()) == np.prod(_v):  # todo: allow coprimes?!
                    n = min(10, self.vmax // 2)
                    ith = self.D * self.K
                    pmax = sympy.ntheory.generate.nextprime(n, ith + 1)
                    p = np.array(list(sympy.ntheory.generate.primerange(n, pmax + 1)))[:ith]  # primes
                    p = [p[-i // 2] if i % 2 else p[i // 2] for i in range(len(p))]  # resort primes
                    _v = np.sort(np.array(p, float).reshape((self.D, self.K)), axis=1)  # resort primes
                    self.logger.warning(f"Periods were not coprime. "
                                        f"Changing values to {str(_v.round(3)).replace(chr(10), ',')}.")
            # else:
            #     vmax = (self._Nmax - 1) / 2 > _v
            #     _v = np.minimum(_v, vmax)

        if _v.size and not np.array_equal(self._v, _v):
            self._v = _v  # set v before D and K
            self.logger.debug(f"self._v = {str(self.v.round(3)).replace(chr(10), ',')}")
            self._l = self.L / self._v
            self.logger.debug(f"self._l = {str(self.l.round(3)).replace(chr(10), ',')}")
            self._UMR = None
            self.D, self.K = self._v.shape  # set l before K and K before f
            self.f = self._f

    @property
    def fmax(self):
        """Maximum temporal frequency,
        i.e. maximum number of periods to shift over."""
        return min((self.Nmin - 1) / 2, self.vmax) if self.FDM and self.static else (self.Nmin - 1) / 2

    @property
    def f(self) -> np.ndarray:
        """Temporal frequency, i.e. number of periods to shift over."""
        if self.D == 1 or len(np.unique(self._f, axis=0)) == 1:  # sets in directions are identical
            f = self._f[0]  # 1D
        else:
            f = self._f  # 2D
        return f

    @f.setter
    def f(self, f: int | float | tuple[int | float] | list[int | float] | np.ndarray | str):
        if isinstance(f, str):
            if f == "auto":
                f = np.ones((self.D, self.K))
            else:
                return

        # make array, cast to dtype, clip
        _f = np.array(f, float).clip(-self.fmax, self.fmax)

        # empty array
        if not _f.size:
            return

        # change ndim to 2 and limit shape
        if _f.ndim == 0:
            _f = np.full((self.D, self.K), _f)
        elif _f.ndim == 1:
            _f = np.vstack([_f[:self._Kmax] for d in range(self.D)])
        elif _f.ndim == 2:
            _f = _f[:self._Dmax, :self._Kmax]
        else:
            _f = _f[:self._Dmax, :self._Kmax, ..., -1]

        D = min(_f.shape[0], self._N.shape[0])
        K = min(_f.shape[1], self._N.shape[1])
        if np.any(_f[:D, :K] % self._N[:D, :K] == 0):
            # _f = np.ones(_f.shape)
            _f[:D, :K][_f[:D, :K] % self._N[:D, :K] == 0] = 1

        if self.FDM:
            if self.static:
                _f = self._v  # periods to shift over = one full revolution
            else:
                if _f.shape != (self.D, self.K) or not np.all(i % 1 == 0 for i in _f) or \
                        len(np.unique(np.abs(_f))) < _f.size:  # assure _f are int and absolute values of _f differ
                    _f = np.arange(1, self.D * self.K + 1, dtype=float).reshape((self.D, self.K))

        if _f.size and 0 not in _f and not np.array_equal(self._f, _f):
            self._f = _f  # set f before D and K
            self.logger.debug(f"self._f = {str((self.f * (-1 if self.reverse else 1)).round(3)).replace(chr(10), ',')}")
            self.D, self.K = self._f.shape
            self.N = self._N  # todo: remove if fractional periods is implemented, log warning

    @property
    def o(self) -> float:
        """Phase offset within interval (-2pi, +2pi)."""
        return self._o

    @o.setter
    def o(self, o: float):
        _o = float(np.abs(o) % (2 * np.pi) * np.sign(o))

        if self._o != _o:
            self._o = _o
            self.logger.debug(f"self._o = {self._o / np.pi} PI")

    @property
    def _ismono(self) -> bool:
        """All hues are monochromatic, i.e. RGB values are identical for each hue."""
        return all(len(set(h)) == 1 for h in self.h)

    @property
    def _isambiguous(self) -> bool:
        """Unambiguous measument range is larger than the screen length."""
        return np.any(self.UMR < self.R * self.alpha)

    @property
    def lmin(self) -> float:
        """Minimum resolvable wavelength [px]."""
        fmax = min((self.Nmin - 1) / 2, self.L / self._lmin) if self.FDM and self.static else (self.Nmin - 1) / 2  # else circular loop
        return min(self._lmin, self.L / fmax) if self.FDM and self.static else self._lmin

    @lmin.setter
    def lmin(self, lmin: float):
        _lmin = float(max(self._lminmin, lmin))

        if self._lmin != _lmin:
            self._lmin = _lmin
            self.logger.debug(f"{self._lmin = }")
            self.logger.debug(f"{self.vmax = }")  # computed upon call
            self.l = self.l  # l triggers v

    @property
    def vmax(self) -> float:
        """Maximum resolvable spatial frequency."""
        return self.L / self.lmin

    @property
    def mode(self) -> str:
        """Mode for wavelengths encoding and decoding."""
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        _mode = str(mode)

        if self._mode != _mode and _mode in self._modes:
            self._mode = _mode
            self.logger.debug(f"{self._mode = }")

    @property
    def reverse(self) -> bool:
        """Flag for shifting fringes in reverse direction."""
        return self._reverse

    @reverse.setter
    def reverse(self, reverse: bool):
        _reverse = bool(reverse)

        if self._reverse != _reverse:
            self._reverse = _reverse
            self.logger.debug(f"{self._reverse = }")
            self.logger.debug(f"self._f = {str((self.f * (-1 if self.reverse else 1)).round(3)).replace(chr(10), ',')}")

    @property
    def verbose(self) -> bool:
        """Flag for additionally returning intermediate results,
        i.e. phase map and reliability/residuals map."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        _verbose = bool(verbose)

        if self._verbose != _verbose:
            self._verbose = _verbose
            self.logger.debug(f"{self._verbose = }")

    @property
    def SSB(self) -> bool:
        """Flag indicating wheather single sideband demodulation is deployed."""
        return self.H == self.K == 1 and np.all(self._N == 1) and self.grid in self._grids[:2]  # todo: allow H > 1 and use decolorizing, then for each color SSB

    @property
    def PU(self) -> str:
        """Phase unwrapping method."""

        if self.SSB:
            PU = "SSB"  # single sideband demodulation
        elif self.K == np.all(self.v <= 1):
            PU = "none"
        elif self._isambiguous:
            PU = "spatial"
        else:
            PU = "temporal"

        return PU

    @property
    def gamma(self) -> float:
        """Gamma correction factor used to compensate the display response curve."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float):
        _gamma = float(min(max(0, gamma), self._gammamax))

        if self._gamma != _gamma and _gamma != 0:
            self._gamma = _gamma
            self.logger.debug(f"{self._gamma = }")

    @property
    def shape(self) -> tuple:
        """Shape of fringe pattern sequence in video shape,
        i.e. (frames, height, with, color channels)."""
        return self.T, self.Y, self.X, self.C

    @property
    def size(self) -> np.uint64:
        """Number of pixels of fringe pattern sequence,
        i.e. frames * height * width * color channels."""
        return float(np.prod(self.shape, dtype=np.uint64))  # using uint64 prevents integer overflow

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by fringe pattern sequence.
        Does not include memory consumed by non-element attributes of the array object."""
        return self.size * self.dtype.itemsize

    @property
    def dtype(self) -> np.dtype:
        """Data type."""
        return np.dtype(self._dtype)  # this is a hotfix for setting _dtype directly as a str in init

    @dtype.setter
    def dtype(self, dtype: np.dtype | str):
        _dtype = np.dtype(dtype)

        if self._dtype != _dtype and str(_dtype) in self._dtypes:
            Imaxold = self.Imax
            self._dtype = _dtype
            self.logger.debug(f"{self._dtype = }")
            self._A = self._A * self.Imax / Imaxold
            self.logger.debug(f"self._A = {self._A}")
            self._B = self._B * self.Imax / Imaxold
            self.logger.debug(f"self._B = {self._B}")

    @property
    def Imax(self) -> int:
        """Maximum gray value."""
        return np.iinfo(self.dtype).max if self.dtype.kind in "ui" else 1

    @property
    def Amin(self):
        """Minimum bias."""
        return self.B / self.Vmax

    @property
    def Amax(self):
        """Maximum bias."""
        return self.Imax - self.B / self.Vmax

    @property
    def A(self) -> float:
        """Bias."""
        return self._A

    @A.setter
    def A(self, A: float):
        _A = float(min(max(self.Amin, A), self.Amax))

        if self._A != _A and _A != 0:
            self._A = _A
            self.logger.debug(f"{self._A = }")

    @property
    def Bmax(self):
        """Maximum amplitude."""
        return min(self.A, self.Imax - self.A) * self.Vmax

    @property
    def B(self) -> float:
        """Amplitude."""
        return self._B

    @B.setter
    def B(self, B: float):
        _B = float(min(max(0, B), self.Bmax))

        if self._B != _B and _B != 0:
            self._B = _B
            self.logger.debug(f"{self._B = }")

    @property
    def betamax(self):
        return 1 / (1 + self.V)

    @property
    def beta(self) -> float:
        """Relative bias i.e. relative mean intensity."""
        return self.A / self.Imax

    @beta.setter
    def beta(self, beta) -> float:
        _beta = float(min(max(0, beta), self.betamax))

        if self._A != _beta * self.Imax or self.B != _beta * self.Imax * self.V:
            self.B = _beta * self.Imax * self.V
            self.A = _beta * self.Imax

    @property
    def Vmax(self):
        """Maximum visibility."""
        return 1 / (self.D * self.K) if self.FDM else 1 / self.D if self.SDM else 1

    @property
    def V(self) -> float:
        """Fringe visibility (fringe contrast)."""
        return self.B / self.A

    @V.setter
    def V(self, V: float):
        _V = float(min(max(0, V), self.Vmax))

        if self._B != _V * self.A:
            self.B = _V * self.A
            self.logger.debug(f"{self.V = }")

    @property
    def Vmin(self) -> float:
        """Minimum fringe visibility (fringe contrast) for measurement to be valid, within interval [0, 1]."""
        return self._Vmin

    @Vmin.setter
    def Vmin(self, Vmin: float):
        _Vmin = float(min(max(0, Vmin), 1))

        if self._Vmin != _Vmin:
            self._Vmin = _Vmin
            self.logger.debug(f"{self._Vmin = }")

    @property
    def r(self) -> int:
        """Number of quantization bits."""
        return 1 if self.dtype.kind in "b" else np.iinfo(
            self.dtype).bits if self.dtype.kind in "ui" else 10 ** np.finfo(self.dtype).precision

    @property
    def Q(self) -> float:
        """Number of quantization levels."""
        return 2 ** self.r

    @property
    def q(self) -> float:
        """Quantization step size."""
        return 1.0 if self.dtype.kind in "uib" else np.finfo(self.dtype).resolution

    @property
    def quant(self) -> float:
        """Quantization noise (standard deviation) [DN]."""
        return float(self.q / np.sqrt(12))  # convert Numpy float64 to Python float

    @property
    def dark(self) -> float:
        """Dark noise of digital camera (standard deviation) [electrons]."""
        return self._dark

    @dark.setter
    def dark(self, dark: float):
        _dark = float(min(max(0, dark), np.sqrt(self.Imax)))

        # _dark = max(_dark, 0.49)  # todo: temporal noise is dominated by quantization noise ->

        _dark = max(0, _dark - self.quant)  # correct for quantization noise contained in dark noise measurement

        if self._dark != _dark:
            self._dark = _dark
            self.logger.debug(f"{self._dark = }")

    @property
    def shot(self) -> float:
        """Shot noise of digital camera (standard deviation) [DN]."""
        return np.sqrt(self.A / self.gain) if self.gain != 0 else 0  # average intensity is bias

    @property
    def esat(self) -> float:
        """Saturation capacity of digital camera (standard deviation) [electrons]."""
        return self._esat

    @esat.setter
    def esat(self, esat: float):
        _esat = float(max(0, esat))

        if _esat == 0:
            return

        if self._esat != _esat:
            self._esat = _esat
            self.logger.debug(f"{self._esat = }")

    @property
    def gain(self) -> float:
        """Overall system gain of digital camera [DN / electrons]."""
        return self._gain

    @gain.setter
    def gain(self, gain: float):
        _gain = min(max(0, gain), 1)

        if self._gain != _gain:
            self._gain = _gain

    @property
    def u(self) -> np.ndarray:
        """Uncertainty of measurement (standard deviation) [px]."""
        ui = np.sqrt(self.gain ** 2 * self.dark ** 2 + self.quant ** 2 + self.gain ** 2 * self.shot ** 2)  # camera noise
        SNR = self.B / ui
        upi = np.sqrt(2) / np.sqrt(self.M) / np.sqrt(self._N) / SNR  # local phase uncertainties
        upin = upi / (2 * np.pi)  # normalized local phase uncertainty
        uxi = upin * self._l  # local positional uncertainties
        ux = np.sqrt(1 / np.sum(1 / uxi ** 2))  # global phase uncertainty (by inverse variance weighting of uxi)
        return float(ux)  # convert Numpy float64 to Python float

    @property
    def DR(self) -> float:
        """Dynamic range."""
        return self.UMR / self.u

    @property
    def DRdB(self) -> float:
        """Dynamic range [dB]."""
        return 20 * np.log10(self.DR)

    @property
    def params(self) -> dict:
        """Base parameters required for en- & decoding fringe patterns."""
        # All property objects which have a setter method i.e. are (usually) not derived from others.
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
        for k, v in self.params.copy().items():  # iterating self.params ensures that only properies with a setter method are set
            if k in params and k != "T":
                setattr(self, k, params[k])

        for k, v in self.params.items():
            if k in params:
                if k in "Nlvf" and np.array(v).ndim != np.array(params[k]).ndim:
                    if not np.array_equal(v, params[k][0]):
                        self.logger.warning(
                            f"'{k}' got overwritten by interdependencies. Choose consistent parameter set."
                        )
                else:
                    if not np.array_equal(v, params[k]):
                        self.logger.warning(
                            f"'{k}' got overwritten by interdependencies. Choose consistent parameter set."
                        )

    # get default values from __init__
    defaults = dict(sorted(dict(zip(__init__.__annotations__, __init__.__defaults__)).items()))

    # restrict instance attributes to the ones listed here
    # commend the next line out or add "__dict__" to circumvent this
    __slots__ = tuple("_" + k for k in defaults.keys()) + ("logger", "_UMR", "_t",)

    # generate glossary
    glossary = {}
    for __k, __v in sorted(vars().items()):
        if not __k.startswith("_"):
            if isinstance(__v, property) and __v.__doc__ is not None:
                glossary[__k] = __v.__doc__

    # generate class docstring
    __doc__ += "\n\nParameters:\n"
    for __k, __v in sorted(vars().items()):
        if not __k.startswith("_"):
            if isinstance(__v, property) and __v.__doc__ is not None:
                __doc__ += f"    {__k}: {__v.__doc__}\n"

    # generate docstring for __init__
    __init__.__doc__ = "Parameter:\n"
    for __k, __v in sorted(vars().items()):
        if __k in defaults:
            if isinstance(__v, property) and __v.__doc__ is not None:
                __init__.__doc__ += f"    {__k} ({__init__.__annotations__[__k]}): {__v.__doc__}\n"

    del __k, __v
