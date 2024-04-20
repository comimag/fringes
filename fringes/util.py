import logging
import time
import typing as tp
import itertools as it

import numpy as np
import numba as nb
import scipy as sp

# import sympy.ntheory.generate
import skimage as ski
import cv2

logger = logging.getLogger(__name__)


def vshape(data: np.ndarray) -> np.ndarray:
    """Standardizes the input data shape.

    Transforms video data into the standardized shape (T, Y, X, C), where
    T is number of frames, Y is height, X is width, and C is number of color channels.

    Parameters
    ----------
    data : ndarray
        Input data of arbitrary shape.

    Returns
    -------
    videodata : ndarray
        Standardized version of data, in shape (T, Y, X, C), where
        T is number of frames, Y is height, X is width, and C is number of color channels.

    Notes
    -----
    Ensures that the array becomes 4-dimensional
    and that the length of the last dimension is in (1, 3, 4).
    To do this, leading dimensions may be flattened.

    Examples
    --------
    >>> import fringes as frng

    >>> data = np.ones(100)
    >>> videodata = frng.vshape(data)
    >>> videodata.shape
    (100, 1, 1, 1)

    >>> data = np.ones(1200, 1920)
    >>> videodata = frng.vshape(data)
    >>> videodata.shape
    (1, 1200, 1820, 1)

    >>> data = np.ones(1200, 1920, 3)
    >>> videodata = frng.vshape(data)
    >>> videodata.shape
    (1, 1200, 1820, 3)

    >>> data = np.ones(100, 1200, 1920)
    >>> videodata = frng.vshape(data)
    >>> videodata.shape
    (100, 1200, 1820, 1)

    >>> data = np.ones(100, 1200, 1920, 3)
    >>> videodata = frng.vshape(data)
    >>> videodata.shape
    (100, 1200, 1820, 3)

    >>> data = np.ones(2, 3, 4, 1200, 1920)
    >>> videodata = frng.vshape(data)
    >>> videodata.shape
    (24, 1200, 1820, 1)
    """

    if data.ndim == 0:
        data = data.reshape(1)  # returns a view

    channels = (1, 3, 4)  # possible number of color channels

    if data.ndim > 4:
        if data.shape[-1] in channels:
            T = np.prod(data.shape[:-3])
            Y, X, C = data.shape[-3:]
        else:
            T = np.prod(data.shape[:-2])
            Y, X = data.shape[-2:]
    elif data.ndim == 4:
        if data.shape[-1] in channels:
            T, Y, X, C = data.shape
        else:
            T = np.prod(data.shape[:2])
            X, Y = data.shape[2:]
            C = 1
    elif data.ndim == 3:
        if data.shape[-1] in channels:
            T = 1
            Y, X, C = data.shape
        else:
            T, Y, X = data.shape
            C = 1
    elif data.ndim == 2:
        T = C = 1
        Y, X = data.shape
    elif data.ndim == 1:
        T = data.shape

    return data.reshape(T, Y, X, C)  # returns a view


def circular_distance(a: np.ndarray, b: np.ndarray, c: float) -> np.ndarray:
    """Circular distance.

    Parameters
    ----------
    a : np.ndarray
        Start points.
    b : np.ndarray
        End points.
    c : float
        Circumference (distance) after which wrapping occurs.

    Returns
    -------
    d : np.ndarray
        Circular distance from a to b.

    Notes
    -----
    For more details, see https://ieeexplore.ieee.org/document/9771407.
    """

    d = c / 2 - np.abs(c / 2 - (a - b))  # todo: check

    return d


# @nb.jit(cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)
# def circ_dist(a, b, c) -> float:
#     d = b - a
#     dmax = c / 2
#
#     # return dmax - np.abs(dmax - d)
#
#     if d > dmax:
#         d -= c
#     elif d < -dmax:
#         d += c
#     return d


def curvature(s: np.ndarray, calibrated: bool = False, normalize: bool = True) -> np.ndarray:  # todo: test
    """Mean curvature map.

    Combuted by differentiating a slope map.

    Parameters
    ----------
    s : np.ndarray
        Slope map.
        It is reshaped to videoshape (frames 'T', height 'Y', width 'X', color channels 'C') before processing.
    calibrated : bool, optional
        Flag indicating whether the input data 's' originates from a calibrated measurement.
        Default is False.
        If this is False, the median value of the computed curvature map is added as an offset,
        so the median value of the final curvature map becomes zero.
    normalize : bool
        Flag indicating whether to use the acrtangent function
        to nonlinearly map the codomain from [-inf, inf] to [-1, 1].
        Default is True.

    Returns
    -------
    c : np.ndarray
        Curvature map.
    """

    t0 = time.perf_counter()

    T, Y, X, C = vshape(s).shape
    s = s.reshape(T, Y, X, C)  # returns a view

    assert T == 2, "Number of direction doesn't equal 2."
    assert X >= 2 and Y >= 2, "Shape too small to calculate numerical gradient."

    c = np.gradient(s[0], axis=0) + np.gradient(s[0], axis=1) + np.gradient(s[1], axis=0) + np.gradient(s[1], axis=1)

    if not calibrated:
        # c -= np.mean(c, axis=(0, 1))
        c -= np.median(c, axis=(0, 1))  # Median is a robust estimator for the mean.

    if map:
        c = np.arctan(c) * 2 / np.pi  # scale [-inf, inf] to [-1, 1]

    logger.info(f"{int(1000 * (time.perf_counter() - t0))}ms")

    return c.reshape(-1, Y, X, C)


def height(curv: np.ndarray, iterations: int = 3) -> np.ndarray:  # todo: test
    """Local height map.

    It is computed by iterative local integration via an inverse laplace filter.
    Think of it as a relief, where height is only relative to the local neighborhood.

    Parameters
    ----------
    curv : np.ndarray
        Curvature map.
        It is reshaped to videoshape (frames 'T', height 'Y', width 'X', color channels 'C') before processing.
    iterations : int, optional
        Number of iterations of the inverse Laplace filter kernel.
        Default is 3.

    Returns
    -------
    z : np.ndarray
        Local height map.
    """

    t0 = time.perf_counter()

    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
    # k *= iterations  # todo

    T, Y, X, C = vshape(curv).shape
    curv = curv.reshape(T, Y, X, C)  # returns a view

    if T == 1:
        curv = np.squeeze(curv, axis=0)  # returns a view

    if curv.min() == curv.max():
        return np.zeros_like(curv)

    z = np.zeros_like(curv)
    for c in range(C):
        for i in range(iterations):
            z[..., c] = (cv2.filter2D(z[..., c], -1, k) - curv[..., c]) / 4

    # todo: residuals
    # filter2(kernel_laplace, z) - cature;

    logger.info(f"{int(1000 * (time.perf_counter() - t0))}ms")

    return z.reshape(-1, Y, X, C)


def bilateral(I: np.ndarray, k: int = 7) -> np.ndarray:  # todo: test
    """Bilateral filter.

    Edge-preserving, denoising filter.

    Parameters
    ----------
    I : np.ndarray
        Input data.
        It is reshaped to videoshape (frames 'T', height 'Y', width 'X', color channels 'C') before processing.
    k : int, optional
        Size of the filter kernel.
        Default is 7.

    Returns
    -------
    out : np.ndarray
        Filtered data.
    """

    t0 = time.perf_counter()

    T, Y, X, C = vshape(I).shape
    I = I.reshape(T, Y, X, C)
    out = np.empty_like(I)

    for t in range(T):
        if C in [1, 3]:
            rv = ski.restoration.denoise_bilateral(I[t], win_size=k, sigma_spatial=1, channel_axis=-1)
            out[t] = rv if C == 3 else rv[..., None]
            # out[t] = cv2.bilateralFilter(I[t], d=k, sigmaColor=np.std(I[t]), sigmaSpace=1)
        else:
            for c in range(C):
                out[t, :, :, c] = ski.restoration.denoise_bilateral(I[t, :, :, c], win_size=k, sigma_spatial=1)
                # out[t, :, :, c] = cv2.bilateralFilter(I[t], d=k, sigmaColor=np.std(I[t, :, :, c]), sigmaSpace=1)

    logger.debug(f"{int(1000 * (time.perf_counter() - t0))}ms")

    return out


@nb.jit(cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)
def _remap_legacy(
    reg: np.ndarray,
    mod: np.ndarray = np.ones(1),
    scale: float = 1,
    Y: int = 0,
    X: int = 0,
    C: int = 0,
) -> np.ndarray:
    if mod.ndim > 1:
        assert reg.shape[1:] == mod.shape[1:]

    if reg.shape[0] == 1:
        # mod = np.vstack(mod, np.zeros_like(mod))
        reg = np.vstack((reg, np.zeros_like(reg)))  # todo: axis

    if X is None:
        X = 0

    if Y is None:
        Y = 0

    X = int(X)
    Y = int(Y)

    if scale <= 0:
        scale = 1

    if Y <= 0:
        Y = max(1, int(np.nanmax(reg[1]) * scale + 0.5))
    else:
        Y = int(Y * scale + 0.5)

    if X <= 0:
        X = max(1, int(np.nanmax(reg[0]) * scale + 0.5))
    else:
        X = int(X * scale + 0.5)

    if C not in [1, 3, 4]:
        if reg.shape[-1] in [3, 4]:
            C = reg.shape[-1]
        else:
            C = 1
            # reg = reg.reshape([s for s in reg.shape] + [C])  # todo: how to get reg[..., 1] if C-axis doesn't exist?

    src = np.zeros((Y, X, C), np.float32)

    Xc = reg.shape[2]
    Yc = reg.shape[1]
    DK = mod.shape[0]
    for xc in nb.prange(Xc):
        for yc in nb.prange(Yc):
            for c in nb.prange(C):
                if not np.isnan(reg[0, yc, xc, c]):
                    xs = int(reg[0, yc, xc, c] * scale + 0.5)  # i.e. rint()
                    if xs < X:
                        if not np.isnan(reg[1, yc, xc, c]):
                            ys = int(reg[1, yc, xc, c] * scale + 0.5)  # i.e. rint()
                            if ys < Y:
                                for dk in nb.prange(DK):
                                    if mod.ndim > 1:
                                        m = mod[dk, yc, xc, c]
                                        if not np.isnan(m):
                                            src[ys, xs, c] += m
                                    else:
                                        src[ys, xs, c] += 1

    mx = src.max()
    if mx > 0:
        src /= mx

    return src


@nb.jit(cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)
def _remap(
    src: np.ndarray,
    reg: np.ndarray,
    mod: np.ndarray = np.ones(1),
) -> np.ndarray:
    Ys, Xs, Cs = src.shape
    Yc, Xc, Cc = reg.shape[1:]

    for xc in nb.prange(Xc):
        for yc in nb.prange(Yc):
            for c in nb.prange(Cs):
                if not np.isnan(reg[0, yc, xc, c]):
                    xs = int(reg[0, yc, xc, c] + 0.5)  # i.e. rint()

                    if xs < Xs:
                        if not np.isnan(reg[1, yc, xc, c]):
                            ys = int(reg[1, yc, xc, c] + 0.5)  # i.e. rint()

                            if ys < Ys:
                                # if mod.ndim > 1 and not np.isnan(mod[yc, xc, c]):  # todo: test shape
                                #     m = mod[yc, xc, c]
                                #     src[ys, xs, c] += m
                                # else:
                                #     src[ys, xs, c] += 1

                                if not np.isnan(mod[yc, xc, c]):
                                    m = mod[yc, xc, c]
                                    src[ys, xs, c] += m
    return src


# # @nb.jit(cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)  # todo
# def filter(combos, K, L, lmin):
#     kroot = L ** (1 / K)
#     if lmin <= kroot:
#         lcombos = np.array([l for l in combos if np.any(l > kroot) and np.lcm.reduce(l) >= L])
#     else:
#         lcombos = np.array([l for l in combos if np.lcm.reduce(l) >= L])
#
#     return lcombos
#
#
# def coprime(n: list[int] | tuple[int] | np.ndarray) -> bool:  # n: iterable  # todo: extend to rational numbers
#     """Test whether numbers are pairwise co-prime.
#
#     Parameters
#     ----------
#     n : list, tuple, np.ndarray
#         Integer numbers.
#
#     Returns
#     -------
#     iscoprime : bool
#         True if numbers are pairwise co-prime, else False.
#     """
#
#     # convert to array and flatten
#     n = np.array(n).ravel()  # returns a view
#
#     # check whether iterable has entries
#     if n.size == 0:
#         return False
#
#     # check whether numbers are integers
#     if not np.all([i % 1 == 0 for i in n]):
#         return False
#
#     # ensure numbers are integers
#     if n.dtype != int:
#         n = n.astype(int, copy=False)
#
#     # check pairwise for coprimality, i.e. gcd(a, b) == 1
#     for i in range(n.size):  # each combination; number of combinations = n.size * (n.size - 1) / 2
#         for j in range(i + 1, n.size):
#             if np.gcd(n[i], n[j]) != 1:
#                 return False
#
#     # alternatively: np.lcm.reduce(n) == np.prod(n)
#
#     return True
#
#
# def extgcd(a, b):
#     """Erweiterter euklidischer Algorithmus.
#
#     https://hwlang.de/krypto/algo/euklid-erweitert.htm
#
#     Parameters
#     ----------
#     a : int
#         Ganzzahl.
#
#     b : int
#         Ganzzahl.
#
#     Returns
#     -------
#     a : int
#         Größter gemeinsamer Teiler von 'a' und 'b'.
#
#     u : int
#         Koeffizienten 'u' und 'v' einer Darstellung von 'a' als ganzzahlige Linearmbination.
#
#     v : int
#         Koeffizienten 'u' und 'v' einer Darstellung von 'a' als ganzzahlige Linearmbination.
#     """
#
#     u, v, s, t = 1, 0, 0, 1
#     while b != 0:
#         q = a // b
#         a, b = b, a - q * b
#         u, s = s, u - q * s
#         v, t = t, v - q * t
#     return a, u, v
#
#
# def modinverse(a, n):
#     """Berechnet das multiplikativ inverse Element von a modulo n.
#
#     https://hwlang.de/krypto/grund/inverses-element.htm
#
#     Parameters
#     ----------
#     a : int
#         Ganzzahl.
#
#     n : int
#         Modul.
#
#     Returns
#     -------
#     mi : int
#         Multiplikativ inverse Element von 'a' modulo 'n'.
#     """
#
#     g, u, v = extgcd(a, n)
#     return u % n
#
#
# def chineseRemainder(nn, rr):
#     """Chinesischer-Restsatz-Algorithmus.
#
#     Der Vorteil dieser Implementierung nach dem Divide-and-Conquer-Prinzip besteht darin,
#     dass in den unteren Rekursionsebenen viele Berechnungen mit kleinen Zahlen stattfinden
#     und erst in den oberen Rekursionsebenen wenige Berechnungen mit großen Zahlen.
#
#     https://hwlang.de/krypto/algo/chinese-remainder.htm
#
#     Parameters
#     ----------
#     nn : np.ndarray, list
#         Liste paarweise teilerfremder Moduln.
#
#     rr : np.ndarray, list
#         Liste der Reste.
#
#     Returns
#     -------
#     mn : int
#         Produkt der Moduln.
#
#     x : int
#         Zahl x nach dem chinesischen Restsatz.
#     """
#
#     if len(nn) == 1:
#         return nn[0], rr[0]
#     else:
#         k = len(nn) // 2
#         m, a = chineseRemainder(nn[:k], rr[:k])
#         n, b = chineseRemainder(nn[k:], rr[k:])
#         g, u, v = extgcd(m, n)
#         x = (b - a) * u % n * m + a
#         return m * n, x
#
#
# def modinv(x, p):
#     """Modular multiplicative inverse.
#
#     y = invmod(x, p) such that x*y == 1 (mod p)
#
#     https://bugs.python.org/issue36027
#
#     Parameters
#     ----------
#
#     x : int
#         Integer.
#
#     p : int
#         Modul.
#
#     Returns
#     -------
#     y : Modular multiplicative inverse.
#     """
#     return pow(x, -1, p)


if __name__ == "__main__":
    pass
