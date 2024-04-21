import logging

import numpy as np

logger = logging.getLogger(__name__)


def img(Y: int = 720, X: int = 720, a: float = 0):
    yy, xx = np.indices((Y, X))
    return rot(xx, yy, a)


def cart(Y: int = 720, X: int = 720, a: float = 0):
    x = np.linspace(-(X - 1) / 2, (X - 1) / 2, X, endpoint=True)
    y = np.linspace((Y - 1) / 2, -(Y - 1) / 2, Y, endpoint=True)
    xx, yy = np.meshgrid(x, y, copy=False)
    return rot(xx, yy, a)


def pol(Y: int = 720, X: int = 720, a: float = 0):
    xx, yy = cart(Y, X)
    pp = np.arctan2(yy, xx) / (2 * np.pi)
    rr = np.sqrt(xx**2 + yy**2) / min(X, Y)
    return rot(pp, rr, a)


def logpol(Y: int = 720, X: int = 720, a: float = 0):
    pp, rr = pol(Y, X)
    ll = np.log(rr) / (2 * np.pi)
    return rot(pp, ll, a)


def rot(uu, vv, a):
    if a % 360 == 0:
        return uu, vv

    t = np.deg2rad(a)

    if a % 90 == 0:
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[c, -s], [s, c]])
        # R = np.matrix([[c, -s], [s, c]])
        ur = R[0, 0] * uu + R[0, 1] * vv
        vr = R[1, 0] * uu + R[1, 1] * vv
        # u = np.dot(uu, R)  # todo: matrix multiplication
        # v = np.dot(vv, R)
    else:
        ur = uu - vv * np.tan(t)
        vr = uu + vv / np.tan(t)

    return ur, vr


def innercirc(Y: int = 720, X: int = 720):
    """Boolean mask with True values inside inscribed circle."""
    return pol(Y, X)[1] <= 0.5
