from collections import namedtuple
import glob
import os
import time
import tempfile

import toml
import numpy as np
import pytest
import subprocess

from fringes import Fringes, curvature, height, __version__


def test_version():
    assert __version__, "Version is not specified."


def test_property_docs():
    f = Fringes()
    for p in dir(f):
        if isinstance(getattr(type(f), p, None), property) and getattr(type(f), p, None).fset is not None:
            assert getattr(type(f), p, None).__doc__ is not None, f"Property '{p}' has no docstring defined."


# def test_init_doc():
#     f = Fringes()
#     assert f.__init__.__doc__.count("\n") == len(f.defaults),\
#         "Not all init parameters have an associated property with a defined docstring."


def test_init():
    f = Fringes()

    for k, v in f.params.items():
        if k in "Nlvf" and f.v.ndim == 1:
            assert np.array_equal(v, f.defaults[k][0]), \
                   f"'{k}' got overwritten by interdependencies. Choose consistent default values in initialization."
        else:
            assert np.array_equal(v, f.defaults[k]) or \
                   f"'{k}' got overwritten by interdependencies. Choose consistent default values in initialization."

    for k in "Nvf":
        assert getattr(f, f"_{k}").shape == (f.D, f.K), f"Set parameter {k} hasn't shape ({f.D}, {f.K})."


def test_set_T():
    f = Fringes()

    for T in range(1, 1001 + 1):  # f._Tmax + 1
        f.T = T
        assert f.T == T, f"Couldn't set 'T' to {T}."


def test_UMR_mutual_divisibility():
    f = Fringes()
    f.l = np.array([20.2, 60.6])

    assert np.array_equal(f.UMR, [60.6] * f.D), "'UMR' is not 60.6."


def test_save_load():
    f = Fringes()
    params = f.params

    with tempfile.TemporaryDirectory() as tempdir:
        for ext in f._loader.keys():
            fname = os.path.join(tempdir, f"params{ext}")

            f.save(fname)
            assert os.path.isfile(fname), "No params-file saved."

            params_loaded = f.load(fname)
            # assert len(params_loaded) == len(params), "A different number of attributes is loaded than saved."

            for k in params_loaded.keys():
                assert k in params, f"Fringes class has no attribute '{k}'"
                assert params_loaded[k] == params[k], \
                    f"Attribute '{k}' in file '{fname}' differs from its corresponding instance attribute."

            for k in params.keys():
                assert k in params_loaded or ext == ".toml" and params[k] is None, f"File '{fname}' has no attribute '{k}'"  # in toml there exists no None
                if k in params_loaded:
                    assert params[k] == params_loaded[k], \
                        f"Instance attribute '{k}' differs from its corresponding attribute in file '{fname}'."


def test_coordinates():
    f = Fringes()
    uv = f.coordinates()
    assert np.array_equal(uv, np.indices((f.Y, f.X))[::-1, :, :, None]), "XY-coordinates are wrong."

    f = Fringes(Y=1)
    uv = f.coordinates()
    assert np.array_equal(uv[0], np.arange(f.X)[None, :, None]), "X-coordinates are wrong."

    f = Fringes(X=1)
    uv = f.coordinates()
    assert np.array_equal(uv[0], np.arange(f.Y)[:, None, None]), "Y-coordinates are wrong."


def test_encoding():
    f = Fringes(Y=100)

    I = f.encode()
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."

    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_encoding_one_direction():
    f = Fringes(Y=100)
    f.axis = 0
    f.D = 1

    I = f.encode()
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."

    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."

    f = Fringes(X=100, Y=1920)
    f.axis = 1
    f.D = 1

    I = f.encode()
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."

    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_encoding_call():
    f = Fringes(Y=100)

    I = f()
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."

    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_encoding_iter():
    f = Fringes(Y=100)

    for t, I in enumerate(f):
        assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
        assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."
    assert t + 1 == f.T, "Number of iterations does't equal number of frames."

    I = np.array(list(frame[0] for frame in f))
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence isn't 4-dimensional."

    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_encoding_frames(fringes=Fringes(Y=100)):
    f = fringes
    
    I = np.array([f.encode(t)[0] for t in range(f.T)])
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence isn't 4-dimensional."

    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_decoding(rm: bool = False):  # todo: rm = True i.e. test decoding time
    f = Fringes(Y=100)
    f.verbose = True

    I = f.encode()

    # test numba compile time
    if rm:
        flist = glob.glob(os.path.join(os.path.dirname(__file__), "..", "fringes", "__pycache__", "decoder*decode*.nbc"))
        for file in flist:
            os.remove(file)
        t0 = time.perf_counter()
        dec = f.decode(I)
        t1 = time.perf_counter()
        assert t1 - t0 < 10 * 60, f"Numba compilation takes longer than 10 minutes: {(t1 - t0) / 60} minutes."

    dec = f.decode(I)

    # assert isinstance(dec, namedtuple), "Return value isn't a 'namedtuple'."  # todo: check for named tuple
    assert all(isinstance(item, np.ndarray) for item in dec), "Return values aren't 'Numpy arrays'."

    # assert np.max(dec.residuals) < 1, "Residuals are larger than 1."  # todo

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."

    if f.mode == "precise":  # todo: decide for each pixel individually i.e. multiply with B later on
        w = f._N * f._v ** 2  # weights of phase measurements are their inverse variances (must be multiplied with m later on)
        idx = [np.argmax(w[d] * (f._N[d] > 2) * (f._v[d] > 0)) for d in range(f.D)]  # the set with most precise phases
    # elif mode == "robust":  # todo: "exhaustive"?
    #     NotImplemented  # todo: try neighboring fringe orders / all permutations = exhaustive search?
    else:  # fast (fallback)
        w = np.ones((f.D, f.K), dtype=np.int_)
        idx = [np.argmax(1 / f._v[d] * (f._N[d] > 2) * (f._v[d] > 0)) for d in range(f.D)]  # the set with the least number of periods
    fid = f.coordinates() // np.array([f.L / f._v[d, idx[d]] for d in range(f.D)])[:, None, None, None]
    #assert np.allclose(dec.orders[[d * f.K + idx[d] for d in range(f.D)]], fid, atol=0), "Errors in fringe orders."


def test_decoding_despike():
    f = Fringes(Y=100)
    I = f.encode()
    I[:, 10, 5, :] = I[:, -5, -10, :]

    dec = f.decode(I, despike=True)
    d = dec.registration - f.coordinates()
    print(d.max())
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_deinterlacing():
    f = Fringes(Y=100)

    I = f.encode()
    I = I.swapaxes(0, 1).reshape(-1, f.T, f.X, f.C)  # interlace
    I = f.deinterlace(I)
    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_alpha():
    f = Fringes(Y=100)
    f.alpha = 1.1

    I = f.encode()
    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.2), "Registration is off more than 0.2."  # todo: 0.1
    assert all([dec.registration[d].max() <= f.R[d] for d in range(f.D)]), "Registration values are larger than R."


def test_dtypes():
    f = Fringes(Y=100)

    for dtype in f._dtypes:
        f.dtype = dtype

        I = f.encode()

        assert I.dtype == f.dtype, "Wrong dtype."


# def test_grids():
#     f = Fringes(Y=100)
#
#     for g in f._grids:
#         f.grid = g
#
#         I = f.encode()
#         dec = f.decode(I)
#
#         d = dec.registration - f.coordinates()
#         assert np.allclose(d, 0, atol=0.1), f"Registration is off more than 0.1 for grid '{f.grid}'."  # todo: fix grids
#
#         # todo: test angles

def test_modes():
    f = Fringes(Y=10)

    I = f.encode()

    for mode in f._modes:
        f.mode = mode

        dec = f.decode(I)

        d = dec.registration - f.coordinates()
        assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


# def test_scaling():
#     f = Fringes(Y=100)
#     f.K = 1
#
#     f.v = 1
#     I = f.encode()
#     dec = f.decode(I)
#
#     d = dec.registration - f.coordinates()
#     # std = np.std(d)
#     # a = 1
#     #assert np.allclose(d, 0, atol=0.5), "Registration is off more than 0.5."  # todo: 0.1


def test_unwrapping():
    f = Fringes()
    f.K = 1
    f.v = 13

    I = f.encode()
    dec = f.decode(I)

    for d in range(f.D):
        grad = np.gradient(dec.registration[d, :, :, 0], axis=1 - d)
        assert np.allclose(grad, 1, atol=0.1), "Gradient of unwrapped phase map isn't close to 0.1."


def test_unwrapping_class_method():
    f = Fringes(Y=100)
    f.K = 1
    f.v = 13

    I = f.encode()
    dec = f.decode(I, verbose=True)
    x = Fringes.unwrap(dec.phase)

    for d in range(f.D):
        grad = np.gradient(x[d, :, :, 0], axis=1 - d)
        assert np.allclose(grad, 0, atol=0.1), "Gradient of unwrapped phase map isn't close to 0.1."


def test_remapping():
    f = Fringes(Y=100)
    f.Y /= 2
    f.verbose = True

    I = f.encode()
    dec = f.decode(I)

    source = f.remap(dec.registration)
    assert np.all(source == 1), "Source doesn't contain only ones."

    source = f.remap(dec.registration, dec.modulation)
    assert np.allclose(source, 1, atol=0.1), "Source doesn't contain only values close to one."


def test_curvature_height():
    f = Fringes(Y=100)

    I = f.encode()
    dec = f.decode(I)

    c = curvature(dec.registration)
    assert np.allclose(c[1:, 1:], 0, atol=0.1), "Curvature if off more than 0.1."  # todo: boarder

    h = height(c)
    assert np.allclose(h[:, 1:], 0, atol=0.1), "Height if off more than 0.1."  # todo: boarder


def test_hues():
    f = Fringes(Y=100)
    f.h = "rggb"

    I = f.encode()
    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."

    f.H = 3  # todo: 2 and take care of M not being a scalar

    I = f.encode()
    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_averaging():
    f = Fringes(Y=100)
    f.M = 2

    I = f.encode()
    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_WDM():
    f = Fringes(Y=100)
    f.N = 3
    f.WDM = True

    I = f.encode()
    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


# def test_SDM():
#     f = Fringes(Y=100)
#     f.SDM = True
#
#     I = f.encode()
#     dec = f.decode(I)
#
#     for d in range(f.D):
#         grad = np.gradient(dec.registration[d, 1:-1, 1:-1, 0], axis=1 - d)  # todo: boarder
#         #assert np.allclose(grad, 1, atol=0.1), "Gradient of registration isn't close to 1."
#
#     d = dec.registration[:, 1:-1, 1:-1, :] - f.coordinates()[:, 1:-1, 1:-1, :]  # todo: boarder
#     assert np.allclose(d, 0, atol=0.5), "Registration is off more than 0.5."


# def test_SDM_WDM():
#     f = Fringes(Y=100)
#     f.N = 3
#     f.SDM = True
#     f.WDM = True
#
#     I = f.encode()
#     dec = f.decode(I)
#
#     for d in range(f.D):
#         grad = np.gradient(dec.registration[d, 1:-1, 1:-1, 0], axis=1 - d)  # todo: boarder
#         #assert np.allclose(grad, 1, atol=0.1), "Gradient of registration isn't close to 1."
#
#     d = dec.registration[:, 1:-1, 1:-1, :] - f.coordinates()[:, 1:-1, 1:-1, :]  # todo: boarder
#     assert np.allclose(d, 0, atol=0.5), "Registration is off more than 0.5."  # todo: 0.1


def test_FDM():
    f = Fringes(Y=100)
    f.FDM = True

    I = f.encode()
    dec = f.decode(I)

    d = dec.registration[:, 1:, 1:, :] - f.coordinates()[:, 1:, 1:, :]  # todo: boarder
    assert np.allclose(d, 0, atol=0.5), "Registration is off more than 0.5."  # todo: 0.1

    f.static = True
    f.N = 1
    I = f.encode()
    dec = f.decode(I)

    d = dec.registration[:, 1:, 1:, :] - f.coordinates()[:, 1:, 1:, :]  # todo: boarder
    assert np.allclose(d, 0, atol=0.2), "Registration is off more than 0.2."  # todo: 0.1


def test_simulation():
    # todo:
    #  no low pass from PSF causing decrease in modulation
    #  no clipping
    #  no hit pixels -> add salt and pepper noise?

    f = Fringes()
    f.V = 0.8
    f.D = 1
    f.X = 832
    f.Y = 2003
    f.N = 4
    f.l = 9, 10, 11

    I = f.encode()
    In = f._simulate(I)
    dec = f.decode(In)
    # d = dec.registration[:, 1:, 1:, :] - f.coordinates()[:, 1:, 1:, :]  # todo: boarder
    dabs = np.abs(dec.registration - f.coordinates())
    dmed = np.nanmedian(dabs)
    davg = np.nanmean(dabs)
    dmax = np.nanmax(dabs)
    assert np.allclose(dmed, 0, atol=0.1), "Median of Registration is off more than 0.1."
    # assert np.allclose(d, 0, atol=0.5), "Registration is off more than 0.5."  # todo: 0.1?


if __name__ == "__main__":
    # todo: test verbose output

    # f = Fringes()
    # f.l = "1, 2, 3"

    # pytest.main()
    subprocess.run(['pytest', '--tb=short', str(__file__)])
