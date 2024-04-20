import os
import toml
import importlib
import logging
import glob
import webbrowser
import argparse

from .fringes import Fringes
from .util import vshape, curvature, height

logger = logging.getLogger(__name__)

_flist = glob.glob(os.path.join(os.path.dirname(__file__), "__pycache__", "decoder*decode*.nbc"))
if not _flist or os.path.getmtime(os.path.join(os.path.dirname(__file__), "decoder.py")) > max(
    os.path.getmtime(file) for file in _flist
):
    logging.warning(
        "The 'decode()'-function has not been compiled yet. "
        "This will take a few minutes (the time depends on your CPU and energy settings)."
    )

# use verion string in pyproject.toml as the single source of truth
try:
    # in order not to confuse an installed version of a package with a local one,
    # first try the local one (not being installed)
    _meta = toml.load(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"))
    __version__ = _meta["project"]["version"]  # Python Packaging User Guide expects version here
except KeyError:
    __version__ = _meta["tool"]["poetry"]["version"]  # Poetry expects version here
except FileNotFoundError:
    __version__ = importlib.metadata.version("fringes")  # installed version


def documentation():
    fname = os.path.join(os.path.dirname(__file__), "..", "docs", "_build", "index.html")
    if os.path.isfile(fname):
        webbrowser.open_new_tab(os.path.join(os.path.dirname(__file__), "..", "docs", "_build", "index.html"))
    else:
        webbrowser.open_new_tab("https://fringes.readthedocs.io")


def main():
    # todo: argparser
    parser = argparse.ArgumentParser
    for p in Fringes.params:
        parser.add_argument(
            "-X",
            "--width",
        )

    args = parser.parse_args()

    q = 1


if __name__ == "__main__":
    main()
