import importlib
import os
import toml

from .fringes import Fringes
from .util import vshape, curvature, height

try:  # PackageNotFoundError
    fname = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    version = toml.load(fname)["tool"]["poetry"]["version"]
except FileNotFoundError or KeyError:
    version = importlib.metadata.version("fringes")

__version__ = version
