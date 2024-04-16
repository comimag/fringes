# Fringes
[![PyPI](https://img.shields.io/pypi/v/fringes)](https://pypi.org/project/fringes/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fringes)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Read the Docs](https://img.shields.io/readthedocs/fringes)](https://fringes.readthedocs.io)
[![PyPI - License](https://img.shields.io/pypi/l/fringes)](https://github.com/comimag/fringes/blob/main/LICENSE.txt)
[![Static Badge](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.10936353-blue)](https://zenodo.org/doi/10.5281/zenodo.10936353)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/fringes)](https://pypistats.org/packages/fringes)

<!---
# todo: DOI
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/comimag/fringes/python-package.yml)
![GitHub top language](https://img.shields.io/github/languages/top/comimag/fringes)
![https://img.shields.io/badge/python-%3E=3.9-blue](https://img.shields.io/badge/python-%3E=3.9-blue)
![GitHub](https://img.shields.io/github/license/comimag/fringes)
[![Downloads](https://static.pepy.tech/badge/fringes)](https://pepy.tech/project/fringes)
--->

<!---
link to  paper, please cite
--->

Easily create customized fringe patterns
and analyse them using phase shifting algorithms.

![coding-cheme}](https://raw.githubusercontent.com/comimag/fringes/main/docs/getting_started/coding-scheme.gif)\
Figure 1: Phase Shifting Coding Scheme.

## Features
- [Parameterize](https://fringes.readthedocs.io/en/latest/user_guide/params.html) the phase shifting algorithm
- [Create](https://fringes.readthedocs.io/en/main/getting_started/fundamentals.html#encoding) and
  [decode](https://fringes.readthedocs.io/en/main/getting_started/fundamentals.html#decoding)
  customized fringe patterns
- Generalized Temporal Phase Unwrapping (GTPU)
- Uncertainty Propagation
- [Optimal Coding Strategy](https://fringes.readthedocs.io/en/latest/user_guide/optimal.html)
- [Multiplexing](https://fringes.readthedocs.io/en/latest/user_guide/mux.html)
- Compute [curvature maps](https://fringes.readthedocs.io/en/latest/user_guide/filter.html#curvature)
- Use many more [filter](https://fringes.readthedocs.io/en/latest/user_guide/filter.html) methods

## Installation
You can install `fringes` directly from [PyPi](https://pypi.org/) via `pip`:

```
pip install fringes
```

## Usage
You instantiate, parameterize and deploy the `Fringes` class:

```python
import fringes as frng  # import module

f = frng.Fringes()      # instantiate class

f.X = 1920              # set width of the fringe patterns
f.Y = 1080              # set height of the fringe patterns
f.K = 2                 # set number of sets
f.N = 4                 # set number of shifts
f.v = [9, 10]           # set spatial frequencies
f.T                     # get number of frames
                            
I = f.encode()          # encode fringe patterns
A, B, X = f.decode(I)   # decode fringe patterns
```

All [parameters](https://fringes.readthedocs.io/en/latest/user_guide/params.html)
are accesible as class properties (managed attributes) of the `Fringes` instance.

For generating the fringe pattern sequence `I`, use the method `encode()`.
It returns a [NumPy array](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) in videoshape (frames, height, width, color channels).

For analyzing (recorded) fringe patterns, use the method `decode()`.
It returns the Numpy arrays brightness `A`, modulation `B` and coordinate `x`.

> Note:\
For the compitationally expensive ``decoding`` we make use of the just-in-time compiler [Numba](https://numba.pydata.org/).
During the first execution, an initial compilation is executed. 
This can take several tens of seconds up to single digit minutes, depending on your CPU and energy settings.
However, for any subsequent execution, the compiled code is cached and the code of the function runs much faster, 
approaching the speeds of code written in C.

## Graphical User Interface
Do you need a GUI? `Fringes` has a sister project which is called `Fringes-GUI`:

https://pypi.org/project/fringes-gui/

## Documentation
The documentation can be found here:

https://fringes.readthedocs.io

## License
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License

## Citation
If you use this software, please cite it using this DOI:
[10.5281/zenodo.10936353](https://zenodo.org/doi/10.5281/zenodo.10936353)\
This DOI represents all versions, i.e. the concept of this software package,
and will always resolve to the latest one.

If you want to cite a specific version,
please choose the respective DOI from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10936353).

## Support
I was looking for a user-friendly tool to configure,
encode and decode customized fringe patterns with phase shifting algorithms.
Since I couldn't find any, I started developing one myself.
It is intended for [non-commercial](#license), academic and educational use.

However, I do this entirely in my free time.
If you like this package and can make use of it, I would be happy about a donation.
It will help me keep it up-to-date and adding more features in the future.

<!---
[![Liberapay](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/comimag/donate/)
[![](https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=EHBGZ229DKUC4)
--->

[![paypal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=PayPal&logoColor=white)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=EHBGZ229DKUC4)

Thank you!
