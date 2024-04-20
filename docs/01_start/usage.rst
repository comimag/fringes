Usage
=====
This package provides the handy ``Fringes`` class,
which handles all the required parameters
for configuring fringe pattern sequences
and provides methods for fringe analysis.

Command-line use
----------------
You instantiate, parameterize and deploy the ``Fringes`` class:

.. code-block:: python

    import fringes as frng      # import the fringes package

    f = frng.Fringes()          # instantiate the Fringes class


All :ref:`parameters <parameters>` are accesible by the respective attributes of the ``Fringes`` instance
(a glossary of them is obtained by the class attribute ``glossary``).
They are implemented as class properties (managed attributes),
which are parsed when setting,
so usually several input types are accepted
(e.g. `bool`, `int`, `float` for scalars
and additionally `list`, `tuple`, `ndarray` for `arrays <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_).
Note that some attributes have subdependencies,
hence dependent attributes might change as well.
Circular dependencies are resolved automatically.

.. code-block:: python

    f.X = 1920                  # set width of the fringe patterns
    f.Y = 1080                  # set height of the fringe patterns
    f.K = 2                     # set number of sets
    f.N = 4                     # set number of shifts
    f.v = [9, 10]               # set spatial frequencies

    f.T                         # get number of frames

For generating the fringe pattern sequence ``I``, use the method ``encode()``.
It returns a `Numpy array <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
in videoshape (frames ``T``, width ``X``, height ``Y``, color channels ``C``).

.. code-block:: python

    I = f.encode()              # encode fringe patterns

For analyzing (recorded) fringe patterns, use the method ``decode()``.
It returns the Numpy arrays brightness ``A``, modulation ``B`` and coordinate ``X``.

.. code-block:: python

    A, B, X = f.decode(I)       # decode fringe patterns


.. note::
  For the compitationally expensive ``decode()``-function
  we make use of the just-in-time compiler `Numba <https://numba.pydata.org/>`_.
  During the first execution, an initial compilation is executed.
  This can take several tens of seconds up to single digit minutes, depending on your CPU and energy settings.
  However, for any subsequent execution, the compiled code is cached and the code of the function runs much faster,
  approaching the speeds of code written in C.

In your application, you can configure the
`logging <https://docs.python.org/3/howto/logging.html#advanced-logging-tutorial>`_
of the ``fringes`` package like so:

.. code-block:: python

    import logging

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s.%(funcName)s(): %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger("fringes")
    logger.setLevel("INFO")


Graphical User Interface
------------------------
Do you prefer to interact with a GUI?
`Fringes` has a sister project which is called `Fringes-GUI`: https://pypi.org/project/fringes-gui/

You can install `Fringes-GUI` directly from `PyPi <https://pypi.org/>`_ with ``pip``::

    pip install fringes-gui


Then you import the ``fringes-gui`` package and call the function ``run()``.

.. code-block:: python

    import fringes_gui as fgui
    fgui.run()

Now the graphical user interface should appear:

.. figure:: GUI.png
    :align: center
    :alt: gui

    Screenshot of the GUI.

Attributes
""""""""""
In the top left corner the attribute widget is located.
It contains the parameter tree which contains all the :ref:`attributes <parameters>` of the `Fringes` class.
If you select a parameter and hover over it, a tool tip will appear,
containing the docstring of the respective attribute of the `Fringes` class.

The visibility does not affect the functionality of the parameters
but is used by the GUI to decide which parameters to display based on the current visibility level.
The purpose is mainly to ensure that the GUI is not cluttered with information that is not
intended at the current visibility level. The following criteria have been used
for the assignment of the recommended visibility level:

- `Beginner` (default):

  Parameters that should be visible in all levels via the GUI.
  The number of parameters with `Beginner` level should be limited to all basic parameters
  so the GUI display is well-organized and easy to use.

- `Expert`:

  Parameters that require a more in-depth knowledge of the system functionality.
  This is the preferred visibility level for all advanced parameters.

- `Guru`:

  Advanced parameters that usually only people with a sound background in phase shifting can make good use of.

- `Experimental`:

  New features that have not been tested yet.
  The system might crash at some point.

Upon every parameter change, all parameters of the `Fringes` instance are saved
to the file `.fringes.yaml` in the user home directory.
When the GUI starts again, the previous parameters are loaded.
To avoid this, just delete the config file
or press the ``reset`` button in the `Methods`_ widget to restore the default parameters.

Methods
"""""""
In the bottem left corner you will find action buttons for the associated methods of the `Fringes` class.
Alternatively, you can use the keyboard shortcuts which are displayed when you hover over them.
The buttons are only active if the necessary data is available, i.e. was enoded, decoded or loaded.

Viewer
""""""
In the center resides the viewer.
If float data is to be displayed, `nan` is replaced by zeros.

Data
""""
In the top right corner the data widget is located.
It lists the data which was encoded, decoded or loaded.

In order to keep the parameters in the :ref:`Parameter Tree <attributes>` consistent with the data,
once a parameter has changed, certain data will be removed
and also certain :ref:`buttons <methods>` will be deactivated.
Also, the data has to fit in order to be able to execute certain functions.
As a consequence, if you load data - e.g. the acquired (distorted) fringe pattern sequence -
the first element of its videoshape has to match the parameter `Frames` in order to be able to decode it.

To display any datum listed in the table in the `Viewer`_, simply select the name of it in the table.

Klick the ``Load`` button to choose data or a config file to load.
With the ``Save`` button, all data including the parameters are saved to the selected directory.
Use the ``Clear all`` button to delete all data.

Please note: By default, the datum `fringes` is decoded.
If you want to decode a datum with a different name (e.g. one that you just loaded),
select its name in the table and klick ``Set data (to be decoded)``.

Log
"""
The logging of the `Fringes` class is displayed here.
The logging level can be set in the :ref:`Parameter Tree <attributes>`.
