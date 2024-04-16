.. default-role:: math

Parameters
==========
All parameters are implemented as class properties (managed attributes).
They are parsed when setting, so usually several input formats are accepted, e.g.
``bool``, ``int``, ``float``, ``str`` for scalars and additionally ``list``, ``tuple``, ``ndarray`` for arrays.

Note that some attributes have subdependencies (cf. :numref:`inter`), hence dependent attributes might change as well.
Circular dependencies are resolved automatically.

.. _inter:
.. figure:: interdependencies.svg
    :align: center
    :alt: interdependencies

    Parameters and their Interdependencies.

Video Shape
-----------
``shape`` is the standardized shape (T, Y, X, C) of the fringe pattern sequence, with

- ``T``: number of frames
- ``Y``: height (in pixel units)
- ``X``: width (in pixel units)
- ``C``: number of color channels

.. ``T`` = ``H`` `\cdot \sum` ``N``.
   If a `multiplexing`_ scheme is activated, ``T`` reduces further.

``L`` is the maximum of ``X`` and ``Y`` and denotes the length (in pixel units) to be encoded.
It can be extended by the factor ``alpha``.

``C`` depends on the :ref:`coloring <coloring and averaging>` and `multiplexing`_ schemes activated.

``size`` is the product of ``shape``.

Coordinate System
------------------
The following coordinate systems can be used by setting ``grid`` to:

- ``image``: The top left corner pixel of the grid is the origin (0, 0) and positive directions are right- resp. downwards.
- ``Cartesian``: The center of grid is the origin (0, 0) and positive directions are right- resp. upwards.
- ``polar``: The center of grid is the origin (0, 0) and positive directions are clockwise resp. outwards.
- ``log-polar``: The center of grid is the origin (0, 0) and positive directions are clockwise resp. outwards.

``indexing`` denotes the indexing convention.
Possible values are:

- ``xy``: Cartesian indexing (defaut) will index the row first.
- ``ij``: Matrix indexing will index the colum first.

``D`` denotes the number of directions to be encoded.

``axis`` is used to define along which axis of the coordinate system (index 0 or 1)
the fringe pattern is shifted if ``D`` = 1.

``angle`` can be used to tilt the coordinate system. The origin remains the same.

Set
---
Each set consists of the following attributes (cf. black box in :numref:`inter`):

- ``N``: number of shifts
- ``l``: wavelength (in pixel units)
- ``v``: spatial frequency, i.e. number of periods (per screen length ``L``)
- ``f``: temporal frequency, i.e. number of periods to shift over

Each is an array with shape (number of directions ``D``, number of sets ``K``).
For example, if ``N``.shape = (2, 3), it means that we encode ``D`` = 2 directions with ``K`` = 3 sets each.
Changing ``D`` or ``K`` directly, changes the shape of all set attributes.
When setting a set attribute with a new shape (``D'``, ``K'``),
``D`` and ``K`` are updated as well as the shape of the other set attributes.

If a set attribute is 1D, then it is stacked to match the number of directions ``D``.

If a set attribute is 0D i.e. a scalar, then all values are simply replaced by the new one.

``l`` and ``v`` are related by ``l`` = ``L`` / ``v``.
When ``L`` changes, ``v`` is kept constant and only ``l`` is changed.

Usually ``f`` = 1 and is essentially only changed if :ref:`frequency division multiplexing <multiplexing>` ``FDM`` is activated.

``reverse`` is a boolean which reverses the direction of the shifts (by multiplying ``f`` with -1).

``p0`` denotes the phase offset, which can be used to
e.g. let the fringe patterns start (at the origin) with a gray value of zero.

Intensity Values
----------------
``dtype`` denotes the data type of the fringe pattern sequence.
Possible values are:

- ``uint8`` (default)
- ``uint16``
- ``float32``
- ``float64``

``nbytes`` is the number of total bytes consumed by fringe pattern sequence.

.. ``q`` is the quantization step size and equals 1 for ``bool``, `2^r` for r-bit ``unsigned integers``,
   and for ``float`` its corresponding `resolution <https://numpy.org/doc/stable/reference/generated/numpy.finfo.html>`_.

``q`` is the quantization step size and equals `2^r` for r-bit ``unsigned integers``
and for ``float`` its corresponding `resolution <https://numpy.org/doc/stable/reference/generated/numpy.finfo.html>`_.

.. ``Imax`` is the maximum gray value and equals 1 for ``float`` and ``bool``,
   and `2^r - 1` for ``unsigned integers`` with r bits.

``Imax`` is the maximum gray value and equals 1 for ``float``
and `2^r - 1` for ``unsigned integers`` with r bits.

``A`` is the offset, also called brightness.
It is limited by ``Imax``.

``B`` is the amplitude of the cosinusoidal fringes.
It is limited by ``Imax``.

``V`` is the fringe :ref:`visibility <visibility and Exposure>` (also called fringe contrast).
``V`` = ``A`` / ``B`` and is within the range [0, 1].

``beta`` is the exposure (relative brightness) and is within the range [0, 1].

``gamma`` denotes the gamma correction factor and can be used to compensate nonlinearities of the display response curve.

Coloring and Averaging
----------------------
The fringe patterns can be colorized by setting the hue ``h``
to any sequence of RGB color triple within the interval [0, 255].
However, black (0, 0, 0) is not allowed.
``h`` must be in shape (``H``, 3):

``H`` is the number of hues and can be set directly; 3 is the length of the RGB color triple.

The hues ``h`` can also be set by assigning any combination of the following characters as a string:

- ``'r'``: red
- ``'g'``: green
- ``'b'``: blue
- ``'c'``: cyan
- ``'m'``: magenta
- ``'y'``: yellow
- ``'w'``: white

``C`` is the number of color channels required for either the set of hues ``h``
or :ref:`wavelength division multiplexing <multiplexing>`.
For example, if all hues are monochromatic, i.e. the RGB values are identical for each hue, ``C`` equals 1, else 3.

Repeating hues will be fused by averaging them before decoding.

``M`` is the number of averaged intensity samples and can be set directly.

Multiplexing
------------
The following multiplexing methods can be activated by setting them to ``True``:

- ``SDM``: Spatial Division Multiplexing

  This results in crossed fringe patterns.
  It can only be activated if we have two directions ``D`` = 2.
  The number of frames ``T`` is reduced by a factor of 2.

- ``WDM``: Wavelength Divison Multiplexing

  The shifts are multiplexed into the color channel, resulting in an RGB fringe pattern.
  All shifts ``N`` must equal 3.
  The number of frames ``T`` is reduced by a factor of 3.

- ``FDM``: Frequency Division Multiplexing

  Here, the directions ``D`` and the sets ``K`` are multiplexed.
  This results in crossed fringe patterns if ``D`` = 2.
  It can only be activated if ``D`` > 1 or ``K`` > 1.
  If you need a static pattern, i.e. one that remains congruent when shifted, set ``static`` to ``True``.

``SDM`` and ``WDM`` can be used together (reducing ``T`` by a factor of 2 * 3 = 6), ``FDM`` with neighter.

``TDM``: By default, the aforementioned multiplexing methods are deactivated,
so we then only have Time Divison Multiplexing.

For more details, please refer to :doc:`Multiplex </user_guide/mux>`.

Unwrapping
----------

``uwr`` denotes the phase unwrapping method and is eihter ``'none'``, ``'temporal'``, ``'spatial'`` or ``'FTM'``.
See :ref:`unwrapping <uwr>` for more details.

.. ``mode`` denotes the mode used for [temporal phase unwrapping](#temporal-phase-unwrapping--tpu-).
   Choose either ``'fast'`` (the default) or ``'precise'``.

``Vmin`` denotes the minimal fringe visibility for the measurement to be balid and is in the interval [0, 1].
During decoding, pixels with less are discarded, which can speed up the computation.

``umax`` denotes the maximal uncertainty required for the measurement to be valid and is in the interval [0, `L`].
It is used in the :ref:`Optimal Coding strategy <optimal coding strategy>`.

``verbose`` can be set to ``True`` to also receive from decoding
the wrapped phase maps `\varphi_i`, the fringe orders `k_i`, the residuals `r`, the uncertainty `u`,
the visibility `V` and the exposure `\beta`.

``FTM`` denotes :ref:`Fourier-transform method <Fourier Transform Method>` and is deployed if ``T`` = 1
and the `coordinate system`_ is eighter ``'image'`` or ``'Cartesian'``.

Quality Metrics
---------------

``UMR`` denotes the unambiguous measurement range.
The coding is only unique within the interval [0, ``UMR``); after that it repeats itself.

The ``UMR`` is derived from ``l`` and ``v``:

- If ``l`` `\in \mathbb{N}`, ``UMR`` = `\mathrm{lcm}(` ``l`` `)`,
  with `\mathrm{lcm}` being the least common multiple.
- Else, if ``v`` `\in \mathbb{N}`, ``UMR`` = ``L`` / `\mathrm{gcd}(` ``v`` `)`,
  with `\mathrm{gcd}` being the greatest common divisor.
- Else, if ``v`` `\lor` ``l`` `\in \mathbb{Q}` , `\mathrm{lcm}` resp. `\mathrm{gcd}` are extended to rational numbers.
- Else, if ``v`` `\land` ``l`` `\in \mathbb{R} \setminus \mathbb{Q}` , ``UMR`` = `\mathrm{prod}(` ``l`` `)`,
  with `\mathrm{prod}` being the product operator.

``eta`` denotes the coding efficiency ``L`` / ``UMR``.
It makes no sense to choose ``UMR`` much larger than ``L``,
because then a significant part of the coding range is not used.

``u`` denotes the minimum possible uncertainty of the measurement in pixels.
It is based on the phase noise model from [1]_
and propagated through the unwrapping process and the phase fusion.
It is influenced by the parameters

- ``M``: number of averaged intensity samples,
- ``N``: number of phase shifts,
- ``l``: wavelengths of the fringes,
- ``B``: measured amplitude

and the measurement hardware [2]_, [3]_

- ``quant``: quantization noise of the light source or camera,
- ``dark``: dark noise of the used camera,
- ``shot``: photon noise of light itself,
- ``gain``: system gain of the used camera.

``SNR`` = ``L`` / ``u`` is the signal-to-noise ratio of the phase shift coding
and is a masure of how many points can be distinguished within the screen length [0, ``L``).
It remains constant if ``L`` and hence ``l`` is scaled (the scaling factor cancels out).

``DR`` = ``UMR`` / ``u`` is the dynamic range of the phase shift coding
and is a measure of how many points can be distinguished within the unambiguous measurement range [0, ``UMR``).
Again, it remains constant if ``L`` and hence ``l`` is scaled (the scaling factor cancels out).

.. [1] `Surrel,
        "Additive noise effect in digital phase detection",
        Applied Optics,
        1997.
        <https://doi.org/10.1364/AO.36.000271>`_

.. [2] `EMVA,
        "Standard for Characterization of Image Sensors and Cameras Release 4.0 Linear",
        European Machine Vision Association,
        2021.
        <https://www.emva.org/standards-technology/emva-1288/emva-standard-1288-downloads-2/>`_

.. [3] `Bothe,
        "Grundlegende Untersuchungen zur Formerfassung mit einem neuartigen Prinzip der Streifenprojektion und Realisierung in einer kompakten 3D-Kamera",
        Dissertation,
        ISBN 978-3-933762-24-5,
        BIAS Bremen,
        2008.
        <https://www.amazon.de/Grundlegende-Untersuchungen-Formerfassung-Streifenprojektion-Strahltechnik/dp/3933762243/ref=sr_1_2?qid=1691575452&refinements=p_27%3AThorsten+B%C3%B6th&s=books&sr=1-2>`_
