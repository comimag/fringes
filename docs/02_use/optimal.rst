Optimal Coding Strategy
=======================

As makes sense intuitively, more sets ``K`` as well as more shifts ``N`` per set reduce the uncertainty ``u`` after decoding.
A minimum of 3 shifts is needed to solve for the 3 unknowns brightness ``A``, modulation ``B`` and coordinates ``x``.
Any additional 2 shifts compensate for one harmonic of the recorded fringe pattern.
Therefore, higher accuracy can be achieved using more shifts ``N``, but the time required to capture them
sets a practical upper limit to the feasible number of shifts.

Generally, shorter wavelengths ``l`` (or equivalently more periods ``v``) reduce the uncertainty ``u``,
but the resolution of the camera and the display must resolve the fringe pattern spatially.
Hence, the used hardware imposes a lower bound for the wavelength ``lmin``
(or upper bound for the number of periods ``vmax``).

Also, small wavelengths might result in a smaller unambiguous measurement range ``UMR``.
If two or more sets ``K`` are used and their wavelengths ``l`` resp. number of periods ``v`` are relative primes,
the unmbiguous measurement range can be increased many times.
As a consequence, one can use much smaller wavelenghts ``l`` (larger number of periods ``v``).

However, it must be assured that the unambiguous measurment range is always equal or larger than both,
the width ``X`` and the height ``Y``.
Else, :ref:`temporal phase unwrapping <tpu>` would yield wrong results and thus instead
:ref:`spatial phase unwrapping <spu>` is used.
Be aware that in the latter case only a relative phase map is obtained,
which lacks the information of where exactly the camera sight rays were looking at during acquisition.

To simplify finding and setting the optimal parameters, one can choose from the followng options:

- ``v`` can be set to ``'optimal'``.
  This automatically determines the optimal integer set of ``v``,
  based on the maximal resolvable spatial frequency ``vmax``.
- Equivalently, ``l`` can also be set to ``'optimal'``.
  This will automatically determine the optimal integer set of ``l``,
  based on the minimal resolvable wavelength ``lmin`` = ``L`` / ``vmax``.
- ``T`` can be set directly, based on the desired acquisition time.
  The optimal ``K``, ``N`` and  - if necessary - the :ref:`multiplexing <multiplex>` will be determined automatically.
- Instead of the options above, one can simply use the function ``optimize()``:
  If ``umax`` is specified, the optimal parameters are determined
  that allow a maximal uncertainty of ``umax`` with a minimum number of frames.
  Else, the parameters of the `Fringes` instance are optimized to yield the minimal uncertainty possible
  using the given number of frames ``T``.

However, these methods only perform optimally
if the recorded modulation ``B`` is known (or can be estimated)
for certain spatial frequencies ``v``.

a) Measure the **modulation transfer function (MTF)** at a given number of sample points:

   1. Set ``K`` to the required number of sample points (usually > 10 is a good value).
   2. Set ``v`` to ``'exponential'``.
      This will create spatial frequencies ``v`` spaced evenly on a log scale (a geometric progression),
      starting from 0 up to ``vmax``.
   3. Encode, acquire and decode the fringe pattern sequence.
   4. Mask the values of ``B`` with *nan* where the camera wasn't looking at the screen.
      The decoded modulation ``B`` can be used as an indicator.
   5. Call ``Bv(B)`` with the estimated modulation from the measurement as the argument.
   6. Finlly, to get the modulation ``B`` at certain spatial frequencies ``v``, simply call ``MTF(v)``.
      This method interpolates the modulation from the measurement at the points ``v``.
b) A linear MTF is assumed [1]_:
   It starts at ``v`` = 0 with B = 1 and ends at ``v`` = ``vmax`` with B = 0.
   Therefore, the optimal wavelength is ``vopt`` = ``vmax`` / 2.

..
   Estimate the **magnification** and the **Point Spread Function (PSF)** of the imaging system:

   1. Set the attributes ``magnification`` and ``PSF``.
   2. Finally, to get the modulation ``B`` at certain spatial frequencies ``v``, simply call ``MTF(v)``.
      Now, this method computes the modulation from the specified attributes ``magnifiction`` and ``PSF`` directly.

c) As a last resort, a constant modulation transfer function is assumed: MTF(``v``) = 1.

.. [1] `Bothe,
        "Grundlegende Untersuchungen zur Formerfassung mit einem neuartigen Prinzip der Streifenprojektion und Realisierung in einer kompakten 3D-Kamera",
        Dissertation,
        ISBN 978-3-933762-24-5,
        BIAS Bremen,
        2008.
        <https://www.amazon.de/Grundlegende-Untersuchungen-Formerfassung-Streifenprojektion-Strahltechnik/dp/3933762243/ref=sr_1_2?qid=1691575452&refinements=p_27%3AThorsten+B%C3%B6th&s=books&sr=1-2>`_
