Troubleshooting
===============

- Decoding takes a long time

  This is most likely due to the just-in-time compiler `Numba <https://numba.pydata.org/>`_,
  which is used for this computationally expensive function:
  During the first execution, an initial compilation is executed. 
  This can take several tens of seconds up to single digit minutes, depending on your CPU.
  However, for any subsequent execution, the compiled code is cached and the code of the function runs much faster, 
  approaching the speeds of code written in C.


- My decoded coordinates show lots of noise

  - Make sure the exposure of your camera is adjusted so that the fringe patterns show up with maximum contrast.
    Avoid under- and overexposure during acquisition.
  - Try using more, sets ``K`` and/or shifts ``N``.
  - If the decoded modulation is much lower than the decoded brightness,
    try to use larger wavelengths ``l`` resp. smaller number of periods ``v``.
  - If the decoded modulation remains low even with very large wavelengths (less than five periods per screen length),
    and you are conducting a deflectometric mesurement, the surface under test is probably too rough.
    Since deflectometry is for specular and glossy surfaces only, it isn't suited for scattering ones.
    You should consider a different measurement technique, e.g. fringe projection.


- My decoded coordinates show systematic offsets

  #. Ensure that the unamboguous measurement range is larger than the pattern length, i.e. ``UMR`` :math:`\ge` ``L``.
     If not, adjust the used wavelengths ``l`` resp. number of periods ``v`` accordingly.
  #. Ensure that the correct frames were captured while acquiring the fringe pattern sequence.
     If the timings are not set correctly, the sequence may be a frame off.
  #. This might occur if either the camera or the display used have a gamma value very different from 1.

    a) Typically, screens have a gamma value of 2.2; therefore compensate by setting the inverse value
       :math:`\gamma^{-1} = 1 / 2.2 \approx 0.45` to the ``gamma`` attribute of the ``Fringes`` instance.
       Alternatively, change the gamma value of the light source or camera directly.
    b) You can use the static method ``gamma_auto_correct()`` to
       automatically estimate and apply the gamma correction factor to linearize the display/camera response curve.
    c) You might also use more shifts ``N`` to compensate for the dominant harmonics of the gamma-nonlinearities.