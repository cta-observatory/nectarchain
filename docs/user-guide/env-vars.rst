.. env-vars:

Environment variables
=====================

We strive for relying as little as possible on environment variables across ``nectarchain``.
However, some environment variables are needed for ``nectarchain`` to work properly, especially to automatically fetch run files from the grid via DIRAC, or to store output results and plots.

Mandatory
---------

:``NECTARCAMDATA``: path to local NectarCAM data. It can contain ``fits.fz`` run files, `~nectarchain.data.container.waveformsContainer.WaveformsContainer` or `~nectarchain.data.container.chargesContainer.ChargesContainer` HDF5 files. This is also where the `~nectarchain.data.management.DataManagement.findrun` method will automatically store NectarCAM run files when fetched from DIRAC.

Optional
--------

:``NECTARCHAIN_LOG``: path for output logs for ``nectarchain``, generally defaulting to ``/tmp``.
:``NECTARCHAIN_FIGURES``: path to store output figures for ``nectarchain``, generally defaulting to ``/tmp``, or configurable (see e.g. `nectarchain.makers.component.spe.spe_algorithm.SPEalgorithm`).