.. dqm:

Quick recipe for the Data Quality Monitoring (DQM) scripts
==========================================================

To launch the DQM, first activate the ``nectarchain`` ``conda`` environment::

    source activate nectarchain

Usage::

    python start_dqm.py -h
    usage: start_dqm.py [-h] [-p] [--write-db] [-n] [-r RUNNB] [--r0] [--max-events MAX_EVENTS] [-i INPUT_FILES [INPUT_FILES ...]] input_paths output_paths

    NectarCAM Data Quality Monitoring tool

    positional arguments:
      input_paths           Input paths
      output_paths          Output paths

    options:
      -h, --help            show this help message and exit
      -p, --plot            Enables plots to be generated
      --write-db            Write DQM output in DQM ZODB data base
      -n, --noped           Enables pedestal subtraction in charge integration
      -r RUNNB, --runnb RUNNB
                            Optional run number, automatically found on DIRAC
      --r0                  Disable all R0->R1 corrections
      --max-events MAX_EVENTS
                            Maximum number of events to loop through in each run slice
      -i INPUT_FILES [INPUT_FILES ...], --input-files INPUT_FILES [INPUT_FILES ...]
                            Local input files

To automatically find and retrieve run files from DIRAC, use the ``-r`` option::

    python start_dqm.py -r 2720 $NECTARCAMDATA $NECTARCAMDATA

To manually use local run files, use the ``-i`` option **after** indicating the positional arguments for input and output directories::

    python start_dqm.py $NECTARCAMDATA $NECTARCAMDATA -i NectarCAM.Run2720.0000.fits.fz NectarCAM.Run2720.0001.fits.fz
