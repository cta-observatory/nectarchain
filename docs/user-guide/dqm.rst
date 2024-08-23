.. _dqm:

Quick recipe for the Data Quality Monitoring script
===================================================

Run locally
-----------

To launch the Data Quality Monitoring (DQM), first activate the ``nectarchain`` ``conda`` environment::

    source activate nectarchain

Usage::

    $ python start_dqm.py -h

To automatically find and retrieve run files from DIRAC, use the ``-r`` option::

    $ python start_dqm.py -r 2720 $NECTARCAMDATA $NECTARCAMDATA

See :ref:`env-vars` for the usage of the ``$NECTARCAMDATA`` environment variable.

To manually use local run files, use the ``-i`` option **after** indicating the positional arguments for input and output directories::

    $ python start_dqm.py $NECTARCAMDATA $NECTARCAMDATA -i NectarCAM.Run2720.0000.fits.fz NectarCAM.Run2720.0001.fits.fz

As a DIRAC job
--------------

The user script `nectarchain/user_scripts/jlenain/dqm_job_submitter/submit_dqm_processor.py` can be used to run the DQM as a DIRAC job::

    $ python submit_dqm_processor.py -h

Under the hood, it calls the ``dqm_processor.sh`` wrapper script, which itself launches an Apptainer instance of the ``nectarchain`` container on the DIRAC worker. This Apptainer image is automatically built and published in CI on releases.

The DQM runs one job per NectarCAM run. It is possible, for instance, to bulk-submit DIRAC jobs for all runs acquired during a given period, e.g.::

    $ d=2023-01-01
    $ while [ "$d" != 2023-03-01 ]; do python submit_dqm_processor.py -d $d; d=$(date -I -d "$d + 1 day"); done

The script will first assess whether DQM jobs have already been run for a given NectarCAM run, based on the output directory on DIRAC where this script stores its output. Look for the ``$DIRAC_OUTDIR`` environment variable in ``dqm_processor.sh``.