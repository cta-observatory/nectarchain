# Quick recipe for the Data Quality Monitoring (DQM) scripts

To launch the DQM, first activate the `nectarchain` `conda` environment:

```shell
source activate nectarchain
```

Usage:

```shell
$ python start_calib.py -h

usage: start_calib.py [-h] [-p] [-n] [-r RUNNB] [-i INPUT_FILES [INPUT_FILES ...]] input_paths output_paths

NectarCAM Data Quality Monitoring tool

positional arguments:
  input_paths           Input paths
  output_paths          Output paths

optional arguments:
  -h, --help            show this help message and exit
  -p, --plot            Enables plots to be generated
  -n, --noped           Enables pedestal subtraction in charge integration
  -r RUNNB, --runnb RUNNB
                        Optional run number, automatically found on DIRAC
  -i INPUT_FILES [INPUT_FILES ...], --input-files INPUT_FILES [INPUT_FILES ...]
                        Local input files
```

To automatically find and retrieve run files from DIRAC, use the `-r` option: 

```shell
python start_calib.py -r 2720 $NECTARCAMDATA $NECTARDIR
```

To manually use local run files, use the `-i` option **after** indicating the positional arguments for input and output directories:
```shell
python start_calib.py $NECTARDATA $NECTARDIR -i NectarCAM.Run2720.0000.fits.fz NectarCAM.Run2720.0001.fits.fz
```
