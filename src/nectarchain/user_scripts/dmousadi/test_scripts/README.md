
# NectarCAM Test Scripts

Automated tests from the Test Readiness Review document for the CTA NectarCAM based on [nectarchain](https://github.com/cta-observatory/nectarchain) and [ctapipe](https://github.com/cta-observatory/ctapipe). This project can test for the following requirements:
- B-TEL-1010 Intensity resolution & B-TEL-1390 Linearity (script [linearity_test.py](tests/linearity_test.py))
- B-TEL-1030 Time resolution (script [pix_couple_tim_uncertainty_test.py](tests/pix_couple_tim_uncertainty_test.py))
- B-TEL-1260 Deadtime, B-TEL-1270 Deadtime Measurement & B-MST-1280 Event Rate (script [deadtime_test.py](tests/deadtime_test.py))
- B-TEL-1370 Pedestal subtraction (script [pedestal_test.py](tests/pedestal_test.py))
- B-TEL-1380 Systematic pixel timing uncertainty (script [pix_tim_uncertainty_test.py](tests/pix_tim_uncertainty_test.py))


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)


## Installation

Instructions on how to install and set up the project:
```
git clone https://drf-gitlab.cea.fr/dmousadi/camera-test-scripts
cd camera-test-scripts
pip install -r requirements.txt
```
If you want to automatically download your data, one of the requirements is also DIRAC, for which you need to have a grid certificate. It is not necessary for this repo, if you have your NectarCAM runs (fits files) locally stored. You can find more information about DIRAC [here](https://gitlab.cta-observatory.org/cta-computing/dpps/workload/CTADIRAC). If you are installing these packages for the first time and getting 'error building wheel', you might need to (re)install some of these: swig, ca-certificates, openssl, boost, protobuff, cmake. 

Once you have set up your environment, if you're not already a nectarchain user you need to set the NECTARCAMDATA environment variable to the directory where you have the NectarCAM runs:
```
conda activate your_env_name
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
nano $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```
In activate.d/env_vars.sh add:
```
export NECTARCAMDATA=your_path_to_nectarcamdata
```
And then in deactivate.d/env_vars.sh:
```
unset NECTARCAMDATA 
```


According to nectarchain, the directory should have this structure:
```
NECTARCAMDATA/runs/{fits_files}
```
If the structure is different in your server (testbench server, looking at you) you could do a symbolic link of the form:
```
ln -s /data/ZFITS/*/*/* nectarcamdatafolder/runs
```


## Usage

The test scripts can be run autonomously, or through a GUI. To run the gui you can just start it with python:
```
python gui.py
```
In the gui you can pick the test and set the parameters needed or keep the default ones. The most important parameter in every test is <b>runlist</b>, where you state which runs you want to process. If you have this runs in your NECTARCAMDATA, they will be found there and processed, otherwise if you don't have them but you're using DIRAC, they will be downloaded there. All tests have an <b>output</b> parameter, where you state where you want to save the plots produced by the tests. The <b>evts</b> parameter indicates the amount of events that are going to be processed (more events = more accurate results but more runtime). The linearity test also requires the corresponding transmission for each of the runs in the list, while in the timing uncertainty between couples of pixels you have to add the path to the csv file containing the PMT transit time values (calculated by Federica, might be stored in a database later). 
To have a better look at the parameters, you can also run the tests on their own with the help option:
```
python tests/linearity_test.py --help
```
When everything is ready, you can press on run test.


## Features

List of key features of the project:

- GUI to run tests automatically
- If runs aren't already present, they will be downloaded with DIRAC
- Tests process runs and create a plot showing the results. Intermediate results for each separate run are saved in HDF5 tables in a folder in NECTARCAMDATA. The plot is saved in a specified folder, as well as in a temporary file which is read by the GUI
- Plot display with zoom and navigation
- Output logs and parameters displayed dynamically

The test scripts essentially loop through the runs using the ctapipe Tool and Component classes, and then process the runs all together to calculate the results that go into the final plot.

## Contributing

You can contribute by adding more tests. The full list of tests is [here](https://docs.google.com/spreadsheets/d/1t5Z9ZHRESB6BYzmMbyFCDDALgTqvSb1KpJ8JMEiZgnI/edit?usp=sharing). It is planned to discuss about the implementation of more of the tests.

Guidelines for contributing to the project:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Create a new Pull Request

## Notes

This project uses a lot of methods developped by Federica Bradascio, and has also taken inspiration from Guillaume Grolleron's tutorials in nectarchain about using the ctapipe Tool class. 

For the in-code documentation I have used Cody (Copilot's free-ish cousin). I have checked everything but I still could have missed some mistake in explaining the code's functions.
