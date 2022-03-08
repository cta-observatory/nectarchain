# nectarchain

Repository for the high level analysis of the NectarCAM data.
The analysis is heavily based on [ctapipe](https://github.com/cta-observatory/ctapipe), adding custom code for mono reconstruction.

master branch status: [![Build Status](https://travis-ci.org/cta-observatory/nectarchain.svg?branch=master)](https://travis-ci.org/cta-observatory/nectarchain)


Current `nectarchain` build uses `ctapipe` master version.

## Install

Here is how you should install:
```
git clone https://github.com/cta-observatory/nectarchain.git
cd nectarchain
conda env create --name nectarchain --file environment.yml
conda activate nectarchain
pip install git+https://github.com/cta-observatory/ctapipe.git#egg=ctapipe
pip install git+https://github.com/cta-observatory/ctapipe_io_nectarcam.git#egg=ctapipe_io_nectarcam
pip install -e .
```
If you are a developper, better you install ctapipe as described in https://cta-observatory.github.io/ctapipe/getting_started/index.html
and periodically perform a "git pull upstream master" in order to be updated with the master

## Contributing

All contribution are welcomed.

Guidelines are the same as [ctapipe's ones](https://cta-observatory.github.io/ctapipe/development/index.html)
See [here](https://cta-observatory.github.io/ctapipe/development/pullrequests.html) how to make a pull request to contribute.


## Report issue / Ask a question

Use GitHub Issues.
