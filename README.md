# nectarchain

Repository for the high level analysis of the NectarCAM data.
The analysis is heavily based on [ctapipe](https://github.com/cta-observatory/ctapipe), adding custom code for NectarCAM calibration.

master branch status: [![Build Status](https://travis-ci.org/cta-observatory/nectarchain.svg?branch=master)](https://travis-ci.org/cta-observatory/nectarchain)

## Installation

```shell
git clone https://github.com/cta-observatory/nectarchain.git
cd nectarchain
# or conda
mamba env create --name nectarchain --file environment.yml
mamba activate nectarchain
pip install .
```
If you are a developer, better you follow the same conventions as `ctapipe`, as described in https://cta-observatory.github.io/ctapipe/getting_started/index.html#developing-a-new-feature-or-code-change, and `pip`-install `nectarchain` in development (_aka_ editable) mode:

```shell
pip install -e .
```

To enable support for DIRAC within the same environment, do the following after the installation of `nectarchain` described above:
```shell
# or conda
mamba activate nectarchain 
mamba install -c conda-forge dirac-grid
conda env config vars set X509_CERT_DIR=${CONDA_PREFIX}/etc/grid-security/certificates X509_VOMS_DIR=${CONDA_PREFIX}/etc/grid-security/vomsdir X509_VOMSES=${CONDA_PREFIX}/etc/grid-security/vomses
# the following is needed for the environment variables, used for DIRAC configuration, to be available:
mamba deactivate
mamba activate nectarchain
pip install CTADIRAC
# optional:
pip install COMDIRAC
dirac-configure
```

`nectarchain` is currently pinned to `ctapipe` version 0.12.

## Contributing

All contribution are welcome.

Guidelines are the same as [ctapipe's ones](https://cta-observatory.github.io/ctapipe/development/index.html)
See [here](https://cta-observatory.github.io/ctapipe/development/pullrequests.html) how to make a pull request to contribute.


## Report issue / Ask a question

Please use GitHub Issues.
