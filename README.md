# nectarchain [![Build Status](https://github.com/cta-observatory/nectarchain/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cta-observatory/nectarchain/actions/workflows/ci.yml?query=workflow%3ACI+branch%3Amain) [![pypi](https://badge.fury.io/py/nectarchain.svg)](https://pypi.org/project/nectarchain) [![conda](https://anaconda.org/conda-forge/nectarchain/badges/version.svg)](https://anaconda.org/conda-forge/nectarchain)

Repository for the high level analysis of the NectarCAM data.
The analysis is heavily based on [ctapipe](https://github.com/cta-observatory/ctapipe), adding custom code for NectarCAM calibration.

## Installation

`nectarchain` is available as a [PyPI](https://pypi.org/project/nectarchain/) or [`conda`](https://anaconda.org/conda-forge/nectarchain) package, or as a [Singularity](https://apptainer.org/news/community-announcement-20211130/)/[Apptainer](https://apptainer.org/) container.

### Using conda/mamba

`conda` is a package manager, distributed e.g. within [Anaconda](https://www.anaconda.com/products/distribution). Use of its re-implementation in C++, `mamba`, is strongly advised instead. `mamba` is shipped e.g. within [Mambaforge](https://mamba.readthedocs.io/en/latest/installation.html) which can advantageously replace Anaconda altogether (lighter and faster).

```shell
mamba create -n nectarchain -c conda-forge nectarchain
```

### Using pip

`nectarchain` can also be manually installed as a PyPI package, albeit following specific requirements which are automatically accounted for through a `conda`/`mamba` installation.

```shell
mamba create -n nectarchain python=3.11
mamba activate nectarchain
pip install nectarchain
```

### As a container

`nectarchain` is planned to be pushed on each release on the [GitHub Container Registry](ghcr.io) as an [Apptainer](https://apptainer.org/) image. Such a container can be instantiated with:

```shell
apptainer shell oras://ghcr.io/cta-observatory/nectarchain:latest
```

The `nectarchain` code is then available under `/opt/cta/nectarchain`.

[DIRAC support](#optional-dirac-support) is fully available and configured within such a container.

#### Note to macOS users

macOS users may experience errors when trying to initialize a proxy to DIRAC when the [DIRAC support is enabled](#optional-dirac-support), especially with recent hardware equipped with M1 or M2 Apple CPU chips. The container alternative can then help having an environment with CTADIRAC fully configured. However, [Apptainer](https://apptainer.org/) is [not readily available on macOS](https://apptainer.org/docs/admin/main/installation.html#mac), but there is a workaround using [`lima` virtualization technology](https://lima-vm.io/) on a Mac.

**TL;DR**

```shell
brew install qemu lima
limactl start template://apptainer
limactl shell apptainer apptainer run --bind $HOME:/home/$USER.linux oras://ghcr.io/cta-observatory/nectarchain:latest
```

If you are running a Mac which CPU is based on ARM architecture (M1 or M2 Apple chips), when starting the `apptainer` container (second line above), please select the `Open an editor to review or modify the current configuration` option and add the following line at the beginning of the configuration file:
```shell
arch: "x86_64"
```
otherwise, please proceed with the `Proceed with the current configuration` option.

The mount point `/tmp/lima` is shared between the host machine and the `apptainer` container, and writable from both.

### Manual installation (for developers)

This is the recommended installation procedure for developers. `nectarchain` should be `pip`-installed in development (_aka_ editable) mode.

```shell
git clone https://github.com/cta-observatory/nectarchain.git
cd nectarchain
mamba env create --name nectarchain --file environment.yml
mamba activate nectarchain
pip install -e .
```

Enable [pre-commit hooks](https://pre-commit.com/), which enforces adherence to PEP8 coding style:

```shell
pre-commit install
```

Please follow the [same conventions as `ctapipe`](https://ctapipe.readthedocs.io/en/latest/getting_started/) regarding settings of Git remotes, and how to contribute to the code with pull requests.

### Optional DIRAC support

_Note_: this is **not** needed if you are using `nectarchain` [as a container](#as-a-container), as DIRAC is already fully installed and configured within.

To enable support for DIRAC within the same environment, do the following after the installation of `nectarchain` described above:

```shell
mamba activate nectarchain 
mamba install dirac-grid
conda env config vars set X509_CERT_DIR=${CONDA_PREFIX}/etc/grid-security/certificates X509_VOMS_DIR=${CONDA_PREFIX}/etc/grid-security/vomsdir X509_VOMSES=${CONDA_PREFIX}/etc/grid-security/vomses
# The following is needed for the environment variables, used for DIRAC configuration, to be available:
mamba deactivate
mamba activate nectarchain
pip install CTADIRAC
dirac-configure
```

Some Mac OS users (running on M1 chip) may experience a `M2Crypto.SSL.SSLError` error when trying to initiate a DIRAC proxy with `dirac-proxy-init`. Instead of:
```shell
mamba install dirac-grid
```
one may try:
```shell
mamba install dirac-grid "voms=2.1.0rc2=h7a71a8a_7"
```
or the [container alternative](#note-to-macos-users) as explained above.

`nectarchain` is currently pinned to `ctapipe` version 0.19.

## Contributing

All contribution are welcome.

Guidelines are the same as [ctapipe's ones](https://ctapipe.readthedocs.io/en/latest/developer-guide/getting-started.html).
See [here](https://ctapipe.readthedocs.io/en/latest/developer-guide/pullrequests.html#pullrequests) how to make a pull request to contribute.


## Report issue / Ask a question

Please use [GitHub Issues](https://github.com/cta-observatory/nectarchain/issues).
