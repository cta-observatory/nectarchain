# nectarchain [![Build Status](https://github.com/cta-observatory/nectarchain/workflows/CI/badge.svg?branch=master)](https://github.com/cta-observatory/nectarchain/actions?query=workflow%3ACI+branch%3Amaster)

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
mamba create -n nectarchain python=3.8
mamba activate nectarchain
pip install nectarchain
```

### As a container

`nectarchain` is planned to be pushed on each release on the [GitHub Container Registry](ghcr.io) as an [Apptainer](https://apptainer.org/) image. Such a container can be instantiated with:

```shell
apptainer shell oras://ghcr.io/cta-observatory/nectarchain:latest
```

The `nectarchain` code is then available under `$CONDA_PREFIX`.

#### Note to Mac OS users

Mac OS users may experience errors when trying to initialize a proxy to DIRAC when the [DIRAC support is enabled](#optional-dirac-support), especially with recent hardware equipped with M1 or M2 Apple CPU chips. The container alternative can then help having an environment with CTADIRAC fully configured. However, [Apptainer](https://apptainer.org/) is [not readily available on Mac OS](https://apptainer.org/docs/admin/main/installation.html#mac), but there is a workaround using [`lima` virtualization technology](https://lima-vm.io/) on a Mac.

**TL;DR**

```shell
brew install qemu lima
limactl start template://apptainer
limactl shell apptainer apptainer run --bind $HOME:/home/$USER.linux oras://ghcr.io/cta-observatory/nectarchain:latest
```

When starting the `apptainer` container (second line above), please select the `Proceed with the current configuration` option.

### Manual installation (for developers)

This is the recommended installation procedure for developers. `nectarchain` should be `pip`-installed in development (_aka_ editable) mode.

```shell
git clone https://github.com/cta-observatory/nectarchain.git
cd nectarchain
mamba env create --name nectarchain --file environment.yml
mamba activate nectarchain
pip install -e .
```

Please follow the [same conventions as `ctapipe`](https://cta-observatory.github.io/ctapipe/getting_started/index.html#developing-a-new-feature-or-code-change) regarding settings of Git remotes for pull requests.

### Optional DIRAC support

To enable support for DIRAC within the same environment, do the following after the installation of `nectarchain` described above:

```shell
mamba activate nectarchain 
mamba install -c conda-forge dirac-grid
conda env config vars set X509_CERT_DIR=${CONDA_PREFIX}/etc/grid-security/certificates X509_VOMS_DIR=${CONDA_PREFIX}/etc/grid-security/vomsdir X509_VOMSES=${CONDA_PREFIX}/etc/grid-security/vomses
# The following is needed for the environment variables, used for DIRAC configuration, to be available:
mamba deactivate
mamba activate nectarchain
pip install CTADIRAC COMDIRAC
dirac-configure
```

`nectarchain` is currently pinned to `ctapipe` version 0.12.

## Contributing

All contribution are welcome.

Guidelines are the same as [ctapipe's ones](https://cta-observatory.github.io/ctapipe/development/index.html)
See [here](https://cta-observatory.github.io/ctapipe/development/pullrequests.html) how to make a pull request to contribute.


## Report issue / Ask a question

Please use [GitHub Issues](https://github.com/cta-observatory/nectarchain/issues).
