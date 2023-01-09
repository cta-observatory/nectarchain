# nectarchain

Repository for the high level analysis of the NectarCAM data.
The analysis is heavily based on [ctapipe](https://github.com/cta-observatory/ctapipe), adding custom code for NectarCAM calibration.

master branch status: [![Build Status](https://travis-ci.org/cta-observatory/nectarchain.svg?branch=master)](https://travis-ci.org/cta-observatory/nectarchain)

## Installation

```
git clone https://github.com/cta-observatory/nectarchain.git
cd nectarchain
conda env create --name nectarchain --file environment.yml
conda activate nectarchain
pip install -e .
```
If you are a developer, better you follow the same conventions as `ctapipe`, as described in https://cta-observatory.github.io/ctapipe/getting_started/index.html#developing-a-new-feature-or-code-change.
 `nectarchain` is currently pinned to `ctapipe` version 0.12.

## Contributing

All contribution are welcome.

Guidelines are the same as [ctapipe's ones](https://cta-observatory.github.io/ctapipe/development/index.html)
See [here](https://cta-observatory.github.io/ctapipe/development/pullrequests.html) how to make a pull request to contribute.


## Report issue / Ask a question

Please use GitHub Issues.
