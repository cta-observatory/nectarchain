================================================
nectarchain |pypi| |conda| |CI| |docs| |codecov|
================================================


Repository for the high level analysis of the NectarCAM data, a camera
to equip the medium-sized telescopes of `CTAO <https://www.ctao.org/>`__
in its Northern site. The analysis is heavily based on
`ctapipe <https://github.com/cta-observatory/ctapipe>`__, adding custom
code for NectarCAM calibration.

``nectarchain`` is currently pinned to ``ctapipe`` version 0.24.0.

This code is under rapid development. It is not recommended for
production use unless you are an expert or developer!

- Code: https://github.com/cta-observatory/nectarchain
- Docs: https://nectarchain.readthedocs.io/

Installation for Users
======================

``nectarchain`` and its dependencies may be installed using the
*Anaconda* or *Miniconda* package system. We recommend creating a conda
virtual environment first, to isolate the installed version and
dependencies from your main environment (this is optional).

The latest version of ``nectarchain`` can be installed via:

::

   mamba install -c conda-forge nectarchain

or via:

::

   pip install nectarchain

**Note**: to install a specific version of ``nectarchain``, take look at
the documentation
`here <https://nectarchain.readthedocs.io/en/latest/user-guide/index.html>`__.

**Note**: ``mamba`` is a C++ reimplementation of conda and can be found
`here <https://github.com/mamba-org/mamba>`__.

Note this is *pre-alpha* software and is not yet stable enough for
end-users (expect large API changes until the first stable 1.0 release).

Developers should follow the development install instructions found in
the
`documentation <https://nectarchain.readthedocs.io/en/latest/developer-guide/index.html>`__.

Contributing
============

All contribution are welcome.

Guidelines are the same as `ctapipe's
ones <https://ctapipe.readthedocs.io/en/latest/developer-guide/getting-started.html>`__.
See
`here <https://ctapipe.readthedocs.io/en/latest/developer-guide/pullrequests.html#pullrequests>`__
how to make a pull request to contribute.

Report issue / Ask a question
=============================

Please use `GitHub
Issues <https://github.com/cta-observatory/nectarchain/issues>`__ to
report issues or `GitHub
Discussions <https://github.com/cta-observatory/nectarchain/discussions>`__
for questions and discussions.

.. |pypi| image:: https://badge.fury.io/py/nectarchain.svg
   :target: https://pypi.org/project/nectarchain
.. |conda| image:: https://anaconda.org/conda-forge/nectarchain/badges/version.svg
   :target: https://anaconda.org/conda-forge/nectarchain
.. |CI| image:: https://github.com/cta-observatory/nectarchain/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/cta-observatory/nectarchain/actions/workflows/ci.yml?query=workflow%3ACI+branch%3Amain
.. |docs| image:: https://readthedocs.org/projects/nectarchain/badge/?version=latest
   :target: https://nectarchain.readthedocs.io/en/latest/?badge=latest
.. |codecov| image:: https://codecov.io/github/cta-observatory/nectarchain/graph/badge.svg?token=TDhZlJtbMv
   :target: https://codecov.io/github/cta-observatory/nectarchain
