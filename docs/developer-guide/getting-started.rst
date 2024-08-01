.. _getting_started_dev:

Getting Started For Developers
==============================

We strongly recommend using the `mambaforge conda distribution <https://github.com/conda-forge/miniforge#mambaforge>`_.

.. warning::

   The following guide is used only if you want to *develop* the
   ``nectarchain`` package, if you just want to write code that uses it
   as a dependency, you can install ``nectarchain`` from PyPI or conda-forge.
   See :ref:`getting_started_users`

Installation
------------

This is the recommended installation procedure for developers. ``nectarchain`` should be ``pip``-installed in development (*aka* editable) mode.

.. code-block:: console

   $ git clone https://github.com/cta-observatory/nectarchain.git
   $ cd nectarchain
   $ mamba env create --name nectarchain --file environment.yml
   $ mamba activate nectarchain
   $ pip install -e .

Enable `pre-commit hooks <https://pre-commit.com/>`_, which enforces adherence to PEP8 coding style:

.. code-block:: console

   $ pre-commit install

Please follow the `same conventions as ctapipe <https://ctapipe.readthedocs.io/en/latest/developer-guide/index.html>`_ regarding settings of git remotes, and how to contribute to the code with pull requests.


.. _optional-dirac-support:

Optional DIRAC support
----------------------

*Note*: this is **not** needed if you are using ``nectarchain`` as a container :ref:`as-a-container`, as DIRAC is already fully installed and configured within.

To enable support for DIRAC within the same environment, do the following after the installation of ``nectarchain`` described above:

.. code-block:: console

   $ mamba activate nectarchain
   $ mamba install dirac-grid
   $ conda env config vars set X509_CERT_DIR=${CONDA_PREFIX}/etc/grid-security/certificates X509_VOMS_DIR=${CONDA_PREFIX}/etc/grid-security/vomsdir X509_VOMSES=${CONDA_PREFIX}/etc/grid-security/vomses
   $ # The following is needed for the environment variables, used for DIRAC configuration, to be available:
   $ mamba deactivate
   $ mamba activate nectarchain
   $ pip install CTADIRAC
   $ dirac-configure


Some Mac OS users (running on M1 chip) may experience a ``M2Crypto.SSL.SSLError`` error when trying to initiate a DIRAC proxy with ``dirac-proxy-init``. Instead of:

.. code-block:: console

   $ mamba install dirac-grid

one may try:

.. code-block:: console

  $ mamba install dirac-grid "voms=2.1.0rc2=h7a71a8a_7"

or the container alternative as explained in  :ref:`troubleshooting`.


Building the documentation
--------------------------

To locally build the documentation, optional dependencies should be installed with:

.. code-block::

    $ pip install -e ".[docs]"

The documentation can then be compiled with:

.. code-block::

    $ make -C docs html

Interactive Development Environment
-----------------------------------

It is recommended that a fully python-aware *interactive development
environment* (IDE) is used to develop code, rather than a basic text
editor. IDEs will automatically mark lines that have style
problems. The recommended IDEs are:

* `PyCharm CE <https://www.jetbrains.com/pycharm>`_ (Jetbrains)
* `VS Code <https://code.visualstudio.com/>`_ (Microsoft)

The IDEs provide a lot of support for avoiding common style and coding
mistakes, and automatic re-formatting.
