.. _getting_started_users:


*************************
Getting Started for Users
*************************

.. warning::

   The following guide is for *users*. If you want to contribute to
   nectarchain as a developer, see :ref:`getting_started_dev`.


Installation
============

``nectarchain`` is available as a `PyPI <https://pypi.org/project/nectarchain/>`_ or `conda <https://anaconda.org/conda-forge/nectarchain>`_ package, or as a `Singularity <https://apptainer.org/news/community-announcement-20211130/>`_/`Apptainer <https://apptainer.org/>`_ container as well.

Using ``conda``/``mamba``
-------------------------

We recommend using the ``mamba`` package manager, which is a C++ reimplementation of ``conda``.
It can be found `here <https://github.com/mamba-org/mamba>`_.
``conda`` is a package manager, distributed e.g. within `Anaconda <https://www.anaconda.com/products/distribution>`_.

To install ``nectarchain`` into an existing conda environment, use:

.. code-block:: console

   $ mamba install -c conda-forge nectarchain

You can also directly create a new environment like this (add more packages as you like):

.. code-block:: console

   $ mamba create -n nectarchain -c conda-forge nectarchain

To install a specific version of ``nectarchain``, you can use the following command:

.. code-block:: console

   $ mamba install -c conda-forge nectarchain=0.1.8

Using ``pip``
-------------

``nectarchain`` can also be manually installed as a PyPI package, albeit following specific requirements which are automatically accounted for through a ``conda``/``mamba`` installation.

.. code-block:: console

   $ mamba create -n nectarchain python=3.11
   $ mamba activate nectarchain
   $ pip install nectarchain

To install a specific version of ``nectarchain``, you can use the following command:

.. code-block:: console

   $ pip install nectarchain==0.1.8

.. _as-a-container:

As an Apptainer container
-------------------------

``nectarchain`` is pushed on each release on the `GitHub Container Registry <ghcr.io>`_ as an `Apptainer <https://apptainer.org/>`_ image. Such a container can be instantiated with:

.. code-block:: console

   $ apptainer shell oras://ghcr.io/cta-observatory/nectarchain:latest

The ``nectarchain`` code is then available under ``/opt/cta/nectarchain`` within the container.

:ref:`DIRAC support<optional-dirac-support>`_ is fully available and configured within such a container.

To use a specific version of ``nectarchain``, you can use the following command:

.. code-block:: console

   $ apptainer shell oras://ghcr.io/cta-observatory/nectarchain:0.1.8

Note to macOS users
^^^^^^^^^^^^^^^^^^^

macOS users may experience errors when trying to initialize a proxy to DIRAC when the :ref:`DIRAC support is enabled<optional-dirac-support>`_, especially with recent hardware equipped with M1 or M2 Apple CPU chips. The container alternative can then help having an environment with CTADIRAC fully configured. However, `Apptainer <https://apptainer.org/>`_ is `not readily available on macOS <https://apptainer.org/docs/admin/main/installation.html#mac>`_, but there is a workaround using `lima virtualization technology <https://lima-vm.io/>`_ on a Mac.

**TL;DR**

.. code-block:: console

   $ brew install qemu lima
   $ limactl start template://apptainer
   $ limactl shell apptainer apptainer run --bind $HOME:/home/$USER.linux oras://ghcr.io/cta-observatory/nectarchain:latest


If you are running a Mac which CPU is based on ARM architecture (M1 or M2 Apple chips), when starting the ``apptainer`` container (second line above), please select the ``Open an editor to review or modify the current configuration`` option and add the following line at the beginning of the configuration file:

.. code-block:: console

   arch: "x86_64"

otherwise, if your Mac is on an Intel CPU chip, please proceed with the ``Proceed with the current configuration`` option.

The mount point ``/tmp/lima`` is shared between the host machine and the ``apptainer`` container, and writable from both.
