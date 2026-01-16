.. _troubleshooting:

Troubleshooting
===============

.. _note_mac_users:

Note to macOS users
-------------------

macOS users may experience errors when trying to initialize a proxy to DIRAC when the
DIRAC support is enabled (see :ref:`optional-dirac-support`), especially with recent
hardware equipped with M1 or M2 Apple CPU chips. Two possible workarounds are proposed
below.

Using a container
^^^^^^^^^^^^^^^^^

The container alternative can then help having an environment with CTADIRAC fully configured.
However, `Apptainer <https://apptainer.org/>`_ is `not readily available on macOS <https://apptainer.org/docs/admin/main/installation.html#mac>`_,
but there is a workaround using `lima virtualization technology <https://lima-vm.io/>`_
on a Mac.

**TL;DR**

.. code-block:: console

   $ brew install qemu lima
   $ limactl start template://apptainer
   $ limactl shell apptainer apptainer run --bind $HOME:/home/$USER.linux oras://ghcr.io/cta-observatory/nectarchain:latest


If you are running a Mac which CPU is based on ARM architecture (M1 or M2 Apple chips),
when starting the ``apptainer`` container (second line above), please select the
``Open an editor to review or modify the current configuration`` option and add the
following line at the beginning of the configuration file:

.. code-block:: console

   arch: "x86_64"

otherwise, if your Mac is on an Intel CPU chip, please proceed with the
``Proceed with the current configuration`` option.

The mount point ``/tmp/lima`` is shared between the host machine and the ``apptainer``
container, and writable from both.
