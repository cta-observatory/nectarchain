.. _getting_started_dev:

Getting Started For Developers
==============================

Building the documentation
--------------------------

To locally build the documentation, optional dependencies should be installed with:

.. code-block::

    $ pip install -e ".[docs]"

The documentation can then be compiled with:

.. code-block::

    $ make -C docs html