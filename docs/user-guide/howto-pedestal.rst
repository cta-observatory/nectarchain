.. _pedestal:

Pedestal estimation tool
-------------------------------
The pedestal estimation tool processes pedestal calibration data (dedicated pedestal runs or runs containing interleaved pedestal events) and returns estimates of the pedestal as a function of gain channel/pixel/sample within the readout window.

Run the tool locally
=========================
To use the pedestal estimation tool, first activate the ``nectarchain`` environment:

.. code-block:: console

    $ mamba activate nectarchain


The user script `nectarchain/user_scripts/ltibaldo/example_pedestal.py` showcases the usage of the tool.

The input data are indentified by run number. See :ref:`env-vars` to set up the ``$NECTARCAMDATA`` environment variable. The pedestal estimation tool inherits the configurable parameters of the `~nectarchain.makers.component.PedestalComponent.PedestalEstimationComponent`.

To run the example script:

.. code-block:: console

    $ python -i example_pedestal.py

Inspect the results
=========================
The results are stored in a `~nectarchain.data.container.pedestalContainer.NectarCAMPedestalContainer`. They are accessible on the fly if the tool is run interactively (as in the example above) and stored in a `.h5` file.

The user script `nectarchain/user_scripts/ltibaldo/show_pedestal_output.py` provides an example of how to access the results from disk and produce some plots:

.. code-block:: console

    $ python -i plot_pedestal_output.py


