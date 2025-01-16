.. _pedestal:

Pedestal estimation tool
-------------------------------
The pedestal estimation tool processes pedestal calibration data
(dedicated pedestal runs or runs containing interleaved pedestal
events) and returns estimates of the pedestal in ADC counts as a function of gain channel/pixel/sample within the readout window.

Run the tool locally
=========================
To use the pedestal estimation tool, first activate the ``nectarchain`` environment:

.. code-block:: console

    $ mamba activate nectarchain


The user script `nectarchain/user_scripts/ltibaldo/example_pedestal.py` showcases the usage of the tool.

The input data are identified by run number. See :ref:`env-vars` to
set up the ``$NECTARCAMDATA`` environment variable. Events in the
selected runs are
processed in slices with a fixed number of events set by the
``events_per_slice`` parameter (see `~nectarchain.makers.core.EventsLoopNectarCAMCalibrationTool`).

The pedestal
estimation tool inherits the configurable parameters of the
`~nectarchain.makers.component.pedestal_component.PedestalEstimationComponent`.
The data can be filtered based on time using the ``ucts_tmin`` and
``ucts_tmax`` parameters and to eliminate outlier waveforms using the ``filter_method`` parameter. Two different methods to exclude outlier
waveforms are implemented:

* ``WaveformsStdFilter`` discards waveforms with a standard deviation
  exceeding the threshold value defined by the parameter
  ``wfs_std_threshold``;
  
* ``ChargeDistributionFilter`` discards waveforms with a total charge integrated over the entire readout window in the tails of the charge distribution, either below ``charge_sigma_low_thr`` or above ``charge_sigma_high_thr``.


To run the example script:

.. code-block:: console

    $ python -i example_pedestal.py

Inspect the results
=========================
The results are stored in a
`~nectarchain.data.container.pedestal_container.NectarCAMPedestalContainer`. The
results include information on pixels that were flagged as having
an abnormal behavior during the computation of the pedestals. The
flags are defined in in
`~nectarchain.data.container.pedestal_container.PedestalFlagBits`. The
results are accessible on the fly if the tool is run interactively (as in the example above) and stored in a `.h5` file.

The user script `nectarchain/user_scripts/ltibaldo/show_pedestal_output.py` provides an example of how to access the results from disk and produce some plots:

.. code-block:: console

    $ python -i plot_pedestal_output.py


