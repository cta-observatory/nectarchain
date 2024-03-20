import tempfile
import os
import numpy as np
import tables
#from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam.constants import N_SAMPLES
from nectarchain.data.container import NectarCAMPedestalContainer
from nectarchain.makers.calibration import PedestalNectarCAMCalibrationTool

# FIXME
# For the moment rely on file stored on disk located at $NECTARCAMDATA/runs/
run_number = 3938
run_file = os.environ['NECTARCAMDATA']+'/runs/NectarCAM.Run3938.30events.fits.fz'
#run_file = get_dataset_path('NectarCAM.Run3938.30events.fits.fz')
# number of working pixels in run
n_pixels = 1834


class TestPedestalCalibrationTool:

    def test_base(self):
        """
        Test basic functionality, including IO on disk
        """

        # setup
        n_slices = 3
        events_per_slice = 10
        max_events = n_slices * events_per_slice
        with tempfile.TemporaryDirectory() as tmpdirname:
            outfile = tmpdirname + "/pedestal.h5"

            # run tool
            tool = PedestalNectarCAMCalibrationTool(
                run_number=run_number,
                run_file=run_file,
                max_events=max_events,
                events_per_slice=events_per_slice,
                log_level=0,
                output_path=outfile,
                overwrite=True,
                filter_method=None,
            )

            tool.initialize()
            tool.setup()

            tool.start()
            output = tool.finish(return_output_component=True)

            # Check output in memory
            assert output.nsamples == N_SAMPLES
            assert np.all(output.nevents == max_events)
            assert np.shape(output.pixels_id) == (n_pixels,)
            assert output.ucts_timestamp_min == np.uint64(1674462932637854793)
            assert output.ucts_timestamp_max == np.uint64(1674462932695877994)
            assert np.shape(output.pedestal_mean_hg) == (n_pixels, N_SAMPLES)
            assert np.shape(output.pedestal_mean_lg) == (n_pixels, N_SAMPLES)
            assert np.shape(output.pedestal_std_hg) == (n_pixels, N_SAMPLES)
            assert np.shape(output.pedestal_std_lg) == (n_pixels, N_SAMPLES)
            assert np.allclose(output.pedestal_mean_hg, 245., atol=20.)
            assert np.allclose(output.pedestal_mean_lg, 245., atol=20.)
            assert np.allclose(output.pedestal_std_hg, 10, atol=10)
            assert np.allclose(output.pedestal_std_lg, 2.5, atol=2.)

            # Check output on disk
            # FIXME: use tables for the moment, update when h5 reader in nectarchain is working
            with tables.open_file(outfile) as h5file:
                for s in range(n_slices):
                    # Check individual groups
                    group_name = "data_{}".format(s + 1)
                    assert group_name in h5file.root.__members__
                    table = h5file.root[group_name][NectarCAMPedestalContainer.__name__][0]
                    assert table['nsamples'] == N_SAMPLES
                    #assert np.all(table['nevents'] == events_per_slice)
                    assert np.shape(table['pixels_id']) == (n_pixels,)
                    assert np.shape(table['pedestal_mean_hg']) == (n_pixels, N_SAMPLES)
                    assert np.shape(table['pedestal_mean_lg']) == (n_pixels, N_SAMPLES)
                    assert np.shape(table['pedestal_std_hg']) == (n_pixels, N_SAMPLES)
                    assert np.shape(table['pedestal_std_lg']) == (n_pixels, N_SAMPLES)
                # Check combined results
                group_name = "data_combined"
                table = h5file.root[group_name][NectarCAMPedestalContainer.__name__][0]
                assert table['nsamples'] == N_SAMPLES
                assert np.all(table['nevents'] == max_events)
                assert np.shape(table['pixels_id']) == (n_pixels,)
                assert table['ucts_timestamp_min'] == np.uint64(1674462932637854793)
                assert table['ucts_timestamp_max'] == np.uint64(1674462932695877994)
                assert np.shape(table['pedestal_mean_hg']) == (n_pixels, N_SAMPLES)
                assert np.shape(table['pedestal_mean_lg']) == (n_pixels, N_SAMPLES)
                assert np.shape(table['pedestal_std_hg']) == (n_pixels, N_SAMPLES)
                assert np.shape(table['pedestal_std_lg']) == (n_pixels, N_SAMPLES)
                assert np.allclose(table['pedestal_mean_hg'], 245., atol=20.)
                assert np.allclose(table['pedestal_mean_lg'], 245., atol=20.)
                assert np.allclose(table['pedestal_std_hg'], 10, atol=10)
                assert np.allclose(table['pedestal_std_lg'], 2.5, atol=2.)

    def test_timesel(self):
        """
        Test time selection
        """

        # setup
        n_slices = 3
        events_per_slice = 10
        max_events = n_slices * events_per_slice
        tmin = 1674462932637860000
        tmax = 1674462932695700000
        with tempfile.TemporaryDirectory() as tmpdirname:
            outfile = tmpdirname + "/pedestal.h5"

            # run tool
            tool = PedestalNectarCAMCalibrationTool(
                run_number=run_number,
                run_file=run_file,
                max_events=max_events,
                events_per_slice=events_per_slice,
                log_level=0,
                output_path=outfile,
                ucts_tmin = tmin,
                ucts_tmax = tmax,
                overwrite=True,
                filter_method=None,
            )

            tool.initialize()
            tool.setup()

            tool.start()
            output = tool.finish(return_output_component=True)

            # check output
            assert output.nsamples == N_SAMPLES
            assert np.all(output.nevents <= max_events)
            assert np.shape(output.pixels_id) == (n_pixels,)
            assert output.ucts_timestamp_min >= tmin
            assert output.ucts_timestamp_max <= tmax
            assert np.shape(output.pedestal_mean_hg) == (n_pixels, N_SAMPLES)
            assert np.shape(output.pedestal_mean_lg) == (n_pixels, N_SAMPLES)
            assert np.shape(output.pedestal_std_hg) == (n_pixels, N_SAMPLES)
            assert np.shape(output.pedestal_std_lg) == (n_pixels, N_SAMPLES)
            assert np.allclose(output.pedestal_mean_hg, 245., atol=20.)
            assert np.allclose(output.pedestal_mean_lg, 245., atol=20.)
            assert np.allclose(output.pedestal_std_hg, 10, atol=10)
            assert np.allclose(output.pedestal_std_lg, 2.5, atol=2.)

    def test_WaveformsStdFilter(self):
        """
        Test filtering based on waveforms standard dev
        """

        # setup
        n_slices = 3
        events_per_slice = 10
        max_events = n_slices * events_per_slice
        with tempfile.TemporaryDirectory() as tmpdirname:
            outfile = tmpdirname + "/pedestal.h5"

            # run tool
            tool = PedestalNectarCAMCalibrationTool(
                run_number=run_number,
                run_file=run_file,
                max_events=max_events,
                events_per_slice=events_per_slice,
                log_level=0,
                output_path=outfile,
                overwrite=True,
                filter_method='WaveformsStdFilter',
                wfs_std_threshold = 4.,
            )

            tool.initialize()
            tool.setup()

            tool.start()
            output = tool.finish(return_output_component=True)

            # check output
            assert output.nsamples == N_SAMPLES
            assert np.all(output.nevents <= max_events)
            assert np.shape(output.pixels_id) == (n_pixels,)
            assert np.shape(output.pedestal_mean_hg) == (n_pixels, N_SAMPLES)
            assert np.shape(output.pedestal_mean_lg) == (n_pixels, N_SAMPLES)
            assert np.shape(output.pedestal_std_hg) == (n_pixels, N_SAMPLES)
            assert np.shape(output.pedestal_std_lg) == (n_pixels, N_SAMPLES)
            assert np.allclose(output.pedestal_mean_hg, 245., atol=20.)
            assert np.allclose(output.pedestal_mean_lg, 245., atol=20.)
            # verify that fluctuations are reduced
            assert np.allclose(output.pedestal_std_hg, 3., atol=2.)
            assert np.allclose(output.pedestal_std_lg, 2.5, atol=2.)

    def test_ChargeDistributionFilter(self):
        """
        Test filtering based on waveforms charge distribution
        """

        # setup
        n_slices = 2
        events_per_slice = 10
        max_events = n_slices * events_per_slice - 1
        with tempfile.TemporaryDirectory() as tmpdirname:
            outfile = tmpdirname + "/pedestal.h5"

            # run tool
            tool = PedestalNectarCAMCalibrationTool(
                run_number=run_number,
                run_file=run_file,
                max_events=max_events,
                events_per_slice=events_per_slice,
                log_level=0,
                output_path=outfile,
                overwrite=True,
                filter_method='ChargeDistributionFilter',
                charge_sigma_low_thr=1.,
                charge_sigma_high_thr=2.,
            )

            tool.initialize()
            tool.setup()

            tool.start()
            output = tool.finish(return_output_component=True)

            # check output
            assert output.nsamples == N_SAMPLES
            assert np.all(output.nevents <= max_events)
            assert np.shape(output.pixels_id) == (n_pixels,)
            assert np.shape(output.pedestal_mean_hg) == (n_pixels, N_SAMPLES)
            assert np.shape(output.pedestal_mean_lg) == (n_pixels, N_SAMPLES)
            assert np.shape(output.pedestal_std_hg) == (n_pixels, N_SAMPLES)
            assert np.shape(output.pedestal_std_lg) == (n_pixels, N_SAMPLES)
            assert np.allclose(output.pedestal_mean_hg, 245., atol=20.)
            assert np.allclose(output.pedestal_mean_lg, 245., atol=20.)
            assert np.allclose(output.pedestal_std_hg, 10., atol=10.)
            assert np.allclose(output.pedestal_std_lg, 2.5, atol=3.)
