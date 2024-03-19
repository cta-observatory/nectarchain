import tempfile

import numpy as np
import tables
from ctapipe_io_nectarcam.constants import N_SAMPLES
from nectarchain.data.container import NectarCAMPedestalContainer
from nectarchain.makers.calibration import PedestalNectarCAMCalibrationTool

# FIXME
# For the moment rely on file stored on disk located at $NECTARCAMDATA/runs/
# or retrived via ctadirac on the fly
run_number = 3938
# number of working pixels in run
n_pixels = 1834


class TestPedestalCalibrationTool:

    def test_base(self):
        # run tool
        n_slices = 3
        events_per_slice = 10
        max_events = n_slices * events_per_slice
        with tempfile.TemporaryDirectory() as tmpdirname:
            outfile = tmpdirname + "/pedestal.h5"

            tool = PedestalNectarCAMCalibrationTool(
                progress_bar=True,
                run_number=run_number,
                max_events=max_events,
                events_per_slice=events_per_slice,
                log_level=20,
                output_path=outfile,
                overwrite=True,
                filter_method='None',
            )

            tool.initialize()
            tool.setup()

            tool.start()
            output = tool.finish(return_output_component=True)

            # Check output in memory
            assert output.nsamples == N_SAMPLES
            assert np.all(output.nevents == max_events)
            assert np.shape(output.pixels_id) == (n_pixels,)
            assert output.ucts_timestamp_min == np.uint64(1674462932636854394)
            assert output.ucts_timestamp_max == np.uint64(1674462932665865994)
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
                    assert np.all(table['nevents'] == events_per_slice)
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
                assert table['ucts_timestamp_min'] == np.uint64(1674462932636854394)
                assert table['ucts_timestamp_max'] == np.uint64(1674462932665865994)
                assert np.shape(table['pedestal_mean_hg']) == (n_pixels, N_SAMPLES)
                assert np.shape(table['pedestal_mean_lg']) == (n_pixels, N_SAMPLES)
                assert np.shape(table['pedestal_std_hg']) == (n_pixels, N_SAMPLES)
                assert np.shape(table['pedestal_std_lg']) == (n_pixels, N_SAMPLES)
                assert np.allclose(table['pedestal_mean_hg'], 245., atol=20.)
                assert np.allclose(table['pedestal_mean_lg'], 245., atol=20.)
                assert np.allclose(table['pedestal_std_hg'], 10, atol=10)
                assert np.allclose(table['pedestal_std_lg'], 2.5, atol=2.)
