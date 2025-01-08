import tempfile

import numpy as np
import tables
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam.constants import N_SAMPLES

from nectarchain.data.container import NectarCAMPedestalContainer, PedestalFlagBits
from nectarchain.makers.calibration import PedestalNectarCAMCalibrationTool

runs = {
    "Run number": [3938, 5288],
    "Run file": [
        get_dataset_path("NectarCAM.Run3938.30events.fits.fz"),
        get_dataset_path("NectarCAM.Run5288.0001.fits.fz"),
    ],
    "N pixels": [1834, 1848],
}


class TestPedestalCalibrationTool:
    def test_base(self):
        """
        Test basic functionality, including IO on disk
        """

        # setup
        n_slices = [3, 2]
        events_per_slice = 10
        max_events = [n_slices[0] * events_per_slice, 13]

        expected_ucts_timestamp_min = [1674462932637854793, 1715007113924900896]
        expected_ucts_timestamp_max = [1674462932695877994, 1715007123524920096]

        for i, run_number in enumerate(runs["Run number"]):
            run_file = runs["Run file"][i]
            n_pixels = runs["N pixels"][i]
            with tempfile.TemporaryDirectory() as tmpdirname:
                outfile = tmpdirname + "/pedestal.h5"

                # run tool
                tool = PedestalNectarCAMCalibrationTool(
                    run_number=run_number,
                    run_file=run_file,
                    max_events=max_events[i],
                    events_per_slice=events_per_slice,
                    log_level=0,
                    output_path=outfile,
                    overwrite=True,
                    filter_method=None,
                    pixel_mask_nevents_min=1,
                )

                # tool.initialize()
                tool.setup()

                tool.start()
                output = tool.finish(return_output_component=True)

                # Check output in memory
                assert output.nsamples == N_SAMPLES
                assert np.all(output.nevents == max_events[i])
                assert np.shape(output.pixels_id) == (n_pixels,)
                assert output.ucts_timestamp_min == np.uint64(
                    expected_ucts_timestamp_min[i]
                )
                assert output.ucts_timestamp_max == np.uint64(
                    expected_ucts_timestamp_max[i]
                )
                assert np.shape(output.pedestal_mean_hg) == (n_pixels, N_SAMPLES)
                assert np.shape(output.pedestal_mean_lg) == (n_pixels, N_SAMPLES)
                assert np.shape(output.pedestal_std_hg) == (n_pixels, N_SAMPLES)
                assert np.shape(output.pedestal_std_lg) == (n_pixels, N_SAMPLES)
                assert np.allclose(output.pedestal_mean_hg, 245.0, atol=20.0)
                assert np.allclose(output.pedestal_mean_lg, 245.0, atol=20.0)
                assert np.allclose(output.pedestal_std_hg, 10, atol=10)
                assert np.allclose(output.pedestal_std_lg, 2.5, atol=2.3)

                # Check output on disk
                # FIXME: use tables for the moment, update when h5 reader in nectarchain
                #  is working
                with tables.open_file(outfile) as h5file:
                    for s in range(n_slices[i]):
                        # Check individual groups
                        group_name = "data_{}".format(s + 1)
                        assert group_name in h5file.root.__members__
                        table = h5file.root[group_name][
                            NectarCAMPedestalContainer.__name__
                        ][0]
                        assert table["nsamples"] == N_SAMPLES
                        assert np.allclose(table["nevents"], events_per_slice, atol=7)
                        assert np.shape(table["pixels_id"]) == (n_pixels,)
                        assert np.shape(table["pedestal_mean_hg"]) == (
                            n_pixels,
                            N_SAMPLES,
                        )
                        assert np.shape(table["pedestal_mean_lg"]) == (
                            n_pixels,
                            N_SAMPLES,
                        )
                        assert np.shape(table["pedestal_std_hg"]) == (
                            n_pixels,
                            N_SAMPLES,
                        )
                        assert np.shape(table["pedestal_std_lg"]) == (
                            n_pixels,
                            N_SAMPLES,
                        )
                    # Check combined results
                    group_name = "data_combined"
                    table = h5file.root[group_name][
                        NectarCAMPedestalContainer.__name__
                    ][0]
                    assert table["nsamples"] == N_SAMPLES
                    assert np.all(table["nevents"] == max_events[i])
                    assert np.shape(table["pixels_id"]) == (n_pixels,)
                    assert table["ucts_timestamp_min"] == np.uint64(
                        expected_ucts_timestamp_min[i]
                    )
                    assert table["ucts_timestamp_max"] == np.uint64(
                        expected_ucts_timestamp_max[i]
                    )
                    assert np.shape(table["pedestal_mean_hg"]) == (n_pixels, N_SAMPLES)
                    assert np.shape(table["pedestal_mean_lg"]) == (n_pixels, N_SAMPLES)
                    assert np.shape(table["pedestal_std_hg"]) == (n_pixels, N_SAMPLES)
                    assert np.shape(table["pedestal_std_lg"]) == (n_pixels, N_SAMPLES)
                    assert np.allclose(table["pedestal_mean_hg"], 245.0, atol=20.0)
                    assert np.allclose(table["pedestal_mean_lg"], 245.0, atol=20.0)
                    assert np.allclose(table["pedestal_std_hg"], 10, atol=10)
                    assert np.allclose(
                        table["pedestal_std_lg"], 2.5, atol=2.0 if i == 0 else 2.3
                    )

    def test_timesel(self):
        """
        Test time selection
        """

        # setup
        n_slices = [3, 2]
        events_per_slice = 10
        max_events = [n_slices[0] * events_per_slice, 13]
        tmin = [1674462932637860000, 1715007113924900000]
        tmax = [1674462932695700000, 1715007123524921000]
        for i, run in enumerate(runs["Run number"]):
            run_number = runs["Run number"][i]
            run_file = runs["Run file"][i]
            n_pixels = runs["N pixels"][i]

            with tempfile.TemporaryDirectory() as tmpdirname:
                outfile = tmpdirname + "/pedestal.h5"

                # run tool
                tool = PedestalNectarCAMCalibrationTool(
                    run_number=run_number,
                    run_file=run_file,
                    max_events=max_events[i],
                    events_per_slice=events_per_slice,
                    log_level=0,
                    output_path=outfile,
                    ucts_tmin=tmin[i],
                    ucts_tmax=tmax[i],
                    overwrite=True,
                    filter_method=None,
                    pixel_mask_nevents_min=1,
                )

                # tool.initialize()
                tool.setup()

                tool.start()
                output = tool.finish(return_output_component=True)

                # check output
                assert output.nsamples == N_SAMPLES
                assert np.all(output.nevents <= max_events[i])
                assert np.shape(output.pixels_id) == (n_pixels,)
                assert output.ucts_timestamp_min >= tmin[i]
                assert output.ucts_timestamp_max <= tmax[i]
                assert np.shape(output.pedestal_mean_hg) == (n_pixels, N_SAMPLES)
                assert np.shape(output.pedestal_mean_lg) == (n_pixels, N_SAMPLES)
                assert np.shape(output.pedestal_std_hg) == (n_pixels, N_SAMPLES)
                assert np.shape(output.pedestal_std_lg) == (n_pixels, N_SAMPLES)
                assert np.allclose(output.pedestal_mean_hg, 245.0, atol=20.0)
                assert np.allclose(output.pedestal_mean_lg, 245.0, atol=20.0)
                assert np.allclose(output.pedestal_std_hg, 10, atol=10)
                assert np.allclose(
                    output.pedestal_std_lg, 2.5, atol=2.0 if i == 0 else 2.3
                )

    def test_WaveformsStdFilter(self):
        """
        Test filtering based on waveforms standard dev
        """

        # setup
        n_slices = [3, 2]
        events_per_slice = 10
        max_events = [n_slices[0] * events_per_slice, 13]
        for i, run in enumerate(runs["Run number"]):
            run_number = runs["Run number"][i]
            run_file = runs["Run file"][i]
            n_pixels = runs["N pixels"][i]
            with tempfile.TemporaryDirectory() as tmpdirname:
                outfile = tmpdirname + "/pedestal.h5"

                # run tool
                tool = PedestalNectarCAMCalibrationTool(
                    run_number=run_number,
                    run_file=run_file,
                    max_events=max_events[i],
                    events_per_slice=events_per_slice,
                    log_level=0,
                    output_path=outfile,
                    overwrite=True,
                    filter_method="WaveformsStdFilter",
                    wfs_std_threshold=4.0,
                    pixel_mask_nevents_min=1,
                )

                # tool.initialize()
                tool.setup()

                tool.start()
                output = tool.finish(return_output_component=True)

                # check output
                assert output.nsamples == N_SAMPLES
                assert np.all(output.nevents <= max_events[i])
                assert np.shape(output.pixels_id) == (n_pixels,)
                assert np.shape(output.pedestal_mean_hg) == (n_pixels, N_SAMPLES)
                assert np.shape(output.pedestal_mean_lg) == (n_pixels, N_SAMPLES)
                assert np.shape(output.pedestal_std_hg) == (n_pixels, N_SAMPLES)
                assert np.shape(output.pedestal_std_lg) == (n_pixels, N_SAMPLES)
                assert np.allclose(output.pedestal_mean_hg, 245.0, atol=20.0)
                assert np.allclose(output.pedestal_mean_lg, 245.0, atol=20.0)
                # verify that fluctuations are reduced
                assert np.allclose(
                    output.pedestal_std_hg, 3.0, atol=2.0 if i == 0 else 4.0
                )
                assert np.allclose(
                    output.pedestal_std_lg, 2.5, atol=2.0 if i == 0 else 2.6
                )

    def test_ChargeDistributionFilter(self):
        """
        Test filtering based on waveforms charge distribution
        """

        # setup
        n_slices = [2, 1]
        events_per_slice = 10
        max_events = [n_slices[0] * events_per_slice - 1, 12]
        for i, run in enumerate(runs["Run number"]):
            run_number = runs["Run number"][i]
            run_file = runs["Run file"][i]
            n_pixels = runs["N pixels"][i]
            with tempfile.TemporaryDirectory() as tmpdirname:
                outfile = tmpdirname + "/pedestal.h5"

                # run tool
                tool = PedestalNectarCAMCalibrationTool(
                    run_number=run_number,
                    run_file=run_file,
                    max_events=max_events[i],
                    events_per_slice=events_per_slice,
                    log_level=0,
                    output_path=outfile,
                    overwrite=True,
                    filter_method="ChargeDistributionFilter",
                    charge_sigma_low_thr=1.0,
                    charge_sigma_high_thr=2.0,
                    pixel_mask_nevents_min=1,
                )

                # tool.initialize()
                tool.setup()

                tool.start()
                output = tool.finish(return_output_component=True)

                # check output
                assert output.nsamples == N_SAMPLES
                assert np.all(output.nevents <= max_events[i])
                assert np.shape(output.pixels_id) == (n_pixels,)
                assert np.shape(output.pedestal_mean_hg) == (n_pixels, N_SAMPLES)
                assert np.shape(output.pedestal_mean_lg) == (n_pixels, N_SAMPLES)
                assert np.shape(output.pedestal_std_hg) == (n_pixels, N_SAMPLES)
                assert np.shape(output.pedestal_std_lg) == (n_pixels, N_SAMPLES)
                assert np.allclose(output.pedestal_mean_hg, 245.0, atol=20.0)
                assert np.allclose(output.pedestal_mean_lg, 245.0, atol=20.0)
                assert np.allclose(output.pedestal_std_hg, 10.0, atol=10.0)
                assert np.allclose(
                    output.pedestal_std_lg, 2.5 if i == 0 else 2.2, atol=3.0
                )

    def test_pixel_mask(self):
        """
        Test that bad pixels are correctly recognized and flagged
        """

        # Flag number
        # setup
        max_events = 10
        # just test one run to be faster
        run_number = runs["Run number"][0]
        run_file = runs["Run file"][0]

        with tempfile.TemporaryDirectory() as tmpdirname:
            outfile = tmpdirname + "/pedestal.h5"

            # Condition on number of events
            # run tool
            tool = PedestalNectarCAMCalibrationTool(
                run_number=run_number,
                run_file=run_file,
                max_events=max_events,
                log_level=0,
                output_path=outfile,
                overwrite=True,
                filter_method=None,
            )

            # tool.initialize()
            tool.setup()

            tool.start()
            output = tool.finish(return_output_component=True)[0]

            # Check that all pixels were flagged as having not enough events
            flag_bit = PedestalFlagBits.NEVENTS
            assert np.all(output.pixel_mask & flag_bit == flag_bit)
            # Check that other flags were not raised
            flag_bits = [
                PedestalFlagBits.MEAN_PEDESTAL,
                PedestalFlagBits.STD_SAMPLE,
                PedestalFlagBits.STD_PIXEL,
            ]
            for flag_bit in flag_bits:
                assert np.all(output.pixel_mask & flag_bit == 0)

            # For all the following tests we set the acceptable values to a range
            # out of what
            # is normal. Since our test run is good we expect to flag all pixels

            # Condition on mean pedestal value
            # run tool
            tool = PedestalNectarCAMCalibrationTool(
                run_number=run_number,
                run_file=run_file,
                max_events=max_events,
                log_level=0,
                output_path=outfile,
                overwrite=True,
                filter_method=None,
                pixel_mask_nevents_min=1,
                pixel_mask_mean_min=1000.0,
                pixel_mask_mean_max=1100.0,
            )

            tool.initialize()
            tool.setup()

            tool.start()
            output = tool.finish(return_output_component=True)[0]

            # Check that all pixels were flagged as having a bad mean
            flag_bit = PedestalFlagBits.MEAN_PEDESTAL
            assert np.all(output.pixel_mask & flag_bit == flag_bit)
            # Check that other flags were not raised
            flag_bits = [
                PedestalFlagBits.NEVENTS,
                PedestalFlagBits.STD_SAMPLE,
                PedestalFlagBits.STD_PIXEL,
            ]
            for flag_bit in flag_bits:
                assert np.all(output.pixel_mask & flag_bit == 0)

            # Condition on sample std
            # run tool
            tool = PedestalNectarCAMCalibrationTool(
                run_number=run_number,
                run_file=run_file,
                max_events=max_events,
                log_level=0,
                output_path=outfile,
                overwrite=True,
                filter_method=None,
                pixel_mask_nevents_min=1,
                pixel_mask_std_sample_min=100.0,
            )

            tool.initialize()
            tool.setup()

            tool.start()
            output = tool.finish(return_output_component=True)[0]

            # Check that all pixels were flagged as
            # having a small sample std
            flag_bit = PedestalFlagBits.STD_SAMPLE
            assert np.all(output.pixel_mask & flag_bit == flag_bit)
            # Check that other flags were not raised
            flag_bits = [
                PedestalFlagBits.NEVENTS,
                PedestalFlagBits.MEAN_PEDESTAL,
                PedestalFlagBits.STD_PIXEL,
            ]
            for flag_bit in flag_bits:
                assert np.all(output.pixel_mask & flag_bit == 0)

            # Condition on pixel std
            # run tool
            tool = PedestalNectarCAMCalibrationTool(
                run_number=run_number,
                run_file=run_file,
                max_events=max_events,
                log_level=0,
                output_path=outfile,
                overwrite=True,
                filter_method=None,
                pixel_mask_nevents_min=1,
                pixel_mask_std_pixel_max=0.01,
            )

            # tool.initialize()
            tool.setup()

            tool.start()
            output = tool.finish(return_output_component=True)[0]

            # Check that all pixels were flagged as having a large pixel std
            flag_bit = PedestalFlagBits.STD_PIXEL
            assert np.all(output.pixel_mask & flag_bit == flag_bit)
            # Check that other flags were not raised
            flag_bits = [
                PedestalFlagBits.NEVENTS,
                PedestalFlagBits.MEAN_PEDESTAL,
                PedestalFlagBits.STD_SAMPLE,
            ]
            for flag_bit in flag_bits:
                assert np.all(output.pixel_mask & flag_bit == 0)
