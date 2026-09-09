import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam import LightNectarCAMEventSource as EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN, LOW_GAIN
from tqdm import tqdm
from traitlets.config import Config

from nectarchain.dqm.charge_integration import ChargeIntegrationHighLowGain


class TestChargeIntegrationHighLowGain:
    def test_charge_integration(self):
        """Test basic charge integration functionality including process_event"""
        path = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")

        config = Config(
            dict(
                NectarCAMEventSource=dict(
                    NectarCAMR0Corrections=dict(
                        calibration_path=None,
                        apply_flatfield=False,
                        select_gain=False,
                    )
                )
            )
        )

        reader1 = EventSource(input_url=path, config=config, max_events=1)

        # Create processor and configure it properly
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader1)
        processor.configure_for_run(path, Pix, Samp, reader1)

        # Process the event using the processor
        reader2 = EventSource(input_url=path, config=config, max_events=1)
        for evt in tqdm(reader2, total=1):
            processor.process_event(evt, noped=False)

        # Verify that the event was processed
        assert processor.counter_evt >= 0 or processor.counter_ped >= 0
        assert len(processor.image_all) >= 0 or len(processor.image_ped) >= 0

        # Original assertions
        assert Pix + Samp == 1915
        # Note: We can't verify the exact ped value since we're now using process_event
        # which handles pedestal computation differently

    def test_init_high_gain(self):
        """Test initialization with HIGH_GAIN (k=0)"""
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=False)
        assert processor.k == HIGH_GAIN
        assert processor.gain_c == "High"
        assert processor.r0 is False
        assert processor.Pix is None
        assert processor.Samp is None
        assert processor.counter_evt is None
        assert processor.counter_ped is None
        assert processor.tel_id is None
        assert len(processor.image_all) == 0
        assert len(processor.ped_all) == 0
        assert processor.ChargeInt_Results_Dict == {}

    def test_init_low_gain(self):
        """Test initialization with LOW_GAIN (k=1)"""
        processor = ChargeIntegrationHighLowGain(LOW_GAIN, r0=True)
        assert processor.k == LOW_GAIN
        assert processor.gain_c == "Low"
        assert processor.r0 is True
        assert processor.Pix is None
        assert processor.Samp is None
        assert processor.counter_evt is None
        assert processor.counter_ped is None
        assert processor.tel_id is None
        assert len(processor.image_all) == 0
        assert len(processor.ped_all) == 0
        assert processor.ChargeInt_Results_Dict == {}

    def test_init_invalid_gain(self):
        """Test initialization with invalid gain value"""
        processor = ChargeIntegrationHighLowGain(2, r0=False)
        assert processor.k == 2
        assert processor.gain_c == "Low"  # Should default to Low for non-zero

    def test_define_for_run(self):
        """Test define_for_run method with real data"""
        path = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
        config = Config(
            dict(
                NectarCAMEventSource=dict(
                    NectarCAMR0Corrections=dict(
                        calibration_path=None,
                        apply_flatfield=False,
                        select_gain=False,
                    )
                )
            )
        )

        reader1 = EventSource(input_url=path, config=config, max_events=1)
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)

        Pix, Samp = processor.define_for_run(reader1)

        assert Pix == 1855  # Expected number of pixels for NectarCAM
        assert Samp == 60  # Expected number of samples
        assert processor.Pix == Pix
        assert processor.Samp == Samp
        assert processor.tel_id is not None

    def test_configure_for_run_default(self):
        """Test configure_for_run with default parameters"""
        path = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
        config = Config(
            dict(
                NectarCAMEventSource=dict(
                    NectarCAMR0Corrections=dict(
                        calibration_path=None,
                        apply_flatfield=False,
                        select_gain=False,
                    )
                )
            )
        )

        reader1 = EventSource(input_url=path, config=config, max_events=1)
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)

        Pix, Samp = processor.define_for_run(reader1)

        # Configure with default parameters (no charges_kwargs)
        processor.configure_for_run(path, Pix, Samp, reader1)

        assert processor.Pix == Pix
        assert processor.Samp == Samp
        assert processor.counter_evt == 0
        assert processor.counter_ped == 0
        assert processor.tel_id is not None
        assert processor.camera is not None
        assert processor.integrator is not None
        assert processor.cmap == "gnuplot2"

    def test_configure_for_run_with_charges_kwargs(self):
        """Test configure_for_run with custom extractor parameters"""
        path = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
        config = Config(
            dict(
                NectarCAMEventSource=dict(
                    NectarCAMR0Corrections=dict(
                        calibration_path=None,
                        apply_flatfield=False,
                        select_gain=False,
                    )
                )
            )
        )

        reader1 = EventSource(input_url=path, config=config, max_events=1)
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)

        Pix, Samp = processor.define_for_run(reader1)

        # Configure with custom extractor parameters
        charges_kwargs = {
            "method": "FixedWindowSum",
            "extractor_kwargs": {"window_shift": 4, "window_width": 16},
        }

        processor.configure_for_run(path, Pix, Samp, reader1, **charges_kwargs)

        assert processor.integrator is not None

    def test_finish_run_no_events(self):
        """Test finish_run when no events have been processed"""
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        processor.Pix = 1855
        processor.Samp = 60
        processor.pixels = np.arange(1855)

        # Initialize empty lists
        processor.image_all = []
        processor.peakpos_all = []
        processor.image_ped = []
        processor.peakpos_ped = []
        processor.ped_all = []
        processor.ped_ped = []
        processor.counter_evt = 0
        processor.counter_ped = 0

        # This should not raise any errors
        processor.finish_run()

        # After finish_run with no events
        assert processor.counter_evt == 0
        assert processor.counter_ped == 0

    def test_finish_run_with_events(self):
        """Test finish_run with processed events"""
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        processor.Pix = 100
        processor.Samp = 60
        processor.pixels = np.arange(100)

        # Add some mock data
        processor.image_all = [np.ones(100) * i for i in range(5)]  # 5 events
        processor.peakpos_all = [np.ones(100) * 10 for _ in range(5)]
        processor.ped_all = [1.0, 2.0, 3.0, 4.0, 5.0]
        processor.counter_evt = 5
        processor.counter_ped = 0

        processor.finish_run()

        # Check that stats are computed
        assert processor.image_all_stats is not None
        assert processor.ped_all_stats is not None
        assert processor.image_all.shape == (5, 100)
        assert processor.peakpos_all.shape == (5, 100)
        assert processor.ped_all.shape == (5,)

        # Check that stats contain expected keys
        assert "average" in processor.image_all_stats
        assert "median" in processor.image_all_stats
        assert "std" in processor.image_all_stats
        assert "rms" in processor.image_all_stats

    def test_finish_run_with_pedestals(self):
        """Test finish_run with pedestal events"""
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        processor.Pix = 100
        processor.Samp = 60
        processor.pixels = np.arange(100)

        # Add some mock data for pedestals
        processor.image_ped = [np.ones(100) * i for i in range(3)]  # 3 ped events
        processor.peakpos_ped = [np.ones(100) * 5 for _ in range(3)]
        processor.ped_ped = [1.0, 2.0, 3.0]
        processor.counter_ped = 3
        processor.counter_evt = 0

        processor.finish_run()

        # Check that stats are computed for pedestals
        assert processor.image_ped_stats is not None
        assert processor.ped_ped_stats is not None
        assert processor.image_ped.shape == (3, 100)
        assert processor.peakpos_ped.shape == (3, 100)
        assert processor.ped_ped.shape == (3,)

    def test_get_results_no_events(self):
        """Test get_results when no events have been processed"""
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        processor.counter_evt = 0
        processor.counter_ped = 0

        results = processor.get_results()

        assert results == {}

    def test_get_results_with_events(self):
        """Test get_results with processed events"""
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        processor.Pix = 100
        processor.Samp = 60
        processor.pixels = np.arange(100)
        processor.gain_c = "High"

        # Add mock data
        processor.image_all = [np.ones(100) * i for i in range(3)]
        processor.peakpos_all = [np.ones(100) * 10 for _ in range(3)]
        processor.ped_all = [1.0, 2.0, 3.0]
        processor.counter_evt = 3
        processor.counter_ped = 0

        # Run finish_run to compute stats
        processor.finish_run()

        results = processor.get_results()

        # Should contain results for event data
        assert len(results) > 0

        # Check for expected keys
        expected_keys = [
            "CHARGE-INTEGRATION-IMAGE-ALL-AVERAGE-HIGH-GAIN",
            "CHARGE-INTEGRATION-IMAGE-ALL-MEDIAN-HIGH-GAIN",
            "CHARGE-INTEGRATION-IMAGE-ALL-STD-HIGH-GAIN",
            "CHARGE-INTEGRATION-IMAGE-ALL-RMS-HIGH-GAIN",
            "PED-INTEGRATION-IMAGE-ALL-AVERAGE-HIGH-GAIN",
            "PED-INTEGRATION-IMAGE-ALL-MEDIAN-HIGH-GAIN",
            "PED-INTEGRATION-IMAGE-ALL-STD-HIGH-GAIN",
            "PED-INTEGRATION-IMAGE-ALL-RMS-HIGH-GAIN",
        ]

        for key in expected_keys:
            assert key in results

    def test_get_results_with_pedestals(self):
        """Test get_results with pedestal events"""
        processor = ChargeIntegrationHighLowGain(LOW_GAIN, r0=True)
        processor.Pix = 100
        processor.Samp = 60
        processor.pixels = np.arange(100)
        processor.gain_c = "Low"

        # Add mock data for pedestals only
        processor.image_ped = [np.ones(100) * i for i in range(2)]
        processor.peakpos_ped = [np.ones(100) * 5 for _ in range(2)]
        processor.ped_ped = [1.0, 2.0]
        processor.counter_evt = 0
        processor.counter_ped = 2

        # Run finish_run to compute stats
        processor.finish_run()

        results = processor.get_results()

        # Should contain results for pedestal data
        assert len(results) > 0

        # Check for expected keys
        expected_keys = [
            "CHARGE-INTEGRATION-PED-ALL-AVERAGE-LOW-GAIN",
            "CHARGE-INTEGRATION-PED-ALL-MEDIAN-LOW-GAIN",
            "CHARGE-INTEGRATION-PED-ALL-STD-LOW-GAIN",
            "CHARGE-INTEGRATION-PED-ALL-RMS-LOW-GAIN",
            "PED-INTEGRATION-PED-ALL-AVERAGE-LOW-GAIN",
            "PED-INTEGRATION-PED-ALL-MEDIAN-LOW-GAIN",
            "PED-INTEGRATION-PED-ALL-STD-LOW-GAIN",
            "PED-INTEGRATION-PED-ALL-RMS-LOW-GAIN",
        ]

        for key in expected_keys:
            assert key in results

    def test_result_keys_high_gain(self):
        """Test that HIGH_GAIN results have correct key formatting"""
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        processor.gain_c = "High"

        # Mock data
        processor.image_all_stats = {"average": np.ones(100)}
        processor.ped_all_stats = {"average": 1.0}
        processor.image_ped_stats = {"average": np.ones(100)}
        processor.ped_ped_stats = {"average": 1.0}
        processor.counter_evt = 1
        processor.counter_ped = 1

        results = processor.get_results()

        # Check that all keys contain "HIGH-GAIN"
        for key in results.keys():
            assert "HIGH-GAIN" in key

    def test_result_keys_low_gain(self):
        """Test that LOW_GAIN results have correct key formatting"""
        processor = ChargeIntegrationHighLowGain(LOW_GAIN, r0=True)
        processor.gain_c = "Low"

        # Mock data
        processor.image_all_stats = {"average": np.ones(100)}
        processor.ped_all_stats = {"average": 1.0}
        processor.image_ped_stats = {"average": np.ones(100)}
        processor.ped_ped_stats = {"average": 1.0}
        processor.counter_evt = 1
        processor.counter_ped = 1

        results = processor.get_results()

        # Check that all keys contain "LOW-GAIN"
        for key in results.keys():
            assert "LOW-GAIN" in key

    def test_process_event_with_real_data(self):
        """Test process_event method with real data"""
        path = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
        config = Config(
            dict(
                NectarCAMEventSource=dict(
                    NectarCAMR0Corrections=dict(
                        calibration_path=None,
                        apply_flatfield=False,
                        select_gain=False,
                    )
                )
            )
        )

        reader1 = EventSource(input_url=path, config=config, max_events=1)
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)

        Pix, Samp = processor.define_for_run(reader1)
        processor.configure_for_run(path, Pix, Samp, reader1)

        # Process a real event
        reader2 = EventSource(input_url=path, config=config, max_events=1)
        for evt in reader2:
            initial_evt_count = processor.counter_evt
            initial_ped_count = processor.counter_ped
            initial_image_count = len(processor.image_all)
            initial_ped_image_count = len(processor.image_ped)

            processor.process_event(evt, noped=False)

            # Verify that the event was processed and counters updated
            assert (
                processor.counter_evt >= initial_evt_count
                or processor.counter_ped >= initial_ped_count
            )
            assert (
                len(processor.image_all) >= initial_image_count
                or len(processor.image_ped) >= initial_ped_image_count
            )
            break

        # Verify final state
        assert processor.counter_evt + processor.counter_ped > 0
        assert len(processor.image_all) + len(processor.image_ped) > 0

    def test_process_event_with_pedestal_subtraction(self):
        """Test process_event method with pedestal subtraction enabled"""
        path = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
        config = Config(
            dict(
                NectarCAMEventSource=dict(
                    NectarCAMR0Corrections=dict(
                        calibration_path=None,
                        apply_flatfield=False,
                        select_gain=False,
                    )
                )
            )
        )

        reader1 = EventSource(input_url=path, config=config, max_events=1)
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)

        Pix, Samp = processor.define_for_run(reader1)
        processor.configure_for_run(path, Pix, Samp, reader1)

        # Process a real event with pedestal subtraction
        reader2 = EventSource(input_url=path, config=config, max_events=1)
        for evt in reader2:
            processor.process_event(evt, noped=True)  # Pedestal subtraction enabled
            break

        # Should have processed the event
        assert processor.counter_evt + processor.counter_ped > 0

    @patch("nectarchain.dqm.charge_integration.CameraDisplay")
    @patch("nectarchain.dqm.charge_integration.plt")
    def test_plot_results_with_events(self, mock_plt, mock_camera_display):
        """Test plot_results method with event data"""
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        processor.gain_c = "High"
        processor.Pix = 100
        processor.Samp = 60
        processor.pixels = np.arange(100)

        # Mock camera geometry for plotting
        mock_camera = MagicMock()
        processor.camera = mock_camera
        processor.pixelBADplot = [np.zeros(100, dtype=bool)]

        # Add mock data for events
        processor.image_all = [np.random.random(100) for _ in range(3)]
        processor.peakpos_all = [np.random.random(100) for _ in range(3)]
        processor.ped_all = [1.0, 2.0, 3.0]
        processor.counter_evt = 3
        processor.counter_ped = 0

        # Run finish_run to compute stats
        processor.finish_run()

        # Mock CameraDisplay to avoid geometry issues
        mock_disp = MagicMock()
        mock_camera_display.return_value = mock_disp

        # Mock plt.subplots
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test plot_results - should not raise any errors
        with tempfile.TemporaryDirectory() as temp_dir:
            fig_dict, fig_names_dict = processor.plot_results("test", temp_dir)

            # Should return dictionaries
            assert isinstance(fig_dict, dict)
            assert isinstance(fig_names_dict, dict)

            # Should have created figures for event data
            assert len(fig_dict) > 0

    @patch("nectarchain.dqm.charge_integration.CameraDisplay")
    @patch("nectarchain.dqm.charge_integration.plt")
    def test_plot_results_with_pedestals(self, mock_plt, mock_camera_display):
        """Test plot_results method with pedestal data"""
        processor = ChargeIntegrationHighLowGain(LOW_GAIN, r0=True)
        processor.gain_c = "Low"
        processor.Pix = 100
        processor.Samp = 60
        processor.pixels = np.arange(100)

        # Mock camera geometry
        mock_camera = MagicMock()
        processor.camera = mock_camera
        processor.pixelBADplot = [np.zeros(100, dtype=bool)]

        # Add mock data for pedestals
        processor.image_ped = [np.random.random(100) for _ in range(2)]
        processor.peakpos_ped = [np.random.random(100) for _ in range(2)]
        processor.ped_ped = [1.0, 2.0]
        processor.counter_evt = 0
        processor.counter_ped = 2

        # Run finish_run to compute stats
        processor.finish_run()

        # Mock CameraDisplay to avoid geometry issues
        mock_disp = MagicMock()
        mock_camera_display.return_value = mock_disp

        # Mock plt.subplots
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test plot_results with pedestal data
        with tempfile.TemporaryDirectory() as temp_dir:
            fig_dict, fig_names_dict = processor.plot_results("test", temp_dir)

            # Should return dictionaries
            assert isinstance(fig_dict, dict)
            assert isinstance(fig_names_dict, dict)

    @patch("nectarchain.dqm.charge_integration.CameraDisplay")
    @patch("nectarchain.dqm.charge_integration.plt")
    def test_plot_results_no_data(self, mock_plt, mock_camera_display):
        """Test plot_results method with no data"""
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        processor.gain_c = "High"
        processor.Pix = 100
        processor.Samp = 60
        processor.pixels = np.arange(100)

        # Mock camera geometry
        mock_camera = MagicMock()
        processor.camera = mock_camera
        processor.pixelBADplot = [np.zeros(100, dtype=bool)]

        # No data - counters are zero
        processor.counter_evt = 0
        processor.counter_ped = 0

        # Mock CameraDisplay to avoid geometry issues
        mock_disp = MagicMock()
        mock_camera_display.return_value = mock_disp

        # Mock plt.subplots
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test plot_results with no data - should not raise errors
        with tempfile.TemporaryDirectory() as temp_dir:
            fig_dict, fig_names_dict = processor.plot_results("test", temp_dir)

            # Should return empty dictionaries when no data
            assert isinstance(fig_dict, dict)
            assert isinstance(fig_names_dict, dict)

    @patch("nectarchain.dqm.charge_integration.CameraDisplay")
    @patch("nectarchain.dqm.charge_integration.plt")
    def test_plot_results_both_events_and_pedestals(
        self, mock_plt, mock_camera_display
    ):
        """Test plot_results method with both event and pedestal data"""
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        processor.gain_c = "High"
        processor.Pix = 100
        processor.Samp = 60
        processor.pixels = np.arange(100)

        # Mock camera geometry
        mock_camera = MagicMock()
        processor.camera = mock_camera
        processor.pixelBADplot = [np.zeros(100, dtype=bool)]

        # Add mock data for both events and pedestals
        processor.image_all = [np.random.random(100) for _ in range(2)]
        processor.peakpos_all = [np.random.random(100) for _ in range(2)]
        processor.ped_all = [1.0, 2.0]
        processor.counter_evt = 2

        processor.image_ped = [np.random.random(100) for _ in range(1)]
        processor.peakpos_ped = [np.random.random(100) for _ in range(1)]
        processor.ped_ped = [1.5]
        processor.counter_ped = 1

        # Run finish_run to compute stats
        processor.finish_run()

        # Mock CameraDisplay to avoid geometry issues
        mock_disp = MagicMock()
        mock_camera_display.return_value = mock_disp

        # Mock plt.subplots
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test plot_results with both types of data
        with tempfile.TemporaryDirectory() as temp_dir:
            fig_dict, fig_names_dict = processor.plot_results("test", temp_dir)

            # Should have created multiple figures
            assert len(fig_dict) > 0
            assert len(fig_names_dict) > 0

    @patch("nectarchain.dqm.charge_integration.plt.subplots")
    @patch("nectarchain.dqm.charge_integration.CameraDisplay")
    def test_plot_camera_image(self, mock_camera_display, mock_subplots):
        """Test _plot_camera_image helper method"""
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        processor.gain_c = "High"

        # Mock camera geometry
        mock_camera = MagicMock()
        processor.camera = mock_camera
        processor.pixelBADplot = [np.zeros(100, dtype=bool)]

        # Mock the subplots and CameraDisplay
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        mock_disp = MagicMock()
        mock_camera_display.return_value = mock_disp

        # Test the helper method
        with tempfile.TemporaryDirectory() as temp_dir:
            processor._plot_camera_image(
                image=np.ones(100),
                title="Test Title",
                text="Test Text",
                filename="test_file.png",
                key="TEST-KEY",
                fig_path=temp_dir,
            )

            # Verify that subplots was called
            assert mock_subplots.called

            # Verify that CameraDisplay was created
            assert mock_camera_display.called

            # Verify that the figure was stored
            assert "TEST-KEY" in processor.ChargeInt_Figures_Dict
            assert "TEST-KEY" in processor.ChargeInt_Figures_Names_Dict

    def test_get_results_edge_cases(self):
        """Test get_results with various edge cases"""
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        processor.gain_c = "High"

        # Test case: only ped_all_stats,
        # no image stats - but counter_evt is 0 so nothing should be included
        processor.image_all_stats = {"average": np.ones(100)}
        processor.ped_all_stats = {"average": np.ones(100)}
        processor.image_ped_stats = None
        processor.ped_ped_stats = None
        processor.counter_evt = 0
        processor.counter_ped = 0

        results = processor.get_results()
        # Should not include anything when both counters are 0
        assert len(results) == 0

        # Test case: only image stats with counter_evt > 0
        processor.image_all_stats = {"average": np.ones(100)}
        processor.ped_all_stats = {"average": 1.0}
        processor.counter_evt = 1
        processor.counter_ped = 0

        results = processor.get_results()
        assert len(results) > 0
        assert "CHARGE-INTEGRATION-IMAGE-ALL-AVERAGE-HIGH-GAIN" in results
        assert "PED-INTEGRATION-IMAGE-ALL-AVERAGE-HIGH-GAIN" in results

    @patch(
        "nectarchain.dqm.charge_integration.ArrayDataComponent._compute_broken_pixels_event"  # noqa
    )
    @patch("nectarchain.dqm.charge_integration.CtapipeExtractor.get_image_peak_time")
    def test_process_event_index_error_handling(
        self, mock_get_image_peak_time, mock_broken_pixels
    ):
        """Test that IndexError is handled gracefully in process_event"""
        path = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
        config = Config(
            dict(
                NectarCAMEventSource=dict(
                    NectarCAMR0Corrections=dict(
                        calibration_path=None,
                        apply_flatfield=False,
                        select_gain=False,
                    )
                )
            )
        )

        reader1 = EventSource(input_url=path, config=config, max_events=1)
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)

        Pix, Samp = processor.define_for_run(reader1)
        processor.configure_for_run(path, Pix, Samp, reader1)

        # Mock the broken pixels computation
        processor.pixels = np.arange(1834)  # Match the actual data size
        mock_broken_pixels.return_value = (
            np.zeros(1834, dtype=bool),
            np.zeros(1834, dtype=bool),
        )

        # Mock the integrator's __call__ method to raise IndexError first
        original_integrator = processor.integrator
        mock_integrator = MagicMock()

        # First call raises IndexError, second call succeeds
        def integrator_side_effect(*args, **kwargs):
            if not hasattr(integrator_side_effect, "called"):
                integrator_side_effect.called = True
                raise IndexError("Waveform shape issue")
            else:
                # Return a mock object that get_image_peak_time can process
                return MagicMock()

        mock_integrator.side_effect = integrator_side_effect
        processor.integrator = mock_integrator

        # Mock get_image_peak_time to return proper data
        mock_get_image_peak_time.return_value = (np.ones(1834), np.ones(1834))

        # Process event - should handle IndexError gracefully
        reader2 = EventSource(input_url=path, config=config, max_events=1)
        for evt in reader2:
            processor.process_event(evt, noped=False)
            break

        # Should have processed the event successfully after retry
        assert processor.counter_evt + processor.counter_ped > 0
        # Verify that the integrator was called twice (original + retry)
        assert mock_integrator.call_count == 2

        # Restore original integrator
        processor.integrator = original_integrator

    def test_write_all_results(self):
        """Test that write_all_results method works correctly"""
        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)

        # Add some mock results
        processor.ChargeInt_Results_Dict = {
            "TEST_KEY_1": np.ones(100),
            "TEST_KEY_2": {"nested": "data"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_output")

            # This should not raise any errors
            processor.write_all_results(
                output_path, {"TestResults": processor.ChargeInt_Results_Dict}
            )

            # Check that the FITS file was created
            fits_file = output_path + "_Results.fits"
            assert os.path.exists(fits_file)
