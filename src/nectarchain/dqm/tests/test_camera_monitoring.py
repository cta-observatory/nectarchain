import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
from astropy import time as astropytime
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam import LightNectarCAMEventSource as EventSource
from ctapipe_io_nectarcam.constants import (
    HIGH_GAIN,
    N_MODULES,
    N_PIXELS,
    N_PIXELS_MODULE,
)
from traitlets.config import Config

from nectarchain.dqm.camera_monitoring import CameraMonitoring


class TestCameraMonitoring:
    """Comprehensive tests for CameraMonitoring processor."""

    @classmethod
    def setup_class(cls):
        """Setup test fixtures once for all tests."""
        cls.paths = [
            get_dataset_path("NectarCAM.Run5288.0001.fits.fz"),
            get_dataset_path("NectarCAM.Run4717.0000.fits.fz"),
            get_dataset_path("NectarCAM.Run3938.30events.fits.fz"),
        ]

        cls.config = Config(
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

    def test_define_for_run(self):
        """Test define_for_run method extracts correct geometry."""
        reader1 = EventSource(input_url=self.paths[2], config=self.config, max_events=1)

        Pix, Samp = CameraMonitoring(HIGH_GAIN, r0=True).define_for_run(reader1)

        # Verify geometry is extracted correctly
        assert Pix == 1855  # NectarCAM 1855 pixels
        assert Samp == 60  # 60 samples
        assert Pix + Samp == 1915

        tel_id = reader1.subarray.tel_ids[0]

        sql_file_date = None
        for evt in reader1:
            run_start1 = evt.nectarcam.tel[tel_id].svc.date
            sql_file_date = astropytime.Time(run_start1, format="unix").iso.split(" ")[
                0
            ]

        assert sql_file_date == "2023-01-23"

    def test_configure_for_run(self):
        """Test configure_for_run method sets up processor correctly."""
        reader1 = EventSource(input_url=self.paths[0], config=self.config, max_events=1)
        tel_id = reader1.subarray.tel_ids[0]
        subarray = reader1.subarray
        run_start1 = next(iter(reader1)).nectarcam.tel[tel_id].svc.date

        processor = CameraMonitoring(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader1)

        processor.configure_for_run(self.paths[0], Pix, Samp, reader1)

        assert processor.Pix == 1855
        assert processor.Samp == 60
        assert processor.tel_id == tel_id
        assert processor.camera is not None
        assert processor.subarray == subarray
        assert processor.run_start1 == run_start1

    def test_configure_for_run_sqlite_filename_construction(self):
        """Test the try clause in configure_for_run,
        to see if the SQLite filename was correctly constructed.
        """
        reader = EventSource(input_url=self.paths[2], config=self.config, max_events=1)
        tel_id = reader.subarray.tel_ids[0]

        # Get expected values
        run_start1 = next(iter(reader)).nectarcam.tel[tel_id].svc.date
        expected_date = astropytime.Time(run_start1, format="unix").iso.split(" ")[0]
        expected_dir = os.path.split(self.paths[2])[0]
        expected_filename = (
            f"{expected_dir}/nectarcam_monitoring_db_{expected_date}.sqlite"
        )

        processor = CameraMonitoring(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader)

        # Mock sqlite3 to capture the filename without requiring a real DB
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            cursor_mock = MagicMock()
            mock_conn.cursor.return_value.__enter__.return_value = cursor_mock
            cursor_mock.fetchall.return_value = []  # Empty result is fine for this test

            processor.configure_for_run(self.paths[2], Pix, Samp, reader)

            # Verify the filename passed to sqlite3.connect
            mock_connect.assert_called_once_with(expected_filename)

    def test_configure_for_run_error_handling(self):
        """Test configure_for_run SQLite error handling."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=1)
        processor = CameraMonitoring(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader)

        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Test error")
            processor.configure_for_run(self.paths[0], Pix, Samp, reader)
        assert processor.DrawerTemp is None

    def test_process_event(self):
        """Test process_event method processes events correctly."""
        reader1 = EventSource(input_url=self.paths[0], config=self.config, max_events=1)

        processor = CameraMonitoring(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader1)

        # Mock SQLite to avoid file-not-found errors during configure
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            cursor_mock = MagicMock()
            mock_conn.cursor.return_value.__enter__.return_value = cursor_mock
            cursor_mock.fetchall.return_value = []  # Empty result is fine

            processor.configure_for_run(self.paths[0], Pix, Samp, reader1)

        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=5)
        evt_count = 0
        for evt in reader:
            processor.process_event(evt, noped=False)
            evt_count += 1

        assert len(processor.event_times) == 5
        assert len(processor.event_id) == 5
        assert evt_count == 5

    def test_finish_run(self):
        """Test finish_run method computes statistics correctly."""
        for path in self.paths:
            # Setup and process events first
            reader = EventSource(input_url=path, config=self.config, max_events=10)
            processor = CameraMonitoring(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)

            # Mock SQLite to avoid file-not-found errors
            with patch("sqlite3.connect") as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value.__enter__.return_value = mock_conn
                cursor_mock = MagicMock()
                mock_conn.cursor.return_value.__enter__.return_value = cursor_mock
                cursor_mock.fetchall.return_value = []
                processor.configure_for_run(path, Pix, Samp, reader)

            # Process events to populate event_times/event_id
            for evt in reader:
                processor.process_event(evt, noped=False)

            # Compute expected run window
            run_start = processor.event_times[np.argmin(processor.event_id)] - 100
            run_end = np.max(processor.event_times) + 100

            # Set DrawerTemp manually with timestamps INSIDE the run window
            mid_time = (run_start + run_end) / 2
            time1 = astropytime.Time(mid_time - 10, format="unix").iso
            time2 = astropytime.Time(mid_time + 10, format="unix").iso

            processor.DrawerTemp = np.array(
                [
                    (
                        1,
                        100,
                        0,
                        time1,
                        20.0,
                        20.5,
                    ),  # (id, ?, drawer_num, timestamp, temp1, temp2)
                    (2, 101, 0, time2, 20.1, 20.6),
                    (3, 102, 1, time1, 21.0, 21.5),
                    (4, 103, 1, time2, 21.1, 21.6),
                ],
                dtype=object,
            )

            processor.finish_run()

            assert isinstance(processor.DrawerTemp1_mean, np.ndarray)
            assert processor.DrawerTemp1_mean.shape[0] == 2 * N_PIXELS_MODULE
            assert isinstance(processor.DrawerTemp2_mean, np.ndarray)
            assert processor.DrawerTemp2_mean.shape[0] == 2 * N_PIXELS_MODULE
            assert isinstance(processor.DrawerTemp1_std, np.ndarray)
            assert processor.DrawerTemp1_std.shape[0] == 2 * N_PIXELS_MODULE
            assert isinstance(processor.DrawerTemp2_std, np.ndarray)
            assert processor.DrawerTemp2_std.shape[0] == 2 * N_PIXELS_MODULE
            assert isinstance(processor.DrawerTemp1_trend, np.ndarray)
            assert isinstance(processor.DrawerTemp2_trend, np.ndarray)
            assert np.array_equal(
                processor.DrawerTemp_mean,
                (processor.DrawerTemp1_mean + processor.DrawerTemp2_mean) / 2,
            )
            assert np.array_equal(
                processor.DrawerTemp_std,
                (processor.DrawerTemp1_std + processor.DrawerTemp2_std) / 2,
            )
            assert np.array_equal(
                processor.DrawerTemp_trend,
                (processor.DrawerTemp1_trend + processor.DrawerTemp2_trend) / 2,
            )

    def test_finish_run_error_handling(self):
        """Test finish_run handles exceptions gracefully."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=5)
        processor = CameraMonitoring(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader)

        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = MagicMock()
            processor.configure_for_run(self.paths[0], Pix, Samp, reader)

        for evt in reader:
            processor.process_event(evt, noped=False)

        processor.DrawerTemp = None  # Trigger exception in finish_run
        processor.finish_run()
        assert processor.DrawerTemp1_mean == []

    def test_finish_run_empty_drawer(self):
        """Test finish_run handles drawers with no data."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=5)
        processor = CameraMonitoring(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader)

        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = MagicMock()
            processor.configure_for_run(self.paths[0], Pix, Samp, reader)

        for evt in reader:
            processor.process_event(evt, noped=False)

        run_start = processor.event_times[np.argmin(processor.event_id)] - 100
        run_end = np.max(processor.event_times) + 100
        mid_time = (run_start + run_end) / 2
        time1 = astropytime.Time(mid_time - 10, format="unix").iso

        # CRITICAL: Use non-consecutive drawer numbers to create a gap
        # Drawer 0 and 2 have data, drawer 1 is empty
        processor.DrawerTemp = np.array(
            [
                (1, 100, 0, time1, 20.0, 20.5),
                (2, 101, 0, time1, 20.1, 20.6),
                (3, 102, 2, time1, 22.0, 22.5),  # Skip drawer 1
            ],
            dtype=object,
        )

        processor.finish_run()
        assert isinstance(processor.DrawerTemp1_mean, np.ndarray)
        # Drawer 1's pixels (indices 7-13) should be NaN
        assert np.isnan(processor.DrawerTemp1_mean[7:14]).all()

    def test_get_results(self):
        """Test get_results method returns proper dictionary structure."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=10)
            processor = CameraMonitoring(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)

            # Mock SQLite to avoid file-not-found errors
            with patch("sqlite3.connect") as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value.__enter__.return_value = mock_conn
                cursor_mock = MagicMock()
                mock_conn.cursor.return_value.__enter__.return_value = cursor_mock
                cursor_mock.fetchall.return_value = []
                processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            # Compute expected run window
            run_start = processor.event_times[np.argmin(processor.event_id)] - 100
            run_end = np.max(processor.event_times) + 100

            # Set DrawerTemp manually with timestamps INSIDE the run window
            mid_time = (run_start + run_end) / 2
            time1 = astropytime.Time(mid_time - 10, format="unix").iso
            time2 = astropytime.Time(mid_time + 10, format="unix").iso

            processor.DrawerTemp = np.array(
                [
                    (
                        1,
                        100,
                        0,
                        time1,
                        20.0,
                        20.5,
                    ),  # (id, ?, drawer_num, timestamp, temp1, temp2)
                    (2, 101, 0, time2, 20.1, 20.6),
                    (3, 102, 1, time1, 21.0, 21.5),
                    (4, 103, 1, time2, 21.1, 21.6),
                ],
                dtype=object,
            )

            processor.finish_run()
            results = processor.get_results()

            assert isinstance(results, dict)
            assert "CAMERA-TEMPERATURE-AVERAGE" in results.keys()
            assert "CAMERA-TEMPERATURE-STD" in results.keys()
            assert "CAMERA-TEMPERATURE-TREND" in results.keys()

    def test_plot_results(self):
        """Test plot_results and helper methods."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=5)
        processor = CameraMonitoring(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader)

        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = MagicMock()
            processor.configure_for_run(self.paths[0], Pix, Samp, reader)

        for evt in reader:
            processor.process_event(evt, noped=False)

        processor.DrawerTemp1_mean = np.random.rand(N_PIXELS)
        processor.DrawerTemp2_mean = np.random.rand(N_PIXELS)
        processor.DrawerTemp1_std = np.random.rand(N_PIXELS)
        processor.DrawerTemp2_std = np.random.rand(N_PIXELS)
        processor.DrawerTemp_mean = np.random.rand(N_PIXELS)
        processor.DrawerTemp_std = np.random.rand(N_PIXELS)

        n_timepoints = 5
        processor.DrawerTemp1_trend = np.random.rand(N_MODULES, n_timepoints)
        processor.DrawerTemp2_trend = np.random.rand(N_MODULES, n_timepoints)
        processor.DrawerTemp_trend = np.random.rand(N_MODULES, n_timepoints)

        time_points = np.linspace(0, 1000, N_MODULES * n_timepoints)
        processor.DrawerTimes_run = time_points

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("matplotlib.pyplot.plot") as mock_plot:  # noqa
                with patch("matplotlib.pyplot.close") as mock_close:
                    fig_dict, name_dict = processor.plot_results("test", tmpdir)
                    # Verify camera display figures were created
                    assert len(fig_dict) == 9
                    assert len(name_dict) == 9
                    mock_close.assert_called()

    def test_plot_results_error_handling(self):
        """Test plot_results exception handling."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=5)
        processor = CameraMonitoring(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader)

        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = MagicMock()
            processor.configure_for_run(self.paths[0], Pix, Samp, reader)

        for evt in reader:
            processor.process_event(evt, noped=False)

        # Force CameraDisplay to fail (wrong shape)
        processor.DrawerTemp_mean = np.random.rand(10)  # 10 != 1855 pixels

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("matplotlib.pyplot.close") as mock_close:
                fig_dict, name_dict = processor.plot_results("test", tmpdir)
                # Exception is caught, so method returns empty dicts
                assert fig_dict == {}
                assert name_dict == {}
                mock_close.assert_not_called()  # Exception happens before plt.close()
