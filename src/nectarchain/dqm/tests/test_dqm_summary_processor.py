import os
import tempfile

import numpy as np
from astropy.table import Table
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam import LightNectarCAMEventSource as EventSource
from traitlets.config import Config

from nectarchain.dqm.dqm_summary_processor import DQMSummary


class TestDQMSummary:
    """Comprehensive tests for the DQMSummary utility."""

    @classmethod
    def setup_class(cls):
        """Setup test fixtures once for all tests."""
        cls.paths = [
            get_dataset_path("NectarCAM.Run5288.0001.fits.fz"),
            get_dataset_path("NectarCAM.Run4717.0000.fits.fz"),
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

    def test_init_default(self):
        """Test default initialization."""
        processor = DQMSummary()
        assert processor.r0 is False
        assert processor.Samp is None
        assert processor.Pix is None
        assert processor.tel_id is None

    def test_init_with_r0(self):
        """Test initialization with r0=True."""
        processor = DQMSummary(r0=True)
        assert processor.r0 is True
        assert processor.Samp is None
        assert processor.Pix is None
        assert processor.tel_id is None

    def test_define_for_run(self):
        """Test define_for_run method extracts correct geometry."""
        reader1 = EventSource(input_url=self.paths[0], config=self.config, max_events=1)

        tel_id = reader1.subarray.tel_ids[0]
        processor = DQMSummary(r0=True)
        Pix, Samp = processor.define_for_run(reader1)

        assert tel_id == processor.tel_id
        assert Pix == 1855  # NectarCAM 1855 pixels
        assert Samp == 60  # 60 samples
        assert Pix + Samp == 1915

    def test_define_for_run_r0_false_skipped(self):
        """Test define_for_run with r0=False is skipped when R1 not available."""
        for path in self.paths:
            reader1 = EventSource(input_url=path, config=self.config, max_events=1)

            processor = DQMSummary(r0=False)
            try:
                Pix, Samp = processor.define_for_run(reader1)
                assert processor.tel_id is not None
                assert Pix == 1855
                assert Samp == 60
            except AttributeError:
                pass

    def test_configure_for_run(self):
        """Test configure_for_run method runs without errors."""
        processor = DQMSummary(r0=True)
        processor.configure_for_run()
        assert True

    def test_process_event(self):
        """Test process_event method runs without errors."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=1)
        processor = DQMSummary(r0=True)
        processor.define_for_run(reader)

        for evt in reader:
            processor.process_event(evt, noped=False)
        assert True

    def test_process_event_with_noped(self):
        """Test process_event with noped=True."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=1)
        processor = DQMSummary(r0=True)
        processor.define_for_run(reader)

        for evt in reader:
            processor.process_event(evt, noped=True)
        assert True

    def test_finish_run(self):
        """Test finish_run method runs without errors."""
        processor = DQMSummary(r0=True)
        M = np.random.rand(1855, 60)
        M_ped = np.random.rand(1855, 60)
        counter_evt = 5
        counter_ped = 3
        processor.finish_run(M, M_ped, counter_evt, counter_ped)
        assert True

    def test_get_results(self):
        """Test get_results method returns None (base class stub)."""
        processor = DQMSummary(r0=True)
        results = processor.get_results()
        assert results is None

    def test_plot_results(self):
        """Test plot_results method runs without errors."""
        processor = DQMSummary(r0=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            M = np.random.rand(1855, 60)
            M_ped = np.random.rand(1855, 60)
            Mean_M_overPix = np.mean(M, axis=1)
            Mean_M_ped_overPix = np.mean(M_ped, axis=1)
            processor.plot_results(
                "test_run", tmpdir, 0, M, M_ped, Mean_M_overPix, Mean_M_ped_overPix
            )
        assert True

    def test_create_hdu_array_1d(self):
        """Test _create_hdu with 1D array."""
        arr = np.array([1.0, 2.0, 3.0])
        hdu = DQMSummary._create_hdu("TEST", arr)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_array_2d(self):
        """Test _create_hdu with 2D array."""
        arr = np.random.rand(1855, 60)
        hdu = DQMSummary._create_hdu("TEST", arr)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_array_0d(self):
        """Test _create_hdu with 0D array (scalar)."""
        scalar = np.array(42.0)
        hdu = DQMSummary._create_hdu("TEST", scalar)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_dict(self):
        """Test _create_hdu with dictionary."""
        data_dict = {"key1": [1, 2, 3], "key2": [4, 5, 6]}
        hdu = DQMSummary._create_hdu("TEST", data_dict)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_table(self):
        """Test _create_hdu with astropy Table."""
        table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        hdu = DQMSummary._create_hdu("TEST", table)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_float(self):
        """Test _create_hdu with float."""
        value = 3.14
        hdu = DQMSummary._create_hdu("TEST", value)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_integer_array(self):
        """Test _create_hdu with integer array."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        hdu = DQMSummary._create_hdu("TEST", arr)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_object_array(self):
        """Test _create_hdu with object array."""
        arr = np.array([1, 2, 3], dtype=np.object_)
        hdu = DQMSummary._create_hdu("TEST", arr)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_write_all_results(self):
        """Test write_all_results method creates a FITS file."""
        processor = DQMSummary(r0=True)
        DICT = {
            "section1": {
                "array_data": np.random.rand(10, 10),
                "scalar_data": 42.0,
            },
            "section2": {
                "table_data": Table({"col1": [1, 2, 3], "col2": [4, 5, 6]}),
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_results")
            processor.write_all_results(output_path, DICT)
            assert os.path.exists(output_path + "_Results.fits")

    def test_write_all_results_with_invalid_content(self):
        """Test write_all_results with content that causes TypeError."""
        processor = DQMSummary(r0=True)
        DICT = {
            "section1": {
                "array_data": np.random.rand(10, 10),
                "invalid_data": {"a": object()},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_results_invalid")
            processor.write_all_results(output_path, DICT)
            assert os.path.exists(output_path + "_Results.fits")

    def test_full_workflow(self):
        """Test complete workflow with base DQMSummary class."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=3)
        processor = DQMSummary(r0=True)

        Pix, Samp = processor.define_for_run(reader)
        processor.configure_for_run()

        for evt in reader:
            processor.process_event(evt, noped=False)

        M = np.random.rand(1855, 60)
        M_ped = np.random.rand(1855, 60)
        processor.finish_run(M, M_ped, 2, 1)
        results = processor.get_results()

        assert Pix == 1855
        assert Samp == 60
        assert processor.tel_id is not None
        assert results is None
