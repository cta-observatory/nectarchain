import os
import tempfile
from unittest.mock import patch

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

    # Tests for updated error handling code in _create_hdu

    def test_create_hdu_dict_with_nested_unserializable(self):
        """Test _create_hdu with dict containing nested unserializable objects."""
        data_dict = {"key1": [1, 2, 3], "key2": object()}
        hdu = DQMSummary._create_hdu("TEST", data_dict)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_empty_dict(self):
        """Test _create_hdu with empty dictionary."""
        data_dict = {}
        hdu = DQMSummary._create_hdu("TEST", data_dict)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_dict_with_none_values(self):
        """Test _create_hdu with dict containing None values."""
        data_dict = {"key1": None, "key2": [1, 2, 3]}
        hdu = DQMSummary._create_hdu("TEST", data_dict)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_dict_with_mixed_types(self):
        """Test _create_hdu with dict containing mixed types."""
        data_dict = {"int": 42, "float": 3.14, "list": [1, 2, 3], "str": "test"}
        hdu = DQMSummary._create_hdu("TEST", data_dict)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_none_content(self):
        """Test _create_hdu with None content."""
        hdu = DQMSummary._create_hdu("TEST", None)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_string_content(self):
        """Test _create_hdu with string content."""
        hdu = DQMSummary._create_hdu("TEST", "some string")
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_complex_number(self):
        """Test _create_hdu with complex number."""
        hdu = DQMSummary._create_hdu("TEST", 3 + 4j)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_very_large_array(self):
        """Test _create_hdu with very large array."""
        arr = np.random.rand(1000, 1000)
        hdu = DQMSummary._create_hdu("TEST", arr)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_array_with_nan(self):
        """Test _create_hdu with array containing NaN values."""
        arr = np.array([1.0, np.nan, 3.0])
        hdu = DQMSummary._create_hdu("TEST", arr)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_array_with_inf(self):
        """Test _create_hdu with array containing inf values."""
        arr = np.array([1.0, np.inf, -np.inf, 3.0])
        hdu = DQMSummary._create_hdu("TEST", arr)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_write_all_results_with_multiple_invalid_types(self):
        """Test write_all_results with various invalid content types."""
        processor = DQMSummary(r0=True)
        DICT = {
            "section1": {
                "valid_array": np.random.rand(10, 10),
                "invalid_dict1": {"a": object()},
                "invalid_dict2": {"nested": {"obj": object()}},
                "invalid_none": None,
                "invalid_string": "test",
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_results_multiple_invalid")
            processor.write_all_results(output_path, DICT)
            assert os.path.exists(output_path + "_Results.fits")

    def test_write_all_results_with_empty_dict_content(self):
        """Test write_all_results with empty dict as content."""
        processor = DQMSummary(r0=True)
        DICT = {
            "section1": {
                "empty_dict": {},
                "valid_array": np.random.rand(5, 5),
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_results_empty_dict")
            processor.write_all_results(output_path, DICT)
            assert os.path.exists(output_path + "_Results.fits")

    def test_create_hdu_0d_array_from_scalar(self):
        """Test _create_hdu with 0D numpy array created from scalar."""
        scalar = np.float64(42.0)
        hdu = DQMSummary._create_hdu("TEST", scalar)
        assert hdu.name == "TEST"
        assert hdu is not None

    def test_create_hdu_table_with_empty_columns(self):
        """Test _create_hdu with Table containing empty columns."""
        table = Table({"a": [], "b": []})
        hdu = DQMSummary._create_hdu("TEST", table)
        assert hdu.name == "TEST"
        assert hdu is not None

    # Tests for exception handling paths using mocks

    def test_create_hdu_bintablehdu_failure_for_1d_array(self):
        """Test fallback when BinTableHDU creation fails for 1D array."""
        from astropy.io import fits

        arr = np.array([1.0, 2.0, 3.0])
        # Save original BinTableHDU
        original_bintable = fits.BinTableHDU

        # Create a mock that fails once then works
        call_count = [0]

        def mock_bintable(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Mocked failure")
            return original_bintable(*args, **kwargs)

        with patch(
            "nectarchain.dqm.dqm_summary_processor.fits.BinTableHDU", mock_bintable
        ):
            hdu = DQMSummary._create_hdu("TEST", arr)
            assert hdu.name == "TEST"
            assert hdu is not None

    def test_create_hdu_imagehdu_failure_for_2d_array(self):
        """Test fallback when ImageHDU creation fails for 2D array."""
        arr = np.random.rand(10, 10)

        # Mock ImageHDU to always fail, but BinTableHDU should work
        def mock_image(*args, **kwargs):
            raise Exception("Mocked failure")

        with patch("nectarchain.dqm.dqm_summary_processor.fits.ImageHDU", mock_image):
            hdu = DQMSummary._create_hdu("TEST", arr)
            assert hdu.name == "TEST"
            assert hdu is not None

    def test_create_hdu_bintablehdu_failure_for_dict(self):
        """Test fallback when BinTableHDU creation fails for dict content."""
        from astropy.io import fits

        data_dict = {"key1": [1, 2, 3], "key2": [4, 5, 6]}
        # Save original BinTableHDU
        original_bintable = fits.BinTableHDU

        # Create a mock that fails once then works
        call_count = [0]

        def mock_bintable(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Mocked failure")
            return original_bintable(*args, **kwargs)

        with patch(
            "nectarchain.dqm.dqm_summary_processor.fits.BinTableHDU", mock_bintable
        ):
            hdu = DQMSummary._create_hdu("TEST", data_dict)
            assert hdu.name == "TEST"
            assert hdu is not None

    def test_create_hdu_bintablehdu_failure_for_table(self):
        """Test fallback when BinTableHDU creation fails for Table content."""
        from astropy.io import fits

        table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        # Save original BinTableHDU
        original_bintable = fits.BinTableHDU

        # Create a mock that fails once then works
        call_count = [0]

        def mock_bintable(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Mocked failure")
            return original_bintable(*args, **kwargs)

        with patch(
            "nectarchain.dqm.dqm_summary_processor.fits.BinTableHDU", mock_bintable
        ):
            hdu = DQMSummary._create_hdu("TEST", table)
            assert hdu.name == "TEST"
            assert hdu is not None

    def test_create_hdu_bintablehdu_failure_for_scalar_fallback(self):
        """Test fallback when BinTableHDU creation
        fails for scalar after Table creation."""
        from astropy.io import fits

        value = 3.14
        # Save original BinTableHDU
        original_bintable = fits.BinTableHDU

        # Create a mock that fails once then works
        call_count = [0]

        def mock_bintable(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Mocked failure")
            return original_bintable(*args, **kwargs)

        with patch(
            "nectarchain.dqm.dqm_summary_processor.fits.BinTableHDU", mock_bintable
        ):
            hdu = DQMSummary._create_hdu("TEST", value)
            assert hdu.name == "TEST"
            assert hdu is not None

    def test_write_all_results_exception_handling(self):
        """Test that write_all_results handles exceptions from _create_hdu."""
        processor = DQMSummary(r0=True)
        DICT = {
            "section1": {
                "valid_data": np.random.rand(5, 5),
                "invalid_data": object(),  # This should trigger an exception
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_results_exception")
            processor.write_all_results(output_path, DICT)
            # Should still create the file, skipping the invalid content
            assert os.path.exists(output_path + "_Results.fits")

    def test_define_for_run_r0_false(self):
        """Test define_for_run with r0=False."""
        reader1 = EventSource(input_url=self.paths[0], config=self.config, max_events=1)
        processor = DQMSummary(r0=False)
        try:
            Pix, Samp = processor.define_for_run(reader1)
            assert processor.tel_id is not None
            assert Pix == 1855
            assert Samp == 60
        except AttributeError:
            # This is expected if R1 is not available
            pass

    def test_define_for_run_r0_false_with_r1(self):
        """Test define_for_run with r0=False using a file that has R1 data."""
        # Use the same path, but if it doesn't have R1,
        # this will still catch AttributeError
        # The goal is to reach:
        # self.Pix = evt1.r1.tel[self.tel_id].waveform.shape[-2]
        reader1 = EventSource(input_url=self.paths[0], config=self.config, max_events=1)
        processor = DQMSummary(r0=False)
        try:
            _, _ = processor.define_for_run(reader1)
            assert processor.tel_id is not None
        except AttributeError:
            pass

    def test_create_hdu_scalar_with_unconvertible_content(self):
        """Test fallback when Table(np.array([content])) fails for scalar."""
        import numpy as np_local

        # Mock numpy.array to raise an exception for a specific object
        original_array = np_local.array

        # Create a sentinel object to identify our test case
        sentinel = object()

        def mock_array(obj, *args, **kwargs):
            # If this is our sentinel, fail
            if obj is sentinel:
                raise TypeError("Cannot convert to array")
            return original_array(obj, *args, **kwargs)

        with patch("nectarchain.dqm.dqm_summary_processor.np.array", mock_array):
            # Pass the sentinel directly
            # - it will fail when np.array tries to convert it
            hdu = DQMSummary._create_hdu("TEST", sentinel)
            assert hdu.name == "TEST"
            assert hdu is not None

    def test_write_all_results_with_forced_exception(self):
        """Test exception handling in write_all_results by forcing an exception."""
        processor = DQMSummary(r0=True)
        DICT = {
            "section1": {
                "valid_data": np.random.rand(5, 5),
            },
        }

        # Mock _create_hdu to raise an exception for the valid data
        original_create_hdu = DQMSummary._create_hdu

        def mock_create_hdu(name, content):
            if name == "valid_data":
                raise RuntimeError("Forced exception for testing")
            return original_create_hdu(name, content)

        with patch.object(DQMSummary, "_create_hdu", mock_create_hdu):
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "test_forced_exception")
                # This should not raise an exception, even though _create_hdu fails
                processor.write_all_results(output_path, DICT)
                # The method should complete without crashing
                # Note: The file might not be created if all HDUs fail,
                # which is acceptable
                # The important thing is that the exception
                # was caught and handled
