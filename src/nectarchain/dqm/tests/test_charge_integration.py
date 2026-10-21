import tempfile

from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam import LightNectarCAMEventSource as EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN, LOW_GAIN
from traitlets.config import Config

from nectarchain.dqm.charge_integration import ChargeIntegrationHighLowGain


class TestChargeIntegrationHighLowGain:
    """Comprehensive tests for ChargeIntegrationHighLowGain processor."""

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

    def test_define_for_run(self):
        """Test define_for_run method extracts correct geometry."""
        reader1 = EventSource(input_url=self.paths[0], config=self.config, max_events=1)

        Pix, Samp = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True).define_for_run(
            reader1
        )

        # Verify geometry is extracted correctly
        assert Pix == 1855  # NectarCAM 1855 pixels
        assert Samp == 60  # 60 samples
        assert Pix + Samp == 1915

    def test_configure_for_run(self):
        """Test configure_for_run method sets up processor correctly."""
        reader1 = EventSource(input_url=self.paths[0], config=self.config, max_events=1)
        tel_id = reader1.subarray.tel_ids[0]

        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader1)

        processor.configure_for_run(self.paths[0], Pix, Samp, reader1)

        assert processor.Pix == 1855
        assert processor.Samp == 60
        assert processor.tel_id == tel_id
        assert processor.counter_evt == 0
        assert processor.counter_ped == 0
        assert processor.camera is not None
        assert processor.integrator is not None

    def test_process_event(self):
        """Test process_event method processes events correctly."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=5)

        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader)
        processor.configure_for_run(self.paths[0], Pix, Samp, reader)

        evt_count = 0
        for evt in reader:
            processor.process_event(evt, noped=False)
            evt_count += 1

        assert processor.counter_evt > 0 or processor.counter_ped > 0
        assert evt_count == 5
        assert len(processor.image_all) + len(processor.image_ped) == 5

    def test_process_event_with_noped(self):
        """Test process_event with pedestal subtraction enabled."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=2)

        processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader)
        processor.configure_for_run(self.paths[0], Pix, Samp, reader)

        for evt in reader:
            processor.process_event(evt, noped=True)

        assert processor.counter_evt > 0 or processor.counter_ped > 0

    def test_finish_run(self):
        """Test finish_run method computes statistics correctly."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=3)

            processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()

            if processor.counter_evt > 0:
                assert processor.image_all_stats is not None
                assert "average" in processor.image_all_stats
                assert "median" in processor.image_all_stats
                assert "std" in processor.image_all_stats
                assert "rms" in processor.image_all_stats
                assert processor.image_all_stats["average"].shape == (1855,)

            if processor.counter_ped > 0:
                assert processor.image_ped_stats is not None
                assert "average" in processor.image_ped_stats

    def test_get_results(self):
        """Test get_results method returns proper dictionary structure."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=3)

            processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()
            results = processor.get_results()

            assert isinstance(results, dict)
            if processor.counter_evt > 0:
                assert any("CHARGE-INTEGRATION-IMAGE-ALL" in k for k in results.keys())
                assert any("PED-INTEGRATION-IMAGE-ALL" in k for k in results.keys())
                assert any("PEAKPOS-ALL" in k for k in results.keys())
            if processor.counter_ped > 0:
                assert any("CHARGE-INTEGRATION-PED-ALL" in k for k in results.keys())
                assert any("PED-INTEGRATION-PED-ALL" in k for k in results.keys())
                assert any("PEAKPOS-PED-ALL" in k for k in results.keys())

    def test_plot_results(self):
        """Test plot_results method generates figures without errors."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=3)

            processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            # Process some events
            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()

            # Create temporary directory for plots
            with tempfile.TemporaryDirectory() as tmpdir:
                fig_dict, name_dict = processor.plot_results("test_run", tmpdir)

                # Verify figures were created (as matplotlib objects)
                assert isinstance(fig_dict, dict)
                assert isinstance(name_dict, dict)
                assert len(fig_dict) > 0

                # Verify all entries in fig_dict are matplotlib Figure objects
                from matplotlib.figure import Figure

                for fig in fig_dict.values():
                    assert isinstance(fig, Figure)

    def test_full_workflow_high_gain(self):
        """Test complete workflow for HIGH_GAIN processor."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=5)
            processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)

            # Full workflow
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()
            results = processor.get_results()

            assert isinstance(results, dict)
            assert processor.counter_evt >= 0
            assert processor.counter_ped >= 0
            assert processor.counter_evt + processor.counter_ped == 5

    def test_full_workflow_low_gain(self):
        """Test complete workflow for LOW_GAIN processor."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=5)
            processor = ChargeIntegrationHighLowGain(LOW_GAIN, r0=True)

            # Full workflow
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()
            results = processor.get_results()

            assert isinstance(results, dict)
            assert processor.counter_evt >= 0
            assert processor.counter_ped >= 0
            assert processor.counter_evt + processor.counter_ped == 5

    def test_with_custom_extractor(self):
        """Test processor with custom charge extractor settings."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=3)

            processor = ChargeIntegrationHighLowGain(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)

            # Configure with custom extractor
            charges_kwargs = {
                "method": "GlobalPeakWindowSum",
                "extractor_kwargs": {"window_shift": 4, "window_width": 16},
            }
            processor.configure_for_run(path, Pix, Samp, reader, **charges_kwargs)

            # Verify custom configuration
            assert processor.integrator is not None

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()
            results = processor.get_results()
            assert isinstance(results, dict)
