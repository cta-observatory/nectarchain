import tempfile

import numpy as np
from ctapipe.containers import EventType
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam import LightNectarCAMEventSource as EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN, LOW_GAIN
from traitlets.config import Config

from nectarchain.dqm.mean_camera_display import MeanCameraDisplayHighLowGain


class TestMeanCameraDisplayHighLowGain:
    """Comprehensive tests for MeanCameraDisplayHighLowGain processor."""

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

        Pix, Samp = MeanCameraDisplayHighLowGain(HIGH_GAIN, r0=True).define_for_run(
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

        processor = MeanCameraDisplayHighLowGain(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader1)

        processor.configure_for_run(self.paths[0], Pix, Samp, reader1)

        assert processor.Pix == 1855
        assert processor.Samp == 60
        assert processor.tel_id == tel_id
        assert processor.counter_evt == 0
        assert processor.counter_ped == 0
        assert processor.camera is not None
        assert isinstance(processor.camera_average, np.ndarray)
        assert isinstance(processor.camera_average_ped, np.ndarray)
        assert processor.pixel_bad is None
        assert processor.pixel_ids is None

    def test_process_event(self):
        """Test process_event method processes events correctly."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=10)
            tel_id = reader.subarray.tel_ids[0]

            processor = MeanCameraDisplayHighLowGain(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            evt_count = 0
            counter_evt = 0
            counter_ped = 0
            camera_average = np.zeros(Pix, dtype=np.float32)
            camera_average_ped = np.zeros(Pix, dtype=np.float32)

            for evt in reader:
                pixel_bad = evt.mon.tel[tel_id].pixel_status.hardware_failing_pixels
                pixel_ids = evt.nectarcam.tel[tel_id].svc.pixel_ids
                processor.process_event(evt, noped=False)

                assert np.array_equal(processor.pixel_bad, pixel_bad)

                if len(pixel_ids) < Pix:
                    padding = np.arange(0, Pix - len(pixel_ids), 1, dtype=int)
                    pixels = np.concatenate([padding, pixel_ids])
                else:
                    pixels = pixel_ids

                if processor.r0:
                    waveforms = evt.r0.tel[processor.tel_id].waveform[processor.k]
                else:
                    wf = evt.r1.tel[processor.tel_id].waveform
                    if wf.ndim == 3:
                        waveforms = wf[processor.k]
                    else:
                        waveforms = wf

                is_ped = evt.trigger.event_type == EventType.SKY_PEDESTAL
                is_phy = evt.trigger.event_type == EventType.SUBARRAY

                charge_per_pixel = waveforms.sum(axis=-1)

                charge_mapped = charge_per_pixel[pixels]

                if is_ped:
                    counter_ped += 1
                    camera_average_ped += charge_mapped
                elif is_phy:
                    counter_evt += 1
                    camera_average += charge_mapped
                else:
                    # TODO: add ids for other event types, e.g., dark pedestals
                    # TODO: this else is wrong, we should have a separate counter
                    # for other event types. It has to be implemented.
                    counter_evt += 1
                    camera_average += charge_mapped

                evt_count += 1

            assert processor.counter_evt > 0 or processor.counter_ped > 0
            assert processor.counter_evt == counter_evt
            assert processor.counter_ped == counter_ped
            assert evt_count == 10
            assert np.array_equal(processor.camera_average, camera_average)
            assert np.array_equal(processor.camera_average_ped, camera_average_ped)

    def test_finish_run(self):
        """Test finish_run method computes statistics correctly."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=3)

            processor = MeanCameraDisplayHighLowGain(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()

            if processor.counter_evt > 0:
                assert processor.camera_average_over_events is not None
                assert processor.camera_average_over_events_over_samp is not None
                assert np.array_equal(
                    processor.camera_average_over_events_over_samp,
                    processor.camera_average_over_events / Samp,
                )

            if processor.counter_ped > 0:
                assert processor.camera_average_ped_over_events is not None
                assert processor.camera_average_ped_over_events_over_samp is not None
                assert np.array_equal(
                    processor.camera_average_ped_over_events_over_samp,
                    processor.camera_average_ped_over_events / Samp,
                )

    def test_get_results(self):
        """Test get_results method returns proper dictionary structure."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=5)

            processor = MeanCameraDisplayHighLowGain(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()
            results = processor.get_results()

            assert isinstance(results, dict)
            if processor.counter_evt > 0:
                assert any(
                    "CAMERA-AVERAGE-PHY-OverEVENTS-OverSamp-HIGH-GAIN" in k
                    for k in results.keys()
                )
            if processor.counter_ped > 0:
                assert any(
                    "CAMERA-AVERAGE-PED-OverEVENTS-OverSamp-HIGH-GAIN" in k
                    for k in results.keys()
                )

            processor = MeanCameraDisplayHighLowGain(LOW_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()
            results = processor.get_results()

            assert isinstance(results, dict)
            if processor.counter_evt > 0:
                assert any(
                    "CAMERA-AVERAGE-PHY-OverEVENTS-OverSamp-LOW-GAIN" in k
                    for k in results.keys()
                )
            if processor.counter_ped > 0:
                assert any(
                    "CAMERA-AVERAGE-PED-OverEVENTS-OverSamp-LOW-GAIN" in k
                    for k in results.keys()
                )

    def test_get_results_nan_handling(self):
        """Test that NaN/Inf values are properly handled in get_results."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=10)

            for gaink in [HIGH_GAIN, LOW_GAIN]:
                processor = MeanCameraDisplayHighLowGain(gaink, r0=True)
                Pix, Samp = processor.define_for_run(reader)
                processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()

            # Inject NaN and Inf values to test the handling
            if processor.counter_evt > 0:
                processor.camera_average_over_events_over_samp[0] = np.nan
                processor.camera_average_over_events_over_samp[1] = np.inf
                processor.camera_average_over_events_over_samp[2] = -np.inf
            if processor.counter_ped > 0:
                processor.camera_average_ped_over_events_over_samp[0] = np.nan
                processor.camera_average_ped_over_events_over_samp[1] = np.inf

            _ = processor.get_results()

            # Verify NaN/Inf were replaced with 0
            if processor.counter_evt > 0:
                assert processor.camera_average_over_events_over_samp[0] == 0
                assert processor.camera_average_over_events_over_samp[1] == 0
                assert processor.camera_average_over_events_over_samp[2] == 0
            if processor.counter_ped > 0:
                assert processor.camera_average_ped_over_events_over_samp[0] == 0
                assert processor.camera_average_ped_over_events_over_samp[1] == 0

    def test_plot_results(self):
        """Test plot_results method generates figures without errors."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=3)

            for gaink in [HIGH_GAIN, LOW_GAIN]:
                processor = MeanCameraDisplayHighLowGain(gaink, r0=True)
                Pix, Samp = processor.define_for_run(reader)
                processor.configure_for_run(path, Pix, Samp, reader)

                for evt in reader:
                    processor.process_event(evt, noped=False)

                processor.finish_run()

                with tempfile.TemporaryDirectory() as tmpdir:
                    fig_dict, name_dict = processor.plot_results("test_run", tmpdir)

                    assert isinstance(fig_dict, dict)
                    assert isinstance(name_dict, dict)
                    # Should have figures if we processed events
                    if processor.counter_evt > 0 or processor.counter_ped > 0:
                        assert len(fig_dict) > 0

                    from matplotlib.figure import Figure

                    for fig in fig_dict.values():
                        assert isinstance(fig, Figure)
