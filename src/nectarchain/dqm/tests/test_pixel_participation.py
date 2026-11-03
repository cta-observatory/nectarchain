import tempfile

import numpy as np
from ctapipe.containers import EventType
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam import LightNectarCAMEventSource as EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN, LOW_GAIN
from traitlets.config import Config

from nectarchain.dqm.pixel_participation import PixelParticipationHighLowGain


class TestPixelParticipationHighLowGain:
    """Comprehensive tests for PixelParticipationHighLowGain processor."""

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

        Pix, Samp = PixelParticipationHighLowGain(HIGH_GAIN, r0=True).define_for_run(
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

        processor = PixelParticipationHighLowGain(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader1)

        BadPixels_ped = np.zeros(Pix, dtype=np.int64)
        BadPixels = np.zeros(Pix, dtype=np.int64)

        processor.configure_for_run(self.paths[0], Pix, Samp, reader1)

        assert processor.Pix == 1855
        assert processor.Samp == 60
        assert processor.tel_id == tel_id
        assert processor.counter_evt == 0
        assert processor.counter_ped == 0
        assert processor.camera is not None
        assert np.array_equal(processor.BadPixels_ped, BadPixels_ped)
        assert np.array_equal(processor.BadPixels, BadPixels)

    def test_process_event(self):
        """Test process_event method processes events correctly."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=5)

        tel_id = reader.subarray.tel_ids[0]

        processor = PixelParticipationHighLowGain(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader)
        BadPixels_ped = np.zeros(Pix, dtype=np.int64)
        BadPixels = np.zeros(Pix, dtype=np.int64)

        processor.configure_for_run(self.paths[0], Pix, Samp, reader)

        evt_count = 0
        for evt in reader:
            processor.process_event(evt, noped=False)
            pixelBAD = evt.mon.tel[tel_id].pixel_status.hardware_failing_pixels[
                HIGH_GAIN
            ]
            pixels = evt.nectarcam.tel[tel_id].svc.pixel_ids
            bad_pixels = np.zeros(Pix, dtype=np.int64)
            np.put(bad_pixels, pixels, pixelBAD[pixels])

            if evt.trigger.event_type == EventType.SKY_PEDESTAL:
                # count sky peds, event id 2
                BadPixels_ped += bad_pixels
            elif evt.trigger.event_type == EventType.SUBARRAY:
                # count standard physics stereo events, event id 32
                BadPixels += bad_pixels
            # TODO: add ids for other event types, e.g., dark pedestals
            # TODO: this else is wrong, we should have a separate counter
            # for other event types, e.g., dark pedestals. It has to be implemented.
            else:
                BadPixels += bad_pixels

            evt_count += 1

        assert processor.counter_evt > 0 or processor.counter_ped > 0
        assert np.array_equal(processor.BadPixels_ped, BadPixels_ped)
        assert np.array_equal(processor.BadPixels, BadPixels)
        assert evt_count == 5

    def test_process_event_with_noped(self):
        """Test process_event with pedestal subtraction enabled."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=2)

        processor = PixelParticipationHighLowGain(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader)
        processor.configure_for_run(self.paths[0], Pix, Samp, reader)

        for evt in reader:
            processor.process_event(evt, noped=True)

        assert processor.counter_evt > 0 or processor.counter_ped > 0

    def test_finish_run(self):
        """Test finish_run method computes statistics correctly."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=3)

            processor = PixelParticipationHighLowGain(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()

    def test_get_results(self):
        """Test get_results method returns proper dictionary structure."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=3)

            processor = PixelParticipationHighLowGain(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()
            results = processor.get_results()

            assert isinstance(results, dict)
            if processor.counter_evt > 0:
                assert any(
                    "CAMERA-BadPix-PHY-OverEVENTS-HIGH-GAIN" in k
                    for k in results.keys()
                )
            if processor.counter_ped > 0:
                assert any(
                    "CAMERA-BadPix-PED-PHY-OverEVENTS-HIGH-GAIN" in k
                    for k in results.keys()
                )

            processor = PixelParticipationHighLowGain(LOW_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()
            results = processor.get_results()

            assert isinstance(results, dict)
            if processor.counter_evt > 0:
                assert any(
                    "CAMERA-BadPix-PHY-OverEVENTS-LOW-GAIN" in k for k in results.keys()
                )
            if processor.counter_ped > 0:
                assert any(
                    "CAMERA-BadPix-PED-PHY-OverEVENTS-LOW-GAIN" in k
                    for k in results.keys()
                )

    def test_plot_results(self):
        """Test plot_results method creates figures correctly."""
        for path in self.paths:
            for gaink in [HIGH_GAIN, LOW_GAIN]:
                reader = EventSource(input_url=path, config=self.config, max_events=5)
                processor = PixelParticipationHighLowGain(gaink, r0=True)
                Pix, Samp = processor.define_for_run(reader)
                processor.configure_for_run(path, Pix, Samp, reader)

                for evt in reader:
                    processor.process_event(evt, noped=False)

                with tempfile.TemporaryDirectory() as tmpdir:
                    fig_dict, name_dict = processor.plot_results("test_run", tmpdir)

                    assert isinstance(fig_dict, dict)
                    assert isinstance(name_dict, dict)

                    gain_display = "High" if gaink == HIGH_GAIN else "Low"

                    if processor.counter_evt > 0:
                        key = f"CAMERA-BADPIX-PHY-DISPLAY-{gain_display}-GAIN"
                        assert key in fig_dict
                        assert key in name_dict
                        assert fig_dict[key] is not None
                        expected_name = f"test_run_Camera_BPX_{gain_display}Gain.png"
                        assert expected_name in name_dict[key]

                    if processor.counter_ped > 0:
                        key = f"CAMERA-BADPIX-PED-DISPLAY-{gain_display}-GAIN"
                        assert key in fig_dict
                        assert key in name_dict
                        assert fig_dict[key] is not None
                        expected_name = f"test_run_Pedestal_BPX_{gain_display}Gain.png"
                        assert expected_name in name_dict[key]

    def test_create_badpixels_plot(self):
        """Test _create_badpixels_plot method directly."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=1)
        for gaink in [HIGH_GAIN, LOW_GAIN]:
            processor = PixelParticipationHighLowGain(gaink, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(self.paths[0], Pix, Samp, reader)

            entity = np.zeros(Pix, dtype=np.int64)
            entity[:5] = 1

            gain_display = "High" if gaink == HIGH_GAIN else "Low"
            title = f"Test Camera BPX {gain_display} gain"
            fig = processor._create_badpixels_plot(entity, title)

            assert fig is not None
