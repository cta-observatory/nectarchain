import tempfile

import numpy as np
from ctapipe.containers import EventType
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam import LightNectarCAMEventSource as EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN, LOW_GAIN
from traitlets.config import Config

from nectarchain.dqm.waveforms import WaveFormsHighLowGain


class TestWaveForms:
    """Comprehensive tests for WaveFormsHighLowGain processor."""

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

        Pix, Samp = WaveFormsHighLowGain(HIGH_GAIN, r0=True).define_for_run(reader1)

        # Verify geometry is extracted correctly
        assert Pix == 1855  # NectarCAM 1855 pixels
        assert Samp == 60  # 60 samples
        assert Pix + Samp == 1915

    def test_configure_for_run(self):
        """Test configure_for_run method sets up processor correctly."""
        reader1 = EventSource(input_url=self.paths[0], config=self.config, max_events=1)
        tel_id = reader1.subarray.tel_ids[0]

        processor = WaveFormsHighLowGain(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader1)

        processor.configure_for_run(self.paths[0], Pix, Samp, reader1)

        assert processor.Pix == 1855
        assert processor.Samp == 60
        assert processor.tel_id == tel_id
        assert processor.counter_evt == 0
        assert processor.counter_ped == 0
        assert processor.wf.shape == (Pix, Samp)
        assert processor.wf_ped.shape == (Pix, Samp)
        assert processor.wf_average.shape == (Pix, Samp)
        assert processor.wf_ped_average.shape == (Pix, Samp)
        assert processor.wf_list_plot is not None

    def test_process_event(self):
        """Test process_event method processes events correctly."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=5)
            tel_id = reader.subarray.tel_ids[0]

            for gain in [HIGH_GAIN, LOW_GAIN]:
                processor = WaveFormsHighLowGain(gain, r0=True)
                Pix, Samp = processor.define_for_run(reader)

                wf = np.zeros((Pix, Samp), dtype=np.float32)
                wf_ped = np.zeros((Pix, Samp), dtype=np.float32)

                processor.configure_for_run(path, Pix, Samp, reader)

                counter_ped = 0
                counter_evt = 0
                evt_count = 0

                for evt in reader:
                    processor.process_event(evt, noped=False)
                    evt_count += 1

                    is_ped = evt.trigger.event_type == EventType.SKY_PEDESTAL
                    is_phy = evt.trigger.event_type == EventType.SUBARRAY

                    if is_ped:
                        counter_ped += 1
                    elif is_phy:
                        counter_evt += 1
                    else:
                        counter_evt += 1

                    if processor.r0:
                        waveforms = evt.r0.tel[tel_id].waveform[gain]
                    else:
                        wf = evt.r1.tel[tel_id].waveform
                        if wf.ndim == 3:
                            waveforms = wf[gain]
                        else:
                            waveforms = wf

                    if is_ped:
                        wf_ped += waveforms
                    elif is_phy:
                        wf += waveforms
                    else:
                        wf += waveforms

                assert processor.counter_evt > 0 or processor.counter_ped > 0
                assert processor.counter_evt == counter_evt
                assert processor.counter_ped == counter_ped
                assert np.array_equal(processor.wf, wf)
                assert np.array_equal(processor.wf_ped, wf_ped)
                assert evt_count == 5

    def test_process_event_with_noped(self):
        """Test process_event with pedestal subtraction enabled."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=2)

        processor = WaveFormsHighLowGain(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader)
        processor.configure_for_run(self.paths[0], Pix, Samp, reader)

        for evt in reader:
            processor.process_event(evt, noped=True)

        assert processor.counter_evt > 0 or processor.counter_ped > 0

    def test_finish_run(self):
        """Test finish_run method computes statistics correctly."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=3)

            processor = WaveFormsHighLowGain(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()

            if processor.counter_evt > 0:
                wf_average = processor.wf / processor.counter_evt
                valid_pixels = ~np.any(
                    np.isinf(wf_average) | np.isnan(wf_average), axis=1
                )
                wf_mean_over_pix = np.mean(wf_average[valid_pixels], axis=0)

                assert np.array_equal(processor.wf_average, wf_average)
                assert np.array_equal(processor.wf_mean_over_pix, wf_mean_over_pix)

            if processor.counter_ped > 0:
                wf_ped_average = processor.wf_ped / processor.counter_ped
                valid_pixels = ~np.any(
                    np.isinf(wf_ped_average) | np.isnan(wf_ped_average), axis=1
                )
                wf_ped_mean_over_pix = np.mean(wf_ped_average[valid_pixels], axis=0)

                assert np.array_equal(processor.wf_ped_average, wf_ped_average)
                assert np.array_equal(
                    processor.wf_ped_mean_over_pix, wf_ped_mean_over_pix
                )

    def test_get_results(self):
        """Test get_results method returns proper dictionary structure."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=5)

            for gain in [HIGH_GAIN, LOW_GAIN]:
                processor = WaveFormsHighLowGain(gain, r0=True)
                Pix, Samp = processor.define_for_run(reader)
                processor.configure_for_run(path, Pix, Samp, reader)

                for evt in reader:
                    processor.process_event(evt, noped=False)

                processor.finish_run()
                results = processor.get_results()

                assert isinstance(results, dict)

                gain_c = "HIGH" if gain == 0 else "LOW"

                if processor.counter_evt > 0:
                    assert any(
                        f"WF-PHY-AVERAGE-PIX-{gain_c}-GAIN" in k for k in results.keys()
                    )
                    assert any(f"WF-PHY-{gain_c}-GAIN" in k for k in results.keys())
                if processor.counter_ped > 0:
                    assert any(
                        f"WF-PED-AVERAGE-PIX-{gain_c}-GAIN" in k for k in results.keys()
                    )
                    assert any(f"WF-PED-{gain_c}-GAIN" in k for k in results.keys())

    def test_plot_results(self):
        """Test plot_results method generates figures without errors."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=5)

            processor = WaveFormsHighLowGain(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()

            with tempfile.TemporaryDirectory() as tmpdir:
                fig_dict, name_dict = processor.plot_results("test_run", tmpdir)

                assert isinstance(fig_dict, dict)
                assert isinstance(name_dict, dict)

                if processor.counter_evt > 0 or processor.counter_ped > 0:
                    assert len(fig_dict) > 0

                from matplotlib.figure import Figure

                for fig in fig_dict.values():
                    assert isinstance(fig, Figure)
