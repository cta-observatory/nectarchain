import numpy as np
from ctapipe.image.extractor import LocalPeakWindowSum  # noqa: F401
from ctapipe.io import EventSource
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam.constants import HIGH_GAIN
from tqdm import tqdm
from traitlets.config import Config

from nectarchain.dqm.mean_waveforms import MeanWaveFormsHighLowGain


class TestMeanWaveForms:
    run_number = 3798
    max_events = 1

    def test_mean_waveforms(self):
        # run_number = 3938
        path = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")

        config = None

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

        Pix, Samp = MeanWaveFormsHighLowGain(HIGH_GAIN).DefineForRun(reader1)

        MeanWaveFormsHighLowGain(HIGH_GAIN).ConfigureForRun(path, Pix, Samp, reader1)

        for evt in tqdm(reader1, total=1):
            self.pixelBAD = evt.mon.tel[0].pixel_status.hardware_failing_pixels
            # MeanWaveFormsHighLowGain(HIGH_GAIN).ProcessEvent(evt, noped = False)
        # MeanWaveFormsHighLowGain(HIGH_GAIN).FinishRun()
        assert Pix + Samp == 1915

        assert np.sum(evt.r0.tel[0].waveform[HIGH_GAIN][0]) == 3932100
