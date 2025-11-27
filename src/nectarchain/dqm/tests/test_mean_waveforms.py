import numpy as np
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam import LightNectarCAMEventSource as EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN
from tqdm import tqdm
from traitlets.config import Config

from nectarchain.dqm.mean_waveforms import MeanWaveFormsHighLowGain


class TestMeanWaveForms:
    def test_mean_waveforms(self):
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
        tel_id = reader1.subarray.tel_ids[0]

        Pix, Samp = MeanWaveFormsHighLowGain(HIGH_GAIN, r0=True).define_for_run(reader1)

        evt = None
        for evt in tqdm(reader1, total=1):
            self.pixelBAD = evt.mon.tel[tel_id].pixel_status.hardware_failing_pixels

        assert Pix + Samp == 1915
        assert np.sum(evt.r0.tel[tel_id].waveform[HIGH_GAIN][0]) == 3932100
