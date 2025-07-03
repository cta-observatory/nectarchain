import numpy as np
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam import LightNectarCAMEventSource as EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN
from tqdm import tqdm
from traitlets.config import Config

from nectarchain.dqm.charge_integration import ChargeIntegrationHighLowGain


class TestChargeIntegrationHighLowGain:
    def test_charge_integration(self):
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

        Pix, Samp = ChargeIntegrationHighLowGain(HIGH_GAIN).define_for_run(reader1)

        ped = None
        for evt in tqdm(reader1, total=1):
            self.pixelBAD = evt.mon.tel[0].pixel_status.hardware_failing_pixels
            waveform = evt.r0.tel[0].waveform[HIGH_GAIN]
            ped = np.mean(waveform[:, 20])

        assert Pix + Samp == 1915
        assert np.sum(ped) == 985.8636118598383
