from ctapipe.io import EventSource
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam.constants import HIGH_GAIN
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
        print(path)

        reader1 = EventSource(input_url=path, config=config, max_events=1)

        Pix, Samp = MeanWaveFormsHighLowGain(HIGH_GAIN).DefineForRun(reader1)
        print(Pix, Samp)
        # self.assertEqual(Pix + Samp, 1915)
        assert Pix + Samp == 1915
