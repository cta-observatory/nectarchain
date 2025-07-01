from ctapipe.io import EventSource
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam.constants import HIGH_GAIN
from tqdm import tqdm
from traitlets.config import Config

from nectarchain.dqm.mean_camera_display import MeanCameraDisplayHighLowGain


class TestMeanCameraDisplayHighLowGain:
    def test_mean_camera_display(self):
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

        Pix, Samp = MeanCameraDisplayHighLowGain(HIGH_GAIN).define_for_run(reader1)

        camera_average = None
        for evt in tqdm(reader1, total=1):
            camera_average = evt.r0.tel[0].waveform[HIGH_GAIN].sum(axis=1)

        assert Pix + Samp == 1915
        assert camera_average.sum(axis=0) == 109723121
