from ctapipe.io import EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN
from traitlets.config import Config

from nectarchain.dqm.mean_camera_display import MeanCameraDisplayHighLowGain


class TestMeanCameraDisplayHighLowGain:
    run_number = 3798
    max_events = 1

    def test_mean_camera_display(self):
        path1 = "/Users/hashkar/Desktop/ashkar_nectar/"
        path2 = "data/runs/NectarCAM.Run3798.0000.fits.fz"
        path = path1 + path2

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

        Pix, Samp = MeanCameraDisplayHighLowGain(HIGH_GAIN).DefineForRun(reader1)
        print(Pix, Samp)
        # self.assertEqual(Pix + Samp, 1915)
        assert Pix + Samp == 1915
