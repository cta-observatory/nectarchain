from astropy import time as astropytime
from ctapipe.io import EventSource
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam.constants import HIGH_GAIN
from tqdm import tqdm
from traitlets.config import Config

from nectarchain.dqm.camera_monitoring import CameraMonitoring


class TestCameraMonitoring:
    run_number = 3798
    max_events = 1

    def test_camera_monitoring(self):
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

        Pix, Samp = CameraMonitoring(HIGH_GAIN).DefineForRun(reader1)

        CameraMonitoring(HIGH_GAIN).ConfigureForRun(path, Pix, Samp, reader1)

        for evt in tqdm(reader1, total=1):
            run_start1 = evt.nectarcam.tel[0].svc.date
            SqlFileDate = astropytime.Time(run_start1, format="unix").iso.split(" ")[0]
        # print("SqlFileDate", SqlFileDate)
        # CameraMonitoring(HIGH_GAIN).FinishRun()
        assert Pix + Samp == 1915
        assert SqlFileDate == "2023-01-23"
