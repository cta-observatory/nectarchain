from astropy import time as astropytime
from ctapipe.io import EventSource
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam.constants import HIGH_GAIN
from tqdm import tqdm
from traitlets.config import Config

from nectarchain.dqm.camera_monitoring import CameraMonitoring


class TestCameraMonitoring:
    def test_camera_monitoring(self):
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

        Pix, Samp = CameraMonitoring(HIGH_GAIN).define_for_run(reader1)

        sql_file_date = None
        for evt in tqdm(reader1, total=1):
            run_start1 = evt.nectarcam.tel[0].svc.date
            sql_file_date = astropytime.Time(run_start1, format="unix").iso.split(" ")[
                0
            ]

        assert Pix + Samp == 1915
        assert sql_file_date == "2023-01-23"
