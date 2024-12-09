import numpy as np
from ctapipe.io import EventSource
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam.constants import HIGH_GAIN
from tqdm import tqdm
from traitlets.config import Config

from nectarchain.dqm.pixel_timeline import PixelTimelineHighLowGain


class TestPixelTimeline:
    def test_pixel_timeline(self):
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

        Pix, Samp = PixelTimelineHighLowGain(HIGH_GAIN).DefineForRun(reader1)

        evt = None
        for evt in tqdm(reader1, total=1):
            self.pixelBAD = evt.mon.tel[0].pixel_status.hardware_failing_pixels

        assert Pix + Samp == 1915
        assert np.sum(evt.nectarcam.tel[0].svc.pixel_ids) == 1719375