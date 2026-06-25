from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam import LightNectarCAMEventSource as EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN
from traitlets.config import Config

from nectarchain.dqm.ping_pong import PingPongMonitoring


class TestPingPongMonitoring:
    def test_ping_pong_monitoring(self):
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

        reader_for_define = EventSource(input_url=path, config=config, max_events=1)

        Pix, Samp = PingPongMonitoring(HIGH_GAIN, r0=True).define_for_run(
            reader_for_define
        )

        assert Pix + Samp == 1915
