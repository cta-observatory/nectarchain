from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam import LightNectarCAMEventSource as EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN
from tqdm import tqdm
from traitlets.config import Config

from nectarchain.dqm.trigger_statistics import TriggerStatistics


class TestTriggerStatistics:
    def test_trigger_statistics(self):
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

        Pix, Samp = TriggerStatistics(HIGH_GAIN).define_for_run(reader1)

        TriggerStatistics(HIGH_GAIN).configure_for_run(path, Pix, Samp, reader1)

        for evt in tqdm(reader1, total=1):
            time = evt.trigger.time.value
            TriggerStatistics(HIGH_GAIN).process_event(evt, noped=False)

        assert Pix + Samp == 1915
        assert time == 1674462932.6398556
