from ctapipe.image.extractor import LocalPeakWindowSum  # noqa: F401
from ctapipe.io import EventSource
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam.constants import HIGH_GAIN
from tqdm import tqdm
from traitlets.config import Config

from nectarchain.dqm.trigger_statistics import TriggerStatistics
from nectarchain.makers import ChargesNectarCAMCalibrationTool


class TestTriggerStatistics:
    run_number = 3798
    max_events = 1

    def test_trigger_statistics(self):
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

        Pix, Samp = TriggerStatistics(HIGH_GAIN).DefineForRun(reader1)

        kwargs = {
            "method": LocalPeakWindowSum,
            "extractor_kwargs": '{"window_width":16,"window_shift":4}',
        }
        charges_kwargs = {}
        tool = ChargesNectarCAMCalibrationTool()
        for key in tool.traits().keys():
            if key in kwargs.keys():
                charges_kwargs[key] = kwargs[key]

        TriggerStatistics(HIGH_GAIN).ConfigureForRun(
            path, Pix, Samp, reader1, charges_kwargs
        )

        for evt in tqdm(reader1, total=1):
            time = evt.trigger.time.value
            TriggerStatistics(HIGH_GAIN).ProcessEvent(evt, noped=False)

        # TriggerStatistics(HIGH_GAIN).FinishRun()
        assert Pix + Samp == 1915
        assert time == 1674462932.6398556
