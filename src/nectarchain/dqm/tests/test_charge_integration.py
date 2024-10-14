import argparse
import json

import numpy as np
from ctapipe.image import LocalPeakWindowSum
from ctapipe.io import EventSource
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam.constants import HIGH_GAIN
from tqdm import tqdm
from traitlets.config import Config

from nectarchain.dqm.charge_integration import ChargeIntegrationHighLowGain
from nectarchain.makers import ChargesNectarCAMCalibrationTool


class TestChargeIntegrationHighLowGain:
    run_number = 3938
    max_events = 1

    def test_charge_integration(self):
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

        parser = argparse.ArgumentParser(
            description="NectarCAM Data Quality Monitoring tool"
        )
        # extractor arguments
        parser.add_argument(
            "--method",
            choices=[
                "FullWaveformSum",
                "FixedWindowSum",
                "GlobalPeakWindowSum",
                "LocalPeakWindowSum",
                "SlidingWindowMaxSum",
                "TwoPassWindowSum",
            ],
            default="LocalPeakWindowSum",
            help="charge extractor method",
            type=str,
        )

        parser.add_argument(
            "--extractor_kwargs",
            default='{"window_shift": 4, "window_width": 16}',
            help="charge extractor kwargs",
            type=json.loads,
        )

        args, leftovers = parser.parse_known_args()

        reader1 = EventSource(input_url=path, config=config, max_events=1)
        subarray = reader1.subarray
        self.integrator = LocalPeakWindowSum(subarray, config=config)

        Pix, Samp = ChargeIntegrationHighLowGain(HIGH_GAIN).DefineForRun(reader1)

        method = args.method
        extractor_kwargs = args.extractor_kwargs

        kwargs = {"method": method, "extractor_kwargs": extractor_kwargs}
        charges_kwargs = {}
        tool = ChargesNectarCAMCalibrationTool()
        for key in tool.traits().keys():
            if key in kwargs.keys():
                charges_kwargs[key] = kwargs[key]

        ChargeIntegrationHighLowGain(HIGH_GAIN).ConfigureForRun(
            path, Pix, Samp, reader1, charges_kwargs
        )

        for evt in tqdm(reader1, total=1):
            self.pixelBAD = evt.mon.tel[0].pixel_status.hardware_failing_pixels
            waveform = evt.r0.tel[0].waveform[HIGH_GAIN]
            ped = np.mean(waveform[:, 20])
            # ChargeIntegrationHighLowGain(HIGH_GAIN).ProcessEvent(evt, noped = False)
        # ChargeIntegrationHighLowGain(HIGH_GAIN).FinishRun()
        assert Pix + Samp == 1915
        assert np.sum(ped) == 985.8636118598383
