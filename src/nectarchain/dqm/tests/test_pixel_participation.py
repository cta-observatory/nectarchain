import os
import sys
import unittest

import numpy as np
from ctapipe.io import EventSource
from traitlets.config import Config

from nectarchain.dqm.pixel_participation import PixelParticipationHighLowGain

script_dir = os.path.dirname(os.path.abspath(__file__))
dqm_dir = os.path.abspath(os.path.join(script_dir, "../../dqm"))
# Add the directory to sys.path
sys.path.insert(0, dqm_dir)


class TestPixelParticipation(unittest.TestCase):
    run_number = 3938
    max_events = 1

    def test_pixel_participation(self):
        # NectarPath = "/Users/hashkar/Desktop/ashkar_nectar/data/"
        # path1 = "NectarCAM.Run3731.0000.fits.fz"
        # path = f"{NectarPath}/runs/{path1}"
        path1 = "$NECTARDATA/runs/NectarCAM.Run3794.0000.fits.fz"

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
        reader1 = EventSource(
            input_url=path1, config=config, max_events=TestPixelParticipation.max_events
        )
        Pix, Samp = PixelParticipationHighLowGain.DefineForRun(reader1)
        self.assertEqual(np.add(Pix, Samp), 1915)


if __name__ == "__main__":
    unittest.main()
