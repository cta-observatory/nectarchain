import logging
import os

import numpy as np
from ctapipe_io_nectarcam import constants
from traitlets.config import Config

from nectarchain.makers.calibration.calibration_pipeline import (
    PipelineNectarCAMCalibrationTool,
)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

os.environ["NECTARCAMDATA"] = "/data/users/pcorrea"

ped_run_number = 6249
FF_run_number = 6252
FF_SPE_run_number = 3936
max_events = 3000
events_per_slice = 1000
progress_bar = True
overwrite = True

config = Config()
# config.FlatfieldNectarCAMCalibrationTool.gain = gain_array.tolist()


def main():
    tool = PipelineNectarCAMCalibrationTool(
        config=config,
        ped_run_number=ped_run_number,
        FF_run_number=FF_run_number,
        FF_SPE_run_number=FF_SPE_run_number,
        max_events=max_events,
        events_per_slice=events_per_slice,
        progress_bar=progress_bar,
        overwrite=overwrite,
    )

    tool.run()


if __name__ == "__main__":
    main()
