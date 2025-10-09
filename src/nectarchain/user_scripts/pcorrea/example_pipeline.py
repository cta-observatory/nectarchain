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
FF_SPE_HHV_run_number = 3942
max_events = 1000
progress_bar = True
overwrite = True

ped_tool_name = "PedestalNectarCAMCalibrationTool"
gain_tool_name = "PhotoStatisticNectarCAMCalibrationTool"
SPE_HHV_result_path = (
    "/data/users/pcorrea/SPEHHV_res/"
    "output_FlatFieldSPEHHVNectarCAMCalibrationTool_run3942_maxEvents1000.h5"
)

config = Config()
# config.FlatfieldNectarCAMCalibrationTool.gain = gain_array.tolist()
# asked_pixels_id_SPE = np.arange(100)
# config.FlatFieldSPENominalNectarCAMCalibrationTool.asked_pixels_id = (
#     asked_pixels_id_SPE.tolist()
# )
config[ped_tool_name].events_per_slice = 500
config[gain_tool_name].multiproc = False
config[gain_tool_name].nproc = 8
# config[gain_tool_name].display = False
config[gain_tool_name].asked_pixels_id = [
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
]


def main():
    tool = PipelineNectarCAMCalibrationTool(
        config=config,
        ped_run_number=ped_run_number,
        FF_run_number=FF_run_number,
        FF_SPE_run_number=FF_SPE_run_number,
        FF_SPE_HHV_run_number=FF_SPE_HHV_run_number,
        SPE_HHV_result_path=SPE_HHV_result_path,
        gain_tool_name=gain_tool_name,
        max_events=max_events,
        progress_bar=progress_bar,
        overwrite=overwrite,
    )

    tool.run()


if __name__ == "__main__":
    main()
