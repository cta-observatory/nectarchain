import logging
import os

import numpy as np
from traitlets.config import Config

from nectarchain.makers.calibration.calibration_pipeline import (
    PipelineNectarCAMCalibrationTool,
)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

# Set NECTARCAMDATA environment
os.environ["NECTARCAMDATA"] = "/data/users/pcorrea"

# Some the run numbers to use for each calibration tool
ped_run_number = 6249
FF_run_number = 6252
FF_SPE_run_number = 3936
FF_SPE_HHV_run_number = 3942

# Specify which tools to use for each step in the calibration
ped_tool_name = "PedestalNectarCAMCalibrationTool"
gain_tool_name = "FlatFieldSPENominalStdNectarCAMCalibrationTool"
FF_tool_name = "FlatfieldNectarCAMCalibrationTool"

# Path for a 1400-V result file of the SPE fit
# used in the SPE combined fit and photostatistic method
SPE_HHV_result_path = (
    "/data/users/pcorrea/SPEHHV_res/"
    "FlatFieldSPEHHVStdNectarCAM_run3942_LocalPeakWindowSum_"
    "window_shift_4_window_width_8.h5"
)

# Some general configurations that will pass to all subtools
max_events = 12000
progress_bar = True
overwrite = True

# Some specific configurations for each subtool
config = Config()
config[ped_tool_name].events_per_slice = 5000
config[gain_tool_name].multiproc = True
config[gain_tool_name].nproc = 8
# config[gain_tool_name].display = False
asked_pixels_id = np.arange(100)
config[gain_tool_name].asked_pixels_id = asked_pixels_id.tolist()


def main():
    tool = PipelineNectarCAMCalibrationTool(
        config=config,
        ped_run_number=ped_run_number,
        FF_run_number=FF_run_number,
        FF_SPE_run_number=FF_SPE_run_number,
        FF_SPE_HHV_run_number=FF_SPE_HHV_run_number,
        SPE_HHV_result_path=SPE_HHV_result_path,
        ped_tool_name=ped_tool_name,
        gain_tool_name=gain_tool_name,
        FF_tool_name=FF_tool_name,
        max_events=max_events,
        progress_bar=progress_bar,
        overwrite=overwrite,
    )

    tool.run()


if __name__ == "__main__":
    main()
