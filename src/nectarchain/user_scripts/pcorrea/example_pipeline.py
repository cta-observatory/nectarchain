import logging
import os

import numpy as np
from ctapipe.core.logging import ColoredFormatter
from traitlets.config import Config

from nectarchain.makers.calibration import (
    FlatfieldNectarCAMCalibrationTool,
    FlatFieldSPECombinedStdNectarCAMCalibrationTool,
    FlatFieldSPEHHVNectarCAMCalibrationTool,
    FlatFieldSPEHHVStdNectarCAMCalibrationTool,
    FlatFieldSPENominalNectarCAMCalibrationTool,
    FlatFieldSPENominalStdNectarCAMCalibrationTool,
    HiLoNectarCAMCalibrationTool,
    PedestalNectarCAMCalibrationTool,
)
from nectarchain.makers.calibration.calibration_pipeline import (
    PipelineNectarCAMCalibrationTool,
)
from nectarchain.makers.calibration.core import NectarCAMCalibrationTool

######################
# Tool configuration #
######################

# Run numbers to use for each calibration tool
ped_run_number = 6249
FF_run_number = 6252
FF_SPE_run_number = 3936
FF_SPE_HHV_run_number = 3942

# Tools to use for each step in the calibration
ped_tool_name = PedestalNectarCAMCalibrationTool.__name__
gain_tool_name = FlatFieldSPENominalStdNectarCAMCalibrationTool.__name__
hilo_tool_name = HiLoNectarCAMCalibrationTool.__name__
FF_tool_name = FlatfieldNectarCAMCalibrationTool.__name__

# Path for a 1400-V result file of the SPE fit
# used in the SPE combined fit and photostatistic method
SPE_HHV_result_path = (
    "/data/users/pcorrea/SPEHHV_res/"
    "FlatFieldSPEHHVStdNectarCAM_run3942_LocalPeakWindowSum_"
    "window_shift_4_window_width_8.h5"
)

config = Config()

# Global configurations for all tools
core_tool_name = NectarCAMCalibrationTool.__name__
config[core_tool_name].max_events = 10000
config[core_tool_name].progress_bar = True
config[core_tool_name].overwrite = True
config[core_tool_name].log_level = "INFO"
config[core_tool_name].camera = "NectarCAMQM"

# Configure pedestal tool
config[ped_tool_name].events_per_slice = 3000
config[ped_tool_name].method = "FullWaveformSum"

# Configure gain tool
config[gain_tool_name].method = "LocalPeakWindowSum"
config[gain_tool_name].extractor_kwargs = {"window_width": 8, "window_shift": 4}
config[gain_tool_name].multiproc = True
config[gain_tool_name].nproc = 8
config[gain_tool_name].asked_pixels_id = np.arange(100).tolist()

# Configure HiLo tool
config[hilo_tool_name].method = config[gain_tool_name].method
config[hilo_tool_name].extractor_kwargs = config[gain_tool_name].extractor_kwargs

# Configure flat-field tool
config[FF_tool_name].charge_extraction_method = "LocalPeakWindowSum"
config[FF_tool_name].window_width = 12
config[FF_tool_name].window_shift = 4


################
# Logger setup #
################

handler = logging.StreamHandler()
handler.setFormatter(
    ColoredFormatter(fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s")
)

log = logging.getLogger(__name__)
log.setLevel(config[core_tool_name].log_level)
log.addHandler(handler)
log.propagate = False


#################
# Main function #
#################


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
        hilo_tool_name=hilo_tool_name,
        FF_tool_name=FF_tool_name,
    )

    tool.run()


if __name__ == "__main__":
    main()
