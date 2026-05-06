import logging

from ctapipe.core import run_tool
from ctapipe.core.logging import ColoredFormatter
from traitlets.config import Config

from nectarchain.makers.calibration import (
    FlatFieldSPENominalNectarCAMCalibrationTool,
    HiLoNectarCAMCalibrationTool,
    PedestalNectarCAMCalibrationTool,
)
from nectarchain.makers.calibration.core import NectarCAMCalibrationTool

######################
# Tool configuration #
######################

core_tool_name = NectarCAMCalibrationTool.__name__
ped_tool_name = PedestalNectarCAMCalibrationTool.__name__
gain_tool_name = FlatFieldSPENominalNectarCAMCalibrationTool.__name__
hilo_tool_name = HiLoNectarCAMCalibrationTool.__name__

config = Config()

# Global configurations for all tools
config[core_tool_name].max_events = 10000
config[core_tool_name].progress_bar = True
config[core_tool_name].overwrite = True
config[core_tool_name].log_level = "INFO"
config[core_tool_name].camera = "NectarCAMQM"

# Configure pedestal tool
config[ped_tool_name].run_number = 6249
config[ped_tool_name].events_per_slice = 3000

# Configure gain tool
config[gain_tool_name].run_number = 3936
config[gain_tool_name].method = "LocalPeakWindowSum"
config[gain_tool_name].extractor_kwargs = {"window_width": 8, "window_shift": 4}
config[gain_tool_name].multiproc = True
config[gain_tool_name].nproc = 8

# Configure HiLo tool
config[hilo_tool_name].run_number = 6252  # FF run
config[hilo_tool_name].method = config[gain_tool_name].method
config[hilo_tool_name].extractor_kwargs = config[gain_tool_name].extractor_kwargs


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
    ped_tool = PedestalNectarCAMCalibrationTool(config=config)
    if ped_tool.output_path.exists():
        log.info(f"Pedestals already computed at {ped_tool.output_path}")
    else:
        run_tool(ped_tool)

    # This might take a while if need to run the tool :)
    gain_tool = FlatFieldSPENominalNectarCAMCalibrationTool(config=config)
    if gain_tool.output_path.exists():
        log.info(f"Gain for HG channel already computed at {gain_tool.output_path}")
    else:
        run_tool(gain_tool)

    hilo_tool = HiLoNectarCAMCalibrationTool(
        config=config,
        pedestal_file=ped_tool.output_path,
        gain_file=gain_tool.output_path,
    )
    run_tool(hilo_tool)

    log.info(
        "Updated gain container with low gain values is written at: "
        f"{hilo_tool.output_path}"
    )


#####################################

if __name__ == "__main__":
    main()
