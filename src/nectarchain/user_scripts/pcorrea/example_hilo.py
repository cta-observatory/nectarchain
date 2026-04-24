import os
import sys

from ctapipe.core import run_tool

from nectarchain.makers.calibration import HiLoNectarCAMCalibrationTool

# Set NECTARCAMDATA environment
os.environ["NECTARCAMDATA"] = "/data/users/pcorrea"

# Input gain file and pedestal file
dir_calib_files = "./calibration_files/"
pedestal_file = dir_calib_files + "pedestal_6249.h5"
gain_file = (
    dir_calib_files + "FlatFieldSPENominalStdNectarCAM_run3936_maxevents50000"
    "_LocalPeakWindowSum_window_shift_4_window_width_16.h5"
)

# Some traits
FF_run_number = 6252
max_events = 5000
progress_bar = True
overwrite = True
method = "LocalPeakWindowSum"
extractor_kwargs = {"window_width": 8, "window_shift": 4}
log_level = "INFO"


def main():
    hilo_tool = HiLoNectarCAMCalibrationTool(
        run_number=FF_run_number,
        max_events=max_events,
        overwrite=overwrite,
        progress_bar=progress_bar,
        log_level=log_level,
        method=method,
        extractor_kwargs=extractor_kwargs,
        gain_file=gain_file,
        pedestal_file=pedestal_file,
    )

    run_tool(hilo_tool)

    log.info(
        "Updated gain container with low gain values is written at: "
        f"{hilo_tool.output_path}"
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
        level=log_level,
    )
    log = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    main()
