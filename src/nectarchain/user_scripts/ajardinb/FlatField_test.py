import os

from nectarchain.makers.calibration import FlatfieldNectarCAMCalibrationTool

# Define the global environment variable NECTARCAMDATA (folder where are the runs)
os.environ["NECTARCAMDATA"] = "./20231222"

run_number = 4940
max_events = 10000
window_width = 14

# Call the tool
tool = FlatfieldNectarCAMCalibrationTool(
    progress_bar=True,
    run_number=run_number,
    max_events=max_events,
    log_level=20,
    window_width=window_width,
    overwrite=True,
)

tool.initialize()
tool.setup()

tool.start()
preFlatFieldOutput = tool.finish(return_output_component=True)[0]
