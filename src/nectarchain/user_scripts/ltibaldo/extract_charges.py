import logging
import os
import pathlib

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

from nectarchain.makers import ChargesNectarCAMCalibrationTool

run_number = 3938
max_events = 100

tool = ChargesNectarCAMCalibrationTool(
    progress_bar=True,
    method = "FullWaveformSum",
    run_number=run_number,
    max_events=max_events,
    log_level=20,
)

tool.initialize()
tool.setup()
tool.start()
tool.finish(return_output_component=False)