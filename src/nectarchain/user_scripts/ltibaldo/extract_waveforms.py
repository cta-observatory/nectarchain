import logging

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

from nectarchain.makers import WaveformsNectarCAMCalibrationTool

run_number = 3938
tool = WaveformsNectarCAMCalibrationTool(
    progress_bar=True,
    run_number=run_number,
    max_events=300,
    log_level=20,
    events_per_slice = 100,
)

tool.initialize()
tool.setup()

tool.start()
output = tool.finish()