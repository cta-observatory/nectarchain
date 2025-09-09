import logging

from ctapipe.core.traits import List

from ..core import EventsLoopNectarCAMCalibrationTool

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class NectarCAMCalibrationTool(EventsLoopNectarCAMCalibrationTool):
    name = "CalibrationTool"
    # PIXELS_ID_COLUMN = "pixels_id"
    # NP_PIXELS = "npixels"

    pixels_id = List(
        default_value=None,
        help="the list of pixel id to apply the components",
        allow_none=True,
    ).tag(config=True)
