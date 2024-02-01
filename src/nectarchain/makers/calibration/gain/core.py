import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

from ctapipe.core.traits import Bool

from ..core import NectarCAMCalibrationTool

__all__ = ["GainNectarCAMCalibrationTool"]


class GainNectarCAMCalibrationTool(NectarCAMCalibrationTool):
    reload_events = Bool(
        default_value=False, help="a flag to re compute the charge from raw data"
    )
