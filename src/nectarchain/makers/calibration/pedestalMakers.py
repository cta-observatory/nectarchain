import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

from .core import NectarCAMCalibrationTool

__all__ = ["PedestalNectarCAMCalibrationTool"]


class PedestalNectarCAMCalibrationTool(NectarCAMCalibrationTool):
    def start(self):
        raise NotImplementedError(
            "The computation of the pedestal calibration is not yet implemented, feel free to contribute !:)"
        )
