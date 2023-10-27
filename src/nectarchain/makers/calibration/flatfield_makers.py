import logging

from .core import CalibrationMaker

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

__all__ = ["FlatfieldMaker"]


class FlatfieldMaker(CalibrationMaker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def make(self):
        raise NotImplementedError(
            "The computation of the flatfield calibration is not yet implemented, "
            "feel free to contribute !:)"
        )
