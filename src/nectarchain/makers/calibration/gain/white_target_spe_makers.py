import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

__all__ = ["GainMaker", "WhiteTargetSPEMaker"]


class GainMaker:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("This class is not yet implemented")


class WhiteTargetSPEMaker(GainMaker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def make(self):
        raise NotImplementedError(
            "The computation of the white target calibration is not yet implemented,"
            "feel free to contribute !:)"
        )
