import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from ..core import CalibrationMaker

__all__ = "GainMaker"

class GainMaker(CalibrationMaker) : 
    """mother class for of the gain calibration
    """

    def __init__(self) -> None:
        super().__init__()
    