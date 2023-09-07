import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from .core import CalibrationMaker


__all__ = ["FlatfieldMaker"]

class FlatfieldMaker(CalibrationMaker) : 
    def __init__(self,**kwargs) : 
        super().__init__(**kwargs)

    def make(self) : 
        raise NotImplementedError("The computation of the flatfield calibration is not yet implemented, feel free to contribute !:)")