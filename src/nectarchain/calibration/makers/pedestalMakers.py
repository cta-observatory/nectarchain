import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from .core import CalibrationMaker

__all__ = ["PedestalMaker"]

class PedestalMaker(CalibrationMaker) : 
    def __init__(self,*args,**kwargs) -> None: 
        super().__init__(*args,**kwargs)
    def make(self) : 
        raise NotImplementedError("The computation of the pedestal calibration is not yet implemented, feel free to contribute !:)")

        