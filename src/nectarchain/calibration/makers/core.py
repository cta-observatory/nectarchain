import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from abc import ABC, abstractmethod
__all__ = ""

class CalibrationMaker(ABC) : 
    """mother class for all the calibration makers that can be defined to compute calibration coeficients from data
    """

    def __new__(cls) : 
        return super().__new__()


    def __init__(self) -> None:
        super().__init__()


    @abstractmethod
    def make(self,*args,**kwargs) :
        pass



