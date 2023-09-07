import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from abc import ABC, ABCMeta, abstractmethod

from astropy.table import QTable,Column
import astropy.units as u
from copy import copy
from datetime import date


__all__ = [""]

class CalibrationMaker(ABC) : 
    """mother class for all the calibration makers that can be defined to compute calibration coeficients from data
    """

#constructors
    def __new__(cls,*args,**kwargs) : 
        return super(CalibrationMaker,cls).__new__(cls)

    def __init__(self,*args,**kwargs) -> None:
        super().__init__()
        self.__pixels_id = kwargs.get("pixels_id", None)
        self.__results = QTable()
        self.__results.add_column(Column(self.__pixels_id,"pixels_id",unit = u.dimensionless_unscaled))
        self.__results.meta['npixels'] = self.npixels
        self.__results.meta['comments'] = f'Produced with NectarChain, Credit : CTA NectarCam {date.today().strftime("%B %d, %Y")}'

#methods
    @abstractmethod
    def make(self,*args,**kwargs) :
        pass

#getters and setters
    @property
    def _pixels_id(self) : return self.__pixels_id
    @_pixels_id.setter
    def _pixels_id(self,value) : self.__pixels_id = value
    @property
    def pixels_id(self) : return copy(self.__pixels_id)

    @property
    def npixels(self) : return len(self.__pixels_id)
    
    @property
    def _results(self) : return self.__results
    @property
    def results(self) : return copy(self.__results)
    



