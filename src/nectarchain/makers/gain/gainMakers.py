import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

import numpy as np
from copy import copy 
from astropy.table import Column
import astropy.units as u

from ..core import CalibrationMaker

__all__ = ["GainMaker"]

class GainMaker(CalibrationMaker) : 
    """mother class for of the gain calibration
    """

#constructors
    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.__high_gain = np.empty((self.npixels),dtype = np.float64)
        self.__low_gain = np.empty((self.npixels),dtype = np.float64)
        self._results.add_column(Column(data = self.__high_gain,name = "high_gain",unit = u.dimensionless_unscaled))
        self._results.add_column(Column(np.empty((self.npixels,2),dtype = np.float64),"high_gain_error",unit = u.dimensionless_unscaled))
        self._results.add_column(Column(data = self.__low_gain,name = "low_gain",unit = u.dimensionless_unscaled))
        self._results.add_column(Column(np.empty((self.npixels,2),dtype = np.float64),"low_gain_error",unit = u.dimensionless_unscaled))
        
        self._results.add_column(Column(np.zeros((self.npixels),dtype = bool),"is_valid",unit = u.dimensionless_unscaled))


#getters and setters
    @property
    def _high_gain(self) : return self.__high_gain
    @_high_gain.setter
    def _high_gain(self,value) : self.__high_gain = value
    @property
    def high_gain(self) : return copy(self.__high_gain)
    @property
    def _low_gain(self) : return self.__low_gain
    @_low_gain.setter
    def _low_gain(self,value) : self.__low_gain = value
    @property
    def low_gain(self) : return copy(self.__low_gain)
    