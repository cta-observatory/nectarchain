import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

import os
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
        self.__gain = np.empty((self.npixels),dtype = np.float64)
        self._results.add_column(Column(data = self.__gain,name = "gain",unit = u.dimensionless_unscaled))
        self._results.add_column(Column(np.empty((self.npixels,2),dtype = np.float64),"gain_error",unit = u.dimensionless_unscaled))
        
        self._results.add_column(Column(np.zeros((self.npixels),dtype = bool),"is_valid",unit = u.dimensionless_unscaled))


#getters and setters
    @property
    def _gain(self) : return self.__gain
    @_gain.setter
    def _gain(self,value) : self.__gain = value
    @property
    def gain(self) : return copy(self.__gain)
    