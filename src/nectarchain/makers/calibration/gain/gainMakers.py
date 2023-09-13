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

class GainMaker(CalibrationMaker):
    """
    A class for gain calibration calculations on data.

    Inherits from the `CalibrationMaker` class and adds functionality specific to gain calibration.

    Members:
        __high_gain (ndarray): Private field to store the high gain values.
        __low_gain (ndarray): Private field to store the low gain values.

    Methods:
        __init__(self, *args, **kwargs): Initializes the `GainMaker` object and sets up the result table with columns for high gain, high gain error, low gain, low gain error, and validity flag.
        _high_gain.setter: Sets the high gain values.
        high_gain(self): Returns a copy of the high gain values.
        _low_gain.setter: Sets the low gain values.
        low_gain(self): Returns a copy of the low gain values.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the `GainMaker` object and sets up the result table with columns for high gain, high gain error, low gain, low gain error, and validity flag.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.__high_gain = np.empty((self.npixels), dtype=np.float64)
        self.__low_gain = np.empty((self.npixels), dtype=np.float64)
        self._results.add_column(Column(data=self.__high_gain, name="high_gain", unit=u.dimensionless_unscaled))
        self._results.add_column(Column(np.empty((self.npixels, 2), dtype=np.float64), "high_gain_error", unit=u.dimensionless_unscaled))
        self._results.add_column(Column(data=self.__low_gain, name="low_gain", unit=u.dimensionless_unscaled))
        self._results.add_column(Column(np.empty((self.npixels, 2), dtype=np.float64), "low_gain_error", unit=u.dimensionless_unscaled))
        self._results.add_column(Column(np.zeros((self.npixels), dtype=bool), "is_valid", unit=u.dimensionless_unscaled))

    @property
    def _high_gain(self):
        """
        Getter for the high gain values.

        Returns:
            ndarray: A copy of the high gain values.
        """
        return self.__high_gain

    @_high_gain.setter
    def _high_gain(self, value):
        """
        Setter for the high gain values.

        Args:
            value (ndarray): The high gain values.
        """
        self.__high_gain = value

    @property
    def high_gain(self):
        """
        Getter for the high gain values.

        Returns:
            ndarray: A copy of the high gain values.
        """
        return copy(self.__high_gain)

    @property
    def _low_gain(self):
        """
        Getter for the low gain values.

        Returns:
            ndarray: A copy of the low gain values.
        """
        return self.__low_gain

    @_low_gain.setter
    def _low_gain(self, value):
        """
        Setter for the low gain values.

        Args:
            value (ndarray): The low gain values.
        """
        self.__low_gain = value

    @property
    def low_gain(self):
        """
        Getter for the low gain values.

        Returns:
            ndarray: A copy of the low gain values.
        """
        return copy(self.__low_gain)
    