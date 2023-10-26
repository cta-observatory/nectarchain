import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers
import os
from collections.abc import Iterable
from copy import copy
from datetime import date
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Column, QTable

from ..core import BaseMaker

__all__ = [""]


class CalibrationMaker(BaseMaker):
    """
    Mother class for all calibration makers that can be defined to compute calibration coefficients from data.

    Attributes:
        _reduced_name (str): A string representing the name of the calibration.
        PIXELS_ID_COLUMN (str): A string representing the name of the column in the result table that stores the pixels id.
        NP_PIXELS (str): A string representing the key in the metadata that stores the number of pixels.

    Members:
        _pixels_id (ndarray): A private property that stores the pixels id.
        _results (QTable): A private property that stores the result table.
    """

    _reduced_name = "Calibration"
    PIXELS_ID_COLUMN = "pixels_id"
    NP_PIXELS = "npixels"

    def __new__(cls, *args, **kwargs):
        """
        Constructor.

        Returns:
            CalibrationMaker: An instance of the CalibrationMaker class.
        """
        return super(CalibrationMaker, cls).__new__(cls)

    def __init__(self, pixels_id, *args, **kwargs) -> None:
        """
        Initialize the CalibrationMaker object.

        Args:
            pixels_id (iterable, np.ndarray): The list of pixels id.
        """
        super().__init__()
        if not (isinstance(pixels_id, Iterable)):
            raise TypeError("pixels_id must be iterable")
        self.__pixels_id = np.array(pixels_id)
        self.__results = QTable()
        self.__results.add_column(
            Column(
                self.__pixels_id,
                __class__.PIXELS_ID_COLUMN,
                unit=u.dimensionless_unscaled,
            )
        )
        self.__results.meta[__class__.NP_PIXELS] = self.npixels
        self.__results.meta[
            "comments"
        ] = f'Produced with NectarChain, Credit : CTA NectarCam {date.today().strftime("%B %d, %Y")}'

    def save(self, path, **kwargs):
        """
        Saves the results to a file in the specified path.

        Args:
            path (str): The path to save the results.
            **kwargs: Additional keyword arguments.

        Keyword Args:
            overwrite (bool): Whether to overwrite an existing file. Defaults to False.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        log.info(f"data saved in {path}")
        self._results.write(
            f"{path}/results_{self._reduced_name}.ecsv",
            format="ascii.ecsv",
            overwrite=kwargs.get("overwrite", False),
        )

    @property
    def _pixels_id(self):
        """
        Get the pixels id.

        Returns:
            ndarray: The pixels id.
        """
        return self.__pixels_id

    @_pixels_id.setter
    def _pixels_id(self, value):
        """
        Set the pixels id.

        Args:
            value (ndarray): The pixels id.
        """
        self.__pixels_id = value

    @property
    def pixels_id(self):
        """
        Get a copy of the pixels id.

        Returns:
            ndarray: A copy of the pixels id.
        """
        return copy(self.__pixels_id)

    @property
    def npixels(self):
        """
        Get the number of pixels.

        Returns:
            int: The number of pixels.
        """
        return len(self.__pixels_id)

    @property
    def _results(self):
        """
        Get the result table.

        Returns:
            QTable: The result table.
        """
        return self.__results

    @property
    def results(self):
        """
        Get a copy of the result table.

        Returns:
            QTable: A copy of the result table.
        """
        return copy(self.__results)
