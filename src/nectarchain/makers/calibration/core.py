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

from ..core import EventsLoopNectarCAMCalibrationTool
from ctapipe.core.traits import List

__all__ = [""]


class NectarCAMCalibrationTool(EventsLoopNectarCAMCalibrationTool):

    name = "CalibrationTool"
    #PIXELS_ID_COLUMN = "pixels_id"
    #NP_PIXELS = "npixels"

    pixels_id = List(default_value = None,
                     help = "the list of pixel id to apply the components",
                     allow_none = True,
                     ).tag(config=  True)

    


'''
    def setup(self) -> None:
        super().setup()
        self.__results = QTable()
        self.__results.add_column(
            Column(
                self.pixels_id,
                self.pixels_id.name,
                unit=u.dimensionless_unscaled,
            )
        )
        self.__results.meta[__class__.NP_PIXELS] = self.npixels
        self.__results.meta[
            "comments"
        ] = f'Produced with NectarChain, Credit : CTA NectarCam {date.today().strftime("%B %d, %Y")}'

    def finish(self, path, **kwargs):
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
'''