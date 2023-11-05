import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import copy

import astropy.units as u
import numpy as np
from astropy.table import Column

from ..core import NectarCAMCalibrationTool

__all__ = ["GainNectarCAMCalibrationTool"]


class GainNectarCAMCalibrationTool(NectarCAMCalibrationTool):
    pass
