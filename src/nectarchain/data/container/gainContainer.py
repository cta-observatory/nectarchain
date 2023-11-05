import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import os
from abc import ABC
from pathlib import Path

import numpy as np
from astropy.io import fits
from ctapipe.containers import Field,partial,Map

from .core import NectarCAMContainer

__all__ = ["GainContainer","SPEfitContainer"]

class GainContainer(NectarCAMContainer) : 
    high_gain = Field(type=np.ndarray, dtype = np.float64, ndim = 2, description="high gain")
    low_gain = Field(type=np.ndarray, dtype = np.float64, ndim = 2, description="low gain")
    pixels_id = Field(type=np.ndarray, dtype = np.uint16, ndim = 1, description="pixel ids")


class SPEfitContainer(GainContainer) : 
    is_valid = Field(type=np.ndarray, dtype = bool, ndim = 1, description="is_valid")
    likelihood = Field(type=np.ndarray, dtype = np.float64, ndim = 1, description="likelihood")
    p_value = Field(type=np.ndarray, dtype = np.float64, ndim = 1, description="p_value")
    pedestal = Field(type=np.ndarray, dtype = np.float64, ndim = 2, description="pedestal")
    pedestalWidth = Field(type=np.ndarray, dtype = np.float64, ndim = 2, description="pedestalWidth")
    resolution = Field(type=np.ndarray, dtype = np.float64, ndim = 2, description="resolution")
    luminosity = Field(type=np.ndarray, dtype = np.float64, ndim = 2, description="luminosity")
    mean = Field(type=np.ndarray, dtype = np.float64, ndim = 2, description="mean")
    n = Field(type=np.ndarray, dtype = np.float64, ndim = 2, description="n")
    pp = Field(type=np.ndarray, dtype = np.float64, ndim = 2, description="pp")
