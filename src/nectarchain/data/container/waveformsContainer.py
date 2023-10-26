import logging
import sys

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import os
from abc import ABC
from enum import Enum
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Column, QTable, Table
from ctapipe.containers import Field,Container,partial,Map
from ctapipe.instrument.subarray import SubarrayDescription
from tqdm import tqdm

from .core import ArrayDataContainer,TriggerMapContainer


class WaveformsContainer(ArrayDataContainer):
    """
    A container that holds information about waveforms from a specific run.

    Fields:
        nsamples (int): The number of samples in the waveforms.
        subarray (SubarrayDescription): The subarray description instance.
        wfs_hg (np.ndarray): An array of high gain waveforms.
        wfs_lg (np.ndarray): An array of low gain waveforms.
    """

    nsamples = Field(
        type=int,
        description="number of samples in the waveforms",
    )
    #subarray = Field(type=SubarrayDescription, description="The subarray  description")
    wfs_hg = Field(type=np.ndarray, dtype = np.uint16, ndim = 3, description="high gain waveforms")
    wfs_lg = Field(type=np.ndarray, dtype = np.uint16, ndim = 3, description="low gain waveforms")


class WaveformsContainers(TriggerMapContainer):
    containers = Field(default_factory=partial(Map, WaveformsContainer),
                       description = "trigger mapping of WaveformContainer"
                       )

