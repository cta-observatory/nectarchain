import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import time
from argparse import ArgumentError

import numpy as np
import numpy.ma as ma
from ctapipe.containers import EventType
from ctapipe.image.extractor import (
    BaselineSubtractedNeighborPeakWindowSum,
    FixedWindowSum,
    FullWaveformSum,
    GlobalPeakWindowSum,
    LocalPeakWindowSum,
    NeighborPeakWindowSum,
    SlidingWindowMaxSum,
    TwoPassWindowSum,
)
from ctapipe.instrument.subarray import SubarrayDescription
from ctapipe_io_nectarcam import constants
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from numba import bool_, float64, guvectorize, int64

from ..data.container import ChargesContainer, WaveformsContainer
from .extractor.utils import CtapipeExtractor

import numpy as np

from ctapipe.containers import EventType
from ctapipe.instrument import SubarrayDescription
from ctapipe_io_nectarcam import constants
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from ctapipe.core.traits import ComponentNameList,Unicode
from tqdm import tqdm

from ..data.container import WaveformsContainer
from .core import EventsLoopNectarCAMCalibrationTool
from .component import NectarCAMComponent,ChargesComponent,get_specific_traits

__all__ = ["ChargesNectarCAMCalibrationTool"]




class ChargesNectarCAMCalibrationTool(EventsLoopNectarCAMCalibrationTool):
    """class use to make the waveform extraction from event read from r0 data"""
    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value = ["ChargesComponent"],                           
        help="List of Component names to be apply, the order will be respected"
    ).tag(config=True)
    
