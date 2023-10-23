import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import copy
from argparse import ArgumentError

import numpy as np
from ctapipe.containers import EventType
from ctapipe.instrument import SubarrayDescription
from ctapipe_io_nectarcam import constants
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from ctapipe.core.traits import ComponentNameList
from tqdm import tqdm

from ..data.container import WaveformsContainer
from .core import EventsLoopNectarCAMCalibrationTool
from .component import NectarCAMComponent

__all__ = ["WaveformsNectarCAMCalibrationTool"]


class WaveformsNectarCAMCalibrationTool(EventsLoopNectarCAMCalibrationTool):
    """class use to make the waveform extraction from event read from r0 data"""
    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value = ["WaveformsComponent"],                           
        help="List of Component names to be apply, the order will be respected"
    ).tag(config=True)
    