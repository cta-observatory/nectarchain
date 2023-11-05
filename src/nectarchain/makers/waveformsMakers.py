import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import copy
from argparse import ArgumentError
import pathlib

import numpy as np
from ctapipe.containers import EventType
from ctapipe.instrument import SubarrayDescription
from ctapipe_io_nectarcam import constants
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from ctapipe.core.traits import ComponentNameList,Path
from tqdm import tqdm
import os

from ..data.container import WaveformsContainer
from .core import EventsLoopNectarCAMCalibrationTool
from .component import NectarCAMComponent

__all__ = ["WaveformsNectarCAMCalibrationTool"]


class WaveformsNectarCAMCalibrationTool(EventsLoopNectarCAMCalibrationTool):
    """class use to make the waveform extraction from event read from r0 data"""
    name = "WaveformsNectarCAMCalibration"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value = ["WaveformsComponent"],                           
        help="List of Component names to be apply, the order will be respected"
    ).tag(config=True)

    def _init_output_path(self) :
        self.output_path = pathlib.Path(f"{os.environ.get('NECTARCAMDATA','/tmp')}/runs/waveforms/{self.name}_run{self.run_number}.h5")


    
    