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
import pathlib
import glob
import os

from ctapipe.containers import EventType
from ctapipe.instrument import SubarrayDescription
from ctapipe_io_nectarcam import constants
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from ctapipe.core.traits import ComponentNameList,Unicode,Bool,Path
from tqdm import tqdm

from ..data.container import WaveformsContainer,TriggerMapContainer,ChargesContainers,WaveformsContainers
from .core import EventsLoopNectarCAMCalibrationTool
from .component import NectarCAMComponent,WaveformsComponent,ChargesComponent

__all__ = ["ChargesNectarCAMCalibrationTool"]


class ChargesNectarCAMCalibrationTool(EventsLoopNectarCAMCalibrationTool):
    """class use to make the waveform extraction from event read from r0 data"""
    name = "ChargesNectarCAMCalibration"

    

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value = ["ChargesComponent"],                           
        help="List of Component names to be apply, the order will be respected"
    ).tag(config=True)

    from_computed_waveforms = Bool(
        default_value = False,
        help = "a flag to compute charge from waveforms stored on disk"
    )

    def __init__(self,*args,**kwargs) : 
           super().__init__(*args,**kwargs)
           #self.__still_finished = False
    def _init_output_path(self) :
        str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(self.extractor_kwargs)
        self.output_path = pathlib.Path(f"{os.environ.get('NECTARCAMDATA','/tmp')}/runs/charges/{self.name}_run{self.run_number}_{self.method}_{str_extractor_kwargs}.h5")


    def start(
        self,
        n_events=np.inf,
        #trigger_type: list = None,
        restart_from_begining: bool = False,
        *args,
        **kwargs,
    ):
        if self.from_computed_waveforms : 
            files = glob.glob(pathlib.Path(f"{os.environ.get('NECTARCAMDATA','/tmp')}/runs/waveforms/*_run{self.run_number}.h5").__str__())
            if len(files) != 1 : 
                self.log.info(f"{len(files)} computed wavforms files found for run {self.run_number}, reload waveforms from event loop")
                super().start(n_events = n_events , restart_from_begining=restart_from_begining, *args, **kwargs)
            else : 
                self.log.info(f"{files[0]} is the computed wavforms files found for run {self.run_number}")
                waveformsContainers = WaveformsComponent.waveformsContainer_from_hdf5(files[0])
                if isinstance(waveformsContainers,WaveformsContainers) : 
                    chargesContainers = ChargesContainers()
                    if isinstance(list(waveformsContainers.containers.keys())[0],EventType) : 
                        self.log.debug('WaveformsContainer file container multiple trigger type')
                        self._init_writer(sliced = False)
                        chargesContainers = ChargesComponent._create_from_waveforms_looping_eventType(
                                                            waveformsContainers = waveformsContainers.containers[key],
                                                            subarray = self.subarray,
                                                            method = self.method,
                                                            **self.extractor_kwargs)
                        self._write_container(container = chargesContainers)
                    else : 
                        self.log.debug('WaveformsContainer file container multiple slices of the run events')
                        for key in waveformsContainers.containers.keys() : 
                            self.log.debug(f"extraction of charge associated to {key}")
                            slice_index = int(key.split('_')[-1])
                            self._init_writer(sliced = True,slice_index=slice_index)
                            chargesContainers = ChargesComponent._create_from_waveforms_looping_eventType(
                                                                    waveformsContainers = waveformsContainers.containers[key],
                                                                    subarray = self.event_source.subarray,
                                                                    method = self.method,
                                                                    **self.extractor_kwargs 
                                                            )
                            self._write_container(container = chargesContainers)
                else : 
                    self.log.debug('WaveformsContainer file container is not mapped with trigger type')
                    self._init_writer(sliced = False)
                    chargesContainers = ChargesComponent.create_from_waveforms(waveformsContainer = waveformsContainers,
                                                            subarray = self.subarray,
                                                            method = self.method,
                                                            **self.extractor_kwargs 
                                                            )
                    self._write_container(container = chargesContainers)
                
                self.writer.close()
                #self.__still_finished = True

        else : 
            super().start(n_events = n_events , restart_from_begining=restart_from_begining, *args, **kwargs)
