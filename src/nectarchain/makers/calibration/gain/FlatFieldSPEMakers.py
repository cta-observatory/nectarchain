import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import astropy.units as u
import numpy as np
from astropy.table import Column
import pathlib
import os
import glob

from ctapipe.core.traits import ComponentNameList,Bool,Path,Integer
from ctapipe.containers import EventType,Container


from .core import GainNectarCAMCalibrationTool
from ...component import NectarCAMComponent,ChargesComponent,ArrayDataComponent
from ...extractor.utils import CtapipeExtractor
from ....data.container.core import NectarCAMContainer,merge_map_ArrayDataContainer,TriggerMapContainer
from ....data.container import SPEfitContainer,ChargesContainer,ChargesContainers
from ....data.management import DataManagement




__all__ = ["FlatFieldSPEHHVNectarCAMCalibrationTool","FlatFieldSPEHHVStdNectarCAMCalibrationTool","FlatFieldSPECombinedStdNectarCAMCalibrationTool"]


class FlatFieldSPEHHVNectarCAMCalibrationTool(GainNectarCAMCalibrationTool):
    name = "FlatFieldSPEHHVNectarCAM"
    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value = ["FlatFieldSingleHHVSPENectarCAMComponent"],                           
        help="List of Component names to be apply, the order will be respected"
    ).tag(config=True)

    #events_per_slice = Integer(
    #    help="feature desactivated for this class",
    #    default_value=None,
    #    allow_none=True,
    #    read_only = True,
    #).tag(config=True)

    def __init__(self,*args,**kwargs) : 
        super().__init__(*args,**kwargs)

        str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(self.extractor_kwargs)
        if not(self.reload_events) : 
            files = DataManagement.find_charges(
                run_number=self.run_number,
                method = self.method,
                str_extractor_kwargs=str_extractor_kwargs,
                max_events=self.max_events,
            )
            if len(files) == 1 : 
                log.warning("You asked events_per_slice but you don't want to reload events and a charges file is on disk, then events_per_slice is set to None")
                self.events_per_slice = None
        


    def _init_output_path(self) :
        str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(self.extractor_kwargs)
        if self.events_per_slice is None : 
            ext = '.h5'
        else : 
            ext = f'_sliced{self.events_per_slice}.h5'
        if self.max_events is None : 
            filename = f"{self.name}_run{self.run_number}_{self.method}_{str_extractor_kwargs}{ext}"
        else : 
            filename = f"{self.name}_run{self.run_number}_maxevents{self.max_events}_{self.method}_{str_extractor_kwargs}{ext}"

        self.output_path = pathlib.Path(f"{os.environ.get('NECTARCAMDATA','/tmp')}/SPEfit/{filename}")

    def start(
            self,
            n_events=np.inf,
            #trigger_type: list = None,
            restart_from_begining: bool = False,
            *args,
            **kwargs,) : 
        str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(self.extractor_kwargs)
        files = DataManagement.find_charges(
            run_number=self.run_number,
            method = self.method,
            str_extractor_kwargs=str_extractor_kwargs,
            max_events=self.max_events,
        )
        if self.reload_events or len(files) != 1 :
            if len(files) != 1 : 
                self.log.info(f"{len(files)} computed charges files found with max_events > {self.max_events} for run {self.run_number} with extraction method {self.method} and {str_extractor_kwargs},\n reload charges from event loop")
            super().start(n_events = n_events , restart_from_begining=restart_from_begining, *args, **kwargs)
        else : 
            self.log.info(f"reading computed charge from files {files[0]}")
            chargesContainers = ChargesContainer.from_hdf5(files[0])
            if isinstance(chargesContainers, ChargesContainer) : 
                self.components[0]._chargesContainers = chargesContainers
            elif isinstance(chargesContainers,ChargesContainers) : 
                self.log.debug("merging along TriggerType")
                self.components[0]._chargesContainers = merge_map_ArrayDataContainer(chargesContainers)
            else : 
                self.log.debug("merging along slices")
                chargesContaienrs_merdes_along_slices = ArrayDataComponent.merge_along_slices(containers_generator=chargesContainers)
                self.log.debug("merging along TriggerType")
                self.components[0]._chargesContainers = merge_map_ArrayDataContainer(chargesContaienrs_merdes_along_slices)

    def _write_container(self, container : Container,index_component : int = 0) -> None:
        #if isinstance(container,SPEfitContainer) : 
        #    self.writer.write(table_name = f"{self.method}_{CtapipeExtractor.get_extractor_kwargs_str(self.extractor_kwargs)}",
        #                      containers = container,
        #    )
        #else : 
        super()._write_container(container = container,index_component= index_component)
                


class FlatFieldSPEHHVStdNectarCAMCalibrationTool(FlatFieldSPEHHVNectarCAMCalibrationTool):
    name = "FlatFieldSPEHHVStdNectarCAM"
    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value = ["FlatFieldSingleHHVSPEStdNectarCAMComponent"],                           
        help="List of Component names to be apply, the order will be respected"
    ).tag(config=True)


class FlatFieldSPECombinedStdNectarCAMCalibrationTool(FlatFieldSPEHHVNectarCAMCalibrationTool):
    name = "FlatFieldCombinedStddNectarCAM"
    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value = ["FlatFieldCombinedSPEStdNectarCAMComponent"],                           
        help="List of Component names to be apply, the order will be respected"
    ).tag(config=True)

    def _init_output_path(self) :
        for word in self.SPE_result.stem.split('_') : 
            if 'run' in word : 
                HHVrun = int(word.split('run')[-1])
        str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(self.extractor_kwargs)
        if self.max_events is None : 
            filename = f"{self.name}_run{self.run_number}_HHV{HHVrun}_{self.method}_{str_extractor_kwargs}.h5"
        else : 
            filename = f"{self.name}_run{self.run_number}_maxevents{self.max_events}_HHV{HHVrun}_{self.method}_{str_extractor_kwargs}.h5"
        
        self.output_path = pathlib.Path(f"{os.environ.get('NECTARCAMDATA','/tmp')}/SPEfit/{filename}")

