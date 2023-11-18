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

from ctapipe.core.traits import ComponentNameList,Bool,Path
from ctapipe.containers import EventType,Container


from .core import GainNectarCAMCalibrationTool
from ...component import NectarCAMComponent,ChargesComponent,ArrayDataComponent
from ...extractor.utils import CtapipeExtractor
from ....data.container.core import NectarCAMContainer,merge_map_ArrayDataContainer,TriggerMapContainer
from ....data.container import SPEfitContainer,ChargesContainer




__all__ = ["FlatFieldSPEHHVNectarCAMCalibrationTool","FlatFieldSPEHHVStdNectarCAMCalibrationTool","FlatFieldSPECombinedStdNectarCAMCalibrationTool"]


class FlatFieldSPEHHVNectarCAMCalibrationTool(GainNectarCAMCalibrationTool):
    name = "FlatFieldSPEHHVNectarCAM"
    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value = ["FlatFieldSingleHHVSPENectarCAMComponent"],                           
        help="List of Component names to be apply, the order will be respected"
    ).tag(config=True)

    compute_charge = Bool(
        default_value = False,
        help = "a flag to re compute the charge from raw data"
    )

    def _init_output_path(self) :
        self.output_path = pathlib.Path(f"{os.environ.get('NECTARCAMDATA','/tmp')}/SPEfit/{self.name}_run{self.run_number}_{self.method}_{CtapipeExtractor.get_extractor_kwargs_str(self.extractor_kwargs)}.h5")

    def start(
            self,
            n_events=np.inf,
            #trigger_type: list = None,
            restart_from_begining: bool = False,
            *args,
            **kwargs,) : 
        str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(self.extractor_kwargs)
        files = glob.glob(pathlib.Path(f"{os.environ.get('NECTARCAMDATA','/tmp')}/runs/charges/*_run{self.run_number}_{self.method}_{str_extractor_kwargs}.h5").__str__())
        if self.compute_charge or len(files) != 1 :
            if len(files) != 1 : 
                self.log.info(f"{len(files)} computed charges files found for run {self.run_number} with extraction method {self.method} and {str_extractor_kwargs},\n reload charges from event loop")
            super().start(n_events = n_events , restart_from_begining=restart_from_begining, *args, **kwargs)
        else : 
            self.log.info(f"reading computed charge from files {files[0]}")
            chargesContainers = ChargesContainer.from_hdf5(files[0])
            if isinstance(chargesContainers, NectarCAMContainer) : 
                self.components[0]._chargesContainers = chargesContainers
            else : 
                if isinstance(list(chargesContainers.containers.keys())[0],EventType) : 
                    self.log.debug("merging along TriggerType")
                    self.components[0]._chargesContainers = merge_map_ArrayDataContainer(chargesContainers)
                else : 
                    self.log.debug("merging along slices")
                    chargesContaienrs_merdes_along_slices = ArrayDataComponent.merge_along_slices(chargesContainers)
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
        self.output_path = pathlib.Path(f"{os.environ.get('NECTARCAMDATA','/tmp')}/SPEfit/{self.name}_run{self.run_number}_HHV{HHVrun}_{self.method}_{CtapipeExtractor.get_extractor_kwargs_str(self.extractor_kwargs)}.h5")

