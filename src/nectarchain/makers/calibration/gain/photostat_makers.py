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
from ....data.container import SPEfitContainer,ChargesContainer


class PhotoStatisticNectarCAMCalibrationTool(GainNectarCAMCalibrationTool):
    name = "PhotoStatisticNectarCAM"
    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value = ["PhotoStatisticNectarCAMComponent"],                           
        help="List of Component names to be apply, the order will be respected"
    ).tag(config=True)
    run_number = Integer(
        help="the FF run number to be treated", 
        default_value=-1
    ).tag(config=True)
    Ped_run_number = Integer(
        help="the FF run number to be treated", 
        default_value=-1
    ).tag(config=True)
    SPE_result = Path(
        help="the path of the SPE result container computed with very high voltage data",
    ).tag(config = True)

    def _init_output_path(self) :
        self.output_path = pathlib.Path(f"{os.environ.get('NECTARCAMDATA','/tmp')}/PhotoStat/{self.name}_run{self.run_number}_{self.method}_{CtapipeExtractor.get_extractor_kwargs_str(self.extractor_kwargs)}.h5")



    def _load_eventsource(self):
        self.log.debug("loading event source")
        self.event_source = self.enter_context(
            self.load_run(self.run_number, self.max_events, run_file=self.run_file)
        )
        
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