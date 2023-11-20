import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import glob
import os
import pathlib

import astropy.units as u
import numpy as np
from astropy.table import Column
from ctapipe.containers import Container, EventType
from ctapipe.core.traits import Bool, ComponentNameList, Integer, Path

from ....data.container import ChargesContainer, SPEfitContainer
from ....data.container.core import (
    NectarCAMContainer,
    TriggerMapContainer,
    merge_map_ArrayDataContainer,
)
from ...chargesMakers import ChargesNectarCAMCalibrationTool
from ...component import ArrayDataComponent, ChargesComponent, NectarCAMComponent
from ...extractor.utils import CtapipeExtractor
from .core import GainNectarCAMCalibrationTool


class PhotoStatisticNectarCAMCalibrationTool(GainNectarCAMCalibrationTool):
    ###TO DO : IMPLEMENT a MOTHER PHOTOSTAT CLASS WITH ONLY 1 RUN WITH FF AND PEDESTAL INTERLEAVED.

    name = "PhotoStatisticNectarCAM"
    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["PhotoStatisticNectarCAMComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)
    run_number = Integer(help="the FF run number to be treated", default_value=-1).tag(
        config=True
    )
    Ped_run_number = Integer(
        help="the FF run number to be treated", default_value=-1
    ).tag(config=True)
    SPE_result = Path(
        help="the path of the SPE result container computed with very high voltage data",
    ).tag(config=True)

    run_file = Path(
        help="desactivated for PhotoStatistic maker",
        default_value=None,
        allow_none=True,
        read_only=True,
    ).tag(config=False)

    def _init_output_path(self):
        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/PhotoStat/{self.name}_FFrun{self.run_number}_{self.method}_{CtapipeExtractor.get_extractor_kwargs_str(self.extractor_kwargs)}_Pedrun{self.Ped_run_number}_FullWaveformSum.h5"
        )

    def _load_eventsource(self, FF_run=True):
        if FF_run:
            self.log.debug("loading FF event source")
            self.event_source = self.enter_context(
                self.load_run(self.run_number, self.max_events)
            )
        else:
            self.log.debug("loading Ped event source")
            self.event_source = self.enter_context(
                self.load_run(self.Ped_run_number, self.max_events)
            )

    def start(
        self,
        n_events=np.inf,
        *args,
        **kwargs,
    ):
        str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
            self.extractor_kwargs
        )
        FF_files = glob.glob(
            pathlib.Path(
                f"{os.environ.get('NECTARCAMDATA','/tmp')}/runs/charges/*_run{self.run_number}_{self.method}_{str_extractor_kwargs}.h5"
            ).__str__()
        )
        Ped_files = glob.glob(
            pathlib.Path(
                f"{os.environ.get('NECTARCAMDATA','/tmp')}/runs/charges/*_run{self.Ped_run_number}_FullWaveformSum_.h5"
            ).__str__()
        )

        if self.compute_charge or len(FF_files) != 1 or len(Ped_files) != 1:
            if len(FF_files) != 1 or len(Ped_files) != 1:
                self.log.info(
                    f"{len(FF_files)} computed charges FF files found for run {self.run_number} with extraction method {self.method} and {str_extractor_kwargs},\n reload charges from event loop"
                )
                self.log.info(
                    f"{len(Ped_files)} computed charges FF files found for run {self.Ped_run_number} with extraction method FullWaveformSum,\n reload charges from event loop"
                )

            super().start(
                n_events=n_events, restart_from_begining=False, *args, **kwargs
            )
            self._setup_eventsource(FF_run=False)
            super().start(
                n_events=n_events, restart_from_begining=False, *args, **kwargs
            )
        else:
            self.log.info(f"reading computed charge from FF file {FF_files[0]}")
            chargesContainers = ChargesContainer.from_hdf5(FF_files[0])
            if isinstance(chargesContainers, NectarCAMContainer):
                self.components[0]._FF_chargesContainers = chargesContainers
            else:
                if isinstance(list(chargesContainers.containers.keys())[0], EventType):
                    self.log.debug("merging along TriggerType")
                    self.components[
                        0
                    ]._FF_chargesContainers = merge_map_ArrayDataContainer(
                        chargesContainers
                    )
                else:
                    self.log.debug("merging along slices")
                    chargesContaienrs_merdes_along_slices = (
                        ArrayDataComponent.merge_along_slices(chargesContainers)
                    )
                    self.log.debug("merging along TriggerType")
                    self.components[
                        0
                    ]._FF_chargesContainers = merge_map_ArrayDataContainer(
                        chargesContaienrs_merdes_along_slices
                    )

            self.log.info(f"reading computed charge from Ped file {Ped_files[0]}")
            chargesContainers = ChargesContainer.from_hdf5(Ped_files[0])
            if isinstance(chargesContainers, NectarCAMContainer):
                self.components[0]._Ped_chargesContainers = chargesContainers
            else:
                if isinstance(list(chargesContainers.containers.keys())[0], EventType):
                    self.log.debug("merging along TriggerType")
                    self.components[
                        0
                    ]._Ped_chargesContainers = merge_map_ArrayDataContainer(
                        chargesContainers
                    )
                else:
                    self.log.debug("merging along slices")
                    chargesContaienrs_merdes_along_slices = (
                        ArrayDataComponent.merge_along_slices(chargesContainers)
                    )
                    self.log.debug("merging along TriggerType")
                    self.components[
                        0
                    ]._Ped_chargesContainers = merge_map_ArrayDataContainer(
                        chargesContaienrs_merdes_along_slices
                    )

    def _write_container(self, container: Container, index_component: int = 0) -> None:
        # if isinstance(container,SPEfitContainer) :
        #    self.writer.write(table_name = f"{self.method}_{CtapipeExtractor.get_extractor_kwargs_str(self.extractor_kwargs)}",
        #                      containers = container,
        #    )
        # else :
        super()._write_container(container=container, index_component=index_component)
