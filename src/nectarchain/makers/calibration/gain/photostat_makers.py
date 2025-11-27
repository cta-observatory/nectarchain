import logging
import os
import pathlib

import numpy as np
from ctapipe.containers import Container
from ctapipe.core.traits import ComponentNameList, Integer, Path

from ....data.container import ChargesContainer, ChargesContainers
from ....data.container.core import merge_map_ArrayDataContainer
from ....data.management import DataManagement
from ...component import ArrayDataComponent, NectarCAMComponent
from ...extractor.utils import CtapipeExtractor
from .core import GainNectarCAMCalibrationTool

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


__all__ = ["PhotoStatisticNectarCAMCalibrationTool"]


class PhotoStatisticNectarCAMCalibrationTool(GainNectarCAMCalibrationTool):
    # TO DO : IMPLEMENT a MOTHER PHOTOSTAT CLASS WITH ONLY 1 RUN WITH FF AND PEDESTAL
    # INTERLEAVED.

    name = "PhotoStatisticNectarCAM"
    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["PhotoStatisticNectarCAMComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)
    run_number = Integer(
        help="the flat-field run number to be treated", default_value=-1
    ).tag(config=True)
    Ped_run_number = Integer(
        help="the pedestal run number to be treated", default_value=-1
    ).tag(config=True)

    run_file = Path(
        help="desactivated for PhotoStatistic maker\
            with FF and pedestal runs separated",
        default_value=None,
        allow_none=True,
        read_only=True,
    ).tag(config=False)

    events_per_slice = Integer(
        help="desactivated for PhotoStatistic maker\
            with FF and pedestal runs separated",
        default_value=None,
        allow_none=True,
        read_only=True,
    ).tag(config=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_output_path(self):
        str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
            method=self.method,
            extractor_kwargs=self.extractor_kwargs,
        )
        if self.max_events is None:
            filename = (
                f"{self.name}_FFrun{self.run_number}_{self.method}"
                f"_{str_extractor_kwargs}_Pedrun{self.Ped_run_number}"
                f"_FullWaveformSum.h5"
            )
        else:
            filename = (
                f"{self.name}_FFrun{self.run_number}_{self.method}"
                f"_{str_extractor_kwargs}_Pedrun{self.Ped_run_number}_"
                f"FullWaveformSum_maxevents{self.max_events}.h5"
            )
        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/PhotoStat/{filename}"
        )

    def _load_eventsource(self, FF_run=True):
        if FF_run:
            self.log.debug("loading FF event source")
            self.event_source = self.enter_context(
                self.load_run(self.run_number, self.max_events, camera=self.camera)
            )
        else:
            self.log.debug("loading Ped event source")
            self.event_source = self.enter_context(
                self.load_run(self.Ped_run_number, self.max_events, camera=self.camera)
            )

    def start(
        self,
        n_events=np.inf,
        *args,
        **kwargs,
    ):
        str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
            method=self.method,
            extractor_kwargs=self.extractor_kwargs,
        )
        try:
            FF_files = DataManagement.find_charges(
                run_number=self.run_number,
                method=self.method,
                str_extractor_kwargs=str_extractor_kwargs,
                max_events=self.max_events,
            )
        except Exception as e:
            self.log.warning(f"{e}", exc_info=True)
            FF_files = []
        try:
            Ped_files = DataManagement.find_charges(
                run_number=self.Ped_run_number,
                max_events=self.max_events,
            )
        except Exception as e:
            self.log.warning(f"{e}", exc_info=True)
            Ped_files = []
        if self.reload_events or len(FF_files) != 1 or len(Ped_files) != 1:
            if len(FF_files) != 1 or len(Ped_files) != 1:
                self.log.info(
                    f"{len(FF_files)} computed charges FF files found"
                    f" with max_events >"
                    f"{self.max_events} for run {self.run_number}"
                    f" with extraction method"
                    f"{self.method} and {str_extractor_kwargs},\n reload charges"
                    f" from event loop"
                )
                self.log.info(
                    f"{len(Ped_files)} computed charges FF files found"
                    f" with max_events >"
                    f"{self.max_events} for run {self.Ped_run_number}"
                    f" with extraction"
                    f" method FullWaveformSum,\n reload charges from event loop"
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
            chargesContainers = ChargesContainers.from_hdf5(FF_files[0])
            if isinstance(chargesContainers, ChargesContainer):
                self.components[0]._FF_chargesContainers = chargesContainers
            else:
                n_slices = 0
                try:
                    while True:
                        next(chargesContainers)
                        n_slices += 1
                except StopIteration:
                    pass
                chargesContainers = ChargesContainers.from_hdf5(FF_files[0])
                if n_slices == 1:
                    self.log.info("merging along TriggerType")
                    self.components[
                        0
                    ]._FF_chargesContainers = merge_map_ArrayDataContainer(
                        next(chargesContainers)
                    )
                else:
                    self.log.info("merging along slices")
                    chargesContaienrs_merdes_along_slices = (
                        ArrayDataComponent.merge_along_slices(
                            containers_generator=chargesContainers
                        )
                    )
                    self.log.info("merging along TriggerType")
                    self.components[
                        0
                    ]._FF_chargesContainers = merge_map_ArrayDataContainer(
                        chargesContaienrs_merdes_along_slices
                    )

            self.log.info(f"reading computed charge from Ped file {Ped_files[0]}")
            chargesContainers = ChargesContainers.from_hdf5(Ped_files[0])
            if isinstance(chargesContainers, ChargesContainer):
                self.components[0]._Ped_chargesContainers = chargesContainers
            else:
                n_slices = 0
                try:
                    while True:
                        next(chargesContainers)
                        n_slices += 1
                except StopIteration:
                    pass
                chargesContainers = ChargesContainers.from_hdf5(Ped_files[0])
                if n_slices == 1:
                    self.log.info("merging along TriggerType")
                    self.components[
                        0
                    ]._Ped_chargesContainers = merge_map_ArrayDataContainer(
                        next(chargesContainers)
                    )
                else:
                    self.log.info("merging along slices")
                    chargesContaienrs_merdes_along_slices = (
                        ArrayDataComponent.merge_along_slices(
                            containers_generator=chargesContainers
                        )
                    )
                    self.log.info("merging along TriggerType")
                    self.components[
                        0
                    ]._Ped_chargesContainers = merge_map_ArrayDataContainer(
                        chargesContaienrs_merdes_along_slices
                    )

    def _write_container(self, container: Container, index_component: int = 0) -> None:
        # if isinstance(container,SPEfitContainer) :
        #    self.writer.write(table_name = f"{self.method}_{CtapipeExtractor.get_extrac
        # tor_kwargs_str(self.extractor_kwargs)}",
        #                      containers = container,
        #    )
        # else :
        super()._write_container(container=container, index_component=index_component)
