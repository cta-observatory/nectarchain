import logging

import numpy as np
from ctapipe.core.traits import ComponentNameList

from nectarchain.makers.component import NectarCAMComponent

from ....data.container import (
    ChargesContainer,
    ChargesContainers,
    merge_map_ArrayDataContainer,
)
from ....data.management import DataManagement
from ...component import ArrayDataComponent
from ...extractor.utils import CtapipeExtractor
from .core import GainNectarCAMCalibrationTool

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

__all__ = ["HiLoNectarCAMCalibrationTool"]


class HiLoNectarCAMCalibrationTool(GainNectarCAMCalibrationTool):
    name = "HiLoNectarCAMCalibrationTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["HiLoComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_output_path(self):
        if self.gain_file is not None:
            self.output_path = self.gain_file.with_name(
                f"{self.gain_file.stem}_hilo_corrected{self.gain_file.suffix}"
            )
        else:
            # The HiLoComponent will raise an error if no gain_file is provided
            super()._init_output_path()

    def start(self, n_events=np.inf, *args, **kwargs):
        str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
            method=self.method,
            extractor_kwargs=self.extractor_kwargs,
        )
        try:
            files = DataManagement.find_charges(
                run_number=self.run_number,
                method=self.method,
                str_extractor_kwargs=str_extractor_kwargs,
                max_events=self.max_events,
            )
        except Exception as e:
            log.warning(e)
            files = []
        if self.reload_events or len(files) != 1:
            if len(files) != 1:
                self.log.info(
                    f"{len(files)} computed charges files found with max_events >"
                    f"{self.max_events} for run {self.run_number} with extraction"
                    f"method {self.method} and {str_extractor_kwargs},\n reload"
                    f"charges from event loop"
                )
            super().start(
                n_events=n_events,
                restart_from_begining=False,
                *args,
                **kwargs,
            )
        else:
            self.log.info(f"reading computed charge from files {files[0]}")
            chargesContainers = ChargesContainers.from_hdf5(files[0])
            if isinstance(chargesContainers, ChargesContainer):
                self.components[0]._chargesContainers = chargesContainers
            else:
                n_slices = 0
                try:
                    while True:
                        next(chargesContainers)
                        n_slices += 1
                except StopIteration:
                    pass
                chargesContainers = ChargesContainers.from_hdf5(files[0])
                if n_slices == 1:
                    self.log.info("merging along TriggerType")
                    self.components[
                        0
                    ]._chargesContainers = merge_map_ArrayDataContainer(
                        next(chargesContainers)
                    )
                else:
                    self.log.info("merging along slices")
                    chargesContainers_merged_along_slices = (
                        ArrayDataComponent.merge_along_slices(
                            containers_generator=chargesContainers
                        )
                    )
                    self.log.info("merging along TriggerType")
                    self.components[
                        0
                    ]._chargesContainers = merge_map_ArrayDataContainer(
                        chargesContainers_merged_along_slices
                    )
