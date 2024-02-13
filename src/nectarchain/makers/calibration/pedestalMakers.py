import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import pathlib
import numpy as np

from ctapipe.core.traits import ComponentNameList, Bool

from .core import NectarCAMCalibrationTool
from ...data.container import WaveformsContainer, WaveformsContainers
from ...data.management import DataManagement
from ...data.container.core import merge_map_ArrayDataContainer
from ..component import ArrayDataComponent, NectarCAMComponent

__all__ = ["PedestalNectarCAMCalibrationTool"]


class PedestalNectarCAMCalibrationTool(NectarCAMCalibrationTool):

    name = "PedestalNectarCAMCalibrationTool"

    reload_events = Bool(
        default_value=False, help="Reload the waveforms from raw data"
    )

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["PedestalEstimationComponent"],
        help="List of Component names to be applied, the order will be respected",
    ).tag(config=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_output_path(self):
        # str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
        #     self.extractor_kwargs
        # )
        if self.events_per_slice is None:
            ext = ".h5"
        else:
            ext = f"_sliced{self.events_per_slice}.h5"
        if self.max_events is None:
            filename = f"{self.name}_run{self.run_number}{ext}"
        else:
            filename = f"{self.name}_run{self.run_number}_maxevents{self.max_events}{ext}"

        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/PedestalEstimation/{filename}"
        )

    def start(
        self,
        max_events=np.inf,
        # trigger_type: list = None,
        restart_from_beginning: bool = False,
        *args,
        **kwargs,
    ):
        # need to implement waveform filter methods
        # str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
        #     self.extractor_kwargs
        # )

        files = DataManagement.find_waveforms(
            run_number=self.run_number,
            max_events=self.max_events,
        )
        if self.reload_events or len(files) != 1:
            if len(files) != 1:
                self.log.info(
                    f"{len(files)} computed waveforms files found with max_events > {self.max_events} for run {self.run_number}, reload waveforms from event loop"
                )
            print("Start parent class", super().name)
            super().start(
                restart_from_beginning=restart_from_beginning,
                *args,
                **kwargs,
            )
        else:
            print('Enter loop to read waveforms')
            self.log.info(f"reading waveforms from files {files[0]}")
            waveformsContainers = WaveformsContainer.from_hdf5(files[0])
            if isinstance(waveformsContainers, WaveformsContainer):
                self.components[0]._waveformsContainers = waveformsContainers
            elif isinstance(waveformsContainers, WaveformsContainers):
                self.log.debug("merging along TriggerType")
                self.components[0]._waveformsContainers = merge_map_ArrayDataContainer(
                    waveformsContainers
                )
            else:
                self.log.debug("merging along slices")
                waveformsContainers_merges_along_slices = (
                    ArrayDataComponent.merge_along_slices(
                        containers_generator=waveformsContainers
                    )
                )
                self.log.debug("merging along TriggerType")
                self.components[0]._waveformsContainers = merge_map_ArrayDataContainer(
                    waveformsContainers_merges_along_slices
                )

        print("Input wfs shape", np.shape(self.components[0]._waveformsContainers.wfs_hg))
