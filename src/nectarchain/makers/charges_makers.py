import logging
import os
import pathlib

import numpy as np
from ctapipe.containers import EventType
from ctapipe.core.traits import Bool, ComponentNameList
from ctapipe.image.extractor import FixedWindowSum  # noqa: F401
from ctapipe.image.extractor import FullWaveformSum  # noqa: F401
from ctapipe.image.extractor import GlobalPeakWindowSum  # noqa: F401
from ctapipe.image.extractor import LocalPeakWindowSum  # noqa: F401
from ctapipe.image.extractor import NeighborPeakWindowSum  # noqa: F401
from ctapipe.image.extractor import SlidingWindowMaxSum  # noqa: F401
from ctapipe.image.extractor import TwoPassWindowSum  # noqa: F401
from ctapipe.image.extractor import (  # noqa: F401
    BaselineSubtractedNeighborPeakWindowSum,
)

from ..data.container import (
    ChargesContainers,
    TriggerMapContainer,
    WaveformsContainer,
    WaveformsContainers,
)
from ..data.management import DataManagement
from .component import ChargesComponent, NectarCAMComponent
from .core import EventsLoopNectarCAMCalibrationTool
from .extractor.utils import CtapipeExtractor

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


__all__ = ["ChargesNectarCAMCalibrationTool"]


class ChargesNectarCAMCalibrationTool(EventsLoopNectarCAMCalibrationTool):
    """class use to make the waveform extraction from event read from r0 data"""

    name = "ChargesNectarCAMCalibration"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["ChargesComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    from_computed_waveforms = Bool(
        default_value=False,
        help="a flag to compute charge from waveforms stored on disk",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_output_path(self):
        str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
            self.extractor_kwargs
        )
        if self.max_events is None:
            filename = f"{self.name}_run{self.run_number}_{self.method}"
            f"{str_extractor_kwargs}.h5"
        else:
            filename = f"{self.name}_run{self.run_number}_maxevents{self.max_events}_"
            f"{self.method}_{str_extractor_kwargs}.h5"

        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/runs/charges/{filename}"
        )

    def start(
        self,
        n_events=np.inf,
        # trigger_type: list = None,
        restart_from_begining: bool = False,
        *args,
        **kwargs,
    ):
        ##cette implémentation est complétement nulle
        if self.from_computed_waveforms:
            files = DataManagement.find_waveforms(
                run_number=self.run_number, max_events=self.max_events
            )
            if len(files) != 1:
                self.log.info(
                    f"{len(files)} computed wavforms files found with max_events >="
                    f"{self.max_events}  for run {self.run_number}, reload waveforms"
                    f"from event loop"
                )
                super().start(
                    n_events=n_events,
                    restart_from_begining=restart_from_begining,
                    *args,
                    **kwargs,
                )
            else:
                self.log.info(
                    f"{files[0]} is the computed wavforms files found"
                    f"with max_events >="
                    f"{self.max_events}  for run {self.run_number}"
                )
                waveformsContainers = WaveformsContainers.from_hdf5(files[0])
                if not (isinstance(waveformsContainers, WaveformsContainer)):
                    n_slices = 0
                    try:
                        while True:
                            next(waveformsContainers)
                            n_slices += 1
                    except StopIteration:
                        pass
                    waveformsContainers = WaveformsContainers.from_hdf5(files[0])
                    if n_slices == 1:
                        self._init_writer(sliced=False)
                        chargesContainers = (
                            ChargesComponent._create_from_waveforms_looping_eventType(
                                waveformsContainers=next(waveformsContainers),
                                subarray=self.event_source.subarray,
                                method=self.method,
                                **self.extractor_kwargs,
                            )
                        )
                        self._write_container(container=chargesContainers)
                    else:
                        self.log.debug(
                            f"WaveformsContainer file contains {n_slices} slices of the run events"
                        )
                        for slice_index, _waveformsContainers in enumerate(
                            waveformsContainers
                        ):
                            self._init_writer(sliced=True, slice_index=slice_index)
                            chargesContainers = ChargesComponent._create_from_waveforms_looping_eventType(  # noqa
                                waveformsContainers=_waveformsContainers,
                                subarray=self.event_source.subarray,
                                method=self.method,
                                **self.extractor_kwargs,
                            )
                            self._write_container(container=chargesContainers)
                else:
                    self.log.debug(
                        "WaveformsContainer file container is a simple \
                        WaveformsContainer (not mapped)"
                    )
                    self._init_writer(sliced=False)
                    chargesContainers = ChargesComponent.create_from_waveforms(
                        waveformsContainer=waveformsContainers,
                        subarray=self.event_source.subarray,
                        method=self.method,
                        **self.extractor_kwargs,
                    )
                    self._write_container(container=chargesContainers)

                self.writer.close()
                # self.__still_finished = True

        else:
            super().start(
                n_events=n_events,
                restart_from_begining=restart_from_begining,
                *args,
                **kwargs,
            )
