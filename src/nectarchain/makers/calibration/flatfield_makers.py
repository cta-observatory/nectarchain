import logging
import os
import pathlib

from ctapipe.core.traits import ComponentNameList

from nectarchain.makers import EventsLoopNectarCAMCalibrationTool
from nectarchain.makers.component import NectarCAMComponent

from .core import NectarCAMCalibrationTool

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

from .core import NectarCAMCalibrationTool

__all__ = ["FlatfieldNectarCAMCalibrationTool"]


class FlatfieldNectarCAMCalibrationTool(EventsLoopNectarCAMCalibrationTool):
    name = "NectarCAM"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["preFlatFieldComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def _init_output_path(self):
        if self.max_events is None:
            filename = f"{self.name}_run{self.run_number}.h5"
        else:
            filename = f"{self.name}_run{self.run_number}_maxevents{self.max_events}.h5"
        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/FlatFieldTests/{filename}"
        )

