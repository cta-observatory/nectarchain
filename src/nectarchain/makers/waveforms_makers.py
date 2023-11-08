import logging

from ctapipe.core.traits import ComponentNameList

from .component import NectarCAMComponent
from .core import EventsLoopNectarCAMCalibrationTool

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

__all__ = ["WaveformsNectarCAMCalibrationTool"]


class WaveformsNectarCAMCalibrationTool(EventsLoopNectarCAMCalibrationTool):
    """class used to make the waveform extraction from event read from r0 data"""

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["WaveformsComponent"],
        help="List of Component names to be applied, the order will be respected",
    ).tag(config=True)
