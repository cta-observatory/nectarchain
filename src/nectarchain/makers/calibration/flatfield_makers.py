import logging
import os
import pathlib

from ctapipe.core.traits import ComponentNameList

from ..component import NectarCAMComponent
from .core import NectarCAMCalibrationTool

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

__all__ = ["FlatfieldNectarCAMCalibrationTool"]


class FlatfieldNectarCAMCalibrationTool(NectarCAMCalibrationTool):
    """Calibration tool for flat-field data processing.

    Computes flat-field coefficients and relative gains from NectarCAM
    flat-field runs.
    """

    name = "FlatfieldNectarCAMCalibrationTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["FlatFieldComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def _init_output_path(self):
        """Set the output path for flat-field calibration results.

        The filename includes the run number and optionally the maximum
        number of events.
        """
        if self.max_events is None:
            filename = f"{self.name}_run{self.run_number}.h5"
        else:
            filename = f"{self.name}_run{self.run_number}_maxevents{self.max_events}.h5"
        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/FlatFieldTests/{filename}"
        )
