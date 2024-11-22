import logging

from .core import NectarCAMCalibrationTool

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


__all__ = ["FlatfieldNectarCAMCalibrationTool"]


#class FlatfieldNectarCAMCalibrationTool(NectarCAMCalibrationTool):
#    def start(self):
#        raise NotImplementedError(
#            "The computation of the flatfield calibration is not yet implemented, feel free to contribute !:)"
#        )

from nectarchain.makers import EventsLoopNectarCAMCalibrationTool


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
