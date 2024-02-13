import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

from .core import NectarCAMCalibrationTool

__all__ = ["PedestalNectarCAMCalibrationTool"]


class PedestalNectarCAMCalibrationTool(NectarCAMCalibrationTool):

    name = "PedestalNectarCAMCalibrationTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["PedestalEstimationComponent"],
        help="List of Component names to be applied, the order will be respected",
    ).tag(config=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
            self.extractor_kwargs
        )
        if not (self.reload_events):
            files = DataManagement.find_waveforms(
                run_number=self.run_number,
                max_events=self.max_events,
            )
            if len(files) == 1:
                log.warning(
                    "You asked events_per_slice but you don't want to reload events and a charges file is on disk, then events_per_slice is set to None"
                )
                self.events_per_slice = None







    def start(self):
        raise NotImplementedError(
            "The computation of the pedestal calibration is not yet implemented, feel free to contribute !:)"
        )
