import logging
import os
import pathlib

from ctapipe.core import run_tool
from ctapipe.core.traits import Integer
from traitlets.config import Config

# from ..extractor.utils import CtapipeExtractor
from .core import NectarCAMCalibrationTool
from .flatfield_makers import FlatfieldNectarCAMCalibrationTool
from .gain.flatfield_spe_makers import FlatFieldSPENominalNectarCAMCalibrationTool
from .gain.photostat_makers import PhotoStatisticNectarCAMCalibrationTool
from .pedestal_makers import PedestalNectarCAMCalibrationTool

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


__all__ = ["PipelineNectarCAMCalibrationTool"]


class PipelineNectarCAMCalibrationTool(NectarCAMCalibrationTool):
    name = "PipelineNectarCAMCalibrationTool"
    description = "Run pedestal -> gain -> flatfield calibrations in sequence"

    ped_run_number = Integer(
        help="Run number for pedestal calibration", default_value=-1
    ).tag(config=True)

    FF_run_number = Integer(
        help="Run number for flat-field calibration", default_value=-1
    ).tag(config=True)

    classes = [
        PedestalNectarCAMCalibrationTool,
        PhotoStatisticNectarCAMCalibrationTool,
        FlatFieldSPENominalNectarCAMCalibrationTool,
        FlatfieldNectarCAMCalibrationTool,
    ]

    def _init_output_path(self):
        # TODO: update calib_filename with right output file (=calibration file)

        if self.events_per_slice is None:
            ext = ".h5"
        else:
            ext = f"_sliced{self.events_per_slice}.h5"
        if self.max_events is None:
            calib_filename = f"{self.name}_run{self.run_number}{ext}"
            ped_filename = f"pedestal_run{self.ped_run_number}{ext}"
            FF_filename = f"flatfield_run{self.ped_run_number}{ext}"
        else:
            calib_filename = (
                f"{self.name}_run{self.run_number}_maxevents{self.max_events}{ext}"
            )
            ped_filename = (
                f"pedestal_run{self.ped_run_number}_maxevents{self.max_events}{ext}"
            )
            FF_filename = (
                f"flatfield_run{self.FF_run_number}_maxevents{self.max_events}{ext}"
            )
        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/calib_pipeline/{calib_filename}"
        )

        # Set intermediate paths for each subtool
        intermediate_dir = os.path.join(os.path.dirname(self.output_path), "tmp")
        self.ped_output_path = pathlib.Path(f"{intermediate_dir}/{ped_filename}")
        self.FF_output_path = pathlib.Path(f"{intermediate_dir}/{FF_filename}")

    def setup(self, *args, **kwargs):
        # Default run_number = -1 will raise Exception
        self.run_number = 0
        log.warning(f"Set run_number = {self.run_number} to avoid exception")

        super().setup(*args, **kwargs)

        # Setup the configuration of all subtools
        config = self._setup_config()

        # This is to ensure that output paths get correct conf values
        if not ("output_path" in kwargs.keys()):
            self._init_output_path()

        self.pedestal_tool = PedestalNectarCAMCalibrationTool(
            parent=self,
            config=config,
            run_number=self.ped_run_number,
            output_path=self.ped_output_path,
        )

        self.flatfield_tool = FlatfieldNectarCAMCalibrationTool(
            parent=self,
            config=config,
            run_number=self.FF_run_number,
            pedestal_file=self.ped_output_path,
            output_path=self.FF_output_path,
        )

    def _setup_config(self):
        """
        Build a merged Config for subtools:
        - Use explicit subtool config if present (highest priority).
        - Else inherit matching values from the parent tool config/traits.
        - Else fall back to subtool defaults.

        Returns:
            config: Merged Config for all subtools
        """
        config = Config()

        for subtool_cls in self.classes:
            subtool_name = subtool_cls.__name__

            # Skip if it's the parent tool
            if subtool_name == self.__class__.__name__:
                continue

            # Start with explicit subtool config if it exists
            subconfig = Config(self.config.get(subtool_name, {}))

            # Check each configurable trait of the subtool
            for name in subtool_cls.class_traits(config=True):
                # Already provided explicitly: keep it
                if name in subconfig:
                    continue
                # Do not overwrite components because they are fixed per tool!
                if name == "componentsList":
                    continue
                # If common trait propagate from parent
                if name in self.traits(config=True):
                    subconfig[name] = getattr(self, name)
                    self.log.debug(
                        f"Propagating {name}={getattr(self, name)!r} "
                        f"from {self.__class__.__name__} -> {subtool_name}"
                    )

            config[subtool_name] = subconfig

        return config

    def start(self):
        run_tool(self.pedestal_tool)
        # TODO: run_tool(self.gain_tool)
        run_tool(self.flatfield_tool)

    def finish(self):
        # TODO: write calibration file
        pass
