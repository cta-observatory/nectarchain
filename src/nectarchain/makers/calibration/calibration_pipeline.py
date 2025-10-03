import logging
import os
import pathlib

from ctapipe.core import run_tool
from ctapipe.core.traits import CaselessStrEnum, Integer, Path
from traitlets.config import Config

from . import flatfield_makers, gain, pedestal_makers
from .core import NectarCAMCalibrationTool

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


__all__ = ["PipelineNectarCAMCalibrationTool"]

PEDESTAL_CALIBRATION_TOOLS = {
    name: getattr(pedestal_makers, name) for name in pedestal_makers.__all__
}
GAIN_CALIBRATION_TOOLS = {name: getattr(gain, name) for name in gain.__all__}
FLATFIELD_CALIBRATION_TOOLS = {
    name: getattr(flatfield_makers, name) for name in flatfield_makers.__all__
}


class PipelineNectarCAMCalibrationTool(NectarCAMCalibrationTool):
    name = "PipelineNectarCAMCalibrationTool"
    description = "Run pedestal -> gain -> flatfield calibrations in sequence"

    ped_run_number = Integer(
        help="Run number for pedestal calibration", default_value=-1
    ).tag(config=True)
    FF_run_number = Integer(
        help="Run number for flat-field calibration", default_value=-1
    ).tag(config=True)
    FF_SPE_run_number = Integer(
        help="Run number for gain calibration using SPE-fit method", default_value=-1
    ).tag(config=True)
    SPEfit_result_path = Path(
        help="Path to SPE fit result for gain calibration using photostatistic method",
        default_value=None,
        allow_none=True,
    ).tag(config=True)

    pedestal_tool_name = CaselessStrEnum(
        list(PEDESTAL_CALIBRATION_TOOLS.keys()),
        help="Name of tool to use for the pedestal calibration",
        default_value="PedestalNectarCAMCalibrationTool",
    ).tag(config=True)
    flatfield_tool_name = CaselessStrEnum(
        list(FLATFIELD_CALIBRATION_TOOLS.keys()),
        help="Name of tool to use for the flatfield calibration",
        default_value="FlatfieldNectarCAMCalibrationTool",
    ).tag(config=True)
    gain_tool_name = CaselessStrEnum(
        list(GAIN_CALIBRATION_TOOLS.keys()),
        help="Name of tool to use for the gain calibration",
        default_value="FlatFieldSPENominalNectarCAMCalibrationTool",
    ).tag(config=True)

    classes = [
        *PEDESTAL_CALIBRATION_TOOLS.values(),
        *GAIN_CALIBRATION_TOOLS.values(),
        *FLATFIELD_CALIBRATION_TOOLS.values(),
    ]

    def setup(self, *args, **kwargs):
        # Default run_number = -1 will raise Exception
        self.run_number = 0
        log.warning(f"Set run_number = {self.run_number} to avoid exception")

        super().setup(*args, **kwargs)

        # Setup the configuration of all subtools
        config = self._setup_config()

        # This is to ensure that default output paths get correct conf values
        if not ("output_path" in kwargs.keys()):
            self._init_output_path()

        # Setup pedestal tool
        pedestal_cls = PEDESTAL_CALIBRATION_TOOLS[self.pedestal_tool_name]
        self.pedestal_tool = pedestal_cls(
            parent=self,
            config=config,
            run_number=self.ped_run_number,
            output_path=self.ped_output_path,
        )
        # Setup gain tool
        gain_cls = GAIN_CALIBRATION_TOOLS[self.gain_tool_name]
        if self.gain_tool_name == "PhotoStatisticNectarCAMCalibrationTool":
            self.gain_tool = gain_cls(
                parent=self,
                config=config,
                run_number=self.FF_run_number,
                Ped_run_number=self.ped_run_number,
                SPE_result=self.SPEfit_result_path,
                output_path=self.gain_output_path,
            )
        else:  # NOTE: not sure if this will work for the SPECombinedFit method
            self.gain_tool = gain_cls(
                parent=self,
                config=config,
                run_number=self.FF_SPE_run_number,
                output_path=self.gain_output_path,
            )
        # Setup flatfield tool
        flatfield_cls = FLATFIELD_CALIBRATION_TOOLS[self.flatfield_tool_name]
        self.flatfield_tool = flatfield_cls(
            parent=self,
            config=config,
            run_number=self.FF_run_number,
            pedestal_file=self.ped_output_path,
            gain_file=self.gain_output_path,
            output_path=self.FF_output_path,
        )

    def _init_output_path(self):
        # TODO: update calib_filename with right output file (=calibration file)

        if self.events_per_slice is None:
            ext = ".h5"
        else:
            ext = f"_sliced{self.events_per_slice}.h5"
        if self.max_events is not None:
            ext = f"_maxevents{self.max_events}{ext}"

        ped_filename = f"output_{self.pedestal_tool_name}_run{self.ped_run_number}{ext}"
        FF_filename = f"output_{self.flatfield_tool_name}_run{self.FF_run_number}{ext}"

        if self.gain_tool_name == "PhotoStatisticNectarCAMCalibrationTool":
            gain_filename = (
                f"output_{self.gain_tool_name}_FFrun_{self.FF_run_number}"
                f"_Pedrun_{self.ped_run_number}_SPEres_{self.SPEfit_result_path}{ext}"
            )
        else:
            gain_filename = (
                f"output_{self.gain_tool_name}_run_{self.FF_SPE_run_number}{ext}"
            )

        # TODO
        calib_filename = f"{self.name}_run{self.run_number}{ext}"
        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/calib_pipeline/{calib_filename}"
        )

        # Set intermediate paths for each subtool
        intermediate_dir = os.path.join(os.path.dirname(self.output_path), "tmp")
        self.ped_output_path = pathlib.Path(f"{intermediate_dir}/{ped_filename}")
        self.gain_output_path = pathlib.Path(f"{intermediate_dir}/{gain_filename}")
        self.FF_output_path = pathlib.Path(f"{intermediate_dir}/{FF_filename}")

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
        run_tool(self.gain_tool)
        run_tool(self.flatfield_tool)

    def finish(self):
        # TODO: write calibration file
        pass
