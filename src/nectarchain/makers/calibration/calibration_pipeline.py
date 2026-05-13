import logging
import os
import pathlib

from ctapipe.core import run_tool
from ctapipe.core.traits import CaselessStrEnum, Integer, Path

from . import flatfield_makers, gain, pedestal_makers
from .core import NectarCAMCalibrationTool

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


__all__ = ["PipelineNectarCAMCalibrationTool"]

PEDESTAL_CALIBRATION_TOOLS = {
    name: getattr(pedestal_makers, name) for name in pedestal_makers.__all__
}
HILO_CALIBRATION_TOOLS = {
    "HiLoNectarCAMCalibrationTool": gain.HiLoNectarCAMCalibrationTool
}
GAIN_CALIBRATION_TOOLS = {
    name: getattr(gain, name)
    for name in gain.__all__
    if name not in HILO_CALIBRATION_TOOLS
}
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
        help="Run number for gain calibration at nominal voltage using SPE-fit method",
        default_value=-1,
    ).tag(config=True)
    FF_SPE_HHV_run_number = Integer(
        help=(
            "Run number for gain calibration at very-high voltage "
            "using SPE-fit method"
        ),
        default_value=-1,
    ).tag(config=True)
    SPE_HHV_result_path = Path(
        help="Path to SPE-fit result at very-high voltage",
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

        # This is to ensure that default output paths get correct conf values
        if not ("output_path" in kwargs.keys()):
            self._init_output_path()

        # Setup a temporary directory to store the results of each step in the
        # calibration pipeline
        self.subtool_res_dir = self.output_path.parent / "tmp"
        self.ped_output_path = (
            self.subtool_res_dir / f"output_{self.pedestal_tool_name}.h5"
        )
        self.gain_output_path = (
            self.subtool_res_dir / f"output_{self.gain_tool_name}.h5"
        )
        self.FF_output_path = (
            self.subtool_res_dir / f"output_{self.flatfield_tool_name}.h5"
        )

        # Setup pedestal tool
        pedestal_cls = PEDESTAL_CALIBRATION_TOOLS[self.pedestal_tool_name]
        self.pedestal_tool = pedestal_cls(
            parent=self,
            run_number=self.ped_run_number,
            output_path=self.ped_output_path,
        )
        # Setup gain tool
        gain_cls = GAIN_CALIBRATION_TOOLS[self.gain_tool_name]
        if "SPENominal" in self.gain_tool_name:
            self.gain_tool = gain_cls(
                parent=self,
                run_number=self.FF_SPE_run_number,
                output_path=self.gain_output_path,
            )
        elif "SPEHHV" in self.gain_tool_name:
            self.gain_tool = gain_cls(
                parent=self,
                run_number=self.FF_SPE_HHV_run_number,
                output_path=self.gain_output_path,
            )
        elif "SPECombined" in self.gain_tool_name:
            self.gain_tool = gain_cls(
                parent=self,
                run_number=self.FF_SPE_run_number,
                SPE_result=self.SPE_HHV_result_path,
                output_path=self.gain_output_path,
            )
        elif "PhotoStatistic" in self.gain_tool_name:
            self.gain_tool = gain_cls(
                parent=self,
                run_number=self.FF_run_number,
                Ped_run_number=self.ped_run_number,
                SPE_result=self.SPE_HHV_result_path,
                output_path=self.gain_output_path,
            )
        # Setup flatfield tool
        flatfield_cls = FLATFIELD_CALIBRATION_TOOLS[self.flatfield_tool_name]
        self.flatfield_tool = flatfield_cls(
            parent=self,
            run_number=self.FF_run_number,
            pedestal_file=self.ped_output_path,
            gain_file=self.gain_output_path,
            output_path=self.FF_output_path,
        )

    def _init_output_path(self):
        # TODO: update calib_filename with right output file (=calibration file)
        # Could be either fits or h5
        calib_filename = f"{self.name}_run{self.run_number}.h5"
        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/calib_pipeline/"
            f"{os.getpid()}/{calib_filename}"
        )

    def start(self):
        run_tool(self.pedestal_tool)
        run_tool(self.gain_tool)
        run_tool(self.flatfield_tool)

    def finish(self):
        # TODO: write calibration file
        pass
