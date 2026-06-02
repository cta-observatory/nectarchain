import logging
import os
import pathlib

from ctapipe.containers import (
    FlatFieldContainer,
    PedestalContainer,
    PixelStatusContainer,
    WaveformCalibrationContainer,
)
from ctapipe.core import ToolConfigurationError, run_tool
from ctapipe.core.traits import Bool, CaselessStrEnum, Integer, Path

from ...data.container import FlatFieldContainer as NectarCAMFlatFieldContainer
from ...data.container import GainContainer as NectarCAMGainContainer
from ...data.container import (
    NectarCAMContainer,
    NectarCAMPedestalContainer,
    flatfield_container,
    gain_container,
    pedestal_container,
)
from ...utils.utils import ContainerUtils
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

NECTARCAM_PEDESTAL_CONTAINER_CLASSES = [
    getattr(pedestal_container, name)
    for name in pedestal_container.__all__
    if issubclass(getattr(pedestal_container, name), NectarCAMContainer)
]
NECTARCAM_GAIN_CONTAINER_CLASSES = [
    getattr(gain_container, name)
    for name in gain_container.__all__
    if issubclass(getattr(gain_container, name), NectarCAMContainer)
]
NECTARCAM_FLATFIELD_CONTAINER_CLASSES = [
    getattr(flatfield_container, name)
    for name in flatfield_container.__all__
    if issubclass(getattr(flatfield_container, name), NectarCAMContainer)
]
NECTARCAM_CONTAINER_CLASSES_DICT = {
    "pedestal": NECTARCAM_PEDESTAL_CONTAINER_CLASSES,
    "gain": NECTARCAM_GAIN_CONTAINER_CLASSES,
    "flatfield": NECTARCAM_FLATFIELD_CONTAINER_CLASSES,
}

OUTPUT_FORMATS = [".h5", ".fits", ".fits.gz"]
GROUP_NAMES = ["data", "data_combined"]


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
    hilo_tool_name = CaselessStrEnum(
        list(HILO_CALIBRATION_TOOLS.keys()),
        help="Name of tool to use for the HiLo",
        default_value="HiLoNectarCAMCalibrationTool",
    ).tag(config=True)

    output_format = CaselessStrEnum(
        OUTPUT_FORMATS,
        help="Format of the category A calibration file",
        default_value=".fits",
    ).tag(config=True)
    save_tmp = Bool(
        default_value=True,
        help=(
            "Option to save tmp subdirectory containing the individual outputs "
            "of each subtool",
        ),
    ).tag(config=True)

    classes = [
        *PEDESTAL_CALIBRATION_TOOLS.values(),
        *GAIN_CALIBRATION_TOOLS.values(),
        *HILO_CALIBRATION_TOOLS.values(),
        *FLATFIELD_CALIBRATION_TOOLS.values(),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Temporary directory to store the results of each step in the
        # calibration pipeline
        self._res_dir_subtools = self.output_path.parent / "tmp"

        # Output paths of calibration subtools
        self._ped_output_path = pathlib.Path()
        self._gain_output_path = pathlib.Path()
        self._hilo_output_path = pathlib.Path()
        self._FF_output_path = pathlib.Path()
        self._output_paths_subtools = {}

        # Dictionary of the `nectarchain` container types of the subtools
        self._nectarcam_containers = {
            "pedestal": NectarCAMPedestalContainer(),
            "gain": NectarCAMGainContainer(),
            "flatfield": NectarCAMFlatFieldContainer(),
        }

        # Dictionary of the `ctapipe` container types that are written in the
        # cat-A calibration file
        self._ctapipe_containers = {
            "calibration": WaveformCalibrationContainer(),
            "flatfield": FlatFieldContainer(),
            "pedestal": PedestalContainer(),
            "pixel_status": PixelStatusContainer(),
        }

    def _init_output_path(self):
        calib_filename = (
            f"{self.name}_Pedrun{self.ped_run_number}_FFrun{self.FF_run_number}_"
            f"FFSPErun{self.FF_SPE_run_number}_FFSPEHHVrun{self.FF_SPE_HHV_run_number}"
            f"{self.output_format}"
        )

        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/calib_pipeline/"
            f"{os.getpid()}/{calib_filename}"
        )

    def setup(self, *args, **kwargs):
        # Default run_number = -1 will raise Exception
        self.run_number = 0
        log.warning(f"Set run_number = {self.run_number} to avoid exception")

        super().setup(*args, **kwargs)

        # Check if output path has correct format
        if not any(self.output_path.name.endswith(end) for end in OUTPUT_FORMATS):
            raise ToolConfigurationError(
                f"Format of output file {self.output_path.name} not valid, must be "
                f"one of {OUTPUT_FORMATS}"
            )

        # Setup pedestal tool
        pedestal_cls = PEDESTAL_CALIBRATION_TOOLS[self.pedestal_tool_name]
        self._ped_output_path = (
            self._res_dir_subtools
            / f"output_{self.pedestal_tool_name}_run{self.ped_run_number}.h5"
        )
        self.pedestal_tool = pedestal_cls(
            parent=self,
            run_number=self.ped_run_number,
            output_path=self._ped_output_path,
        )

        # Setup gain tool
        gain_cls = GAIN_CALIBRATION_TOOLS[self.gain_tool_name]
        if "SPENominal" in self.gain_tool_name:
            self._gain_output_path = (
                self._res_dir_subtools
                / f"output_{self.gain_tool_name}_run{self.FF_SPE_run_number}.h5"
            )
            self.gain_tool = gain_cls(
                parent=self,
                run_number=self.FF_SPE_run_number,
                output_path=self._gain_output_path,
            )
        elif "SPEHHV" in self.gain_tool_name:
            self._gain_output_path = (
                self._res_dir_subtools / f"output_{self.gain_tool_name}_"
                f"run{self.FF_SPE_HHV_run_number}.h5"
            )
            self.gain_tool = gain_cls(
                parent=self,
                run_number=self.FF_SPE_HHV_run_number,
                output_path=self._gain_output_path,
            )
        elif "SPECombined" in self.gain_tool_name:
            for word in self.SPE_HHV_result_path.name.split("_"):
                if "run" in word:
                    self.FF_SPE_HHV_run_number = int(word.split("run")[-1])
                    break
            if self.FF_SPE_HHV_run_number == -1:
                self.log.warning(
                    f"HHV run number not specified in {self.SPE_HHV_result_path}"
                )
            self._gain_output_path = (
                self._res_dir_subtools / f"output_{self.gain_tool_name}_"
                f"run{self.FF_SPE_run_number}_"
                f"HHVrun{self.FF_SPE_HHV_run_number}.h5"
            )
            self.gain_tool = gain_cls(
                parent=self,
                run_number=self.FF_SPE_run_number,
                SPE_result=self.SPE_HHV_result_path,
                output_path=self._gain_output_path,
            )
        elif "PhotoStatistic" in self.gain_tool_name:
            for word in self.SPE_HHV_result_path.name.split("_"):
                if "run" in word:
                    self.FF_SPE_HHV_run_number = int(word.split("run")[-1])
                    break
            if self.FF_SPE_HHV_run_number == -1:
                self.log.warning(
                    f"HHV run number not specified in {self.SPE_HHV_result_path}"
                )
            self._gain_output_path = (
                self._res_dir_subtools / f"output_{self.gain_tool_name}_"
                f"FFrun{self.FF_SPE_run_number}_"
                f"Pedrun{self.ped_run_number}_"
                f"HHVrun{self.FF_SPE_HHV_run_number}.h5"
            )
            self.gain_tool = gain_cls(
                parent=self,
                run_number=self.FF_run_number,
                Ped_run_number=self.ped_run_number,
                SPE_result=self.SPE_HHV_result_path,
                output_path=self._gain_output_path,
            )

        # Setup HiLo tool
        hilo_cls = HILO_CALIBRATION_TOOLS[self.hilo_tool_name]
        self._hilo_output_path = self._gain_output_path.with_name(
            f"{self._gain_output_path.stem}"
            f"_hilo_corrected{self._gain_output_path.suffix}"
        )
        self.hilo_tool = hilo_cls(
            parent=self,
            run_number=self.FF_run_number,
            pedestal_file=self._ped_output_path,
            gain_file=self._gain_output_path,
            output_path=self._hilo_output_path,
        )

        # Setup flatfield tool
        flatfield_cls = FLATFIELD_CALIBRATION_TOOLS[self.flatfield_tool_name]
        self._FF_output_path = (
            self._res_dir_subtools
            / f"output_{self.flatfield_tool_name}_run{self.FF_run_number}.h5"
        )
        self.flatfield_tool = flatfield_cls(
            parent=self,
            run_number=self.FF_run_number,
            pedestal_file=self._ped_output_path,
            gain_file=self._hilo_output_path,
            output_path=self._FF_output_path,
        )

        # Fill the dictionary of subtool output paths
        # NOTE: the "gain" path should be the one with HiLo correction
        self._output_paths_subtools = {
            "pedestal": self._ped_output_path,
            "gain": self._hilo_output_path,
            "flatfield": self._FF_output_path,
        }

    def start(self):
        run_tool(self.pedestal_tool)
        run_tool(self.gain_tool)
        run_tool(self.hilo_tool)
        run_tool(self.flatfield_tool)

    def finish(self):
        self._read_containers_from_subtool_outputs()

    def _read_containers_from_subtool_outputs(self):
        for key in self._output_paths_subtools.keys():
            self._nectarcam_containers[key] = ContainerUtils.get_container_from_hdf5(
                self._output_paths_subtools[key],
                NECTARCAM_CONTAINER_CLASSES_DICT[key],
                group_names=GROUP_NAMES,
            )

    # def _add_missing_pixels(self):
    #     """
    #     Identifies NectarCAM containers with missing pixels due to hardware failure
    #     (e.g. an incomplete camera). The missing pixels are then padded with default
    #     values.
    #     """

    #     log.info("Checking for missing pixels in input data...")

    #     hardware_working_pixels = np.ones((N_GAINS, N_PIXELS), dtype=bool)

    #     for key, container in self._nectarcam_containers.items():
    #         # First identify missing pixels
    #         for ch in range(N_GAINS):
    #             hardware_working_pixels[ch] = np.logical_and(
    #                 hardware_working_pixels[ch],
    #                 np.isin(PIXEL_INDEX, container.pixels_id),
    #             )
    #         # Then add missing pixels_to_container
    #         ContainerUtils.add_missing_pixels_to_container(
    #             container, pad_value=self.default_values[key]
    #         )

    #     # Set the hardware failing pixels status in the pixel status container
    #     self.output_containers["pixel_status"].hardware_failing_pixels = (
    #         ~hardware_working_pixels
    #     )

    #     return
