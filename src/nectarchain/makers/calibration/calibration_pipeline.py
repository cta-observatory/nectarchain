import logging
import os
import pathlib

import astropy.units as u
import numpy as np
from ctapipe.containers import (
    FlatFieldContainer,
    PedestalContainer,
    PixelStatusContainer,
    WaveformCalibrationContainer,
)
from ctapipe.core import ToolConfigurationError, run_tool
from ctapipe.core.traits import Bool, CaselessStrEnum, Integer, Path
from ctapipe_io_nectarcam.constants import N_GAINS, N_PIXELS, N_SAMPLES

from ...data.container import FlatFieldContainer as NectarCAMFlatFieldContainer
from ...data.container import GainContainer as NectarCAMGainContainer
from ...data.container import (
    NectarCAMContainer,
    NectarCAMPedestalContainer,
    flatfield_container,
    gain_container,
    pedestal_container,
)
from ...utils.constants import PEDESTAL_DEFAULT
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

        self._fill_ctapipe_containers()

    def _read_containers_from_subtool_outputs(self):
        for key in self._output_paths_subtools.keys():
            self._nectarcam_containers[key] = ContainerUtils.get_container_from_hdf5(
                self._output_paths_subtools[key],
                NECTARCAM_CONTAINER_CLASSES_DICT[key],
                group_names=GROUP_NAMES,
            )
            ContainerUtils.add_missing_pixels_to_container(
                self._nectarcam_containers[key], pad_value=np.nan
            )

    def _fill_ctapipe_containers(self):
        """
        Fills all `ctapipe` containers to be written in the Category A calibration
        file from `nectarchain` containers.
        """

        self.log.info(f"Filling ctapipe containers: {self._ctapipe_containers}...")

        # Initialize hardware failing pixels here, they will be tagged in each
        # subfunction
        self._ctapipe_containers["pixel_status"].hardware_failing_pixels = np.zeros(
            (N_GAINS, N_PIXELS), dtype=bool
        )

        # Copy data from NectarCAMPedestalContainer to output containers
        self._copy_from_nectarcam_pedestal_container()

    def _copy_from_nectarcam_pedestal_container(self):
        """
        Copies calibration data from a `NectarCAMPedestalContainer` to the `ctapipe`
        containers to be written in the Category A calibration file.
        """

        self.log.info(
            f"Copying data from {self._nectarcam_containers['pedestal'].__name__} "
            f"to ctapipe containers..."
        )

        # Combine high gain and low gain pedestal arrays
        pedestal_mean_per_pixel_per_sample = self._combine_hg_and_lg(
            self._nectarcam_containers["pedestal"].pedestal_mean_hg,
            self._nectarcam_containers["pedestal"].pedestal_mean_lg,
        )
        pedestal_std_per_pixel_per_sample = self._combine_hg_and_lg(
            self._nectarcam_containers["pedestal"].pedestal_std_hg,
            self._nectarcam_containers["pedestal"].pedestal_std_lg,
        )

        # Update hardware failing pixels and add default values for fields of interest
        mask = np.logical_or(
            np.isnan(pedestal_mean_per_pixel_per_sample),
            pedestal_mean_per_pixel_per_sample == PEDESTAL_DEFAULT,
        )
        self._ctapipe_containers["pixel_status"].hardware_failing_pixels[
            mask[..., 0]
        ] = True
        pedestal_mean_per_pixel_per_sample[mask] = PEDESTAL_DEFAULT
        pedestal_std_per_pixel_per_sample[mask] = 0

        # Compute mean and std of pedestal per pixel
        pedestal_mean_per_pixel = self._get_pedestal_mean_per_pixel(
            pedestal_mean_per_pixel_per_sample
        )
        pedestal_std_per_pixel = self._get_pedestal_std_per_pixel(
            pedestal_std_per_pixel_per_sample,
        )

        # Set default pedestal values for bad pixels
        pedestal_mean_per_pixel_with_default = np.where(
            self._nectarcam_containers["pedestal"].pixel_mask,
            PEDESTAL_DEFAULT,
            pedestal_mean_per_pixel,
        )

        # Fill WaveformCalibrationContainer with pedestals
        self._ctapipe_containers[
            "calibration"
        ].pedestal_per_sample = pedestal_mean_per_pixel_with_default

        # Fill PedestalContainer
        # NOTE: normally in `ctapipe`, n_events is a float, here it's an array of shape
        # (`N_pixels`)
        self._ctapipe_containers["pedestal"].n_events = self._nectarcam_containers[
            "pedestal"
        ].nevents.astype(np.int64)
        self._ctapipe_containers["pedestal"].sample_time = (
            np.mean(
                [
                    self._nectarcam_containers["pedestal"].ucts_timestamp_max,
                    self._nectarcam_containers["pedestal"].ucts_timestamp_min,
                ]
            )
            * u.ns
        ).to(u.s)
        self._ctapipe_containers["pedestal"].sample_time_min = (
            self._nectarcam_containers["pedestal"].ucts_timestamp_min * u.ns
        ).to(u.s)
        self._ctapipe_containers["pedestal"].sample_time_max = (
            self._nectarcam_containers["pedestal"].ucts_timestamp_max * u.ns
        ).to(u.s)
        self._ctapipe_containers["pedestal"].charge_mean = pedestal_mean_per_pixel
        self._ctapipe_containers["pedestal"].charge_std = pedestal_std_per_pixel

        # Fill PixelStatusContainer with pedestal pixel status
        self._ctapipe_containers[
            "pixel_status"
        ].pedestal_failing_pixels = self._nectarcam_containers["pedestal"].pixel_mask

    @staticmethod
    def _combine_hg_and_lg(high_gain_array, low_gain_array):
        """Combines high-gain and low-gain arrays into one array."""

        combined_array = np.stack([high_gain_array, low_gain_array], axis=0)

        return combined_array

    @staticmethod
    def _get_pedestal_mean_per_pixel(pedestal_mean_per_pixel_per_sample):
        """Computes the mean pedestal per pixel."""

        pedestal_mean_per_pixel = np.mean(pedestal_mean_per_pixel_per_sample, axis=-1)

        return pedestal_mean_per_pixel

    @staticmethod
    def _get_pedestal_std_per_pixel(pedestal_std_per_pixel_per_sample):
        "Computes the std of a pedestal per pixel."

        pedestal_std_per_pixel = (
            np.sqrt(
                np.sum(
                    pedestal_std_per_pixel_per_sample**2,
                    axis=-1,
                )
            )
            / N_SAMPLES
        )

        return pedestal_std_per_pixel
