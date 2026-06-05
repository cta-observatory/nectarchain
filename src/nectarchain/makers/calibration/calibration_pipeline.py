import os
import pathlib
import shutil

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from ctapipe.containers import (
    FlatFieldContainer,
    PedestalContainer,
    PixelStatusContainer,
    WaveformCalibrationContainer,
)
from ctapipe.core import Provenance, ToolConfigurationError, run_tool
from ctapipe.core.traits import Bool, CaselessStrEnum, Integer, Path
from ctapipe.io import HDF5TableWriter
from ctapipe.io import metadata as meta
from ctapipe_io_nectarcam.constants import (
    LOW_GAIN,
    N_GAINS,
    N_PIXELS,
    N_SAMPLES,
    PIXEL_INDEX,
)

from ...data.container import FlatFieldContainer as NectarCAMFlatFieldContainer
from ...data.container import GainContainer as NectarCAMGainContainer
from ...data.container import (
    NectarCAMContainer,
    NectarCAMPedestalContainer,
    flatfield_container,
    gain_container,
    pedestal_container,
)
from ...utils.constants import (
    FLATFIELD_DEFAULT,
    GAIN_DEFAULT,
    HILO_DEFAULT,
    PEDESTAL_DEFAULT,
)
from ...utils.metadata import (
    add_metadata_to_hdu,
    get_ctapipe_metadata,
    get_local_metadata,
)
from ...utils.utils import ContainerUtils
from . import flatfield_makers, gain, pedestal_makers
from .core import NectarCAMCalibrationTool

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
PROV_OUTPUT_ROLES = {"create_calibration_file": "catA.r1.mon.tel.camera.calibration"}


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
        default_value=False,
        help=(
            "Option to save tmp subdirectory containing the individual outputs "
            "of each subtool",
        ),
    ).tag(config=True)
    all_default = Bool(
        default_value=False,
        help=(
            "Option to create a calibration file with only default "
            "calibration coefficients"
        ),
    )

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
        if self.all_default:
            calib_filename = f"{self.name}_DEFAULT_CALIB_VALUES{self.output_format}"
        else:
            calib_filename = (
                f"{self.name}_Pedrun{self.ped_run_number}_FFrun{self.FF_run_number}_"
                f"FFSPErun{self.FF_SPE_run_number}_"
                f"FFSPEHHVrun{self.FF_SPE_HHV_run_number}"
                f"{self.output_format}"
            )

        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/calib_pipeline/"
            f"{os.getpid()}/{calib_filename}"
        )

    def setup(self, *args, **kwargs):
        # Default run_number = -1 will raise Exception
        self.run_number = 0
        self.log.warning(f"Set run_number = {self.run_number} to avoid exception")

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
        if self.all_default:
            return
        else:
            run_tool(self.pedestal_tool)
            run_tool(self.gain_tool)
            run_tool(self.hilo_tool)
            run_tool(self.flatfield_tool)

    def finish(self):
        if self.all_default:
            self._fill_nectarcam_containers_with_default_values()
        else:
            self._read_nectarcam_containers_from_subtool_outputs()
        self._fill_ctapipe_containers_from_nectarcam_containers()
        self._write_catA_calibration_file()
        if not self.save_tmp:
            self.log.info(
                f"Removing temporary subtool result directory: {self._res_dir_subtools}"
            )
            shutil.rmtree(self._res_dir_subtools)

    def _read_nectarcam_containers_from_subtool_outputs(self):
        """
        Reads `nectarchain` containers from the different NectarCAM calibration tools.
        Missing pixels are padded with NaN values here.
        NOTE: Some of the subtools already pad `nectarchain` containers with default
        calibration values. This is taken into account later to identify hardware
        missing pixels.
        """

        self.log.info(f"Reading {NectarCAMContainer.__name__} outputs from subtools...")

        for key in self._output_paths_subtools.keys():
            self._nectarcam_containers[key] = ContainerUtils.get_container_from_hdf5(
                self._output_paths_subtools[key],
                NECTARCAM_CONTAINER_CLASSES_DICT[key],
                group_names=GROUP_NAMES,
            )
            ContainerUtils.add_missing_pixels_to_container(
                self._nectarcam_containers[key], pad_value=np.nan
            )

    def _fill_nectarcam_containers_with_default_values(self):
        """
        Construct nectarcam containers filled with default calibration coefficients.
        """

        self.log.info(
            f"Filling {NectarCAMContainer.__name__}s filled with default values..."
        )

        # Set pixels id (all pixels)
        for key in self._nectarcam_containers.keys():
            self._nectarcam_containers[key].pixels_id = PIXEL_INDEX

        # No bad pixels
        self._nectarcam_containers["pedestal"].pixel_mask = np.full(
            (N_GAINS, N_PIXELS), True, dtype=bool
        )
        self._nectarcam_containers["gain"].is_valid = np.full(
            (N_PIXELS), True, dtype=bool
        )
        self._nectarcam_containers["flatfield"].bad_pixels = None

        # Number of events (1 per pixel)
        self._nectarcam_containers["pedestal"].nevents = np.ones(N_PIXELS)
        self._nectarcam_containers["flatfield"].event_id = np.ones((1,))

        # FF charges (dummy value)
        self._nectarcam_containers["flatfield"].amp_int_per_pix_per_event = np.ones(
            (1, N_GAINS, N_PIXELS)
        )

        # Default timestamps (my birthday :)
        default_timestamp = 791054100000000000  # ns
        self._nectarcam_containers["pedestal"].ucts_timestamp_min = default_timestamp
        self._nectarcam_containers["pedestal"].ucts_timestamp_max = default_timestamp
        self._nectarcam_containers["flatfield"].ucts_timestamp = default_timestamp

        # Default pedestal values
        self._nectarcam_containers["pedestal"].pedestal_mean_hg = np.full(
            (N_PIXELS, N_SAMPLES), PEDESTAL_DEFAULT
        )
        self._nectarcam_containers["pedestal"].pedestal_mean_lg = np.full(
            (N_PIXELS, N_SAMPLES), PEDESTAL_DEFAULT
        )
        self._nectarcam_containers["pedestal"].pedestal_std_hg = np.zeros(
            (N_PIXELS, N_SAMPLES)
        )
        self._nectarcam_containers["pedestal"].pedestal_std_lg = np.zeros(
            (N_PIXELS, N_SAMPLES)
        )

        # Default gain values
        self._nectarcam_containers["gain"].high_gain = np.full(
            (N_PIXELS, 1), GAIN_DEFAULT
        )
        self._nectarcam_containers["gain"].low_gain = np.full(
            (N_PIXELS, 1), GAIN_DEFAULT / HILO_DEFAULT
        )

        # Default flatfield values
        self._nectarcam_containers["flatfield"].eff_coef = np.full(
            (1, N_GAINS, N_PIXELS), FLATFIELD_DEFAULT
        )

    def _fill_ctapipe_containers_from_nectarcam_containers(self):
        """
        Fills all `ctapipe` containers to be written in the Category A calibration
        file from `nectarchain` containers.
        """

        self.log.info(f"Filling ctapipe containers: {self._ctapipe_containers}...")

        # Identify unusable pixels
        self._set_unusable_pixels()

        # Set times in WaveformCalibrationContainer
        self._set_times_in_waveform_calibration_container()

        # Copy data from NectarCAMPedestalContainer to ctapipe containers
        self._copy_from_nectarcam_pedestal_container()

        # Copy data from NectarCAMGainContainer to ctapipe containers
        self._copy_from_nectarcam_gain_container()

        # Copy data from NectarCAMFlatfieldContainer to ctapipe containers
        self._copy_from_nectarcam_flatfield_container()

        # Set default values for bad pixels
        self._set_default_values_in_waveform_calibration_container()

    def _set_unusable_pixels(self):
        """
        Tags bad pixels identified by each `NectarCAMCalibrationTool` to write to
        the `ctapipe` containers.

        NOTE: Pixels tagged as bad during the gain computation are taken into account
        for the `unusable_pixels` field of the `WaveFormCalibrationContainer`.
        """

        # First identify hardware failing pixels
        self._set_hardware_failing_pixels()

        self.log.info("Identifying unusable pixels...")

        # Identify pixels tagged as bad during pedestal computation
        pedestal_failing_pixels = self._nectarcam_containers["pedestal"].pixel_mask

        # Identify pixels tagged as bad during gain computation
        gain_failing_pixels = np.logical_not(
            self._combine_hg_and_lg(
                self._nectarcam_containers["gain"].is_valid,
                self._nectarcam_containers["gain"].is_valid,
            )
        )

        # Identify pixels tagged as bad during flatfield computation
        flatfield_failing_pixels = np.tile(
            np.isin(
                self._nectarcam_containers["flatfield"].pixels_id,
                self._nectarcam_containers["flatfield"].bad_pixels,
            ),
            (N_GAINS, 1),
        )

        # Fill relevant ctapipe containers
        self._ctapipe_containers[
            "pixel_status"
        ].pedestal_failing_pixels = pedestal_failing_pixels
        self._ctapipe_containers[
            "pixel_status"
        ].flatfield_failing_pixels = flatfield_failing_pixels

        pixel_status_arrays = [
            self._ctapipe_containers["pixel_status"].hardware_failing_pixels,
            pedestal_failing_pixels,
            gain_failing_pixels,  # Take into account gain in overall status!
            flatfield_failing_pixels,
        ]
        self._ctapipe_containers["calibration"].unusable_pixels = np.logical_or.reduce(
            pixel_status_arrays
        )

    def _set_hardware_failing_pixels(self):
        """
        Tags hardware failing pixels from each `NectarCAMContainer` to write to
        the `PixelStatusContainer` of `ctapipe`. Generally these are missing pixels.

        NOTE: Since each `NectarCAMContainer` is expanded to the full camera,
        hardware failing pixels are tagged by either NaN or default calibration values.
        """

        self.log.info("Identifying hardware failing pixels...")

        hardware_failing_pixels = np.zeros((N_GAINS, N_PIXELS), dtype=bool)

        # Check for hardware failing pixels in pedestal container
        # Only need to check first sample of "pedestal waveform"
        pedestal = self._combine_hg_and_lg(
            self._nectarcam_containers["pedestal"].pedestal_mean_hg[..., 0],
            self._nectarcam_containers["pedestal"].pedestal_mean_lg[..., 0],
        )
        # These will be tagged by either NaN or default values
        mask_ped = np.logical_or(
            np.isnan(pedestal),
            pedestal == PEDESTAL_DEFAULT,
        )
        hardware_failing_pixels[mask_ped] = True

        # Check for hardware failing pixels in gain container
        # Only need to check high gain, since low gain is determined directly from that
        gain = self._nectarcam_containers["gain"].high_gain[..., 0]
        # These will be tagged by either NaN or default values
        mask_gain = np.logical_or(
            np.isnan(gain),
            gain == GAIN_DEFAULT,
        )
        hardware_failing_pixels[:, mask_gain] = True

        # Check for hardware failing pixels in FF container
        # Only need to check the first event
        eff_coef = self._nectarcam_containers["flatfield"].eff_coef[0]
        # These will be tagged by either NaN or default values
        mask_FF = np.logical_or(
            np.isnan(eff_coef),
            eff_coef == FLATFIELD_DEFAULT,
        )
        hardware_failing_pixels[mask_FF] = True

        # Fill relevant ctapipe container
        self._ctapipe_containers[
            "pixel_status"
        ].hardware_failing_pixels = hardware_failing_pixels

    def _set_times_in_waveform_calibration_container(self):
        """
        Sets the times for the output `WaveformCalibrationContainer`.
        The `time` field is assumed to be the mean of `time_min` and `time_max`.
        TODO: Take into account time of gain calibration. In LSTCAM this is done
        implictly with the same FF run, since they use the PhotoStatistic method.
        There is also no timing information currently in the `NectarCAMGainContainer`.
        """

        self.log.info("Identifying run times...")

        time_min_ped = (
            self._nectarcam_containers["pedestal"].ucts_timestamp_min * u.ns
        ).to(u.s)
        time_min_FF = (
            np.min(self._nectarcam_containers["flatfield"].ucts_timestamp, axis=0)
            * u.ns
        ).to(u.s)
        time_min = min(time_min_ped, time_min_FF)

        time_max_ped = (
            self._nectarcam_containers["pedestal"].ucts_timestamp_max * u.ns
        ).to(u.s)
        time_max_FF = (
            np.max(self._nectarcam_containers["flatfield"].ucts_timestamp, axis=0)
            * u.ns
        ).to(u.s)
        time_max = max(time_max_ped, time_max_FF)

        time = (time_min + time_max) / 2.0

        self._ctapipe_containers["calibration"].time_min = time_min
        self._ctapipe_containers["calibration"].time_max = time_max
        self._ctapipe_containers["calibration"].time = time

        # Set the run_start trait required for the metadata in the final calibration
        # file. Take the same time as the first event in the calibration run.
        self.run_start = Time(time_min, format="unix", scale="utc")

    def _copy_from_nectarcam_pedestal_container(self):
        """
        Copies calibration data from a `NectarCAMPedestalContainer` to the `ctapipe`
        containers to be written in the Category A calibration file.
        """

        self.log.info(
            "Copying data from "
            f"{self._nectarcam_containers['pedestal'].__class__.__name__} "
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

        # Compute mean and std of pedestal per pixel
        pedestal_mean_per_pixel = self._get_pedestal_mean_per_pixel(
            pedestal_mean_per_pixel_per_sample
        )
        pedestal_std_per_pixel = self._get_pedestal_std_per_pixel(
            pedestal_std_per_pixel_per_sample,
        )

        # Fill WaveformCalibrationContainer with pedestals
        self._ctapipe_containers[
            "calibration"
        ].pedestal_per_sample = pedestal_mean_per_pixel

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

    def _copy_from_nectarcam_gain_container(self):
        """
        Copies calibration data from a `NectarCAMGainContainer` to the `ctapipe`
        containers to be written in the Category A calibration file.
        """

        self.log.info(
            "Copying data from "
            f"{self._nectarcam_containers['gain'].__class__.__name__} "
            f"to ctapipe containers..."
        )

        # Combine high gain and low gain arrays
        gain_per_pixel = self._combine_hg_and_lg(
            self._nectarcam_containers["gain"].high_gain[..., 0],
            self._nectarcam_containers["gain"].low_gain[..., 0],
        )

        # Fill WaveformCalibrationContainer with gains
        self._ctapipe_containers["calibration"].n_pe = np.divide(
            1.0,
            gain_per_pixel,
            out=np.zeros_like(gain_per_pixel),
            where=gain_per_pixel != 0,
        )

    def _copy_from_nectarcam_flatfield_container(self):
        """
        Copies calibration data from a `NectarCAMFlatFieldContainer` to the `ctapipe`
        containers to be written in the Category A calibration file.
        """

        self.log.info(
            "Copying data from "
            f"{self._nectarcam_containers['flatfield'].__class__.__name__} "
            f"to ctapipe containers..."
        )

        eff_coef_per_pixel_per_event = self._nectarcam_containers[
            "flatfield"
        ].eff_coef.astype(np.float64)
        charge_per_pixel_per_event = self._nectarcam_containers[
            "flatfield"
        ].amp_int_per_pix_per_event.astype(np.float64)

        FF_pixel_mask = eff_coef_per_pixel_per_event == 0

        # Mask bad pixels for FF coefficient computations
        FF_coef_per_pixel_per_event_masked = np.ma.masked_array(
            1.0 / eff_coef_per_pixel_per_event,
            mask=np.broadcast_to(FF_pixel_mask, eff_coef_per_pixel_per_event.shape),
        )

        # Compute mean, median, std of FF coefficients, for bad pixels fill with 0
        FF_coef_per_pixel_mean = np.ma.mean(
            FF_coef_per_pixel_per_event_masked, axis=0
        ).filled(0)
        FF_coef_per_pixel_median = np.ma.median(
            FF_coef_per_pixel_per_event_masked, axis=0
        ).filled(0)
        FF_coef_per_pixel_std = np.ma.std(
            FF_coef_per_pixel_per_event_masked, axis=0
        ).filled(0)

        # Compute mean, median, std of charges
        charge_per_pixel_mean = np.mean(charge_per_pixel_per_event, axis=0)
        charge_per_pixel_median = np.median(charge_per_pixel_per_event, axis=0)
        charge_per_pixel_std = np.std(charge_per_pixel_per_event, axis=0)

        # Fill WaveformCalibrationContainer with FF corrections
        self._ctapipe_containers["calibration"].dc_to_pe = (
            FF_coef_per_pixel_mean * self._ctapipe_containers["calibration"].n_pe
        )

        # Fill FlatFieldContainer
        self._ctapipe_containers["flatfield"].sample_time = (
            np.mean(self._nectarcam_containers["flatfield"].ucts_timestamp) * u.ns
        ).to(u.s)
        self._ctapipe_containers["flatfield"].sample_time_min = (
            np.min(self._nectarcam_containers["flatfield"].ucts_timestamp) * u.ns
        ).to(u.s)
        self._ctapipe_containers["flatfield"].sample_time_max = (
            np.max(self._nectarcam_containers["flatfield"].ucts_timestamp) * u.ns
        ).to(u.s)
        self._ctapipe_containers["flatfield"].n_events = self._nectarcam_containers[
            "flatfield"
        ].event_id.shape[0]
        self._ctapipe_containers["flatfield"].charge_mean = charge_per_pixel_mean
        self._ctapipe_containers["flatfield"].charge_median = charge_per_pixel_median
        self._ctapipe_containers["flatfield"].charge_std = charge_per_pixel_std
        self._ctapipe_containers[
            "flatfield"
        ].relative_gain_mean = FF_coef_per_pixel_mean
        self._ctapipe_containers[
            "flatfield"
        ].relative_gain_median = FF_coef_per_pixel_median
        self._ctapipe_containers["flatfield"].relative_gain_std = FF_coef_per_pixel_std

    def _set_default_values_in_waveform_calibration_container(self):
        """
        Sets default values for unusable pixels in the `WaveformCalibrationContainer`.
        """

        self.log.info("Setting default calibration values for unusable pixels...")

        mask = self._ctapipe_containers["calibration"].unusable_pixels

        # Set default pedestal values
        self._ctapipe_containers["calibration"].pedestal_per_sample[
            mask
        ] = PEDESTAL_DEFAULT

        # Set default gain values and correct for HiLo ratio
        self._ctapipe_containers["calibration"].n_pe[mask] = 1 / GAIN_DEFAULT
        self._ctapipe_containers["calibration"].n_pe[LOW_GAIN] *= HILO_DEFAULT

        # Set default flatfield values
        self._ctapipe_containers["calibration"].dc_to_pe[mask] = (
            self._ctapipe_containers["calibration"].n_pe[mask] * FLATFIELD_DEFAULT
        )

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

    def _write_catA_calibration_file(self):
        """
        Writes output containers in Category A calibration file.
        Majority is copied from `lstcam_calib`.
        """
        self.log.info(f"Writing Cat-A calibration file at: {self.output_path}")

        # Get ctapipe metadata
        ctapipe_metadata = get_ctapipe_metadata("Cat-A pixel calibration coefficients")

        # Get local metadata
        local_metadata = get_local_metadata(
            self.tel_id,
            str(self.provenance_log.resolve()),
            self.run_start.iso,
        )

        # Write output file in hdf5 format
        if self.output_path.name.endswith(".h5"):
            with HDF5TableWriter(self.output_path) as writer:
                for key, container in self._ctapipe_containers.items():
                    writer.write(f"tel_{self.tel_id}/{key}", [container])

                # add metadata
                meta.write_to_hdf5(ctapipe_metadata.to_dict(), writer.h5file)
                meta.write_to_hdf5(local_metadata.as_dict(), writer.h5file)

        # Write output file in fits or fits.gz format
        elif self.output_path.name.endswith(".fits") or self.output_path.name.endswith(
            ".fits.gz"
        ):
            primary_hdu = fits.PrimaryHDU()
            add_metadata_to_hdu(ctapipe_metadata.to_dict(fits=True), primary_hdu)
            add_metadata_to_hdu(local_metadata.as_dict(), primary_hdu)

            hdul = fits.HDUList(primary_hdu)

            for key, container in self._ctapipe_containers.items():
                # Patch for Fields that are not filled like `time_correction` in the
                # WaveformCalibrationContainer -> replace None with np.nan
                container_dict = container.as_dict()
                for k, v in container_dict.items():
                    if v is None:
                        container_dict[k] = np.nan

                t = Table([container_dict])

                # Workaround for astropy#17930, attach missing units
                for col, value in self._ctapipe_containers[key].items():
                    if unit := getattr(value, "unit", None):
                        t[col].unit = unit

                hdul.append(fits.BinTableHDU(t, name=key))

            hdul.writeto(self.output_path, overwrite=self.overwrite)

        # Update provenance
        Provenance().add_output_file(
            self.output_path, role=PROV_OUTPUT_ROLES["create_calibration_file"]
        )
