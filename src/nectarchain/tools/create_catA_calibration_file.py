"""
Tool to create a category A calibration file.
Inspired by: `lstcam_calib.tools.create_calibration_file
<https://gitlab.cta-observatory.org/cta-array-elements/lst/analysis/lstcam_calib/-/blob/main/src/
lstcam_calib/tools/create_calibration_file.py>`.

Assumes that calibration of pedestal, gain, and flat field are performed by their
respective `NectarCAMCalibrationTool`. Each of these creates their own h5 file with
calibration factor. This tool takes those h5 files, and creates a dedicated output file.

NOTE: For now gain is assumed to be computed using the SPE fit method.

Possible TODO: in the full calibration pipeline, directly write the (CatA)
calibration file as the output. This would be more in the philosophy of `nectarchain`
and `ctapipe`.
"""

import logging

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
from ctapipe.core import Provenance, Tool, ToolConfigurationError, traits
from ctapipe.io import HDF5TableWriter
from ctapipe.io import metadata as meta
from ctapipe_io_nectarcam.constants import (
    HIGH_GAIN,
    LOW_GAIN,
    N_GAINS,
    N_PIXELS,
    N_SAMPLES,
    PIXEL_INDEX,
)

from nectarchain.data.container.flatfield_container import (
    FlatFieldContainer as NectarCAMFlatFieldContainer,
)
from nectarchain.data.container.gain_container import (
    GainContainer as NectarCAMGainContainer,
)
from nectarchain.data.container.gain_container import (
    SPEfitContainer as NectarCAMSPEfitContainer,
)
from nectarchain.data.container.pedestal_container import NectarCAMPedestalContainer
from nectarchain.utils.metadata import (
    add_metadata_to_hdu,
    get_ctapipe_metadata,
    get_local_metadata,
)

# Outputs allowed for Cat-A calibration file as done in `lstcam_calib`
OUTPUT_FORMATS = ["fits.gz", "fits", "h5"]

# Provenance output role as done in `lstcam_calib`
PROV_OUTPUT_ROLES = {"create_calibration_file": "catA.r1.mon.tel.camera.calibration"}

# Set logger
logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class CalibrationWriterNectarCAM(Tool):
    """
    Tool that generates a (h5 or fits) file with MST Cat-A NectarCAM calibration
    coefficients.
    """

    name = "CalibrationWriterNectarCAM"
    description = "Generate file with MST Cat-A NectarCAM calibration coefficients"

    pedestal_file = traits.Path(
        help="Path to h5 file with pedestal calibration coefficients",
    ).tag(config=True)

    gain_file = traits.Path(
        help="Path to h5 file with gain calibration coefficients",
    ).tag(config=True)

    spe = traits.Bool(
        default_value=False,
        help="Flag to specify if gain calibration was done with SPE fit method",
    ).tag(config=True)

    flatfield_file = traits.Path(
        help="Path to h5 file with flat-field calibration coefficients",
    ).tag(config=True)

    output_file = traits.Path(
        default_value="CatACalibrationFile.h5",
        directory_ok=False,
        help="Name of the output file (allowed format: fits, fits.gz or h5)",
    ).tag(config=True)

    aliases = {
        ("p", "pedestal-file"): "CalibrationWriterNectarCAM.pedestal_file",
        ("g", "gain-file"): "CalibrationWriterNectarCAM.gain_file",
        ("f", "flatfield-file"): "CalibrationWriterNectarCAM.flatfield_file",
        ("o", "output-file"): "CalibrationWriterNectarCAM.output_file",
        "spe": "CalibrationWriterNectarCAM.spe",
    }

    def __init__(self, **kwargs):
        """Initialize class and set some custom attributes."""
        super().__init__(**kwargs)

        self.flags.update(
            {
                "spe": (
                    {"CalibrationWriterNectarCAM": {"spe": True}},
                    "Use SPE fit method for gain calibration",
                ),
            }
        )

        self.input_containers = {
            "pedestal": NectarCAMPedestalContainer(),
            "gain": NectarCAMGainContainer(),
            "flatfield": NectarCAMFlatFieldContainer(),
        }

        self.output_containers = {
            "calibration": WaveformCalibrationContainer(),
            "flatfield": FlatFieldContainer(),
            "pedestal": PedestalContainer(),
            "pixel_status": PixelStatusContainer(),
        }

        self.input_files = {
            "pedestal": self.pedestal_file,
            "gain": self.gain_file,
            "flatfield": self.flatfield_file,
        }

        self.group_names = {
            "pedestal": "data_combined",
            "gain": "data",
            "flatfield": "data",
        }

    def setup(self):
        """Open input files and set up containers."""

        log.info("Setting up CalibrationWriterNectarCAM tool...")

        # Check input format
        for key in self.input_files:
            if not self.input_files[key].name.endswith("h5"):
                raise ToolConfigurationError(
                    f"Suffix of input file '{self.input_files[key].name}' not valid,"
                    f"must be h5"
                )

        # Check output format
        if not any(self.output_file.name.endswith(end) for end in OUTPUT_FORMATS):
            raise ToolConfigurationError(
                f"Suffix of output file '{self.output_file.name}' not valid, must be"
                f"one of {OUTPUT_FORMATS}"
            )

        # Change input gain container if SPE method was used for gain calibration
        if self.spe:
            self.input_containers["gain"] = NectarCAMSPEfitContainer

        # Load NectarCAM calibration data
        for key in self.input_containers.keys():
            self.input_containers[key] = next(
                self.input_containers[key].from_hdf5(
                    self.input_files[key], group_name=self.group_names[key]
                )
            )

        # Set run start time
        self.run_start = None

        # Set telescope id to dummy value
        # NOTE: needs to be updated!!
        self.tel_id = 0

        return

    def start(self):
        """
        Fill `ctapipe` containers for Category A calibration file from `nectarchain`
        containers.
        """
        # Zero-pad missing pixels that are missing from hardware
        self._add_missing_pixels()

        # Fill the output containers from the NectarCAM calibration containers
        self._fill_output_containers()

        # Update run start time
        self._set_run_start_time()

        return

    def _add_missing_pixels(self):
        """
        Zero-pads NectarCAM containers with missing pixels due to hardware failing
        pixels (e.g. an incomplete camera).
        """

        log.info("Checking for missing pixels in input data...")

        hardware_failing_pixels = np.ones((N_GAINS, N_PIXELS), dtype=bool)

        for container in self.input_containers.values():
            pixels_id = container.pixels_id

            for name, field in zip(container.keys(), container.values()):
                if isinstance(field, np.ndarray):
                    # Find pixel axis if there is one
                    pixel_axis = None
                    for i, dim in enumerate(field.shape):
                        if dim == len(pixels_id):
                            pixel_axis = i
                            break
                    if pixel_axis is None:
                        continue

                    # Reshape fields to fully pixelated camera with zero-padding
                    shape_new_field = list(field.shape)
                    shape_new_field[pixel_axis] = N_PIXELS
                    new_field = np.zeros(shape_new_field, dtype=field.dtype)

                    # Copy data in slices so that the correct axis is zero-padded
                    # Also sorts the arrays in terms of `PIXEL_INDEX`
                    pixel_pos = np.searchsorted(PIXEL_INDEX, pixels_id)
                    slc = [slice(None)] * field.ndim
                    slc[pixel_axis] = pixel_pos
                    new_field[tuple(slc)] = field

                    # Update the container
                    setattr(container, name, new_field)

            # Update the pixels_id with the full camera
            setattr(container, "pixels_id", PIXEL_INDEX)

            # If pixels are missing it's due to hardware, so update the pixel status
            for ch in range(N_GAINS):
                hardware_failing_pixels[ch] = np.logical_and(
                    hardware_failing_pixels[ch],
                    np.isin(PIXEL_INDEX, pixels_id),
                )

        # Set the hardware failing pixels status in the pixel status container
        self.output_containers[
            "pixel_status"
        ].hardware_failing_pixels = hardware_failing_pixels
        return

    def _fill_output_containers(self):
        """
        Fills all `ctapipe` containers to be written in the Category A calibration
        file from `nectarchain` containers.
        """

        log.info("Filling output containers from NectarCAM calibration data...")

        # Copy data from NectarCAMPedestalContainer to output containers
        self._copy_from_nectarcam_pedestal_container()

        # Copy data from GainContainer / SPEfitContainer (NectarCAM) to output
        # containers
        self._copy_from_nectarcam_gain_container()

        # Copy data from FlatFieldContainer (NectarCAM) to output containers
        self._copy_from_nectarcam_flatfield_container()

        # Set usable pixels in WaveformCalibrationContainer
        self._set_usable_pixels()

        # Set times in WaveformCalibrationContainer
        self._set_times()

        return

    def _copy_from_nectarcam_pedestal_container(self):
        """
        Copies calibration data from a `NectarCAMPedestalContainer` to the `ctapipe`
        containers to be written in the Category A calibration file.
        """

        log.info(f"Copying data from {type(self.input_containers['pedestal'])}...")

        # Combine high gain and low gain pedestal arrays
        pedestal_mean_per_pixel_per_sample = self._combine_hg_and_lg(
            self.input_containers["pedestal"].pedestal_mean_hg,
            self.input_containers["pedestal"].pedestal_mean_lg,
        )
        pedestal_std_per_pixel_per_sample = self._combine_hg_and_lg(
            self.input_containers["pedestal"].pedestal_std_hg,
            self.input_containers["pedestal"].pedestal_std_lg,
        )

        # Compute mean and std of pedestal per pixel
        pedestal_mean_per_pixel = self._get_pedestal_mean_per_pixel(
            pedestal_mean_per_pixel_per_sample
        )
        pedestal_std_per_pixel = self._get_pedestal_std_per_pixel(
            pedestal_std_per_pixel_per_sample,
        )

        # Fill WaveformCalibrationContainer with pedestals
        self.output_containers[
            "calibration"
        ].pedestal_per_sample = pedestal_mean_per_pixel

        # Fill PedestalContainer
        # NOTE: normally in `ctapipe`, n_events is a float, here it's an array of shape
        # (`N_pixels`)
        self.output_containers["pedestal"].n_events = self.input_containers[
            "pedestal"
        ].nevents
        self.output_containers["pedestal"].sample_time = (
            np.mean(
                [
                    self.input_containers["pedestal"].ucts_timestamp_max,
                    self.input_containers["pedestal"].ucts_timestamp_min,
                ]
            )
            * u.ns
        ).to(u.s)
        self.output_containers["pedestal"].sample_time_min = (
            self.input_containers["pedestal"].ucts_timestamp_min * u.ns
        ).to(u.s)
        self.output_containers["pedestal"].sample_time_max = (
            self.input_containers["pedestal"].ucts_timestamp_max * u.ns
        ).to(u.s)
        self.output_containers["pedestal"].charge_mean = pedestal_mean_per_pixel
        self.output_containers["pedestal"].charge_std = pedestal_std_per_pixel

        # Fill PixelStatusContainer with pedestal pixel status
        self.output_containers[
            "pixel_status"
        ].pedestal_failing_pixels = self.input_containers["pedestal"].pixel_mask

        return

    def _copy_from_nectarcam_gain_container(self):
        """
        Copies calibration data from a `NectarCAMGainContainer` or
        `NectarCAMSPEfitContainer` to the `ctapipe` containers to be written in the
        Category A calibration file.
        """

        log.info(f"Copying data from {type(self.input_containers['gain'])}...")

        # Combine high gain and low gain arrays
        gain_per_pixel = self._combine_hg_and_lg(
            self.input_containers["gain"].high_gain[..., 0],
            self.input_containers["gain"].low_gain[..., 0],
        )

        # NOTE: for now there is no HiLo correction applied and the gain is only
        # computed for the high gain channel. For now we make the assumption that the
        # HiLo correction factor is 13.1
        gain_per_pixel[LOW_GAIN] = gain_per_pixel[HIGH_GAIN] / 13.1

        # Fill WaveformCalibrationContainer with gains
        self.output_containers["calibration"].n_pe = np.divide(
            1.0,
            gain_per_pixel,
            out=np.zeros_like(gain_per_pixel),
            where=gain_per_pixel != 0,
        )

        # NOTE: there is no more information stored in output containers.

        return

    def _copy_from_nectarcam_flatfield_container(self):
        """
        Copies calibration data from a `NectarCAMFlatFieldContainer` to the `ctapipe`
        containers to be written in the Category A calibration file.
        """

        log.info(f"Copying data from {type(self.input_containers['flatfield'])}...")

        FF_coeff_per_pixel_per_event = self.input_containers["flatfield"].FF_coef
        charge_per_pixel_per_event = self.input_containers[
            "flatfield"
        ].amp_int_per_pix_per_event
        FF_pixel_mask = np.isin(
            self.input_containers["flatfield"].pixels_id,
            self.input_containers["flatfield"].bad_pixels[0],
        )

        # Mask bad pixels for FF coefficient computations
        FF_coeff_per_pixel_per_event_masked = np.ma.masked_array(
            FF_coeff_per_pixel_per_event,
            mask=np.broadcast_to(FF_pixel_mask, FF_coeff_per_pixel_per_event.shape),
        )

        # Compute mean, median, std of FF coefficients, for bad pixels fill with 0
        FF_coeff_per_pixel_mean = np.ma.mean(
            FF_coeff_per_pixel_per_event_masked, axis=0
        ).filled(0)
        FF_coeff_per_pixel_median = np.ma.median(
            FF_coeff_per_pixel_per_event_masked, axis=0
        ).filled(0)
        FF_coeff_per_pixel_std = np.ma.std(
            FF_coeff_per_pixel_per_event_masked, axis=0
        ).filled(0)

        # Compute mean, median, std of charges
        charge_per_pixel_mean = np.mean(charge_per_pixel_per_event, axis=0)
        charge_per_pixel_median = np.median(charge_per_pixel_per_event, axis=0)
        charge_per_pixel_std = np.std(charge_per_pixel_per_event, axis=0)

        # Expand dimensions of FF failing pixels to cover both high gain and low gain
        FF_failing_pixels = self._combine_hg_and_lg(FF_pixel_mask, FF_pixel_mask)

        # Fill WaveformCalibrationContainer with FF corrections
        self.output_containers["calibration"].dc_to_pe = (
            FF_coeff_per_pixel_mean * self.output_containers["calibration"].n_pe
        )

        # Fill FlatFieldContainer
        self.output_containers["flatfield"].sample_time = (
            np.mean(self.input_containers["flatfield"].ucts_timestamp) * u.ns
        ).to(u.s)
        self.output_containers["flatfield"].sample_time_min = (
            np.min(self.input_containers["flatfield"].ucts_timestamp) * u.ns
        ).to(u.s)
        self.output_containers["flatfield"].sample_time_max = (
            np.max(self.input_containers["flatfield"].ucts_timestamp) * u.ns
        ).to(u.s)
        self.output_containers["flatfield"].n_events = self.input_containers[
            "flatfield"
        ].event_id.shape[0]
        self.output_containers["flatfield"].charge_mean = charge_per_pixel_mean
        self.output_containers["flatfield"].charge_median = charge_per_pixel_median
        self.output_containers["flatfield"].charge_std = charge_per_pixel_std
        self.output_containers["flatfield"].relative_gain_mean = FF_coeff_per_pixel_mean
        self.output_containers[
            "flatfield"
        ].relative_gain_median = FF_coeff_per_pixel_median
        self.output_containers["flatfield"].relative_gain_std = FF_coeff_per_pixel_std

        # Fill PixelStatusContainer with pedestal pixel status
        self.output_containers[
            "pixel_status"
        ].flatfield_failing_pixels = FF_failing_pixels

        return

    def _set_usable_pixels(self):
        """Sets the pixel status for the output `WaveFormCalibrationContainer`."""

        log.info("Updating pixel status...")

        pixel_status_arrays = [
            self.output_containers["pixel_status"].hardware_failing_pixels,
            self.output_containers["pixel_status"].pedestal_failing_pixels,
            self.output_containers["pixel_status"].flatfield_failing_pixels,
        ]

        self.output_containers["calibration"].unusable_pixels = np.logical_and.reduce(
            pixel_status_arrays
        )

        return

    def _set_times(self):
        """
        Sets the times for the output `WaveformCalibrationContainer`.
        TODO: Take into account time of gain calibration. In LST this is one implictly
        with the same FF run. To be updated!
        """

        log.info("Updating times...")

        time_min = (
            np.min(
                [
                    self.output_containers["pedestal"].sample_time_min.to_value("s"),
                    self.output_containers["flatfield"].sample_time_min.to_value("s"),
                ]
            )
            * u.s
        )
        time_max = (
            np.max(
                [
                    self.output_containers["pedestal"].sample_time_max.to_value("s"),
                    self.output_containers["flatfield"].sample_time_max.to_value("s"),
                ]
            )
            * u.s
        )
        time = (time_min + time_max) / 2.0

        self.output_containers["calibration"].time_min = time_min
        self.output_containers["calibration"].time_max = time_max
        self.output_containers["calibration"].time = time

        return

    def _set_run_start_time(self):
        """
        Sets the run start time required for the metadata in the final calibration
        file. Take the same time as the first event in the calibration run.
        TODO: set the *actual* time of the run start.
        """

        self.run_start = Time(
            self.output_containers["calibration"].time_min, format="unix", scale="utc"
        )

        return

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

    def finish(self):
        """
        Writes output containers in Category A calibration file.
        Majority is copied from `lstcam_calib`.
        """
        log.info(f"Writing Cat-A calibration file at: {self.output_file}")

        # Get ctapipe metadata
        ctapipe_metadata = get_ctapipe_metadata("Cat-A pixel calibration coefficients")

        # Get local metadata
        local_metadata = get_local_metadata(
            self.tel_id,
            str(self.provenance_log.resolve()),
            self.run_start.iso,
        )

        # Write output file in hdf5 format
        if self.output_file.name.endswith(".h5"):
            with HDF5TableWriter(self.output_file) as writer:
                for key, container in self.output_containers.items():
                    writer.write(f"tel_{self.tel_id}/{key}", [container])

                # add metadata
                meta.write_to_hdf5(ctapipe_metadata.to_dict(), writer.h5file)
                meta.write_to_hdf5(local_metadata.as_dict(), writer.h5file)

        # Write output file in fits or fits.gz format
        elif self.output_file.name.endswith(".fits") or self.output_file.name.endswith(
            ".fits.gz"
        ):
            primary_hdu = fits.PrimaryHDU()
            add_metadata_to_hdu(ctapipe_metadata.to_dict(fits=True), primary_hdu)
            add_metadata_to_hdu(local_metadata.as_dict(), primary_hdu)

            hdul = fits.HDUList(primary_hdu)

            for key, container in self.output_containers.items():
                # Patch for Fields that are not filled like `time_correction` in the
                # WaveformCalibrationContainer -> replace None with np.nan
                container_dict = container.as_dict()
                for k, v in container_dict.items():
                    if v is None:
                        container_dict[k] = np.nan

                t = Table([container_dict])

                # Workaround for astropy#17930, attach missing units
                for col, value in self.output_containers[key].items():
                    if unit := getattr(value, "unit", None):
                        t[col].unit = unit

                hdul.append(fits.BinTableHDU(t, name=key))

            hdul.writeto(self.output_file, overwrite=self.overwrite)

        # Update provenance
        Provenance().add_output_file(
            self.output_file, role=PROV_OUTPUT_ROLES["create_calibration_file"]
        )

        return


def main():
    exe = CalibrationWriterNectarCAM()
    exe.run()


if __name__ == "__main__":
    main()
