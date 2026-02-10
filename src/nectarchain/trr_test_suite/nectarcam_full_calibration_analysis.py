"""
NectarCAM Full Calibration and Analysis Pipeline

This script performs a complete calibration pipeline including:
1. Pedestal computation
2. Gain (SPE fit) computation
3. Flatfield computation
4. Charge extraction
5. Calibrated charge computation (pedestal subtraction, gain correction, FF correction)
6. Plotting of all calibration parameters vs temperature

Author: Generated for NectarCAM calibration analysis
Date: 2026-02-10
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ctapipe_io_nectarcam import constants

# NectarChain imports
from nectarchain.data.container import ChargesContainer
from nectarchain.makers import (
    ChargesNectarCAMCalibrationTool,
    WaveformsNectarCAMCalibrationTool,
)
from nectarchain.makers.calibration import (
    FlatfieldNectarCAMCalibrationTool,
    FlatFieldSPEHHVNectarCAMCalibrationTool,
    FlatFieldSPEHHVStdNectarCAMCalibrationTool,
    FlatFieldSPENominalNectarCAMCalibrationTool,
    FlatFieldSPENominalStdNectarCAMCalibrationTool,
    PedestalNectarCAMCalibrationTool,
)
from nectarchain.makers.extractor.utils import CtapipeExtractor

# Argument parser
parser = argparse.ArgumentParser(
    prog="nectarcam_full_calibration_analysis.py",
    description="Complete NectarCAM calibration pipeline",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Run numbers for different calibration types
parser.add_argument(
    "--pedestal_runs",
    nargs="+",
    required=True,
    help="Run number(s) for pedestal calibration",
    type=int,
)
parser.add_argument(
    "--gain_runs",
    nargs="+",
    required=True,
    help="Run number(s) for gain (SPE) calibration",
    type=int,
)
parser.add_argument(
    "--flatfield_runs",
    nargs="+",
    required=True,
    help="Run number(s) for flatfield calibration",
    type=int,
)
parser.add_argument(
    "--charge_runs",
    nargs="+",
    required=True,
    help="Run number(s) for charge extraction",
    type=int,
)

# Temperature data
parser.add_argument(
    "--temperatures",
    nargs="+",
    required=True,
    help="Temperature values corresponding to charge runs (in same order)",
    type=float,
)

# Camera selection
parser.add_argument(
    "-c",
    "--camera",
    default="NectarCAM",
    help="Process data for a specific NectarCAM camera (default: NectarCAM)",
    type=str,
)

# Max events
parser.add_argument(
    "--max_events_pedestal",
    default=None,
    help="Max events for pedestal runs",
    type=int,
)
parser.add_argument(
    "--max_events_gain",
    nargs="+",
    default=None,
    help="Max events for gain runs (1 value/run, or a 1 value for all)",
    type=int,
)
parser.add_argument(
    "--max_events_flatfield",
    default=10000,
    help="Max events for flatfield runs",
    type=int,
)
parser.add_argument(
    "--max_events_charge",
    default=None,
    help="Max events for charge runs",
    type=int,
)

# Bad pixels
parser.add_argument(
    "--bad_pixels",
    nargs="+",
    default=None,
    help="List of bad pixel IDs to exclude from analysis",
    type=int,
)
parser.add_argument(
    "--use_bad_pixels",
    action="store_true",
    default=False,
    help="Apply bad pixel masking in the analysis",
)

# Processing options
parser.add_argument(
    "--recompute_pedestal",
    action="store_true",
    default=True,
    help="Force recomputation of pedestal calibration",
)
parser.add_argument(
    "--recompute_gain",
    action="store_true",
    default=True,
    help="Force recomputation of gain calibration",
)
parser.add_argument(
    "--recompute_flatfield",
    action="store_true",
    default=True,
    help="Force recomputation of flatfield calibration",
)
parser.add_argument(
    "--recompute_charge",
    action="store_true",
    default=True,
    help="Force recomputation of charge extraction",
)
parser.add_argument(
    "--recompute_all",
    action="store_true",
    default=True,
    help="Force recomputation of all calibrations",
)

# Gain-specific options
parser.add_argument(
    "--HHV",
    action="store_true",
    default=False,
    help="Gain runs taken at high voltage (HHV)",
)
parser.add_argument(
    "--free_pp_n",
    action="store_true",
    default=False,
    help="Let pp and n parameters free in SPE fit",
)
parser.add_argument(
    "--gain_display",
    action="store_true",
    default=False,
    help="Display SPE histograms for each pixel during gain computation",
)
parser.add_argument(
    "--gain_asked_pixels_id",
    nargs="+",
    default=None,
    help="Pixel IDs to process during gain computation (default: all pixels)",
    type=int,
)
parser.add_argument(
    "--gain_reload_events",
    action="store_true",
    default=False,
    help="Force re-computation of waveforms from fits.fz files for gain runs",
)
parser.add_argument(
    "--gain_overwrite",
    action="store_true",
    default=False,
    help="Force overwrite of existing gain output files on disk",
)
parser.add_argument(
    "--gain_events_per_slice",
    type=int,
    default=None,
    help="Split raw gain data with this many events per slice",
)
parser.add_argument(
    "--gain_multiproc",
    action="store_true",
    default=False,
    help="Use multiprocessing for gain computation",
)
parser.add_argument(
    "--gain_nproc",
    type=int,
    default=8,
    help="Number of processes to use when --gain_multiproc is set",
)
parser.add_argument(
    "--gain_chunksize",
    type=int,
    default=1,
    help="Chunk size per process when --gain_multiproc is set",
)

# Pedestal-specific options
parser.add_argument(
    "--events_per_slice",
    type=int,
    default=300,
    help="Events per slice for pedestal computation",
)
parser.add_argument(
    "--filter_method",
    type=str,
    default="WaveformsStdFilter",
    help="Filter method for pedestal computation",
)
parser.add_argument(
    "--wfs_std_threshold",
    type=float,
    default=4.0,
    help="Waveform std threshold for pedestal filtering",
)

# Flatfield-specific options
parser.add_argument(
    "--flatfield_window_width",
    type=int,
    default=12,
    help="Window width for flatfield charge extraction",
)
parser.add_argument(
    "--flatfield_window_shift",
    type=int,
    default=4,
    help="Window shift for flatfield charge extraction",
)

# Charge extraction options
parser.add_argument(
    "--charge_method",
    choices=[
        "FullWaveformSum",
        "FixedWindowSum",
        "GlobalPeakWindowSum",
        "LocalPeakWindowSum",
        "SlidingWindowMaxSum",
        "TwoPassWindowSum",
    ],
    default="LocalPeakWindowSum",
    help="Charge extractor method",
    type=str,
)
parser.add_argument(
    "--charge_extractor_kwargs",
    default='{"window_width": 16, "window_shift": 4}',
    help="Charge extractor kwargs (JSON format)",
    type=str,
)

# Output options
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Output directory for calibration files (default: $NECTARCAMDATA)",
)
parser.add_argument(
    "--figure_dir",
    type=str,
    default=None,
    help="Output directory for figures (default: $NECTARCHAIN_FIGURES)",
)

# Verbosity
parser.add_argument(
    "-v",
    "--verbosity",
    help="Set the verbosity level of logger",
    default="INFO",
    choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
    type=str,
)


class NectarCAMCalibrationPipeline:
    """Complete calibration pipeline for NectarCAM data"""

    def __init__(self, args, log):
        self.args = args
        self.log = log

        # Setup directories
        self.data_dir = Path(args.output_dir or os.environ.get("NECTARCAMDATA", "/tmp"))
        self.figure_dir = Path(
            args.figure_dir or os.environ.get("NECTARCHAIN_FIGURES", "/tmp")
        )

        # Create subdirectories
        self.pedestal_dir = self.data_dir / "pedestals"
        self.gain_dir = self.data_dir / "gains"
        self.flatfield_dir = self.data_dir / "flatfields"
        self.charge_dir = self.data_dir / "charges"
        self.calibrated_dir = self.data_dir / "calibrated_charges"

        for d in [
            self.pedestal_dir,
            self.gain_dir,
            self.flatfield_dir,
            self.charge_dir,
            self.calibrated_dir,
            self.figure_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Parse extractor kwargs
        self.extractor_kwargs = json.loads(args.charge_extractor_kwargs)

        # Store results
        self.pedestal_results = {}
        self.gain_results = {}
        self.flatfield_results = {}
        self.charge_results = {}
        self.calibrated_charge_results = {}

        self.log.info(f"Data directory: {self.data_dir}")
        self.log.info(f"Figure directory: {self.figure_dir}")

    def get_pedestal_output_path(self, run_number):
        """Get output path for pedestal calibration"""
        return self.pedestal_dir / f"pedestal_{run_number}.h5"

    def get_gain_output_path(self, run_number):
        """Get output path for gain calibration"""
        method_str = CtapipeExtractor.get_extractor_kwargs_str(
            self.args.charge_method, self.extractor_kwargs
        )
        return self.gain_dir / f"gain_SPE_{run_number}_{method_str}.h5"

    def get_flatfield_output_path(self, run_number, iteration=2):
        """Get output path for flatfield calibration"""
        return self.flatfield_dir / f"{iteration}FF_{run_number}.h5"

    def get_charge_output_path(self, run_number):
        """Get output path for charge extraction"""
        method_str = CtapipeExtractor.get_extractor_kwargs_str(
            self.args.charge_method, self.extractor_kwargs
        )
        return self.charge_dir / f"charges_{run_number}_{method_str}.h5"

    def compute_pedestal(self, run_number):
        """Compute pedestal calibration for a run"""
        output_path = self.get_pedestal_output_path(run_number)

        if output_path.exists() and not (
            self.args.recompute_pedestal or self.args.recompute_all
        ):
            self.log.info(f"Pedestal for run {run_number} already exists, loading...")
            self.pedestal_results[run_number] = output_path
            return output_path

        self.log.info(f"Computing pedestal for run {run_number}...")

        tool = PedestalNectarCAMCalibrationTool(
            progress_bar=True,
            camera=self.args.camera,
            run_number=run_number,
            max_events=self.args.max_events_pedestal,
            events_per_slice=self.args.events_per_slice,
            log_level=logging.getLevelName(self.log.level),
            output_path=str(output_path),
            overwrite=True,
            filter_method=self.args.filter_method,
            wfs_std_threshold=self.args.wfs_std_threshold,
        )

        tool.initialize()
        tool.setup()
        tool.start()
        tool.finish(return_output_component=True)
        logging.info(f"Pedestal computation completed for run {run_number}")
        logging.info(f"Pedestal saved to {output_path}")

        self.pedestal_results[run_number] = output_path
        self.log.info(f"Pedestal saved to {output_path}")

        return output_path

    def _build_gain_kwargs(self):
        """
        Build the **kwargs dict passed to the SPE tool, mirroring the pattern
        used in gain_SPEfit_computation.py.  Only gain-relevant keys are
        included so that unrelated pipeline arguments are never forwarded.
        """
        kwargs = dict(
            method=self.args.charge_method,
            extractor_kwargs=self.extractor_kwargs,
            log_level=logging.getLevelName(self.log.level),
            overwrite=self.args.gain_overwrite,
            reload_events=self.args.gain_reload_events,
            display_toggle=self.args.gain_display,
            multiproc=self.args.gain_multiproc,
            nproc=self.args.gain_nproc,
            chunksize=self.args.gain_chunksize,
        )
        # Optional keys – only add when a value was explicitly provided so we
        # do not override tool defaults with None.
        if self.args.gain_asked_pixels_id is not None:
            kwargs["asked_pixels_id"] = self.args.gain_asked_pixels_id
        if self.args.gain_events_per_slice is not None:
            kwargs["events_per_slice"] = self.args.gain_events_per_slice
        return kwargs

    def compute_gain(self, run_number, max_events=None):
        """
        Compute gain calibration using SPE fit for a single run.

        Parameters
        ----------
        run_number : int
            Run number to process.
        max_events : int or None
            Maximum number of events to load (mirrors the per-run
            ``max_events`` list logic of gain_SPEfit_computation.py).
        """
        output_path = self.get_gain_output_path(run_number)

        if output_path.exists() and not (
            self.args.recompute_gain or self.args.recompute_all
        ):
            self.log.info(f"Gain for run {run_number} already exists, loading...")
            self.gain_results[run_number] = output_path
            return output_path

        self.log.info(f"Computing gain (SPE fit) for run {run_number}...")

        # Select the appropriate tool class (same logic as reference script)
        if self.args.HHV:
            tool_class = (
                FlatFieldSPEHHVNectarCAMCalibrationTool
                if self.args.free_pp_n
                else FlatFieldSPEHHVStdNectarCAMCalibrationTool
            )
        else:
            tool_class = (
                FlatFieldSPENominalNectarCAMCalibrationTool
                if self.args.free_pp_n
                else FlatFieldSPENominalStdNectarCAMCalibrationTool
            )

        # Build kwargs the same way the reference script does, then pass
        # run-level positional info separately (camera, run_number,
        # max_events, progress_bar) so they are never duplicated in kwargs.
        gain_kwargs = self._build_gain_kwargs()

        try:
            tool = tool_class(
                progress_bar=True,
                camera=self.args.camera,
                run_number=run_number,
                max_events=max_events,
                **gain_kwargs,
            )
            tool.setup()

            # Build figpath the same way as in the reference script
            extractor_kwargs_str = CtapipeExtractor.get_extractor_kwargs_str(
                tool.method,
                tool.extractor_kwargs,
            )

            base_figpath = (
                f"{self.figure_dir}/{tool.name}"
                f"_run{tool.run_number}"
                f"_{tool.method}"
                f"_{extractor_kwargs_str}"
            )

            if gain_kwargs.get("reload_events") and max_events is not None:
                figpath = f"{base_figpath}_maxevents{max_events}"
            else:
                figpath = base_figpath

            tool.start(figpath=figpath)
            tool.finish(figpath=figpath)

        except Exception as e:
            self.log.warning(
                f"Gain computation failed for run {run_number}: {e}", exc_info=True
            )
            raise

        self.gain_results[run_number] = output_path
        self.log.info(f"Gain computation completed for run {run_number}")

        return output_path

    def get_gain_from_flatfield(self, ff_output):
        """Calculate gain from flatfield output"""
        amp_int_per_pix_per_event = ff_output.amp_int_per_pix_per_event[:, :, :]
        amp_int_per_pix_mean = np.mean(amp_int_per_pix_per_event, axis=0)
        amp_int_per_pix_var = np.var(amp_int_per_pix_per_event, axis=0)
        gain = np.divide(
            amp_int_per_pix_var, amp_int_per_pix_mean, where=amp_int_per_pix_mean != 0.0
        )
        return gain

    def get_hi_lo_ratio(self, ff_output):
        """Calculate high gain to low gain ratio"""
        gain = self.get_gain_from_flatfield(ff_output)
        hi_lo_ratio = gain[constants.HIGH_GAIN] / gain[constants.LOW_GAIN]
        return hi_lo_ratio

    def compute_flatfield(self, run_number):
        """Compute flatfield calibration for a run (two-pass method)"""
        output_path_final = self.get_flatfield_output_path(run_number, iteration=2)

        if output_path_final.exists() and not (
            self.args.recompute_flatfield or self.args.recompute_all
        ):
            self.log.info(f"Flatfield for run {run_number} already exists, loading...")
            self.flatfield_results[run_number] = output_path_final
            return output_path_final

        self.log.info(f"Computing flatfield for run {run_number} (two-pass method)...")

        # First pass: default gain and hi/lo values
        self.log.info("First pass with default gain and hi/lo values...")

        gain_default = 58.0
        hi_lo_ratio_default = 13.0
        gain_array = np.ones(shape=(constants.N_GAINS, constants.N_PIXELS))
        gain_array[0] = gain_array[0] * gain_default
        gain_array[1] = gain_array[1] * gain_default / hi_lo_ratio_default

        bad_pixels_array = []

        output_path_1 = self.get_flatfield_output_path(run_number, iteration=1)

        tool = FlatfieldNectarCAMCalibrationTool(
            progress_bar=True,
            camera=self.args.camera,
            run_number=run_number,
            max_events=self.args.max_events_flatfield,
            log_level=logging.getLevelName(self.log.level),
            charge_extraction_method=None,
            charge_integration_correction=False,
            window_width=self.args.flatfield_window_width,
            window_shift=self.args.flatfield_window_shift,
            overwrite=True,
            gain=gain_array.tolist(),
            bad_pix=bad_pixels_array,
            output_path=str(output_path_1),
        )

        tool.initialize()
        tool.setup()
        tool.start()
        ff_output_1 = tool.finish(return_output_component=True)[0]

        self.log.info(f"First pass completed, intermediate file: {output_path_1}")

        # Second pass: updated gain and hi/lo values
        self.log.info("Second pass with updated gain and hi/lo values...")

        updated_gain = self.get_gain_from_flatfield(ff_output_1)

        tool = FlatfieldNectarCAMCalibrationTool(
            progress_bar=True,
            camera=self.args.camera,
            run_number=run_number,
            max_events=self.args.max_events_flatfield,
            log_level=logging.getLevelName(self.log.level),
            charge_extraction_method=None,
            charge_integration_correction=False,
            window_width=self.args.flatfield_window_width,
            window_shift=self.args.flatfield_window_shift,
            overwrite=True,
            gain=updated_gain.tolist(),
            bad_pix=bad_pixels_array,
            output_path=str(output_path_final),
        )

        tool.initialize()
        tool.setup()
        tool.start()
        tool.finish(return_output_component=True)[0]

        self.flatfield_results[run_number] = output_path_final
        self.log.info(f"Flatfield saved to {output_path_final}")

        return output_path_final

    def compute_charge(self, run_number):
        """Compute charge extraction for a run"""
        output_path = self.get_charge_output_path(run_number)

        if output_path.exists() and not (
            self.args.recompute_charge or self.args.recompute_all
        ):
            self.log.info(f"Charges for run {run_number} already exists, loading...")
            container = ChargesContainer.from_hdf5(output_path)
            self.charge_results[run_number] = container
            return container

        self.log.info(f"Computing charges for run {run_number}...")

        # First check if waveforms exist
        wfs_tool = WaveformsNectarCAMCalibrationTool(
            progress_bar=True,
            camera=self.args.camera,
            run_number=run_number,
            max_events=self.args.max_events_charge,
            log_level=logging.getLevelName(self.log.level),
            overwrite=self.args.recompute_charge or self.args.recompute_all,
        )

        try:
            wfs_tool.setup()
            wfs_tool.start()
            wfs_tool.finish()
            self.log.info(f"Waveforms extracted for run {run_number}")
        except Exception as e:
            self.log.error(f"Error extracting waveforms: {e}", exc_info=True)
            raise

        # Now compute charges
        charge_tool = ChargesNectarCAMCalibrationTool(
            progress_bar=True,
            camera=self.args.camera,
            run_number=run_number,
            max_events=self.args.max_events_charge,
            from_computed_waveforms=True,
            method=self.args.charge_method,
            extractor_kwargs=self.extractor_kwargs,
            log_level=logging.getLevelName(self.log.level),
            overwrite=True,
        )

        charge_tool.setup()
        charge_tool.start()
        output = charge_tool.finish()

        self.charge_results[run_number] = output
        self.log.info(f"Charges saved to {output_path}")

        return output

    def compute_calibrated_charge(
        self, charge_run, pedestal_run, gain_run, flatfield_run
    ):
        """
        Compute calibrated charge by applying:
        1. Pedestal subtraction
        2. Gain correction (division)
        3. Flatfield correction (multiplication)
        """
        import h5py

        self.log.info(f"Computing calibrated charge for run {charge_run}...")

        # Make sure all calibration data exists
        if charge_run not in self.charge_results:
            self.compute_charge(charge_run)
        if pedestal_run not in self.pedestal_results:
            self.compute_pedestal(pedestal_run)
        if gain_run not in self.gain_results:
            self.compute_gain(gain_run)
        if flatfield_run not in self.flatfield_results:
            self.compute_flatfield(flatfield_run)

        # Load charge data
        charge_path = (
            self.charge_results[charge_run]
            if isinstance(self.charge_results[charge_run], (str, Path))
            else self.get_charge_output_path(charge_run)
        )
        try:
            with h5py.File(charge_path, "r") as f:
                if "data/ChargesContainer/charges_hg" in f:
                    charges_hg = f["data/ChargesContainer/charges_hg"][:]
                    charges_lg = f["data/ChargesContainer/charges_lg"][:]
                    charges = np.stack([charges_hg, charges_lg], axis=1)
                elif "charges" in f:
                    charges = f["charges"][:]
                else:
                    self.log.error(f"Available keys in charge file: {list(f.keys())}")
                    for key in f.keys():
                        content = (
                            list(f[key].keys())
                            if hasattr(f[key], "keys")
                            else "dataset"
                        )
                        self.log.error("  %s: %s", key, content)
                    raise KeyError("Cannot find charge data in HDF5 file")
        except Exception as e:
            self.log.error(f"Error loading charge data: {e}", exc_info=True)
            return None

        # Load pedestal data
        pedestal_path = (
            self.pedestal_results[pedestal_run]
            if isinstance(self.pedestal_results[pedestal_run], (str, Path))
            else self.get_pedestal_output_path(pedestal_run)
        )
        try:
            with h5py.File(pedestal_path, "r") as f:
                if "pedestal_mean_hg" in f:
                    ped_hg = f["pedestal_mean_hg"][:]
                    ped_lg = f["pedestal_mean_lg"][:]
                    pedestals = np.stack([ped_hg, ped_lg], axis=0)
                elif "pedestal" in f:
                    pedestals = f["pedestal"][:]
                else:
                    self.log.error(f"Available keys in pedestal file: {list(f.keys())}")
                    raise KeyError("Cannot find pedestal data in HDF5 file")
        except Exception as e:
            self.log.error(f"Error loading pedestal data: {e}", exc_info=True)
            return None

        # Load gain data
        gain_path = (
            self.gain_results[gain_run]
            if isinstance(self.gain_results[gain_run], (str, Path))
            else self.get_gain_output_path(gain_run)
        )
        try:
            with h5py.File(gain_path, "r") as f:
                if "gain" in f:
                    gains = f["gain"][:]
                elif "high_gain" in f:
                    gains = np.stack([f["high_gain"][:], f["low_gain"][:]], axis=0)
                else:
                    self.log.warning(
                        "Cannot find gain data, using flatfield-derived gain"
                    )
                    flatfield_path = (
                        self.flatfield_results[flatfield_run]
                        if isinstance(
                            self.flatfield_results[flatfield_run], (str, Path)
                        )
                        else self.get_flatfield_output_path(flatfield_run)
                    )
                    with h5py.File(flatfield_path, "r") as ff_file:
                        if "amp_int_per_pix_per_event" in ff_file:
                            amp_data = ff_file["amp_int_per_pix_per_event"][:]
                            amp_mean = np.mean(amp_data, axis=0)
                            amp_var = np.var(amp_data, axis=0)
                            gains = np.divide(amp_var, amp_mean, where=amp_mean != 0.0)
                        else:
                            raise KeyError("Cannot compute gain from flatfield")
        except Exception as e:
            self.log.error(f"Error loading gain data: {e}", exc_info=True)
            return None

        # Load flatfield data
        flatfield_path = (
            self.flatfield_results[flatfield_run]
            if isinstance(self.flatfield_results[flatfield_run], (str, Path))
            else self.get_flatfield_output_path(flatfield_run)
        )
        try:
            with h5py.File(flatfield_path, "r") as f:
                if "FF_coef" in f:
                    ff_coef = np.mean(f["FF_coef"][:], axis=0)
                elif "flatfield" in f:
                    ff_coef = f["flatfield"][:]
                else:
                    self.log.error(
                        f"Available keys in flatfield file: {list(f.keys())}"
                    )
                    raise KeyError("Cannot find flatfield data in HDF5 file")
        except Exception as e:
            self.log.error(f"Error loading flatfield data: {e}", exc_info=True)
            return None

        # Ensure proper shapes
        if charges.ndim == 3:
            if charges.shape[0] == 2:
                # Shape is (gains, events, pixels)
                # – transpose to (events, gains, pixels)
                charges = np.transpose(charges, (1, 0, 2))

        # Apply calibration: (charge - pedestal) / gain * FF
        calibrated_charges = np.zeros_like(charges)

        for gain_idx in range(min(charges.shape[1], 2)):
            for pixel_idx in range(charges.shape[2]):
                ped_subtracted = (
                    charges[:, gain_idx, pixel_idx] - pedestals[gain_idx, pixel_idx]
                )
                if gains[gain_idx, pixel_idx] != 0:
                    gain_corrected = ped_subtracted / gains[gain_idx, pixel_idx]
                else:
                    gain_corrected = ped_subtracted
                calibrated_charges[:, gain_idx, pixel_idx] = (
                    gain_corrected * ff_coef[gain_idx, pixel_idx]
                )

        # Apply bad pixel mask if requested
        if self.args.use_bad_pixels and self.args.bad_pixels:
            for bad_pix in self.args.bad_pixels:
                if bad_pix < calibrated_charges.shape[2]:
                    calibrated_charges[:, :, bad_pix] = np.nan

        self.calibrated_charge_results[charge_run] = {
            "calibrated_charges": calibrated_charges,
            "raw_charges": charges,
            "pedestals": pedestals,
            "gains": gains,
            "flatfield": ff_coef,
        }

        self.log.info(f"Calibrated charge computed for run {charge_run}")

        return calibrated_charges

    def plot_calibration_vs_temperature(self):
        """Plot all calibration parameters and charges vs temperature"""
        self.log.info("Creating calibration vs temperature plots...")

        if len(self.args.charge_runs) != len(self.args.temperatures):
            self.log.error("Number of charge runs must match number of temperatures")
            return

        temperatures = np.array(self.args.temperatures)
        n_temps = len(temperatures)
        n_pixels = constants.N_PIXELS
        gain_idx = constants.HIGH_GAIN

        pedestals_vs_temp = np.zeros((n_temps, n_pixels))
        gains_vs_temp = np.zeros((n_temps, n_pixels))
        ff_vs_temp = np.zeros((n_temps, n_pixels))
        raw_charge_mean_vs_temp = np.zeros((n_temps, n_pixels))
        calib_charge_mean_vs_temp = np.zeros((n_temps, n_pixels))

        for idx, (charge_run, temp) in enumerate(
            zip(self.args.charge_runs, temperatures)
        ):
            if charge_run in self.calibrated_charge_results:
                result = self.calibrated_charge_results[charge_run]
                pedestals_vs_temp[idx] = result["pedestals"][gain_idx]
                gains_vs_temp[idx] = result["gains"][gain_idx]
                ff_vs_temp[idx] = result["flatfield"][gain_idx]
                raw_charge_mean_vs_temp[idx] = np.nanmean(
                    result["raw_charges"][:, gain_idx, :], axis=0
                )
                calib_charge_mean_vs_temp[idx] = np.nanmean(
                    result["calibrated_charges"][:, gain_idx, :], axis=0
                )

        if self.args.use_bad_pixels and self.args.bad_pixels:
            for bad_pix in self.args.bad_pixels:
                if bad_pix < n_pixels:
                    pedestals_vs_temp[:, bad_pix] = np.nan
                    gains_vs_temp[:, bad_pix] = np.nan
                    ff_vs_temp[:, bad_pix] = np.nan
                    raw_charge_mean_vs_temp[:, bad_pix] = np.nan
                    calib_charge_mean_vs_temp[:, bad_pix] = np.nan

        def _is_bad(pix):
            return (
                self.args.use_bad_pixels
                and self.args.bad_pixels
                and pix in self.args.bad_pixels
            )

        plots = [
            (
                "Pedestal vs Temperature",
                "Pedestal (ADC counts)",
                pedestals_vs_temp,
                "pedestal_vs_temperature.png",
            ),
            (
                "Gain vs Temperature",
                "Gain (ADC/p.e.)",
                gains_vs_temp,
                "gain_vs_temperature.png",
            ),
            (
                "Flatfield vs Temperature",
                "Flatfield Coefficient",
                ff_vs_temp,
                "flatfield_vs_temperature.png",
            ),
            (
                "Raw Charge vs Temperature",
                "Raw Charge (ADC counts)",
                raw_charge_mean_vs_temp,
                "raw_charge_vs_temperature.png",
            ),
            (
                "Calibrated Charge vs Temperature (Per Pixel)",
                "Calibrated Charge (p.e.)",
                calib_charge_mean_vs_temp,
                "calibrated_charge_vs_temperature_perpixel.png",
            ),
        ]

        for title, ylabel, data, filename in plots:
            fig, ax = plt.subplots(figsize=(10, 6))
            for pix in range(0, n_pixels, 50):
                if not _is_bad(pix):
                    ax.plot(temperatures, data[:, pix], "o-", alpha=0.3, markersize=3)
            ax.set_xlabel("Temperature (°C)", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            output_file = self.figure_dir / filename
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            self.log.info(f"Plot saved to {output_file}")
            plt.close()

        # Camera-average calibrated charge
        fig, ax = plt.subplots(figsize=(10, 6))
        mean_calib = np.nanmean(calib_charge_mean_vs_temp, axis=1)
        std_calib = np.nanstd(calib_charge_mean_vs_temp, axis=1)
        ax.errorbar(
            temperatures,
            mean_calib,
            yerr=std_calib,
            fmt="o-",
            capsize=5,
            markersize=8,
            linewidth=2,
            color="blue",
            ecolor="blue",
        )
        ax.set_xlabel("Temperature (°C)", fontsize=12)
        ax.set_ylabel("Mean Calibrated Charge (p.e.)", fontsize=12)
        ax.set_title("Camera Average Calibrated Charge vs Temperature", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_file = (
            self.figure_dir / "calibrated_charge_vs_temperature_camera_average.png"
        )
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        self.log.info(f"Average calib. charge vs temp plot saved to {output_file}")
        plt.close()

    def plot_individual_calibration_parameters(self):
        """Create individual plots for each calibration parameter"""
        self.log.info("Creating individual calibration parameter plots...")

        for run in self.args.pedestal_runs:
            if run in self.pedestal_results:
                self._plot_pedestal(run)

        for run in self.args.gain_runs:
            if run in self.gain_results:
                self._plot_gain(run)

        for run in self.args.flatfield_runs:
            if run in self.flatfield_results:
                self._plot_flatfield(run)

        for run in self.args.charge_runs:
            if run in self.charge_results:
                self._plot_charge(run)

    def _plot_pedestal(self, run_number):
        """Plot pedestal distribution"""
        import h5py

        pedestal_path = self.pedestal_results.get(
            run_number
        ) or self.get_pedestal_output_path(run_number)

        if not Path(pedestal_path).exists():
            self.log.warning(f"Pedestal file not found for run {run_number}")
            return

        try:
            with h5py.File(pedestal_path, "r") as f:
                if "pedestal_mean_hg" in f:
                    ped_hg = f["pedestal_mean_hg"][:]
                    ped_lg = f["pedestal_mean_lg"][:]
                elif "pedestal" in f:
                    pedestals = f["pedestal"][:]
                    ped_hg = pedestals[0]
                    ped_lg = pedestals[1]
                else:
                    self.log.warning(
                        f"Cannot plot pedestal for run {run_number}: data not found"
                    )
                    return
        except Exception as e:
            self.log.warning(f"Error loading pedestal data for plotting: {e}")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Pedestal Distribution - Run {run_number}")
        for ax, data, title in zip(axes, [ped_hg, ped_lg], ["High Gain", "Low Gain"]):
            ax.hist(data, bins=50, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Pedestal (ADC counts)")
            ax.set_ylabel("Number of pixels")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_file = self.figure_dir / f"pedestal_run{run_number}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        self.log.info(f"Pedestal plot saved to {output_file}")
        plt.close()

    def _plot_gain(self, run_number):
        """Plot gain distribution"""
        import h5py

        gain_path = self.gain_results.get(run_number) or self.get_gain_output_path(
            run_number
        )

        if not Path(gain_path).exists():
            self.log.warning(f"Gain file not found for run {run_number}")
            return

        try:
            with h5py.File(gain_path, "r") as f:
                if "gain" in f:
                    gains = f["gain"][:]
                elif "high_gain" in f:
                    gains = np.stack([f["high_gain"][:], f["low_gain"][:]], axis=0)
                else:
                    self.log.warning(
                        f"Cannot plot gain for run {run_number}: data not found"
                    )
                    return
        except Exception as e:
            self.log.warning(f"Error loading gain data for plotting: {e}")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Gain Distribution - Run {run_number}")
        for ax, idx, title in zip(
            axes,
            [constants.HIGH_GAIN, constants.LOW_GAIN],
            ["High Gain", "Low Gain"],
        ):
            ax.hist(gains[idx], bins=50, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Gain (ADC/p.e.)")
            ax.set_ylabel("Number of pixels")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_file = self.figure_dir / f"gain_run{run_number}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        self.log.info(f"Gain plot saved to {output_file}")
        plt.close()

    def _plot_flatfield(self, run_number):
        """Plot flatfield coefficient distribution"""
        import h5py

        ff_path = self.flatfield_results.get(
            run_number
        ) or self.get_flatfield_output_path(run_number)

        if not Path(ff_path).exists():
            self.log.warning(f"Flatfield file not found for run {run_number}")
            return

        try:
            with h5py.File(ff_path, "r") as f:
                if "FF_coef" in f:
                    ff_coef = np.mean(f["FF_coef"][:], axis=0)
                elif "flatfield" in f:
                    ff_coef = f["flatfield"][:]
                else:
                    self.log.warning(
                        f"Cannot plot flatfield for run {run_number}: data not found"
                    )
                    return
        except Exception as e:
            self.log.warning(f"Error loading flatfield data for plotting: {e}")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Flatfield Coefficient Distribution - Run {run_number}")
        for ax, idx, title in zip(
            axes,
            [constants.HIGH_GAIN, constants.LOW_GAIN],
            ["High Gain", "Low Gain"],
        ):
            ax.hist(ff_coef[idx], bins=50, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Flatfield Coefficient")
            ax.set_ylabel("Number of pixels")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_file = self.figure_dir / f"flatfield_run{run_number}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        self.log.info(f"Flatfield plot saved to {output_file}")
        plt.close()

    def _plot_charge(self, run_number):
        """Plot charge distribution"""
        import h5py

        charge_path = self.charge_results.get(
            run_number
        ) or self.get_charge_output_path(run_number)

        if not Path(charge_path).exists():
            self.log.warning(f"Charge file not found for run {run_number}")
            return

        try:
            with h5py.File(charge_path, "r") as f:
                if "data/ChargesContainer/charges_hg" in f:
                    charges_hg = f["data/ChargesContainer/charges_hg"][:]
                    charges_lg = f["data/ChargesContainer/charges_lg"][:]
                    charges = np.stack([charges_hg, charges_lg], axis=1)
                elif "charges" in f:
                    charges = f["charges"][:]
                else:
                    self.log.warning(
                        f"Cannot plot charge for run {run_number}: data not found"
                    )
                    return
        except Exception as e:
            self.log.warning(f"Error loading charge data for plotting: {e}")
            return

        if charges.ndim == 3 and charges.shape[0] == 2:
            charges = np.transpose(charges, (1, 0, 2))

        mean_charges = np.mean(charges, axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Charge Distribution - Run {run_number}")
        for ax, idx, title in zip(
            axes,
            [constants.HIGH_GAIN, constants.LOW_GAIN],
            ["High Gain", "Low Gain"],
        ):
            ax.hist(mean_charges[idx], bins=50, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Mean Charge (ADC counts)")
            ax.set_ylabel("Number of pixels")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_file = self.figure_dir / f"charge_run{run_number}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        self.log.info(f"Charge plot saved to {output_file}")
        plt.close()

    def run_pipeline(self):
        """Run the complete calibration pipeline"""
        start_time = time.time()

        self.log.info("=" * 80)
        self.log.info("Starting NectarCAM Calibration Pipeline")
        self.log.info("=" * 80)

        # Step 1: Compute pedestals
        self.log.info("\n" + "=" * 80)
        self.log.info("STEP 1: Pedestal Calibration")
        self.log.info("=" * 80)
        for run in self.args.pedestal_runs:
            try:
                self.compute_pedestal(run)
            except Exception as e:
                self.log.error(
                    f"Error computing pedestal for run {run}: {e}", exc_info=True
                )

        # Step 2: Compute gains
        # Mirror the reference script's per-run max_events list logic.
        self.log.info("\n" + "=" * 80)
        self.log.info("STEP 2: Gain (SPE) Calibration")
        self.log.info("=" * 80)
        gain_max_events = self.args.max_events_gain
        if gain_max_events is None:
            gain_max_events = [None] * len(self.args.gain_runs)
        elif len(gain_max_events) == 1:
            gain_max_events = gain_max_events * len(self.args.gain_runs)

        for run, max_ev in zip(self.args.gain_runs, gain_max_events):
            try:
                self.compute_gain(run, max_events=max_ev)
            except Exception as e:
                self.log.error(
                    f"Error computing gain for run {run}: {e}", exc_info=True
                )

        # Step 3: Compute flatfields
        self.log.info("\n" + "=" * 80)
        self.log.info("STEP 3: Flatfield Calibration")
        self.log.info("=" * 80)
        for run in self.args.flatfield_runs:
            try:
                self.compute_flatfield(run)
            except Exception as e:
                self.log.error(
                    f"Error computing flatfield for run {run}: {e}", exc_info=True
                )

        # Step 4: Compute charges
        self.log.info("\n" + "=" * 80)
        self.log.info("STEP 4: Charge Extraction")
        self.log.info("=" * 80)
        for run in self.args.charge_runs:
            try:
                self.compute_charge(run)
            except Exception as e:
                self.log.error(
                    f"Error computing charge for run {run}: {e}", exc_info=True
                )

        # Step 5: Compute calibrated charges
        self.log.info("\n" + "=" * 80)
        self.log.info("STEP 5: Calibrated Charge Computation")
        self.log.info("=" * 80)
        pedestal_run = self.args.pedestal_runs[0]
        gain_run = self.args.gain_runs[0]
        flatfield_run = self.args.flatfield_runs[0]

        for charge_run in self.args.charge_runs:
            try:
                self.compute_calibrated_charge(
                    charge_run, pedestal_run, gain_run, flatfield_run
                )
            except Exception as e:
                self.log.error(
                    f"Error computing calibrated charge for run {charge_run}: {e}",
                    exc_info=True,
                )

        # Step 6: Create plots
        self.log.info("\n" + "=" * 80)
        self.log.info("STEP 6: Creating Plots")
        self.log.info("=" * 80)
        try:
            self.plot_individual_calibration_parameters()
        except Exception as e:
            self.log.error(f"Error creating individual plots: {e}", exc_info=True)

        try:
            self.plot_calibration_vs_temperature()
        except Exception as e:
            self.log.error(f"Error creating temperature plots: {e}", exc_info=True)

        elapsed_time = time.time() - start_time
        self.log.info("\n" + "=" * 80)
        self.log.info("Pipeline Complete!")
        self.log.info(f"Total execution time: {elapsed_time:.2f} seconds")
        self.log.info("=" * 80)


def main():
    """Main function"""
    args = parser.parse_args()

    # Setup logging
    log_dir = Path(os.environ.get("NECTARCHAIN_LOG", "/tmp")) / str(os.getpid())
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"nectarcam_calibration_{os.getpid()}.log"

    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
        level=args.verbosity,
        filename=str(log_file),
    )

    log = logging.getLogger(__name__)
    log.setLevel(args.verbosity)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(args.verbosity)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)

    logging.getLogger("numba").setLevel(logging.WARNING)

    log.info(f"Log file: {log_file}")
    log.info(f"Arguments: {vars(args)}")

    pipeline = NectarCAMCalibrationPipeline(args, log)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
