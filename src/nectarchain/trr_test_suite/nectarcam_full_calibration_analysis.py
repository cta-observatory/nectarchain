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
# from nectarchain.data.container import ChargesContainer
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
    help="Max events for gain runs (1/run, or 1 applied to all)",
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
    default=False,
    help="Force recomputation of pedestal calibration",
)
parser.add_argument(
    "--recompute_gain",
    action="store_true",
    default=False,
    help="Force recomputation of gain calibration",
)
parser.add_argument(
    "--recompute_flatfield",
    action="store_true",
    default=False,
    help="Force recomputation of flatfield calibration",
)
parser.add_argument(
    "--recompute_charge",
    action="store_true",
    default=False,
    help="Force recomputation of charge extraction",
)
parser.add_argument(
    "--recompute_all",
    action="store_true",
    default=False,
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

        # Root of NectarCAM data (mirrors $NECTARCAMDATA)
        self.nectarcam_data = Path(os.environ.get("NECTARCAMDATA", "/tmp"))

        # Pipeline output directory (may equal nectarcam_data if not overridden)
        self.data_dir = Path(args.output_dir or self.nectarcam_data)
        self.figure_dir = Path(
            args.figure_dir or os.environ.get("NECTARCHAIN_FIGURES", "/tmp")
        )

        # Pipeline-managed subdirectories
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

        # Default directories that nectarchain tools write to when no
        # output_path is explicitly given (relative to $NECTARCAMDATA).
        # Extend this list if the framework changes its conventions.
        self._tool_default_search_dirs = {
            "gain": [
                self.gain_dir,
                self.nectarcam_data / "SPEfit",  # actual tool default (from log)
                self.nectarcam_data / "SPEfit" / "data",  # alternate layout
            ],
            "charge": [
                self.charge_dir,
                self.nectarcam_data
                / "runs"
                / "charges",  # actual tool default (from log)
                self.nectarcam_data / "charges",
            ],
            "flatfield": [
                self.flatfield_dir,
                self.nectarcam_data / "flatfield",
                self.nectarcam_data / "flatfields",
            ],
            "pedestal": [
                self.pedestal_dir,
                self.nectarcam_data / "pedestals",
                self.nectarcam_data / "runs" / "pedestals",
            ],
        }

        # Parse extractor kwargs
        self.extractor_kwargs = json.loads(args.charge_extractor_kwargs)

        # Store results  (path or container object, keyed by run number)
        self.pedestal_results = {}
        self.gain_results = {}
        self.flatfield_results = {}
        self.charge_results = {}
        self.calibrated_charge_results = {}

        self.log.info(f"NECTARCAMDATA root : {self.nectarcam_data}")
        self.log.info(f"Pipeline data dir  : {self.data_dir}")
        self.log.info(f"Figure directory   : {self.figure_dir}")

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------

    def _find_file(self, filename, kind):
        """
        Search for *filename* (exact basename) across all candidate directories
        for *kind*.  Returns the first existing :class:`~pathlib.Path` or
        ``None``.
        """
        for directory in self._tool_default_search_dirs.get(kind, []):
            candidate = Path(directory) / filename
            if candidate.exists():
                self.log.debug(f"[_find_file] found {filename} in {directory}")
                return candidate
        # Last-resort: recursive glob under NECTARCAMDATA
        matches = list(self.nectarcam_data.rglob(filename))
        if matches:
            self.log.debug(f"[_find_file] found {filename} via rglob: {matches[0]}")
            return matches[0]
        return None

    def _find_file_for_run(self, run_number, kind):
        """
        Search for any file matching ``*run{run_number}*`` across all candidate
        directories for *kind*.  This is used when the exact filename is not
        known (e.g. the tool embeds method/kwargs in the name).

        Returns the first existing :class:`~pathlib.Path` or ``None``.
        """
        pattern = f"*run{run_number}*"
        for directory in self._tool_default_search_dirs.get(kind, []):
            d = Path(directory)
            if d.is_dir():
                matches = sorted(d.glob(pattern))
                if matches:
                    self.log.debug(
                        f"[_find_file_for_run] run {run_number} ({kind}) "
                        f"found {matches[0]} in {d}"
                    )
                    return matches[0]
        # Last-resort: recursive glob
        matches = sorted(self.nectarcam_data.rglob(pattern))
        # Filter to only .h5 files to avoid matching raw fits.fz files
        matches = [m for m in matches if m.suffix == ".h5"]
        if matches:
            self.log.debug(
                f"[_find_file_for_run] run {run_number} ({kind}) "
                f"found via rglob: {matches[0]}"
            )
            return matches[0]
        return None

    def _resolve_tool_output_path(self, tool, expected_path, kind):
        """
        Return the actual path where *tool* wrote (or will write) its output.

        Priority:
        1. ``tool.output_path`` attribute (set by the tool itself after
           ``setup()``), if it exists and is non-empty.
        2. *expected_path* if it already exists on disk.
        3. ``_find_file`` search across default directories.
        4. Fall back to *expected_path* (caller will fail gracefully).
        """
        # 1. Trust the tool's own resolved output_path
        tool_path = getattr(tool, "output_path", None)
        if tool_path:
            p = Path(tool_path)
            if p.exists():
                return p
            # Tool set a path but file isn't written yet – remember it so we
            # can look for it after finish()
            self.log.debug(f"Tool output_path={tool_path} (not yet written)")
            return p

        # 2. Expected path already on disk
        if Path(expected_path).exists():
            return Path(expected_path)

        # 3. Search default dirs (useful when tool already ran previously)
        filename = Path(expected_path).name
        found = self._find_file(filename, kind)
        if found:
            return found

        return Path(expected_path)

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
        # overwrite must be True whenever we are (re)computing, otherwise the
        # tool raises an exception if the output file already exists on disk.
        overwrite = (
            self.args.gain_overwrite
            or self.args.recompute_gain
            or self.args.recompute_all
        )
        kwargs = dict(
            method=self.args.charge_method,
            extractor_kwargs=self.extractor_kwargs,
            log_level=logging.getLevelName(self.log.level),
            overwrite=overwrite,
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
        expected_path = self.get_gain_output_path(run_number)

        if not (self.args.recompute_gain or self.args.recompute_all):
            # Try expected path first, then search default dirs
            if expected_path.exists():
                self.log.info(f"Gain for run {run_number} found at {expected_path}")
                self.gain_results[run_number] = expected_path
                return expected_path
            found = self._find_file_for_run(run_number, "gain")
            if found:
                self.log.info(f"Gain for run {run_number} found at {found}")
                self.gain_results[run_number] = found
                return found
            self.log.info(
                f"Gain for run {run_number} not found on disk – will compute."
            )

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

            # Resolve the path the tool itself will write to (may differ from
            # expected_path when the tool uses its own default directory).
            actual_path = self._resolve_tool_output_path(tool, expected_path, "gain")
            self.log.info(f"Gain output will be written to: {actual_path}")

            # Build figpath the same way as in the reference script
            if gain_kwargs.get("reload_events") and max_events is not None:
                extractor_str = CtapipeExtractor.get_extractor_kwargs_str(
                    tool.method,
                    tool.extractor_kwargs,
                )

                figpath = (
                    f"{self.figure_dir}/{tool.name}"
                    f"_run{tool.run_number}"
                    f"_maxevents{max_events}"
                    f"_{tool.method}"
                    f"_{extractor_str}"
                )
            else:
                extractor_str = CtapipeExtractor.get_extractor_kwargs_str(
                    tool.method,
                    tool.extractor_kwargs,
                )

                figpath = (
                    f"{self.figure_dir}/{tool.name}"
                    f"_run{tool.run_number}"
                    f"_{tool.method}"
                    f"_{extractor_str}"
                )

            tool.start(figpath=figpath)
            tool.finish(figpath=figpath)

            # After finish(), re-resolve in case the file appeared at a
            # different location than setup() reported.
            if not actual_path.exists():
                found = self._find_file_for_run(run_number, "gain")
                if found:
                    actual_path = found
                    self.log.info(f"Gain file located via search at {actual_path}")

        except Exception as e:
            self.log.warning(
                f"Gain computation failed for run {run_number}: {e}", exc_info=True
            )
            raise

        self.gain_results[run_number] = actual_path
        self.log.info(
            f"Gain computation completed for run {run_number} → {actual_path}"
        )

        return actual_path

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
        expected_path = self.get_charge_output_path(run_number)

        if not (self.args.recompute_charge or self.args.recompute_all):
            if expected_path.exists():
                self.log.info(f"Charges for run {run_number} found at {expected_path}")
                self.charge_results[run_number] = expected_path
                return expected_path
            found = self._find_file_for_run(run_number, "charge")
            if found:
                self.log.info(f"Charges for run {run_number} found at {found}")
                self.charge_results[run_number] = found
                return found
            self.log.info(
                f"Charges for run {run_number} not found on disk – will compute."
            )

        self.log.info(f"Computing charges for run {run_number}...")

        # Waveforms step
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

        # Charge extraction step
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

        # Resolve the path the tool itself will write to
        actual_path = self._resolve_tool_output_path(
            charge_tool, expected_path, "charge"
        )
        self.log.info(f"Charge output will be written to: {actual_path}")

        charge_tool.start()
        charge_tool.finish()

        # Re-resolve after finish in case the file appeared elsewhere
        if not actual_path.exists():
            found = self._find_file_for_run(run_number, "charge")
            if found:
                actual_path = found
                self.log.info(f"Charge file located via search at {actual_path}")

        self.charge_results[run_number] = actual_path
        self.log.info(f"Charges saved to {actual_path}")

        return actual_path

    def _get_result_path(self, results_dict, run_number, get_default_fn, kind):
        """
        Return an existing Path for *run_number* from *results_dict*.

        Tries, in order:
        1. The stored value (if it is a Path/str and exists on disk).
        2. The pipeline-default path returned by *get_default_fn*.
        3. ``_find_file`` by exact filename across all known default dirs.
        4. ``_find_file_for_run`` by run-number glob across all known default dirs.

        Raises ``FileNotFoundError`` if nothing is found.
        """
        stored = results_dict.get(run_number)
        if stored is not None and isinstance(stored, (str, Path)):
            p = Path(stored)
            if p.exists():
                return p

        default = Path(get_default_fn(run_number))
        if default.exists():
            return default

        found = self._find_file(default.name, kind)
        if found:
            self.log.info(
                f"[_get_result_path] {kind} run {run_number} located at {found}"
            )
            results_dict[run_number] = found
            return found

        found = self._find_file_for_run(run_number, kind)
        if found:
            self.log.info(
                "[_get_result_path] %s run %s  via glob at %s",
                kind,
                run_number,
                found,
            )

            results_dict[run_number] = found
            return found

        raise FileNotFoundError(
            f"Cannot find {kind} file for run {run_number}. "
            f"and recursively under {self.nectarcam_data}."
        )

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

        # Resolve actual on-disk paths using stored results + fallback search
        try:
            charge_path = self._get_result_path(
                self.charge_results, charge_run, self.get_charge_output_path, "charge"
            )
        except FileNotFoundError as e:
            self.log.error(str(e))
            return None

        try:
            pedestal_path = self._get_result_path(
                self.pedestal_results,
                pedestal_run,
                self.get_pedestal_output_path,
                "pedestal",
            )
        except FileNotFoundError as e:
            self.log.error(str(e))
            return None

        try:
            gain_path = self._get_result_path(
                self.gain_results, gain_run, self.get_gain_output_path, "gain"
            )
        except FileNotFoundError as e:
            self.log.error(str(e))
            return None

        try:
            flatfield_path = self._get_result_path(
                self.flatfield_results,
                flatfield_run,
                lambda r: self.get_flatfield_output_path(r, iteration=2),
                "flatfield",
            )
        except FileNotFoundError as e:
            self.log.error(str(e))
            return None

        self.log.info(f"  charge    : {charge_path}")
        self.log.info(f"  pedestal  : {pedestal_path}")
        self.log.info(f"  gain      : {gain_path}")
        self.log.info(f"  flatfield : {flatfield_path}")

        # Load charge data - CORRECTED to read structured arrays properly

        # Load charge data - CORRECTED to handle shape mismatches
        try:
            with h5py.File(charge_path, "r") as f:
                all_keys = self._explore_hdf5_keys(f)
                self.log.debug(f"[charge] HDF5 keys: {all_keys}")

                charges_hg = charges_lg = None

                # Try structured array format first
                for container_path in [
                    "data/ChargesContainer_0",
                    "data/ChargesContainer",
                ]:
                    if container_path in f:
                        container = f[container_path]
                        # Look for datasets like "FLATFIELD", "PEDESTAL", etc.
                        for dataset_name in container.keys():
                            dataset = container[dataset_name]
                            if len(dataset) > 0:
                                dataset0 = dataset[
                                    0
                                ]  # First element of structured array
                                if "charges_hg" in dataset0.dtype.names:
                                    charges_hg = dataset0["charges_hg"]
                                    charges_lg = dataset0["charges_lg"]

                                    self.log.info(
                                        f"  Loaded from {container_path}/{dataset_name}"
                                    )
                                    self.log.info(
                                        f"  charges_hg shape: {charges_hg.shape}"
                                    )
                                    self.log.info(
                                        f"  charges_lg shape: {charges_lg.shape}"
                                    )

                                    # Handle shape mismatches
                                    if charges_hg.shape != charges_lg.shape:
                                        self.log.warning(
                                            "Shape mismatch detected! "
                                            "charges_hg: %s, charges_lg: %s",
                                            charges_hg.shape,
                                            charges_lg.shape,
                                        )

                                        # Find common shape
                                        min_shape = tuple(
                                            min(s1, s2)
                                            for s1, s2 in zip(
                                                charges_hg.shape, charges_lg.shape
                                            )
                                        )
                                        self.log.info(
                                            f"  Truncating to common shape: {min_shape}"
                                        )

                                        if len(min_shape) == 2:
                                            charges_hg = charges_hg[
                                                : min_shape[0], : min_shape[1]
                                            ]
                                            charges_lg = charges_lg[
                                                : min_shape[0], : min_shape[1]
                                            ]
                                        elif len(min_shape) == 3:
                                            charges_hg = charges_hg[
                                                : min_shape[0],
                                                : min_shape[1],
                                                : min_shape[2],
                                            ]
                                            charges_lg = charges_lg[
                                                : min_shape[0],
                                                : min_shape[1],
                                                : min_shape[2],
                                            ]

                                    break
                        if charges_hg is not None:
                            break

                if charges_hg is None:
                    raise KeyError(
                        f"Cannot find charge data. Available keys: {all_keys}"
                    )

                # charges shape: may be
                # (n_slices, n_events_per_slice, n_pixels) or (n_events, n_pixels)
                # Reshape to (total_events, n_pixels)
                if charges_hg.ndim == 3:
                    self.log.info("reshaping from 3d to 2d")
                    charges_hg = charges_hg.reshape(-1, charges_hg.shape[-1])
                    charges_lg = charges_lg.reshape(-1, charges_lg.shape[-1])
                    self.log.info(
                        "After reshape - charges_hg: %s, charges_lg: %s",
                        charges_hg.shape,
                        charges_lg.shape,
                    )

                # Final shape check before stacking
                if charges_hg.shape != charges_lg.shape:
                    self.log.error("  Still have shape mismatch after processing!")
                    self.log.error(f"  charges_hg: {charges_hg.shape}")
                    self.log.error(f"  charges_lg: {charges_lg.shape}")
                    # Take the smaller one
                    n_events = min(charges_hg.shape[0], charges_lg.shape[0])
                    n_pixels = min(charges_hg.shape[1], charges_lg.shape[1])
                    charges_hg = charges_hg[:n_events, :n_pixels]
                    charges_lg = charges_lg[:n_events, :n_pixels]
                    self.log.info(f"  Truncated to: {charges_hg.shape}")

                # Stack to (total_events, 2, n_pixels)
                charges = np.stack([charges_hg, charges_lg], axis=1)

                self.log.info(f"  Final charges shape: {charges.shape}")
                self.log.info(f"  HG mean: {np.mean(charges[:, 0, :]):.2f} ADC")
                self.log.info(f"  LG mean: {np.mean(charges[:, 1, :]):.2f} ADC")

        except Exception as e:
            self.log.error(f"Error loading charge data: {e}", exc_info=True)
            return None

        # Load pedestal data
        try:
            ped_hg, ped_lg = self._load_hg_lg_from_hdf5(pedestal_path, "pedestal")
            pedestals = np.stack([ped_hg, ped_lg], axis=0)
            self.log.info(
                f"  Pedestals: HG mean={np.mean(ped_hg):.2f}, shape={ped_hg.shape}"
            )
        except Exception as e:
            self.log.error(f"Error loading pedestal data: {e}", exc_info=True)
            return None

        # Load gain data - USE DEFAULT IF MISSING
        try:
            # First check if gain file actually exists
            if not Path(gain_path).exists():
                self.log.warning(f"Gain file does not exist: {gain_path}")
                raise FileNotFoundError("Gain file missing")

            gain_hg, gain_lg = self._load_hg_lg_from_hdf5(gain_path, "gain")
            gains = np.stack([gain_hg, gain_lg], axis=0)
            self.log.info(
                f"  Gains: HG mean={np.mean(gain_hg):.2f}, shape={gain_hg.shape}"
            )

        except Exception as e:
            # USE DEFAULT GAIN VALUES
            self.log.warning(f"Cannot load gain file ({e})")
            self.log.warning("Using DEFAULT gain values: HG=58.0, LG=4.46 ADC/p.e.")

            # Default gains based on NectarCAM typical values
            default_hg_gain = 58.0  # ADC/p.e.
            default_lg_gain = 4.46  # ADC/p.e. (58/13)

            # Get number of pixels from charges
            n_pixels = charges.shape[2]

            # Create gain arrays with default values
            gain_hg = np.full(n_pixels, default_hg_gain)
            gain_lg = np.full(n_pixels, default_lg_gain)
            gains = np.stack([gain_hg, gain_lg], axis=0)

            self.log.info(
                f"  Using default gains: HG={default_hg_gain}, LG={default_lg_gain}"
            )

        # Load flatfield data
        try:
            ff_hg, ff_lg = self._load_hg_lg_from_hdf5(flatfield_path, "flatfield")
            ff_coef = np.stack([ff_hg, ff_lg], axis=0)
            self.log.info(
                f"  FF coef: HG mean={np.mean(ff_hg):.3f}, shape={ff_hg.shape}"
            )
        except Exception as e:
            self.log.error(f"Error loading flatfield data: {e}", exc_info=True)
            return None

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

        self.log.info(f"✓ Calibrated charge computed for run {charge_run}")
        self.log.info(f"  Raw HG mean: {np.mean(charges[:, 0, :]):.2f} ADC")
        self.log.info(
            f"  Calibrated HG mean: {np.mean(calibrated_charges[:, 0, :]):.2f} p.e."
        )

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

    def _explore_hdf5_keys(self, f, prefix="", max_depth=4):
        """Recursively collect all dataset paths in an HDF5 file (for debugging)."""
        paths = []

        def _recurse(obj, path, depth):
            if depth > max_depth:
                return
            if hasattr(obj, "keys"):
                for k in obj.keys():
                    _recurse(obj[k], f"{path}/{k}", depth + 1)
            else:
                paths.append(path)

        _recurse(f, "", 0)
        return paths

    def _load_hg_lg_from_hdf5(self, filepath, kind):
        """
        Try to load [high_gain, low_gain] arrays from an HDF5 file written by
        a nectarchain tool.  Probes multiple known key layouts and falls back
        to a full-key dump so we can log exactly what is available.

        Returns (hg_array, lg_array) or raises KeyError.
        """
        import h5py

        with h5py.File(filepath, "r") as f:
            all_keys = self._explore_hdf5_keys(f)
            self.log.debug(f"[{kind}] HDF5 keys in {filepath}: {all_keys}")

            if kind == "pedestal":
                for container_path in [
                    "data_1/NectarCAMPedestalContainer_0",
                    "data/NectarCAMPedestalContainer_0",
                    "data_combined/NectarCAMPedestalContainer_0",
                ]:
                    if container_path in f:
                        data = f[container_path]
                        # Access structured array fields - shape is (1,) so take [0]
                        ped_hg = data["pedestal_mean_hg"][0]  # Shape: (pixels, samples)
                        ped_lg = data["pedestal_mean_lg"][0]
                        # Average over samples dimension to get (pixels,)
                        return np.mean(ped_hg, axis=1), np.mean(ped_lg, axis=1)

                # Fallback: direct access (older format)
                for hg_key, lg_key in [
                    ("pedestal_mean_hg", "pedestal_mean_lg"),
                ]:
                    if hg_key in f:
                        ped_hg = f[hg_key][:]
                        ped_lg = f[lg_key][:]
                        if ped_hg.ndim == 2:
                            ped_hg = np.mean(ped_hg, axis=1)
                            ped_lg = np.mean(ped_lg, axis=1)
                        return ped_hg, ped_lg

            elif kind == "gain":
                # Correct layout: /data/SPEfitContainer_0 is a structured array
                for container_path in [
                    "data/SPEfitContainer_0",
                    "data/SPEfitContainer",
                ]:
                    if container_path in f:
                        data = f[container_path]
                        # Access structured array fields - shape is (1,) so take [0]
                        gain_hg = data["high_gain"][0]
                        gain_lg = data["low_gain"][0]
                        return gain_hg, gain_lg

                # Fallback: direct access
                for hg_key, lg_key in [
                    ("high_gain", "low_gain"),
                    ("gain_hg", "gain_lg"),
                ]:
                    if hg_key in f:
                        return f[hg_key][:], f[lg_key][:]

            elif kind == "flatfield":
                # FlatfieldNectarCAMCalibrationTool layout
                for container_path in [
                    "data/FlatFieldContainer_0",
                    "data/FlatFieldContainer",
                ]:
                    if container_path in f:
                        # FF_coef may be inside the container
                        if "FF_coef" in f[container_path]:
                            arr = f[container_path]["FF_coef"][:]
                            # Shape may be (events, gains, pixels) - average over events
                            if arr.ndim == 3:
                                arr = np.mean(arr, axis=0)
                            return arr[0], arr[1]

                # Direct access
                for key in ["FF_coef", "data/FF_coef"]:
                    if key in f:
                        arr = f[key][:]
                        if arr.ndim == 3:
                            arr = np.mean(arr, axis=0)
                        return arr[0], arr[1]

            elif kind == "charge":
                for container_path in [
                    "data/ChargesContainer_0",
                    "data/ChargesContainer",
                ]:
                    if container_path in f:
                        container = f[container_path]
                        # The container has datasets like "FLATFIELD", "PEDESTAL", etc.
                        # Find the first dataset
                        for dataset_name in container.keys():
                            dataset = container[dataset_name]
                            if len(dataset) > 0:
                                dataset0 = dataset[
                                    0
                                ]  # First element of structured array
                                # Check if this has charge data
                                if "charges_hg" in dataset0.dtype.names:
                                    charges_hg = dataset0["charges_hg"]
                                    charges_lg = dataset0["charges_lg"]
                                    # charges shape: (N_events, N_pixels) or (N_pixels,)
                                    # Return mean over events if needed
                                    if charges_hg.ndim > 1:
                                        return np.mean(charges_hg, axis=0), np.mean(
                                            charges_lg, axis=0
                                        )
                                    else:
                                        return charges_hg, charges_lg

                # Fallback: direct access
                for hg_key, lg_key in [
                    ("charges_hg", "charges_lg"),
                    ("data/charges_hg", "data/charges_lg"),
                ]:
                    if hg_key in f:
                        charges_hg = f[hg_key][:]
                        charges_lg = f[lg_key][:]
                        if charges_hg.ndim > 1:
                            return np.mean(charges_hg, axis=0), np.mean(
                                charges_lg, axis=0
                            )
                        else:
                            return charges_hg, charges_lg

        raise KeyError(
            f"Cannot find {kind} data in {filepath}. "
            f"Available HDF5 paths: {all_keys}"
        )

    def _plot_pedestal(self, run_number):
        """Plot pedestal distribution"""
        try:
            path = self._get_result_path(
                self.pedestal_results,
                run_number,
                self.get_pedestal_output_path,
                "pedestal",
            )
        except FileNotFoundError as e:
            self.log.warning(str(e))
            return

        try:
            ped_hg, ped_lg = self._load_hg_lg_from_hdf5(path, "pedestal")
        except Exception as e:
            self.log.warning(f"Cannot plot pedestal run {run_number}: {e}")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Pedestal Distribution - Run {run_number}")
        for ax, data, title in zip(axes, [ped_hg, ped_lg], ["High Gain", "Low Gain"]):
            ax.hist(data.ravel(), bins=50, alpha=0.7, edgecolor="black")
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
        try:
            path = self._get_result_path(
                self.gain_results, run_number, self.get_gain_output_path, "gain"
            )
        except FileNotFoundError as e:
            self.log.warning(str(e))
            return

        try:
            gain_hg, gain_lg = self._load_hg_lg_from_hdf5(path, "gain")
        except Exception as e:
            self.log.warning(f"Cannot plot gain run {run_number}: {e}")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Gain Distribution - Run {run_number}")
        for ax, data, title in zip(axes, [gain_hg, gain_lg], ["High Gain", "Low Gain"]):
            ax.hist(data.ravel(), bins=50, alpha=0.7, edgecolor="black")
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
        try:
            path = self._get_result_path(
                self.flatfield_results,
                run_number,
                lambda r: self.get_flatfield_output_path(r, iteration=2),
                "flatfield",
            )
        except FileNotFoundError as e:
            self.log.warning(str(e))
            return

        try:
            ff_hg, ff_lg = self._load_hg_lg_from_hdf5(path, "flatfield")
        except Exception as e:
            self.log.warning(f"Cannot plot flatfield run {run_number}: {e}")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Flatfield Coefficient Distribution - Run {run_number}")
        for ax, data, title in zip(axes, [ff_hg, ff_lg], ["High Gain", "Low Gain"]):
            ax.hist(data.ravel(), bins=50, alpha=0.7, edgecolor="black")
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
        try:
            path = self._get_result_path(
                self.charge_results, run_number, self.get_charge_output_path, "charge"
            )
        except FileNotFoundError as e:
            self.log.warning(str(e))
            return

        try:
            charge_hg, charge_lg = self._load_hg_lg_from_hdf5(path, "charge")
        except Exception as e:
            self.log.warning(f"Cannot plot charge run {run_number}: {e}")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Charge Distribution - Run {run_number}")
        for ax, data, title in zip(
            axes, [charge_hg, charge_lg], ["High Gain", "Low Gain"]
        ):
            ax.hist(data.ravel(), bins=50, alpha=0.7, edgecolor="black")
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
