"""
NectarCAM Full Calibration and Analysis Pipeline

This script performs a complete calibration pipeline including:
1. Pedestal computation
2. Gain (SPE fit) computation
3. Flatfield computation
4. Charge extraction
5. Calibrated charge computation (pedestal subtraction, gain correction, FF correction)
6. Plotting of all calibration parameters vs temperature

The script is designed to be flexible and configurable via command-line arguments,
allowing users to specify run numbers, processing options, and output directories.
It also includes robust path resolution logic to handle the various output locations
used by the nectarchain tools.

Don't forget to set environment variable NECTARCAMDATA and  NECTRCHAIN_FIGURES
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import h5py
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
    help="Max events for gain runs (1/ run, or a single value applied to all)",
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

# Pipeline mode
parser.add_argument(
    "--mode",
    choices=["full", "pedestal", "gain", "flatfield", "charge"],
    default="full",
    help=(
        "Pipeline mode: 'full' runs complete pipeline with calibrated charge, "
        "'pedestal' runs only pedestal computation and plots, "
        "'gain' runs only gain computation and plots, "
        "'flatfield' runs only flatfield computation and plots, "
        "'charge' runs only charge extraction and plots"
    ),
    type=str,
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
        Search for any HDF5 file matching ``*run{run_number}*`` across all
        candidate directories for *kind*.

        Validates candidates by checking the HDF5 magic bytes (first 8 bytes)
        rather than opening with h5py/PyTables, which would register the file
        in PyTables' global open-file registry and prevent later write-mode opens.

        Returns the first valid HDF5 :class:`~pathlib.Path` or ``None``.
        """
        HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

        def _is_valid_hdf5(path):
            try:
                with open(path, "rb") as fh:
                    return fh.read(8) == HDF5_MAGIC
            except OSError:
                return False

        pattern = f"*run{run_number}*"
        for directory in self._tool_default_search_dirs.get(kind, []):
            d = Path(directory)
            if d.is_dir():
                for candidate in sorted(
                    p for p in d.glob(pattern) if p.suffix == ".h5"
                ):
                    if _is_valid_hdf5(candidate):
                        self.log.debug(
                            f"[_find_file_for_run] run {run_number} ({kind}) "
                            f"found {candidate} in {d}"
                        )
                        return candidate

        # Last-resort: recursive glob under NECTARCAMDATA
        for candidate in sorted(
            p for p in self.nectarcam_data.rglob(pattern) if p.suffix == ".h5"
        ):
            if _is_valid_hdf5(candidate):
                self.log.debug(
                    f"[_find_file_for_run] run {run_number} ({kind}) "
                    f"found via rglob: {candidate}"
                )
                return candidate
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

    def _explore_hdf5_keys(self, f, max_depth=8):
        """
        Recursively collect all Dataset paths in an HDF5 file/group using
        h5py's own type system.  Only ``h5py.Dataset`` objects are added as
        leaves; ``h5py.Group`` objects are always recursed into regardless of
        whether they appear empty to a ``hasattr(keys)`` check.

        This avoids the bug where PyTables Group nodes look like leaf objects
        to a generic ``hasattr`` test, causing the caller to receive group
        paths instead of dataset paths.
        """

        paths = []

        def _recurse(obj, path, depth):
            if depth > max_depth:
                return
            if isinstance(obj, h5py.Dataset):
                paths.append(path)
            elif isinstance(obj, h5py.Group):
                for k in obj.keys():
                    _recurse(obj[k], f"{path}/{k}", depth + 1)
            # anything else (e.g. NamedType) is silently skipped

        _recurse(f, "", 0)
        return paths

    def _load_hg_lg_from_hdf5(self, filepath, kind):
        """
        Load (high_gain, low_gain) arrays from an HDF5 file produced by any
        nectarchain calibration tool.

        Returns (hg_array, lg_array) where each is 1D array of shape (n_pixels,)
        or raises KeyError with full path dump.
        """
        import h5py
        import tables

        with h5py.File(filepath, "r") as f:
            # Collect all Dataset paths (Groups are recursed, never returned)
            all_keys = self._explore_hdf5_keys(f)

            # Also collect top-level Group names for debug when no datasets found
            top_groups = list(f.keys())
            self.log.debug(
                f"[{kind}] {filepath}\n"
                f"  top groups : {top_groups}\n"
                f"  datasets   : {all_keys}"
            )

            # ------------------------------------------------------------------
            # PEDESTAL
            # ------------------------------------------------------------------
            if kind == "pedestal":
                # The paths exist but are PyTables structured arrays
                # Use PyTables to read them
                h5file = tables.open_file(str(filepath), mode="r")
                try:
                    # Try data_combined first (final merged data)
                    for node_path in [
                        "/data_combined/NectarCAMPedestalContainer_0",
                        "/data_1/NectarCAMPedestalContainer_0",
                    ]:
                        if node_path in h5file:
                            table = h5file.get_node(node_path)
                            if len(table) > 0:
                                record = table[0]
                                ped_hg = record[
                                    "pedestal_mean_hg"
                                ]  # Shape: (pixels, samples)
                                ped_lg = record["pedestal_mean_lg"]
                                h5file.close()

                                self.log.debug(f"  pedestal raw shape: {ped_hg.shape}")

                                # Average over samples to get (pixels,)
                                if ped_hg.ndim > 1:
                                    ped_hg = np.mean(ped_hg, axis=1)
                                    ped_lg = np.mean(ped_lg, axis=1)

                                self.log.debug(
                                    f"  pedestal final shape: {ped_hg.shape}"
                                )
                                return ped_hg, ped_lg
                    h5file.close()
                except Exception as e:
                    logging.exception(f"Error reading pedestal from {filepath}: {e}")
                    if h5file.isopen:
                        h5file.close()
                    raise

            # ------------------------------------------------------------------
            # GAIN
            # ------------------------------------------------------------------
            elif kind == "gain":
                # Path exists: /data/SPEfitContainer_0
                # It's a PyTables structured array
                h5file = tables.open_file(str(filepath), mode="r")
                try:
                    if "/data/SPEfitContainer_0" in h5file:
                        table = h5file.get_node("/data/SPEfitContainer_0")
                        if len(table) > 0:
                            record = table[0]
                            gain_hg = record["high_gain"]
                            gain_lg = record["low_gain"]
                            h5file.close()

                            self.log.debug(
                                f"  gain shapes: HG={gain_hg.shape}, LG={gain_lg.shape}"
                            )

                            # Flatten to 1D if needed
                            if gain_hg.ndim > 1:
                                # Take mean or just flatten - depends on structure
                                # Take first column or mean
                                if gain_hg.shape[1] == 3:
                                    # [value, error_low, error_high]
                                    # Take first column (the actual value)
                                    gain_hg = gain_hg[:, 0]
                                    gain_lg = gain_lg[:, 0]
                                else:
                                    # Unknown structure, flatten
                                    gain_hg = gain_hg.ravel()
                                    gain_lg = gain_lg.ravel()

                            self.log.debug(
                                f"  gain shapes: HG={gain_hg.shape}, LG={gain_lg.shape}"
                            )
                            return gain_hg, gain_lg
                    h5file.close()
                except Exception as e:
                    logging.exception(f"Error {filepath}: {e}")
                    if h5file.isopen:
                        h5file.close()
                    raise

            # ------------------------------------------------------------------
            # FLATFIELD
            # ------------------------------------------------------------------
            elif kind == "flatfield":
                # Path exists: /data/FlatFieldContainer_0
                # It's a PyTables structured array
                h5file = tables.open_file(str(filepath), mode="r")
                try:
                    if "/data/FlatFieldContainer_0" in h5file:
                        table = h5file.get_node("/data/FlatFieldContainer_0")
                        if len(table) > 0:
                            record = table[0]
                            if "FF_coef" in record.dtype.names:
                                ff_coef = record["FF_coef"]  # Shape varies
                                h5file.close()

                                self.log.debug(f"  FF_coef raw shape: {ff_coef.shape}")

                                # Average over events if 3D:
                                if ff_coef.ndim == 3:
                                    ff_coef = np.mean(ff_coef, axis=0)

                                # Now should be (gains, pixels)
                                if ff_coef.ndim == 2 and ff_coef.shape[0] >= 2:
                                    ff_hg = ff_coef[0]
                                    ff_lg = ff_coef[1]
                                elif ff_coef.ndim == 1:
                                    ff_hg = ff_coef
                                    ff_lg = ff_coef
                                else:
                                    # Unexpected shape
                                    self.log.warning(
                                        f"  Unexpected FF_coef shape: {ff_coef.shape}"
                                    )
                                    if ff_coef.ndim >= 2:
                                        ff_hg = ff_coef[0]
                                        ff_lg = ff_coef[0]
                                    else:
                                        ff_hg = ff_coef
                                        ff_lg = ff_coef

                                # Check for inf/nan and replace with 1.0
                                if not np.all(np.isfinite(ff_hg)):
                                    n_bad = np.sum(~np.isfinite(ff_hg))
                                    self.log.warning(
                                        f"  FF_coef HG has {n_bad} non-finite values, "
                                        f"replacing with 1.0"
                                    )
                                    ff_hg = np.where(np.isfinite(ff_hg), ff_hg, 1.0)

                                if not np.all(np.isfinite(ff_lg)):
                                    n_bad = np.sum(~np.isfinite(ff_lg))
                                    self.log.warning(
                                        f"  FF_coef LG has {n_bad} non-finite values, "
                                        f"replacing with 1.0"
                                    )
                                    ff_lg = np.where(np.isfinite(ff_lg), ff_lg, 1.0)

                                self.log.debug(
                                    f"  FF_coef final: HG shape={ff_hg.shape}, "
                                    f"mean={np.mean(ff_hg):.3f},LGshape={ff_lg.shape}, "
                                    f"mean={np.mean(ff_lg):.3f}"
                                )

                                return ff_hg, ff_lg
                    h5file.close()
                except Exception as e:
                    logging.exception(f"Error {filepath}: {e}")
                    if h5file.isopen:
                        h5file.close()
                    raise

            # ------------------------------------------------------------------
            # CHARGE
            # ------------------------------------------------------------------
            elif kind == "charge":
                # Paths exist:
                # /data/ChargesContainer_0/FLATFIELD,
                # /data/ChargesContainer_0/SKY_PEDESTAL
                # These are PyTables structured arrays
                h5file = tables.open_file(str(filepath), mode="r")
                try:
                    if "/data/ChargesContainer_0" in h5file:
                        container = h5file.get_node("/data/ChargesContainer_0")

                        hg_parts, lg_parts = [], []

                        # Iterate over trigger datasets
                        for child in container._f_iter_nodes("Table"):
                            if len(child) > 0:
                                record = child[0]
                                if "charges_hg" in record.dtype.names:
                                    charges_hg = record["charges_hg"]
                                    charges_lg = record["charges_lg"]

                                    self.log.debug(
                                        f"  {child._v_name} raw shapes: "
                                        f"HG={charges_hg.shape}, LG={charges_lg.shape}"
                                    )

                                    # Flatten to (events, pixels)
                                    hg_parts.append(
                                        charges_hg.reshape(-1, charges_hg.shape[-1])
                                    )
                                    lg_parts.append(
                                        charges_lg.reshape(-1, charges_lg.shape[-1])
                                    )

                        h5file.close()

                        if hg_parts:
                            hg = np.concatenate(hg_parts, axis=0)
                            lg = np.concatenate(lg_parts, axis=0)

                            self.log.debug(
                                f"  charge final shapes: HG={hg.shape}, LG={lg.shape}"
                            )

                            # Return mean over events to get (pixels,)
                            return np.mean(hg, axis=0), np.mean(lg, axis=0)

                    h5file.close()
                except Exception as e:
                    logging.exception(f"Error {filepath}: {e}")
                    if h5file.isopen:
                        h5file.close()
                    raise

        raise KeyError(
            f"Cannot find {kind} data in {filepath}.\n"
            f"  Top-level groups : {top_groups}\n"
            f"  All dataset paths: {all_keys}"
        )

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
            self.log.info(f"[_get_result_path] {kind} run {run_number} located {found}")
            results_dict[run_number] = found
            return found

        raise FileNotFoundError(
            f"Cannot find {kind} file for run {run_number}. "
            f"In: {[str(d) for d in self._tool_default_search_dirs.get(kind, [])]} "
            f"and recursively under {self.nectarcam_data}."
        )

    def _load_or_default_calibration(self, run_number, kind):
        """
        Load calibration data for a run, with fallback to default values.

        Returns (hg_array, lg_array) tuple.

        Default values:
        - pedestal: 250 ADC for both HG and LG
        - gain: 58.0 ADC/p.e. for HG, 4.46 ADC/p.e. for LG
        - flatfield: 1.0 for both HG and LG
        """
        n_pixels = constants.N_PIXELS

        # Define defaults
        defaults = {
            "pedestal": (
                np.full(n_pixels, 250.0),  # HG
                np.full(n_pixels, 250.0),  # LG
            ),
            "gain": (
                np.full(n_pixels, 58.0),  # HG
                np.full(n_pixels, 4.46),  # LG (58/13)
            ),
            "flatfield": (
                np.full(n_pixels, 1.0),  # HG
                np.full(n_pixels, 1.0),  # LG
            ),
        }

        # Try to load from file
        try:
            if kind == "pedestal":
                path = self._get_result_path(
                    self.pedestal_results,
                    run_number,
                    self.get_pedestal_output_path,
                    "pedestal",
                )
            elif kind == "gain":
                path = self._get_result_path(
                    self.gain_results,
                    run_number,
                    self.get_gain_output_path,
                    "gain",
                )
            elif kind == "flatfield":
                path = self._get_result_path(
                    self.flatfield_results,
                    run_number,
                    lambda r: self.get_flatfield_output_path(r, iteration=2),
                    "flatfield",
                )
            else:
                raise ValueError(f"Unknown calibration kind: {kind}")

            # Load from HDF5
            hg, lg = self._load_hg_lg_from_hdf5(path, kind)
            self.log.info(f"✓ Loaded {kind} from {path}")
            return hg, lg

        except (FileNotFoundError, KeyError, Exception) as e:
            self.log.warning(
                f"Cannot load {kind} for run {run_number}: {e}. "
                f"Using default values."
            )
            hg_default, lg_default = defaults[kind]
            self.log.info(
                f"  {kind.capitalize()} defaults: "
                f"HG={hg_default[0]:.2f}, LG={lg_default[0]:.2f}"
            )
            return hg_default, lg_default

    def compute_calibrated_charge(
        self, charge_run, pedestal_run, gain_run, flatfield_run
    ):
        """
        Compute calibrated charge by applying:
        1. Pedestal subtraction
        2. Gain correction (division)
        3. Flatfield correction (multiplication)
        """

        self.log.info(f"Computing calibrated charge for run {charge_run}...")

        # Strategy: Try to compute/load each component, fall back to defaults if needed

        # 1. Charge data (required - no default)
        if charge_run not in self.charge_results:
            try:
                self.compute_charge(charge_run)
            except Exception as e:
                self.log.error(f"Cannot compute charge for run {charge_run}: {e}")
                return None

        try:
            charge_path = self._get_result_path(
                self.charge_results, charge_run, self.get_charge_output_path, "charge"
            )
        except FileNotFoundError as e:
            self.log.error(f"Charge file required but not found: {e}")
            return None

        # 2-4. Pedestal, Gain, Flatfield (will use defaults if not found)
        # Try to compute first, then load or use defaults
        for run, kind, compute_fn in [
            (pedestal_run, "pedestal", self.compute_pedestal),
            (gain_run, "gain", self.compute_gain),
            (flatfield_run, "flatfield", self.compute_flatfield),
        ]:
            results_dict = getattr(self, f"{kind}_results")
            if run not in results_dict:
                try:
                    self.log.info(f"Attempting to compute {kind} for run {run}...")
                    compute_fn(run)
                except Exception as e:
                    self.log.warning(
                        f"Cannot compute {kind} for run {run}: {e}. "
                        f"Will try to load or use defaults."
                    )

        self.log.info(f"  charge    : {charge_path}")

        # Load charge data – full event-by-event array needed (not mean).
        # Structure: data/ChargesContainer_0/<TRIGGER_TYPE>/charges_hg|lg
        # where TRIGGER_TYPE is a PyTables structured array dataset
        try:
            import tables

            # Use PyTables to read the structured arrays
            h5file = tables.open_file(str(charge_path), mode="r")

            try:
                container = h5file.root.data.ChargesContainer_0

                hg_parts, lg_parts = [], []

                # Load only FLATFIELD trigger type
                if "FLATFIELD" in container._v_children:
                    dataset = container._f_get_child("FLATFIELD")

                    if len(dataset) > 0:
                        # Access the structured array
                        data_record = dataset[0]

                        # Check if it has charges_hg field
                        if "charges_hg" in dataset.dtype.names:
                            charges_hg = data_record["charges_hg"]
                            charges_lg = data_record["charges_lg"]

                            self.log.info(
                                f"  Loaded charges from FLATFIELD: "
                                f"HGshape{charges_hg.shape}, LGshape{charges_lg.shape}"
                            )

                            # Handle shape mismatch
                            if charges_hg.shape != charges_lg.shape:
                                self.log.warning(
                                    f"HG:{charges_hg.shape}, LG:{charges_lg.shape}"
                                )
                                # Truncate to common shape
                                min_events = min(
                                    charges_hg.shape[0], charges_lg.shape[0]
                                )
                                min_pixels = min(
                                    charges_hg.shape[1], charges_lg.shape[1]
                                )
                                charges_hg = charges_hg[:min_events, :min_pixels]
                                charges_lg = charges_lg[:min_events, :min_pixels]
                                self.log.info(f"  Truncated to: {charges_hg.shape}")

                            # Flatten if 3D
                            if charges_hg.ndim == 3:
                                charges_hg = charges_hg.reshape(
                                    -1, charges_hg.shape[-1]
                                )
                                charges_lg = charges_lg.reshape(
                                    -1, charges_lg.shape[-1]
                                )

                            hg_parts.append(charges_hg)
                            lg_parts.append(charges_lg)
                else:
                    raise KeyError("FLATFIELD dataset not found in ChargesContainer_0")

                h5file.close()

                if not hg_parts:
                    raise KeyError(
                        f"Cannot find fields in ChargesContainer_0. "
                        f"Available datasets: {list(container._v_children.keys())}"
                    )

                # Concatenate all trigger types
                charges_hg = np.concatenate(
                    hg_parts, axis=0
                )  # (total_events, n_pixels)
                charges_lg = np.concatenate(lg_parts, axis=0)
                charges = np.stack([charges_hg, charges_lg], axis=1)
                # charges: (total_events, 2, n_pixels)

                self.log.info(
                    f"  Total charge shape: {charges.shape} "
                    f"({len(hg_parts)} trigger types combined)"
                )

            except Exception as e:
                logging.exception(f"Error {charge_path}: {e}")
                if h5file.isopen:
                    h5file.close()
                raise

        except Exception as e:
            self.log.error(f"Error loading charge data: {e}", exc_info=True)
            return None

        # Load pedestal data (with defaults if needed)
        ped_hg, ped_lg = self._load_or_default_calibration(pedestal_run, "pedestal")
        pedestals = np.stack([ped_hg, ped_lg], axis=0)
        self.log.info(
            f"  Pedestals: HG mean={np.mean(ped_hg):.2f}, shape={ped_hg.shape}"
        )

        # Load gain data (with defaults if needed)
        gain_hg, gain_lg = self._load_or_default_calibration(gain_run, "gain")
        gains = np.stack([gain_hg, gain_lg], axis=0)
        self.log.info(f"  Gains: HG mean={np.mean(gain_hg):.2f}, shape={gain_hg.shape}")

        # Load flatfield data (with defaults if needed)
        ff_hg, ff_lg = self._load_or_default_calibration(flatfield_run, "flatfield")
        ff_coef = np.stack([ff_hg, ff_lg], axis=0)
        self.log.info(f"  FF coef: HG mean={np.mean(ff_hg):.3f}, shape={ff_hg.shape}")

        # charges: (total_events, 2, n_pixels)
        # pedestals, gains, ff_coef: (2, n_pixels)
        # Apply calibration: (charge - pedestal) / gain * FF
        self.log.info("  Applying calibration...")
        self.log.info(f"    charges shape: {charges.shape}")
        self.log.info(f"    pedestals shape: {pedestals.shape}")
        self.log.info(f"    gains shape: {gains.shape}")
        self.log.info(f"    ff_coef shape: {ff_coef.shape}")

        # Vectorized calibration (much faster than loops)
        ped_subtracted = charges - pedestals[np.newaxis, :, :]  # Broadcast pedestals

        # Avoid division by zero
        gain_corrected = np.divide(
            ped_subtracted,
            gains[np.newaxis, :, :],
            out=np.zeros_like(ped_subtracted),
            where=gains[np.newaxis, :, :] != 0,
        )

        calibrated_charges = gain_corrected * ff_coef[np.newaxis, :, :]

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

        # Initialise with NaN so missing runs produce gaps in plots, not zeros
        pedestals_vs_temp = np.full((n_temps, n_pixels), np.nan)
        gains_vs_temp = np.full((n_temps, n_pixels), np.nan)
        ff_vs_temp = np.full((n_temps, n_pixels), np.nan)
        raw_charge_mean_vs_temp = np.full((n_temps, n_pixels), np.nan)
        calib_charge_mean_vs_temp = np.full((n_temps, n_pixels), np.nan)

        # Track which temperature indices actually have data (for axis limits)
        valid_temp_mask = np.zeros(n_temps, dtype=bool)

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
                valid_temp_mask[idx] = True
            else:
                self.log.warning(
                    f"No calibrated charge result for run {charge_run} "
                    f"(T={temp}°C) – row will appear as NaN in plots."
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

        n_valid = int(valid_temp_mask.sum())
        missing_note = (
            f" [{n_valid}/{n_temps} runs have data]" if n_valid < n_temps else ""
        )

        for title, ylabel, data, filename in plots:
            if not np.any(np.isfinite(data)):
                self.log.warning(f"Skipping '{title}' – no valid data available.")
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            # Only valid temperature rows
            t_valid = temperatures[valid_temp_mask]
            data_valid = data[valid_temp_mask]

            # -------------------------------
            # 1) Per-pixel sampled lines
            # -------------------------------
            for pix in range(0, n_pixels, 50):
                if not _is_bad(pix):
                    ax.plot(
                        t_valid,
                        data_valid[:, pix],
                        "o-",
                        alpha=0.5,
                        markersize=3,
                        linewidth=1,
                    )

            # -------------------------------
            # 2) Mean and Std across pixels
            # -------------------------------
            mean_vals = np.nanmean(data_valid, axis=1)
            std_vals = np.nanstd(data_valid, axis=1)

            # Hard mean line
            ax.plot(
                t_valid,
                mean_vals,
                "k-",
                linewidth=3,
                label="Mean",
                zorder=10,
            )

            # Std band
            ax.fill_between(
                t_valid,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.2,
                label="±1σ",
                zorder=9,
            )

            # -------------------------------
            ax.set_xlabel("Temperature (°C)", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title + missing_note, fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            output_file = self.figure_dir / filename
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            self.log.info(f"Plot saved to {output_file}")
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
        """Run the calibration pipeline according to --mode argument"""
        start_time = time.time()

        self.log.info("=" * 80)
        self.log.info(
            f"Starting NectarCAM Calibration Pipeline - Mode: {self.args.mode.upper()}"
        )
        self.log.info("=" * 80)

        # Mode-specific execution
        if self.args.mode == "pedestal":
            self._run_pedestal_only()
        elif self.args.mode == "gain":
            self._run_gain_only()
        elif self.args.mode == "flatfield":
            self._run_flatfield_only()
        elif self.args.mode == "charge":
            self._run_charge_only()
        elif self.args.mode == "full":
            self._run_full_pipeline()
        else:
            self.log.error(f"Unknown mode: {self.args.mode}")
            return

        elapsed_time = time.time() - start_time
        self.log.info("\n" + "=" * 80)
        self.log.info("Pipeline Complete!")
        self.log.info(f"Total execution time: {elapsed_time:.2f} seconds")
        self.log.info("=" * 80)

    def _run_pedestal_only(self):
        """Run only pedestal computation and plotting"""
        self.log.info("\n" + "=" * 80)
        self.log.info("PEDESTAL ONLY MODE")
        self.log.info("=" * 80)

        for run in self.args.pedestal_runs:
            try:
                self.compute_pedestal(run)
            except Exception as e:
                self.log.error(
                    f"Error computing pedestal for run {run}: {e}", exc_info=True
                )

        # Plot
        self.log.info("\nCreating pedestal plots...")
        for run in self.args.pedestal_runs:
            if run in self.pedestal_results:
                self._plot_pedestal(run)

    def _run_gain_only(self):
        """Run only gain computation and plotting"""
        self.log.info("\n" + "=" * 80)
        self.log.info("GAIN ONLY MODE")
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

        # Plot
        self.log.info("\nCreating gain plots...")
        for run in self.args.gain_runs:
            if run in self.gain_results:
                self._plot_gain(run)

    def _run_flatfield_only(self):
        """Run only flatfield computation and plotting"""
        self.log.info("\n" + "=" * 80)
        self.log.info("FLATFIELD ONLY MODE")
        self.log.info("=" * 80)

        for run in self.args.flatfield_runs:
            try:
                self.compute_flatfield(run)
            except Exception as e:
                self.log.error(
                    f"Error computing flatfield for run {run}: {e}", exc_info=True
                )

        # Plot
        self.log.info("\nCreating flatfield plots...")
        for run in self.args.flatfield_runs:
            if run in self.flatfield_results:
                self._plot_flatfield(run)

    def _run_charge_only(self):
        """Run only charge extraction and plotting"""
        self.log.info("\n" + "=" * 80)
        self.log.info("CHARGE ONLY MODE")
        self.log.info("=" * 80)

        for run in self.args.charge_runs:
            try:
                self.compute_charge(run)
            except Exception as e:
                self.log.error(
                    f"Error computing charge for run {run}: {e}", exc_info=True
                )

        # Plot
        self.log.info("\nCreating charge plots...")
        for run in self.args.charge_runs:
            if run in self.charge_results:
                self._plot_charge(run)

    def _run_full_pipeline(self):
        """Run the complete calibration pipeline with all steps"""
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
        for pedestal_run, gain_run, flatfield_run, charge_run in zip(
            self.args.pedestal_runs,
            self.args.gain_runs,
            self.args.flatfield_runs,
            self.args.charge_runs,
        ):
            try:
                self.compute_calibrated_charge(
                    charge_run,
                    pedestal_run,
                    gain_run,
                    flatfield_run,
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
