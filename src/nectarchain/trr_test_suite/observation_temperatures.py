import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ctapipe.io import HDF5TableReader
from ctapipe_io_nectarcam import constants

try:
    from ..utils.constants import (
        FLATFIELD_DEFAULT,
        GAIN_DEFAULT,
        GROUP_NAMES_PEDESTAL,
        HILO_DEFAULT,
        PEDESTAL_DEFAULT,
    )
except ImportError:
    import importlib.util as _ilu
    from pathlib import Path as _Path

    _constants_path = _Path(__file__).resolve().parent.parent / "utils" / "constants.py"
    _spec = _ilu.spec_from_file_location("constants", _constants_path)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    FLATFIELD_DEFAULT = _mod.FLATFIELD_DEFAULT
    GAIN_DEFAULT = _mod.GAIN_DEFAULT
    GROUP_NAMES_PEDESTAL = _mod.GROUP_NAMES_PEDESTAL
    HILO_DEFAULT = _mod.HILO_DEFAULT
    PEDESTAL_DEFAULT = _mod.PEDESTAL_DEFAULT

# nectarchain containers – read back from HDF5 via ctapipe's HDF5TableReader,
# exactly as nectarchain's own calibration_pipeline.py does.
from nectarchain.data.container import (
    ChargesContainer,
    FlatFieldContainer,
    NectarCAMPedestalContainer,
    SPEfitContainer,
)

# nectarchain makers
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

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def get_args():
    """Parses command-line arguments for the NectarCAM calibration pipeline."""
    parser = argparse.ArgumentParser(
        description="""NectarCAM Full Calibration and Analysis Pipeline

Runs tests on the influence of observation temperatures on pedestal, charge,
and flat-field to validate requirements B-ENV-0210 and B-ENV-0230.

Pipeline stages:
  1. Pedestal computation
  2. Gain (SPE fit) computation
  3. Flatfield computation  (two-pass)
  4. Charge extraction
  5. Calibrated charge  (pedestal sub + gain corr + FF corr)
  6. Plots vs temperature  (--mode full only)

Use --mode to run only a subset of stages.
Set $NECTARCAMDATA and $NECTARCHAIN_FIGURES before running.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pedestal_runs", nargs="+", default=[6735, 6744, 6753, 6762, 6771], type=int
    )
    parser.add_argument(
        "--gain_runs", nargs="+", default=[6735, 6744, 6753, 6762, 6771], type=int
    )
    parser.add_argument(
        "--flatfield_runs", nargs="+", default=[6735, 6744, 6753, 6762, 6771], type=int
    )
    parser.add_argument(
        "--charge_runs", nargs="+", default=[6735, 6744, 6753, 6762, 6771], type=int
    )

    parser.add_argument(
        "--temperatures",
        nargs="+",
        default=[25, 20, 14, 10, 5],
        type=float,
        help="Temperatures (°C) in the same order as --charge_runs",
    )

    parser.add_argument("-c", "--camera", default="NectarCAMQM", type=str)

    parser.add_argument("--max_events_pedestal", default=1000, type=int)
    parser.add_argument(
        "--max_events_gain",
        default=1000,
        type=int,
        help="Maximum number of events to process for gain runs",
    )
    parser.add_argument("--max_events_flatfield", default=1000, type=int)
    parser.add_argument("--max_events_charge", default=1000, type=int)

    parser.add_argument("--bad_pixels", nargs="+", default=None, type=int)
    parser.add_argument("--use_bad_pixels", action="store_true", default=False)

    parser.add_argument("--recompute_pedestal", action="store_true", default=False)
    parser.add_argument("--recompute_gain", action="store_true", default=False)
    parser.add_argument("--recompute_flatfield", action="store_true", default=False)
    parser.add_argument("--recompute_charge", action="store_true", default=False)
    parser.add_argument("--recompute_all", action="store_true", default=False)

    # Gain-specific
    parser.add_argument("--HHV", action="store_true", default=False)
    parser.add_argument("--free_pp_n", action="store_true", default=False)
    parser.add_argument("--gain_display", action="store_true", default=False)
    parser.add_argument("--gain_asked_pixels_id", nargs="+", default=None, type=int)
    parser.add_argument("--gain_reload_events", action="store_true", default=False)
    parser.add_argument("--gain_overwrite", action="store_true", default=False)
    parser.add_argument("--gain_events_per_slice", type=int, default=None)
    parser.add_argument("--gain_multiproc", action="store_true", default=False)
    parser.add_argument("--gain_nproc", type=int, default=8)
    parser.add_argument("--gain_chunksize", type=int, default=1)

    # Pedestal-specific
    parser.add_argument("--events_per_slice", type=int, default=300)
    parser.add_argument("--filter_method", type=str, default="WaveformsStdFilter")
    parser.add_argument("--wfs_std_threshold", type=float, default=4.0)

    # Flatfield-specific
    parser.add_argument("--flatfield_window_width", type=int, default=16)
    parser.add_argument("--flatfield_window_shift", type=int, default=4)

    # Charge extraction
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
        type=str,
    )
    parser.add_argument(
        "--charge_extractor_kwargs",
        default='{"window_width": 16, "window_shift": 4}',
        type=str,
    )

    parser.add_argument(
        "--figure_dir",
        type=str,
        default=None,
        help="Figure output directory (default: $NECTARCHAIN_FIGURES)",
    )

    parser.add_argument(
        "--mode",
        choices=["full", "pedestal", "gain", "flatfield", "charge"],
        default="full",
        type=str,
    )
    parser.add_argument(
        "--plot_vs_temp",
        action="store_true",
        default=False,
        help="Also produce vs-temperature plots in partial modes",
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        default="INFO",
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        type=str,
    )
    return parser


# ---------------------------------------------------------------------------
# HDF5 reading helpers
# ---------------------------------------------------------------------------


def _read_container(path, container_class, table_name=None, index_component=0):
    """Yield containers from an HDF5 file using ctapipe's HDF5TableReader."""
    if table_name is None:
        table_name = f"/data/{container_class.__name__}_{index_component}"
    with HDF5TableReader(path) as reader:
        yield from reader.read(table_name=table_name, containers=container_class)


def _load_pedestal(path):
    """Return (hg, lg) mean pedestal arrays, shape (n_pixels,).

    Uses NectarCAMPedestalContainer.from_hdf5(), the built-in nectarchain
    reader, which knows exactly where the data is stored regardless of which
    group name (data, data_1, data_combined, …) was used when writing.
    """
    container = next(NectarCAMPedestalContainer.from_hdf5(str(path)))
    hg = np.mean(container["pedestal_mean_hg"], axis=1)
    lg = np.mean(container["pedestal_mean_lg"], axis=1)
    return hg, lg


def _load_gain(path):
    """Return (hg, lg) SPE gain arrays, shape (n_pixels,).

    Uses SPEfitContainer.from_hdf5(), the built-in nectarchain reader.
    """
    container = next(SPEfitContainer.from_hdf5(str(path)))
    hg = np.asarray(container["high_gain"])
    lg = np.asarray(container["low_gain"])
    if hg.ndim == 2:
        hg, lg = hg[:, 0], lg[:, 0]
    return hg, lg


def _load_flatfield(path):
    """Return (hg, lg) flat-field coefficient arrays, shape (n_pixels,).

    Uses FlatFieldContainer.from_hdf5(), the built-in nectarchain reader.
    """
    container = next(FlatFieldContainer.from_hdf5(str(path)))
    ff = np.asarray(container["FF_coef"])
    if ff.ndim == 3:
        ff = np.mean(ff, axis=0)
    ff_hg = np.where(np.isfinite(ff[0]), ff[0], FLATFIELD_DEFAULT)
    ff_lg = np.where(np.isfinite(ff[1]), ff[1], FLATFIELD_DEFAULT)
    return ff_hg, ff_lg


def _load_charges(path):
    """Return (hg, lg) per-event charge arrays, shape (n_events, n_pixels)."""
    import tables

    with tables.open_file(str(path), mode="r") as h5:
        if "/data/ChargesContainer_0" not in h5:
            raise RuntimeError(f"No /data/ChargesContainer_0 in {path}")
        parent = h5.get_node("/data/ChargesContainer_0")
        child_names = [c._v_name for c in parent._f_iter_nodes("Table")]

    preferred_hg, preferred_lg = [], []
    fallback_hg, fallback_lg = [], []

    for name in child_names:
        table_path = f"/data/ChargesContainer_0/{name}"
        try:
            container = next(_read_container(path, ChargesContainer, table_path))
        except Exception:
            continue
        hg = np.atleast_2d(container.charges_hg)
        lg = np.atleast_2d(container.charges_lg)
        if "FLATFIELD" in name.upper():
            preferred_hg.append(hg)
            preferred_lg.append(lg)
        else:
            fallback_hg.append(hg)
            fallback_lg.append(lg)

    hg_parts = preferred_hg if preferred_hg else fallback_hg
    lg_parts = preferred_lg if preferred_lg else fallback_lg

    if not hg_parts:
        raise RuntimeError(f"No charge data found in {path}")

    return np.concatenate(hg_parts, axis=0), np.concatenate(lg_parts, axis=0)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class NectarCAMObsTempPipeline:
    """Complete NectarCAM calibration pipeline."""

    def __init__(self, args, log):
        self.args = args
        self.log = log

        self.nectarcam_data = Path(os.environ.get("NECTARCAMDATA", "/tmp"))
        self.figure_dir = Path(
            args.figure_dir or os.environ.get("NECTARCHAIN_FIGURES", "/tmp")
        )
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.extractor_kwargs = json.loads(args.charge_extractor_kwargs)

        # run_number → Path of the HDF5 output written by the tool
        self.pedestal_results = {}
        self.gain_results = {}
        self.flatfield_results = {}
        self.charge_results = {}

        # run_number → dict of calibrated arrays
        self.calibrated_charge_results = {}

        self.log.info(f"NECTARCAMDATA : {self.nectarcam_data}")
        self.log.info(f"Figure dir    : {self.figure_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_recompute(self, kind):
        return getattr(self.args, f"recompute_{kind}", False) or self.args.recompute_all

    def _cached_path(self, results_dict, run_number):
        """Return a cached Path if the file still exists on disk, else None."""
        p = results_dict.get(run_number)
        if p is not None and Path(p).exists():
            return Path(p)
        return None

    def _tool_output_path(self, tool):
        """Return the path the tool resolved for its output after setup()."""
        p = getattr(tool, "output_path", None)
        if p:
            return Path(p)
        raise RuntimeError(
            f"{type(tool).__name__} did not expose output_path after setup()."
        )

    def _resolve_existing_output(self, tool_instance, kind, run_number, results_dict):
        """
        Call setup() on an already-constructed tool to resolve its output path,
        then check if that file already exists on disk.

        To avoid setup() opening/truncating the existing file, we temporarily
        set overwrite=False on the tool before calling setup(), then restore
        it afterward if we still need to run the tool.

        Returns the Path if the file exists and recompute is not requested,
        else None (caller must proceed to start()/finish()).
        """
        try:
            tool_instance.overwrite = False
        except Exception:
            pass

        tool_instance.setup()
        candidate = self._tool_output_path(tool_instance)

        if candidate.exists() and not self._should_recompute(kind):
            self.log.info(
                f"  ✓ Found existing {kind} file for run {run_number}: {candidate}"
            )
            results_dict[run_number] = candidate
            # DEBUG: check whether setup() left any handles open on this file
            import tables as _tables

            _open = list(_tables.file._open_files.get_handlers_by_name(str(candidate)))
            self.log.debug(
                f"[DEBUG] open handles on {candidate.name} after setup(): "
                f"{[h.filename + ' mode=' + h.mode for h in _open] or 'none'}"
            )
            for _fh in _open:
                self.log.debug(f"[DEBUG] force-closing handle: {_fh.filename}")
                _fh.close()
            return candidate

        if candidate.exists() and self._should_recompute(kind):
            self.log.info(
                f"  Recompute requested – overwriting {kind} for run {run_number}"
            )

        # Restore overwrite so start()/finish() can write the file
        try:
            tool_instance.overwrite = True
        except Exception:
            pass

        return None  # caller must proceed to .start() / .finish()

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_pedestal(self, run_number):
        """Compute (or reuse) pedestal for *run_number*. Returns output Path."""
        self.log.info(f"Pedestal run {run_number}: checking for existing output…")

        tool = PedestalNectarCAMCalibrationTool(
            progress_bar=True,
            run_number=run_number,
            max_events=self.args.max_events_pedestal,
            events_per_slice=self.args.events_per_slice,
            log_level=logging.getLevelName(self.log.level),
            overwrite=True,
            filter_method=self.args.filter_method,
            wfs_std_threshold=self.args.wfs_std_threshold,
        )

        existing = self._resolve_existing_output(
            tool, "pedestal", run_number, self.pedestal_results
        )
        if existing:
            return existing

        self.log.info(f"Computing pedestal for run {run_number}…")
        tool.start()
        tool.finish(return_output_component=True)
        output_path = self._tool_output_path(tool)

        self.pedestal_results[run_number] = output_path
        self.log.info(f"Pedestal saved to {output_path}")
        return output_path

    def compute_gain(self, run_number):
        """Compute (or reuse) SPE gain for *run_number*. Returns output Path."""
        self.log.info(f"Gain run {run_number}: checking for existing output…")

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

        overwrite = self.args.gain_overwrite or self._should_recompute("gain")
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
        if self.args.gain_asked_pixels_id is not None:
            kwargs["asked_pixels_id"] = self.args.gain_asked_pixels_id
        if self.args.gain_events_per_slice is not None:
            kwargs["events_per_slice"] = self.args.gain_events_per_slice

        tool = tool_class(
            progress_bar=True,
            camera=self.args.camera,
            run_number=run_number,
            max_events=self.args.max_events_gain,
            **kwargs,
        )

        existing = self._resolve_existing_output(
            tool, "gain", run_number, self.gain_results
        )
        if existing:
            return existing

        self.log.info(f"Computing gain (SPE fit) for run {run_number}…")
        extractor_str = CtapipeExtractor.get_extractor_kwargs_str(
            tool.method, tool.extractor_kwargs
        )
        figpath = (
            f"{self.figure_dir}/{tool.name}"
            f"_run{tool.run_number}_{tool.method}_{extractor_str}"
        )
        tool.start(figpath=figpath)
        tool.finish(figpath=figpath)
        output_path = self._tool_output_path(tool)

        self.gain_results[run_number] = output_path
        self.log.info(f"Gain saved to {output_path}")
        return output_path

    def compute_flatfield(self, run_number):
        """Compute (or reuse) flatfield for *run_number* (two-pass). Returns Path."""
        self.log.info(f"Flatfield run {run_number}: checking for existing output…")

        gain_array = np.ones((constants.N_GAINS, constants.N_PIXELS))
        gain_array[0] *= GAIN_DEFAULT
        gain_array[1] *= GAIN_DEFAULT / HILO_DEFAULT

        common = dict(
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
            bad_pix=[],
        )

        # Use a pass-1 tool only to resolve the output path and check existence.
        # Pass-2 will write to the same path (same run + same common kwargs).
        probe_tool = FlatfieldNectarCAMCalibrationTool(
            gain=gain_array.tolist(), **common
        )
        existing = self._resolve_existing_output(
            probe_tool, "flatfield", run_number, self.flatfield_results
        )
        if existing:
            return existing

        self.log.info(f"Computing flatfield for run {run_number} (two-pass)…")

        # --- Pass 1: default gain values (probe_tool already set up) ---
        self.log.info("  First pass…")
        probe_tool.start()
        ff_out_1 = probe_tool.finish(return_output_component=True)[0]

        # Estimate gain from Var(amp)/Mean(amp) on the pass-1 amplitudes
        amp = ff_out_1.amp_int_per_pix_per_event  # (n_events, n_gains, n_pixels)
        updated_gain = np.divide(
            np.var(amp, axis=0),
            np.mean(amp, axis=0),
            where=np.mean(amp, axis=0) != 0.0,
            out=np.ones((constants.N_GAINS, constants.N_PIXELS)),
        )

        # --- Pass 2: updated gain values ---
        self.log.info("  Second pass…")
        tool2 = FlatfieldNectarCAMCalibrationTool(gain=updated_gain.tolist(), **common)
        tool2.setup()
        output_path = self._tool_output_path(tool2)
        tool2.start()
        tool2.finish(return_output_component=True)

        self.flatfield_results[run_number] = output_path
        self.log.info(f"Flatfield saved to {output_path}")
        return output_path

    def compute_charge(self, run_number):
        """Compute (or reuse) charges for *run_number*. Returns output Path."""
        self.log.info(f"Charge run {run_number}: checking for existing output…")

        # We probe with the ChargesNectarCAMCalibrationTool because that is
        # the tool that produces the final charge HDF5 file we care about.
        chg_probe = ChargesNectarCAMCalibrationTool(
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
        existing = self._resolve_existing_output(
            chg_probe, "charge", run_number, self.charge_results
        )
        if existing:
            return existing

        self.log.info(f"Computing charges for run {run_number}…")

        # Step 1 – waveforms (needed before charge extraction)
        wfs = WaveformsNectarCAMCalibrationTool(
            progress_bar=True,
            camera=self.args.camera,
            run_number=run_number,
            max_events=self.args.max_events_charge,
            log_level=logging.getLevelName(self.log.level),
            overwrite=self._should_recompute("charge"),
        )
        wfs.setup()
        wfs.start()
        wfs.finish()

        # Step 2 – charge extraction (chg_probe is already set up)
        output_path = self._tool_output_path(chg_probe)
        chg_probe.start()
        chg_probe.finish()

        self.charge_results[run_number] = output_path
        self.log.info(f"Charges saved to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Loading with physics-motivated defaults
    # ------------------------------------------------------------------

    def _load_or_default(self, run_number, kind):
        """Load (hg, lg) arrays; return defaults on any failure."""
        n = constants.N_PIXELS
        defaults = {
            "pedestal": (np.full(n, PEDESTAL_DEFAULT), np.full(n, PEDESTAL_DEFAULT)),
            "gain": (np.full(n, GAIN_DEFAULT), np.full(n, GAIN_DEFAULT / HILO_DEFAULT)),
            "flatfield": (np.full(n, FLATFIELD_DEFAULT), np.full(n, FLATFIELD_DEFAULT)),
        }
        loaders = {
            "pedestal": (_load_pedestal, self.pedestal_results),
            "gain": (_load_gain, self.gain_results),
            "flatfield": (_load_flatfield, self.flatfield_results),
        }
        loader_fn, cache = loaders[kind]
        path = cache.get(run_number)
        if path and Path(path).exists():
            try:
                hg, lg = loader_fn(path)
                self.log.info(f"  ✓ {kind} loaded from {path}")
                return hg, lg
            except Exception as e:
                self.log.warning(f"  Cannot read {kind} ({path}): {e}")
        self.log.warning(f"  Using default {kind} values for run {run_number}.")
        return defaults[kind]

    # ------------------------------------------------------------------
    # Calibrated charge
    # ------------------------------------------------------------------

    def compute_calibrated_charge(
        self, charge_run, pedestal_run, gain_run, flatfield_run
    ):
        """Pedestal sub + gain corr + FF correction → calibrated p.e. charges.

        If a pre-computed charge file exists on disk (and --recompute_charge is
        not set) the calibration step is still performed using the on-disk data
        so that calibrated_charge_results is always populated.
        """
        self.log.info(f"Computing calibrated charge for run {charge_run}…")

        # ------------------------------------------------------------------
        # 1. Ensure charge file is available (compute if missing)
        # ------------------------------------------------------------------
        charge_path = self._cached_path(self.charge_results, charge_run)
        if charge_path is None:
            # Try to compute (or locate) the charge file.
            try:
                charge_path = self.compute_charge(charge_run)
            except Exception as e:
                self.log.error(f"Cannot obtain charge for run {charge_run}: {e}")
                return None

        if charge_path is None or not charge_path.exists():
            self.log.error(f"Charge file missing for run {charge_run}")
            return None

        try:
            charges_hg, charges_lg = _load_charges(charge_path)
        except Exception as e:
            self.log.error(
                f"Cannot load charges from {charge_path}: {e}", exc_info=True
            )
            return None

        # (n_events, 2, n_pixels)
        charges = np.stack([charges_hg, charges_lg], axis=1)
        self.log.info(f"  Charge shape : {charges.shape}")

        # ------------------------------------------------------------------
        # 2. Ensure calibration files are available (compute / locate each)
        # ------------------------------------------------------------------
        for run, kind, fn in [
            (pedestal_run, "pedestal", self.compute_pedestal),
            (gain_run, "gain", self.compute_gain),
            (flatfield_run, "flatfield", self.compute_flatfield),
        ]:
            if not self._cached_path(getattr(self, f"{kind}_results"), run):
                try:
                    fn(run)
                except Exception as e:
                    self.log.warning(f"  {kind} run {run} unavailable: {e}")

        ped_hg, ped_lg = self._load_or_default(pedestal_run, "pedestal")
        gain_hg, gain_lg = self._load_or_default(gain_run, "gain")
        ff_hg, ff_lg = self._load_or_default(flatfield_run, "flatfield")

        pedestals = np.stack([ped_hg, ped_lg], axis=0)  # (2, n_pixels)
        gains = np.stack([gain_hg, gain_lg], axis=0)
        ff_coef = np.stack([ff_hg, ff_lg], axis=0)

        # ------------------------------------------------------------------
        # 3. Apply calibration
        # ------------------------------------------------------------------
        ped_sub = charges - pedestals[np.newaxis, :, :]
        gain_corr = np.divide(
            ped_sub,
            gains[np.newaxis, :, :],
            out=np.zeros_like(ped_sub),
            where=gains[np.newaxis, :, :] != 0,
        )
        calibrated = gain_corr * ff_coef[np.newaxis, :, :]

        if self.args.use_bad_pixels and self.args.bad_pixels:
            for pix in self.args.bad_pixels:
                if pix < calibrated.shape[2]:
                    calibrated[:, :, pix] = np.nan

        self.calibrated_charge_results[charge_run] = {
            "calibrated_charges": calibrated,
            "raw_charges": charges,
            "pedestals": pedestals,
            "gains": gains,
            "flatfield": ff_coef,
        }
        self.log.info(f"Calibrated charge computed for run {charge_run}")
        return calibrated

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_calibration_vs_temperature(self, plot_types=None):
        """Plot calibration parameters and charges vs temperature."""
        if len(self.args.charge_runs) != len(self.args.temperatures):
            self.log.error("--charge_runs and --temperatures must have the same length")
            return

        temps = np.array(self.args.temperatures)
        n_temps = len(temps)
        n_pixels = constants.N_PIXELS
        g_idx = constants.HIGH_GAIN

        if plot_types is None:
            plot_types = [
                "pedestal",
                "gain",
                "flatfield",
                "raw_charge",
                "calibrated_charge",
            ]

        def nan2d():
            return np.full((n_temps, n_pixels), np.nan)

        arrays = {k: nan2d() for k in plot_types}
        valid = np.zeros(n_temps, dtype=bool)

        for i, (run, _) in enumerate(zip(self.args.charge_runs, temps)):
            r = self.calibrated_charge_results.get(run)
            if r is None:
                self.log.warning(f"No result for run {run}")
                continue
            if "pedestal" in arrays:
                arrays["pedestal"][i] = r["pedestals"][g_idx]
            if "gain" in arrays:
                arrays["gain"][i] = r["gains"][g_idx]
            if "flatfield" in arrays:
                arrays["flatfield"][i] = r["flatfield"][g_idx]
            if "raw_charge" in arrays:
                arrays["raw_charge"][i] = np.nanmean(
                    r["raw_charges"][:, g_idx, :], axis=0
                )
            if "calibrated_charge" in arrays:
                arrays["calibrated_charge"][i] = np.nanmean(
                    r["calibrated_charges"][:, g_idx, :], axis=0
                )
            valid[i] = True

        bad_set = set(self.args.bad_pixels or [])
        if self.args.use_bad_pixels and bad_set:
            for arr in arrays.values():
                arr[:, list(bad_set)] = np.nan

        meta = {
            "pedestal": (
                "Pedestal vs Temperature",
                "Pedestal (ADC counts)",
                "pedestal_vs_temperature.png",
            ),
            "gain": (
                "Gain vs Temperature",
                "Gain (ADC/p.e.)",
                "gain_vs_temperature.png",
            ),
            "flatfield": (
                "Flatfield vs Temperature",
                "Flatfield Coefficient",
                "flatfield_vs_temperature.png",
            ),
            "raw_charge": (
                "Raw Charge vs Temperature",
                "Raw Charge (ADC counts)",
                "raw_charge_vs_temperature.png",
            ),
            "calibrated_charge": (
                "Calibrated Charge vs Temperature",
                "Calibrated Charge (p.e.)",
                "calibrated_charge_vs_temperature.png",
            ),
        }
        note = f" [{int(valid.sum())}/{n_temps} runs]" if valid.sum() < n_temps else ""

        for ptype in plot_types:
            if ptype not in meta or not np.any(np.isfinite(arrays.get(ptype, [[]]))):
                self.log.warning(f"Skipping '{ptype}' – no valid data.")
                continue
            title, ylabel, fname = meta[ptype]
            d = arrays[ptype]
            t_v = temps[valid]
            d_v = d[valid]

            fig, ax = plt.subplots(figsize=(10, 6))
            for pix in range(0, n_pixels, 50):
                if pix not in bad_set:
                    ax.plot(
                        t_v, d_v[:, pix], "o-", alpha=0.5, markersize=3, linewidth=1
                    )
            mean_v = np.nanmean(d_v, axis=1)
            std_v = np.nanstd(d_v, axis=1)
            ax.plot(t_v, mean_v, "k-", lw=3, label="Mean", zorder=10)
            ax.fill_between(
                t_v, mean_v - std_v, mean_v + std_v, alpha=0.2, label="±1σ", zorder=9
            )
            ax.set_xlabel("Temperature (°C)", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title + note, fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            out = self.figure_dir / fname
            plt.savefig(out, dpi=150, bbox_inches="tight")
            self.log.info(f"Saved {out}")
            plt.close()

    def plot_individual_calibration_parameters(self):
        """Histogram plots for each computed calibration parameter."""
        specs = [
            (
                self.args.pedestal_runs,
                self.pedestal_results,
                _load_pedestal,
                "pedestal",
                "Pedestal (ADC counts)",
            ),
            (
                self.args.gain_runs,
                self.gain_results,
                _load_gain,
                "gain",
                "Gain (ADC/p.e.)",
            ),
            (
                self.args.flatfield_runs,
                self.flatfield_results,
                _load_flatfield,
                "flatfield",
                "Flatfield Coefficient",
            ),
            (
                self.args.charge_runs,
                self.charge_results,
                lambda p: tuple(np.mean(a, axis=0) for a in _load_charges(p)),
                "charge",
                "Mean Charge (ADC counts)",
            ),
        ]
        for runs, cache, loader, kind, xlabel in specs:
            for run in runs:
                path = cache.get(run)
                if not (path and Path(path).exists()):
                    continue
                try:
                    hg, lg = loader(path)
                except Exception as e:
                    self.log.warning(f"Cannot load {kind} run {run}: {e}")
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f"{kind.capitalize()} Distribution – Run {run}")
                for ax, data, title in zip(axes, [hg, lg], ["High Gain", "Low Gain"]):
                    ax.hist(data.ravel(), bins=50, alpha=0.7, edgecolor="black")
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("Number of pixels")
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3)
                plt.tight_layout()
                out = self.figure_dir / f"{kind}_run{run}.png"
                plt.savefig(out, dpi=150, bbox_inches="tight")
                self.log.info(f"Saved {out}")
                plt.close()

    # ------------------------------------------------------------------
    # Pipeline modes
    # ------------------------------------------------------------------

    def run_pipeline(self):
        start = time.time()
        self.log.info("=" * 80)
        self.log.info(
            f"NectarCAM Calibration Pipeline – mode: {self.args.mode.upper()}"
        )
        self.log.info("=" * 80)
        {
            "pedestal": self._run_pedestal_only,
            "gain": self._run_gain_only,
            "flatfield": self._run_flatfield_only,
            "charge": self._run_charge_only,
            "full": self._run_full_pipeline,
        }[self.args.mode]()
        self.log.info(f"Pipeline complete in {time.time() - start:.1f} s")

    def _run_each(self, runs, fn, label):
        for run in runs:
            try:
                fn(run)
            except Exception as e:
                self.log.error(f"{label} run {run}: {e}", exc_info=True)

    def _populate_for_temp(self, run, kind, loader):
        """Load *kind* data and stage it in calibrated_charge_results for vs-T plots."""
        cache = getattr(self, f"{kind}_results")
        path = cache.get(run)
        if not (path and Path(path).exists()):
            return
        try:
            hg, lg = loader(path)
            arr = np.stack([hg, lg], axis=0)  # (2, n_pixels)
            empty = np.zeros((1, 2, arr.shape[1]))
            self.calibrated_charge_results[run] = {
                "pedestals": arr if kind == "pedestal" else np.zeros_like(arr),
                "gains": arr if kind == "gain" else np.ones_like(arr),
                "flatfield": arr if kind == "flatfield" else np.ones_like(arr),
                "raw_charges": empty,
                "calibrated_charges": empty,
            }
        except Exception as e:
            self.log.warning(f"Cannot prepare {kind} vs T for run {run}: {e}")

    def _run_pedestal_only(self):
        self.log.info("\n--- PEDESTAL ONLY ---")
        self._run_each(self.args.pedestal_runs, self.compute_pedestal, "Pedestal")
        self.plot_individual_calibration_parameters()
        if self.args.plot_vs_temp:
            for r in self.args.pedestal_runs:
                self._populate_for_temp(r, "pedestal", _load_pedestal)
            self.plot_calibration_vs_temperature(["pedestal"])

    def _run_gain_only(self):
        self.log.info("\n--- GAIN ONLY ---")
        self._run_each(self.args.gain_runs, self.compute_gain, "Gain")
        self.plot_individual_calibration_parameters()
        if self.args.plot_vs_temp:
            for r in self.args.gain_runs:
                self._populate_for_temp(r, "gain", _load_gain)
            self.plot_calibration_vs_temperature(["gain"])

    def _run_flatfield_only(self):
        self.log.info("\n--- FLATFIELD ONLY ---")
        self._run_each(self.args.flatfield_runs, self.compute_flatfield, "Flatfield")
        self.plot_individual_calibration_parameters()
        if self.args.plot_vs_temp:
            for r in self.args.flatfield_runs:
                self._populate_for_temp(r, "flatfield", _load_flatfield)
            self.plot_calibration_vs_temperature(["flatfield"])

    def _run_charge_only(self):
        self.log.info("\n--- CHARGE ONLY ---")
        self._run_each(self.args.charge_runs, self.compute_charge, "Charge")
        self.plot_individual_calibration_parameters()
        if self.args.plot_vs_temp:
            for run in self.args.charge_runs:
                path = self.charge_results.get(run)
                if not (path and Path(path).exists()):
                    continue
                try:
                    hg, lg = _load_charges(path)
                    ch_mean = np.stack(
                        [np.mean(hg, axis=0), np.mean(lg, axis=0)], axis=0
                    )
                    ch_exp = ch_mean[np.newaxis, :, :]
                    self.calibrated_charge_results[run] = {
                        "pedestals": np.zeros_like(ch_mean),
                        "gains": np.ones_like(ch_mean),
                        "flatfield": np.ones_like(ch_mean),
                        "raw_charges": ch_exp,
                        "calibrated_charges": ch_exp,
                    }
                except Exception as e:
                    self.log.warning(f"Cannot prepare charge vs T for run {run}: {e}")
            self.plot_calibration_vs_temperature(["raw_charge"])

    def _run_full_pipeline(self):
        self.log.info("\n=== STEP 1: Pedestal ===")
        self._run_each(self.args.pedestal_runs, self.compute_pedestal, "Pedestal")

        self.log.info("\n=== STEP 2: Gain (SPE) ===")
        self._run_each(self.args.gain_runs, self.compute_gain, "Gain")

        self.log.info("\n=== STEP 3: Flatfield ===")
        self._run_each(self.args.flatfield_runs, self.compute_flatfield, "Flatfield")

        self.log.info("\n=== STEP 4: Charge Extraction ===")
        self._run_each(self.args.charge_runs, self.compute_charge, "Charge")

        self.log.info("\n=== STEP 5: Calibrated Charge ===")
        for ped_r, gain_r, ff_r, chg_r in zip(
            self.args.pedestal_runs,
            self.args.gain_runs,
            self.args.flatfield_runs,
            self.args.charge_runs,
        ):
            try:
                self.compute_calibrated_charge(chg_r, ped_r, gain_r, ff_r)
            except Exception as e:
                self.log.error(f"Calibrated charge run {chg_r}: {e}", exc_info=True)

        self.log.info("\n=== STEP 6: Plots ===")
        try:
            self.plot_individual_calibration_parameters()
        except Exception as e:
            self.log.error(f"Individual plots error: {e}", exc_info=True)
        try:
            self.plot_calibration_vs_temperature()
        except Exception as e:
            self.log.error(f"Temperature plots error: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = get_args()
    args = parser.parse_args()

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
    # Force DEBUG so all [DEBUG] diagnostic lines are visible.
    # Revert to args.verbosity once the file-handle issue is resolved.
    log.setLevel(logging.DEBUG)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.DEBUG)
    h.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    log.addHandler(h)
    logging.getLogger("numba").setLevel(logging.WARNING)

    log.info(f"Log  : {log_file}")
    log.info(f"Args : {vars(args)}")

    NectarCAMObsTempPipeline(args, log).run_pipeline()


if __name__ == "__main__":
    main()
