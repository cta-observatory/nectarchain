import argparse
import copy
import json
import os
import sys
import time
from multiprocessing import freeze_support
from pathlib import Path

from nectarchain.makers.calibration import (
    FlatFieldSPEHHVNectarCAMCalibrationTool,
    FlatFieldSPEHHVStdNectarCAMCalibrationTool,
    FlatFieldSPENominalNectarCAMCalibrationTool,
    FlatFieldSPENominalStdNectarCAMCalibrationTool,
)
from nectarchain.makers.extractor.utils import CtapipeExtractor
from nectarchain.utils.constants import ALLOWED_CAMERAS

parser = argparse.ArgumentParser(
    prog="gain_SPEfit_computation.py",
    description=f"compute high gain with SPE fit for one run at very very high voltage (~1400V) or at nominal voltage (it can often fail). By default, output data are saved in $NECTARCAMDATA/../SPEfit/data/",
)
# run numbers
parser.add_argument(
    "-r", "--run_number", nargs="+", default=[], help="run(s) list", type=int
)

parser.add_argument(
    "-c",
    "--camera",
    choices=ALLOWED_CAMERAS,
    default=[camera for camera in ALLOWED_CAMERAS if "QM" in camera][0],
    help="""Process data for a specific NectarCAM camera.
Default: NectarCAMQM (Qualification Model).""",
    type=str,
)

# max events to be loaded
parser.add_argument(
    "-m",
    "--max_events",
    nargs="+",
    # default=[],
    help="max events to be load",
    type=int,
)

# boolean arguments
parser.add_argument(
    "--reload_events",
    action="store_true",
    default=False,
    help="to force re-computation of waveforms from fits.fz files",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    default=False,
    help="to force overwrite files on disk",
)

parser.add_argument(
    "--events_per_slice",
    type=int,
    default=None,
    help="will slplit the raw data in fits.fz file with events_per_slice events per slices",
)

# extractor arguments
parser.add_argument(
    "--method",
    choices=[
        "FullWaveformSum",
        "FixedWindowSum",
        "GlobalPeakWindowSum",
        "LocalPeakWindowSum",
        "SlidingWindowMaxSum",
        "TwoPassWindowSum",
    ],
    default="LocalPeakWindowSum",
    help="charge extractor method",
    type=str,
)
parser.add_argument(
    "--extractor_kwargs",
    default={"window_width": 8, "window_shift": 4},
    help="charge extractor kwargs",
    type=json.loads,
)

# verbosity argument
parser.add_argument(
    "-v",
    "--verbosity",
    help="set the verbosity level of logger",
    default="INFO",
    choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
    type=str,
)

# output figures path
parser.add_argument(
    "--display",
    action="store_true",
    default=False,
    help="to plot the SPE histogram for each pixel",
)
parser.add_argument(
    "--figpath",
    type=str,
    default=f"{os.environ.get('NECTARCHAIN_FIGURES','/tmp')}/",
    help="output figure path",
)

# pixels selected
parser.add_argument(
    "-p", "--asked_pixels_id", nargs="+", default=None, help="pixels selected", type=int
)

# for let free pp and n :
parser.add_argument(
    "--free_pp_n", action="store_true", default=False, help="to let free pp and n"
)

# to say if it is a run taken at HHV
parser.add_argument(
    "--HHV",
    action="store_true",
    default=False,
    help="to say that these runs are taken at HHV, it will change the configuration file used",
)

# multiprocessing args
parser.add_argument(
    "--multiproc", action="store_true", default=False, help="to use multiprocessing"
)
parser.add_argument(
    "--nproc", help="nproc used for multiprocessing", default=8, type=int
)
parser.add_argument(
    "--chunksize", help="chunksize used for multiprocessing", default=1, type=int
)
args = parser.parse_args()


def main(
    log,
    **kwargs,
):
    run_number = kwargs.pop("run_number")
    max_events = kwargs.pop("max_events", [None for i in range(len(run_number))])
    if max_events is None:
        max_events = [None for i in range(len(run_number))]

    log.info(f"max_events : {max_events}")

    figpath = args.figpath

    if args.HHV:
        if args.free_pp_n:
            _class = FlatFieldSPEHHVNectarCAMCalibrationTool
        else:
            _class = FlatFieldSPEHHVStdNectarCAMCalibrationTool
    else:
        if args.free_pp_n:
            _class = FlatFieldSPENominalNectarCAMCalibrationTool
        else:
            _class = FlatFieldSPENominalStdNectarCAMCalibrationTool

    for _run_number, _max_events in zip(run_number, max_events):
        try:
            tool = _class(
                progress_bar=True,
                camera=args.camera,
                run_number=_run_number,
                max_events=_max_events,
                **kwargs,
            )
            tool.setup()
            if args.reload_events and not (_max_events is None):
                _figpath = f"{figpath}/{tool.name}_run{tool.run_number}_maxevents{_max_events}_{tool.method}_{CtapipeExtractor.get_extractor_kwargs_str(tool.method,tool.extractor_kwargs)}"
            else:
                _figpath = f"{figpath}/{tool.name}_run{tool.run_number}_{tool.method}_{CtapipeExtractor.get_extractor_kwargs_str(tool.method,tool.extractor_kwargs)}"
            tool.start(figpath=_figpath, display=args.display)
            tool.finish(figpath=_figpath, display=args.display)
        except Exception as e:
            log.warning(e, exc_info=True)


if __name__ == "__main__":
    import logging

    # to quiet numba
    logging.getLogger("numba").setLevel(logging.WARNING)

    t = time.time()

    args = parser.parse_args()
    kwargs = copy.deepcopy(vars(args))

    kwargs["log_level"] = args.verbosity
    if any("pydevd" in mod for mod in sys.modules):
        kwargs["max_events"] = [1000]
        kwargs["asked_pixels_id"] = [1, 10, 100, 1000]
        kwargs["run_number"] = [3942]
        kwargs["overwrite"] = True
        kwargs["HHV"] = True
        kwargs["method"] = "LocalPeakWindowSum"
        kwargs["extractor_kwargs"] = {"window_width": 10, "peak_search_window": 4}
        kwargs["log_level"] = "DEBUG"

    os.makedirs(
        f"{os.environ.get('NECTARCHAIN_LOG','/tmp')}/{os.getpid()}/figures",
        exist_ok=True,
    )
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
        level=args.verbosity,
        filename=f"{os.environ.get('NECTARCHAIN_LOG','/tmp')}/{os.getpid()}/{Path(__file__).stem}_{os.getpid()}.log",
    )

    log = logging.getLogger(__name__)
    log.setLevel(args.verbosity)
    ##tips to add message to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(args.verbosity)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)

    kwargs.pop("verbosity")
    kwargs.pop("figpath")
    kwargs.pop("display")
    kwargs.pop("HHV")
    kwargs.pop("free_pp_n")

    # kwargs['run_number'] = [3942]
    # kwargs['overwrite'] = True
    # kwargs['asked_pixels_id'] = [45,600,800]
    # args.HHV = True
    # kwargs['multiproc'] = True
    # args.display = True
    # args.figpath = "/home/ggroller/projects/nectarchain/src/nectarchain/user_scripts/ggrolleron/local/figures"

    log.info(f"arguments passed to main are : {kwargs}")
    main(log=log, **kwargs)
    log.info(f"time for execution is {time.time() - t:.2e} sec")
