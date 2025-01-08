import argparse
import copy
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

from nectarchain.data.management import DataManagement
from nectarchain.makers.calibration import (
    FlatFieldSPECombinedStdNectarCAMCalibrationTool,
)
from nectarchain.makers.extractor.utils import CtapipeExtractor

parser = argparse.ArgumentParser(
    prog="gain_SPEfit_combined_computation.py",
    description=f"compute high gain with SPE fit for one run at nominal voltage from a SPE result from a run at 1400V. By default, output data are saved in $NECTARCAMDATA/../SPEfit/data/",
)
# run numbers
parser.add_argument(
    "-r", "--run_number", nargs="+", default=[], help="run(s) list", type=int
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
    default={"window_width": 10, "window_shift": 4},
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

# combined fit
parser.add_argument(
    "--same_luminosity",
    action="store_true",
    default=False,
    help="if luminosity for VVH and nominal data is the same",
)
parser.add_argument(
    "--HHV_run_number",
    help="HHV run number of which the SPE fit has ever been performed",
    type=int,
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

    str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
        args.extractor_kwargs
    )
    path = DataManagement.find_SPE_HHV(
        run_number=args.HHV_run_number,
        method=args.method,
        str_extractor_kwargs=str_extractor_kwargs,
    )
    if len(path) == 1:
        log.info(
            f"{path[0]} found associated to HHV run {args.HHV_run_number}, method {args.method} and extractor kwargs {str_extractor_kwargs}"
        )
    else:
        _text = f"no file found in $NECTARCAM_DATA/../SPEfit associated to HHV run {args.HHV_run_number}, method {args.method} and extractor kwargs {str_extractor_kwargs}"
        log.error(_text)
        raise FileNotFoundError(_text)
    for _run_number, _max_events in zip(run_number, max_events):
        try:
            tool = FlatFieldSPECombinedStdNectarCAMCalibrationTool(
                progress_bar=True,
                run_number=_run_number,
                max_events=_max_events,
                SPE_result=path[0],
                **kwargs,
            )
            tool.setup()
            tool.start()
            if args.reload_events and not (_max_events is None):
                _figpath = f"{figpath}/{tool.name}_run{tool.run_number}_maxevents{_max_events}_{tool.method}_{str_extractor_kwargs}"
            else:
                _figpath = f"{figpath}/{tool.name}_run{tool.run_number}_{tool.method}_{str_extractor_kwargs}"
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
    kwargs.pop("HHV_run_number")

    # args.HHV_run_number = 3942
    # kwargs['run_number'] = [3936]
    # kwargs['overwrite'] = True
    # kwargs['asked_pixels_id'] = [45,600,800]
    # kwargs['multiproc'] = False
    # args.display = True
    # args.figpath = "/home/ggroller/projects/nectarchain/src/nectarchain/user_scripts/ggrolleron/local/figures"

    log.info(f"arguments passed to main are : {kwargs}")
    main(log=log, **kwargs)
    log.info(f"time for execution is {time.time() - t:.2e} sec")
