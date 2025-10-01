import argparse
import copy
import glob
import json
import logging
import os
import sys
from pathlib import Path

from nectarchain.data.management import DataManagement
from nectarchain.makers.calibration import PhotoStatisticNectarCAMCalibrationTool
from nectarchain.makers.extractor.utils import CtapipeExtractor

parser = argparse.ArgumentParser(
    prog="gain_SPEfit_computation.py",
    description=f"compute high and low gain with the Photo-statistic\
        method, output data are saved in $NECTARCAMDATA/../PhotoStat/",
)
# run numbers
parser.add_argument(
    "--FF_run_number", nargs="+", default=[], help="run(s) list", type=int
)
parser.add_argument(
    "--Ped_run_number", nargs="+", default=[], help="run(s) list", type=int
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
    default="GlobalPeakWindowSum",
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
    "--figpath",
    type=str,
    default=f"{os.environ.get('NECTARCHAIN_FIGURES','/tmp')}/",
    help="output figure path",
)

# pixels selected
parser.add_argument(
    "-p", "--asked_pixels_id", nargs="+", default=None, help="pixels selected", type=int
)

parser.add_argument(
    "--SPE_run_number",
    help="run number of which the SPE fit has ever been performed",
    type=int,
)
parser.add_argument(
    "--SPE_config",
    choices=[
        "HHVfree",
        "HHVfixed",
        "nominal",
    ],
    help="SPE configuration to use, either HHVfree, HHVfixed or nominal.\
        From ICRC2025 proceedings, we recommend to use resoltion at nominal for the SPE fit.",
)

args = parser.parse_args()


def main(
    log,
    **kwargs,
):
    FF_run_number = kwargs.pop("FF_run_number")
    Ped_run_number = kwargs.pop("Ped_run_number")

    if len(FF_run_number) != len(Ped_run_number):
        raise Exception("The number of FF and Ped runs must be the same")

    max_events = kwargs.pop("max_events", [None for i in range(len(FF_run_number))])
    if max_events is None:
        max_events = [None for i in range(len(FF_run_number))]

    log.info(f"max_events : {max_events}")

    figpath = args.figpath

    str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
        method=args.method, extractor_kwargs=args.extractor_kwargs
    )
    if args.SPE_config is None:
        raise ValueError(
            "You must specify the SPE_config to use, either HHVfree, HHVfixed or nominal"
        )
    if args.SPE_config == "HHVfree":
        path = DataManagement.find_SPE_HHV(
            run_number=args.SPE_run_number,
            method=args.method,
            str_extractor_kwargs=str_extractor_kwargs,
            free_pp_n=True,
        )
    elif args.SPE_config == "HHVfixed":
        path = DataManagement.find_SPE_HHV(
            run_number=args.SPE_run_number,
            method=args.method,
            str_extractor_kwargs=str_extractor_kwargs,
            free_pp_n=False,
        )
    elif args.SPE_config == "nominal":
        path = DataManagement.find_SPE_nominal(
            run_number=args.SPE_run_number,
            method=args.method,
            str_extractor_kwargs=str_extractor_kwargs,
            free_pp_n=False,
        )
    log.info(f"Using SPE path : {path[0]}")

    for _FF_run_number, _Ped_run_number, _max_events in zip(
        FF_run_number, Ped_run_number, max_events
    ):
        try:
            tool = PhotoStatisticNectarCAMCalibrationTool(
                progress_bar=True,
                run_number=_FF_run_number,
                max_events=_max_events,
                Ped_run_number=_Ped_run_number,
                SPE_result=path[0],
                **kwargs,
            )
            tool.setup()
            if args.reload_events and not (_max_events is None):
                _figpath = f"{figpath}/{tool.name}_run{tool.run_number}_maxevents{_max_events}_{tool.method}_{str_extractor_kwargs}"
            else:
                _figpath = f"{figpath}/{tool.name}_run{tool.run_number}_{tool.method}_{str_extractor_kwargs}"
            tool.start()
            tool.finish(figpath=_figpath)
        except Exception as e:
            log.warning(e, exc_info=True)


if __name__ == "__main__":
    import logging

    # to quiet numba
    logging.getLogger("numba").setLevel(logging.WARNING)

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
    kwargs.pop("SPE_run_number")

    kwargs["overwrite"] = True
    log.info(f"arguments passed to main are : {kwargs}")
    main(log=log, **kwargs)
