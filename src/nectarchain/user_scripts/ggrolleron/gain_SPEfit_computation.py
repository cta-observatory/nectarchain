import logging
import os
import sys
import time
from pathlib import Path

os.makedirs(os.environ.get("NECTARCHAIN_LOG"), exist_ok=True)

# to quiet numba
logging.getLogger("numba").setLevel(logging.WARNING)


import argparse

# import seaborn as sns
from nectarchain.data.container import ChargeContainer
from nectarchain.makers.calibration.gain.FlatFieldSPEMakers import (
    FlatFieldSingleHHVSPEMaker,
    FlatFieldSingleHHVStdSPEMaker,
)

parser = argparse.ArgumentParser(
    prog="gain_SPEfit_computation.py",
    description="compute high gain with SPE fit for one run at very very high voltage (~1400V) or at nominal voltage (it can often fail). Output data are saved in $NECTARCAMDATA/../SPEfit/data/{multipath}{args.voltage_tag}-{SPEpath}-{args.run_number}-{args.chargeExtractorPath}/",
)

# run numbers
parser.add_argument("-r", "--run_number", help="spe run", type=int)

parser.add_argument(
    "--overwrite",
    action="store_true",
    default=False,
    help="to force overwrite files on disk",
)

parser.add_argument(
    "--voltage_tag",
    type=str,
    default="",
    help="tag for voltage specifcication (1400V or nominal), used to setup the output path. See help for more details",
)

# output figures and path extension
parser.add_argument(
    "--display", action="store_true", default=False, help="whether to save plot or not"
)
parser.add_argument(
    "--output_fig_tag", type=str, default="", help="tag to set output figure path"
)

# pixels selected
parser.add_argument(
    "-p", "--pixels", nargs="+", default=None, help="pixels selected", type=int
)

# for let free pp and n :
parser.add_argument(
    "--free_pp_n", action="store_true", default=False, help="to let free pp and n"
)

# multiprocessing args
parser.add_argument(
    "--multiproc", action="store_true", default=False, help="to use multiprocessing"
)
parser.add_argument("--nproc", help="nproc used for multiprocessing", type=int)
parser.add_argument("--chunksize", help="chunksize used for multiprocessing", type=int)

# extractor arguments
parser.add_argument(
    "--chargeExtractorPath",
    help="charge extractor path where charges are saved",
    type=str,
)

# verbosity argument
parser.add_argument(
    "-v",
    "--verbosity",
    help="set the verbosity level of logger",
    default="info",
    choices=["fatal", "debug", "info", "warning"],
    type=str,
)


def main(args):
    figpath = f"{os.environ.get('NECTARCHAIN_FIGURES')}/"
    figpath_ext = "" if args.output_fig_tag == "" else f"-{args.output_fig_tag}"

    multipath = "MULTI-" if args.multiproc else ""
    SPEpath = "SPE" if args.free_pp_n else "SPEStd"

    charge_run_1400V = ChargeContainer.from_file(
        f"{os.environ.get('NECTARCAMDATA')}/charges/{args.chargeExtractorPath}/",
        args.run_number,
    )

    if args.free_pp_n:
        gain_Std = FlatFieldSingleHHVSPEMaker.create_from_chargeContainer(
            signal=charge_run_1400V
        )

    else:
        gain_Std = FlatFieldSingleHHVStdSPEMaker.create_from_chargeContainer(
            signal=charge_run_1400V
        )

    t = time.time()
    gain_Std.make(
        pixels_id=args.pixels,
        multiproc=args.multiproc,
        display=args.display,
        nproc=args.nproc,
        chunksize=args.chunksize,
        figpath=figpath
        + f"/{multipath}{args.voltage_tag}-{SPEpath}-{args.run_number}-{args.chargeExtractorPath}{figpath_ext}",
    )

    log.info(f"fit time =  {time.time() - t } sec")
    gain_Std.save(
        f"{os.environ.get('NECTARCAMDATA')}/../SPEfit/data/{multipath}{args.voltage_tag}-{SPEpath}-{args.run_number}-{args.chargeExtractorPath}/",
        overwrite=args.overwrite,
    )
    conv_rate = (
        len(gain_Std._results[gain_Std._results["is_valid"]]) / gain_Std.npixels
        if args.pixels is None
        else len(gain_Std._results[gain_Std._results["is_valid"]]) / len(args.pixels)
    )
    log.info(f"convergence rate : {conv_rate}")


if __name__ == "__main__":
    args = parser.parse_args()
    logginglevel = logging.FATAL
    if args.verbosity == "warning":
        logginglevel = logging.WARNING
    elif args.verbosity == "info":
        logginglevel = logging.INFO
    elif args.verbosity == "debug":
        logginglevel = logging.DEBUG

    os.makedirs(f"{os.environ.get('NECTARCHAIN_LOG')}/{os.getpid()}/figures")
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
        level=logginglevel,
        filename=f"{os.environ.get('NECTARCHAIN_LOG')}/{os.getpid()}/{Path(__file__).stem}_{os.getpid()}.log",
    )

    log = logging.getLogger(__name__)
    log.setLevel(logginglevel)
    ##tips to add message to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logginglevel)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)

    main(args)
