import argparse
import copy
import json
import os
import sys
from pathlib import Path

from nectarchain.data.container import (
    ChargesContainer,
    ChargesContainers,
    WaveformsContainer,
    WaveformsContainers,
)
from nectarchain.makers import (
    ChargesNectarCAMCalibrationTool,
    WaveformsNectarCAMCalibrationTool,
)

prefix = "NectarCAM"
cameras = [f"{prefix}" + "QM"]
cameras.extend([f"{prefix + str(i)}" for i in range(2, 10)])

parser = argparse.ArgumentParser(
    prog="load_wfs_compute_charge",
    description="This program load waveforms from fits.fz run files and compute charge",
)

# run numbers
parser.add_argument(
    "-r", "--run_number", nargs="+", default=[], help="run(s) list", type=int
)

parser.add_argument(
    "-c",
    "--camera",
    choices=cameras,
    default=[camera for camera in cameras if "QM" in camera][0],
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

parser.add_argument(
    "--only_wfs",
    action="store_true",
    default=False,
    help="to only reload wfs",
)
# boolean arguments
parser.add_argument(
    "--reload_wfs",
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
    help="will split the raw data in fits.fz file with events_per_slice events per slices",
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
    default="FullWaveformSum",
    help="charge extractor method",
    type=str,
)
parser.add_argument(
    "--extractor_kwargs",
    default={},
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

args = parser.parse_args()


def main(
    log,
    **kwargs,
):
    # print(kwargs)
    run_number = kwargs.pop("run_number")
    max_events = kwargs.pop("max_events", [None for i in range(len(run_number))])
    if max_events is None:
        max_events = [None for i in range(len(run_number))]

    log.info(max_events)

    tool = WaveformsNectarCAMCalibrationTool()
    waveforms_kwargs = {}
    for key in tool.traits().keys():
        if key in kwargs.keys():
            waveforms_kwargs[key] = kwargs[key]

    tool = ChargesNectarCAMCalibrationTool()
    charges_kwargs = {}
    for key in tool.traits().keys():
        if key in kwargs.keys():
            charges_kwargs[key] = kwargs[key]

    log.info(f"WaveformsNectarCAMCalibrationTool kwargs are {waveforms_kwargs}")
    log.info(f"ChargesNectarCAMCalibrationTool kwargs are {charges_kwargs}")

    for _run_number, _max_events in zip(run_number, max_events):
        try:
            if kwargs.get("only_wfs", False) or kwargs.get("reload_wfs", False):
                log.info("reloading waveforms")
                tool = WaveformsNectarCAMCalibrationTool(
                    progress_bar=True,
                    run_number=_run_number,
                    camera=args.camera,
                    max_events=_max_events,
                    **waveforms_kwargs,
                )
                tool.setup()
                tool.start()
                tool.finish()

                if not (kwargs.get("only_wfs", False)):
                    tool = ChargesNectarCAMCalibrationTool(
                        progress_bar=True,
                        run_number=_run_number,
                        camera=args.camera,
                        max_events=_max_events,
                        from_computed_waveforms=True,
                        **charges_kwargs,
                    )
                    tool.setup()
                    tool.start()
                    tool.finish()
            else:
                log.info("trying to compute charges from waveforms yet extracted")
                tool = ChargesNectarCAMCalibrationTool(
                    progress_bar=True,
                    run_number=_run_number,
                    camera=args.camera,
                    max_events=_max_events,
                    from_computed_waveforms=True,
                    **charges_kwargs,
                )
                tool.setup()
                tool.start()
                tool.finish()
        except Exception as e:
            log.error(e, exc_info=True)


if __name__ == "__main__":
    import logging

    # to quiet numba
    logging.getLogger("numba").setLevel(logging.WARNING)

    # run of interest
    # spe_run_number = [2633,2634,3784]
    # ff_run_number = [2608]
    # ped_run_number = [2609]
    # spe_nevents = [49227,49148,-1]

    args = parser.parse_args()
    kwargs = copy.deepcopy(vars(args))
    kwargs["log_level"] = args.verbosity

    if any("pydevd" in mod for mod in sys.modules):
        kwargs["max_events"] = [1000]
        kwargs["reload_wfs"] = False
        kwargs["run_number"] = [6511]
        kwargs["overwrite"] = True
        kwargs["method"] = "LocalPeakWindowSum"
        kwargs["extractor_kwargs"] = {"window_width": 10, "peak_search_window": 4}
        kwargs["events_per_slice"] = 800
        kwargs["log_level"] = "DEBUG"

    os.makedirs(os.environ.get("NECTARCHAIN_LOG", "/tmp"), exist_ok=True)
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

    log.info(f"arguments passed to main are : {kwargs}")
    # kwargs['reload_wfs'] = True
    # kwargs['run_number'] = [3784]#[5436]
    # kwargs['overwrite'] = True
    # kwargs['events_per_slice'] = 2000
    main(log=log, **kwargs)
