import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.makedirs(os.environ.get("NECTARCHAIN_LOG"), exist_ok=True)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    filename=f"{os.environ.get('NECTARCHAIN_LOG','/tmp')}/{Path(__file__).stem}_{os.getpid()}.log",
)
log = logging.getLogger(__name__)
import copy
from nectarchain.data.container import (
    ChargesContainer,
    ChargesContainers,
    WaveformsContainer,
    WaveformsContainers,
)
from nectarchain.makers import WaveformsNectarCAMCalibrationTool,ChargesNectarCAMCalibrationTool

parser = argparse.ArgumentParser(
    prog="load_wfs_compute_charge",
    description="This program load waveforms from fits.fz run files and compute charge",
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
    #default=[],
    help="max events to be load",
    type=int,
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
    default="LocalPeakWindowSum",
    help="charge extractor method",
    type=str,
)
parser.add_argument(
    "--extractor_kwargs",
    default={"window_width": 16, "window_shift": 4},
    help="charge extractor kwargs",
    type=json.loads,
)

# verbosity argument
parser.add_argument(
    "-v",
    "--verbosity",
    help="set the verbosity level of logger",
    default="INFO",
    choices=["DEBUG","INFO","WARN","ERROR","CRITICAL"],
    type=str,
)

args = parser.parse_args()

def main(log,
    **kwargs,
):
    # print(kwargs)
    run_number = kwargs.pop("run_number")
    max_events = kwargs.pop(
        "max_events", [None for i in range(len(run_number))]
    )
    if max_events is None : 
        max_events = [None for i in range(len(run_number))]

    log.info(max_events)



    tool = WaveformsNectarCAMCalibrationTool()
    waveforms_kwargs = {}
    for key in tool.traits().keys() : 
        if key in kwargs.keys() : 
            waveforms_kwargs[key] = kwargs[key]

    tool = ChargesNectarCAMCalibrationTool()
    charges_kwargs = {}
    for key in tool.traits().keys() : 
        if key in kwargs.keys() : 
            charges_kwargs[key] = kwargs[key]

    log.info(f"WaveformsNectarCAMCalibrationTool kwargs are {waveforms_kwargs}")
    log.info(f"ChargesNectarCAMCalibrationTool kwargs are {charges_kwargs}")

    for _run_number,_max_events in zip(run_number,max_events) : 
        try : 
            if kwargs.get("reload_wfs",False) : 
                log.info("reloading waveforms")
                tool = WaveformsNectarCAMCalibrationTool(progress_bar = True,
                                                         run_number = _run_number,
                                                         max_events = _max_events,
                                                         **waveforms_kwargs
                                                         )
                tool.setup()
                tool.start()
                tool.finish()

                tool = ChargesNectarCAMCalibrationTool(progress_bar = True,
                                                        run_number = _run_number,
                                                        max_events = _max_events,
                                                        from_computed_waveforms = True,
                                                        **charges_kwargs
                                                         )
                tool.setup()
                tool.start()
                tool.finish()
            else : 
                log.info("trying to compute charges from waveforms yet extracted")
                tool = ChargesNectarCAMCalibrationTool(progress_bar = True,
                                                        run_number = _run_number,
                                                        max_events = _max_events,
                                                        from_computed_waveforms = True,
                                                        **charges_kwargs
                                                         )


                tool.setup()
                tool.start()
                tool.finish()
        except Exception as e : 
            log.error(e,exc_info = True)

    


if __name__ == "__main__":
    # run of interest
    # spe_run_number = [2633,2634,3784]
    # ff_run_number = [2608]
    # ped_run_number = [2609]
    # spe_nevents = [49227,49148,-1]

    args = parser.parse_args()
    kwargs = copy.deepcopy(vars(args))

    kwargs["log_level"] = args.verbosity
    
    os.makedirs(f"{os.environ.get('NECTARCHAIN_LOG','/tmp')}/{os.getpid()}/figures")
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

    main(log = log, **kwargs)
