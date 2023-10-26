import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

logging.getLogger("numba").setLevel(logging.WARNING)
logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    filename=f"{os.environ.get('NECTARCHAIN_LOG')}/{Path(__file__).stem}_{os.getpid()}.log",
)
log = logging.getLogger(__name__)

from nectarchain.data.container import (
    ChargeContainer,
    ChargeContainers,
    WaveformsContainer,
    WaveformsContainers,
)

parser = argparse.ArgumentParser(
    prog="load_wfs_compute_charge",
    description="This program load waveforms from fits.fz run files and compute charge",
)

# run numbers
parser.add_argument(
    "-s", "--spe_run_number", nargs="+", default=[], help="spe run list", type=int
)
parser.add_argument(
    "-p", "--ped_run_number", nargs="+", default=[], help="ped run list", type=int
)
parser.add_argument(
    "-f", "--ff_run_number", nargs="+", default=[], help="FF run list", type=int
)

# max events to be loaded
parser.add_argument(
    "--spe_max_events",
    nargs="+",
    # default=[],
    help="spe max events to be load",
    type=int,
)
parser.add_argument(
    "--ped_max_events",
    nargs="+",
    # default=[],
    help="ped max events to be load",
    type=int,
)
parser.add_argument(
    "--ff_max_events",
    nargs="+",
    # default=[],
    help="FF max events to be load",
    type=int,
)

# n_events in runs
parser.add_argument(
    "--spe_nevents",
    nargs="+",
    # default=[],
    help="spe n events to be load",
    type=int,
)
parser.add_argument(
    "--ped_nevents",
    nargs="+",
    # default=[],
    help="ped n events to be load",
    type=int,
)
parser.add_argument(
    "--ff_nevents",
    nargs="+",
    # default=[],
    help="FF n events to be load",
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
    "--split",
    action="store_true",
    default=False,
    help="split waveforms extraction with 1 file per fits.fz raw data file",
)

# extractor arguments
parser.add_argument(
    "--extractorMethod",
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
    default="info",
    choices=["fatal", "debug", "info", "warning"],
    type=str,
)

args = parser.parse_args()

# control shape of arguments lists
for arg in ["spe", "ff", "ped"]:
    run_number = eval(f"args.{arg}_run_number")
    max_events = eval(f"args.{arg}_max_events")
    nevents = eval(f"args.{arg}_nevents")

    if not (max_events is None) and len(max_events) != len(run_number):
        e = Exception(f"{arg}_run_number and {arg}_max_events must have same length")
        log.error(e, exc_info=True)
        raise e
    if not (nevents is None) and len(nevents) != len(run_number):
        e = Exception(f"{arg}_run_number and {arg}_nevents must have same length")
        log.error(e, exc_info=True)
        raise e


def load_wfs_no_split(i, runs_list, max_events, nevents, overwrite):
    """method to load waveforms without splitting

    Args:
        i (int): index in the run list
        runs_list (list): the run number list
        max_events (list): max_events list
        nevents (list): nevents list
        overwrite (bool): to overwrite

    Returns:
        WaveformsContainer: the output waveformsContainer
    """
    log.info("loading wfs not splitted")
    wfs = WaveformsContainer(runs_list[i], max_events=max_events[i], nevents=nevents[i])
    wfs.load_wfs()
    wfs.write(f"{os.environ['NECTARCAMDATA']}/waveforms/", overwrite=overwrite)
    return wfs


def load_wfs_charge_split(
    i, runs_list, max_events, overwrite, charge_childpath, extractor_kwargs
):
    """_summary_

    Args:
        i (int): index in the run list
        runs_list (list): the run number list
        max_events (list): max_events list
        nevents (list): nevents list
        overwrite (bool): to overwrite
        charge_childpath (str): the extraction method
        extractor_kwargs (dict): the charge extractor kwargs

    Returns:
        WaveformsContainers,ChargeContainers: the output WaveformsContainers and ChargeContainers
    """

    log.info("splitting wafevorms extraction with raw data list files")
    log.debug(f"creation of the WaveformsContainers")
    wfs = WaveformsContainers(runs_list[i], max_events=max_events[i], init_arrays=False)
    log.info(f"computation of charge with {charge_childpath}")
    log.info("splitting charge computation with raw data list files")
    charge = ChargeContainers()
    for j in range(wfs.nWaveformsContainer):
        log.debug(f"reader events for file {j}")
        wfs.load_wfs(index=j)
        wfs.write(
            f"{os.environ['NECTARCAMDATA']}/waveforms/", index=j, overwrite=overwrite
        )
        log.debug(f"computation of charge for file {j}")
        charge.append(
            ChargeContainer.from_waveforms(
                wfs.waveformsContainer[j], method=charge_childpath, **extractor_kwargs
            )
        )
        log.debug(f"deleting waveformsContainer at index {j} to free RAM")
        wfs.waveformsContainer[j] = WaveformsContainer.__new__(WaveformsContainer)

    log.info("merging charge")
    charge = charge.merge()
    return wfs, charge


def load_wfs_charge_split_from_wfsFiles(wfsFiles, charge_childpath, extractor_kwargs):
    """_summary_

    Args:
        wfsFiles (list): list of the waveformsContainer FITS files
        charge_childpath (str): the extraction method
        extractor_kwargs (dict): the charge extractor kwargs

    Returns:
        None,ChargeContainers: the output ChargeContainers (return tuple with None to keep same structure as load_wfs_charge_split)
    """
    charge = ChargeContainers()
    for j, file in enumerate(wfsFiles):
        log.debug(f"loading wfs from file {file}")
        wfs = WaveformsContainer.load(file)
        log.debug(f"computation of charge for file {file}")
        charge.append(
            ChargeContainer.from_waveforms(
                wfs, method=charge_childpath, **extractor_kwargs
            )
        )
        log.debug(f"deleting waveformsContainer from {file} to free RAM")
        del wfs.wfs_hg
        del wfs.wfs_lg
        del wfs.ucts_timestamp
        del wfs.ucts_busy_counter
        del wfs.ucts_event_counter
        del wfs.event_type
        del wfs.event_id
        del wfs.trig_pattern_all
        del wfs
        # gc.collect()

    log.info("merging charge")
    charge = charge.merge()
    return None, charge


def load_wfs_compute_charge(
    runs_list: list,
    reload_wfs: bool = False,
    overwrite: bool = False,
    charge_extraction_method: str = "FullWaveformSum",
    **kwargs,
) -> None:
    """this method is used to load waveforms from zfits files and compute charge with an user specified method

    Args:
        runs_list (list): list of runs for which you want to perfrom waveforms and charge extraction
        reload_wfs (bool, optional): argument used to reload waveforms from pre-loaded waveforms (in fits format) or from zfits file. Defaults to False.
        overwrite (bool, optional): to overwrite file on disk. Defaults to False.
        charge_extraction_method (str, optional): ctapipe charge extractor. Defaults to "FullWaveformSum".

    Raises:
        e : an error occurred during zfits loading from ctapipe EventSource
    """

    # print(runs_list)
    # print(charge_extraction_method)
    # print(overwrite)
    # print(reload_wfs)
    # print(kwargs)

    max_events = kwargs.get("max_events", [None for i in range(len(runs_list))])
    nevents = kwargs.get("nevents", [-1 for i in range(len(runs_list))])

    charge_childpath = kwargs.get("charge_childpath", charge_extraction_method)
    extractor_kwargs = kwargs.get("extractor_kwargs", {})

    split = kwargs.get("split", False)

    for i in range(len(runs_list)):
        log.info(f"treating run {runs_list[i]}")
        log.info("waveform computation")
        if not (reload_wfs):
            log.info(
                f"trying to load waveforms from {os.environ['NECTARCAMDATA']}/waveforms/"
            )
            try:
                if split:
                    files = glob.glob(
                        f"{os.environ['NECTARCAMDATA']}/waveforms/waveforms_run{runs_list[i]}_*.fits"
                    )
                    if len(files) == 0:
                        raise FileNotFoundError(f"no splitted waveforms found")
                    else:
                        wfs, charge = load_wfs_charge_split_from_wfsFiles(
                            files, charge_childpath, extractor_kwargs
                        )

                else:
                    wfs = WaveformsContainer.load(
                        f"{os.environ['NECTARCAMDATA']}/waveforms/waveforms_run{runs_list[i]}.fits"
                    )
            except FileNotFoundError as e:
                log.warning(
                    f"argument said to not reload waveforms from zfits files but computed waveforms not found at {os.environ['NECTARCAMDATA']}/waveforms/waveforms_run{runs_list[i]}.fits"
                )
                log.warning(f"reloading from zfits files")
                if split:
                    wfs, charge = load_wfs_charge_split(
                        i,
                        runs_list,
                        max_events,
                        overwrite,
                        charge_childpath,
                        extractor_kwargs,
                    )
                else:
                    wfs = load_wfs_no_split(
                        i, runs_list, max_events, nevents, overwrite
                    )
            except Exception as e:
                log.error(e, exc_info=True)
                raise e
        else:
            if split:
                wfs, charge = load_wfs_charge_split(
                    i,
                    runs_list,
                    max_events,
                    overwrite,
                    charge_childpath,
                    extractor_kwargs,
                )
            else:
                wfs = load_wfs_no_split(i, runs_list, max_events, nevents, overwrite)

        if not (split):
            log.info(f"computation of charge with {charge_childpath}")
            charge = ChargeContainer.from_waveforms(
                wfs, method=charge_childpath, **extractor_kwargs
            )
        del wfs

        charge.write(
            f"{os.environ['NECTARCAMDATA']}/charges/{path}/", overwrite=overwrite
        )
        del charge


def main(
    spe_run_number: list = [],
    ff_run_number: list = [],
    ped_run_number: list = [],
    **kwargs,
):
    # print(kwargs)

    spe_nevents = kwargs.pop("spe_nevents", [-1 for i in range(len(spe_run_number))])
    ff_nevents = kwargs.pop("ff_nevents", [-1 for i in range(len(ff_run_number))])
    ped_nevents = kwargs.pop("ped_nevents", [-1 for i in range(len(ped_run_number))])

    spe_max_events = kwargs.pop(
        "spe_max_events", [None for i in range(len(spe_run_number))]
    )
    ff_max_events = kwargs.pop(
        "ff_max_events", [None for i in range(len(ff_run_number))]
    )
    ped_max_events = kwargs.pop(
        "ped_max_events", [None for i in range(len(ped_run_number))]
    )

    runs_list = spe_run_number + ff_run_number + ped_run_number
    nevents = spe_nevents + ff_nevents + ped_nevents
    max_events = spe_max_events + ff_max_events + ped_max_events

    charge_extraction_method = kwargs.get("extractorMethod", "FullWaveformSum")

    load_wfs_compute_charge(
        runs_list=runs_list,
        charge_extraction_method=charge_extraction_method,
        nevents=nevents,
        max_events=max_events,
        **kwargs,
    )


if __name__ == "__main__":
    # run of interest
    # spe_run_number = [2633,2634,3784]
    # ff_run_number = [2608]
    # ped_run_number = [2609]
    # spe_nevents = [49227,49148,-1]

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

    arg = vars(args)
    log.info(f"arguments are : {arg}")

    key_to_pop = []
    for key in arg.keys():
        if arg[key] is None:
            key_to_pop.append(key)

    for key in key_to_pop:
        arg.pop(key)

    log.info(f"arguments passed to main are : {arg}")

    path = args.extractorMethod
    if args.extractorMethod in ["GlobalPeakWindowSum", "LocalPeakWindowSum"]:
        path += f"_{args.extractor_kwargs['window_shift']}-{args.extractor_kwargs['window_width']-args.extractor_kwargs['window_shift']}"
    elif args.extractorMethod in ["SlidingWindowMaxSum"]:
        path += f"_{args.extractor_kwargs['window_width']}"
    elif args.extractorMethod in ["FixedWindowSum"]:
        path += f"_{args.extractor_kwargs['peak_index']}_{args.extractor_kwargs['window_shift']}-{args.extractor_kwargs['window_width']-args.extractor_kwargs['window_shift']}"

    arg["path"] = path

    main(**arg)
