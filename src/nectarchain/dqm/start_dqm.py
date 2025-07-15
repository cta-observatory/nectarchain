import argparse
import json
import logging
import os
import sys
import time

# ctapipe imports
from ctapipe_io_nectarcam import LightNectarCAMEventSource as EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN, LOW_GAIN
from matplotlib import pyplot as plt
from tqdm import tqdm
from traitlets.config import Config

from nectarchain.dqm.camera_monitoring import CameraMonitoring
from nectarchain.dqm.charge_integration import ChargeIntegrationHighLowGain
from nectarchain.dqm.db_utils import DQMDB
from nectarchain.dqm.mean_camera_display import MeanCameraDisplayHighLowGain
from nectarchain.dqm.mean_waveforms import MeanWaveFormsHighLowGain
from nectarchain.dqm.pixel_participation import PixelParticipationHighLowGain
from nectarchain.dqm.pixel_timeline import PixelTimelineHighLowGain
from nectarchain.dqm.trigger_statistics import TriggerStatistics
from nectarchain.makers import ChargesNectarCAMCalibrationTool

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


def main():
    """
    Main DQM script
    """
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="NectarCAM Data Quality Monitoring tool"
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Enables plots to be generated"
    )
    parser.add_argument(
        "--write-db", action="store_true", help="Write DQM output in DQM ZODB data base"
    )
    parser.add_argument(
        "-n",
        "--noped",
        action="store_true",
        help="Enables pedestal subtraction in charge integration",
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
        default='{"window_shift": 4, "window_width": 16}',
        help="charge extractor kwargs",
        type=json.loads,
    )

    parser.add_argument(
        "-r",
        "--runnb",
        help="Optional run number, automatically found on DIRAC",
        type=int,
    )
    parser.add_argument(
        "--r0", action="store_true", help="Disable all R0->R1 corrections"
    )
    parser.add_argument(
        "--max-events",
        default=None,
        type=int,
        help="Maximum number of events to loop through in each run slice",
    )
    parser.add_argument("-i", "--input-files", nargs="+", help="Local input files")
    parser.add_argument("--log", default="info", help="debug output", type=str)

    parser.add_argument("input_paths", help="Input paths")
    parser.add_argument("output_paths", help="Output paths")

    args, leftovers = parser.parse_known_args()

    log.setLevel(args.log.upper())

    # Reading arguments, paths and plot-boolean
    NectarPath = args.input_paths
    log.info(f"Input file path: {NectarPath}")

    # Defining and printing the paths of the output files.
    output_path = args.output_paths
    log.info(f"Output path: {output_path}")

    if args.runnb is not None:
        # Grab runs automatically from DIRAC is the -r option is provided
        from nectarchain.data.management import DataManagement

        dm = DataManagement()
        _, filelist = dm.findrun(args.runnb)
        args.input_files = [s.name for s in filelist]
    elif args.input_files is None:
        log.error("Input files should be provided, exiting...")
        sys.exit(1)

    # OTHERWISE READ THE RUNS FROM ARGS
    path1 = args.input_files[0]

    # THE PATH OF INPUT FILES
    path = f"{NectarPath}/runs/{path1}"
    log.debug(f"Input files:\n{path}")
    for arg in args.input_files[1:]:
        log.debug(arg)

    # Defining and printing the options
    PlotFig = args.plot
    noped = args.noped
    method = args.method
    extractor_kwargs = args.extractor_kwargs

    log.info(f"Plot: {PlotFig}")
    log.info(f"Noped: {noped}")
    log.info(f"method: {method}")
    log.info(f"extractor_kwargs: {extractor_kwargs}")

    kwargs = {"method": method, "extractor_kwargs": extractor_kwargs}
    charges_kwargs = {}
    tool = ChargesNectarCAMCalibrationTool()
    for key in tool.traits().keys():
        if key in kwargs.keys():
            charges_kwargs[key] = kwargs[key]
    log.info(f"charges_kwargs: {charges_kwargs}")

    def GetName(RunFile):
        name = RunFile.split("/")[-1]
        name = name.split(".")[0] + "_" + name.split(".")[1]
        log.debug(name)
        return name

    def CreateFigFolder(name, type):
        if type == 0:
            folder = "Plots"

        ParentFolderName = name.split("_")[0] + "_" + name.split("_")[1]
        ChildrenFolderName = "./" + ParentFolderName + "/" + name + "_calib"
        FolderPath = f"{output_path}/output/{ChildrenFolderName}/{folder}"

        if not os.path.exists(FolderPath):
            os.makedirs(FolderPath)

        return ParentFolderName, ChildrenFolderName, FolderPath

    start = time.time()

    # INITIATE
    path = path
    print(path)

    # Read and seek
    config = None
    if args.r0:
        config = Config(
            dict(
                NectarCAMEventSource=dict(
                    NectarCAMR0Corrections=dict(
                        calibration_path=None,
                        apply_flatfield=False,
                        select_gain=False,
                    )
                )
            )
        )

    reader = EventSource(input_url=path, config=config, max_events=args.max_events)
    reader1 = EventSource(input_url=path, config=config, max_events=1)
    # print(reader.file_list)

    name = GetName(path)
    ParentFolderName, ChildrenFolderName, fig_path = CreateFigFolder(name, 0)
    ResPath = f"{output_path}/output/{ChildrenFolderName}/{name}"

    # LIST OF PROCESSES TO RUN
    ####################################################################################
    processors = [
        TriggerStatistics(HIGH_GAIN),
        MeanWaveFormsHighLowGain(HIGH_GAIN),
        MeanWaveFormsHighLowGain(LOW_GAIN),
        MeanCameraDisplayHighLowGain(HIGH_GAIN),
        MeanCameraDisplayHighLowGain(LOW_GAIN),
        ChargeIntegrationHighLowGain(HIGH_GAIN),
        ChargeIntegrationHighLowGain(LOW_GAIN),
        CameraMonitoring(HIGH_GAIN),
        PixelParticipationHighLowGain(HIGH_GAIN),
        PixelParticipationHighLowGain(LOW_GAIN),
        PixelTimelineHighLowGain(HIGH_GAIN),
        PixelTimelineHighLowGain(LOW_GAIN),
    ]

    # LIST OF DICT RESULTS
    NESTED_DICT = {}  # The final results dictionary

    NESTED_DICT_KEYS = [
        "Results_TriggerStatistics",
        "Results_MeanWaveForms_HighGain",
        "Results_MeanWaveForms_LowGain",
        "Results_MeanCameraDisplay_HighGain",
        "Results_MeanCameraDisplay_LowGain",
        "Results_ChargeIntegration_HighGain",
        "Results_ChargeIntegration_LowGain",
        "Results_CameraMonitoring",
        "Results_PixelParticipation_HighGain",
        "Results_PixelParticipation_LowGain",
        "Results_PixelTimeline_HighGain",
        "Results_PixelTimeline_LowGain",
    ]

    # START
    for p in processors:
        Pix, Samp = p.define_for_run(reader1)
        break

    for p in processors:
        p.configure_for_run(path, Pix, Samp, reader1, **charges_kwargs)

    for evt in tqdm(
        reader, total=args.max_events if args.max_events else len(reader), unit="ev"
    ):
        for p in processors:
            p.process_event(evt, noped)

    # for the rest of the event files
    for arg in args.input_files[1:]:
        path2 = f"{NectarPath}/runs/{arg}"
        print(path2)

        with EventSource(
            input_url=path2, config=config, max_events=args.max_events
        ) as reader:
            for evt in tqdm(
                reader,
                total=args.max_events if args.max_events else len(reader),
                unit="ev",
            ):
                for p in processors:
                    p.process_event(evt, noped)

    for p in processors:
        p.finish_run()

    dict_num = 0
    for p in processors:
        NESTED_DICT[NESTED_DICT_KEYS[dict_num]] = p.get_results()
        dict_num += 1

    # Write all results in 1 fits file:
    p.write_all_results(ResPath, NESTED_DICT)
    if args.write_db:
        db = DQMDB(read_only=False)
        if db.insert(name, NESTED_DICT):
            db.commit_and_close()
        else:
            db.abort_and_close()

    # if plot option in arguments, it will construct the figures and save them
    if PlotFig:
        for p in processors:
            processor_figure_dict, processor_figure_name_dict = p.plot_results(
                name, fig_path
            )

            for fig_plot in processor_figure_dict:
                fig = processor_figure_dict[fig_plot]
                SavePath = processor_figure_name_dict[fig_plot]
                fig.savefig(SavePath)
                plt.close()

    end = time.time()
    print(f"Processing time: {end-start:.2f} s.")


if __name__ == "__main__":
    main()
