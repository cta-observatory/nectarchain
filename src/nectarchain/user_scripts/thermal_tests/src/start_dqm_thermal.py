import argparse
import json
import os
import sys
import time

# ctapipe imports
from ctapipe.io import EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN, LOW_GAIN
from matplotlib import pyplot as plt
from tqdm import tqdm
from traitlets.config import Config

from nectarchain.dqm.charge_integration import ChargeIntegrationHighLowGain
from nectarchain.dqm.db_utils import DQMDB
from nectarchain.makers import ChargesNectarCAMCalibrationTool


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

    parser.add_argument("input_paths", help="Input paths")
    parser.add_argument("output_paths", help="Output paths")

    args, leftovers = parser.parse_known_args()

    # Reading arguments, paths and plot-boolean
    NectarPath = args.input_paths
    print("Input file path:", NectarPath)

    # Defining and printing the paths of the output files.
    output_path = args.output_paths
    print("Output path:", output_path)

    if args.runnb is not None:
        # Grab runs automatically from DIRAC is the -r option is provided
        from nectarchain.data.management import DataManagement

        dm = DataManagement()
        _, filelist = dm.findrun(args.runnb)
        args.input_files = [s.name for s in filelist]
    elif args.input_files is None:
        print("Input files should be provided, exiting...")
        sys.exit(1)

    # OTHERWISE READ THE RUNS FROM ARGS
    path1 = args.input_files[0]

    # THE PATH OF INPUT FILES
    path = f"{NectarPath}/runs/{path1}"
    print("Input files:")
    print(path)
    for arg in args.input_files[1:]:
        print(arg)

    # Defining and printing the options
    PlotFig = args.plot
    noped = args.noped
    method = args.method
    extractor_kwargs = args.extractor_kwargs

    print("Plot:", PlotFig)
    print("Noped:", noped)
    print("method:", method)
    print("extractor_kwargs:", extractor_kwargs)

    kwargs = {"method": method, "extractor_kwargs": extractor_kwargs}
    charges_kwargs = {}
    tool = ChargesNectarCAMCalibrationTool()
    for key in tool.traits().keys():
        if key in kwargs.keys():
            charges_kwargs[key] = kwargs[key]
    print(charges_kwargs)

    def GetName(RunFile):
        name = RunFile.split("/")[-1]
        name = (
            name.split(".")[0] + "_" + name.split(".")[1]
        )  # + '_' +name.split('.')[2]
        print(name)
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
    ParentFolderName, ChildrenFolderName, FigPath = CreateFigFolder(name, 0)
    ResPath = f"{output_path}/output/{ChildrenFolderName}/{name}"

    # LIST OF PROCESSES TO RUN
    ####################################################################################
    processors = [
        ChargeIntegrationHighLowGain(HIGH_GAIN),
        ChargeIntegrationHighLowGain(LOW_GAIN),
    ]

    # LIST OF DICT RESULTS
    NESTED_DICT = {}  # The final results dictionary

    NESTED_DICT_KEYS = [
        "Results_ChargeIntegration_HighGain",
        "Results_ChargeIntegration_LowGain",
    ]

    # START
    for p in processors:
        Pix, Samp = p.DefineForRun(reader1)
        break

    for p in processors:
        p.ConfigureForRun(path, Pix, Samp, reader1, **charges_kwargs)

    for evt in tqdm(
        reader, total=args.max_events if args.max_events else len(reader), unit="ev"
    ):
        for p in processors:
            p.ProcessEvent(evt, noped)

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
                    p.ProcessEvent(evt, noped)

    for p in processors:
        p.FinishRun()

    dict_num = 0
    for p in processors:
        NESTED_DICT[NESTED_DICT_KEYS[dict_num]] = p.GetResults()
        dict_num += 1

    # Write all results in 1 fits file:
    p.WriteAllResults(ResPath, NESTED_DICT)
    if args.write_db:
        db = DQMDB(read_only=False)
        if db.insert(name, NESTED_DICT):
            db.commit_and_close()
        else:
            db.abort_and_close()

    # if plot option in arguments, it will construct the figures and save them
    if PlotFig:
        for p in processors:
            processor_figure_dict, processor_figure_name_dict = p.PlotResults(
                name, FigPath
            )

            for fig_plot in processor_figure_dict:
                fig = processor_figure_dict[fig_plot]
                SavePath = processor_figure_name_dict[fig_plot]
                fig.savefig(SavePath)
                plt.close()

    end = time.time()
    print(f"Processing time: {end-start:.2f} s.")

    # TODO
    # Reduce code by using loops: for figs and results


if __name__ == "__main__":
    main()
