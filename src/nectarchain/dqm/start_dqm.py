import argparse
import os
import sys
import time

from camera_monitoring import CameraMonitoring
from charge_integration import ChargeIntegrationHighLowGain
from pixel_participation import PixelParticipation_HighLowGain
from pixel_timeline import PixelTimeline_HighLowGain
from mean_camera_display import MeanCameraDisplay_HighLowGain
from mean_waveforms import MeanWaveFormsHighLowGain
from trigger_statistics import TriggerStatistics

from ctapipe.io import EventSeeker, EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN, LOW_GAIN
from db_utils import DQMDB
from matplotlib import pyplot as plt


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="NectarCAM Data Quality Monitoring tool")
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
parser.add_argument(
    "-r", "--runnb", help="Optional run number, automatically found on DIRAC", type=int
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


# Defining and printing the paths of the input files.

if args.runnb is not None:
    # Grab runs automatically from DIRAC is the -r option is provided
    from nectarchain.data.container import utils

    dm = utils.DataManagement()
    _, filelist = dm.findrun(args.runnb)
    args.input_files = [s.name for s in filelist]
elif args.input_files is None:
    print("Input files should be provided, exiting...")
    sys.exit(1)

# OTHERWISE READ THE RUNS FROM ARGS
path1 = args.input_files[0]

# THE PATH OF INPUT FILES
path = f"{NectarPath}/{path1}"
print("Input files:")
print(path)
for arg in args.input_files[1:]:
    print(arg)

# Defining and priting the options
PlotFig = args.plot
noped = args.noped

print("Plot:", PlotFig)
print("Noped:", noped)


def GetName(RunFile):
    name = RunFile.split("/")[-1]
    name = name.split(".")[0] + "_" + name.split(".")[1]  # + '_' +name.split('.')[2]
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
cmap = "gnuplot2"

# Read and seek
reader = EventSource(input_url=path)
seeker = EventSeeker(reader)
reader1 = EventSource(input_url=path, max_events=1)
# print(reader.file_list)

name = GetName(path)
ParentFolderName, ChildrenFolderName, FigPath = CreateFigFolder(name, 0)
ResPath = f"{output_path}/output/{ChildrenFolderName}/{name}"


# LIST OF PROCESSES TO RUN
########################################################################################
a = TriggerStatistics(HIGH_GAIN)
b = MeanWaveFormsHighLowGain(HIGH_GAIN)
c = MeanWaveFormsHighLowGain(LOW_GAIN)
d = MeanCameraDisplay_HighLowGain(HIGH_GAIN)
e = MeanCameraDisplay_HighLowGain(LOW_GAIN)
f = ChargeIntegrationHighLowGain(HIGH_GAIN)
g = ChargeIntegrationHighLowGain(LOW_GAIN)
h = CameraMonitoring(HIGH_GAIN)
i = PixelParticipation_HighLowGain(HIGH_GAIN)
j = PixelParticipation_HighLowGain(LOW_GAIN)
k = PixelTimeline_HighLowGain(HIGH_GAIN)
l = PixelTimeline_HighLowGain(LOW_GAIN)

processors = list()

processors.append(a)
processors.append(b)
processors.append(c)
processors.append(d)
processors.append(e)
processors.append(f)
processors.append(g)
processors.append(h)
processors.append(i)
processors.append(j)
processors.append(k)
processors.append(l)


# LIST OF DICT RESULTS
Results_TriggerStatistics = {}
Results_MeanWaveForms_HighGain = {}
Results_MeanWaveForms_LowGain = {}
Results_MeanCameraDisplay_HighGain = {}
Results_MeanCameraDisplay_LowGain = {}
Results_ChargeIntegration_HighGain = {}
Results_ChargeIntegration_LowGain = {}
Results_CameraMonitoring = {}
Results_PixelParticipation_HighGain = {}
Results_PixelParticipation_LowGain = {}
Results_PixelTimeline_HighGain = {}
Results_PixelTimeline_LowGain = {}

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

#NESTED_DICT_KEYS = ["Results_PixelParticipation_HighGain", "Results_PixelTimeline_HighGain"]

# START
for p in processors:
    Pix, Samp = p.DefineForRun(reader1)
    break

for p in processors:
    p.ConfigureForRun(path, Pix, Samp, reader1)

for i, evt in enumerate(reader):
    for p in processors:
        p.ProcessEvent(evt, noped)

# for the rest of the event files
for arg in args.input_files[1:]:
    path2 = f"{NectarPath}/{arg}"
    print(path2)

    reader = EventSource(input_url=path2)
    seeker = EventSeeker(reader)

    for i, evt in enumerate(reader):
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
        processor_figure_dict, processor_figure_name_dict = p.PlotResults(name, FigPath)

        for fig_plot in processor_figure_dict:
            fig = processor_figure_dict[fig_plot]
            SavePath = processor_figure_name_dict[fig_plot]
            fig.savefig(SavePath)
            plt.close()


end = time.time()
print("Processing time:", end - start)

# TODO
# Reduce code by using loops: for figs and results
