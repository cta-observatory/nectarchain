import os
import sys

from matplotlib import pyplot as plt
import argparse
#from multiprocessing import Process
import time

from ctapipe.io import EventSource, EventSeeker
from ctapipe_io_nectarcam.constants import LOW_GAIN, HIGH_GAIN

from mean_waveforms import MeanWaveForms_HighLowGain
from mean_camera_display import MeanCameraDisplay_HighLowGain
from charge_integration import ChargeIntegration_HighLowGain
from trigger_statistics import TriggerStatistics
from camera_monitoring import CameraMonitoring



# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='My script')
parser.add_argument('-p', '--plot',
                     action='store_true',
                     help='Enables plots to be generated')
parser.add_argument('-n', '--noped',
                     action='store_true',
                     help='Enables pedestal subtraction in intergration')
parser.add_argument('-r', '--runnb',
                    help='Optional run number, automatically found on DIRAC',
                    type=int)
parser.add_argument('-i', '--input-files',
                    nargs='+',
                    help='Input files')

parser.add_argument('input_paths', help='Input paths')
parser.add_argument('output_paths', help='Output paths')

args, leftovers = parser.parse_known_args()

# Reading arguments, paths and plot-boolean
NectarPath = args.input_paths  # str(os.environ['NECTARDIR'])
print("Input file path:", NectarPath)

# Defining and printing the paths of the output files.
output_path = args.output_paths
print("Output path:", output_path)


# Defining and printing the paths of the input files.

#Read run automatocally is the -r option is provided...NOT TESTED YET
if args.runnb is not None:
    from nectarchain.calibration.container import utils
    dm = utils.DataManagement()
    _, filelist = dm.findrun(args.runnb)
    args.input_files = filelist
    path1 = filelist[0].name
else:  # OTHERWISE READ THE RUNS FROM ARGS
    if args.input_files is not None:
        path1 = args.input_files[0]
    else:
        print('Input files should be provided, exiting...')
        sys.exit(1)

#THE PATH OF INPUT FILES
path = f'{NectarPath}/{path1}'
print("Input files:")
print(path)
for arg in args.input_files[1:]:
    print(arg)

#Defining and priting the options
PlotFig = args.plot
noped = args.noped

print("Plot:", PlotFig)
print("Noped:", noped)




def GetName(RunFile):
    name = RunFile.split("/")[-1]
    name = name.split(".")[0] + "_" + name.split(".")[1]  # + '_' +name.split('.')[2]
    print(name)
    return name


    ParentFolderName = name.split('_')[0] + '_' + name.split('_')[1]
    ChildrenFolderName = './' + ParentFolderName +'/' + name + '_calib'
    FolderPath = output_path + 'output/%s/%s/' %(ChildrenFolderName, folder)

    if not os.path.exists(FolderPath):
        os.makedirs(FolderPath)

    return ParentFolderName, ChildrenFolderName, FolderPath


start = time.time()

# INITIATE
path = path
print(path)
cmap = 'gnuplot2'

# Read and seek
reader = EventSource(input_url=path)
seeker = EventSeeker(reader)
reader1 = EventSource(input_url=path, max_events=1)
# print(reader.file_list)

name = GetName(path)
ParentFolderName, ChildrenFolderName, FigPath = CreateFigFolder(name, 0)
ResPath = output_path + 'output/%s/%s' %(ChildrenFolderName, name)
#######################################################################################################################




                                                  ########################







#LIST OF PROCESSES TO RUN
#######################################################################################################################
a = TriggerStatistics(HIGH_GAIN)
b = MeanWaveForms_HighLowGain(HIGH_GAIN) #0 is for high gain and 1 is for low gain
c = MeanWaveForms_HighLowGain(LOW_GAIN)
d = MeanCameraDisplay_HighLowGain(HIGH_GAIN)
e = MeanCameraDisplay_HighLowGain(LOW_GAIN)
f = ChargeIntegration_HighLowGain(HIGH_GAIN)
g = ChargeIntegration_HighLowGain(LOW_GAIN)
h = CameraMonitoring(HIGH_GAIN)

processors = list()

processors.append(a)
processors.append(b)
processors.append(c)
processors.append(d)
processors.append(e)
processors.append(f)
processors.append(g)
processors.append(h)


# LIST OF DICT RESULTS
Results_MeanWaveForms_HighGain = {}
Results_MeanWaveForms_LowGain = {}
Results_MeanCameraDisplay_HighGain = {}
Results_MeanCameraDisplay_LowGain = {}
Results_ChargeIntegration_HighGain = {}
Results_ChargeIntegration_LowGain = {}
Results_TriggerStatistics = {}
Results_CameraMonitoring = {}

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
]
# NESTED_DICT_KEYS = ["Results_CameraMonitoring"]

# START
for p in processors:
    Chan, Samp = p.DefineForRun(reader1)
    break

for p in processors:
    p.ConfigureForRun(path, Chan, Samp, reader1)

for i, evt in enumerate(reader):
	for p in processors:
		p.ProcessEvent(evt, noped)
        
#for the rest of the event files
for arg in args.input_files[1:]:
    path2 = NectarPath + arg
    print(path2)

    reader=EventSource(input_url=path2)
    seeker = EventSeeker(reader)

    for i, evt in enumerate(reader):
        for p in processors:
            p.ProcessEvent(evt, noped)

for p in processors:
    p.FinishRun()

dict_num = 0
for p in processors:
    # True if want to compute plots, sedond true if want to save results
    NESTED_DICT[NESTED_DICT_KEYS[dict_num]] = p.GetResults()
    dict_num += 1


name = name #in order to allow to change the name easily
p.WriteAllResults(ResPath, NESTED_DICT) #if we want to write all results in 1 fits file we do this. 

#if -plot in args it will construct the figures and save them
if PlotFig == True:
    for p in processors:
        processor_figure_dict, processor_figure_name_dict  = p.PlotResults(name, FigPath)
    
        for fig_plot in processor_figure_dict:
            fig = processor_figure_dict[fig_plot]
            SavePath = processor_figure_name_dict[fig_plot]
            plt.gcf()
            fig.savefig(SavePath)
            
    plt.clf()
    plt.cla()
    plt.close()


end = time.time()
print("Processing time:", end - start)

# TODOS
# Reduce code by using loops: for figs and results
# MONGO: store results
