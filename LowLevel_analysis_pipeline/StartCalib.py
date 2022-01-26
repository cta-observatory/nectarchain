import os
import sys

from matplotlib import pyplot as plt

#from multiprocessing import Process

import time

from ctapipe.io import EventSource, EventSeeker


from MeanWaveForms import MeanWaveForms_HighLowGain
from MeanCameraDisplay import MeanCameraDisplay_HighLowGain
from ChargeIntegration import ChargeIntegration_HighLowGain
from TriggerStatistics import TriggerStatistics_HighLowGain


print(sys.argv)
path = sys.argv[1] # path of the Run file: ./NectarCAM.Run2720.0000.fits.fz

SystemPath = str(os.environ['NECTARPROCESSINGDIR'])
NectarPath = str(os.environ['NECTARDIR'])
DataPath = str(os.environ['NECTARDATA'])

def GetName(RunFile):
    name = RunFile.split('/')[-1]
    name = name.split('.')[0] + '_' + name.split('.')[1]# + '_' +name.split('.')[2]
    print(name)
    return name

def CreateFigFolder(name, type):
    if(type == 0):
        folder = 'Plots'

    ParentFolderName = name.split('_')[0] + '_' + name.split('_')[1]
    ChildrenFolderName = './' + ParentFolderName +'/' + name + '_calib'
    FolderPath = NectarPath + 'output/%s/%s/' %(ChildrenFolderName, folder)

    if not os.path.exists(FolderPath):
        os.makedirs(FolderPath)

    return ParentFolderName, ChildrenFolderName, FolderPath


start = time.time()

#INITIATE
#######################################################################################################################
path = path
cmap = 'gnuplot2'

#Read and seek
reader=EventSource(input_url=path)
seeker = EventSeeker(reader)
#print(reader.file_list)
name = GetName(path)
ParentFolderName, ChildrenFolderName, FigPath = CreateFigFolder(name, 0)
ResPath = NectarPath + 'output/%s/%s' %(ChildrenFolderName, name)
#######################################################################################################################




                                                  ########################







#LIST OF PROCESSES TO RUN
#######################################################################################################################
a = MeanWaveForms_HighLowGain(0) #0 is for high gain and 1 is for low gain
b = MeanWaveForms_HighLowGain(1)
c = MeanCameraDisplay_HighLowGain(0)
d = MeanCameraDisplay_HighLowGain(1)
e = ChargeIntegration_HighLowGain(0)
f = ChargeIntegration_HighLowGain(1)
g = TriggerStatistics_HighLowGain(0)

processors = list()

#processors.append(a)
#processors.append(b)
#processors.append(c)
#processors.append(d)
#processors.append(e)
#processors.append(f)
processors.append(g)
#######################################################################################################################





                                                 ########################




#LIST OF DICT RESULTS
#######################################################################################################################
Results_MeanWaveForms_HighGain ={}
Results_MeanWaveForms_LowGain = {}
Results_MeanCameraDisplay_HighGain = {}
Results_MeanCameraDisplay_LowGain = {}
Results_ChargeIntegration_HighGain = {}
Results_ChargeIntegration_LowGain = {}
Results_TriggerStatistics_HighGain = {}
Results_TriggerStatistics_LowGain = {}

NESTED_DICT = {} #The final results dictionary
#NESTED_DICT_KEYS = ["Results_MeanWaveForms_HighGain", "Results_MeanWaveForms_LowGain", "Results_MeanCameraDisplay_HighGain", "Results_MeanCameraDisplay_LowGain", "Results_ChargeIntegration_HighGain", "Results_ChargeIntegration_LowGain", "Results_TriggerStatistics"]
NESTED_DICT_KEYS = ["Results_TriggerStatistics"]


#######################################################################################################################




                                                  ########################



#START
#######################################################################################################################
for p in processors:
    Chan, Samp = p.DefineForRun(path)
    break
    
for p in processors:  
    p.ConfigureForRun(path, Chan, Samp)

for i, evt in enumerate(reader):
	for p in processors:
		p.ProcessEvent(evt)

for arg in sys.argv[2:]:
    reader=EventSource(input_url=arg)
    seeker = EventSeeker(reader)

    for i, evt in enumerate(reader):
        for p in processors:
            p.ProcessEvent(evt)

for p in processors:
    p.FinishRun()

dict_num = 0
for p in processors:
    NESTED_DICT[NESTED_DICT_KEYS[dict_num]] = p.GetResults() #True if want to compute plots, sedond true if want to save results
    dict_num += 1


name = name #in order to allow to change the name easily
p.WriteAllResults(ResPath, NESTED_DICT) #if we want to write all results in 1 pickle file we do this. 

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
print("Processing time:", end-start)

#TODOS
#######################################################################################################################
#change name of processors: event summary
#Reduce code by using loops: for figs and results
#MONGO: store results 
#Note2: Plot should not be called by GetResults. Plot should make and return figures not write them. Higher level class. 

