
from Processor import *

class TriggerStatistics_HighLowGain(Processor):

    def __init__(self, gaink):

        self.k = gaink
        return None

    def ConfigureForRun(self,path, Chan, Samp):
        #define number of channels and samples
        self.Chan = Chan
        self.Samp= Samp
        
        self.image_triggers = []

    def ProcessEvent(self, evt):
        trigger_type = evt.trigger.event_type
        self.image_triggers.append(trigger_type)

    def FinishRun(self):
        self.image_triggers = np.array(self.image_triggers)
        self.image_trigger_1 = self.image_triggers[self.image_triggers == 1]
        self.image_trigger_32 = self.image_triggers[self.image_triggers == 32]
        self.image_trigger_129 = self.image_triggers[self.image_triggers == 129]

    
    def GetResults(self):
        if (self.k==0):
            gain_c = 'High'
        if (self.k ==1):
            gain_c = 'Low'

        self.TriggerStat_Results_Dict = {}
        self.Trigger_stats_array = [len(self.image_trigger_1), len(self.image_trigger_32), len(self.image_trigger_129)]
        self.TriggerStat_Results_Dict["TRIGGER-STATS-%s-GAIN" %gain_c] = self.Trigger_stats_array
        return self.TriggerStat_Results_Dict


    def PlotResults(self,name,FigPath):

        if (self.k==0):
            gain_c = 'High'
        if (self.k ==1):
            gain_c = 'Low'

        self.TriggerStat_Figures_Dict = {}
        self.TriggerStat_Figures_Names_Dict = {}
        fig, ax = plt.subplots()
        ax.hist(self.image_triggers, 100, color = 'r', linewidth=1, log = True, alpha = 1, label = 'Trigger types')
        for rect in ax.patches:
            height = rect.get_height()
            ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 

        plt.xticks([1,32,129], ["1","32","129"])
        plt.title("Trigger Statistics %s Gain" %gain_c)
        plt.xlabel("Trigger type")
        full_name = name + '_Trigger_Statistics_%sGain.png' %gain_c  
        FullPath = FigPath +full_name

        self.TriggerStat_Figures_Dict["TRIGGER-STATISTICS-%s-GAIN" %gain_c] = fig
        self.TriggerStat_Figures_Names_Dict["TRIGGER-STATISTICS-%s-GAIN" %gain_c] = FullPath

        return  self.TriggerStat_Figures_Dict, self.TriggerStat_Figures_Names_Dict



