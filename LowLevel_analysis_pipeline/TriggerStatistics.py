
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
        self.event_times = []

    def ProcessEvent(self, evt):
        trigger_type = evt.trigger.event_type
        trigger_time = evt.trigger.time


        self.image_triggers.append(trigger_type)
        self.event_times.append(trigger_time)

        self.run_start = evt.nectarcam.tel[0].svc.date

    def FinishRun(self):
        self.image_triggers = np.array(self.image_triggers)
        self.image_trigger_1 = self.image_triggers[self.image_triggers == 1]
        self.image_trigger_32 = self.image_triggers[self.image_triggers == 32]
        #self.image_trigger_129 = self.image_triggers[self.image_triggers == 129]

        self.triggers = np.unique(self.image_triggers)


        self.event_times = np.array(self.event_times)

        self.event_ped_times = self.event_times[self.image_triggers == 32]
        self.event_phy_times = self.event_times[self.image_triggers == 1]

        mask = ((self.image_triggers != 1) & (self.image_triggers != 32))
        self.event_other_times = self.event_times[mask]

        self.event_ped_times = self.event_ped_times[self.event_ped_times > self.run_start]
        self.event_phy_times = self.event_phy_times[self.event_phy_times > self.run_start]
        self.event_other_times = self.event_other_times[self.event_other_times > self.run_start]
        self.event_wrong_times = self.event_times[self.event_times < self.run_start]
        self.event_times = self.event_times[self.event_times > self.run_start]

    
    def GetResults(self):
        self.TriggerStat_Results_Dict = {}
        self.TriggerStat_Results_Dict["TRIGGER-STATS"] = self.triggers
        return self.TriggerStat_Results_Dict


    def PlotResults(self,name,FigPath):

        self.TriggerStat_Figures_Dict = {}
        self.TriggerStat_Figures_Names_Dict = {}
        fig1, ax = plt.subplots()
        ax.hist(self.image_triggers, 100, color = 'r', linewidth=1, log = True, alpha = 1, label = 'Trigger types')
        for rect in ax.patches:
            height = rect.get_height()
            ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 

        plt.xticks(self.triggers)
        plt.title("Trigger Statistics")
        plt.xlabel("Trigger type")
        plt.grid()
        full_name = name + '_Trigger_Statistics.png'  
        FullPath = FigPath +full_name

        self.TriggerStat_Figures_Dict["TRIGGER-STATISTICS"] = fig1
        self.TriggerStat_Figures_Names_Dict["TRIGGER-STATISTICS"] = FullPath


        fig2, ax = plt.subplots()
        ax.hist(self.event_times, 50, color = 'grey', linewidth=1, log = True, alpha = 0.2, label = 'All events (%s + %s invisible)' %(len(self.event_times), len(self.event_wrong_times)))
        ax.hist(self.event_phy_times, 50, color = 'cyan', linewidth=1, log = True, alpha = 0.2, label = 'Pedestal events (%s)' %len(self.event_ped_times))
        ax.hist(self.event_ped_times, 50, color = 'orange', linewidth=1, log = True, alpha = 0.2, label = 'Physical events (%s)' %len(self.event_phy_times))
        ax.hist(self.event_other_times, 50, color = 'brown', linewidth=1, log = True, alpha = 0.2, label = 'Other events (%s)' %len(self.event_other_times))
        plt.legend()
        plt.xlabel("Time")
        plt.grid()
        plt.title("Trigger rates, run start at %s" %astropytime.Time(self.run_start, format='unix').iso)
        full_name = name + '_Event_rate.png'  
        FullPath = FigPath +full_name

        self.TriggerStat_Figures_Dict["EVENT-TIME"] = fig2
        self.TriggerStat_Figures_Names_Dict["EVENT-TIME"] = FullPath

        return  self.TriggerStat_Figures_Dict, self.TriggerStat_Figures_Names_Dict

#TODO
#GET TRIGGER RATES VS EVENT NUMBER
#continue GetResults
#adjust histogram displays

