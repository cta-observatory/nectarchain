import math

import numpy as np
from astropy import time as astropytime
from dqm_summary_processor import DQMSummary
from matplotlib import pyplot as plt


class TriggerStatistics(DQMSummary):
    def __init__(self, gaink):
        self.k = gaink
        self.Pix = None
        self.Samp = None
        self.event_type = []
        self.event_times = []
        self.event_id = []
        self.run_times = []
        self.run_start1 = None
        self.run_start = None
        self.run_end = None
        self.event_ped_times = None
        self.event_phy_times = None
        self.event_other_times = None
        self.event_ped_id = None
        self.event_phy_id = None
        self.event_other_id = None
        self.event_wrong_times = None
        self.TriggerStat_Results_Dict = {}
        self.TriggerStat_Figures_Dict = {}
        self.TriggerStat_Figures_Names_Dict = {}


    def ConfigureForRun(self, path, Pix, Samp, Reader1):
        # define number of pixels and samples
        self.Pix = Pix
        self.Samp = Samp

    def ProcessEvent(self, evt, noped):
        trigger_type = evt.trigger.event_type.value
        trigger_time = evt.trigger.time.value
        trigger_id = evt.index.event_id
        trigger_run_time = evt.nectarcam.tel[0].svc.date

        self.event_type.append(trigger_type)
        self.event_times.append(trigger_time)
        self.event_id.append(trigger_id)
        self.run_times.append(trigger_run_time)

    def FinishRun(self):
        self.triggers = np.unique(self.event_type)
        pedestal_num = 32
        physical_num = 2

        self.event_id = np.array(self.event_id)
        self.run_times = np.array(self.run_times)
        self.event_times = np.array(self.event_times)
        self.event_type = np.array(self.event_type)

        self.run_start1 = self.run_times[self.event_id == np.min(self.event_id)]

        # Choose between the following two methods. time for max id can be sometimes 0.
        # self.run_end = self.event_times[self.event_id == np.max(self.event_id)]
        self.run_start = self.event_times[self.event_id == np.min(self.event_id)]
        # self.run_start = np.min(self.event_times)
        self.run_end = np.max(self.event_times)

        self.event_ped_times = self.event_times[self.event_type == pedestal_num]
        self.event_phy_times = self.event_times[self.event_type == physical_num]
        mask = (self.event_type != physical_num) & (self.event_type != pedestal_num)
        self.event_other_times = self.event_times[mask]

        self.event_ped_id = self.event_id[self.event_type == pedestal_num]
        self.event_phy_id = self.event_id[self.event_type == physical_num]
        mask = (self.event_type != physical_num) & (self.event_type != pedestal_num)
        self.event_other_id = self.event_id[mask]

        self.event_ped_times = self.event_ped_times[
            self.event_ped_times > self.run_start
        ]
        self.event_phy_times = self.event_phy_times[
            self.event_phy_times > self.run_start
        ]
        self.event_other_times = self.event_other_times[
            self.event_other_times > self.run_start
        ]
        self.event_wrong_times = self.event_times[self.event_times < self.run_start]
        self.event_times = self.event_times[self.event_times > self.run_start]

    def GetResults(self):
        self.TriggerStat_Results_Dict["TRIGGER-TYPES"] = self.triggers
        self.TriggerStat_Results_Dict[
            "TRIGGER-STATISTICS"
        ] = "All: %s, Physical: %s, Pedestals: %s, Others: %s, Wrong times: %s" % (
            len(self.event_times),
            len(self.event_phy_times),
            len(self.event_ped_times),
            len(self.event_other_times),
            len(self.event_wrong_times),
        )
        self.TriggerStat_Results_Dict[
            "START-TIMES"
        ] = "Run start time: %s, First event: %s, Last event: %s" % (
            self.run_start1,
            self.run_start,
            self.run_end,
        )
        return self.TriggerStat_Results_Dict

    def PlotResults(self, name, FigPath):
        w = 1
        n1 = np.array(self.event_times.max() - self.event_times.min(), dtype=object)
        n = math.ceil(n1 / w)

        fig1, ax = plt.subplots()
        ax.hist(
            self.event_type,
            100,
            color="r",
            linewidth=1,
            log=True,
            alpha=1,
            label="Trigger types",
        )
        for rect in ax.patches:
            height = rect.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
        plt.xticks(self.triggers)
        plt.title("Trigger Statistics")
        plt.xlabel("Trigger type")
        plt.grid()
        full_name = name + "_Trigger_Statistics.png"
        FullPath = FigPath + full_name

        self.TriggerStat_Figures_Dict["TRIGGER-STATISTICS"] = fig1
        self.TriggerStat_Figures_Names_Dict["TRIGGER-STATISTICS"] = FullPath
        plt.close()

        w = 15
        n1 = self.event_times.max() - self.event_times.min()
        n = math.ceil(n1 / w)

        fig2, ax = plt.subplots()
        ax.hist(
            self.event_times - self.run_start,
            n,
            color="grey",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="All events (%s + %s invisible)"
            % (len(self.event_times), len(self.event_wrong_times)),
        )
        ax.hist(
            self.event_phy_times - self.run_start,
            n,
            color="cyan",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="Physical events (%s)" % len(self.event_phy_times),
        )
        ax.hist(
            self.event_ped_times - self.run_start,
            n,
            color="orange",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="Pedestal events (%s)" % len(self.event_ped_times),
        )
        ax.hist(
            self.event_other_times - self.run_start,
            n,
            color="brown",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="Other events (%s)" % len(self.event_other_times),
        )
        plt.legend()
        plt.xlabel("Time")
        plt.grid()
        plt.title(
            "Trigger rates, run start at %s"
            % astropytime.Time(self.run_start, format="unix").iso
        )
        full_name = name + "_Event_rate.png"
        FullPath = FigPath + full_name

        self.TriggerStat_Figures_Dict["EVENT-TIME"] = fig2
        self.TriggerStat_Figures_Names_Dict["EVENT-TIME"] = FullPath
        plt.close()

        fig3, ax = plt.subplots()
        ax.hist(
            self.event_id,
            n,
            color="grey",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="All events (%s)" % len(self.event_id),
        )
        ax.hist(
            self.event_phy_id,
            n,
            color="orange",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="Physical events (%s)" % len(self.event_phy_id),
        )
        ax.hist(
            self.event_ped_id,
            n,
            color="cyan",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="Pedestal events (%s)" % len(self.event_ped_id),
        )
        ax.hist(
            self.event_other_id,
            n,
            color="brown",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="Other events (%s)" % len(self.event_other_id),
        )
        plt.legend()
        plt.xlabel("ID")
        plt.grid()
        plt.title(
            "Trigger IDs, run start at %s"
            % astropytime.Time(self.run_start, format="unix").iso
        )
        full_name = name + "_Event_IDs.png"
        FullPath = FigPath + full_name

        self.TriggerStat_Figures_Dict["EVENT-ID"] = fig3
        self.TriggerStat_Figures_Names_Dict["EVENT-ID"] = FullPath
        plt.close()

        return self.TriggerStat_Figures_Dict, self.TriggerStat_Figures_Names_Dict


# TODO
# continue GetResults
# adjust histogram displays
# Choose between starting since run star time or event start time ?
