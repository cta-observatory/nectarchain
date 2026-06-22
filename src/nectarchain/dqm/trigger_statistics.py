import math
import os

import numpy as np
from astropy import time as astropytime
from ctapipe.containers import EventType
from matplotlib import pyplot as plt

from .dqm_summary_processor import DQMSummary

__all__ = ["TriggerStatistics"]


class TriggerStatistics(DQMSummary):
    def __init__(self, gaink, r0=False):
        self.k = gaink
        self.Pix = None
        self.Samp = None
        self.tel_id = None
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
        super().__init__(r0)

    def configure_for_run(self, path, Pix, Samp, Reader1, **kwargs):
        # define number of pixels and samples
        self.Pix = Pix
        self.Samp = Samp
        self.tel_id = Reader1.subarray.tel_ids[0]

    def process_event(self, evt, noped):
        trigger_type = evt.trigger.event_type.value
        trigger_time = evt.trigger.time.value
        trigger_id = evt.index.event_id
        trigger_run_time = evt.nectarcam.tel[self.tel_id].svc.date

        self.event_type.append(trigger_type)
        self.event_times.append(trigger_time)
        self.event_id.append(trigger_id)
        self.run_times.append(trigger_run_time)

    def finish_run(self):
        self.triggers = np.unique(self.event_type)

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

        # First, save wrong times (events before run_start) from unfiltered data
        is_wrong = self.event_times < self.run_start
        self.event_wrong_times = self.event_times[is_wrong]

        # Filter by run_start - only keep events after run_start
        valid = self.event_times > self.run_start
        self.event_id = self.event_id[valid]
        self.event_times = self.event_times[valid]
        self.event_type = self.event_type[valid]

        # Compute masks for subsets on the filtered arrays
        self._ped_mask = self.event_type == EventType.SKY_PEDESTAL.value
        # sky pedestal, event id 2
        self._phy_mask = self.event_type == EventType.SUBARRAY.value
        # standard physics stereo, event id 32
        # TODO: add ids and event time selection for
        # other event types, e.g., dark pedestals
        self._other_mask = ~self._ped_mask & ~self._phy_mask

    def get_results(self):
        self.TriggerStat_Results_Dict["TRIGGER-TYPES"] = self.triggers

        # Count using masks (no array creation)
        n_ped = self._ped_mask.sum()
        n_phy = self._phy_mask.sum()
        n_other = self._other_mask.sum()

        self.TriggerStat_Results_Dict["TRIGGER-STATISTICS"] = {
            "All": [len(self.event_times)],
            "Physical": [n_phy],
            "Pedestals": [n_ped],
            "Others": [n_other],
            "Wrong times": [len(self.event_wrong_times)],
        }

        from astropy.table import Table

        # Use masks directly on main arrays - no intermediate copies
        # ALL events
        self.TriggerStat_Results_Dict["TRIGGER-EVENTS-ALL"] = Table(
            {
                "Timestamps": self.event_times,
                "IDs": self.event_id,
            }
        )

        # PHY events
        if n_phy > 0:
            self.TriggerStat_Results_Dict["TRIGGER-EVENTS-PHY"] = Table(
                {
                    "Timestamps": self.event_times[self._phy_mask],
                    "IDs": self.event_id[self._phy_mask],
                }
            )

        # PED events
        if n_ped > 0:
            self.TriggerStat_Results_Dict["TRIGGER-EVENTS-PED"] = Table(
                {
                    "Timestamps": self.event_times[self._ped_mask],
                    "IDs": self.event_id[self._ped_mask],
                }
            )

        # OTHERS events
        if n_other > 0:
            self.TriggerStat_Results_Dict["TRIGGER-EVENTS-OTHERS"] = Table(
                {
                    "Timestamps": self.event_times[self._other_mask],
                    "IDs": self.event_id[self._other_mask],
                }
            )

        # WRONG times (only has Timestamps, no IDs)
        if len(self.event_wrong_times) > 0:
            self.TriggerStat_Results_Dict["TRIGGER-EVENTS-WRONG"] = Table(
                {
                    "Timestamps": self.event_wrong_times,
                }
            )

        self.TriggerStat_Results_Dict["START-TIMES"] = {
            "Run start time": [self.run_start1],
            "First event": [self.run_start],
            "Last event": [self.run_end],
        }
        return self.TriggerStat_Results_Dict

    def plot_results(self, name, fig_path):
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
        FullPath = os.path.join(fig_path, full_name)

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
        # Use masks directly on main arrays
        ax.hist(
            self.event_times[self._phy_mask] - self.run_start,
            n,
            color="cyan",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="Physical events (%s)" % self._phy_mask.sum(),
        )
        ax.hist(
            self.event_times[self._ped_mask] - self.run_start,
            n,
            color="orange",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="Pedestal events (%s)" % self._ped_mask.sum(),
        )
        ax.hist(
            self.event_times[self._other_mask] - self.run_start,
            n,
            color="brown",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="Other events (%s)" % self._other_mask.sum(),
        )
        plt.legend(loc="upper right")
        plt.xlabel("Time")
        plt.grid()
        plt.title(
            "Trigger rates, run start at %s"
            % astropytime.Time(self.run_start, format="unix").iso
        )
        full_name = name + "_Event_rate.png"
        FullPath = os.path.join(fig_path, full_name)

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
        # Use masks directly on main arrays
        ax.hist(
            self.event_id[self._phy_mask],
            n,
            color="orange",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="Physical events (%s)" % self._phy_mask.sum(),
        )
        ax.hist(
            self.event_id[self._ped_mask],
            n,
            color="cyan",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="Pedestal events (%s)" % self._ped_mask.sum(),
        )
        ax.hist(
            self.event_id[self._other_mask],
            n,
            color="brown",
            linewidth=1,
            log=True,
            alpha=0.5,
            label="Other events (%s)" % self._other_mask.sum(),
        )
        plt.legend(loc="upper right")
        plt.xlabel("ID")
        plt.grid()
        plt.title(
            "Trigger IDs, run start at %s"
            % astropytime.Time(self.run_start, format="unix").iso
        )
        full_name = name + "_Event_IDs.png"
        FullPath = os.path.join(fig_path, full_name)

        self.TriggerStat_Figures_Dict["EVENT-ID"] = fig3
        self.TriggerStat_Figures_Names_Dict["EVENT-ID"] = FullPath
        plt.close()

        return self.TriggerStat_Figures_Dict, self.TriggerStat_Figures_Names_Dict


# TODO
# continue get_results
# adjust histogram displays
# Choose between starting since run star time or event start time ?
