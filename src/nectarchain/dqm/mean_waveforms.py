from dqm_summary_processor import dqm_summary
from matplotlib import pyplot as plt
import numpy as np


class MeanWaveForms_HighLowGain(dqm_summary):

    def __init__(self, gaink):

        self.k = gaink
        return None

    def ConfigureForRun(self, path, Chan, Samp, Reader1):
        # define number of channels and samples
        self.Chan = Chan
        self.Samp = Samp

        # redefine everything
        self.Mwf = np.zeros((self.Chan, self.Samp))
        self.Mwf_ped = np.zeros((self.Chan, self.Samp))
        self.counter_evt = 0
        self.counter_ped = 0

        self.Mwf_average = np.zeros((self.Chan, self.Samp))
        self.Mwf_ped_average = np.zeros((self.Chan, self.Samp))
        self.Mwf_Mean_overChan = []
        self.Mwf_ped_Mean_overChan = []

        self.wf_list_plot = list(range(1, self.Samp + 1))  # used for plotting later on

        return None

    def ProcessEvent(self, evt):
        if evt.trigger.event_type.value == 32:  # count peds
            self.counter_ped += 1
        else:
            self.counter_evt += 1

        for ichan in range(self.Chan):
            # loop over channels, 1855 should be redefined as a variable
            if evt.trigger.event_type.value == 32:  # only peds now
                self.Mwf_ped[ichan, :] += evt.r0.tel[0].waveform[self.k][
                    ichan]  # fill channels one by one and sum them for peds only
            else:
                self.Mwf[ichan, :] += evt.r0.tel[0].waveform[self.k][
                    ichan]  # fill channels one by one and sum them
        return None

    def FinishRun(self):
        # if (self.k == 0):
        #   gain_c = 'High'
        # if (self.k == 1):
        #    gain_c = 'Low'

        self.Mwf_average = self.Mwf / self.counter_evt  # get average
        # get average over channels
        self.Mwf_Mean_overChan = np.mean(self.Mwf_average, axis=0)

        if self.counter_ped > 0:
            # get average pedestals
            self.Mwf_ped_average = self.Mwf_ped / self.counter_ped
            self.Mwf_ped_Mean_overChan = np.mean(self.Mwf_ped_average, axis=0)

        return None

    def GetResults(self):

        # INITIATE DICT
        self.MeanWaveForms_Results_Dict = {}

        # ASSIGN RESUTLS TO DICT
        if (self.k == 0):
            self.MeanWaveForms_Results_Dict[
                "WF-PHY-AVERAGE-HIGH-GAIN"] = self.Mwf_average
            self.MeanWaveForms_Results_Dict[
                "WF-PHY-AVERAGE-CHAN-HIGH-GAIN"] = self.Mwf_Mean_overChan
            if self.counter_ped > 0:
                self.MeanWaveForms_Results_Dict[
                    "WF-PED-AVERAGE-HIGH-GAIN"] = self.Mwf_ped_average
                self.MeanWaveForms_Results_Dict[
                    "WF-AVERAGE-PED-CHAN-HIGH-GAIN"] = self.Mwf_ped_Mean_overChan

        if (self.k == 1):
            self.MeanWaveForms_Results_Dict["WF-AVERAGE-LOW-GAIN"] = self.Mwf_average
            self.MeanWaveForms_Results_Dict[
                "WF-AVERAGE-CHAN-LOW-GAIN"] = self.Mwf_Mean_overChan
            if self.counter_ped > 0:
                self.MeanWaveForms_Results_Dict[
                    "WF-PHY-PED-AVERAGE-LOW-GAIN"] = self.Mwf_ped_average
                self.MeanWaveForms_Results_Dict[
                    "WF-PHY-AVERAGE-PED-CHAN-LOW-GAIN"] = self.Mwf_ped_Mean_overChan

        return self.MeanWaveForms_Results_Dict

    def PlotResults(self, name, FigPath):
        self.MeanWaveForms_Figures_Dict = {}
        self.MeanWaveForms_Figures_Names_Dict = {}

        wf_list = np.array(self.wf_list_plot)

        counter_fig = 0
        colors = ['blue', 'red']
        # colors2 = ['cyan', 'orange']
        titles = ['Physical', 'Pedestals']

        # Mean_plot_array = [self.Mwf_Mean_overChan, self.Mwf_ped_Mean_overChan]

        # Set characters of gain: high or lo
        if (self.k == 0):
            gain_c = 'High'
        if (self.k == 1):
            gain_c = 'Low'

        full_fig, full_ax = plt.subplots()
        if self.counter_ped > 0:
            array_plot = [self.Mwf_average, self.Mwf_ped_average]
        else:
            array_plot = [self.Mwf_average]

        for x in array_plot:

            part_fig, part_ax = plt.subplots()

            for ichan in range(self.Chan):
                full_ax.plot(wf_list, x[ichan, :], color=colors[counter_fig],
                             alpha=0.005, linewidth=1)
                part_ax.plot(wf_list, x[ichan, :], color=colors[counter_fig],
                             alpha=0.005, linewidth=1)

            # Mean_plot = Mean_plot_array[counter_fig]

            # full_ax_return = full_ax.plot(wf_list, Mean_plot,
            #                              color=colors2[counter_fig], alpha=1,
            #                              linewidth=3,
            #                              label='Mean ' + titles[counter_fig])
            # part_ax_return = part_ax.plot(wf_list, Mean_plot,
            #                              color=colors2[counter_fig], alpha=1,
            #                              linewidth=3,
            #                              label='Mean ' + titles[counter_fig])
            part_ax.set_title(
                'Mean Waveforms %s (%s Gain)' % (titles[counter_fig], gain_c))
            part_ax.set_xlabel('Samples')
            part_ax.set_ylabel('Amplitude (DC)')
            part_ax.legend()
            part_ax.grid()

            part_name = name + '_MeanWaveforms_%s_%sGain.png' % (
                titles[counter_fig], gain_c
            )
            PartPath = FigPath + part_name

            self.MeanWaveForms_Figures_Dict[
                "FIGURE-WF-%s-%s-GAIN" % (titles[counter_fig], gain_c)] = part_fig
            self.MeanWaveForms_Figures_Names_Dict[
                "FIGURE-WF-%s-%s-GAIN" % (titles[counter_fig], gain_c)] = PartPath

            plt.close()

            counter_fig += 1

        full_ax.set_title('Mean Waveforms Combined Plot (%s Gain)' % gain_c)
        full_ax.set_xlabel('Samples')
        full_ax.set_ylabel('Amplitude (DC)')
        full_ax.legend()
        full_ax.grid()

        full_name = name + '_MeanWaveforms_CombinedPlot_%sGain.png' % gain_c
        FullPath = FigPath + full_name
        self.MeanWaveForms_Figures_Dict[
            "FIGURE-WF-COMBINED-%s-GAIN" % gain_c] = full_fig
        self.MeanWaveForms_Figures_Names_Dict[
            "FIGURE-WF-COMBINED-%s-GAIN" % gain_c] = FullPath

        plt.close()

        return self.MeanWaveForms_Figures_Dict, self.MeanWaveForms_Figures_Names_Dict
