import numpy as np
from dqm_summary_processor import DQMSummary
from matplotlib import pyplot as plt


class MeanWaveFormsHighLowGain(DQMSummary):
    def __init__(self, gaink):
        self.k = gaink
        self.Pix = None
        self.Samp = None
        self.Mwf = None
        self.Mwf_ped = None
        self.counter_evt = None
        self.counter_ped = None
        self.Mwf_average = None
        self.Mwf_ped_average = None
        self.Mwf_Mean_overPix = []
        self.Mwf_ped_Mean_overPix = []
        self.MeanWaveForms_Results_Dict = {}
        self.MeanWaveForms_Figures_Dict = {}
        self.MeanWaveForms_Figures_Names_Dict = {}

    def ConfigureForRun(self, path, Pix, Samp, Reader1):
        # define number of pixels and samples
        self.Pix = Pix
        self.Samp = Samp
        self.Mwf = np.zeros((self.Pix, self.Samp))
        self.Mwf_ped = np.zeros((self.Pix, self.Samp))
        self.counter_evt = 0
        self.counter_ped = 0
        self.Mwf_average = np.zeros((self.Pix, self.Samp))
        self.Mwf_ped_average = np.zeros((self.Pix, self.Samp))
        self.wf_list_plot = list(range(1, self.Samp + 1))  # used for plotting later on

        return None

    def ProcessEvent(self, evt, noped):
        if evt.trigger.event_type.value == 32:  # count peds
            self.counter_ped += 1
        else:
            self.counter_evt += 1

        for ipix in range(self.Pix):
            # loop over pixels, 1855 should be redefined as a variable
            if evt.trigger.event_type.value == 32:  # only peds now
                self.Mwf_ped[ipix, :] += evt.r0.tel[0].waveform[self.k][
                    ipix
                ]  # fill pixels one by one and sum them for peds only
            else:
                self.Mwf[ipix, :] += evt.r0.tel[0].waveform[self.k][
                    ipix
                ]  # fill pixels one by one and sum them
        return None

    def FinishRun(self):
        # if (self.k == 0):
        #   gain_c = 'High'
        # if (self.k == 1):
        #    gain_c = 'Low'

        self.Mwf_average = self.Mwf / self.counter_evt  # get average
        # get average over pixels
        self.Mwf_Mean_overPix = np.mean(self.Mwf_average, axis=0)

        if self.counter_ped > 0:
            # get average pedestals
            self.Mwf_ped_average = self.Mwf_ped / self.counter_ped
            self.Mwf_ped_Mean_overPix = np.mean(self.Mwf_ped_average, axis=0)

        return None

    def GetResults(self):

        # ASSIGN RESUTLS TO DICT
        if self.k == 0:
            self.MeanWaveForms_Results_Dict[
                "WF-PHY-AVERAGE-PIX-HIGH-GAIN"
            ] = self.Mwf_Mean_overPix
            if self.counter_ped > 0:
                self.MeanWaveForms_Results_Dict[
                    "WF-AVERAGE-PED-PIX-HIGH-GAIN"
                ] = self.Mwf_ped_Mean_overPix

        if self.k == 1:
            self.MeanWaveForms_Results_Dict[
                "WF-AVERAGE-PIX-LOW-GAIN"
            ] = self.Mwf_Mean_overPix
            if self.counter_ped > 0:
                self.MeanWaveForms_Results_Dict[
                    "WF-PHY-AVERAGE-PED-PIX-LOW-GAIN"
                ] = self.Mwf_ped_Mean_overPix

        return self.MeanWaveForms_Results_Dict

    def PlotResults(self, name, FigPath):

        wf_list = np.array(self.wf_list_plot)

        counter_fig = 0
        colors = ["blue", "red"]
        # colors2 = ['cyan', 'orange']
        titles = ["Physical", "Pedestals"]

        # Set characters of gain: high or lo
        if self.k == 0:
            gain_c = "High"
        if self.k == 1:
            gain_c = "Low"

        full_fig, full_ax = plt.subplots()
        if self.counter_ped > 0:
            array_plot = [self.Mwf_average, self.Mwf_ped_average]
        else:
            array_plot = [self.Mwf_average]

        for x in array_plot:
            part_fig, part_ax = plt.subplots()

            for ipix in range(self.Pix):
                full_ax.plot(
                    wf_list,
                    x[ipix, :],
                    color=colors[counter_fig],
                    alpha=0.005,
                    linewidth=1,
                )
                part_ax.plot(
                    wf_list,
                    x[ipix, :],
                    color=colors[counter_fig],
                    alpha=0.005,
                    linewidth=1,
                )

            part_ax.set_title(
                "Mean Waveforms %s (%s Gain)" % (titles[counter_fig], gain_c)
            )
            part_ax.set_xlabel("Samples")
            part_ax.set_ylabel("Amplitude (DC)")
            # part_ax.legend()
            part_ax.grid()

            part_name = name + "_MeanWaveforms_%s_%sGain.png" % (
                titles[counter_fig],
                gain_c,
            )
            PartPath = FigPath + part_name

            self.MeanWaveForms_Figures_Dict[
                "FIGURE-WF-%s-%s-GAIN" % (titles[counter_fig], gain_c)
            ] = part_fig
            self.MeanWaveForms_Figures_Names_Dict[
                "FIGURE-WF-%s-%s-GAIN" % (titles[counter_fig], gain_c)
            ] = PartPath

            plt.close()

            counter_fig += 1

        full_ax.set_title("Mean Waveforms Combined Plot (%s Gain)" % gain_c)
        full_ax.set_xlabel("Samples")
        full_ax.set_ylabel("Amplitude (DC)")
        # full_ax.legend()
        full_ax.grid()

        full_name = name + "_MeanWaveforms_CombinedPlot_%sGain.png" % gain_c
        FullPath = FigPath + full_name
        self.MeanWaveForms_Figures_Dict[
            "FIGURE-WF-COMBINED-%s-GAIN" % gain_c
        ] = full_fig
        self.MeanWaveForms_Figures_Names_Dict[
            "FIGURE-WF-COMBINED-%s-GAIN" % gain_c
        ] = FullPath

        plt.close()

        return self.MeanWaveForms_Figures_Dict, self.MeanWaveForms_Figures_Names_Dict
