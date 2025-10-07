try:
    from dataclasses import dataclass, field

    import matplotlib
    import numpy as np
    import numpy.ma as ma
    from ctapipe.visualization import CameraDisplay
    from matplotlib import pyplot as plt

    matplotlib.use("TkAgg")
    from CalibrationData import CalibrationCameraDisplay
    from IPython import embed
    from Utils import GetCamera

except ImportError as e:
    print(e)
    raise SystemExit


class PedestalEvoInfo:
    def __init__(self):
        self.run = None
        # 1D : Entries
        self.times = None  # entriesx

        # 4D Entries, channel, pixel, samples :
        self.positions = None
        self.widths = None

        # 3D Entries, channel, pixel
        self.meanpositions = None
        self.meanwidths = None
        self.temperatures1 = None
        self.temperatures2 = None

        # 2D : channel, pixel
        self.slopes_t1_pedmean = None
        self.intercepts_t1_pedmean = None
        self.corr_t1_pedmean = None

        self.slopes_t2_pedmean = None
        self.intercepts_t2_pedmean = None
        self.corr_t2_pedmean = None

        self.camdisplays = dict()
        self.pixfigures = dict()

    def StoreFigure(self, name, fig):
        fig.canvas.manager.set_window_title(name)

        if name in self.pixfigures:
            plt.close(self.pixfigures[name])

        self.pixfigures[name] = fig

        ## Store also the slope vs slice ?

    def ShowPixelPedestalTemperature1(self, pixid):
        if self.slopes_t1_pedmean is None:
            return

        run_info = f"\nrun {self.run}" if self.run is not None else ""

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # print(f'{self.temperatures1.shape = }')
        for chan in range(2):
            chan_info = "High-Gain" if chan == 0 else "Low-Gain"
            chan_info += f" Pix: {pixid}"

            temps1 = self.temperatures1[:, chan, pixid]
            peds = self.meanpositions[:, chan, pixid]
            slope = self.slopes_t1_pedmean[chan, pixid]
            intercept = self.intercepts_t1_pedmean[chan, pixid]

            min_temps1 = np.min(temps1)
            max_temps1 = np.max(temps1)
            min_peds_fit = slope * min_temps1 + intercept
            max_peds_fit = slope * max_temps1 + intercept

            axs[chan].plot(temps1, peds, "o", label="Measure")
            axs[chan].plot(
                [min_temps1, max_temps1],
                [min_peds_fit, max_peds_fit],
                "-",
                color="red",
                label="Fit",
            )
            axs[chan].set_title(f"Mean Pedestal Vs Temperature\n{chan_info}{run_info}")
            axs[chan].set_xlabel("Temperature 1 (C)")
            axs[chan].set_ylabel("Average Pedestal Position")
            axs[chan].grid()
            axs[chan].legend()

        run_title = "run_" + str(self.run) + "_" if self.run is not None else ""
        canname = f"{run_title}PedestalVsTemperatureEvolution_Pixel_{pixid}"

        self.StoreFigure(canname, fig)

        plt.show(block=False)  ## block = False ?

    def ShowPixelTemperaturesEvolution(self, pixid):
        run_info = f"\nrun {self.run}" if self.run is not None else ""

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

        temps1 = self.temperatures1[:, 0, pixid]
        temps2 = self.temperatures2[:, 0, pixid]
        times = self.times

        axs.plot(times, temps1, label="Temp. 1")
        axs.plot(times, temps2, label="Temp. 2")
        axs.grid()
        axs.legend()
        axs.set_xlabel("Time")
        axs.set_ylabel("Temperature (C)")
        axs.set_title(f"Temperature Vs Time Pixel {pixid}{run_info}")

        run_title = "run_" + str(self.run) + "_" if self.run is not None else ""
        canname = f"{run_title}TemperatureEvolution_Pixel_{pixid}"

        # fig.canvas.manager.set_window_title(canname)
        self.StoreFigure(canname, fig)

        # plt.show(block=True)
        plt.show(block=False)

    def ShowTemperatureEvolution(self):
        run_info = f"\nrun {self.run}" if self.run is not None else ""

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        fig.subplots_adjust(right=0.99, left=0.05, bottom=0.06, top=0.92)

        run_title = "run_" + str(self.run) + "_" if self.run is not None else ""
        # fig.canvas.manager.set_window_title(f'{run_title}TemperatureEvolution')
        self.StoreFigure(f"{run_title}TemperatureEvolution", fig)

        # map of average temperature
        # map of rms temperature
        # map of max-min temperature

        # 3D Entries, channel, pixel

        temps1 = self.temperatures1[:, 0, :]
        mean_temp1 = temps1.mean(axis=0)
        stddev_temp1 = temps1.std(axis=0)
        delta_temp1 = temps1.max(axis=0) - temps1.min(axis=0)

        temps2 = self.temperatures2[:, 0, :]
        mean_temp2 = temps2.mean(axis=0)
        stddev_temp2 = temps2.std(axis=0)
        delta_temp2 = temps2.max(axis=0) - temps2.min(axis=0)

        cam_mean_temp1 = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=mean_temp1,
            ax=axs[0][0],
            allow_pick=True,
            title=f"Mean Temperature 1{run_info}",
            show_frame=False,
        )
        cam_mean_temp1.add_colorbar()
        cam_mean_temp1.set_function(self.ShowPixelTemperaturesEvolution)

        cam_stddev_temp1 = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=stddev_temp1,
            ax=axs[0][1],
            allow_pick=True,
            title=f"Std Dev Temperature 1{run_info}",
            show_frame=False,
        )
        cam_stddev_temp1.add_colorbar()
        cam_stddev_temp1.set_function(self.ShowPixelTemperaturesEvolution)

        cam_delta_temp1 = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=delta_temp1,
            ax=axs[0][2],
            allow_pick=True,
            title=f"Max-Min Temperature 1{run_info}",
            show_frame=False,
        )
        cam_delta_temp1.add_colorbar()
        cam_delta_temp1.set_function(self.ShowPixelTemperaturesEvolution)

        cam_mean_temp2 = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=mean_temp2,
            ax=axs[1][0],
            allow_pick=True,
            title=f"Mean Temperature 2",
            show_frame=False,
        )
        cam_mean_temp2.add_colorbar()
        cam_mean_temp2.set_function(self.ShowPixelTemperaturesEvolution)

        cam_stddev_temp2 = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=stddev_temp2,
            ax=axs[1][1],
            allow_pick=True,
            title=f"Std Dev Temperature 2",
            show_frame=False,
        )
        cam_stddev_temp2.add_colorbar()
        cam_stddev_temp2.set_function(self.ShowPixelTemperaturesEvolution)

        cam_delta_temp2 = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=delta_temp2,
            ax=axs[1][2],
            allow_pick=True,
            title=f"Max-Min Temperature 2",
            show_frame=False,
        )
        cam_delta_temp2.add_colorbar()
        cam_delta_temp2.set_function(self.ShowPixelTemperaturesEvolution)

        self.camdisplays["cam_mean_temp1"] = cam_mean_temp1
        self.camdisplays["cam_stddev_temp1"] = cam_stddev_temp1
        self.camdisplays["cam_delta_temp1"] = cam_delta_temp1

        self.camdisplays["cam_mean_temp2"] = cam_mean_temp2
        self.camdisplays["cam_stddev_temp2"] = cam_stddev_temp2
        self.camdisplays["cam_delta_temp2"] = cam_delta_temp2

    def ShowPedestalTemperature1Correlation(self):
        run_info = f"\nrun {self.run}" if self.run is not None else ""

        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
        fig.subplots_adjust(right=1, left=0.05, bottom=0.06, top=0.92)

        run_title = "run_" + str(self.run) + "_" if self.run is not None else ""
        self.StoreFigure(f"{run_title}PedestalVsTemperature1Evolution", fig)

        for chan in range(2):
            chan_info = "High-Gain" if chan == 0 else "Low-Gain"

            if chan == 0:
                corr_title = f"Correlation Mean Ped Vs Temp 1\n{chan_info}{run_info}"
                corr2_title = f"R^2 Mean Ped Vs Temp 1\n{chan_info}{run_info}"
                slope_title = f"Slope of Mean Ped Vs Temp 1\n{chan_info}{run_info}"
                intercept_title = (
                    f"Intercept of Mean Ped Vs Temp 1\n{chan_info}{run_info}"
                )
            else:
                corr_title = f"{chan_info}"
                corr2_title = f"{chan_info}"
                slope_title = f"{chan_info}"
                intercept_title = f"{chan_info}"

            cam_disp_corr = CalibrationCameraDisplay(
                geometry=GetCamera(),
                cmap="seismic",
                image=self.corr_t1_pedmean[chan],
                ax=axs[chan][0],
                title=corr_title,
                show_frame=False,
                allow_pick=True,
            )
            cam_disp_corr.add_colorbar()
            cam_disp_corr.set_limits_minmax(-1.0, 1.0)
            cam_disp_corr.set_function(self.ShowPixelPedestalTemperature1)

            cam_disp_corr2 = CalibrationCameraDisplay(
                geometry=GetCamera(),
                cmap="turbo",
                image=np.power(self.corr_t1_pedmean[chan], 2.0),
                ax=axs[chan][1],
                title=corr2_title,
                show_frame=False,
                allow_pick=True,
            )
            cam_disp_corr2.add_colorbar()
            cam_disp_corr2.set_limits_minmax(-1.0, 1.0)
            cam_disp_corr2.set_function(self.ShowPixelPedestalTemperature1)

            cam_disp_slope = CalibrationCameraDisplay(
                geometry=GetCamera(),
                cmap="seismic",
                image=self.slopes_t1_pedmean[chan],
                ax=axs[chan][2],
                title=slope_title,
                show_frame=False,
                allow_pick=True,
            )
            cam_disp_slope.add_colorbar()
            cam_disp_slope.set_limits_minmax(-1.5, 1.5)
            cam_disp_slope.set_function(self.ShowPixelPedestalTemperature1)

            cam_disp_intercept = CalibrationCameraDisplay(
                geometry=GetCamera(),
                cmap="turbo",
                image=self.intercepts_t1_pedmean[chan],
                ax=axs[chan][3],
                title=intercept_title,
                show_frame=False,
                allow_pick=True,
            )
            cam_disp_intercept.add_colorbar()
            cam_disp_intercept.set_function(self.ShowPixelPedestalTemperature1)

            self.camdisplays[f"cam_disp_corr_chan{chan}"] = cam_disp_corr
            self.camdisplays[f"cam_disp_corr2_chan{chan}"] = cam_disp_corr2
            self.camdisplays[f"cam_disp_slope_chan{chan}"] = cam_disp_slope
            self.camdisplays[f"cam_disp_intercept_chan{chan}"] = cam_disp_intercept

    def ShowPixelPedestals(self, pixid):
        run_info = f"\nrun {self.run}" if self.run is not None else ""
        run_title = "run_" + str(self.run) + "_" if self.run is not None else ""

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        fig.subplots_adjust(right=0.99, left=0.05, bottom=0.06, top=0.92)
        self.StoreFigure(f"{run_title}Pedestals_Pixel{pixid}", fig)

        colormap = matplotlib.cm.get_cmap("inferno")

        x = np.arange(self.positions.shape[3])
        # print(x)
        for i in range(1, self.positions.shape[0]):
            frac = i / self.positions.shape[0]
            axs[0][0].plot(x, self.positions[i, 0, pixid, :], color=colormap(frac))
            axs[0][1].plot(x, self.positions[i, 1, pixid, :], color=colormap(frac))
            axs[1][0].plot(x, self.widths[i, 0, pixid, :], color=colormap(frac))
            axs[1][1].plot(x, self.widths[i, 1, pixid, :], color=colormap(frac))

        axs[0][0].set_title(f"Pixel {pixid} HG Pedestal Positions{run_info}")
        axs[0][0].set_xlim(np.min(x), np.max(x))
        axs[0][0].set_xlabel("slice")
        axs[0][0].set_ylabel("ped pos")

        axs[0][1].set_title(f"Pixel {pixid} LG Pedestal Positions{run_info}")
        axs[0][1].set_xlim(np.min(x), np.max(x))
        axs[0][1].set_xlabel("slice")
        axs[0][1].set_ylabel("ped pos")

        axs[1][0].set_title(f"Pixel {pixid} HG Pedestal Width")
        axs[1][0].set_xlim(np.min(x), np.max(x))
        axs[1][0].set_xlabel("slice")
        axs[1][0].set_ylabel("ped width")

        axs[1][1].set_title(f"Pixel {pixid} LG Pedestal Width")
        axs[1][1].set_xlim(np.min(x), np.max(x))
        axs[1][1].set_xlabel("slice")
        axs[1][1].set_ylabel("ped width")

        axs[0][0].grid()
        axs[0][1].grid()
        axs[1][0].grid()
        axs[1][1].grid()

        plt.show(block=False)

    def ShowPixelPedestalEvolution(self, pixid):
        run_info = f"\nrun {self.run}" if self.run is not None else ""
        run_title = "run_" + str(self.run) + "_" if self.run is not None else ""

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        fig.subplots_adjust(right=0.99, left=0.1, bottom=0.06, top=0.92)
        self.StoreFigure(f"{run_title}PedestalEvolution_Pixel{pixid}", fig)

        times = self.times
        meanpos_hi = self.meanpositions[:, 0, pixid]
        meanpos_lo = self.meanpositions[:, 1, pixid]
        stdpos_hi = self.meanwidths[:, 0, pixid]
        stdpos_lo = self.meanwidths[:, 1, pixid]

        # embed()
        axs[0][0].plot(self.times, meanpos_hi, "-")
        axs[0][0].set_title(f"Pixel {pixid} HG Pedestal Position{run_info}")
        axs[0][0].set_xlim(min(times), max(times))
        axs[0][0].set_xlabel("time")
        axs[0][0].set_ylabel("ped pos")

        axs[0][1].plot(self.times, meanpos_lo, "-")
        axs[0][1].set_title(f"Pixel {pixid} LG Pedestal Position{run_info}")
        axs[0][1].set_xlim(min(times), max(times))
        axs[0][1].set_xlabel("time")
        axs[0][1].set_ylabel("ped pos")

        axs[1][0].plot(self.times, stdpos_hi, "-")
        axs[1][0].set_title(f"Pixel {pixid} HG Pedestal Width")
        axs[1][0].set_xlim(min(times), max(times))
        axs[1][0].set_xlabel("time")
        axs[1][0].set_ylabel("ped width")

        axs[1][1].plot(self.times, stdpos_lo, "-")
        axs[1][1].set_title(f"Pixel {pixid} LG Pedestal Width")
        axs[1][1].set_xlim(min(times), max(times))
        axs[1][1].set_xlabel("time")
        axs[1][1].set_ylabel("ped width")

        plt.show(block=False)

    def ShowPedestalEvolution(self):
        run_info = f"\nrun {self.run}" if self.run is not None else ""
        run_title = "run_" + str(self.run) + "_" if self.run is not None else ""

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        fig.subplots_adjust(right=0.99, left=0.05, bottom=0.06, top=0.92)
        self.StoreFigure(f"{run_title}MaxMinPedestal", fig)

        min_ped = self.positions.min(axis=0)
        max_ped = self.positions.max(axis=0)
        delta_ped = max_ped - min_ped
        delta_max_ped = delta_ped.max(axis=2)

        cam_disp_maxminhg = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=delta_max_ped[0],
            ax=axs[0],
            title=f"High-Gain Maximum Pedestal Position Change in Slice{run_info}",
            show_frame=False,
            allow_pick=True,
        )
        cam_disp_maxminhg.add_colorbar()
        cam_disp_maxminhg.set_function(self.ShowPixelPedestals)

        cam_disp_maxminlg = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=delta_max_ped[1],
            ax=axs[1],
            title=f"Low-Gain Maximum Pedestal Position Change in Slice{run_info}",
            show_frame=False,
            allow_pick=True,
        )
        cam_disp_maxminlg.add_colorbar()
        cam_disp_maxminlg.set_function(self.ShowPixelPedestals)

        self.camdisplays[f"cam_disp_maxminhg"] = cam_disp_maxminhg
        self.camdisplays[f"cam_disp_maxminlg"] = cam_disp_maxminlg

        ## Pedesal : Mean of mean, rms of mean, mean of rms, rms of rms
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(5 * 4, 5 * 2))
        fig.subplots_adjust(right=0.99, left=0.05, bottom=0.06, top=0.92)
        self.StoreFigure(f"{run_title}PedestalEvolution", fig)
        # Quand on clic, on a l'évolution de la moyenne et de la rms pendant tout le run

        meanmeanpos = self.meanpositions.mean(axis=0)
        stdmeanpos = self.meanpositions.std(axis=0)
        meanmeanwidths = self.meanwidths.mean(axis=0)
        stdmeanwidths = self.meanwidths.std(axis=0)

        cam_disp_meanmeanhg = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=meanmeanpos[0],
            ax=axs[0][0],
            title=f"Mean of Mean HG Pedestal{run_info}",
            show_frame=False,
            allow_pick=True,
        )
        cam_disp_meanmeanhg.add_colorbar()
        cam_disp_meanmeanhg.set_function(self.ShowPixelPedestalEvolution)
        self.camdisplays[f"cam_disp_meanmeanhg"] = cam_disp_meanmeanhg

        cam_disp_stdmeanhg = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=stdmeanpos[0],
            ax=axs[0][1],
            title=f"StdDev of Mean HG Pedestal{run_info}",
            show_frame=False,
            allow_pick=True,
        )
        cam_disp_stdmeanhg.add_colorbar()
        cam_disp_stdmeanhg.set_function(self.ShowPixelPedestalEvolution)
        self.camdisplays[f"cam_disp_stdmeanhg"] = cam_disp_stdmeanhg

        cam_disp_meanstdhg = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=meanmeanwidths[0],
            ax=axs[0][2],
            title=f"Mean HG Pedestal Width{run_info}",
            show_frame=False,
            allow_pick=True,
        )
        cam_disp_meanstdhg.add_colorbar()
        cam_disp_meanstdhg.set_function(self.ShowPixelPedestalEvolution)
        self.camdisplays[f"cam_disp_meanstdhg"] = cam_disp_meanstdhg

        cam_disp_stdstdhg = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=stdmeanwidths[0],
            ax=axs[0][3],
            title=f"StdDev HG Pedestal Width{run_info}",
            show_frame=False,
            allow_pick=True,
        )
        cam_disp_stdstdhg.add_colorbar()
        cam_disp_stdstdhg.set_function(self.ShowPixelPedestalEvolution)
        self.camdisplays[f"cam_disp_stdstdhg"] = cam_disp_stdstdhg

        #############
        cam_disp_meanmeanlg = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=meanmeanpos[1],
            ax=axs[1][0],
            title=f"Mean of Mean LG Pedestal",
            show_frame=False,
            allow_pick=True,
        )
        cam_disp_meanmeanlg.add_colorbar()
        cam_disp_meanmeanlg.set_function(self.ShowPixelPedestalEvolution)
        self.camdisplays[f"cam_disp_meanmeanlg"] = cam_disp_meanmeanlg

        cam_disp_stdmeanlg = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=stdmeanpos[1],
            ax=axs[1][1],
            title=f"StdDev of Mean LG Pedestal",
            show_frame=False,
            allow_pick=True,
        )
        cam_disp_stdmeanlg.add_colorbar()
        cam_disp_stdmeanlg.set_function(self.ShowPixelPedestalEvolution)
        self.camdisplays[f"cam_disp_stdmeanlg"] = cam_disp_stdmeanlg

        cam_disp_meanstdlg = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=meanmeanwidths[1],
            ax=axs[1][2],
            title=f"Mean LG Pedestal Width",
            show_frame=False,
            allow_pick=True,
        )
        cam_disp_meanstdlg.add_colorbar()
        cam_disp_meanstdlg.set_function(self.ShowPixelPedestalEvolution)
        self.camdisplays[f"cam_disp_meanstdlg"] = cam_disp_meanstdlg

        cam_disp_stdstdlg = CalibrationCameraDisplay(
            geometry=GetCamera(),
            cmap="turbo",
            image=stdmeanwidths[1],
            ax=axs[1][3],
            title=f"StdDev LG Pedestal Width",
            show_frame=False,
            allow_pick=True,
        )
        cam_disp_stdstdlg.add_colorbar()
        cam_disp_stdstdlg.set_function(self.ShowPixelPedestalEvolution)
        self.camdisplays[f"cam_disp_stdstdlg"] = cam_disp_stdstdlg
