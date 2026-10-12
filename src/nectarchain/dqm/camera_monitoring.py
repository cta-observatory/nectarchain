import logging
import os
import sqlite3

import numpy as np
from astropy import time as astropytime
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.visualization import CameraDisplay
from ctapipe_io_nectarcam import constants
from matplotlib import pyplot as plt

from .dqm_summary_processor import DQMSummary

__all__ = ["CameraMonitoring"]

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class CameraMonitoring(DQMSummary):
    """Monitor camera drawer temperatures from a SQLite monitoring database.

    Retrieves drawer temperature data, computes per-pixel statistics
    (mean, std, trend), and produces camera display and trend figures.
    """

    def __init__(self, gaink, r0=False):
        """Initialize camera monitoring attributes.

        Parameters
        ----------
        gaink : int
            Gain index (0 for high gain, 1 for low gain).
        r0 : bool, optional
            Whether to use r0 waveforms (default False).
        """
        self.k = gaink
        self.Pix = None
        self.Samp = None
        self.camera = None
        self.cmap = None
        self.subarray = None
        self.tel_id = None
        self.event_id = []
        self.event_times = []
        self.DrawerTemp = None
        self.run_start = None
        self.run_end = None
        self.DrawerTimes = None
        self.DrawerTimes_new = None
        self.DrawerTemp12 = None
        self.DrawerTemp22 = None
        self.DrawerNum2 = None
        self.DrawerTemp1_mean = []
        self.DrawerTemp2_mean = []
        self.DrawerTemp1_std = []
        self.DrawerTemp2_std = []
        self.DrawerTemp_mean = []
        self.DrawerTemp_std = []
        self.DrawerTemp_trend = []
        self.CameraMonitoring_Results_Dict = {}
        self.ChargeInt_Figures_Dict = {}
        self.ChargeInt_Figures_Names_Dict = {}
        super().__init__(r0)

    def configure_for_run(self, path, Pix, Samp, Reader1, **kwargs):
        """Configure the monitoring processor for a new run.

        Parameters
        ----------
        path : str
            Path to the input data file.
        Pix : int
            Number of pixels.
        Samp : int
            Number of waveform samples.
        Reader1 : ctapipe_io_nectarcam.NectarCAMEventSource
            Event reader providing subarray and camera geometry.
        **kwargs
            Additional keyword arguments (unused).
        """
        # define number of pixels and samples
        self.Pix = Pix
        self.Samp = Samp
        self.tel_id = Reader1.subarray.tel_ids[0]

        self.camera = Reader1.subarray.tel[self.tel_id].camera.geometry.transform_to(
            EngineeringCameraFrame()
        )
        self.cmap = "gnuplot2"

        self.subarray = Reader1.subarray

        self.run_start1 = next(iter(Reader1)).nectarcam.tel[self.tel_id].svc.date

        SqlFileDate = astropytime.Time(self.run_start1, format="unix").iso.split(" ")[0]
        log.debug(f"SqlFileDate is {SqlFileDate}")

        SqlFilePath = os.path.split(path)[0]
        SqlFileName = (
            SqlFilePath + "/nectarcam_monitoring_db_" + SqlFileDate + ".sqlite"
        )
        log.info(f"SqlFileName: {SqlFileName}")
        try:
            with sqlite3.connect(SqlFileName) as con:
                cursor = con.cursor()
                cursor.execute("SELECT * FROM monitoring_drawer_temperatures;")
                self.DrawerTemp = cursor.fetchall()
        except sqlite3.Error as err:
            log.error(
                f"Drawer temperature could not be retrieved. Received error "
                f"code: {err}"
            )

    def process_event(self, evt, noped):
        """Record trigger time and event ID for the current event.

        Parameters
        ----------
        evt : ctapipe.io.DataEventContainer
            The event container.
        noped : bool
            Whether to subtract pedestal (unused here).
        """
        trigger_time = evt.trigger.time.value
        trigger_id = evt.index.event_id

        self.event_times.append(trigger_time)
        self.event_id.append(trigger_id)

    def finish_run(self):
        """Compute per-pixel drawer temperature statistics.

        Calculates the mean, standard deviation, and time trend of
        drawer temperatures, expanding drawer-level values to all
        pixels in each module.
        """
        try:
            self.event_id = np.array(self.event_id)
            self.event_times = np.array(self.event_times)

            min_evt_idx = np.argmin(self.event_id)
            self.run_start = self.event_times[min_evt_idx] - 100
            self.run_end = np.max(self.event_times) + 100

            self.DrawerTemp = np.array(self.DrawerTemp)
            self.DrawerTimes = astropytime.Time(
                np.array(self.DrawerTemp[:, 3], dtype=str), format="iso"
            ).unix

            run_mask = (self.DrawerTimes > self.run_start) & (
                self.DrawerTimes < self.run_end
            )
            self.DrawerTimes_run = self.DrawerTimes[run_mask]
            self.DrawerTemp12 = self.DrawerTemp[:, 4][run_mask]
            self.DrawerTemp22 = self.DrawerTemp[:, 5][run_mask]
            self.DrawerNum2 = self.DrawerTemp[:, 2][run_mask]

            TotalDrawers = int(np.max(self.DrawerNum2))
            n_drawers = TotalDrawers + 1

            drawer_temp1_mean = np.zeros(n_drawers)
            drawer_temp2_mean = np.zeros(n_drawers)
            drawer_temp1_std = np.zeros(n_drawers)
            drawer_temp2_std = np.zeros(n_drawers)

            for i in range(n_drawers):
                mask = self.DrawerNum2 == i
                temp1 = self.DrawerTemp12[mask]
                temp2 = self.DrawerTemp22[mask]

                if len(temp1) > 0:
                    drawer_temp1_mean[i] = np.mean(temp1)
                    drawer_temp1_std[i] = np.std(temp1)
                    drawer_temp2_mean[i] = np.mean(temp2)
                    drawer_temp2_std[i] = np.std(temp2)
                else:
                    drawer_temp1_mean[i] = np.nan
                    drawer_temp1_std[i] = np.nan
                    drawer_temp2_mean[i] = np.nan
                    drawer_temp2_std[i] = np.nan

            # NectarCAM has 7 pixels per drawer
            self.DrawerTemp1_mean = np.repeat(
                drawer_temp1_mean, constants.N_PIXELS_MODULE
            )
            self.DrawerTemp2_mean = np.repeat(
                drawer_temp2_mean, constants.N_PIXELS_MODULE
            )
            self.DrawerTemp1_std = np.repeat(
                drawer_temp1_std, constants.N_PIXELS_MODULE
            )
            self.DrawerTemp2_std = np.repeat(
                drawer_temp2_std, constants.N_PIXELS_MODULE
            )

            self.DrawerTemp1_trend = np.array(
                [self.DrawerTemp12[self.DrawerNum2 == ii] for ii in range(n_drawers)]
            )
            self.DrawerTemp2_trend = np.array(
                [self.DrawerTemp22[self.DrawerNum2 == ii] for ii in range(n_drawers)]
            )

            self.DrawerTemp_trend = (
                self.DrawerTemp1_trend + self.DrawerTemp2_trend
            ) / 2.0
            self.DrawerTemp_mean = (self.DrawerTemp1_mean + self.DrawerTemp2_mean) / 2
            self.DrawerTemp_std = (self.DrawerTemp1_std + self.DrawerTemp2_std) / 2

        except Exception as err:
            log.error(
                f"Drawer temperature could not be retrieved. Received error "
                f"code: {err}"
            )

    def get_results(self):
        """Return the camera temperature results dictionary.

        Returns
        -------
        dict
            Dictionary with keys CAMERA-TEMPERATURE-AVERAGE,
            CAMERA-TEMPERATURE-STD, and CAMERA-TEMPERATURE-TREND.
        """
        try:
            self.CameraMonitoring_Results_Dict[
                "CAMERA-TEMPERATURE-AVERAGE"
            ] = self.DrawerTemp_mean
            self.CameraMonitoring_Results_Dict[
                "CAMERA-TEMPERATURE-STD"
            ] = self.DrawerTemp_std
            self.CameraMonitoring_Results_Dict[
                "CAMERA-TEMPERATURE-TREND"
            ] = self.DrawerTemp_trend
        except Exception as err:
            log.error(
                f"Drawer temperature could not be retrieved. Received error "
                f"code: {err}"
            )

        return self.CameraMonitoring_Results_Dict

    def _create_camera_display_figure(
        self, data, title, label, key_prefix, filename_suffix, name, fig_path
    ):
        """Create a camera display figure with consistent styling."""

        fig, _ = plt.subplots()
        disp = CameraDisplay(self.camera)
        disp.image = data
        disp.cmap = self.cmap
        disp.axes.text(1.8, -0.3, label, fontsize=12, rotation=90)
        disp.add_colorbar()
        plt.title(title)

        full_name = f"{name}_CameraTemperature_{filename_suffix}.png"
        full_path = os.path.join(fig_path, full_name)

        self.ChargeInt_Figures_Dict[key_prefix] = fig
        self.ChargeInt_Figures_Names_Dict[key_prefix] = full_path

        plt.close()
        return fig, full_path

    def _create_trend_figure(
        self, data, title, filename_suffix, name, fig_path, drawer_times
    ):
        """Create a trend plot figure."""

        fig, _ = plt.subplots()
        for ii in range(data.shape[0]):
            plt.plot(
                drawer_times[ii],
                data[ii],
                color="blue",
                alpha=0.5,
            )
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        plt.title(title)

        full_name = f"{name}_CameraTemperature_{filename_suffix}.png"
        full_path = os.path.join(fig_path, full_name)

        self.ChargeInt_Figures_Dict[
            f"CAMERA-TEMPERATURE-IMAGE-{filename_suffix.upper()}"
        ] = fig
        self.ChargeInt_Figures_Names_Dict[
            f"CAMERA-TEMPERATURE-IMAGE-{filename_suffix.upper()}"
        ] = full_path

        plt.close()
        return fig, full_path

    def plot_results(self, name, fig_path):
        """Generate camera temperature display and trend figures.

        Parameters
        ----------
        name : str
            Run name prefix for output filenames.
        fig_path : str
            Directory path for saving figure files.

        Returns
        -------
        tuple of dict
            (figures_dict, filenames_dict) mapping plot keys to
            matplotlib figures and their save paths.
        """
        try:
            # Camera display plots (6 figures)
            camera_plots = [
                (
                    self.DrawerTemp_mean,
                    "Camera temperature average",
                    "Temperature",
                    "CAMERA-TEMPERATURE-IMAGE-AVERAGE",
                    "Mean",
                ),
                (
                    self.DrawerTemp1_mean,
                    "Camera temperature average 1",
                    "Temperature 1",
                    "CAMERA-TEMPERATURE-IMAGE-AVERAGE-1",
                    "average1",
                ),
                (
                    self.DrawerTemp2_mean,
                    "Camera temperature average 2",
                    "Temperature 2",
                    "CAMERA-TEMPERATURE-IMAGE-AVERAGE-2",
                    "average2",
                ),
                (
                    self.DrawerTemp_std,
                    "Camera temperature std",
                    "Temperature",
                    "CAMERA-TEMPERATURE-IMAGE-STD",
                    "Std",
                ),
                (
                    self.DrawerTemp1_std,
                    "Camera temperature std 1",
                    "Temperature 1",
                    "CAMERA-TEMPERATURE-IMAGE-STD-1",
                    "Std1",
                ),
                (
                    self.DrawerTemp2_std,
                    "Camera temperature std 2",
                    "Temperature 2",
                    "CAMERA-TEMPERATURE-IMAGE-STD-2",
                    "Std2",
                ),
            ]

            for data, title, label, key, suffix in camera_plots:
                self._create_camera_display_figure(
                    data, title, label, key, suffix, name, fig_path
                )

            drawer_times = self.DrawerTimes_run.reshape(self.DrawerTemp1_trend.shape)
            drawer_times = np.tile(
                np.unique(drawer_times), (self.DrawerTemp_trend.shape[0], 1)
            )

            # Trend plots (3 figures)
            trend_plots = [
                (self.DrawerTemp_trend, "Camera temperature trend", "Trend"),
                (self.DrawerTemp1_trend, "Camera temperature trend 1", "Trend1"),
                (self.DrawerTemp2_trend, "Camera temperature trend 2", "Trend2"),
            ]

            for data, title, suffix in trend_plots:
                self._create_trend_figure(
                    data, title, suffix, name, fig_path, drawer_times
                )

        except Exception as err:
            log.error(f"Received error code: {err}")

        return self.ChargeInt_Figures_Dict, self.ChargeInt_Figures_Names_Dict
