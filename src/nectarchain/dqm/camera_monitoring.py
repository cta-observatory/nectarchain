from dqm_summary_processor import dqm_summary
from matplotlib import pyplot as plt
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from astropy import time as astropytime
import numpy as np
import sqlite3


class CameraMonitoring(dqm_summary):
    def __init__(self, gaink):
        self.k = gaink
        return None

    def ConfigureForRun(self, path, Chan, Samp, Reader1):
        # define number of channels and samples
        self.Chan = Chan
        self.Samp = Samp

        self.camera = CameraGeometry.from_name("NectarCam", 3)
        self.cmap = "gnuplot2"

        self.subarray = Reader1.subarray

        self.event_id = []
        self.event_times = []

        for i, evt1 in enumerate(Reader1):
            self.run_start1 = evt1.nectarcam.tel[0].svc.date

        SqlFileDate = (astropytime.Time(self.run_start1, format="unix").iso).split(" ")[
            0
        ]

        SqlFilePath = ""
        for i in range(len(path.split("/")) - 1):
            SqlFilePath = SqlFilePath + path.split("/")[i] + "/"

        SqlFileName = SqlFilePath + "nectarcam_monitoring_db_" + SqlFileDate + ".sqlite"
        print("SqlFileName", SqlFileName)

        con = sqlite3.connect(SqlFileName)
        cursor = con.cursor()
        # print(cursor.fetchall())
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        # TempData = cursor.execute('''SELECT * FROM monitoring_drawer_temperatures''')
        # print(TempData.description)

        self.DrawerTemp = cursor.fetchall()
        cursor.close()

    def ProcessEvent(self, evt):
        trigger_time = evt.trigger.time.value
        trigger_id = evt.index.event_id

        self.event_times.append(trigger_time)
        self.event_id.append(trigger_id)

    def FinishRun(self):
        self.event_id = np.array(self.event_id)
        self.event_times = np.array(self.event_times)

        self.run_start = self.event_times[self.event_id == np.min(self.event_id)] - 100
        self.run_end = np.max(self.event_times) + 100

        self.DrawerTemp = np.array(self.DrawerTemp)
        self.DrawerTimes = np.array(self.DrawerTemp[:, 3])

        for i in range(len(self.DrawerTimes)):
            self.DrawerTimes[i] = astropytime.Time(
                self.DrawerTimes[i], format="iso"
            ).unix

        self.DrawerTemp11 = self.DrawerTemp[:, 4][self.DrawerTimes > self.run_start]
        self.DrawerTemp21 = self.DrawerTemp[:, 5][self.DrawerTimes > self.run_start]
        self.DrawerNum1 = self.DrawerTemp[:, 2][self.DrawerTimes > self.run_start]

        self.DrawerTimes_new = self.DrawerTimes[self.DrawerTimes > self.run_start]

        self.DrawerTemp12 = self.DrawerTemp11[self.DrawerTimes_new < self.run_end]
        self.DrawerTemp22 = self.DrawerTemp21[self.DrawerTimes_new < self.run_end]
        self.DrawerNum2 = self.DrawerNum1[self.DrawerTimes_new < self.run_end]

        self.DrawerTemp1_mean = []
        self.DrawerTemp2_mean = []
        TotalDrawers = np.max(self.DrawerNum2)

        for i in range(TotalDrawers + 1):
            for j in range(7):
                self.DrawerTemp1_mean.append(
                    np.mean(self.DrawerTemp12[self.DrawerNum2 == i])
                )
                self.DrawerTemp2_mean.append(
                    np.mean(self.DrawerTemp22[self.DrawerNum2 == i])
                )
        self.DrawerTemp1_mean = np.array(self.DrawerTemp1_mean)
        self.DrawerTemp2_mean = np.array(self.DrawerTemp2_mean)

        self.DrawerTemp_mean = (self.DrawerTemp1_mean + self.DrawerTemp2_mean) / 2

    def GetResults(self):
        self.CameraMonitoring_Results_Dict = {}
        self.CameraMonitoring_Results_Dict[
            "CAMERA-TEMPERATURE-AVERAGE"
        ] = self.DrawerTemp_mean

        return self.CameraMonitoring_Results_Dict

    def PlotResults(self, name, FigPath):
        self.ChargeInt_Figures_Dict = {}
        self.ChargeInt_Figures_Names_Dict = {}

        fig, disp = plt.subplots()
        disp = CameraDisplay(self.camera)
        disp.image = self.DrawerTemp_mean
        disp.cmap = plt.cm.coolwarm
        disp.axes.text(1.8, -0.3, "Temperature", fontsize=12, rotation=90)
        disp.add_colorbar()
        plt.title("Camera temperature average")
        full_name = name + "_CameraTemperature_Mean.png"
        FullPath = FigPath + full_name
        self.ChargeInt_Figures_Dict["CAMERA-TEMPERATURE-IMAGE-AVERAGE"] = fig
        self.ChargeInt_Figures_Names_Dict["CAMERA-TEMPERATURE-IMAGE-AVERAGE"] = FullPath

        plt.close()

        fig1, disp = plt.subplots()
        disp = CameraDisplay(self.camera)
        disp.image = self.DrawerTemp1_mean
        disp.cmap = plt.cm.coolwarm
        disp.axes.text(1.8, -0.3, "Temperature 1", fontsize=12, rotation=90)
        disp.add_colorbar()
        plt.title("Camera temperature average 1")
        full_name = name + "_CameraTemperature_average1.png"
        FullPath = FigPath + full_name
        self.ChargeInt_Figures_Dict["CAMERA-TEMPERATURE-IMAGE-AVERAGE-1"] = fig1
        self.ChargeInt_Figures_Names_Dict[
            "CAMERA-TEMPERATURE-IMAGE-AVERAGE-1"
        ] = FullPath

        plt.close()

        fig2, disp = plt.subplots()
        disp = CameraDisplay(self.camera)
        disp.image = self.DrawerTemp2_mean
        disp.cmap = plt.cm.coolwarm
        disp.axes.text(1.8, -0.3, "Temperature 2", fontsize=12, rotation=90)
        disp.add_colorbar()
        plt.title("Camera temperature average 2")
        full_name = name + "_CameraTemperature_average2.png"
        FullPath = FigPath + full_name
        self.ChargeInt_Figures_Dict["CAMERA-TEMPERATURE-IMAGE-AVERAGE-2"] = fig2
        self.ChargeInt_Figures_Names_Dict[
            "CAMERA-TEMPERATURE-IMAGE-AVERAGE-2"
        ] = FullPath

        plt.close()

        return self.ChargeInt_Figures_Dict, self.ChargeInt_Figures_Names_Dict
