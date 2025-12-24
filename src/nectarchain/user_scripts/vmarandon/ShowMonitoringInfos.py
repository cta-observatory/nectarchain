try:
    import argparse
    import os
    import sys
    import time
    from datetime import datetime, timedelta

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.ma as ma
    from CalibrationCameraDisplay import CalibrationCameraDisplay
    from DataUtils import GetFirstLastEventTime
    from DBHandler2 import DBInfos, ModuleArray, PixelArray, to_datetime
    from matplotlib import dates
    from Utils import (
        ConvertTitleToName,
        CustomFormatter,
        GetCamera,
        GetDefaultDataPath,
        GetDefaultDBPath,
        GetMirrorModules,
    )

    matplotlib.use("TkAgg")
    # matplotlib.use("qt5cairo")
except ImportError as err:
    print(err)
    raise SystemExit


class BaseDisplayTimeInfos:
    def __init__(
        self,
        name,
        times,
        datas,
        infos="",
        units="",
        run=None,
        begin_time=None,
        end_time=None,
    ):
        self.name = name
        self.times = times
        self.datas = datas
        self.infos = infos
        self.units = units
        self.run = run
        self.begin_time = begin_time
        self.end_time = end_time
        self.figures = list()
        self.axs = list()

    def run_title(self):
        return "" if self.run is None else f"Run: {self.run}"

    def run_name(self):
        return "" if self.run is None else f"run_{self.run}"

    def has_run(self):
        return self.run is not None

    def title_unit(self, parenthesis=False):
        return f"({self.units})" if parenthesis else f"{self.units}"


class BaseCameraDisplayTimeInfos(BaseDisplayTimeInfos):
    def __init__(self, camfigsize=5, camera=None, cmap="turbo", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camfigsize = camfigsize
        self.camera = GetCamera() if camera is None else camera
        self.cmap = cmap
        self.camobjects = dict()

    def show(self, save_plots=False):
        if self.datas is None:
            print("No data stored --> Don't show anything !")
            return

        mean = None
        std = None
        min = None
        max = None

        if isinstance(self.datas, ModuleArray) or self.datas.shape[0] == 265:
            mean = np.nanmean(self.datas, axis=-1).to_pixel()
            std = np.nanstd(self.datas, axis=-1, ddof=1).to_pixel()
            min = np.nanmin(self.datas, axis=-1).to_pixel()
            max = np.nanmax(self.datas, axis=-1).to_pixel()
            # mean = np.repeat( np.nanmean(self.datas,axis=-1), 7 )
            # std  = np.repeat( np.nanstd(self.datas,axis=-1,ddof=1), 7)
            # min  = np.repeat( np.nanmin(self.datas,axis=-1), 7)
            # max  = np.repeat( np.nanmax(self.datas,axis=-1), 7)
        elif isinstance(self.datas, PixelArray) or self.datas.shape[0] == 1855:
            # mean = self.datas.nanmean(axis=-1)
            # std  = self.datas.nanstd(axis=-1,ddof=1)
            # min  = self.datas.nanmin(axis=-1)
            # max  = self.datas.nanmax(axis=-1)
            mean = np.nanmean(self.datas, axis=-1)
            std = np.nanstd(self.datas, axis=-1, ddof=1)
            min = np.nanmin(self.datas, axis=-1)
            max = np.nanmax(self.datas, axis=-1)
        else:
            print(f"No method for data of type : {type(self.datas)} --> Implement me !")
            return

        mask = (mean == 0.0) | np.isnan(mean) | np.isinf(mean)
        mean = ma.array(mean, mask=mask)
        std = ma.array(std, mask=mask)
        min = ma.array(min, mask=mask)
        max = ma.array(max, mask=mask)
        # print(min)
        # print(max)

        nrows = 2
        ncols = 2
        size = self.camfigsize

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows)
        )

        ax = axs[0][0]
        title = f"Mean {self.name} {self.infos}"
        if self.has_run():
            title += f"\n{self.run_title()}"
        cam_disp_mean = CalibrationCameraDisplay(
            geometry=self.camera,
            cmap=self.cmap,
            image=mean,
            title=title,
            ax=ax,
            show_frame=False,
            allow_pick=True,
            norm="lin",
        )
        cam_disp_mean.highlight_pixels(range(1855), color="grey", linewidth=0.2)
        cam_disp_mean.add_colorbar()
        cam_disp_mean.colorbar.set_label(self.title_unit())
        cam_disp_mean.set_function(self.show_pixel_evolution)
        self.camobjects["mean"] = cam_disp_mean

        ax = axs[0][1]
        title = f"Std {self.name} {self.infos}"
        if self.has_run():
            title += f"\n{self.run_title()}"
        cam_disp_std = CalibrationCameraDisplay(
            geometry=self.camera,
            cmap=self.cmap,
            image=std,
            title=title,
            ax=ax,
            show_frame=False,
            allow_pick=True,
            norm="lin",
        )
        cam_disp_std.highlight_pixels(range(1855), color="grey", linewidth=0.2)
        cam_disp_std.add_colorbar()
        cam_disp_std.colorbar.set_label(self.title_unit())
        cam_disp_std.set_function(self.show_pixel_evolution)
        self.camobjects["std"] = cam_disp_std

        ax = axs[1][0]
        title = f"Min {self.name} {self.infos}"
        if self.has_run():
            title += f"\n{self.run_title()}"
        cam_disp_min = CalibrationCameraDisplay(
            geometry=self.camera,
            cmap=self.cmap,
            image=min,
            title=title,
            ax=ax,
            show_frame=False,
            allow_pick=True,
            norm="lin",
        )
        cam_disp_min.highlight_pixels(range(1855), color="grey", linewidth=0.2)
        cam_disp_min.add_colorbar()
        cam_disp_min.colorbar.set_label(self.title_unit())
        cam_disp_min.set_function(self.show_pixel_evolution)
        self.camobjects["min"] = cam_disp_min

        ax = axs[1][1]
        title = f"Max {self.name} {self.infos}"
        if self.has_run():
            title += f"\n{self.run_title()}"
        cam_disp_max = CalibrationCameraDisplay(
            geometry=self.camera,
            cmap=self.cmap,
            image=max,
            title=title,
            ax=ax,
            show_frame=False,
            allow_pick=True,
            norm="lin",
        )
        cam_disp_max.highlight_pixels(range(1855), color="grey", linewidth=0.2)
        cam_disp_max.add_colorbar()
        cam_disp_max.colorbar.set_label(self.title_unit())
        cam_disp_max.set_function(self.show_pixel_evolution)
        self.camobjects["max"] = cam_disp_max

        fig.tight_layout()

        if save_plots:
            outname = ""
            if self.has_run:
                outname += "run_" + f"{self.run}" + "_"
            outname += self.name + "_" + ConvertTitleToName(self.infos)
            outname = ConvertTitleToName(outname)  ## weird but it also do some cleaning
            outname += ".png"
            fig.savefig(outname)

        self.figures.append(fig)
        self.axs.append(axs)

    def show_pixel_evolution(self, pixid):
        if self.datas is None:
            print("No data stored --> Don't show anything !")
            return
        if self.times is None:
            print(f"No time stored --> Don't show anything for pixel {pixid}")
            return
        print(f"Show Evolution for Pixel : {pixid}")
        nrows = 1
        ncols = 1
        size_y = 4
        size_x = 2 * size_y
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(size_x * ncols, size_y * nrows)
        )
        ax = axs
        xs = self.times
        ys = None
        if isinstance(self.datas, ModuleArray):
            ys = self.datas[pixid // 7]
        elif isinstance(self.datas, PixelArray):
            ys = self.datas[pixid]
        else:
            try:
                ys = self.datas[pixid]
            except Exception as err:
                print(err)
                return

        ax.plot(xs, ys, "o-")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel(f"{self.name} {self.infos} {self.title_unit(parenthesis=True)}")
        ax.set_title(
            f"{self.name} {self.infos} Evolution\nPixel: {pixid} [Module: {pixid//7}, Pos: {pixid%7}]"
        )
        ax.xaxis.set_major_formatter(dates.DateFormatter("%d/%m %H:%M"))
        ax.tick_params("x", rotation=30)
        ax.grid()

        fig.tight_layout()

        # save figures to keep them alive
        self.figures.append(fig)
        self.axs.append(axs)

        plt.show()


class ModuleTemperatureDisplayInfos(BaseCameraDisplayTimeInfos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infos = "Temperature"
        self.units = "$^\circ$C"


class PixelHVDisplayInfos(BaseCameraDisplayTimeInfos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infos = "HV"
        self.units = "V"


class PixelCurrentDisplayInfos(BaseCameraDisplayTimeInfos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infos = "Current"
        self.units = "mA"


class CameraRateDisplayInfos(BaseCameraDisplayTimeInfos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infos = "Rate"
        self.units = "Hz"


class DisplayTimeInfos(BaseDisplayTimeInfos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def show(self, save_plots=False):
        if self.datas is None:
            print("No data stored --> Don't show anything !")
            return

        nrows = 1
        ncols = 1
        size_y = 4
        size_x = 2 * size_y
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(size_x * ncols, size_y * nrows)
        )
        self.figures.append(fig)
        self.axs.append(axs)
        ax = axs

        xs = self.times
        ys = self.datas

        ax.plot(xs, ys, "o-")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel(f"{self.name} {self.infos} {self.title_unit(parenthesis=True)}")

        title = f"{self.name} {self.infos} Evolution"
        if self.has_run():
            title += f"\n{self.run_title()}"
        ax.set_title(title)
        ax.xaxis.set_major_formatter(dates.DateFormatter("%d/%m %H:%M"))
        ax.tick_params("x", rotation=30)
        ax.grid()

        fig.tight_layout()

        if save_plots:
            outname = ""
            if self.has_run:
                outname += "run_" + f"{self.run}" + "_"
            outname += self.name + "_" + ConvertTitleToName(self.infos)
            outname = ConvertTitleToName(outname)  ## weird but it also do some cleaning
            outname += ".png"
            fig.savefig(outname)

        # save figures to keep them alive
        self.figures.append(fig)
        self.axs.append(axs)

        # plt.show()


class TemperatureDisplayInfos(DisplayTimeInfos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infos = "Temperature"
        self.units = "$^\circ$C"


class MultipleTemperatureDisplayInfos(DisplayTimeInfos):
    def __init__(self, temperatures, *args, **kwargs):
        super().__init__(datas=next(iter(temperatures.values()), None), *args, **kwargs)
        self.infos = "Temperature"
        self.units = "$^\circ$C"
        self.temperatures = temperatures

    def show(self, save_plots=False):
        if self.datas is None:
            print("No data stored --> Don't show anything !")
            return
        # print("HERE HERE HERE HERE HERE !!!!!!!!!!!!")
        nrows = 1
        ncols = 1
        size_y = 4
        size_x = 2 * size_y
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(size_x * ncols, size_y * nrows)
        )
        self.figures.append(fig)
        self.axs.append(axs)
        ax = axs

        xs = self.times
        for label, temperature in self.temperatures.items():
            ax.plot(xs, temperature, "o-", label=label)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel(f"{self.name} {self.infos} {self.title_unit(parenthesis=True)}")

        title = f"{self.name} {self.infos} Evolution"
        if self.has_run():
            title += f"\n{self.run_title()}"
        ax.set_title(title)
        ax.xaxis.set_major_formatter(dates.DateFormatter("%d/%m %H:%M"))
        ax.tick_params("x", rotation=30)
        ax.grid()
        ax.legend()

        fig.tight_layout()

        if save_plots:
            outname = ""
            if self.has_run:
                outname += "run_" + f"{self.run}" + "_"
            outname += self.name + "_" + ConvertTitleToName(self.infos)
            outname = ConvertTitleToName(outname)  ## weird but it also do some cleaning
            outname += ".png"
            fig.savefig(outname)

        # save figures to keep them alive
        self.figures.append(fig)
        self.axs.append(axs)


class CurrentDisplayInfos(DisplayTimeInfos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infos = "Current"
        self.units = "A"


class VoltageDisplayInfos(DisplayTimeInfos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infos = "Voltage"
        self.units = "V"


class WaterCoolingTemperatureInfos(DisplayTimeInfos):
    def __init__(self, temp_in, temp_out, *args, **kwargs):
        super().__init__(datas=temp_in, *args, **kwargs)
        self.infos = "Temperature"
        self.units = "$^\circ$C"
        self.temp_in = temp_in
        self.temp_out = temp_out

    def show(self, save_plots=False):
        if self.datas is None:
            print("No data stored --> Don't show anything !")
            return

        nrows = 1
        ncols = 1
        size_y = 4
        size_x = 2 * size_y
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(size_x * ncols, size_y * nrows)
        )
        self.figures.append(fig)
        self.axs.append(axs)
        ax = axs

        xs = self.times
        ys1 = self.temp_in
        ys2 = self.temp_out

        ax.plot(xs, ys1, "o-", label="In")
        ax.plot(xs, ys2, "o-", label="Out")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel(f"{self.name} {self.infos} {self.title_unit(parenthesis=True)}")

        title = f"{self.name} {self.infos} Evolution"
        if self.has_run():
            title += f"\n{self.run_title()}"
        ax.set_title(title)
        ax.xaxis.set_major_formatter(dates.DateFormatter("%d/%m %H:%M"))
        ax.tick_params("x", rotation=30)
        ax.grid()
        ax.legend()

        fig.tight_layout()

        if save_plots:
            outname = ""
            if self.has_run:
                outname += "run_" + f"{self.run}" + "_"
            outname += self.name + "_" + ConvertTitleToName(self.infos)
            outname = ConvertTitleToName(outname)  ## weird but it also do some cleaning
            outname += ".png"
            fig.savefig(outname)

        # save figures to keep them alive
        self.figures.append(fig)
        self.axs.append(axs)


class PhaseDisplayInfos(DisplayTimeInfos):
    def __init__(self, phase1, phase2, phase3, *args, **kwargs):
        super().__init__(datas=phase1, *args, **kwargs)
        # self.infos = "Current"
        # self.units = "A"
        self.phase1 = phase1
        self.phase2 = phase2
        self.phase3 = phase3

    def show(self, save_plots=False):
        if self.datas is None:
            print("No data stored --> Don't show anything !")
            return

        nrows = 1
        ncols = 1
        size_y = 4
        size_x = 2 * size_y
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(size_x * ncols, size_y * nrows)
        )
        self.figures.append(fig)
        self.axs.append(axs)
        ax = axs

        xs = self.times
        ys1 = self.phase1
        ys2 = self.phase2
        ys3 = self.phase3

        ax.plot(xs, ys1, "o-", label="Phase 1")
        ax.plot(xs, ys2, "o-", label="Phase 2")
        ax.plot(xs, ys3, "o-", label="Phase 3")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel(f"{self.name} {self.infos} {self.title_unit(parenthesis=True)}")

        title = f"{self.name} {self.infos} Evolution"
        if self.has_run():
            title += f"\n{self.run_title()}"
        ax.set_title(title)
        ax.xaxis.set_major_formatter(dates.DateFormatter("%d/%m %H:%M"))
        ax.tick_params("x", rotation=30)
        ax.grid()
        ax.legend()

        fig.tight_layout()

        if save_plots:
            outname = ""
            if self.has_run:
                outname += "run_" + f"{self.run}" + "_"
            outname += self.name + "_" + ConvertTitleToName(self.infos)
            outname = ConvertTitleToName(outname)  ## weird but it also do some cleaning
            outname += ".png"
            fig.savefig(outname)

        # save figures to keep them alive
        self.figures.append(fig)
        self.axs.append(axs)


class CurrentPhaseDisplayInfos(PhaseDisplayInfos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infos = "Current"
        self.units = "A"


class VoltagePhaseDisplayInfos(PhaseDisplayInfos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infos = "Voltage"
        self.units = "V"


from dateutil.parser import ParserError, parse

# class DateParser(argparse.Action):
#     def __call__(self, parser, namespace, values, option_strings=None):
#         print(f"values: [{values}]")
#         print(f'tupes: {type(values)}')
#         print(f"values == None ? {values=='None'}")
#         try:
#             setattr(namespace, self.dest, parse(values))
#         except Exception as err:
#             setattr(namespace, self.dest, None)

#         #     if values != 'None':
#         #     #dt = datetime.strptime(values, '%Y-%m-%d %H:%M')
#         #     #setattr(namespace, self.dest, dt)
#         #     setattr(namespace, self.dest, parse(values))
#         # else:
#         #     setattr(namespace, self.dest, None)


def valid_date(s):
    try:
        # return datetime.strptime(s, "%Y-%m-%d %H:%M")
        return parse(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Not a valid date: '{s}'. Expected format: YYYY-MM-DD HH:MM"
        )


# parser = argparse.ArgumentParser(description="Process optional start and end datetime arguments.")


def ShowMonitoringInfos(arglist):
    p = argparse.ArgumentParser(
        description="Show monitoring informations from DB",
        epilog="examples:\n" "\t python %(prog)s --run 123456  \n",
        formatter_class=CustomFormatter,
    )

    p.add_argument("--run", dest="run", type=int, help="Run number to be converted")
    p.add_argument(
        "--data-path",
        dest="dataPath",
        type=str,
        default=None,
        help="Path to the rawdata directory. The program will recursively search in all directory for matching rawdata",
    )
    p.add_argument(
        "--db-path",
        dest="dbPath",
        type=str,
        default=None,
        help="Path to the directory containing db files",
    )
    p.add_argument(
        "--telid",
        dest="telId",
        type=int,
        default=0,
        help="Telescope id for which we show the data distributions",
    )
    p.add_argument("--savefig", dest="savefig", action="store_true", help="Save figure")
    p.add_argument("--batch", dest="batch", action="store_true", help="batch mode")
    p.add_argument(
        "--tstart",
        dest="tstart",
        type=valid_date,
        default=None,
        help="Set a start date (in UTC)",
    )
    p.add_argument(
        "--tend",
        dest="tend",
        type=valid_date,
        default=None,
        help="Set a end date (in UTC)",
    )

    args = p.parse_args(arglist)
    if args.run is None and (args.tstart is None or args.tend is None):
        p.print_help()
        return -1

    path = GetDefaultDataPath() if args.dataPath is None else args.dataPath
    dbpath = GetDefaultDBPath() if args.dbPath is None else args.dbPath

    ## Read event times
    try:
        if args.run is not None:
            begin_time, end_time = GetFirstLastEventTime(args.run, path=path)
        elif args.tstart is not None and args.tend is not None:
            begin_time = to_datetime(args.tstart)
            end_time = to_datetime(args.tend)
            print(f"{begin_time = }")
            print(f"{end_time = }")
        else:
            print("Invalid arguments")
    except Exception as err:
        print(err)
        return -1
    if begin_time is None or end_time is None:
        print(f"Invalid begin or end time [{begin_time},{end_time}]")
        return -1

    begin_time = to_datetime(begin_time)
    end_time = to_datetime(end_time)

    delta_t = (end_time - begin_time).total_seconds()
    if delta_t < 60.0:
        safety_margin_minutes = 2
        print(
            f"Run: {args.run} is {delta_t}s --> Not enough, adding {safety_margin_minutes} minutes "
        )
        end_time = end_time + timedelta(seconds=safety_margin_minutes * 60)

    print(f"{begin_time = }")
    print(f"{end_time = }")

    print(f"dbpath: [{dbpath}]")
    dbinfos = DBInfos.init_from_time(begin_time, end_time, dbpath=dbpath, verbose=True)
    # from IPython import embed
    # embed()
    dbinfos.connect(
        "monitoring_drawer_temperatures",
        "monitoring_ib",
        "monitoring_channel_voltages",
        "monitoring_l0_scalers",
        "monitoring_l1_scalers",
        "monitoring_channel_currents",
        "monitoring_tib_scalers",
        "monitoring_ucts",
        "monitoring_ecc",
        "monitoring_ffcls",
    )

    ########## Temperatures
    # import matplotlib
    # print(matplotlib.get_backend())

    # print(matplotlib.get_backend())
    # #from IPython import embed
    # #embed()

    try:
        tfeb1 = dbinfos.tel[args.telId].monitoring_drawer_temperatures.tfeb1
        mtdi1 = ModuleTemperatureDisplayInfos(
            run=args.run, name="FEB 1", times=tfeb1.times, datas=tfeb1.datas
        )
        mtdi1.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        tfeb2 = dbinfos.tel[args.telId].monitoring_drawer_temperatures.tfeb2
        mtdi2 = ModuleTemperatureDisplayInfos(
            run=args.run, name="FEB 2", times=tfeb2.times, datas=tfeb2.datas
        )
        mtdi2.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    ## Show Mirror
    mirror_modules = GetMirrorModules()
    # from IPython import embed
    # embed()
    try:
        tfeb1m_datas = tfeb1.datas[mirror_modules]
        mtdi1m = ModuleTemperatureDisplayInfos(
            run=args.run,
            name="FEB 1 - Mirror",
            times=tfeb1.times,
            datas=tfeb1m_datas - tfeb1.datas,
            cmap="coolwarm",
        )
        mtdi1m.show(save_plots=args.savefig)
        # pass
    except Exception as err:
        print(err)

    try:
        tfeb2m_datas = tfeb2.datas[mirror_modules]
        mtdi2m = ModuleTemperatureDisplayInfos(
            run=args.run,
            name="FEB 2 - Mirror",
            times=tfeb2.times,
            datas=tfeb2m_datas - tfeb2.datas,
            cmap="coolwarm",
        )
        mtdi2m.show(save_plots=args.savefig)
        # pass
    except Exception as err:
        print(err)

    try:
        try:
            tib = dbinfos.tel[args.telId].monitoring_ib.temperature
        except Exception as err:
            print(f"{err} --> Fallback to camera 0")
            try:
                tib = dbinfos.tel[0].monitoring_ib.temperature
            except Exception as err:
                tib = None
        mtdi3 = ModuleTemperatureDisplayInfos(
            run=args.run, name="IB", times=tib.times, datas=tib.datas
        )
        mtdi3.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        l0s = dbinfos.tel[args.telId].monitoring_l0_scalers.l0_scaler
        l0di = CameraRateDisplayInfos(
            run=args.run, name="L0", times=l0s.times, datas=l0s.datas
        )
        l0di.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        l1s = dbinfos.tel[args.telId].monitoring_l1_scalers.l1_scaler
        l1di = CameraRateDisplayInfos(
            run=args.run, name="L1", times=l1s.times, datas=l1s.datas
        )
        l1di.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        hvs = dbinfos.tel[args.telId].monitoring_channel_voltages.voltage
        hvdi = PixelHVDisplayInfos(
            run=args.run, name="", times=hvs.times, datas=hvs.datas
        )
        hvdi.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        hvcs = dbinfos.tel[args.telId].monitoring_channel_currents.current
        hvcdi = PixelCurrentDisplayInfos(
            run=args.run, name="", times=hvcs.times, datas=hvcs.datas
        )
        hvcdi.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        hvlcs = dbinfos.tel[args.telId].monitoring_channel_currents.load_current
        hvlcdi = PixelCurrentDisplayInfos(
            run=args.run, name="Load", times=hvlcs.times, datas=hvlcs.datas
        )
        hvlcdi.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        tibt = dbinfos.tel[args.telId].monitoring_tib_scalers.temperature
        tibdti = TemperatureDisplayInfos(
            run=args.run, name="TIB", times=tibt.times, datas=tibt.datas
        )
        tibdti.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        uctst = dbinfos.tel[args.telId].monitoring_ucts.temp
        uctsdti = TemperatureDisplayInfos(
            run=args.run, name="UCTS", times=uctst.times, datas=uctst.datas
        )
        uctsdti.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        ecc_temp_infos = dict()
        ecc_temp_infos[1] = "Ambiant Front Right Top"
        ecc_temp_infos[2] = "Ambiant Front Right Bottom"
        ecc_temp_infos[3] = "Ambiant Front Left Bottom"
        ecc_temp_infos[4] = "Ambiant Front Left Top"

        ecc_temp_infos[5] = "MH Front Right Top"
        ecc_temp_infos[6] = "MH Front Middle Top"
        ecc_temp_infos[7] = "MH Front Left Top"
        ecc_temp_infos[8] = "MH Middle Left Top"
        ecc_temp_infos[13] = "MH Front Middle Bottom"

        ecc_temp_infos[9] = "Ambiant Rear Right Top"
        ecc_temp_infos[10] = "Ambiant Rear Left Top"
        ecc_temp_infos[11] = "Ambiant Rear Left Bottom"
        ecc_temp_infos[12] = "Ambiant Rear Right Bottom"

        ecc_temp_infos[14] = "Water Input"
        ecc_temp_infos[15] = "Water Output"

        ecc_temp_infos[16] = "DTC Crate"

        ecc_temp_datas = dict()
        ecc_temperatures = dict()
        ecc_times = dict()
        for i in range(17):
            try:
                if i == 14 or i == 15:
                    ## Water temperature will have a special treatment
                    continue
                ecctemps = getattr(
                    dbinfos.tel[args.telId].monitoring_ecc, f"temp_{i:02}"
                )
                ecc_temperatures[ecc_temp_infos[i]] = ecctemps.datas
                ecc_times[ecc_temp_infos[i]] = ecctemps.times
                varname = ecc_temp_infos.get(i, "Unknown")
                ecc_temp_datas[i] = TemperatureDisplayInfos(
                    run=args.run,
                    name=varname,
                    times=ecctemps.times,
                    datas=ecctemps.datas,
                )
                ecc_temp_datas[i].show(save_plots=args.savefig)
            except Exception as err:
                print(err)

        ecc_avg_temp = TemperatureDisplayInfos(
            run=args.run,
            name="Avg Temp",
            times=dbinfos.tel[args.telId].monitoring_ecc.temp_avg.times,
            datas=dbinfos.tel[args.telId].monitoring_ecc.temp_avg.datas,
        )
        ecc_avg_temp.show(save_plots=args.savefig)

        ecc_all_temp = MultipleTemperatureDisplayInfos(
            run=args.run,
            name="ECC",
            times=next(iter(ecc_times.values())),
            temperatures=ecc_temperatures,
        )
        ecc_all_temp.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        ecc_water_in = dbinfos.tel[args.telId].monitoring_ecc.temp_14
        ecc_water_out = dbinfos.tel[args.telId].monitoring_ecc.temp_15
        ecc_water_delta = ecc_water_out.datas - ecc_water_in.datas

        ecc_water_temp_single = WaterCoolingTemperatureInfos(
            run=args.run,
            name="Water Temperature",
            times=ecc_water_in.times,
            temp_in=ecc_water_in.datas,
            temp_out=ecc_water_out.datas,
        )
        ecc_water_temp_single.show(save_plots=args.savefig)

        ecc_water_temp = TemperatureDisplayInfos(
            run=args.run,
            name="Water Heating",
            times=ecc_water_in.times,
            datas=ecc_water_delta,
        )
        ecc_water_temp.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        water_flow = dbinfos.tel[args.telId].monitoring_ecc.water_flow
        ecc_water_flow = DisplayTimeInfos(
            run=args.run,
            name="Water",
            infos="Flow",
            units="l/min",
            times=water_flow.times,
            datas=water_flow.datas,
        )
        ecc_water_flow.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        water_flow_m3_per_s = water_flow.datas * 1e-3 / 60.0
        ## meg 30% at 14 deg
        rho = 1048
        cp = 3661
        dissipated_energy = cp * rho * ecc_water_delta * water_flow_m3_per_s
        water_dissipated_energy = DisplayTimeInfos(
            run=args.run,
            name="Water Dissipated",
            infos="Power",
            units="W",
            times=water_flow.times,
            datas=dissipated_energy,
        )
        water_dissipated_energy.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        cur_phase1 = dbinfos.tel[args.telId].monitoring_ecc.current_phase1
        cur_phase2 = dbinfos.tel[args.telId].monitoring_ecc.current_phase2
        cur_phase3 = dbinfos.tel[args.telId].monitoring_ecc.current_phase3

        cur_3phase = CurrentPhaseDisplayInfos(
            run=args.run,
            name="Phase",
            times=cur_phase1.times,
            phase1=cur_phase1.datas,
            phase2=cur_phase2.datas,
            phase3=cur_phase3.datas,
        )
        cur_3phase.show(save_plots=args.savefig)

        total_current = CurrentDisplayInfos(
            run=args.run,
            name="Phase Sum",
            times=cur_phase1.times,
            datas=(cur_phase1.datas + cur_phase2.datas + cur_phase3.datas),
        )
        total_current.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        #        from IPython import embed
        #        embed()
        voltage_phase1 = dbinfos.tel[args.telId].monitoring_ecc.voltage_phase1
        voltage_phase2 = dbinfos.tel[args.telId].monitoring_ecc.voltage_phase2
        voltage_phase3 = dbinfos.tel[args.telId].monitoring_ecc.voltage_phase3

        voltage_3phase = VoltagePhaseDisplayInfos(
            run=args.run,
            name="Phase",
            times=voltage_phase1.times,
            phase1=voltage_phase1.datas,
            phase2=voltage_phase2.datas,
            phase3=voltage_phase3.datas,
        )
        voltage_3phase.show(save_plots=args.savefig)
        # total_voltage = VoltageDisplayInfos(run=args.run,name="Phase Sum",times=voltage_phase1.times,datas=(voltage_phase1.datas+voltage_phase2.datas+voltage_phase3.datas))
        # total_voltage.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        kva1 = cur_phase1.datas * voltage_phase1.datas
        kva2 = cur_phase2.datas * voltage_phase2.datas
        kva3 = cur_phase3.datas * voltage_phase3.datas
        kvatot = kva1 + kva2 + kva3

        kva_3phase = PhaseDisplayInfos(
            run=args.run,
            name="Phase",
            infos="VoltAmpere",
            units="VA",
            times=voltage_phase1.times,
            phase1=kva1,
            phase2=kva2,
            phase3=kva3,
        )
        kva_3phase.show(save_plots=args.savefig)

        kva_sumphase = DisplayTimeInfos(
            run=args.run,
            name="Phase Sum",
            infos="VoltAmpere",
            units="VA",
            times=voltage_phase1.times,
            datas=kvatot,
        )
        kva_sumphase.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        # print("power")
        power1 = dbinfos.tel[args.telId].monitoring_ecc.active_power_phase1
        power2 = dbinfos.tel[args.telId].monitoring_ecc.active_power_phase2
        power3 = dbinfos.tel[args.telId].monitoring_ecc.active_power_phase3
        powerups = dbinfos.tel[args.telId].monitoring_ecc.active_power_ups
        powertot = power1.datas + power2.datas + power3.datas + powerups.datas

        power_3phase = PhaseDisplayInfos(
            run=args.run,
            name="Phase",
            infos="Active Power",
            units="kW",
            times=power1.times,
            phase1=power1.datas,
            phase2=power2.datas,
            phase3=power3.datas,
        )
        power_3phase.show(save_plots=args.savefig)

        power_upsphase = DisplayTimeInfos(
            run=args.run,
            name="UPS",
            infos="Active Power",
            units="kW",
            times=power1.times,
            datas=powerups.datas,
        )
        power_upsphase.show(save_plots=args.savefig)

        power_sumphase = DisplayTimeInfos(
            run=args.run,
            name="Phase Sum",
            infos="Active Power",
            units="kW",
            times=power1.times,
            datas=powertot,
        )
        power_sumphase.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        ## Hidden Loss
        # dissipated_energy = cp*rho*ecc_water_delta*water_flow_m3_per_s

        # water_dissipated_energy = DisplayTimeInfos(run=args.run,name="Water Dissipated",infos="Power",units="W",times=water_flow.times,datas=dissipated_energy)
        # water_dissipated_energy.show(save_plots=args.savefig)

        hidden_power = 1000.0 * powertot - dissipated_energy
        hidden_dissipated_energy = DisplayTimeInfos(
            run=args.run,
            name="Hidden Dissipated",
            infos="Power",
            units="W",
            times=water_flow.times,
            datas=hidden_power,
        )
        hidden_dissipated_energy.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    try:
        ## Flatfield
        ff_temps = dbinfos.monitoring_ffcls.temperature.datas
        ff_times = dbinfos.monitoring_ffcls.temperature.times
        ff_disp = TemperatureDisplayInfos(
            run=args.run, name="FlatField", times=ff_times, datas=ff_temps
        )
        ff_disp.show(save_plots=args.savefig)
    except Exception as err:
        print(err)

    ## UCTS

    ## PDB

    ## Average Temperature

    ##

    ## Water Temperature

    # from IPython import embed
    # embed()

    if not args.batch:
        plt.show()

    return 0

    # run_dbinfos[run] = dbinfos

    ## Show Temperature Stats (mean,std,min,max)

    ## Show Temperature asymmetry

    ## Show HV stats (mean, std, min, max)

    ## If long run : Show time evolution of some information


if __name__ == "__main__":
    retval = ShowMonitoringInfos(sys.argv[1:])
    sys.exit(retval)
