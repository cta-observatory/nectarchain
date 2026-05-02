# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: nectardev2
#     language: python
#     name: python3
# ---

# %%
try:
    from dateutil.parser import parse, ParserError
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from matplotlib import dates
    import pandas as pd
    import numpy as np
    import astropy.units as u
    import warnings
    import datetime

    from ctapipe.instrument import CameraGeometry
    from ctapipe.coordinates import EngineeringCameraFrame
    from ctapipe.visualization import CameraDisplay

    # in vmarandon scripts at the moment
    from nectarchain.utils.dbhandler import DBInfos, to_datetime

    # from CalibrationCameraDisplay import CalibrationCameraDisplay
    # from Utils import GetCamera

except ImportError as err:
    print(err)

try:
    from ctapipe.instrument.warnings import FromNameWarning

    CTAPipeWarningExist = True
except ImportError:
    CTAPipeWarningExist = False


def GetCamera(cam_name="NectarCam-003"):  # Copied from my Utils code
    if CTAPipeWarningExist:
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=FromNameWarning)
            # Let's ignore the warning for the moment
            cam = CameraGeometry.from_name(cam_name).transform_to(
                EngineeringCameraFrame()
            )
    else:
        cam = CameraGeometry.from_name(cam_name).transform_to(EngineeringCameraFrame())

    return cam


# %%
# path where the data are if using a run number
path = "/Users/vm273425/Programs/NectarCAM/data_ssd/camera2/"

# path of where is the sqlite file (can be in a sub-directory)
db_data_path = "/Users/vm273425/Programs/NectarCAM/data_ssd/camera2/monitoring"

telid = 2

# %%
# define the time interval
try:
    # expected date in UTC
    begin_time = to_datetime(parse("2025-11-10 16:00:00"))
    end_time = to_datetime(parse("2025-11-11 14:00:00"))
except ParserError as err:
    print(err)

print(begin_time, end_time)

# %%
# Create DB Instance
dbinfos = DBInfos.init_from_time(
    begin_time, end_time, dbpath=db_data_path, verbose=False
)

# Could be done using a run number way also, it will look for a run:
# dbinfos = DBInfos.init_from_run(run, path=path, dbpath=db_data_path,verbose=False)

# %%
# show available tables
dbinfos.show_available_tables()

# %%
# Show available infos per table
dbinfos.show_available_infos()

# %%
# Load the information in memory (can be long :-( )
dbinfos.connect(
    "monitoring_ffcls", "monitoring_drawer_temperatures", "monitoring_channel_voltages"
)  # Add any table that are present in the DB
# dbinfos.connect("*") # Load everything
# dbinfos.connect("monitoring_drawer_temperatures","monitoring_channel_currents","monitoring_channel_voltages","monitoring_ib","monitoring_ffcls")
# dbinfos.connect("monitoring_drawer_temperatures","monitoring_ffcls","monitoring_tib_scalers")

# %%
# Retrieve temperature of the modules
temperatures = dbinfos.tel[telid].monitoring_drawer_temperatures.tfeb1.datas
temperatures_times = dbinfos.tel[telid].monitoring_drawer_temperatures.tfeb1.times
print(type(temperatures))

# %%
# Interpolation can be done with the method "at"
delta_t = (end_time - begin_time).total_seconds()
steps = 100

interp_times = [
    begin_time + n * datetime.timedelta(seconds=delta_t / steps) for n in range(steps)
]
interp_tfeb1_temps = dbinfos.tel[telid].monitoring_drawer_temperatures.tfeb1.at(
    interp_times
)

pix = 1855 // 2
central_interp_tfeb1 = interp_tfeb1_temps.to_pixel()[pix]
central_datas_tfeb1 = dbinfos.tel[
    telid
].monitoring_drawer_temperatures.tfeb1.datas.to_pixel()[pix]
central_times_tfeb1 = dbinfos.tel[telid].monitoring_drawer_temperatures.tfeb1.times

plt.figure(figsize=(18, 6))
plt.plot(central_times_tfeb1, central_datas_tfeb1)
plt.plot(interp_times, central_interp_tfeb1, "o")

# %%
nrows = 1
ncols = 1
size_y = 4
size_x = 3 * size_y
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(size_x * ncols, size_y * nrows)
)

ax = axs
modules = [124, 132, 140]
for m in modules:
    ax.plot(temperatures_times, temperatures[m], label=f"Module: {m}")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel(f"FEB 1 Temperature")
ax.set_title(f"FEB 1 Temperature Evolution")
ax.xaxis.set_major_formatter(dates.DateFormatter("%d/%m %H:%M"))
ax.tick_params("x", rotation=30)
ax.legend()
ax.grid()
fig.tight_layout()

# %%
# Plot temperature in the camera

pix_temperature = np.nanmean(temperatures, axis=-1).to_pixel()
# temperatures is of type ModuleArray which is a numpy array with an additionnal function to convert to pixel

nrows = 1
ncols = 1
size = 6
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows))
cam = CameraDisplay(
    geometry=GetCamera(),
    cmap="turbo",
    image=pix_temperature,
    title=f"FEB 1 Temperature",
    ax=axs,
    show_frame=False,
    allow_pick=True,
    norm="lin",
)
cam.highlight_pixels(range(1855), color="grey", linewidth=0.2)
# cam.highlight_pixels( range(7*132,7*133), color = "white" ) # highlight the central module
# CalibrationCameraDisplay is an overload of the CameraDisplay that can highlight differently pixels and easily add function if pixels are clicked
cam.add_colorbar()
fig.tight_layout()
# cam1.colorbar.set_label(f'%')

# %%
# Retrieve hv of the pixels
hvs = dbinfos.tel[telid].monitoring_channel_voltages.voltage.datas
hvs_times = dbinfos.tel[telid].monitoring_channel_voltages.voltage.times
print(type(hvs))

# %%
nrows = 1
ncols = 1
size_y = 4
size_x = 3 * size_y
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(size_x * ncols, size_y * nrows)
)

ax = axs
pixels = [124 * 7 + 3, 132 * 7 + 3, 140 * 7 + 3]
for p in pixels:
    ax.plot(hvs_times, hvs[p], label=f"Pixel: {p}")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel(f"HV (V)")
ax.set_title(f"HV Evolution")
ax.xaxis.set_major_formatter(dates.DateFormatter("%d/%m %H:%M"))
ax.tick_params("x", rotation=30)
ax.legend()
ax.grid()
fig.tight_layout()

# %%
# Plot HV in the camera

hv_means = np.nanmean(hvs, where=hvs > 400, axis=-1)
hv_stds = np.nanstd(hvs, where=hvs > 400, axis=-1, ddof=1)

# hv is of type PixelArray which is a numpy array with additionnal feature

nrows = 1
ncols = 2
size = 6
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows))

ax = axs[0]
cam = CameraDisplay(
    geometry=GetCamera(),
    cmap="turbo",
    image=hv_means,
    title=f"Average HV",
    ax=ax,
    show_frame=False,
    allow_pick=True,
    norm="lin",
)
cam.highlight_pixels(range(1855), color="grey", linewidth=0.2)
# cam.highlight_pixels( range(7*132,7*133), color = "white" ) # highlight the central module
cam.add_colorbar()

ax = axs[1]
cams = CameraDisplay(
    geometry=GetCamera(),
    cmap="turbo",
    image=hv_stds,
    title=f"Std HV",
    ax=ax,
    show_frame=False,
    allow_pick=True,
    norm="lin",
)
cams.highlight_pixels(range(1855), color="grey", linewidth=0.2)
# cams.highlight_pixels( range(7*132,7*133), color = "white" ) # highlight the central module
cams.add_colorbar()
# CalibrationCameraDisplay is an overload of the CameraDisplay that can highlight differently pixels and easily add function if pixels are clicked


fig.tight_layout()

# %%
