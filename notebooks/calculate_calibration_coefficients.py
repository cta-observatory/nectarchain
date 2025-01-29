# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from corsikaio.subblocks import event_end_types
from sqlalchemy import values
# %load_ext autoreload


from ctapipe.io import EventSource, EventSeeker
import sys
 
from matplotlib import pyplot as plt
import numpy as np
# %matplotlib inline
import sys
from scipy.stats import norm
from traitlets.config.loader import Config 
from ctapipe import utils

# ctapipe modules
from ctapipe.visualization import CameraDisplay
from ctapipe.image.extractor import *

from ctapipe.io.hdf5tableio import HDF5TableWriter, HDF5TableReader
from ctapipe.instrument import TelescopeDescription, CameraGeometry
from ctapipe.coordinates import EngineeringCameraFrame

camera = CameraGeometry.from_name("NectarCam-003").transform_to(EngineeringCameraFrame())



# %%
# %env NECTARCAMDATA = /tmp/amikhno/scratch
import os

tmpdir = f"/tmp/{os.environ['USER']}/scratch/runs"
if not os.path.isdir(tmpdir):
    print(f"{tmpdir} does not exist yet, I will create it for you")
    os.makedirs(tmpdir)

# %%
import DIRAC
import ZEO
from astropy.io import fits
from DIRAC.Interfaces.API.Dirac import Dirac
from glob import glob
from nectarchain.dqm.db_utils import DQMDB

cmap = "gnuplot2"

run = 5922  # 22/05

from nectarchain.data.management import DataManagement

dm = DataManagement()
_, filelist = dm.findrun(run)

print(filelist)

# %%
pwd = glob(f"{tmpdir}/NectarCAM.*.fits.fz")
print(pwd)

# %%
from ctapipe.containers import EventType

source = EventSource.from_url(input_url=filelist[0])

# %%
for event in source:
    print(event.index.event_id, event.trigger.event_type, event.trigger.time)

# %%
# ?event

# %%
for event in source:
    # select only flatfield events
    print(f"read event id: {event.index.event_id }, trigger {event.trigger.event_type}")
    if event.trigger.event_type == EventType.FLATFIELD:
        break
    # elif event.trigger.event_type == SKY_PEDESTAL or  eevent.trigger.event_type == SUBARRAY:
    #    break

print(f"read event id: {event.index.event_id }, trigger {event.trigger.event_type}")

# %%
pix = 441

fig = plt.figure(300, figsize=(12, 6))
label = "r0"
label1 = "r1 = r0 - 250"
chan = 0
plt.subplot(1, 2, 1)
plt.plot(event.r1.tel[0].waveform[chan, pix], label=label)
plt.plot(event.r0.tel[0].waveform[chan, pix], label=label1)
plt.title(f"pixel {pix}, channel {[chan]}")
plt.ylim(-100, 3000)
plt.legend()
chan = 1
plt.subplot(1, 2, 2)
plt.ylim(-100, 500)
plt.plot(event.r1.tel[0].waveform[chan, pix], label=label)
plt.plot(event.r0.tel[0].waveform[chan, pix], label=label1)
plt.title(f"pixel {pix}, channel {[chan]}")
plt.legend()
# plt.savefig(f"flatfield_outliers_pixel{pix}_event_{event.r0.event_id}_run{run}.png")

# %%
# plot R1 waveform of module [module]
mod = 1
tel_id = 0
module_id = event.nectarcam.tel[0].svc.module_ids[mod]


def view_waveform(chan=0, pix_id=6):
    waveform = event.r1.tel[tel_id].waveform
    plt.plot(waveform[chan, pix_id], label=f"pixel {pix_id}")

    plt.title(
        f"module {module_id},  channel {[chan]}",
    )
    max_now = waveform[chan, pix_id].max()
    min_now = waveform[chan, pix_id].min()

    plt.legend()
    plt.ylabel("DC", fontsize=15)
    plt.xlabel("ns", fontsize=15)


# module number

module = 63
module_rank = np.where(event.nectarcam.tel[0].svc.module_ids == module)

# ids of pixel in module
pixels_mod = event.nectarcam.tel[0].svc.pixel_ids[
    module_rank[0][0] * 7 : module_rank[0][0] * 7 + 7
]

# channel
chan = 0

fig = plt.figure(num=0, figsize=(8, 8))

for i, pix in enumerate(pixels_mod):
    view_waveform(chan=chan, pix_id=pix)

# plt.savefig(f"Run{run}_waveform_channel_{channel[chan]}_module_{module_id}.png")

# %%
# Comupte pedestals
waveform = event.r0.tel[0].waveform[1]
ped = np.mean(waveform[:, 20])

# %%
waveform

# %%
window = 15
# wfs = []
wfs = event.r0.tel[0].waveform
ma.shape(wfs)

# %%
ped_mean = np.mean(wfs[:, :, 0:window], axis=2)
ped_mean
ped_mean

# %%
inf_mask = np.isinf(ped_mean)
inf_mask

# %%
ped_mean_expanded = ped_mean[:, :, None]  # Shape: (2, 1855, 1)
inf_mask_expanded = inf_mask[:, :, None]  # Shape: (2, 1855, 1)

# Perform subtraction and set wfs_pedsub to 0 where ped_mean is `inf`
wfs_pedsub = np.where(inf_mask_expanded, 0, wfs - ped_mean_expanded)

# Should be (2, 1855, 60)

# %%
wfs_pedsub[:, :, 20]

# %%
wfs_pedsub[0]

# %%
# integrate the charge on 12 ns around the peak value
from ctapipe_io_nectarcam import constants  # to get the constants

n_pixels = 1855
broken_pixel = np.zeros(n_pixels, dtype=bool)

config = Config({"LocalPeakWindowSum": {"window_shift": 4, "window_width": 10}})
integrator = LocalPeakWindowSum(source.subarray, config=config)


waveform = event.r1.tel[0].waveform

# %%
from ctapipe_io_nectarcam import constants

constants.HIGH_GAIN

import numpy as np

gain = constants.HIGH_GAIN

if gain == constants.HIGH_GAIN:
    selected_gain_channel = np.zeros(constants.N_PIXELS)
elif gain == constants.LOW_GAIN:
    selected_gain_channel = np.ones(constants.N_PIXELS)
else:
    print("something is wrong")
    selected_gain_channel = None

print(selected_gain_channel)
high_gain_channel = np.round(selected_gain_channel).astype(int)

# %%
gain_channels = np.stack((high_gain_channel, low_gain), axis=0)
gain_channels

# %%
broken_pixels = np.stack((broken_pixel, broken_pixel), axis=0)
broken_pixels

# %%
waveforms = event.r1.tel[0].waveform[0, np.arange(n_pixels), :]
waveforms.shape

# %%
print(f"tel_id type: {type(tel_id)}")
print(f"gain_channels type: {type(gain_channels)}")
print(f"broken_pixel type: {type(broken_pixel)}")

# %%
waveform = event.r0.tel[0].waveform
image = integrator(wfs_pedsub, 0, selected_gain_channel=0, broken_pixels=broken_pixels)

# %%
image

# %%
fig = plt.figure(figsize=(16, 16))
for chan in np.arange(2):
    ax = plt.subplot(2, 2, chan + 1)

    disp = CameraDisplay(source.subarray.tel[0].camera.geometry)
    disp.image = image.image[chan]

    # disp.set_limits_minmax(2000,4000)
    disp.cmap = plt.cm.coolwarm
    disp.axes.text(2.0, 0, f"channel {[chan]} charge (DC)", fontsize=10, rotation=90)
    disp.add_colorbar()

    ax = plt.subplot(2, 2, chan + 3)
    disp = CameraDisplay(source.subarray.tel[0].camera.geometry)
    disp.image = image.peak_time[chan]
    disp.cmap = plt.cm.coolwarm
    disp.set_limits_minmax(10, 30)
    disp.axes.text(2.0, 0, f"channel {[chan]} time (ns)", fontsize=10, rotation=90)

    disp.add_colorbar()

    disp.update()

# plt.savefig(f"Run{run}_event_{event.nectarcam.tel[0].evt.event_id}_charge_time.png")

# %%
image.image

# %%
mod = 1
event.nectarcam.tel[0].svc.pixel_ids[mod * 7 : mod * 7 + 7]

# %%
image.peak_time[0, 70] - integrator.window_shift.tel[0]

# %%
int(max(image.peak_time[chan, pix] - integrator.window_shift.tel[0], 0))

# %%
# Plot the part of the waveform that is integrated
# (this work only after the line above)

fig = plt.figure(0, figsize=(14, 8))

# consider only 36 samples
samples = np.arange(0, 60)

# chose the module
mod = 1
module_id = event.nectarcam.tel[0].svc.module_ids[mod]
# find pixel index in module
pix_in_mod = event.nectarcam.tel[0].svc.pixel_ids[mod * 7 : mod * 7 + 7]

for chan in np.arange(2):
    plt.subplot(1, 2, chan + 1)

    for i, pix in enumerate(pix_in_mod):
        # samples used to calculate the charge
        start = int(max(image.peak_time[chan, pix] - integrator.window_shift.tel[0], 0))
        stop = int(min(start + integrator.window_width.tel[0], 60))
        used_samples = np.arange(start, stop)
        used = wfs_pedsub[chan, pix, start:stop]

        plt.plot(
            wfs_pedsub[
                chan,
                pix,
            ],
            color="b",
            label="all samples",
        )
        plt.plot(used_samples, used, color="r", label="integrated samples")

        if i == 0:
            plt.legend()

        # plt.ylim(-150,300)
        plt.ylabel("[DC]", fontsize=20)
        plt.xlabel(f"channel {[chan]}  waveforms in module {module_id}", fontsize=10)
        # plt.ylim(-50,4000)
        # plt.legend()
        fig.savefig(f"Run{run}_waverforms_module_zoom_{module_id}.png")

        # fig.savefig(f"Run{run}_event_75_all.png")

# %%
signal_mean = np.mean(image.image[:], axis=-1)
signal_var = np.var(image.image[:], axis=-1)

# %%
signal_var, signal_mean
gain = signal_mean / signal_var
gain

# %%
pixel_spec_signal = image.image[:] / (np.expand_dims(gain[:], axis=-1))
pixel_spec_signal

# %%
mean_camera_signal = np.expand_dims((signal_mean[:] / gain[:]), axis=-1)
mean_camera_signal

# %%
# Ralative efficience

eff = pixel_spec_signal[:] / mean_camera_signal[:]
eff

# %%
ff = np.divide(1, eff, out=np.zeros_like(eff, dtype=float), where=eff != 0)
ff

# %%

# %%
fig = plt.figure(figsize=(16, 16))
for chan in np.arange(2):
    ax = plt.subplot(2, 2, chan + 1)

    disp = CameraDisplay(source.subarray.tel[0].camera.geometry)
    disp.image = ff[chan]

    # disp.set_limits_minmax(2000,4000)
    # disp.cmap = plt.cm.coolwarm
    disp.axes.text(2.0, 0, f"FF-coefs, channel - {[chan]}", fontsize=10, rotation=90)

    disp.add_colorbar()

    disp.update()

# %%
# Illuminating a weird pixel

ff_new = ff.copy()
ff_new[0, 1702] = 0
ff_new[1, 1702] = 0

# %%
fig = plt.figure(figsize=(16, 16))
for chan in np.arange(2):
    ax = plt.subplot(2, 2, chan + 1)

    disp = CameraDisplay(source.subarray.tel[0].camera.geometry)
    disp.image = ff_new[chan]

    # disp.set_limits_minmax(2000,4000)
    # disp.cmap = plt.cm.coolwarm
    disp.axes.text(2.0, 0, f"FF-coefs, channel - {[chan]}", fontsize=15, rotation=90)

    disp.add_colorbar()

    disp.update()

# %%
# To take into account wavefront shape

pixel_x = source.subarray.tel[0].camera.geometry.pix_x
pixel_y = source.subarray.tel[0].camera.geometry.pix_y

# Flasher center (x0, y0)
x0, y0 = 0.0, 0.0  # Adjust based on your flasher's location

# Compute distances from flasher center
distances = np.sqrt((pixel_x - x0) ** 2 + (pixel_y - y0) ** 2).value
distances

# %%
sigma = 8.0  # TO BE CHECKED
amplitude = 2.0  # normalize later

# Gaussian intensity
intensity = amplitude * np.exp(-(distances**2) / (2 * sigma**2))
expected_response = intensity / np.sum(intensity)

# %%

# %%
# Apply
pixel_spec_signal = image.image[:] / (
    np.expand_dims(gain[:], axis=-1) * expected_response
)
eff = pixel_spec_signal[:] / mean_camera_signal[:]
ff_gaus = np.divide(1, eff, out=np.zeros_like(eff, dtype=float), where=eff != 0)

# %%
fig = plt.figure(figsize=(16, 16))
for chan in np.arange(2):
    ax = plt.subplot(2, 2, chan + 1)

    disp = CameraDisplay(source.subarray.tel[0].camera.geometry)
    disp.image = ff_gaus[chan]

    # disp.set_limits_minmax(2000,4000)
    # disp.cmap = plt.cm.coolwarm
    disp.axes.text(2.0, 0, f"FF-coefs, channel - {[chan]}", fontsize=10, rotation=90)

    disp.add_colorbar()

    disp.update()

# %%
ff_gaus_new = ff_gaus.copy()
ff_gaus_new[0, 1702] = 0
ff_gaus_new[1, 1702] = 0

# %%
fig = plt.figure(figsize=(16, 16))
for chan in np.arange(2):
    ax = plt.subplot(2, 2, chan + 1)

    disp = CameraDisplay(source.subarray.tel[0].camera.geometry)
    disp.image = ff_gaus_new[chan]

    # disp.set_limits_minmax(2000,4000)
    # disp.cmap = plt.cm.coolwarm
    disp.axes.text(2.0, 0, f"FF-coefs, channel - {[chan]}", fontsize=15, rotation=90)

    disp.add_colorbar()

    disp.update()

# %%
image.image[:, 1702] = [0, 0]
signal_mean = np.mean(image.image[:], axis=-1)
signal_var = np.var(image.image[:], axis=-1)
gain = signal_mean / signal_var
print(gain)

pixel_spec_signal = image.image[:] / (np.expand_dims(gain[:], axis=-1))
mean_camera_signal = np.expand_dims((signal_mean[:] / gain[:]), axis=-1)
eff = pixel_spec_signal[:] / mean_camera_signal[:]
ff_test = np.divide(
    1, eff * expected_response, out=np.zeros_like(eff, dtype=float), where=eff != 0
)

# %%
fig = plt.figure(figsize=(16, 16))
for chan in np.arange(2):
    ax = plt.subplot(2, 2, chan + 1)

    disp = CameraDisplay(source.subarray.tel[0].camera.geometry)
    disp.image = ff_test[chan]

    # disp.set_limits_minmax(2000,4000)
    # disp.cmap = plt.cm.coolwarm
    disp.axes.text(2.0, 0, f"FF-coefs, channel - {[chan]}", fontsize=15, rotation=90)

    disp.add_colorbar()

    ax = plt.subplot(2, 2, chan + 3)
    disp = CameraDisplay(source.subarray.tel[0].camera.geometry)

    disp.image = 1 / expected_response
    # disp.image = 1/expected_response

    disp.axes.text(2.0, 0, f"channel {[chan]} time (ns)", fontsize=10, rotation=90)

    disp.add_colorbar()

    disp.update()

# %%

# %%

# %%
# use the tool to write calibration coefficients
# you can call it also as "python write_camera_calibration.py --help"

from nectarchain.tools.write_camera_calibration import CalibrationHDF5Writer

calibration_tool = CalibrationHDF5Writer()

# %%
calibration_tool.run()

# %%
# read back the monitoring containers written with the tool write_camera_calibration.py
from ctapipe.containers import FlatFieldContainer, WaveformCalibrationContainer
from ctapipe.io.hdf5tableio import HDF5TableWriter, HDF5TableReader

ff_data = FlatFieldContainer()
cal_data = WaveformCalibrationContainer()

with HDF5TableReader("calibration.hdf5") as h5_table:
    assert h5_table._h5file.isopen == True

    for cont in h5_table.read("/tel_0/flatfield", ff_data):
        print(cont.as_dict())

    for cont in h5_table.read("/tel_0/calibration", cal_data):
        print(cont.as_dict())

        break

h5_table.close()

chan = 0
values = 1 / cont.dc_to_pe[chan]
# Perform some plots
fig = plt.figure(13, figsize=(16, 5))
disp = CameraDisplay(camera)
disp.image = values
# disp.set_limits_minmax(0,1)
disp.cmap = plt.cm.coolwarm
disp.axes.text(2.4, 0, "photon electrons", rotation=90)
disp.add_colorbar()

#
select = np.logical_not(cont.unusable_pixels[0])


fig = plt.figure(12, figsize=(16, 5))
plt.hist(values[select], color="r", histtype="step", bins=50, stacked=True, fill=False)
plt.title(f"ADC per photon-electrons, mean={np.mean(values[select]):5.0f} ADC")

# %%
