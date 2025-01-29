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
# %matplotlib inline

# %%
import numpy as np
from matplotlib import pyplot as plt

from astropy import time as astropytime
from ctapipe.io import EventSource, EventSeeker
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from ctapipe.image import (
    tailcuts_clean,
    dilate,
    hillas_parameters,
    HillasParameterizationError,
)

# %%
lfns = [
    "/vo.cta.in2p3.fr/nectarcam/2024/20240910/NectarCAM.Run5661.0000.fits.fz",
    "/vo.cta.in2p3.fr/nectarcam/2024/20240910/NectarCAM.Run5661.0001.fits.fz",
    "/vo.cta.in2p3.fr/nectarcam/2024/20240910/NectarCAM.Run5661.0002.fits.fz",
    "/vo.cta.in2p3.fr/nectarcam/2024/20240910/NectarCAM.Run5661.0003.fits.fz",
    "/vo.cta.in2p3.fr/nectarcam/2024/20240910/NectarCAM.Run5661.0005.fits.fz"
    "/vo.cta.in2p3.fr/nectarcam/2024/20240910/NectarCAM.Run5661.0006.fits.fz",
    "/vo.cta.in2p3.fr/nectarcam/2024/20240910/NectarCAM.Run5661.0007.fits.fz",
]
cmap = "gnuplot2"

# %%
import os

tmpdir = f"/tmp/{os.environ['USER']}/scratch"
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

dirac = Dirac()

dirac.getFile(lfn=lfns, destDir=tmpdir, printOutput=True)

# %%
pwd = glob(f"{tmpdir}/NectarCAM.*.fits.fz")
print(pwd)

# %%
source = EventSource.from_url(input_url=pwd[0], max_events=100)

for event in source:
    print(event.index.event_id, event.trigger.event_type, event.trigger.time)

# %%
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.io import EventSource, EventSeeker
from ctapipe.instrument import CameraGeometry

camera = CameraGeometry.from_name("NectarCam-003").transform_to(
    EngineeringCameraFrame()
)
disp = CameraDisplay(geometry=camera)

chan = 0
disp.image = event.mon.tel[0].pixel_status.hardware_failing_pixels[chan]
disp.set_limits_minmax(0, 1)
disp.cmap = plt.cm.coolwarm
disp.add_colorbar()

# %%
from ctapipe.image.extractor import NeighborPeakWindowSum

extractor = NeighborPeakWindowSum(source.subarray)

# %%
n_pixels = 1855
broken_pixels = np.zeros(n_pixels, dtype=bool)

# %%
waveforms = event.r1.tel[0].waveform[0, np.arange(n_pixels), :]
n_pixels = waveforms.shape[1]

charge = extractor(waveforms, 0, selected_gain_channel=0, broken_pixels=broken_pixels)

# %%
charge

# %%
source = EventSource.from_url(input_url=pwd[0])
import pandas as pd

charge_df = pd.DataFrame()
pixel_id = 0  # look at a particular pixel
times = []
charges = []
for event in source:
    charge = extractor(
        event.r1.tel[0].waveform[0, np.arange(n_pixels), :],
        0,
        selected_gain_channel=0,
        broken_pixels=broken_pixels,
    )  # Extract signal (charge)
    time = event.trigger.time.value + charge.peak_time[pixel_id]
    print(time)
    times.append(time)
    charge_pix = charge.image[pixel_id]
    print(charge_pix)
    charges.append(charge_pix)
    print(event.index.event_id)

# %%
charge_df["time"] = times
charge_df["charge"] = charges
plt.scatter(charge_df["time"], charge_df["charge"])

# %%
from ctapipe.containers import EventType

source = EventSource.from_url(input_url=pwd[0])
channel = [0, 1]

tel_id = 0

# read first pedestal event

for i, event in enumerate(source):
    if event.trigger.event_type == EventType.SKY_PEDESTAL:
        break

# %%
print(f"read event id: {event.index.event_id}, trigger {event.trigger.event_type}")

# %%
disp = CameraDisplay(source.subarray.tel[0].camera.geometry)

chan = 0
disp.image = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[chan]
disp.set_limits_minmax(0, 1)
disp.cmap = plt.cm.coolwarm
disp.axes.text(2.4, 0, "failing pixels", rotation=90)
disp.add_colorbar()


# %%
def view_waveform(chan=0, pix_id=6):
    waveform = event.r1.tel[tel_id].waveform
    plt.plot(waveform[chan, pix_id], label=f"pixel {pix_id}")

    plt.title(
        f"module {module}, channel {channel[chan]}",
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
#
# channel
chan = 0
# ids of pixel in module
# pixels_mod=event.nectarcam.tel[0].svc.pixel_ids[module*7:module*7+7]

fig = plt.figure(num=0, figsize=(12, 12))

for i, pix in enumerate(pixels_mod):
    view_waveform(chan=chan, pix_id=pix)

# %%
# integrate the charge on 12 ns around the peak value
from traitlets.config.loader import Config
from ctapipe.image.extractor import FixedWindowSum

config = Config({"FixedWindowSum": {"window_width": 12}})
# integrator = LocalPeakWindowSum(config=config)
integrator = FixedWindowSum(source.subarray, config=config)


waveform = event.r0.tel[0].waveform
image = integrator(waveform, 0, selected_gain_channel=0, broken_pixels=broken_pixels)

fig = plt.figure(figsize=(16, 8))
for chan in np.arange(2):
    ax = plt.subplot(1, 2, chan + 1)

    disp = CameraDisplay(source.subarray.tel[0].camera.geometry)
    disp.image = image.image[chan]
    # disp.set_limits_minmax(0,200)
    disp.cmap = plt.cm.coolwarm
    disp.axes.text(2.0, 0, f"{channel[chan]} charge (DC)", rotation=90)
    disp.add_colorbar()

    disp.update()

# %%
fig = plt.figure(0, figsize=(12, 6))

# consider only 36 samples
samples = np.arange(0, 60)

# chose the module
mod = 3
window_start = 12

# find pixel index in module
pix_in_mod = event.nectarcam.tel[0].svc.pixel_ids[mod * 7 : mod * 7 + 7]

for chan in np.arange(2):
    plt.subplot(1, 2, chan + 1)

    for i, pix in enumerate(pix_in_mod):
        # samples used to calculate the charge
        start = 12
        stop = 60
        used_samples = np.arange(start, stop)
        used = waveform[chan, pix, start:stop]

        plt.plot(waveform[chan, pix], color="b", label="all samples")
        plt.plot(used_samples, used, color="r", label="integrated samples")

        if i == 0:
            plt.legend()
        plt.ylabel("[ADC]")
        plt.xlabel(f"{channel[chan]}  waveforms in module {mod}")
        plt.ylim(0, 500)

# %%
source = EventSource.from_url(input_url=pwd[0])

for event in source:
    print("Id: {},  Telescopes: {}".format(event.count, len(event.r0.tel)))

# %%
source.subarray.camera_types

# %%
len(event.r0.tel), len(event.r1.tel)

# %%
from ctapipe.calib import CameraCalibrator

calibrator = CameraCalibrator(subarray=source.subarray)

# %%
calibrator(event)

# %%

# %%

# %%
event.mon.tel[tel_id].pedestal

# %%
fig = plt.figure(11, figsize=(16, 5))
# mask=  np.logical_or(ped_data.charge_median_outliers, status_data.hardware_failing_pixels)

image = ped_data.charge_median
# plt.savefig(f"std_{channel[chan]}.png")
chan = 0

ax = plt.subplot(1, 2, 1)
disp = CameraDisplay(source.subarray.tel[0].camera.geometry)
# disp.highlight_pixels(mask[chan])
disp.image = image[chan]
disp.cmap = plt.cm.coolwarm
disp.add_colorbar()
disp.axes.text(2.4, 0, "charge median", rotation=90)
ax = plt.subplot(1, 2, 2)

chan = 1

disp = CameraDisplay(source.subarray.tel[0].camera.geometry)
# disp.highlight_pixels(mask[chan])
disp.image = image[chan]
disp.cmap = plt.cm.coolwarm
disp.axes.text(2.4, 0, "charge median", rotation=90)
disp.add_colorbar()
# plt.savefig(f"Run{run}_pedestal_median_r1_over_camera.png")
# plot data
fig = plt.figure(10, figsize=(16, 5))


image = ped_data.charge_std
mask = np.logical_or(ped_data.charge_std_outliers, status_data.hardware_failing_pixels)
chan = 0
ax = plt.subplot(1, 2, 1)
disp = CameraDisplay(source.subarray.tel[0].camera.geometry)
# disp.highlight_pixels(mask[chan])
disp.image = image[chan]
disp.cmap = plt.cm.coolwarm
disp.axes.text(2.4, 0, "charge std", rotation=90)
disp.add_colorbar()

ax = plt.subplot(1, 2, 2)
chan = 1
disp = CameraDisplay(source.subarray.tel[0].camera.geometry)
# disp.highlight_pixels(mask[chan])
disp.image = image[chan]
disp.cmap = plt.cm.coolwarm
disp.add_colorbar()
disp.axes.text(2.4, 0, "charge std", rotation=90)
# plt.savefig(f"Run{run}_pedestal_std_r1_over_camera.png")
#
# plt.savefig(f"pedestal_median_r1.png")

# %%
print(ped_data.charge_std_outliers)

# %%

# %%
plt.pcolormesh(teldata.waveform[0])
plt.colorbar()
plt.ylim(700, 750)
plt.xlabel("sample number")
plt.ylabel("pixel_id")
print("waveform[0] is an array of shape (N_pix,N_slice) =", teldata.waveform[0].shape)

# %%
trace = teldata.waveform[0][719]

plt.plot(trace, drawstyle="steps")

# %%
for pix_id in range(718, 723):
    plt.plot(
        teldata.waveform[0][pix_id], label="pix {}".format(pix_id), drawstyle="steps"
    )

plt.legend()

# %%
camgeom = source.subarray.tel[0].camera.geometry

# %%
data = teldata.waveform[0]
peds = data[:, 10:20].mean(axis=1)
sums = data[:, 25:35].sum(axis=1) / (18)

# %%
phist = plt.hist(peds, bins=50, range=[0, 150])

plt.title("Pedestal Distribution of all pixels for a single event")

# %%
plt.plot(sums - peds)

plt.xlabel("pixel id")

plt.ylabel("Pedestal-subtracted Signal")

# %%
# we can also subtract the pedestals from the traces themselves, which would be needed to compare peaks properly

for ii in range(300, 310):
    plt.plot(data[ii] - peds[ii], drawstyle="steps", label="pix{}".format(ii))

plt.legend()

# %%
title = "CT24, run {} event {} ped-sub".format(event.index.obs_id, event.index.event_id)
disp = CameraDisplay(camgeom, title=title)
disp.image = sums - peds
disp.cmap = plt.cm.RdBu_r
disp.add_colorbar()
disp.set_limits_percent(95)  # autoscale

# %%
# ?source

# %%

# %%
event.trigger.time

# %% [markdown]
# ## Look at waveform image for a particular event

# %%
evt = seeker.get_event_index(25)
import time
from IPython import display

adcsum = evt.r0.tel[0].waveform[0].sum(axis=1)

camera = CameraGeometry.from_name("NectarCam-003")

for i in range(len(evt.r0.tel[0].waveform[0].T)):
    image = evt.r0.tel[0].waveform[0].T[i]
    plt.clf()

    fig = plt.figure(figsize=(13, 9))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    disp2 = CameraDisplay(
        geometry=camera, image=adcsum, ax=ax1, title="Sum ADC", cmap=cmap
    )
    # disp2.cmap = cmap
    # disp2.add_colorbar()

    disp = CameraDisplay(
        geometry=camera,
        image=image,
        ax=ax2,
        title="Waveform (ADC), T={} ns".format(i),
        cmap=cmap,
    )
    # disp.cmap = cmap
    disp.add_colorbar()

    display.display(plt.gcf())
    display.clear_output(wait=True)
    time.sleep(0.2)

# %% [markdown]
# ## Or look at an integrated charge image for a particular event

# %%
evt = seeker.get_event_index(0)
image = evt.r0.tel[0].waveform.sum(axis=2)
disp = CameraDisplay(geometry=camera, image=image[0], cmap=cmap)

# %%
run_start = astropytime.Time(evt.nectarcam.tel[0].svc.date, format="unix").iso
print("Run started at {}".format(run_start))

# %%
evt.index.event_id

# %% [markdown]
# ## Extract interleaved pedestals

# %%
# Evaluate pedestal from interleaved pedestals from same input run
max_events = 500
read_ped = EventSource(input_url=path[0])
peds = []
for i, ev in enumerate(read_ped):
    if len(peds) > max_events:
        break
    if ev.trigger.event_type == 32:
        # print('Event {}, trigger type {}'.format(i,ev.r0.tel[0].trigger_type))
        wfs = ev.r0.tel[0].waveform
        wfs_hi = wfs[0].sum(axis=1)
        peds.append(wfs_hi)
peds = np.array(peds)
peds = peds.mean(axis=0)

# %%
peds.shape

# %%
disp = CameraDisplay(geometry=camera, image=peds, cmap=cmap)
disp.cmap = cmape
disp.add_colorbar()

# %%
camera

# %%
plt.plot(peds)

# %% [markdown]
# ## Calibration

# %%
adc_to_pe = 58.0
evt = next(iter(seeker))
print("Event {}, trigger type {}".format(evt.index.event_id, evt.trigger.event_type))
if evt.trigger.event_type == 1:
    raw = evt.r0.tel[0].waveform[0].sum(axis=1)
    charges = (raw - peds) / adc_to_pe
disp = CameraDisplay(geometry=camera, image=charges, cmap="gnuplot2")
disp.cmap = cmap
disp.add_colorbar()

# Comment: if this cell says that "charges" is not defined it's because the event type is not 1. Re-run it.

# %% [markdown]
# ## Hillas cleaning

# %%
cleanmask = tailcuts_clean(
    camera,
    charges,
    picture_thresh=10,
    boundary_thresh=5,
    min_number_picture_neighbors=3,
)
charges[~cleanmask] = 0
try:
    hillas_param = hillas_parameters(camera, charges)
    disp = CameraDisplay(geometry=camera, image=charges, cmap="gnuplot2")
    disp.cmap = cmap
    disp.add_colorbar()
    disp.overlay_moments(
        hillas_param,
        with_label=False,
        color="red",
        alpha=0.7,
        linewidth=2,
        linestyle="dashed",
    )
    disp.highlight_pixels(cleanmask, color="white", alpha=0.3, linewidth=2)
    print(hillas_param)
except HillasParameterizationError:
    pass
print("Cleaned image: charge = {} pe".format(charges.sum()))

# %% [markdown]
# ## Loop over events

# %%
import time
from IPython import display

for i, evt in enumerate(reader):
    if evt.trigger.event_type == 1:
        raw = evt.r0.tel[0].waveform[0].sum(axis=1)
        charges = (raw - peds) / adc_to_pe
        cleanmask = tailcuts_clean(
            camera,
            charges,
            picture_thresh=10,
            boundary_thresh=5,
            min_number_picture_neighbors=3,
        )
        charges[~cleanmask] = 0

        plt.clf()
        disp = CameraDisplay(geometry=camera, image=charges, cmap="gnuplot2")
        disp.cmap = cmap
        disp.add_colorbar()
        try:
            hillas_param = hillas_parameters(camera, charges)
            disp.overlay_moments(
                hillas_param,
                with_label=False,
                color="red",
                alpha=0.7,
                linewidth=2,
                linestyle="dashed",
            )
            disp.highlight_pixels(cleanmask, color="white", alpha=0.3, linewidth=2)
        except HillasParameterizationError:
            pass
        display.display(plt.gcf())
        display.clear_output(wait=True)
        time.sleep(0.2)

# %%

# %%

# %%
