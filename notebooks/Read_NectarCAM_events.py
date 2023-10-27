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

import time

# %%
import numpy as np
from astropy import time as astropytime
from ctapipe.image import HillasParameterizationError, hillas_parameters, tailcuts_clean
from ctapipe.instrument import CameraGeometry
from ctapipe.io import EventSeeker, EventSource
from ctapipe.visualization import CameraDisplay
from IPython import display
from matplotlib import pyplot as plt

# %%
path = "../obs/NectarCAM.Run1388.0001.fits.fz"
cmap = "gnuplot2"

# %%
reader = EventSource(input_url=path)

# %%
seeker = EventSeeker(reader)

# %% [markdown]
# ## Look at waveform image for a particular event

# %%
evt = seeker.get_event_index(25)
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
read_ped = EventSource(input_url=path)
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
disp = CameraDisplay(geometry=camera, image=peds, cmap=cmap)
disp.cmap = cmap
disp.add_colorbar()

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

# Comment: if this cell says that "charges" is not defined it's because the event
# type is not 1. Re-run it.

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
