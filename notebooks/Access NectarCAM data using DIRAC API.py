# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python (nectar-dev)
#     language: python
#     name: nectar-dev
# ---

# %% [markdown]
# # How to access NectarCAM data from the EGI grid directly using the DIRAC API ?
#
# In this short notebook, we will see how to access data stored on the grid using the
# DIRAC API, without the hassle of first manually downloading them locally.
#
# In order to achieve this, you will obviously need a `conda` environment in which all
# relevant code is installed, such as `ctapipe`, `nectarchain`, but also `CTADIRAC`
# itself. Please refer to the `nectarchain` installation procedure at
# <https://github.com/cta-observatory/nectarchain> and follow the instructions to enable
# DIRAC support.
#
# You will also need to have an active proxy for EGI, initialized e.g. with:
#
# ```
# dirac-proxy-init -U -M -g cta_nectarcam
# ```
#
# You can also check whether you currently have an active proxy with the command
# `dirac-proxy-info`.

# %%
# %matplotlib inline
import os
from glob import glob

from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.io import EventSeeker, EventSource
from ctapipe.visualization import CameraDisplay
from DIRAC.Interfaces.API.Dirac import Dirac

# %%
dirac = Dirac()

# %%
# ?dirac.getFile

# %%
lfns = [
    "/vo.cta.in2p3.fr/nectarcam/2022/20220411/NectarCAM.Run3169.0000.fits.fz",
    "/vo.cta.in2p3.fr/nectarcam/2022/20220411/NectarCAM.Run3169.0001.fits.fz",
]

# %%
tmpdir = f"/tmp/{os.environ['USER']}/scratch"
if not os.path.isdir(tmpdir):
    print(f"{tmpdir} does not exist yet, I will create it for you")
    os.makedirs(tmpdir)

# %%
dirac.getFile(lfn=lfns, destDir=tmpdir, printOutput=True)

# %% [markdown]
# **You are now ready to work with `ctapipe` as usual!**

# %%
path = glob(f"{tmpdir}/NectarCAM.*.fits.fz")
path.sort()
reader = EventSource(input_url=path[0])
seeker = EventSeeker(reader)

# Get some event, and display camera charges for the high gain channel
# (no time window optimization)
evt = seeker.get_event_index(10)
image = evt.r0.tel[0].waveform.sum(axis=2)
camera = CameraGeometry.from_name("NectarCam-003").transform_to(
    EngineeringCameraFrame()
)
disp = CameraDisplay(geometry=camera, image=image[0])

# %%
