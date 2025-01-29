# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to access NectarCAM data from the EGI grid directly using the DIRAC API ?
#
# In this short notebook, we will see how to access data stored on the grid using the DIRAC API, without the hassle of first manually downloading them locally.
#
# In order to achieve this, you will obviously need a `conda` environment in which all relevant code is installed, such as `ctapipe`, `nectarchain`, but also `CTADIRAC` itself. Please refer to the
#  `nectarchain` installation procedure at <https://github.com/cta-observatory/nectarchain> and follow the instructions to enable DIRAC support.
#
# You will also need to have an active proxy for EGI, initialized e.g. with:
#
# ```
# dirac-proxy-init -M -g cta_nectarcam
# ```
#
# You can also check whether you currently have an active proxy with the command `dirac-proxy-info`.

# %%
from DIRAC.Interfaces.API.Dirac import Dirac

# %%
dirac = Dirac()

# %%
# ?dirac.getFile

# %%
lfns = [
    "/vo.cta.in2p3.fr/nectarcam/2025/20250107/NectarCAM.Run5922.0000.fits.fz",
    "/vo.cta.in2p3.fr/nectarcam/2025/20250107/NectarCAM.Run5922.0001.fits.fz",
    "/vo.cta.in2p3.fr/nectarcam/2025/20250107/NectarCAM.Run5922.0002.fits.fz",
    "/vo.cta.in2p3.fr/nectarcam/2025/20250107/NectarCAM.Run5922.0003.fits.fz",
    "/vo.cta.in2p3.fr/nectarcam/2025/20250107/NectarCAM.Run5922.0004.fits.fz",
    "/vo.cta.in2p3.fr/nectarcam/2025/20250107/NectarCAM.Run5922.0005.fits.fz",
    "/vo.cta.in2p3.fr/nectarcam/2025/20250107/NectarCAM.Run5922.0006.fits.fz",
    "/vo.cta.in2p3.fr/nectarcam/2025/20250107/NectarCAM.Run5922.0007.fits.fz",
]

# %%
import os

tmpdir = f"/tmp/{os.environ['USER']}/scratch/runs"
if not os.path.isdir(tmpdir):
    print(f"{tmpdir} does not exist yet, I will create it for you")
    os.makedirs(tmpdir)

# %%
dirac.getFile(lfn=lfns, destDir=tmpdir, printOutput=True)

# %% [markdown]
# **You are now ready to work with `ctapipe` as usual!**

# %%
# %matplotlib inline
import numpy as np
from matplotlib import pyplot as plt

from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.io import EventSource, EventSeeker
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

# %%
from glob import glob

path = glob(f"{tmpdir}/NectarCAM.*.fits.fz")
path.sort()
reader = EventSource(input_url=path[0])
seeker = EventSeeker(reader)

# Get some event, and display camera charges for the high gain channel (no time window optimization)
evt = seeker.get_event_index(10)
image = evt.r0.tel[0].waveform.sum(axis=2)
camera = CameraGeometry.from_name("NectarCam-003").transform_to(
    EngineeringCameraFrame()
)
disp = CameraDisplay(geometry=camera, image=image[0])

# %%
reader

# %%

# %%
