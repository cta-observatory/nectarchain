# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: nectar-dev
#     language: python
#     name: nectar-dev
# ---

# %% [markdown]
# # How to access NectarCAM data from the EGI grid directly using through DIRAC within `nectarchain` ?
#
# In this short notebook, we will see how to access data stored on the grid through DIRAC within `nectarchain` itself, without the hassle of first manually downloading them locally.
#
# In order to achieve this, you will obviously need a `conda` environment in which all relevant code is installed, such as `ctapipe`, `nectarchain`, but also `CTADIRAC` itself. Please refer to the `nectarchain` installation procedure at <https://github.com/cta-observatory/nectarchain> and follow the instructions to enable DIRAC support.
#
# You will also need to have an active proxy for EGI, initialized e.g. with:
#
# ```
# dirac-proxy-init -M -g ctao_nectarcam
# ```
#
# You can also check whether you currently have an active proxy with the command `dirac-proxy-info`.

# %%
from nectarchain.data import DataManagement

# %%
dm = DataManagement()
dm.findrun(6881)

# %% [markdown]
# Once the files are downloaded, the same command will *not* fetch them again, but will detect that the data are already locally present:

# %%
dm.findrun(6881)

# %% [markdown]
# It is also possible to fetch data using the DIRAC API directly. Under the hood, this is exactly what is done within `nectarchain` in the example above.

# %%
from DIRAC.Interfaces.API.Dirac import Dirac

# %%
dirac = Dirac()

# %%
# ?dirac.getFile

# %%
lfns = [
    "/ctao/nectarcam/NectarCAMQM/2025/20250722/NectarCAM.Run6881.0000.fits.fz",
    "/ctao/nectarcam/NectarCAMQM/2025/20250722/NectarCAM.Run6881.0001.fits.fz",
]

# %%
import os

tmpdir = f"{os.environ['NECTARCAMDATA']}/runs"
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
from traitlets.config import Config

from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.io import EventSource, EventSeeker
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

# %%
from glob import glob

path = glob(f"{tmpdir}/NectarCAM.*.fits.fz")
path.sort()

config = Config(
    dict(
        NectarCAMEventSource=dict(
            NectarCAMR0Corrections=dict(
                calibration_path=None,
                apply_flatfield=False,
                select_gain=False,
            )
        )
    )
)

reader = EventSource(input_url=path[0], config=config, max_events=100)

tel_id = reader.subarray.tel_ids[0]

# Get some event, and display camera charges for the high gain channel (no time window optimization)
evt = next(iter(reader))
image = evt.r0.tel[tel_id].waveform.sum(axis=2)
camera = reader.subarray.tel[tel_id].camera.geometry.transform_to(
    EngineeringCameraFrame()
)
disp = CameraDisplay(geometry=camera, image=image[0])

# %%
