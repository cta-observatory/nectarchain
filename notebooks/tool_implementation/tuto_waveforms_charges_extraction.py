# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: nectarchain
#     language: python
#     name: python3
# ---

# %%
import logging
import os

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers
import pathlib

import numpy as np
from ctapipe.containers import EventType
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry

from nectarchain.display import ContainerDisplay
from nectarchain.makers import (
    ChargesNectarCAMCalibrationTool,
    WaveformsNectarCAMCalibrationTool,
)
from nectarchain.makers.component import get_valid_component

# %%
get_valid_component()

# %%
run_number = 3942

# %%
tool = WaveformsNectarCAMCalibrationTool(
    progress_bar=True,
    run_number=run_number,
    max_events=500,
    log_level=20,
    overwrite=True,
    output_path=pathlib.Path(os.environ.get("NECTARCAMDATA", "/tmp"))
    / "tutorials/"
    / f"Waveforms_{run_number}.h5",
)
tool

# %%
tool.initialize()

# %%
tool.setup()

# %%
tool.start()

# %%
w_output = tool.finish(return_output_component=True)[0]
w_output

# %%
w_output.containers

# %%

# %% [markdown]
# ### Now for the charge extraction

# %%
tool = ChargesNectarCAMCalibrationTool(
    progress_bar=True,
    method="LocalPeakWindowSum",
    extractor_kwargs={"window_width": 12, "window_shift": 4},
    run_number=run_number,
    max_events=500,
    log_level=20,
    from_computed_waveforms=False,
    overwrite=True,
    output_path=pathlib.Path(os.environ.get("NECTARCAMDATA", "/tmp"))
    / "tutorials/"
    / f"Charges_{run_number}_LocalPeakWindowSum_12-4.h5",
)
tool

# %%
tool.initialize()
tool.setup()

# %%
tool.start()

# %%
c_output = tool.finish(return_output_component=True)[0]
c_output

# %%
c_output.containers

# %% [markdown]
# ### Display

# %%
geom = CameraGeometry.from_name("NectarCam-003").transform_to(EngineeringCameraFrame())

# %%
image = w_output.containers[EventType.FLATFIELD].wfs_hg.sum(axis=2)[23]
max_id = w_output.containers[EventType.FLATFIELD].pixels_id[np.argmax(image)]
max_id

# %%
disp = ContainerDisplay.display(
    w_output.containers[EventType.FLATFIELD], evt=23, geometry=geom
)
# disp.highlight_pixels(max_id,color = 'r',linewidth = 3) #to check the validity of geometry
disp.show()

# %%
ContainerDisplay.display(
    c_output.containers[EventType.FLATFIELD], evt=23, geometry=geom
)

# %%
