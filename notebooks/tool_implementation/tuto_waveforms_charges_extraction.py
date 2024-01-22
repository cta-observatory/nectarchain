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
from nectarchain.makers import (
    ChargesNectarCAMCalibrationTool,
    WaveformsNectarCAMCalibrationTool,
)
from nectarchain.makers.component import get_valid_component
from nectarchain.data.container import (
    ChargesContainers,
    ChargesContainer,
    WaveformsContainer,
    WaveformsContainers,
)
from ctapipe.io import HDF5TableReader
from ctapipe.containers import EventType

# %%
get_valid_component()

# %%
run_number = 3942

# %%
tool = WaveformsNectarCAMCalibrationTool(
    progress_bar=True, run_number=run_number, max_events=500, log_level=20
)
tool

# %%
tool = ChargesNectarCAMCalibrationTool(
    progress_bar=True,
    method="LocalPeakWindowSum",
    extractor_kwargs={"window_width": 12, "window_shift": 4},
    run_number=run_number,
    max_events=500,
    log_level=20,
)
tool

# %%
tool.initialize()

# %%
tool.setup()

# %%
tool.start()

# %%
output = tool.finish(return_output_component=True)[0]
output

# %%
output.containers

# %%
