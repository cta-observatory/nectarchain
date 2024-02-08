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
import pathlib

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

from nectarchain.makers.calibration import (
    FlatFieldSPEHHVStdNectarCAMCalibrationTool,
    FlatFieldSPENominalStdNectarCAMCalibrationTool,
)

# %%
run_number = 3942

# %%
os.environ["NECTARCAMDATA"]

# %%
# !ls -lh $NECTARCAMDATA/runs/*

# %%
tool = FlatFieldSPEHHVStdNectarCAMCalibrationTool(
    progress_bar=True,
    method="LocalPeakWindowSum",
    extractor_kwargs={"window_width": 12, "window_shift": 4},
    multiproc=True,
    nproc=2,
    run_number=run_number,
    max_events=1000,
    log_level=20,
    reload_events=False,
    # events_per_slice = 200,
    overwrite=True,
    asked_pixels_id=[52, 48, 78, 94],
    output_path=pathlib.Path(os.environ.get("NECTARCAMDATA", "/tmp"))
    / "tutorials/"
    / f"SPEfit_{run_number}.h5",
)

# %%
tool

# %%
tool.initialize()

# %%
tool.setup()

# %%
tool.start()

# %%
output = tool.finish(return_output_component=True, display=True, figpath=os.getcwd())
output

# %%
output[0].resolution

# %%
