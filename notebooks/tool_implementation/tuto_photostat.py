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

# %% [markdown]
# # Tutorial for gain computation with the Photo-statitic method

# %%
import logging
import os
import pathlib

import matplotlib.pyplot as plt

from nectarchain.data.management import DataManagement
from nectarchain.makers.calibration import PhotoStatisticNectarCAMCalibrationTool
from nectarchain.makers.extractor.utils import CtapipeExtractor

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


# %%
extractor_kwargs = {"window_width": 12, "window_shift": 4}

method = "LocalPeakWindowSum"
HHV_run_number = 3942

Ped_run_number = 3938
FF_run_number = 3937

# %%
str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
    method=method, extractor_kwargs=extractor_kwargs
)
path = DataManagement.find_SPE_HHV(
    run_number=HHV_run_number,
    method=method,
    str_extractor_kwargs=str_extractor_kwargs,
)
if len(path) == 1:
    log.info(
        f"{path[0]} found associated to HHV run {HHV_run_number},"
        f"method {method} and extractor kwargs {str_extractor_kwargs}"
    )
else:
    _text = (
        f"no file found in $NECTARCAM_DATA/../SPEfit associated to HHV run"
        f"{HHV_run_number}, method {method} and extractor kwargs {str_extractor_kwargs}"
    )
    log.error(_text)
    raise FileNotFoundError(_text)

# %% [markdown]
#  WARNING : for now you can't split the event loop in slice for the Photo-statistic
# method, however in case of the charges havn't been computed on disk, the loop over
# events will only store the charge, therefore memory errors should happen rarely

# %%
tool = PhotoStatisticNectarCAMCalibrationTool(
    progress_bar=True,
    run_number=FF_run_number,
    Ped_run_number=Ped_run_number,
    SPE_result=path[0],
    method="LocalPeakWindowSum",
    extractor_kwargs={"window_width": 12, "window_shift": 4},
    max_events=10000,
    log_level=20,
    reload_events=False,
    overwrite=True,
    output_path=pathlib.Path(os.environ.get("NECTARCAMDATA", "/tmp"))
    / "tutorials/"
    / f"Photostat_FF{FF_run_number}_Ped{Ped_run_number}.h5",
)
tool

# %%
tool.initialize()

# %%
tool.setup()

# %%
tool.start()

# %%
output = tool.finish(return_output_component=True)
output

# %%
plt.plot(output[0].pixels_id, output[0].high_gain.T[0])
plt.xlabel("pixels_id")
plt.ylabel("high gain")

# %%
