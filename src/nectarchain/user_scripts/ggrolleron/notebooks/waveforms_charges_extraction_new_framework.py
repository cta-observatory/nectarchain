# %%
from ctapipe.containers import EventType
from ctapipe.io import HDF5TableReader

from nectarchain.data.container import (
    ChargesContainer,
    ChargesContainers,
    WaveformsContainer,
    WaveformsContainers,
)
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
    progress_bar=True, run_number=run_number, max_events=500, log_level=10
)
tool

# %%
tool = ChargesNectarCAMCalibrationTool(
    progress_bar=True,
    method="LocalPeakWindowSum",
    extractor_kwargs={"window_width": 16, "window_shift": 4},
    run_number=run_number,
    max_events=5000,
    log_level=10,
)
tool

# %%
tool.initialize()

# %%
tool.setup()

# %%
tool.start()

# %%
tool.finish()

# %%
container = WaveformsContainers()
trigger_type = EventType.__members__


with HDF5TableReader(
    f"/tmp/EventsLoopNectarCAMCalibrationTool_{run_number}.h5"
) as reader:
    for key, trigger in trigger_type.items():
        try:
            tableReader = reader.read(
                table_name=f"/data/WaveformsContainer_0/{trigger.name}",
                containers=WaveformsContainer,
            )
            container.containers[trigger] = next(tableReader)
        except Exception as err:
            print(err)

# %%
container.containers

# %%
container = ChargesContainers()
trigger_type = EventType.__members__


with HDF5TableReader(
    f"/tmp/EventsLoopNectarCAMCalibrationTool_{run_number}.h5"
) as reader:
    for key, trigger in trigger_type.items():
        try:
            tableReader = reader.read(
                table_name=f"/data/ChargesContainer_0/{trigger.name}",
                containers=ChargesContainer,
            )
            container.containers[trigger] = next(tableReader)
        except Exception as err:
            print(err)

# %%
container.containers

import matplotlib.pyplot as plt
import numpy as np

# %%
from nectarchain.makers.component import ChargesComponent

# %%
counts, charge = ChargesComponent.histo_hg(container.containers[EventType.FLATFIELD])
charge.shape, counts.shape

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.errorbar(
    charge[30], counts[30], np.sqrt(counts[30]), zorder=0, fmt=".", label="data"
)

# %%
