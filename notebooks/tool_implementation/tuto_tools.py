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
# # Tutorial to use ctapipe tools and component

# %% [markdown]
# ## Context

# %% [markdown]
# Tool and Component are 2 modules of ctapipe, Tool is a high level module to analyse
# raw data (fits.fz files).
# This module use Component to perform computation on the raw data. Basically, we can
# create a class (MyTool) which inherits of Tool, where we can define two Component
# (Comp_A and Comp_B). Thus, with an instance of MyTool, we can loop over event within
# raw data, and for each event apply successively Comp_A, Comp_B.
#
# A ctapipe tutorial is accessible here:
# https://ctapipe.readthedocs.io/en/stable/auto_examples/core/command_line_tools.html
# #sphx-glr-auto-examples-core-command-line-tools-py
#
# You can find documentation of ctapipe Tool and Component:
#
# https://ctapipe.readthedocs.io/en/stable/api-reference/tools/index.html
#
#
# https://ctapipe.readthedocs.io/en/stable/api/ctapipe.core.Component.html
#
# Within nectarchain, we implemented within the nectarchain.makers module both a top
# level Tool and Component from which all the nectarchain Component and Tool should
# inherit.
#
# In this tutorial, we explain quickly how we can use Tool and Component to develop the
# nectarchain software, as an example there is the implementation of a PedestalTool
# which extract pedestal

# %%
# ### Imports

import os
import pathlib

import matplotlib.pyplot as plt

# %%
import numpy as np
from ctapipe.containers import Field
from ctapipe.core.traits import ComponentNameList, Integer
from ctapipe.io import HDF5TableReader
from ctapipe_io_nectarcam import constants
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer

from nectarchain.data.container import NectarCAMContainer
from nectarchain.makers import EventsLoopNectarCAMCalibrationTool
from nectarchain.makers.component import NectarCAMComponent

# %%
tools = EventsLoopNectarCAMCalibrationTool()
tools

# %%
tools.classes


# %% [markdown]
# The only thing to add is to fill the componentList field, which contains
# the names of the component to be applied on events.
#
# Then we will define a very simple component to compute the pedestal of each event.

# %% [markdown]
# ### Definition of container to store extracted data on disk


# %%
class MyContainer(NectarCAMContainer):
    run_number = Field(
        type=np.uint16,
        description="run number associated to the waveforms",
    )
    npixels = Field(
        type=np.uint16,
        description="number of effective pixels",
    )
    pixels_id = Field(type=np.ndarray, dtype=np.uint16, ndim=1, description="pixel ids")
    ucts_timestamp = Field(
        type=np.ndarray, dtype=np.uint64, ndim=1, description="events ucts timestamp"
    )
    event_type = Field(
        type=np.ndarray, dtype=np.uint8, ndim=1, description="trigger event type"
    )
    event_id = Field(type=np.ndarray, dtype=np.uint32, ndim=1, description="event ids")

    pedestal_hg = Field(
        type=np.ndarray, dtype=np.uint16, ndim=2, description="The high gain pedestal"
    )
    pedestal_lg = Field(
        type=np.ndarray, dtype=np.uint16, ndim=2, description="The low gain pedestal"
    )


# %% [markdown]
# ### Definition of our Component


# %%
class MyComp(NectarCAMComponent):
    window_shift = Integer(
        default_value=4,
        help="the time in ns before the peak to extract charge",
    ).tag(config=True)

    window_width = Integer(
        default_value=12,
        help="the duration of the extraction window in ns",
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, *args, **kwargs):
        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **kwargs
        )
        # If you want you can add here members of MyComp, they will contain interesting
        # quantity during the event loop process

        self.__ucts_timestamp = []
        self.__event_type = []
        self.__event_id = []

        self.__pedestal_hg = []
        self.__pedestal_lg = []

    # This method need to be defined !
    def __call__(self, event: NectarCAMDataContainer, *args, **kwargs):
        self.__event_id.append(np.uint32(event.index.event_id))
        self.__event_type.append(event.trigger.event_type.value)
        self.__ucts_timestamp.append(
            event.nectarcam.tel[self.tel_id].evt.ucts_timestamp
        )

        wfs = []
        wfs.append(
            event.r0.tel[self.tel_id].waveform[constants.HIGH_GAIN][self.pixels_id]
        )
        wfs.append(
            event.r0.tel[self.tel_id].waveform[constants.LOW_GAIN][self.pixels_id]
        )

        # The main work is here !
        for i, pedestal in enumerate([self.__pedestal_hg, self.__pedestal_lg]):
            index_peak = np.argmax(wfs[i])
            signal_start = index_peak - self.window_shift
            signal_stop = index_peak + self.window_width - self.window_shift
            if signal_start < 0:
                signal_stop = self.window_width
                signal_start = 0
            if signal_stop > constants.N_SAMPLES:
                signal_stop = constants.N_SAMPLES
                signal_start = constants.N_SAMPLES - self.window_width
            pedestal.append(
                (
                    np.sum(wfs[i][:, 0:signal_start], axis=1)
                    + np.sum(wfs[i][:, signal_stop:], axis=1)
                )
                / (constants.N_SAMPLES - self.window_width)
            )

    # This method need to be defined !
    def finish(self):
        output = MyContainer(
            run_number=MyContainer.fields["run_number"].type(self._run_number),
            npixels=MyContainer.fields["npixels"].type(self._npixels),
            pixels_id=MyContainer.fields["pixels_id"].dtype.type(self._pixels_id),
            ucts_timestamp=MyContainer.fields["ucts_timestamp"].dtype.type(
                self.__ucts_timestamp
            ),
            event_type=MyContainer.fields["event_type"].dtype.type(self.__event_type),
            event_id=MyContainer.fields["event_id"].dtype.type(self.__event_id),
            pedestal_hg=MyContainer.fields["pedestal_hg"].dtype.type(
                self.__pedestal_hg
            ),
            pedestal_lg=MyContainer.fields["pedestal_lg"].dtype.type(
                self.__pedestal_lg
            ),
        )
        return output


# %% [markdown]
# ### Definition of our Tool

# %% [markdown]
# Now we can define out Tool, we have just to add our component "MyComp"
# in the ComponentList:


# %%
def get_valid_component():
    return NectarCAMComponent.non_abstract_subclasses()


class MyTool(EventsLoopNectarCAMCalibrationTool):
    name = "PedestalTutoNectarCAM"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["MyComp"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def _init_output_path(self):
        if self.max_events is None:
            filename = f"{self.name}_run{self.run_number}.h5"
        else:
            filename = f"{self.name}_run{self.run_number}_maxevents{self.max_events}.h5"
        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/tutorials/{filename}"
        )


# %%
tool = MyTool(
    progress_bar=True,
    run_number=4943,
    max_events=500,
    log_level=20,
    window_width=14,
    overwrite=True,
)

# %%
tool.componentsList

# %%
tool

# %% [markdown]
# First we have to initialize the tool :

# %%
tool.initialize()

# %% [markdown]
# Then to setup, it will in particular setup the Components :

# %%
tool.setup()

# %% [markdown]
# The following command will just start the tool and apply components
# looping over events.

# %%
tool.start()

# %% [markdown]
# Then, we finish the tool. Behind this command the component will be finalized and will
# create an output container which will be written on disk and can be returned

# %%
output = tool.finish(return_output_component=True)[0]

# %%
output

# %% [markdown]
# The following file has been written :

# %%
# !ls -lh $NECTARCAMDATA/tutorials

# %% [markdown]
# The shape of pedestal is (n_events,n_pixels)

# %%
output.pedestal_hg.shape

# %% [markdown]
# To have a look to a random pixel pedestal evolution :

# %%
fix, ax = plt.subplots(1, 1)
i = np.random.randint(0, len(output.pixels_id))

ax.plot(
    (output.ucts_timestamp - output.ucts_timestamp[0]) / 1e6,
    output.pedestal_hg[:, i],
    linewidth=0.5,
)
ax.set_ylabel("Charge [ADC]")
ax.set_xlabel("time [ms]")

# %% [markdown]
# If you want to load container thereafter :

# %%
container_loaded = next(
    MyContainer._container_from_hdf5(
        f"{os.environ.get('NECTARCAMDATA','/tmp')}/tutorials/PedestalTutoNectarCAM_"
        f"run4943_maxevents500.h5",
        MyContainer,
    )
)
container_loaded.validate()
container_loaded

# %% [markdown]
# ## Going further

# %% [markdown]
# An argument that are implemented in EventsLoopNectarCAMCalibrationTool is
# `event_per_slice`. This argument allows to split all the events within the raw
# data fits.fz file in slices. It allows, for each slice, to loop over events and
# write container on disk. This mechanism allows to save RAM.
# The resulting HDF5 file that is written on disk can be easily loaded thereafter.
# There is only one HDF5 file for the whole run, which is a mapping between slices
# and containers filled by computed quantity from components.

# %%
tool = MyTool(
    progress_bar=True,
    run_number=4943,
    max_events=2000,
    log_level=20,
    events_per_slice=1000,
    overwrite=True,
)
tool

# %%
tool.initialize()
tool.setup()

# %%
tool.start()

# %%
output = tool.finish(return_output_component=True)[0]
output


# %%
# !h5ls -r $NECTARCAMDATA/tutorials/PedestalTutoNectarCAM_run4943_maxevents2000.h5

# %%
# container_loaded = ArrayDataContainer._container_from_hdf5(
#     f"{os.environ.get('NECTARCAMDATA','/tmp')}/tutorials/PedestalTutoNectarCAM_"
#     f"run4943_maxevents2000.h5",MyContainer)
# container_loaded


# %%
def read_hdf5_sliced(path):
    container = MyContainer()
    container_class = MyContainer
    with HDF5TableReader(path) as reader:
        for data in reader._h5file.root.__members__:
            # print(data)
            data_cont = eval(f"reader._h5file.root.{data}.__members__")[0]
            # print(data_cont)
            tableReader = reader.read(
                table_name=f"/{data}/{data_cont}", containers=container_class
            )
            # container.containers[data].containers[trigger] = next(tableReader)
            container = next(tableReader)
            yield container


# %%
container_loaded = read_hdf5_sliced(
    f"{os.environ.get('NECTARCAMDATA','/tmp')}/tutorials/PedestalTutoNectarCAM_"
    f"run4943_maxevents2000.h5"
)
for i, container in enumerate(container_loaded):
    print(
        f"Container {i} is filled by events from {container.event_id[0]} to"
        f"{container.event_id[-1]}"
    )

# %%
