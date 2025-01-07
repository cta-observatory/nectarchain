import copy
import logging
import os
import pathlib
from datetime import datetime

import numpy as np
from ctapipe.containers import Container, EventType
from ctapipe.core import Component, Tool
from ctapipe.core.container import FieldValidationError
from ctapipe.core.traits import (
    Bool,
    ComponentNameList,
    Integer,
    Path,
    classes_with_traits,
    flag,
)
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from ctapipe.io import HDF5TableWriter
from ctapipe.io.datawriter import DATA_MODEL_VERSION
from ctapipe_io_nectarcam import LightNectarCAMEventSource
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from tables.exceptions import HDF5ExtError
from tqdm.auto import tqdm
from traitlets import default

from ..data import DataManagement
from ctapipe_io_nectarcam import LightNectarCAMEventSource
from ..data.container.core import NectarCAMContainer, TriggerMapContainer
from ..utils import ComponentUtils
from .component import NectarCAMComponent, get_valid_component

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

__all__ = [
    "EventsLoopNectarCAMCalibrationTool",
    "DelimiterLoopNectarCAMCalibrationTool",
]

"""The code snippet is a part of a class hierarchy for data processing.
It includes the `BaseMaker` abstract class, the `EventsLoopMaker` and `ArrayDataMaker`
subclasses.
These classes are used to perform computations on data from a specific run."""


class BaseNectarCAMCalibrationTool(Tool):
    """Mother class for all the makers, the role of makers is to do computation on the
    data."""

    name = "BaseNectarCAMCalibration"

    progress_bar = Bool(
        help="show progress bar during processing", default_value=False
    ).tag(config=True)

    @default("provenance_log")
    def _default_provenance_log(self):
        return (
            f"{os.environ.get('NECTARCHAIN_LOG', '/tmp')}/"
            f"{self.name}_{os.getpid()}_{datetime.now()}.provenance.log"
        )

    @default("log_file")
    def _default_log_file(self):
        return (
            f"{os.environ.get('NECTARCHAIN_LOG', '/tmp')}/"
            f"{self.name}_{os.getpid()}_{datetime.now()}.log"
        )

    @staticmethod
    def load_run(
        run_number: int, max_events: int = None, run_file: str = None
    ) -> LightNectarCAMEventSource:
        """Static method to load from $NECTARCAMDATA directory data for specified run
        with max_events.

        Parameters
        ----------
        run_number : int
            run_id
        maxevents : int, optional
            max of events to be loaded. Defaults to -1, to load everything.
        run_file : optional
            if provided, will load this run file

        Returns
        -------
        List[ctapipe_io_nectarcam.LightNectarCAMEventSource]
            List of EventSource for each run files.
        """
        # Load the data from the run file.
        if run_file is None:
            generic_filename, _ = DataManagement.findrun(run_number)
            log.info(f"{str(generic_filename)} will be loaded")
            eventsource = LightNectarCAMEventSource(
                input_url=generic_filename, max_events=max_events
            )
        else:
            log.info(f"{run_file} will be loaded")
            eventsource = LightNectarCAMEventSource(
                input_url=run_file, max_events=max_events
            )
        return eventsource


class EventsLoopNectarCAMCalibrationTool(BaseNectarCAMCalibrationTool):
    """
    A class for data processing and computation on events from a specific run.

    Args:
        run_number (int): The ID of the run to be processed.
        max_events (int, optional): The maximum number of events to be loaded.
        Defaults to None.
        run_file (optional): The specific run file to be loaded.

    Example Usage:
        maker = EventsLoopMaker(run_number=1234, max_events=1000)
        maker.make(n_events=500)


    """

    name = "EventsLoopNectarCAMCalibration"

    description = (
        __doc__ + f" This currently uses data model version {DATA_MODEL_VERSION}"
    )
    examples = """To be implemented"""

    aliases = {
        ("i", "input"): "EventsLoopNectarCAMCalibrationTool.run_file",
        ("r", "run-number"): "EventsLoopNectarCAMCalibrationTool.run_number",
        ("m", "max-events"): "EventsLoopNectarCAMCalibrationTool.max_events",
        ("o", "output"): "EventsLoopNectarCAMCalibrationTool.output_path",
        "events-per-slice": "EventsLoopNectarCAMCalibrationTool.events_per_slice",
    }

    flags = {
        "overwrite": (
            {"HDF5TableWriter": {"overwrite": False}},
            "Overwrite output file if it exists",
        ),
        **flag(
            "progress",
            "EventsLoopNectarCAMCalibrationTool.progress_bar",
            "show a progress bar during event processing",
            "don't show a progress bar during event processing",
        ),
    }

    classes = (
        [
            HDF5TableWriter,
        ]
        + classes_with_traits(LightNectarCAMEventSource)
        + classes_with_traits(NectarCAMComponent)
    )

    run_number = Integer(help="run number to be treated", default_value=-1).tag(
        config=True
    )

    output_path = Path(
        help="output filename",
        default_value=pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/runs/"
            f"{name}_run{run_number.default_value}.h5"
        ),
    ).tag(config=True)

    max_events = Integer(
        help="maximum number of events to be loaded",
        default_value=None,
        allow_none=True,
    ).tag(config=True)

    run_file = Path(
        help="file name to be loaded",
        default_value=None,
        allow_none=True,
    ).tag(config=True)

    componentsList = ComponentNameList(
        NectarCAMComponent,
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    events_per_slice = Integer(
        help="number of events that will be treat before to pull the buffer and write"
        "to disk, if None, all the events will be loaded",
        default_value=None,
        allow_none=True,
    ).tag(config=True)

    def __new__(cls, *args, **kwargs):
        """This method is used to pass to the current instance of Tool the traits
        defined in the components provided in the componentsList trait.
        WARNING : This method is maybe not the best way to do it, need to discuss with
        ctapipe developers.
        """
        _cls = super(EventsLoopNectarCAMCalibrationTool, cls).__new__(
            cls, *args, **kwargs
        )
        log.warning(
            "the componentName in componentsList must be defined in the "
            "nectarchain.makers.component module, otherwise the import of the "
            "componentName will raise an error"
        )
        for componentName in _cls.componentsList:
            class_name = ComponentUtils.get_class_name_from_ComponentName(componentName)
            configurable_traits = ComponentUtils.get_configurable_traits(class_name)
            _cls.add_traits(**configurable_traits)
            _cls.aliases.update(
                {key: f"{componentName}.{key}" for key in configurable_traits.keys()}
            )
        return _cls

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not ("output_path" in kwargs.keys()):
            self._init_output_path()

    def _init_output_path(self):
        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/"
            f"runs/{self.name}_run{self.run_number}.h5"
        )

    def _load_eventsource(self, *args, **kwargs):
        self.log.debug("loading event source")
        self.event_source = self.enter_context(
            self.load_run(self.run_number, self.max_events, run_file=self.run_file)
        )

    def _get_provided_component_kwargs(self, componentName: str):
        class_name = ComponentUtils.get_class_name_from_ComponentName(componentName)
        component_kwargs = ComponentUtils.get_configurable_traits(class_name)
        output_component_kwargs = {}
        for key in component_kwargs.keys():
            if hasattr(self, key):
                output_component_kwargs[key] = getattr(self, key)
        return output_component_kwargs

    def _init_writer(self, sliced: bool = False, slice_index: int = 0, group_name=None):
        if hasattr(self, "writer"):
            self.writer.close()

        if sliced:
            if group_name is None:
                if slice_index == 0:
                    if self.overwrite:
                        try:
                            log.info(
                                f"overwrite set to true, trying to "
                                f"remove file {self.output_path}"
                            )
                            os.remove(self.output_path)
                            log.info(f"{self.output_path} removed")
                        except OSError:
                            pass
                    else:
                        if os.path.exists(self.output_path):
                            raise Exception(
                                f"file {self.output_path} does exist,\n set overwrite "
                                f"to True if you want to overwrite"
                            )
                group_name = f"data_{slice_index}"
            self.log.info(
                f"initialization of writer in sliced mode (output written "
                f"to {group_name})"
            )
            mode = "a"
        else:
            self.log.info("initialization of writer in full mode")
            if self.overwrite:
                try:
                    log.info(
                        f"overwrite set to true, trying to remove "
                        f"file {self.output_path}"
                    )
                    os.remove(self.output_path)
                    log.info(f"{self.output_path} removed")
                except OSError:
                    pass
            else:
                if os.path.exists(self.output_path):
                    raise Exception(
                        f"file {self.output_path} does exist,\n set overwrite to True "
                        f"if you want to overwrite"
                    )
            if group_name is None:
                group_name = "data"
            mode = "w"
        try:
            os.makedirs(self.output_path.parent, exist_ok=True)
            self.writer = self.enter_context(
                HDF5TableWriter(
                    filename=self.output_path,
                    parent=self,
                    mode=mode,
                    group_name=group_name,
                )
            )
        except HDF5ExtError as err:
            self.log.warning(err.args[0], exc_info=True)
            self.log.warning("retry with w mode instead of a")
            self.writer = self.enter_context(
                HDF5TableWriter(
                    filename=self.output_path,
                    parent=self,
                    mode="w",
                    group_name=group_name,
                )
            )
        except Exception as err:
            self.log.error(err, exc_info=True)
            raise err

    def setup(self, *args, **kwargs):
        self.log.info("setup of the Tool")
        if self.run_number == -1:
            raise Exception("run_number need to be set up")
        self._setup_eventsource(*args, **kwargs)

        self._setup_components(*args, **kwargs)

        if self.output_path.exists() and self.overwrite:
            os.remove(self.output_path)

        self._init_writer(sliced=not (self.events_per_slice is None), slice_index=1)

        self._n_traited_events = 0

        # self.comp = MyComponent(parent=self)
        # self.comp2 = SecondaryMyComponent(parent=self)
        # self.comp3 = TelescopeWiseComponent(parent=self, subarray=subarray)
        # self.advanced = AdvancedComponent(parent=self)

    def _setup_eventsource(self, *args, **kwargs):
        self._load_eventsource(*args, **kwargs)
        self.__npixels = self._event_source.camera_config.num_pixels
        self.__pixels_id = self._event_source.nectarcam_service.pixel_ids

    def _setup_components(self, *args, **kwargs):
        self.log.info("setup of components")
        self.components = []
        for componentName in self.componentsList:
            if componentName in get_valid_component():
                component_kwargs = self._get_provided_component_kwargs(componentName)
                self.components.append(
                    #    self.add_component(
                    Component.from_name(
                        componentName,
                        subarray=self.event_source.subarray,
                        parent=self,
                        **component_kwargs,
                    )
                    #    )
                )

    def start(
        self,
        n_events=np.inf,
        # trigger_type: list = None,
        restart_from_begining: bool = False,
        *args,
        **kwargs,
    ):
        """
        Method to extract data from the EventSource.

        Parameters
        ----------
        n_events: int, optional
            The maximum number of events to process. Default is np.inf.
        restart_from_begining: bool, optional
            Whether to restart the event source reader. Default is False.
        args
            Additional arguments that can be passed to the method.
        kwargs
            Additional keyword arguments that can be passed to the method.

        Returns
        -------
        The output container created by the _make_output_container method.
        """
        if ~np.isfinite(n_events) and (self.events_per_slice is None):
            self.log.warning(
                "neither needed events number specified or events per slice, it may "
                "cause a memory error"
            )
        # if isinstance(trigger_type, EventType) or trigger_type is None:
        #    trigger_type = [trigger_type]
        # for _trigger_type in trigger_type:
        #    self._init_trigger_type(_trigger_type)

        if restart_from_begining:
            self.log.debug(
                "restart from beginning : creation of the EventSource " "reader"
            )
            self._load_eventsource()

        n_events_in_slice = 0
        slice_index = 1
        for i, event in enumerate(
            tqdm(
                self._event_source,
                desc=self._event_source.__class__.__name__,
                total=(
                    len(self._event_source)
                    if self._event_source.max_events is None
                    else int(np.min((self._event_source.max_events, n_events)))
                ),
                unit="ev",
                disable=not self.progress_bar,
            )
        ):
            # if i % 100 == 0:
            #    self.log.info(f"reading event number {i}")
            for component in self.components:
                component(event, *args, **kwargs)
                self._n_traited_events += 1
                n_events_in_slice += 1
            if self._n_traited_events >= n_events:
                break

            if self.split_run(n_events_in_slice, event):
                self.log.info(f"slice number {slice_index} is full, pulling buffer")
                self._finish_components(*args, **kwargs)
                self.writer.close()
                slice_index += 1
                self._init_writer(sliced=True, slice_index=slice_index)
                self._setup_components()
                n_events_in_slice = 0

    def split_run(self, n_events_in_slice : int = None, event : NectarCAMDataContainer = None):
        """Method to decide if criteria to end a run slice are met"""
        condition = (
            self.events_per_slice is not None
            and n_events_in_slice >= self.events_per_slice
        )
        return condition

    def finish(self, return_output_component=False, *args, **kwargs):
        self.log.info("finishing Tool")

        output = self._finish_components(*args, **kwargs)

        self.writer.close()
        super().finish()
        self.log.warning("Shutting down.")
        if return_output_component:
            return output

    def _finish_components(self, *args, **kwargs):
        self.log.info("finishing components and writing to output file")
        output = []
        for component in self.components:
            output.append(component.finish(*args, **kwargs))
        log.info(output)
        for i, _output in enumerate(output):
            if not (_output is None):
                self._write_container(_output, i)
        return output

    def _write_container(self, container: Container, index_component: int = 0) -> None:
        try:
            container.validate()
            if isinstance(container, NectarCAMContainer):
                self.writer.write(
                    table_name=f"{container.__class__.__name__}_{index_component}",
                    containers=container,
                )
            elif isinstance(container, TriggerMapContainer):
                for key in container.containers.keys():
                    self.writer.write(
                        table_name=f"{container.containers[key].__class__.__name__}_"
                        f"{index_component}/{key.name}",
                        containers=container.containers[key],
                    )
            else:
                raise TypeError(
                    "component output must be an instance of TriggerMapContainer or "
                    "NectarCAMContainer"
                )
        except FieldValidationError as e:
            log.warning(e, exc_info=True)
            self.log.warning("the container has not been written")
        except Exception as e:
            log.error(e, exc_info=True)
            self.log.error(e.args[0], exc_info=True)
            raise e

    @property
    def event_source(self):
        """
        Getter method for the _event_source attribute.
        """
        return copy.copy(self._event_source)

    @event_source.setter
    def event_source(self, value):
        """
        Setter method to set a new LightNectarCAMEventSource to the _reader attribute.

        Args:
            value: a LightNectarCAMEventSource instance.
        """
        if isinstance(value, LightNectarCAMEventSource):
            self._event_source = value
        else:
            raise TypeError("The reader must be a LightNectarCAMEventSource")

    @property
    def _npixels(self):
        """
        Getter method for the _npixels attribute.
        """
        return self.__npixels

    @property
    def _pixels_id(self):
        """
        Getter method for the _pixels_id attribute.
        """
        return self.__pixels_id

    @property
    def npixels(self):
        """
        Getter method for the npixels attribute.
        """
        return copy.deepcopy(self.__npixels)

    @property
    def pixels_id(self):
        """
        Getter method for the pixels_id attribute.
        """
        return copy.deepcopy(self.__pixels_id)


class DelimiterLoopNectarCAMCalibrationTool(EventsLoopNectarCAMCalibrationTool):
    """
    Class that will split data based on the EventType UNKNOWN.
    Each time this particular type is seen, it will trigger the change of slice.
    Note that the UNKONWN event will be seen by the component, so it must be filtered
    there.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def split_run(self, n_events_in_slice : int = None, event : NectarCAMDataContainer = None):
        """Method to decide if criteria to end a run slice is met"""
        condition = event.trigger.event_type == EventType.UNKNOWN
        return condition


def main():
    """run the tool"""
    tool = EventsLoopNectarCAMCalibrationTool()
    tool.run()


if __name__ == "__main__":
    main()
