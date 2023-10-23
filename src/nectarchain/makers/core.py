import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import copy
import pathlib
from abc import ABC, abstractmethod

import numpy as np
from ctapipe.containers import EventType
from ctapipe.core import Tool,Component
from ctapipe.core.traits import Bool, Integer, Path, classes_with_traits, flag, ComponentNameList
from ctapipe.instrument import CameraGeometry
from ctapipe.io import HDF5TableWriter
from ctapipe.io.datawriter import DATA_MODEL_VERSION
from ctapipe_io_nectarcam import NectarCAMEventSource, constants
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer



from tqdm.auto import tqdm

from ..data import DataManagement
from ..data.container.core import ArrayDataContainer,NectarCAMContainer,TriggerMapContainer
from .component import *

__all__ = ["EventsLoopNectarCAMCalibrationTool"]

"""The code snippet is a part of a class hierarchy for data processing. 
It includes the `BaseMaker` abstract class, the `EventsLoopMaker` and `ArrayDataMaker` subclasses. 
These classes are used to perform computations on data from a specific run."""


class BaseNectarCAMCalibrationTool(Tool):
    """Mother class for all the makers, the role of makers is to do computation on the data."""

    name = "BaseNectarCAMCalibration"

    progress_bar = Bool(
        help="show progress bar during processing", default_value=False
    ).tag(config=True)

    @staticmethod
    def load_run(
        run_number: int, max_events: int = None, run_file: str = None
    ) -> NectarCAMEventSource:
        """Static method to load from $NECTARCAMDATA directory data for specified run with max_events

        Args:self.__run_number = run_number
            run_number (int): run_id
            maxevents (int, optional): max of events to be loaded. Defaults to -1, to load everythings.
            run_file (optional) : if provided, will load this run file
        Returns:
            List[ctapipe_io_nectarcam.NectarCAMEventSource]: List of EventSource for each run files
        """
        # Load the data from the run file.
        if run_file is None:
            generic_filename, _ = DataManagement.findrun(run_number)
            log.info(f"{str(generic_filename)} will be loaded")
            eventsource = NectarCAMEventSource(
                input_url=generic_filename, max_events=max_events
            )
        else:
            log.info(f"{run_file} will be loaded")
            eventsource = NectarCAMEventSource(
                input_url=run_file, max_events=max_events
            )
        return eventsource


class EventsLoopNectarCAMCalibrationTool(BaseNectarCAMCalibrationTool):
    """
    A class for data processing and computation on events from a specific run.

    Args:
        run_number (int): The ID of the run to be processed.
        max_events (int, optional): The maximum number of events to be loaded. Defaults to None.
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
    }

    flags = {
        "overwrite": (
            {"HDF5TableWriter": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
        **flag(
            "progress",
            "ProcessorTool.progress_bar",
            "show a progress bar during event processing",
            "don't show a progress bar during event processing",
        ),
    }

    classes = (
        [
            HDF5TableWriter,
        ] 
        + classes_with_traits(NectarCAMEventSource)
        + classes_with_traits(NectarCAMComponent)

    )


    output_path = Path(
        help="output filename", default_value=pathlib.Path("/tmp/EventsLoopNectarCAMCalibrationTool.h5")
    ).tag(config=True)

    run_number = Integer(help="run number to be treated", default_value=-1).tag(
        config=True
    )

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

    componentsList = ComponentNameList(NectarCAMComponent,
        help="List of Component names to be apply, the order will be respected"
    ).tag(config=True)

    def __new__(cls,*args,**kwargs) : 
        """This method is used to pass to the current instance of Tool the traits defined 
        in the components provided in the componentsList trait. 
        WARNING : This method is maybe not the best way to do it, need to discuss with ctapipe developpers. 
        """
        _cls = super(EventsLoopNectarCAMCalibrationTool,cls).__new__(cls,*args,**kwargs)
        log.warning("the componentName in componentsList must be defined in the nectarchain.makers.component module, otherwise the import of the componentName will raise an error")
        for componentName in _cls.componentsList : 
            configurable_traits = get_configurable_traits(eval(componentName))
            _cls.add_traits(**configurable_traits)
            _cls.aliases.update({key : f"{componentName}.{key}" for key in configurable_traits.keys()})
        return _cls

    def _load_eventsource(self):
        self.event_source = self.enter_context(
            self.load_run(self.run_number, self.max_events, run_file=self.run_file)
        )

    def _get_provided_component_kwargs(self,componentName : str) : 
        component_kwargs = get_configurable_traits(eval(componentName))
        output_component_kwargs = {}
        for key in component_kwargs.keys() : 
            if hasattr(self,key) : 
                output_component_kwargs[key] = getattr(self,key)
        return output_component_kwargs

    def setup(self, *args, **kwargs):
        if self.run_number == -1:
            raise Exception("run_number need to be set up")
        self._load_eventsource()
        self.__npixels = self._event_source.camera_config.num_pixels
        self.__pixels_id = self._event_source.camera_config.expected_pixels_id

        self.components = []
        for componentName in self.componentsList : 
            if componentName in get_valid_component():
                component_kwargs = self._get_provided_component_kwargs(componentName)
                self.components.append(
                #    self.add_component(
                        Component.from_name(
                            componentName,
                            subarray = self.event_source.subarray, 
                            parent=self,
                            **component_kwargs,
                            )
                    #    )
                    )
            
        self.writer = self.enter_context(
            HDF5TableWriter(
                filename = pathlib.Path(f"{self.output_path.parent}/{self.output_path.stem}_{self.run_number}{self.output_path.suffix}"),
                parent = self,
                group_name = "data"
                )
            )

        # self.comp = MyComponent(parent=self)
        # self.comp2 = SecondaryMyComponent(parent=self)
        # self.comp3 = TelescopeWiseComponent(parent=self, subarray=subarray)
        # self.advanced = AdvancedComponent(parent=self)


    def start(
        self,
        n_events=np.inf,
        #trigger_type: list = None,
        restart_from_begining: bool = False,
        *args,
        **kwargs,
    ):
        """
        Method to extract data from the EventSource.

        Args:
            n_events (int, optional): The maximum number of events to process. Default is np.inf.
            restart_from_begining (bool, optional): Whether to restart the event source reader. Default is False.
            *args: Additional arguments that can be passed to the method.
            **kwargs: Additional keyword arguments that can be passed to the method.

        Returns:
            The output container created by the _make_output_container method.
        """
        if ~np.isfinite(n_events):
            self.log.warning(
                "no needed events number specified, it may cause a memory error"
            )
        #if isinstance(trigger_type, EventType) or trigger_type is None:
        #    trigger_type = [trigger_type]
        #for _trigger_type in trigger_type:
        #    self._init_trigger_type(_trigger_type)

        if restart_from_begining:
            self.log.debug("restart from begining : creation of the EventSource reader")
            self._load_eventsource()

        n_traited_events = 0
        for i, event in enumerate(
            tqdm(
                self._event_source,
                desc=self._event_source.__class__.__name__,
                total=min(self._event_source.max_events, n_events),
                unit="ev",
                disable=not self.progress_bar,
            )
        ):
            if i % 100 == 0:
                self.log.info(f"reading event number {i}")
            for component in self.components : 
                component(event,*args,**kwargs)
                n_traited_events += 1
            if n_traited_events >= n_events:
                break

    def finish(self, *args, **kwargs):
        # self.write = self.enter_context(
        #    HDF5TableWriter(filename=filename, parent=self)
        # )
        output = []
        for component in self.components : 
            output.append(component.finish(*args,**kwargs))
        log.info(output)
        for _output in output : 
            if isinstance(_output,NectarCAMContainer) : 
                self.writer.write(table_name = str(_output.__class__.__name__),
                                  containers = _output,
                )
            elif isinstance(_output,TriggerMapContainer) : 
                for i,key in enumerate(_output.containers.keys()) : 
                    self.writer.write(table_name = f"{_output.containers[key].__class__.__name__}_{i}/{key.name}",
                                      containers = _output.containers[key],
                    )
            else : 
                raise TypeError("component output must be an instance of TriggerMapContainer or NectarCAMContainer")

        self.writer.close()
        super().finish()
        self.log.warning("Shutting down.")

    @property
    def event_source(self):
        """
        Getter method for the _event_source attribute.
        """
        return copy.copy(self._event_source)

    @event_source.setter
    def event_source(self, value):
        """
        Setter method to set a new NectarCAMEventSource to the _reader attribute.

        Args:
            value: a NectarCAMEventSource instance.
        """
        if isinstance(value, NectarCAMEventSource):
            self._event_source = value
        else:
            raise TypeError("The reader must be a NectarCAMEventSource")

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


def main():
    """run the tool"""
    tool = EventsLoopNectarCAMCalibrationTool()
    tool.run()


if __name__ == "__main__":
    main()