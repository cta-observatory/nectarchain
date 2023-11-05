import logging
import sys

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import copy

import numpy as np
from ctapipe.containers import Field,Container,partial,Map
from ctapipe.core.container import FieldValidationError


__all__ = ["ArrayDataContainer","TriggerMapContainer","get_array_keys","merge_map_ArrayDataContainer"]
def get_array_keys(container : Container) : 
        keys = []
        for field in container.fields : 
            if field.type == np.ndarray : 
                keys.append(field.key)
        return keys


class NectarCAMContainer(Container) : 
    """base class for the NectarCAM containers. This contaner cannot berecursive, to be directly written with a HDF5TableWriter """

class ArrayDataContainer(NectarCAMContainer):
    """
    A container that holds information about waveforms from a specific run.

    Attributes:
        run_number (int): The run number associated with the waveforms.
        nevents (int): The number of events.
        npixels (int): The number of pixels.
        camera (str): The name of the camera.
        pixels_id (np.ndarray): An array of pixel IDs.
        broken_pixels_hg (np.ndarray): An array of high gain broken pixels.
        broken_pixels_lg (np.ndarray): An array of low gain broken pixels.
        ucts_timestamp (np.ndarray): An array of events' UCTS timestamps.
        ucts_busy_counter (np.ndarray): An array of UCTS busy counters.
        ucts_event_counter (np.ndarray): An array of UCTS event counters.
        event_type (np.ndarray): An array of trigger event types.
        event_id (np.ndarray): An array of event IDs.
        trig_pattern_all (np.ndarray): An array of trigger patterns.
        trig_pattern (np.ndarray): An array of reduced trigger patterns.
        multiplicity (np.ndarray): An array of events' multiplicities.
    """

    run_number = Field(
        type=np.uint16,
        description="run number associated to the waveforms",
    )
    nevents = Field(
        type=np.uint64,
        description="number of events",
    )
    npixels = Field(
        type=np.uint16,
        description="number of effective pixels",
    )
    pixels_id = Field(type=np.ndarray, dtype = np.uint16, ndim = 1,description="pixel ids")
    broken_pixels_hg = Field(type=np.ndarray, dtype = bool, ndim = 2, description="high gain broken pixels")
    broken_pixels_lg = Field(type=np.ndarray, dtype = bool, ndim = 2, description="low gain broken pixels")
    camera = Field(
        type=str,
        description="camera name",
    )
    ucts_timestamp = Field(type=np.ndarray, dtype = np.uint64, ndim = 1, description="events ucts timestamp")
    ucts_busy_counter = Field(type=np.ndarray, dtype = np.uint32, ndim = 1, description="ucts busy counter")
    ucts_event_counter = Field(type=np.ndarray, dtype = np.uint32, ndim = 1, description="ucts event counter")
    event_type = Field(type=np.ndarray, dtype = np.uint8, ndim = 1, description="trigger event type")
    event_id = Field(type=np.ndarray, dtype = np.uint32, ndim = 1, description="event ids")
    trig_pattern_all = Field(type=np.ndarray, dtype = bool, ndim = 3, description="trigger pattern")
    trig_pattern = Field(type=np.ndarray, dtype = bool, ndim = 2, description="reduced trigger pattern")
    multiplicity = Field(type=np.ndarray, dtype = np.uint16, ndim = 1, description="events multiplicity")

    

class TriggerMapContainer(Container) : 
    containers = Field(default_factory=partial(Map, Container),
                       description = "trigger mapping of Container"
                       )

    def validate(self) : 
        super().validate()
        for i,container in enumerate(self.containers) : 
            if i==0 :
                container_type = type(container)
            else : 
                if not(isinstance(container,container_type)) : 
                    raise FieldValidationError("all the containers mapped must have the same type to be merged ")

def merge_map_ArrayDataContainer(triggerMapContainer : TriggerMapContainer) : 
    triggerMapContainer.validate()
    keys = list(triggerMapContainer.containers.keys())
    output_container = copy.deepcopy(triggerMapContainer.containers[keys[0]])
    for key in keys[1:] : 
        for field in get_array_keys(triggerMapContainer.containers[key]):
            output_container[field] = np.concatenate((output_container[field],triggerMapContainer.containers[key][field]),axis = 0)
        if "nevents" in output_container.fields : 
            output_container.nevents += triggerMapContainer.containers[key].nevents
    return output_container

#class TriggerMapArrayDataContainer(TriggerMapContainer):
#    containers = Field(default_factory=partial(Map, ArrayDataContainer),
#                       description = "trigger mapping of arrayDataContainer"
#                       )




