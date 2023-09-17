import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from abc import ABC, abstractmethod

from ctapipe_io_nectarcam import NectarCAMEventSource
import numpy as np
import copy

from ctapipe.containers import EventType
from ctapipe.instrument import CameraGeometry
from ctapipe_io_nectarcam import constants
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer



from ..data import DataManagement
from ..data.container.core import ArrayDataContainer

__all__ = ["ArrayDataMaker"]

"""The code snippet is a part of a class hierarchy for data processing. 
It includes the `BaseMaker` abstract class, the `EventsLoopMaker` and `ArrayDataMaker` subclasses. 
These classes are used to perform computations on data from a specific run."""

class BaseMaker(ABC):
    """Mother class for all the makers, the role of makers is to do computation on the data. 
    """
    @abstractmethod
    def make(self, *args, **kwargs):
        """
        Abstract method that needs to be implemented by subclasses.
        This method is the main one, which computes and does the work. 
        """
        pass
    @staticmethod
    def load_run(run_number : int,max_events : int = None, run_file = None) -> NectarCAMEventSource: 
        """Static method to load from $NECTARCAMDATA directory data for specified run with max_events

        Args:self.__run_number = run_number
            run_number (int): run_id
            maxevents (int, optional): max of events to be loaded. Defaults to 0, to load everythings.
            run_file (optional) : if provided, will load this run file
        Returns:
            List[ctapipe_io_nectarcam.NectarCAMEventSource]: List of EventSource for each run files
        """
        # Load the data from the run file.
        if run_file is None : 
            generic_filename,_ = DataManagement.findrun(run_number)
            log.info(f"{str(generic_filename)} will be loaded")
            eventsource = NectarCAMEventSource(input_url=generic_filename,max_events=max_events)
        else :  
            log.info(f"{run_file} will be loaded")
            eventsource = NectarCAMEventSource(input_url=run_file,max_events=max_events)
        return eventsource


class EventsLoopMaker(BaseMaker):
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

    def __init__(self, run_number: int, max_events: int = None, run_file=None, *args, **kwargs):
        """
        Constructor method that initializes the EventsLoopMaker object.

        Args:
            run_number (int): The ID of the run to be loaded.
            max_events (int, optional): The maximum number of events to be loaded. Defaults to None.
            run_file (optional): The specific run file to be loaded.
        """
        super().__init__(*args, **kwargs)

        self.__run_number = run_number
        self.__run_file = run_file
        self.__max_events = max_events

        self.__reader = __class__.load_run(run_number, max_events, run_file=run_file)

        # from reader members
        self.__npixels = self.__reader.camera_config.num_pixels
        self.__pixels_id = self.__reader.camera_config.expected_pixels_id

        log.info(f"N pixels : {self.npixels}")

    def make(self, n_events=np.inf, restart_from_beginning : bool =False, *args, **kwargs):
        """
        Method to iterate over the events and perform computations on each event.

        Args:
            n_events (int, optional): The number of events to process. Defaults to np.inf.
            restart_from_beginning (bool, optional): Whether to restart from the beginning of the run. Defaults to False.
        """
        if restart_from_beginning:
            log.debug('restart from beginning : creation of the EventSource reader')
            self.__reader = __class__.load_run(self.__run_number, self.__max_events, run_file=self.__run_file)

        n_traited_events = 0
        for i, event in enumerate(self.__reader):
            if i % 100 == 0:
                log.info(f"reading event number {i}")
            self._make_event(event, *args, **kwargs)
            n_traited_events += 1
            if n_traited_events >= n_events:
                break

    @abstractmethod
    def _make_event(self, event: NectarCAMDataContainer):
        """
        Abstract method that needs to be implemented by subclasses.
        This method is called for each event in the run to perform computations on the event.

        Args:
            event (NectarCAMDataContainer): The event to perform computations on.
        """
        pass

    @property
    def _run_file(self):
        """
        Getter method for the _run_file attribute.
        """
        return self.__run_file

    @property
    def _max_events(self):
        """
        Getter method for the _max_events attribute.
        """
        return self.__max_events

    @property
    def _reader(self):
        """
        Getter method for the _reader attribute.
        """
        return self.__reader

    @_reader.setter
    def _reader(self, value):
        """
        Setter method to set a new NectarCAMEventSource to the _reader attribute.

        Args:
            value: a NectarCAMEventSource instance.
        """
        if isinstance(value, NectarCAMEventSource):
            self.__reader = value
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
    def _run_number(self):
        """
        Getter method for the _run_number attribute.
        """
        return self.__run_number

    @property
    def reader(self):
        """
        Getter method for the reader attribute.
        """
        return copy.deepcopy(self.__reader)

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

    @property
    def run_number(self):
        """
        Getter method for the run_number attribute.
        """
        return copy.deepcopy(self.__run_number)

    

class ArrayDataMaker(EventsLoopMaker) : 
    """
    Class used to loop over the events of a run and to extract informations that are stored in arrays. 
    Example Usage:
    - Create an instance of the ArrayDataMaker class
    maker = ArrayDataMaker(run_number=1234, max_events=1000)

    - Perform data processing on the specified run
    maker.make(n_events=500, trigger_type=[EventType.SKY])

    - Access the computed data
    ucts_timestamp = maker.ucts_timestamp(EventType.SKY)
    event_type = maker.event_type(EventType.SKY)

    Inputs:
    - run_number (int): The ID of the run to be processed.
    - max_events (int, optional): The maximum number of events to be loaded. Defaults to None, which loads all events.
    - run_file (optional): The specific run file to be loaded.

    Flow:
    1. The ArrayDataMaker class is initialized with the run number, maximum events, or a run file.
    2. The make method is called to perform data processing on the specified run.
    3. The _make_event method is called for each event in the run to extract and store relevant data.
    4. The computed data is stored in instance variables.
    5. The computed data can be accessed using getter methods.

    Outputs:
    - Computed data such as UCTS timestamps, event types, and event IDs can be accessed using getter methods provided by the ArrayDataMaker class.
    """

    TEL_ID = 0
    CAMERA_NAME = "NectarCam-003"
    CAMERA = CameraGeometry.from_name(CAMERA_NAME)
    def __init__(self,run_number : int,max_events : int = None,run_file = None,*args,**kwargs):
        """construtor

        Args:
            run_number (int): id of the run to be loaded
            maxevents (int, optional): max of events to be loaded. Defaults to 0, to load everythings.
            nevents (int, optional) : number of events in run if known (parameter used to save computing time)
            run_file (optional) : if provided, will load this run file
        """
        super().__init__(run_number,max_events,run_file,*args,**kwargs)
        self.__nsamples =  self._reader.camera_config.num_samples
        
        #data we want to compute
        self.__ucts_timestamp = {}
        self.__ucts_busy_counter = {}
        self.__ucts_event_counter = {}
        self.__event_type = {}
        self.__event_id = {}
        self.__trig_patter_all = {}
        self.__broken_pixels_hg = {}
        self.__broken_pixels_lg = {}

    def _init_trigger_type(self, trigger : EventType, **kwargs):
        """
        Initializes empty lists for different trigger types in the ArrayDataMaker class.

        Args:
            trigger (EventType): The trigger type for which the lists are being initialized.

        Returns:
            None. The method only initializes the empty lists for the trigger type.
        """
        name = __class__._get_name_trigger(trigger)
        self.__ucts_timestamp[f"{name}"] = []
        self.__ucts_busy_counter[f"{name}"] = []
        self.__ucts_event_counter[f"{name}"] = []
        self.__event_type[f"{name}"] = []
        self.__event_id[f"{name}"] = []
        self.__trig_patter_all[f"{name}"] = []
        self.__broken_pixels_hg[f"{name}"] = []
        self.__broken_pixels_lg[f"{name}"] = []


    @staticmethod
    def _compute_broken_pixels(wfs_hg, wfs_lg, **kwargs):
        """
        Computes broken pixels for high and low gain waveforms.
        Args:
            wfs_hg (ndarray): High gain waveforms.
            wfs_lg (ndarray): Low gain waveforms.
            **kwargs: Additional keyword arguments.
        Returns:
            tuple: Two arrays of zeros with the same shape as `wfs_hg` (or `wfs_lg`) but without the last dimension.
        """
        log.warning("computation of broken pixels is not yet implemented")
        return np.zeros((wfs_hg.shape[:-1]), dtype=bool), np.zeros((wfs_hg.shape[:-1]), dtype=bool)
    
    @staticmethod
    def _compute_broken_pixels_event(event: NectarCAMDataContainer, pixels_id : np.ndarray, **kwargs):
            """
            Computes broken pixels for a specific event and pixel IDs.
            Args:
                event (NectarCAMDataContainer): An event.
                pixels_id (list or np.ndarray): IDs of pixels.
                **kwargs: Additional keyword arguments.
            Returns:
                tuple: Two arrays of zeros with the length of `pixels_id`.
            """
            log.warning("computation of broken pixels is not yet implemented")
            return np.zeros((len(pixels_id)), dtype=bool), np.zeros((len(pixels_id)), dtype=bool)

    @staticmethod
    def _get_name_trigger(trigger: EventType):
        """            
        Gets the name of a trigger event.
        Args:
            trigger (EventType): A trigger event.
        Returns:
            str: The name of the trigger event.
        """
        if trigger is None:
            name = "None"
        else:
            name = trigger.name
        return name

    
    def make(self, n_events=np.inf, trigger_type: list = None, restart_from_begining : bool=False, *args, **kwargs):
        """
        Method to extract data from the EventSource.

        Args:
            n_events (int, optional): The maximum number of events to process. Default is np.inf.
            trigger_type (list[EventType], optional): Only events with the specified trigger types will be used. Default is None.
            restart_from_begining (bool, optional): Whether to restart the event source reader. Default is False.
            *args: Additional arguments that can be passed to the method.
            **kwargs: Additional keyword arguments that can be passed to the method.

        Returns:
            The output container created by the _make_output_container method.
        """
        if ~np.isfinite(n_events):
            log.warning('no needed events number specified, it may cause a memory error')
        if isinstance(trigger_type, EventType) or trigger_type is None:
            trigger_type = [trigger_type]
        for _trigger_type in trigger_type:
            self._init_trigger_type(_trigger_type)

        if restart_from_begining:
            log.debug('restart from begining : creation of the EventSource reader')
            self._reader = __class__.load_run(self._run_number, self._max_events, run_file=self._run_file)

        n_traited_events = 0
        for i, event in enumerate(self._reader):
            if i % 100 == 0:
                log.info(f"reading event number {i}")
            for trigger in trigger_type:
                if (trigger is None) or (trigger == event.trigger.event_type):
                    self._make_event(event, trigger, *args, **kwargs)
                    n_traited_events += 1
            if n_traited_events >= n_events:
                break

        return self._make_output_container(trigger_type, *args, **kwargs)


    def _make_event(self, event : NectarCAMDataContainer, trigger : EventType, *args, **kwargs):
        """
        Method to extract data from the event.

        Args:
            event (NectarCAMDataContainer): The event object.
            trigger (EventType): The trigger type.
            *args: Additional arguments that can be passed to the method.
            **kwargs: Additional keyword arguments that can be passed to the method.

        Returns:
            If the return_wfs keyword argument is True, the method returns the high and low gain waveforms from the event.
        """
        name = __class__._get_name_trigger(trigger)
        self.__event_id[f'{name}'].append(np.uint16(event.index.event_id))
        self.__ucts_timestamp[f'{name}'].append(event.nectarcam.tel[__class__.TEL_ID].evt.ucts_timestamp)
        self.__event_type[f'{name}'].append(event.trigger.event_type.value)
        self.__ucts_busy_counter[f'{name}'].append(event.nectarcam.tel[__class__.TEL_ID].evt.ucts_busy_counter)
        self.__ucts_event_counter[f'{name}'].append(event.nectarcam.tel[__class__.TEL_ID].evt.ucts_event_counter)
        self.__trig_patter_all[f'{name}'].append(event.nectarcam.tel[__class__.TEL_ID].evt.trigger_pattern.T)

        if kwargs.get("return_wfs", False):
            get_wfs_hg = event.r0.tel[0].waveform[constants.HIGH_GAIN][self.pixels_id]
            get_wfs_lg = event.r0.tel[0].waveform[constants.LOW_GAIN][self.pixels_id]
            return get_wfs_hg, get_wfs_lg


    @abstractmethod
    def _make_output_container(self) : pass

    @staticmethod
    def select_container_array_field(container: ArrayDataContainer, pixel_id: np.ndarray, field: str) -> np.ndarray:
        """
        Selects specific fields from an ArrayDataContainer object based on a given list of pixel IDs.

        Args:
            container (ArrayDataContainer): An object of type ArrayDataContainer that contains the data.
            pixel_id (ndarray): An array of pixel IDs for which the data needs to be selected.
            field (str): The name of the field to be selected from the container.

        Returns:
            ndarray: An array containing the selected data for the given pixel IDs.
        """
        mask_contain_pixels_id = np.array([pixel in container.pixels_id for pixel in pixel_id], dtype=bool)
        for pixel in pixel_id[~mask_contain_pixels_id]:
            log.warning(f"You asked for pixel_id {pixel} but it is not present in this container, skip this one")
        res = np.array([np.take(container[field], np.where(container.pixels_id == pixel)[0][0], axis=1) for pixel in pixel_id[mask_contain_pixels_id]])
        ####could be nice to return np.ma.masked_array(data = res, mask = container.broken_pixels_hg.transpose(res.shape[1],res.shape[0],res.shape[2]))
        return res



    @staticmethod
    def merge(container_a : ArrayDataContainer,container_b : ArrayDataContainer) -> ArrayDataContainer : 
        """method to merge 2 ArrayDataContainer into one single ArrayDataContainer

        Returns:
            ArrayDataContainer: the merged object
        """
        if type(container_a) != type(container_b) : 
            raise Exception("The containers have to be instnace of the same class")

        if np.array_equal(container_a.pixels_id,container_b.pixels_id) : 
            raise Exception("The containers have not the same pixels ids")
        
        merged_container = container_a.__class__.__new__()

        for field in container_a.keys() : 
            if ~isinstance(container_a[field],np.ndarray) : 
                if container_a[field] != container_b[field] : 
                    raise Exception(f"merge impossible because of {field} filed (values are {container_a[field]} and {container_b[field]}")
        
        for field in container_a.keys() : 
            if isinstance(container_a[field],np.ndarray) : 
                merged_container[field] = np.concatenate(container_a[field],container_a[field],axis = 0)
            else : 
                merged_container[field] = container_a[field]

        return merged_container


    
    @property
    def nsamples(self) : 
        """
        Returns a deep copy of the nsamples attribute.
    
        Returns:
            np.ndarray: A deep copy of the nsamples attribute.
        """
        return copy.deepcopy(self.__nsamples)

    @property
    def _nsamples(self) : 
        """
        Returns the nsamples attribute.
    
        Returns:
            np.ndarray: The nsamples attribute.
        """
        return self.__nsamples

    def nevents(self,trigger : EventType) : 
        """
        Returns the number of events for the specified trigger type.
    
        Args:
            trigger (EventType): The trigger type for which the number of events is requested.
        
        Returns:
            int: The number of events for the specified trigger type.
        """
        return len(self.__event_id[__class__._get_name_trigger(trigger)])

    @property
    def _broken_pixels_hg(self) : 
        """
        Returns the broken_pixels_hg attribute.
    
        Returns:
            np.ndarray: The broken_pixels_hg attribute.
        """
        return self.__broken_pixels_hg

    def broken_pixels_hg(self,trigger : EventType) : 
        """
        Returns an array of broken pixels for high gain for the specified trigger type.
    
        Args:
            trigger (EventType): The trigger type for which the broken pixels for high gain are requested.
        
        Returns:
            np.ndarray: An array of broken pixels for high gain for the specified trigger type.
        """
        return np.array(self.__broken_pixels_hg[__class__._get_name_trigger(trigger)],dtype = bool)

    @property
    def _broken_pixels_lg(self) : 
        """
        Returns the broken_pixels_lg attribute.
    
        Returns:
            np.ndarray: The broken_pixels_lg attribute.
        """
        return self.__broken_pixels_lg

    def broken_pixels_lg(self,trigger : EventType) : 
        """
        Returns an array of broken pixels for low gain for the specified trigger type.
    
        Args:
            trigger (EventType): The trigger type for which the broken pixels for low gain are requested.
        
        Returns:
            np.ndarray: An array of broken pixels for low gain for the specified trigger type.
        """
        return np.array(self.__broken_pixels_lg[__class__._get_name_trigger(trigger)],dtype = bool)

    def ucts_timestamp(self,trigger : EventType) : 
        """
        Returns an array of UCTS timestamps for the specified trigger type.
    
        Args:
            trigger (EventType): The trigger type for which the UCTS timestamps are requested.
        
        Returns:
            np.ndarray: An array of UCTS timestamps for the specified trigger type.
        """
        return np.array(self.__ucts_timestamp[__class__._get_name_trigger(trigger)],dtype = np.uint64)

    def ucts_busy_counter(self,trigger : EventType) : 
        """
        Returns an array of UCTS busy counters for the specified trigger type.
    
        Args:
            trigger (EventType): The trigger type for which the UCTS busy counters are requested.
        
        Returns:
            np.ndarray: An array of UCTS busy counters for the specified trigger type.
        """
        return np.array(self.__ucts_busy_counter[__class__._get_name_trigger(trigger)],dtype = np.uint32)

    def ucts_event_counter(self,trigger : EventType) : 
        """
        Returns an array of UCTS event counters for the specified trigger type.
    
        Args:
            trigger (EventType): The trigger type for which the UCTS event counters are requested.
        
        Returns:
            np.ndarray: An array of UCTS event counters for the specified trigger type.
        """
        return np.array(self.__ucts_event_counter[__class__._get_name_trigger(trigger)],dtype = np.uint32)

    def event_type(self,trigger : EventType) : 
        """
        Returns an array of event types for the specified trigger type.
    
        Args:
            trigger (EventType): The trigger type for which the event types are requested.
        
        Returns:
            np.ndarray: An array of event types for the specified trigger type.
        """
        return np.array(self.__event_type[__class__._get_name_trigger(trigger)],dtype = np.uint8)

    def event_id(self,trigger : EventType) : 
        """
        Returns an array of event IDs for the specified trigger type.
    
        Args:
            trigger (EventType): The trigger type for which the event IDs are requested.
        
        Returns:
            np.ndarray: An array of event IDs for the specified trigger type.
        """
        return np.array(self.__event_id[__class__._get_name_trigger(trigger)],dtype = np.uint32)

    def multiplicity(self,trigger : EventType) :  
        """
        Returns an array of multiplicities for the specified trigger type.
    
        Args:
            trigger (EventType): The trigger type for which the multiplicities are requested.
        
        Returns:
            np.ndarray: An array of multiplicities for the specified trigger type.
        """
        tmp = self.trig_pattern(trigger)
        if len(tmp) == 0 : 
            return np.array([])
        else : 
            return np.uint16(np.count_nonzero(tmp,axis = 1))

    def trig_pattern(self,trigger : EventType) :  
        """
        Returns an array of trigger patterns for the specified trigger type.
    
        Args:
            trigger (EventType): The trigger type for which the trigger patterns are requested.
        
        Returns:
            np.ndarray: An array of trigger patterns for the specified trigger type.
        """
        tmp = self.trig_pattern_all(trigger)
        if len(tmp) == 0 : 
            return np.array([])
        else : 
            return tmp.any(axis = 2)

    def trig_pattern_all(self,trigger : EventType) :  
        """
        Returns an array of trigger patterns for all events for the specified trigger type.
    
        Args:
            trigger (EventType): The trigger type for which the trigger patterns for all events are requested.
        
        Returns:
            np.ndarray: An array of trigger patterns for all events for the specified trigger type.
        """
        return np.array(self.__trig_patter_all[f"{__class__._get_name_trigger(trigger)}"],dtype = bool)

