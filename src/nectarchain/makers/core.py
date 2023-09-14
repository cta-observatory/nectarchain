import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from abc import ABC, abstractmethod

from ctapipe_io_nectarcam import NectarCAMEventSource
import numpy as np
from ctapipe.containers import EventType
from ctapipe.visualization import CameraDisplay
from ctapipe.coordinates import CameraFrame,EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry,SubarrayDescription,TelescopeDescription
import copy

from ..data import DataManagement

__all__ = ["BaseMaker"]

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
    def load_run(run_number : int,max_events : int = None, run_file = None) : 
        """Static method to load from $NECTARCAMDATA directory data for specified run with max_events

        Args:self.__run_number = run_number
            run_number (int): run_id
            maxevents (int, optional): max of events to be loaded. Defaults to 0, to load everythings.
            run_file (optional) : if provided, will load this run file
        Returns:
            List[ctapipe_io_nectarcam.NectarCAMEventSource]: List of EventSource for each run files
        """
        if run_file is None : 
            generic_filename,_ = DataManagement.findrun(run_number)
            log.info(f"{str(generic_filename)} will be loaded")
            eventsource = NectarCAMEventSource(input_url=generic_filename,max_events=max_events)
        else :  
            log.info(f"{run_file} will be loaded")
            eventsource = NectarCAMEventSource(input_url=run_file,max_events=max_events)
        return eventsource


class LoopEventsMaker(BaseMaker):
    def __init__(self,run_number : int,max_events : int = None,run_file = None,*args,**kwargs):
        """construtor

        Args:
            run_number (int): id of the run to be loaded
            maxevents (int, optional): max of events to be loaded. Defaults to 0, to load everythings.
            nevents (int, optional) : number of events in run if known (parameter used to save computing time)
            run_file (optional) : if provided, will load this run file
        """
        super().__init__(*args,**kwargs)

        self.__run_number = run_number
        self.__run_file = run_file
        self.__max_events = max_events

        self.__reader = __class__.load_run(run_number,max_events,run_file = run_file)

        #from reader members
        self.__npixels = self.__reader.camera_config.num_pixels
        self.__pixels_id = self.__reader.camera_config.expected_pixels_id
        
        log.info(f"N pixels : {self.npixels}")

    def make(self,n_events = np.inf, restart_from_begining = False) : 
        if restart_from_begining : 
            log.debug('restart from begining : creation of the EventSource reader')
            self.__reader = __class__.load_run(self.__run_number,self.__max_events,run_file = self.__run_file)

        n_traited_events = 0
        for i,event in enumerate(self.__reader):
            if i%100 == 0:
                log.info(f"reading event number {i}")
            self._make_event(event)
            n_traited_events += 1
            if n_traited_events >= n_events : 
                break
    

    @abstractmethod
    def _make_event(self, event : EventType) : pass

    @property 
    def _run_file(self) : return self.__run_file     
    @property
    def _max_events(self) : return self.__max_events
    @property
    def _reader(self) : return self.__reader
    @property
    def _npixels(self) : return self.__npixels
    @property
    def _pixels_id(self) : return self.__pixels_id
    @property
    def _run_number(self) : return self.__run_number
    @property
    def reader(self) : return copy.deepcopy(self.__reader)
    @property
    def npixels(self) : return copy.deepcopy(self.__npixels)
    @property
    def pixels_id(self) : return copy.deepcopy(self.__pixels_id)
    @property
    def run_number(self) : return copy.deepcopy(self.__run_number)

    

class ArrayDataMaker(LoopEventsMaker) : 
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
        
        #data we want to compute
        self.__ucts_timestamp = {}
        self.__ucts_busy_counter = {}
        self.__ucts_event_counter = {}
        self.__event_type = {}
        self.__event_id = {}
        self.__trig_patter_all = {}
        self.__broken_pixels_hg = {}
        self.__broken_pixels_lg = {}

    def _init_trigger_type(self,trigger,**kwargs) :
        name = __class__._get_name_trigger(trigger)
        self.__ucts_timestamp[f"{name}"] = []
        self.__ucts_busy_counter[f"{name}"] = []
        self.__ucts_event_counter[f"{name}"] = []
        self.__event_type[f"{name}"] = []
        self.__event_id[f"{name}"] = []
        self.__trig_patter_all[f"{name}"] = []
        self.__broken_pixels_hg[f"{name}"] = []
        self.__broken_pixels_lg[f"{name}"] = []

    def _compute_broken_pixels(self,wfs_hg_event,wfs_lg_event,**kwargs) : 
        log.warning("computation of broken pixels is not yet implemented")
        return np.zeros((self.npixels),dtype = bool),np.zeros((self.npixels),dtype = bool)
    
    @staticmethod
    def _get_name_trigger(trigger : EventType) : 
        if trigger is None : 
            name = "None"
        else : 
            name = trigger.name
        return name
    
    def make(self,n_events = np.inf, trigger_type : list = None, restart_from_begining = False) : 
        """mathod to extract data from the EventSource 

        Args:
            trigger_type (list[EventType], optional): only events with the asked trigger type will be use. Defaults to None.
            compute_trigger_patern (bool, optional): To recompute on our side the trigger patern. Defaults to False.
        """
        if ~np.isfinite(n_events) : 
            log.warning('no needed events number specified, it may cause a memory error')
        if isinstance(trigger_type,EventType) or trigger_type is None : 
            trigger_type = [trigger_type]
        for _trigger_type in trigger_type :
            self._init_trigger_type(_trigger_type) 

        if restart_from_begining : 
            log.debug('restart from begining : creation of the EventSource reader')
            self.__reader = __class__.load_run(self.__run_number,self.__max_events,run_file = self.__run_file)

        n_traited_events = 0
        for i,event in enumerate(self.__reader):
            if i%100 == 0:
                log.info(f"reading event number {i}")
            for trigger in trigger_type : 
                if (trigger is None) or (trigger == event.trigger.event_type) : 
                    self._make_event(event,trigger)
                    n_traited_events += 1
            if n_traited_events >= n_events : 
                break

        return self._make_output_container()


    def _make_event(self,event,trigger) : 
        name = __class__._get_name_trigger(trigger)
        self.__event_id[f'{name}'].append(np.uint16(event.index.event_id))
        self.__ucts_timestamp[f'{name}'].append(event.nectarcam.tel[__class__.TEL_ID].evt.ucts_timestamp)
        self.__event_type[f'{name}'].append(event.trigger.event_type.value)
        self.__ucts_busy_counter[f'{name}'].append(event.nectarcam.tel[__class__.TEL_ID].evt.ucts_busy_counter)
        self.__ucts_event_counter[f'{name}'].append(event.nectarcam.tel[__class__.TEL_ID].evt.ucts_event_counter)
        self.__trig_patter_all[f'{name}'].append(event.nectarcam.tel[__class__.TEL_ID].evt.trigger_pattern.T)

        broken_pixels_hg,broken_pixels_lg = self._compute_broken_pixels()
        self.__broken_pixels_hg[f'{name}'].append(broken_pixels_hg.tolist())
        self.__broken_pixels_lg[f'{name}'].append(broken_pixels_lg.tolist())

    @abstractmethod
    def _make_output_container(self) : pass

    @staticmethod
    def create_from_events_list(events_list : list,
                                run_number : int,
                                npixels : int,
                                nsamples : int,
                                subarray,
                                pixels_id : int,
                                ) : 
        cls = __class__.__new__()

        cls.__run_number =  run_number
        cls.__nevents = len(events_list)
        cls.__npixels = npixels
        cls.__nsamples = nsamples
        cls.__subarray = subarray
        cls.__pixels_id = pixels_id

        cls.__ucts_timestamp = {}
        cls.__ucts_busy_counter = {}
        cls.__ucts_event_counter = {}
        cls.__event_type = {}
        cls.__event_id = {}
        cls.__trig_patter_all = {}
        cls.__broken_pixels_hg = {}
        cls.__broken_pixels_lg = {}
        
        return cls


    def nevents(self,trigger) : return len(self.__event_id[__class__._get_name_trigger(trigger)])
    @property
    def _broken_pixels_hg(self) : return self.__broken_pixels_hg
    def broken_pixels_hg(self,trigger) : return np.array(self.__broken_pixels_hg[__class__._get_name_trigger(trigger)],dtype = bool)
    @property
    def _broken_pixels_lg(self) : return self.__broken_pixels_lg
    def broken_pixels_lg(self,trigger) : return np.array(self.__broken_pixels_lg[__class__._get_name_trigger(trigger)],dtype = bool)
    def ucts_timestamp(self,trigger) : return np.array(self.__ucts_timestamp[__class__._get_name_trigger(trigger)],dtype = np.uint64)
    def ucts_busy_counter(self,trigger) : return np.array(self.__ucts_busy_counter[__class__._get_name_trigger(trigger)],dtype = np.uint32)
    def ucts_event_counter(self,trigger) : return np.array(self.__ucts_event_counter[__class__._get_name_trigger(trigger)],dtype = np.uint32)
    def event_type(self,trigger) : return np.array(self.__event_type[__class__._get_name_trigger(trigger)],dtype = np.uint8)
    def event_id(self,trigger) : return np.array(self.__event_id[__class__._get_name_trigger(trigger)],dtype = np.uint32)
    def multiplicity(self,trigger) :  
        tmp = self.trig_pattern(trigger)
        if len(tmp) == 0 : 
            return np.array([])
        else : 
            return np.uint16(np.count_nonzero(tmp,axis = 1))
    def trig_pattern(self,trigger) :  
        tmp = self.trig_pattern_all(trigger)
        if len(tmp) == 0 : 
            return np.array([])
        else : 
            return tmp.any(axis = 2)
    def trig_pattern_all(self,trigger) :  return np.array(self.__trig_patter_all[f"{__class__._get_name_trigger(trigger)}"],dtype = bool)

