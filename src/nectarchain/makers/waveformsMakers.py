import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from argparse import ArgumentError
import numpy as np
from tqdm import tqdm
import copy

from ctapipe.instrument import SubarrayDescription
from ctapipe_io_nectarcam import constants
from ctapipe.containers import EventType
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer


from ..data.container import WaveformsContainer
from .core import ArrayDataMaker

__all__ = ["WaveformsMaker"]

class WaveformsMaker(ArrayDataMaker) :
    """class use to make the waveform extraction from event read from r0 data
    """

#constructors
    def __init__(self,run_number : int,max_events : int = None,run_file = None,*args,**kwargs):
        """construtor

        Args:
            run_number (int): id of the run to be loaded
            maxevents (int, optional): max of events to be loaded. Defaults to 0, to load everythings.
            nevents (int, optional) : number of events in run if known (parameter used to save computing time)
            run_file (optional) : if provided, will load this run file
        """
        super().__init__(run_number,max_events,run_file,*args,**kwargs)

        self.__geometry = self._reader.subarray.tel[__class__.TEL_ID].camera
        self.__subarray =  self._reader.subarray
    
        self.__wfs_hg = {}
        self.__wfs_lg = {}

    @staticmethod
    def create_from_events_list(events_list : list,
                                run_number : int,
                                npixels : int,
                                nsamples :int,
                                subarray : SubarrayDescription,
                                pixels_id : int,
                                ) : 
        """Create a container for the extracted waveforms from a list of events.

        Args:
            events_list (list[NectarCAMDataContainer]): A list of events to extract waveforms from.
            run_number (int): The ID of the run to be loaded.
            npixels (int): The number of pixels in the waveforms.
            nsamples (int): The number of samples in the waveforms.
            subarray (SubarrayDescription): The subarray description instance.
            pixels_id (int): The ID of the pixels to extract waveforms from.

        Returns:
            WaveformsContainer: A container object that contains the extracted waveforms and other relevant information.
        """
        container = WaveformsContainer(
            run_number = run_number,
            npixels = npixels,
            nsamples = nsamples,
            subarray = subarray,
            camera = __class__.CAMERA_NAME,
            pixels_id = pixels_id,
        )

        ucts_timestamp = []
        ucts_busy_counter = []
        ucts_event_counter = []
        event_type = []
        event_id = []
        trig_pattern_all = []
        wfs_hg = []
        wfs_lg = []

        for event in tqdm(events_list) : 
            ucts_timestamp.append(event.nectarcam.tel[__class__.TEL_ID].evt.ucts_timestamp)
            ucts_busy_counter.append(event.nectarcam.tel[__class__.TEL_ID].evt.ucts_busy_counter)
            ucts_event_counter.append(event.nectarcam.tel[__class__.TEL_ID].evt.ucts_event_counter)
            event_type.append(event.trigger.event_type.value)
            event_id.append(event.index.event_id)
            trig_pattern_all.append(event.nectarcam.tel[__class__.TEL_ID].evt.trigger_pattern.T) 
            wfs_hg.append(event.r0.tel[0].waveform[constants.HIGH_GAIN][pixels_id])
            wfs_lg.append(event.r0.tel[0].waveform[constants.HIGH_GAIN][pixels_id])

        container.wfs_hg = np.array(wfs_hg,dtype = np.uint16)
        container.wfs_lg = np.array(wfs_lg,dtype = np.uint16)

        container.ucts_timestamp = np.array(ucts_timestamp,dtype = np.uint64)
        container.ucts_busy_counter = np.array(ucts_busy_counter,dtype = np.uint32)
        container.ucts_event_counter = np.array(ucts_event_counter,dtype = np.uint32)
        container.event_type = np.array(event_type,dtype = np.uint8)
        container.event_id = np.array(event_id,dtype = np.uint32)
        container.trig_pattern_all = np.array(trig_pattern_all,dtype =bool )
        container.trig_pattern = container.trig_pattern_all.any(axis = 2)
        container.multiplicity = np.uint16(np.count_nonzero(container.trig_pattern,axis = 1))

        broken_pixels = __class__._compute_broken_pixels(container.wfs_hg,container.wfs_lg)
        container.broken_pixels_hg = broken_pixels[0]         
        container.broken_pixels_lg = broken_pixels[1]
        return container

            
    def _init_trigger_type(self,trigger_type : EventType,**kwargs) : 
        """Initialize the waveformsMaker following the trigger type.

        Args:
            trigger_type: The type of trigger.

        """
        super()._init_trigger_type(trigger_type,**kwargs)
        name = __class__._get_name_trigger(trigger_type)
        log.info(f"initialization of the waveformsMaker following trigger type : {name}")
        self.__wfs_hg[f"{name}"] = []
        self.__wfs_lg[f"{name}"] = []
       

    def _make_event(self,
                event : NectarCAMDataContainer,
                trigger : EventType,
                *args,
                **kwargs
                ) : 
        """Process an event and extract waveforms.

        Args:
            event (NectarCAMDataContainer): The event to process and extract waveforms from.
            trigger (EventType): The type of trigger for the event.

        """
        wfs_hg_tmp=np.zeros((self.npixels,self.nsamples),dtype = np.uint16)
        wfs_lg_tmp=np.zeros((self.npixels,self.nsamples),dtype = np.uint16)

        wfs_hg_tmp,wfs_lg_tmp = super()._make_event(event = event,
                                                    trigger = trigger,
                                                    return_wfs = True,
                                                    *args,**kwargs)
        name = __class__._get_name_trigger(trigger)

        self.__wfs_hg[f'{name}'].append(wfs_hg_tmp.tolist())
        self.__wfs_lg[f'{name}'].append(wfs_lg_tmp.tolist())

        broken_pixels_hg,broken_pixels_lg = __class__._compute_broken_pixels_event(event,self._pixels_id)
        self._broken_pixels_hg[f'{name}'].append(broken_pixels_hg.tolist())
        self._broken_pixels_lg[f'{name}'].append(broken_pixels_lg.tolist())

    def _make_output_container(self,trigger_type : EventType,*args,**kwargs) :
        """Make the output container for the selected trigger types.

        Args:
            trigger_type (EventType): The selected trigger types.

        Returns:
            list[WaveformsContainer]: A list of output containers for the selected trigger types.
        """
        output = []
        for trigger in trigger_type :
            waveformsContainer = WaveformsContainer(
                run_number = self._run_number,
                npixels = self._npixels,
                nsamples = self._nsamples,
                subarray = self._subarray,
                camera = self.CAMERA_NAME,
                pixels_id = self._pixels_id,
                nevents = self.nevents(trigger),
                wfs_hg = self.wfs_hg(trigger),
                wfs_lg = self.wfs_lg(trigger),
                broken_pixels_hg = self.broken_pixels_hg(trigger),
                broken_pixels_lg = self.broken_pixels_lg(trigger),
                ucts_timestamp = self.ucts_timestamp(trigger),
                ucts_busy_counter = self.ucts_busy_counter(trigger),
                ucts_event_counter = self.ucts_event_counter(trigger),
                event_type = self.event_type(trigger),
                event_id = self.event_id(trigger),   
                trig_pattern_all = self.trig_pattern_all(trigger),         
                trig_pattern = self.trig_pattern(trigger),
                multiplicity = self.multiplicity(trigger)
            )
            output.append(waveformsContainer)
        return output

    @staticmethod
    def sort(waveformsContainer :WaveformsContainer, method : str = 'event_id') : 
        """Sort the waveformsContainer based on a specified method.

        Args:
            waveformsContainer (WaveformsContainer): The waveformsContainer to be sorted.
            method (str, optional): The sorting method. Defaults to 'event_id'.

        Returns:
            WaveformsContainer: The sorted waveformsContainer.
        """
        output = WaveformsContainer(
            run_number = waveformsContainer.run_number,
            npixels = waveformsContainer.npixels,
            nsamples = waveformsContainer.nsamples,
            subarray = waveformsContainer.subarray,
            camera = waveformsContainer.camera,
            pixels_id = waveformsContainer.pixels_id,
            nevents = waveformsContainer.nevents
        )
        if method == 'event_id' :
            index = np.argsort(waveformsContainer.event_id)
            for field in waveformsContainer.keys() :
                if not(field in ["run_number","npixels","nsamples","subarray","camera","pixels_id","nevents"]) : 
                    output[field] = waveformsContainer[field][index]
        else : 
            raise ArgumentError(f"{method} is not a valid method for sorting")
        return output

    @staticmethod
    def select_waveforms_hg(waveformsContainer:WaveformsContainer,pixel_id : np.ndarray) :
        """Select HIGH GAIN waveforms from the container.

        Args:
            waveformsContainer (WaveformsContainer): The container object that contains the waveforms.
            pixel_id (np.ndarray): An array of pixel IDs to select specific waveforms from the container.

        Returns:
            np.ndarray: An array of selected waveforms from the container.
        """ 
        res = __class__.select_container_array_field(container = waveformsContainer,pixel_id = pixel_id,field = 'wfs_lg')
        res = res.transpose(1,0,2)
        return res

    @staticmethod
    def select_waveforms_lg(waveformsContainer:WaveformsContainer,pixel_id : np.ndarray) : 
        """Select LOW GAIN waveforms from the container.

        Args:
            waveformsContainer (WaveformsContainer): The container object that contains the waveforms.
            pixel_id (np.ndarray): An array of pixel IDs to select specific waveforms from the container.

        Returns:
            np.ndarray: An array of selected waveforms from the container.
        """ 
        res = __class__.select_container_array_field(container = waveformsContainer,pixel_id = pixel_id,field = 'wfs_hg')
        res = res.transpose(1,0,2)
        return res



    @property
    def _geometry(self):
        """
        Returns the private __geometry attribute of the WaveformsMaker class.

        :return: The value of the private __geometry attribute.
        """
        return self.__geometry

    @property
    def _subarray(self):
        """
        Returns the private __subarray attribute of the WaveformsMaker class.

        :return: The value of the private __subarray attribute.
        """
        return self.__subarray
    @property
    def geometry(self):
        """
        Returns a deep copy of the geometry attribute.

        Returns:
            A deep copy of the geometry attribute.
        """
        return copy.deepcopy(self.__geometry)

    @property
    def subarray(self):
        """
        Returns a deep copy of the subarray attribute.

        Returns:
            A deep copy of the subarray attribute.
        """
        return copy.deepcopy(self.__subarray)

    def wfs_hg(self, trigger: EventType):
        """
        Returns the waveform data for the specified trigger type.

        Args:
            trigger (EventType): The type of trigger for which the waveform data is requested.

        Returns:
            An array of waveform data for the specified trigger type.
        """
        return np.array(self.__wfs_hg[__class__._get_name_trigger(trigger)], dtype=np.uint16)

    def wfs_lg(self, trigger: EventType):
        """
        Returns the waveform data for the specified trigger type in the low gain channel.

        Args:
            trigger (EventType): The type of trigger for which the waveform data is requested.

        Returns:
            An array of waveform data for the specified trigger type in the low gain channel.
        """
        return np.array(self.__wfs_lg[__class__._get_name_trigger(trigger)], dtype=np.uint16)