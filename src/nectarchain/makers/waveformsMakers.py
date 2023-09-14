import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from argparse import ArgumentError
import numpy as np
from matplotlib import pyplot as plt
import copy
import os
import glob
from pathlib import Path

from enum import Enum

from tqdm import tqdm

from astropy.io import fits
from astropy.table import QTable,Column,Table
import astropy.units as u

from ctapipe.visualization import CameraDisplay
from ctapipe.coordinates import CameraFrame,EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry,SubarrayDescription,TelescopeDescription

from ctapipe_io_nectarcam import NectarCAMEventSource
from ctapipe_io_nectarcam import constants


from ..data import DataManagement

import sys

from ctapipe.containers import EventType
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
        self.__nsamples =  self._reader.camera_config.num_samples
    
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
        container = WaveformsContainer()
        container.run_number = run_number
        container.npixels = npixels
        container.nsamples = nsamples
        container.subarray = subarray
        container.camera = __class__.CAMERA_NAME
        container.pixels_id = pixels_id

        ucts_timestamp = []
        ucts_busy_counter = []
        ucts_event_counter = []
        event_type = []
        event_id = []
        trig_pattern_all = []
        wfs_hg = []
        wfs_lg = []

        for event in tqdm(events_list) : 
            ucts_timestamp.append(event.ucts_timestamp)
            ucts_busy_counter.append(event.ucts_busy_counter)
            ucts_event_counter.append(event.ucts_event_counter)
            event_type.append(event.event_type)
            event_id.append(event.event_id)
            trig_pattern_all.append(event.trig_pattern_all) 
            broken_pixels = __class__._compute_broken_pixels_event(event,pixels_id)

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

        broken_pixels = __class__._compute_broken_pixels()
        container.broken_pixels_hg = broken_pixels[0]         
        container.broken_pixels_lg = broken_pixels[1]
        return container

            
    def _init_trigger_type(self,trigger_type,**kwargs) : 
        super()._init_trigger_type(trigger_type,**kwargs)
        name = __class__._get_name_trigger(trigger_type)
        log.info(f"initialization of the waveformsMaker following trigger type : {name}")
        self.__wfs_hg[f"{name}"] = []
        self.__wfs_lg[f"{name}"] = []
       

    def _make_event(self,
                event,
                trigger : EventType,
                *args,
                **kwargs
                ) : 
        wfs_hg_tmp=np.zeros((self.npixels,self.nsamples),dtype = np.uint16)
        wfs_lg_tmp=np.zeros((self.npixels,self.nsamples),dtype = np.uint16)

        super()._make_event(event = event,
                            trigger = trigger,
                            wfs_hg = wfs_hg_tmp,
                            wfs_lg = wfs_lg_tmp)
        name = __class__._get_name_trigger(trigger)

        self.__wfs_hg[f'{name}'].append(wfs_hg_tmp.tolist())
        self.__wfs_lg[f'{name}'].append(wfs_lg_tmp.tolist())

        broken_pixels_hg,broken_pixels_lg = __class__._compute_broken_pixels(wfs_hg_tmp,wfs_lg_tmp)
        self._broken_pixels_hg[f'{name}'].append(broken_pixels_hg.tolist())
        self._broken_pixels_lg[f'{name}'].append(broken_pixels_lg.tolist())

    def _make_output_container(self,trigger_type) :
        output = []
        for trigger in trigger_type :
            waveformsContainer = WaveformsContainer(
                run_number = self.run_number,
                npixels = self.npixels,
                nsamples = self.nsamples,
                subarray = self.subarray,
                camera = self.CAMERA_NAME,
                pixels_id = self.pixels_id,
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
    def sort(waveformsContainer :WaveformsContainer, method = 'event_id') : 
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
            for field in waveformsContainer() :
                if not(field in ["run_number","npixels","nsamples","subarray","camera","pixels_id","nevents"]) : 
                    output[field] = waveformsContainer[field][index]
            #output.ucts_busy_counter = waveformsContainer.ucts_busy_counter[index]
            #output.ucts_event_counter = waveformsContainer.ucts_event_counter[index]
            #output.event_type = waveformsContainer.event_type[index]
            #output.event_id = waveformsContainer.event_id[index] 
            #output.trig_pattern_all = waveformsContainer.trig_pattern_all[index]
            #output.trig_pattern = waveformsContainer.trig_pattern[index]
            #output.multiplicity = waveformsContainer.multiplicity[index]
#
            #output.wfs_hg = waveformsContainer.wfs_hg[index]
            #output.wfs_lg = waveformsContainer.wfs_lg[index]
            #output.broken_pixels_hg = waveformsContainer.broken_pixels_hg[index]
            #output.broken_pixels_lg = waveformsContainer.broken_pixels_lg[index]
        else : 
            raise ArgumentError(f"{method} is not a valid method for sorting")
        return output

    @staticmethod
    def select_waveforms_hg(waveformsContainer:WaveformsContainer,pixel_id : np.ndarray) : 
        res = __class__.select_container_array_field(container = waveformsContainer,pixel_id = pixel_id,field = 'wfs_lg')
        res = res.transpose(1,0,2)
        return res

    @staticmethod
    def select_waveforms_lg(waveformsContainer:WaveformsContainer,pixel_id : np.ndarray) : 
        res = __class__.select_container_array_field(container = waveformsContainer,pixel_id = pixel_id,field = 'wfs_hg')
        res = res.transpose(1,0,2)
        return res


    @property
    def nsamples(self) : return self.__nsamples
    @property
    def geometry(self) : return self.__geometry
    @property
    def subarray(self) : return self.__subarray
    def wfs_hg(self,trigger) : return np.array(self.__wfs_hg[__class__._get_name_trigger(trigger)],dtype = np.uint16)
    def wfs_lg(self,trigger) : return np.array(self.__wfs_lg[__class__._get_name_trigger(trigger)],dtype = np.uint16)
    