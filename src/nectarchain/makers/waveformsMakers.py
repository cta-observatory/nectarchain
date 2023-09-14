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

from ..data import DataManagement

import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from ctapipe.containers import EventType
from ..data.container import WaveformsContainer

from nectarchain.makers import BaseMaker

__all__ = ["WaveformsMaker"]

class WaveformsMaker(BaseMaker) :
    """class use to make the waveform extraction from event read from r0 data
    """
    TEL_ID = 0
    CAMERA_NAME = "NectarCam-003"
    CAMERA = CameraGeometry.from_name(CAMERA_NAME)

#constructors
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

        self.__reader = WaveformsMaker.load_run(run_number,max_events,run_file = run_file)

        #from reader members
        self.__npixels = self.__reader.camera_config.num_pixels
        self.__nsamples =  self.__reader.camera_config.num_samples
        self.__geometry = self.__reader.subarray.tel[WaveformsMaker.TEL_ID].camera
        self.__subarray =  self.__reader.subarray
        self.__pixels_id = self.__reader.camera_config.expected_pixels_id
        

        log.info(f"N pixels : {self.npixels}")
        
        #data we want to compute
        self.__wfs_hg = {}
        self.__wfs_lg = {}
        self.__ucts_timestamp = {}
        self.__ucts_busy_counter = {}
        self.__ucts_event_counter = {}
        self.__event_type = {}
        self.__event_id = {}
        self.__trig_patter_all = {}
        self.__broken_pixels_hg = {}
        self.__broken_pixels_lg = {}

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

        cls.__wfs_hg = {}
        cls.__wfs_lg = {}
        cls.__ucts_timestamp = {}
        cls.__ucts_busy_counter = {}
        cls.__ucts_event_counter = {}
        cls.__event_type = {}
        cls.__event_id = {}
        cls.__trig_patter_all = {}
        cls.__broken_pixels_hg = {}
        cls.__broken_pixels_lg = {}
        
        cls.__init_trigger_type(None)

        for event in tqdm(events_list):
            cls._make_event(event,None)

        return cls.__make_output_container([None])

            
    def __init_trigger_type(self,trigger_type,**kwargs) : 
        name = WaveformsMaker.__get_name_trigger(trigger_type)
        log.info(f"initiamization of the waveformsMaker following trigger type : {name}")
        self.__wfs_hg[f"{name}"] = []
        self.__wfs_lg[f"{name}"] = []
        self.__ucts_timestamp[f"{name}"] = []
        self.__ucts_busy_counter[f"{name}"] = []
        self.__ucts_event_counter[f"{name}"] = []
        self.__event_type[f"{name}"] = []
        self.__event_id[f"{name}"] = []
        self.__trig_patter_all[f"{name}"] = []
        self.__broken_pixels_hg[f"{name}"] = []
        self.__broken_pixels_lg[f"{name}"] = []

    

    def __compute_broken_pixels(self,wfs_hg_event,wfs_lg_event,**kwargs) : 
        log.warning("computation of broken pixels is not yet implemented")
        return np.zeros((self.npixels),dtype = bool),np.zeros((self.npixels),dtype = bool)
    
    @staticmethod
    def __get_name_trigger(trigger : EventType) : 
        if trigger is None : 
            name = "None"
        else : 
            name = trigger.name
        return name

    def make(self,n_events = np.inf, trigger_type : list = None, restart_from_begining = False):
        """mathod to extract waveforms data from the EventSource 

        Args:
            trigger_type (list[EventType], optional): only events with the asked trigger type will be use. Defaults to None.
            compute_trigger_patern (bool, optional): To recompute on our side the trigger patern. Defaults to False.
        """
        if ~np.isfinite(n_events) : 
            log.warning('no needed events number specified, it may cause a memory error')
        if isinstance(trigger_type,EventType) or trigger_type is None : 
            trigger_type = [trigger_type]

        if restart_from_begining : 
            log.debug('restart from begining : creation of the EventSource reader')
            self.__reader = WaveformsMaker.load_run(self.__run_number,self.__max_events,run_file = self.__run_file)
        
        for _trigger_type in trigger_type :
            self.__init_trigger_type(_trigger_type) 

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

        return self.__make_output_container(trigger_type)

    def _make_event(self,
                event,
                trigger : EventType
                ) : 
        name = WaveformsMaker.__get_name_trigger(trigger)
        self.__event_id[f'{name}'].append(np.uint16(event.index.event_id))
        self.__ucts_timestamp[f'{name}'].append(event.nectarcam.tel[WaveformsMaker.TEL_ID].evt.ucts_timestamp)
        self.__event_type[f'{name}'].append(event.trigger.event_type.value)
        self.__ucts_busy_counter[f'{name}'].append(event.nectarcam.tel[WaveformsMaker.TEL_ID].evt.ucts_busy_counter)
        self.__ucts_event_counter[f'{name}'].append(event.nectarcam.tel[WaveformsMaker.TEL_ID].evt.ucts_event_counter)
        self.__trig_patter_all[f'{name}'].append(event.nectarcam.tel[WaveformsMaker.TEL_ID].evt.trigger_pattern.T)

        wfs_hg_tmp=np.zeros((self.npixels,self.nsamples),dtype = np.uint16)
        wfs_lg_tmp=np.zeros((self.npixels,self.nsamples),dtype = np.uint16)

        for pix in range(self.npixels):
            wfs_lg_tmp[pix]=event.r0.tel[0].waveform[1,self.pixels_id[pix]]
            wfs_hg_tmp[pix]=event.r0.tel[0].waveform[0,self.pixels_id[pix]]

        self.__wfs_hg[f'{name}'].append(wfs_hg_tmp.tolist())
        self.__wfs_lg[f'{name}'].append(wfs_lg_tmp.tolist())

        broken_pixels_hg,broken_pixels_lg = self.__compute_broken_pixels(wfs_hg_tmp,wfs_lg_tmp)
        self.__broken_pixels_hg[f'{name}'].append(broken_pixels_hg.tolist())
        self.__broken_pixels_lg[f'{name}'].append(broken_pixels_lg.tolist())

    def __make_output_container(self,trigger_type) :
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
    def sort(waveformsContainer, method = 'event_id') : 
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
            output.ucts_timestamp = waveformsContainer.ucts_timestamp[index]
            output.ucts_busy_counter = waveformsContainer.ucts_busy_counter[index]
            output.ucts_event_counter = waveformsContainer.ucts_event_counter[index]
            output.event_type = waveformsContainer.event_type[index]
            output.event_id = waveformsContainer.event_id[index] 
            output.trig_pattern_all = waveformsContainer.trig_pattern_all[index]
            output.trig_pattern = waveformsContainer.trig_pattern[index]
            output.multiplicity = waveformsContainer.multiplicity[index]

            output.wfs_hg = waveformsContainer.wfs_hg[index]
            output.wfs_lg = waveformsContainer.wfs_lg[index]
            output.broken_pixels_hg = waveformsContainer.broken_pixels_hg[index]
            output.broken_pixels_lg = waveformsContainer.broken_pixels_lg[index]
        else : 
            raise ArgumentError(f"{method} is not a valid method for sorting")
        return output
    
    @staticmethod
    ##methods used to display
    def display(waveformsContainer,evt,geometry, cmap = 'gnuplot2') : 
        """plot camera display

        Args:
            evt (int): event index
            cmap (str, optional): colormap. Defaults to 'gnuplot2'.

        Returns:
            CameraDisplay: thoe cameraDisplay plot
        """
        image = waveformsContainer.wfs_hg.sum(axis=2)
        disp = CameraDisplay(geometry=geometry, image=image[evt], cmap=cmap)
        disp.add_colorbar()
        return disp

    @staticmethod
    def plot_waveform_hg(waveformsContainer,evt,**kwargs) :
        """plot the waveform of the evt

        Args:
            evt (int): the event index

        Returns:
            tuple: the figure and axes
        """
        if 'figure' in kwargs.keys() and 'ax' in kwargs.keys() :
            fig = kwargs.get('figure')
            ax = kwargs.get('ax')
        else : 
            fig,ax = plt.subplots(1,1)
        ax.plot(waveformsContainer.wfs_hg[evt].T)
        return fig,ax

    @staticmethod
    def select_waveforms_hg(waveformsContainer,pixel_id : np.ndarray) : 
        """method to extract waveforms HG from a list of pixel ids 
        The output is the waveforms HG with a shape following the size of the input pixel_id argument
        Pixel in pixel_id which are not present in the WaveformsContaineur pixels_id are skipped 
        Args:
            pixel_id (np.ndarray): array of pixel ids you want to extract the waveforms
        Returns:
            (np.ndarray): waveforms array in the order of specified pixel_id
        """
        mask_contain_pixels_id = np.array([pixel in waveformsContainer.pixels_id for pixel in pixel_id],dtype = bool)
        for pixel in pixel_id[~mask_contain_pixels_id] : log.warning(f"You asked for pixel_id {pixel} but it is not present in this WaveformsContainer, skip this one")
        res = np.array([waveformsContainer.wfs_hg[:,np.where(waveformsContainer.pixels_id == pixel)[0][0],:] for pixel in pixel_id[mask_contain_pixels_id]])
        res = res.transpose(1,0,2)
        ####could be nice to return np.ma.masked_array(data = res, mask = waveformsContainer.broken_pixels_hg.transpose(res.shape[1],res.shape[0],res.shape[2]))
        return res

    @staticmethod
    def select_waveforms_lg(waveformsContainer,pixel_id : np.ndarray) : 
        """method to extract waveforms LG from a list of pixel ids 
        The output is the waveforms LG with a shape following the size of the input pixel_id argument
        Pixel in pixel_id which are not present in the WaveformsContaineur pixels_id are skipped 

        Args:
            pixel_id (np.ndarray): array of pixel ids you want to extract the waveforms

        Returns:
            (np.ndarray): waveforms array in the order of specified pixel_id
        """
        mask_contain_pixels_id = np.array([pixel in waveformsContainer.pixels_id for pixel in pixel_id],dtype = bool)
        for pixel in pixel_id[~mask_contain_pixels_id] : log.warning(f"You asked for pixel_id {pixel} but it is not present in this WaveformsContainer, skip this one")
        res =  np.array([waveformsContainer.wfs_lg[:,np.where(waveformsContainer.pixels_id == pixel)[0][0],:] for pixel in pixel_id[mask_contain_pixels_id]])
        res = res.transpose(1,0,2)
        return res


    @property 
    def _run_file(self) : return self.__run_file     
    @property
    def _max_events(self) : return self.__max_events
    @property
    def reader(self) : return self.__reader
    @property
    def npixels(self) : return self.__npixels
    @property
    def nsamples(self) : return self.__nsamples
    @property
    def geometry(self) : return self.__geometry
    @property
    def subarray(self) : return self.__subarray
    @property
    def pixels_id(self) : return self.__pixels_id
    @property
    def run_number(self) : return self.__run_number
    def nevents(self,trigger) : return len(self.__event_id[WaveformsMaker.__get_name_trigger(trigger)])
    def wfs_hg(self,trigger) : return np.array(self.__wfs_hg[WaveformsMaker.__get_name_trigger(trigger)],dtype = np.uint16)
    def wfs_lg(self,trigger) : return np.array(self.__wfs_lg[WaveformsMaker.__get_name_trigger(trigger)],dtype = np.uint16)
    def broken_pixels_hg(self,trigger) : return np.array(self.__broken_pixels_hg[WaveformsMaker.__get_name_trigger(trigger)],dtype = bool)
    def broken_pixels_lg(self,trigger) : return np.array(self.__broken_pixels_lg[WaveformsMaker.__get_name_trigger(trigger)],dtype = bool)
    def ucts_timestamp(self,trigger) : return np.array(self.__ucts_timestamp[WaveformsMaker.__get_name_trigger(trigger)],dtype = np.uint64)
    def ucts_busy_counter(self,trigger) : return np.array(self.__ucts_busy_counter[WaveformsMaker.__get_name_trigger(trigger)],dtype = np.uint32)
    def ucts_event_counter(self,trigger) : return np.array(self.__ucts_event_counter[WaveformsMaker.__get_name_trigger(trigger)],dtype = np.uint32)
    def event_type(self,trigger) : return np.array(self.__event_type[WaveformsMaker.__get_name_trigger(trigger)],dtype = np.uint8)
    def event_id(self,trigger) : return np.array(self.__event_id[WaveformsMaker.__get_name_trigger(trigger)],dtype = np.uint32)
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
    def trig_pattern_all(self,trigger) :  return np.array(self.__trig_patter_all[f"{WaveformsMaker.__get_name_trigger(trigger)}"],dtype = bool)

