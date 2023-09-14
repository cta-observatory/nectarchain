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
                                nsamples : int,
                                subarray,
                                pixels_id : int,
                                ) : 
        cls = super().create_from_events_list(events_list = events_list,
                                run_number = run_number,
                                npixels = npixels,
                                pixels_id = pixels_id
                                )
        
        cls.__nsamples = nsamples
        cls.__subarray = subarray

        cls.__wfs_hg = {}
        cls.__wfs_lg = {}
        
        cls._init_trigger_type(None)
        
        for event in tqdm(events_list):
            cls._make_event(event,None)

        return cls._make_output_container([None])

            
    def _init_trigger_type(self,trigger_type,**kwargs) : 
        super()._init_trigger_type(trigger_type,**kwargs)
        name = __class__._get_name_trigger(trigger_type)
        log.info(f"initialization of the waveformsMaker following trigger type : {name}")
        self.__wfs_hg[f"{name}"] = []
        self.__wfs_lg[f"{name}"] = []



       

    def _make_event(self,
                event,
                trigger : EventType
                ) : 
        super()._make_event(event = event,
                            trigger = trigger)
        name = __class__._get_name_trigger(trigger)

        wfs_hg_tmp=np.zeros((self.npixels,self.nsamples),dtype = np.uint16)
        wfs_lg_tmp=np.zeros((self.npixels,self.nsamples),dtype = np.uint16)

        for pix in range(self.npixels):
            wfs_lg_tmp[pix]=event.r0.tel[0].waveform[1,self.pixels_id[pix]]
            wfs_hg_tmp[pix]=event.r0.tel[0].waveform[0,self.pixels_id[pix]]

        self.__wfs_hg[f'{name}'].append(wfs_hg_tmp.tolist())
        self.__wfs_lg[f'{name}'].append(wfs_lg_tmp.tolist())

        broken_pixels_hg,broken_pixels_lg = self._compute_broken_pixels(wfs_hg_tmp,wfs_lg_tmp)
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
    def nsamples(self) : return self.__nsamples
    @property
    def geometry(self) : return self.__geometry
    @property
    def subarray(self) : return self.__subarray
    def wfs_hg(self,trigger) : return np.array(self.__wfs_hg[__class__._get_name_trigger(trigger)],dtype = np.uint16)
    def wfs_lg(self,trigger) : return np.array(self.__wfs_lg[__class__._get_name_trigger(trigger)],dtype = np.uint16)
    