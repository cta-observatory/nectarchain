from argparse import ArgumentError
import numpy as np
from matplotlib import pyplot as plt
import copy
import os
from pathlib import Path

from enum import Enum

from astropy.io import fits
from astropy.table import QTable,Column,Table
import astropy.units as u

from ctapipe.visualization import CameraDisplay
from ctapipe.coordinates import CameraFrame,EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry

from ctapipe_io_nectarcam import NectarCAMEventSource
from ctapipe.containers import EventType
from ctapipe.io import EventSource, EventSeeker

from .utils import DataManagment,ChainGenerator

import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

__all__ = ["WaveformsContainer"]


class WaveformsContainer() :
    """class used to load run and load waveforms from r0 data
    """
    TEL_ID = 0
    CAMERA = CameraGeometry.from_name("NectarCam-003")

    def __init__(self,run_number : int,max_events : int = None,nevents : int = -1,merge_file = True):
        """construtor

        Args:
            run_number (int): id of the run to be loaded
            maxevents (int, optional): max of events to be loaded. Defaults to 0, to load everythings.
            nevents (int, optional) : number of events in run if known (parameter used to save computing time)
            merge_file (optional) : if True will load all fits.fz files of the run, else merge_file can be integer to select the run fits.fz file according to this number
            
        """

        self.__run_number = run_number
        #gerer ici le fait de traiter plusieurs fichiers ou simplement 1 par 1
        self.__reader = WaveformsContainer.load_run(run_number,max_events,merge_file = merge_file)

        #from reader members
        self.__npixels = self.__reader.camera_config.num_pixels
        self.__nsamples =  self.__reader.camera_config.num_samples
        self.__geometry = self.__reader.subarray.tel[WaveformsContainer.TEL_ID].camera
        self.__pixels_id = self.__reader.camera_config.expected_pixels_id
        
        #set camera properties
        log.info(f"N pixels : {self.npixels}")

        #run properties
        if nevents != -1 :
            self.__nevents = nevents if max_events is None else min(max_events,nevents) #self.check_events()
        else :
            self.__nevents = self.check_events()
            #reload file (bc check_events has drained reader generator)
            self.__reader = WaveformsContainer.load_run(run_number,max_events,merge_file = merge_file)
        log.info(f"N_events : {self.nevents}")

        
        

        


        #define empty members which will be filled therafter
        self.wfs_hg = np.empty((self.nevents,self.npixels,self.nsamples),dtype = np.uint16)
        self.wfs_lg = np.empty((self.nevents,self.npixels,self.nsamples),dtype = np.uint16)
        self.ucts_timestamp = np.empty((self.nevents),dtype = np.uint64)
        self.ucts_busy_counter = np.empty((self.nevents),dtype = np.uint16)
        self.ucts_event_counter = np.empty((self.nevents),dtype = np.uint16)
        self.event_type = np.empty((self.nevents),dtype = np.uint8)
        self.event_id = np.empty((self.nevents),dtype = np.uint16)
        self.trig_pattern_all = np.empty((self.nevents,self.npixels,4),dtype = bool)
        #self.trig_pattern = np.empty((self.nevents,self.npixels),dtype = bool)
        #self.multiplicity = np.empty((self.nevents,self.npixels),dtype = np.uint16)


    @staticmethod
    def load_run(run_number : int,max_events : int = None, merge_file = True) : 
        """Static method to load from $NECTARCAMDATA directory data for specified run with max_events

        Args:
            run_number (int): run_id
            maxevents (int, optional): max of events to be loaded. Defaults to 0, to load everythings.
            merge_file (optional) : if True will load all fits.fz files of the run, else merge_file can be integer to select the run fits.fz file according to this number
        Returns:
            List[ctapipe_io_nectarcam.NectarCAMEventSource]: List of EventSource for each run files
        """
        generic_filename,filenames = DataManagment.findrun(run_number)
        if merge_file : 
            log.info(f"{str(generic_filename)} will be loaded")
            eventsource = NectarCAMEventSource(input_url=generic_filename,max_events=max_events)
        else : 
            if isinstance(merge_file,int) : 
                eventsource = NectarCAMEventSource(input_url=filenames[merge_file],max_events=max_events)
        return eventsource
        
    def check_events(self):
        """method to check triggertype of events and counting number of events in all readers
            it prints the trigger type when trigger type change over events (to not write one line by events)

        Returns:
            int : number of events
        """
        log.info("checking trigger type")
        has_changed = True
        previous_trigger = None
        n_events = 0
        for i,event in enumerate(self.__reader):
            if previous_trigger is not None :
                has_changed = (previous_trigger.value != event.trigger.event_type.value)
            previous_trigger = event.trigger.event_type
            if has_changed : 
                log.info(f"event {i} is {previous_trigger} : {previous_trigger.value} trigger type")
            n_events+=1
        return n_events



    def load_wfs(self,compute_trigger_patern = False):
        wfs_hg_tmp=np.zeros((self.npixels,self.nsamples),dtype = np.uint16)
        wfs_lg_tmp=np.zeros((self.npixels,self.nsamples),dtype = np.uint16)

        n_traited_events = 0
        for i,event in enumerate(self.__reader):
            if i%100 == 0:
                log.info(f"reading event number {i}")

            self.event_id[i] = np.uint16(event.index.event_id)
            self.ucts_timestamp[i]=event.nectarcam.tel[WaveformsContainer.TEL_ID].evt.ucts_timestamp
            self.event_type[i]=event.trigger.event_type.value
            self.ucts_busy_counter[i] = event.nectarcam.tel[WaveformsContainer.TEL_ID].evt.ucts_busy_counter
            self.ucts_event_counter[i] = event.nectarcam.tel[WaveformsContainer.TEL_ID].evt.ucts_event_counter


            self.trig_pattern_all[i] = event.nectarcam.tel[WaveformsContainer.TEL_ID].evt.trigger_pattern.T
            #self.trig_pattern[i] = event.nectarcam.tel[WaveformsContainer.TEL_ID].evt.trigger_pattern.any(axis = 0)

            for pix in range(self.npixels):
                wfs_lg_tmp[pix]=event.r0.tel[0].waveform[1,pix]
                wfs_hg_tmp[pix]=event.r0.tel[0].waveform[0,pix]

            self.wfs_hg[i] = wfs_hg_tmp
            self.wfs_lg[i] = wfs_lg_tmp

        #self.multiplicity = np.count_nonzero(self.trig_pattern,axis = 1)


        #if compute_trigger_patern and np.max(self.trig_pattern) == 0:
        #    self.compute_trigger_patern()


    def write(self,path : str, **kwargs) : 
        log.info(f"saving in {path}")
        os.makedirs(path,exist_ok = True)

        hdr = fits.Header()
        hdr['RUN'] = self.__run_number
        hdr['NEVENTS'] = self.nevents
        hdr['NPIXELS'] = self.npixels
        hdr['NSAMPLES'] = self.nsamples



        hdr['COMMENT'] = f"The waveforms containeur for run {self.__run_number} : primary is the pixels id, 2nd HDU : high gain waveforms, 3rd HDU : low gain waveforms, 4th HDU :  event properties and 5th HDU trigger paterns."

        primary_hdu = fits.PrimaryHDU(self.pixels_id,header=hdr)

        wfs_hg_hdu = fits.ImageHDU(self.wfs_hg)
        wfs_lg_hdu = fits.ImageHDU(self.wfs_lg)


        col1 = fits.Column(array = self.event_id, name = "event_id", format = '1I')
        col2 = fits.Column(array = self.event_type, name = "event_type", format = '1I')
        col3 = fits.Column(array = self.ucts_timestamp, name = "ucts_timestamp", format = '1K')
        col4 = fits.Column(array = self.ucts_busy_counter, name = "ucts_busy_counter", format = '1I')
        col5 = fits.Column(array = self.ucts_event_counter, name = "ucts_event_counter", format = '1I')
        col6 = fits.Column(array = self.multiplicity, name = "multiplicity", format = '1I')

        coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6])
        event_properties = fits.BinTableHDU.from_columns(coldefs)

        col1 = fits.Column(array = self.trig_pattern_all, name = "trig_pattern_all", format = '7420L',dim = f'({self.npixels},4)')
        col2 = fits.Column(array = self.trig_pattern, name = "trig_pattern", format = '1855L')
        coldefs = fits.ColDefs([col1, col2])
        trigger_patern = fits.BinTableHDU.from_columns(coldefs)

        hdul = fits.HDUList([primary_hdu, wfs_hg_hdu, wfs_lg_hdu,event_properties,trigger_patern])
        try : 
            hdul.writeto(Path(path)/f"waveforms_run{self.run_number}.fits",overwrite=kwargs.get('overwrite',False))
            log.info(f"runs saved in {Path(path)}/waveforms_run{self.run_number}.fits")
        except OSError as e : 
            log.warning(e)
        except Exception as e :
            log.error(e,exc_info = True)
            raise e



    @classmethod
    def load(cls,path : str) : 
        log.info(f"loading from {path}")
        hdul = fits.open(Path(path))

        cls.__run_number = hdul[0].header['RUN'] 
        cls.nevents = hdul[0].header['NEVENTS'] 
        cls.npixels = hdul[0].header['NPIXELS'] 
        cls.nsamples = hdul[0].header['NSAMPLES'] 


        cls.pixels_id = hdul[0].data
        cls.wfs_hg = hdul[1].data
        cls.wfs_lg = hdul[2].data

        table_prop = hdul[3].data
        cls.event_id = table_prop["event_id"]
        cls.event_type = table_prop["event_type"]
        cls.ucts_timestamp = table_prop["ucts_timestamp"]
        cls.ucts_busy_counter = table_prop["ucts_busy_counter"]
        cls.ucts_event_counter = table_prop["ucts_event_counter"]

        table_trigger = hdul[4].data
        cls.trig_pattern_all = table_trigger["trig_pattern_all"]

        return cls

    

    


    def compute_trigger_patern(self) : 
        #mean.shape nevents * npixels
        mean,std =np.mean(self.wfs_hg,axis = 2),np.std(self.wfs_hg,axis = 2)
        self.trig_pattern = self.wfs_hg.max(axis = 2) > (mean + 3 * std)



    ##methods used to display
    def display(self,evt,cmap = 'gnuplot2') : 
        image = self.wfs_hg.sum(axis=2)
        disp = CameraDisplay(geometry=WaveformsContainer.CAMERA, image=image[evt], cmap=cmap)
        disp.add_colorbar()
        return disp

    def plot_waveform(self,evt) :
        fig,ax = plt.subplots(1,1)
        ax.plot(self.wfs_hg[evt].T)
        return fig,ax



    @property
    def reader(self) : return self.__reader

    @property
    def npixels(self) : return self.__npixels

    @property
    def nsamples(self) : return self.__nsamples

    @property
    def geometry(self) : return self.__geometry

    @property
    def pixels_id(self) : return self.__pixels_id

    @property
    def nevents(self) : return self.__nevents

    @property
    def run_number(self) : return self.__run_number


    #physical properties
    @property
    def multiplicity(self) :  return np.uint16(np.count_nonzero(self.trig_pattern,axis = 1))

    @property
    def trig_pattern(self) :  return self.trig_pattern_all.any(axis = 2)