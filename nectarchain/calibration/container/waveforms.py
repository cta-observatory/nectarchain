from argparse import ArgumentError
import numpy as np
from matplotlib import pyplot as plt
import copy

from ctapipe.visualization import CameraDisplay
from ctapipe.coordinates import CameraFrame,EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry

from ctapipe_io_nectarcam import NectarCAMEventSource
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
        """

        self.__run_number = run_number
        #gerer ici le fait de traiter plusieurs fichiers ou simplement 1 par 1
        self.__reader = WaveformsContainer.load_run(run_number,max_events,merge_file = merge_file)

        #set camera properties
        log.info(f"N pixels : {self.npixels}")

        #run properties
        if nevents != -1 :
            self.__nevents = nevents if max_events is None else min(max_events,nevents) #self.check_events()
        else :
            self.__nevents = self.check_events()
            #reload file (bc check_events has drained reader generator)
            self.__reader = WaveformsContainer.load_run(run_number,max_events)
        log.info(f"N_events : {self.nevents}")

        


        #define empty members which will be filled therafter
        self.wfs_hg = np.empty((self.nevents,self.npixels,self.nsamples),dtype = np.uint16)
        self.wfs_lg = np.empty((self.nevents,self.npixels,self.nsamples),dtype = np.uint16)
        self.ucts_timestamp = np.empty((self.nevents),dtype = np.uint64)
        self.ucts_busy_counter = np.empty((self.nevents),dtype = np.uint16)
        self.ucts_event_counter = np.empty((self.nevents),dtype = np.uint16)
        self.wfs_triggertype = np.empty((self.nevents),dtype = object)
        self.wfs_evtid = np.empty((self.nevents),dtype = np.uint16)
        self.trig_pattern_all = np.empty((self.nevents,self.npixels,4),dtype = bool)
        self.trig_pattern = np.empty((self.nevents,self.npixels),dtype = bool)
        self.multiplicity = np.empty((self.nevents,self.npixels),dtype = np.uint16)


    @staticmethod
    def load_run(run_number : int,max_events : int = None) : 
        """Static method to load from $NECTARCAMDATA directory data for specified run with max_events

        Args:
            run_number (int): run_id
            maxevents (int, optional): max of events to be loaded. Defaults to 0, to load everythings.

        Returns:
            List[ctapipe_io_nectarcam.NectarCAMEventSource]: List of EventSource for each run files
        """
        filename,_ = DataManagment.findrun(run_number)
        log.info(f"{str(filename)} will be loaded")
        return NectarCAMEventSource(input_url=filename,max_events=max_events)
        
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

            self.wfs_evtid[i] = np.uint16(event.index.event_id)
            self.ucts_timestamp[i]=event.nectarcam.tel[WaveformsContainer.TEL_ID].evt.ucts_timestamp
            self.wfs_triggertype[i]=event.trigger.event_type
            self.ucts_busy_counter[i] = event.nectarcam.tel[WaveformsContainer.TEL_ID].evt.ucts_busy_counter
            self.ucts_event_counter[i] = event.nectarcam.tel[WaveformsContainer.TEL_ID].evt.ucts_event_counter


            self.trig_pattern_all[i] = event.nectarcam.tel[WaveformsContainer.TEL_ID].evt.trigger_pattern
            self.trig_pattern[i] = event.nectarcam.tel[WaveformsContainer.TEL_ID].evt.trigger_pattern.any(axis = 0)

            for pix in range(self.npixels):
                wfs_lg_tmp[pix]=event.r0.tel[0].waveform[1,pix]
                wfs_hg_tmp[pix]=event.r0.tel[0].waveform[0,pix]

            self.wfs_hg[i] = wfs_hg_tmp
            self.wfs_lg[i] = wfs_lg_tmp

        self.multiplicity = np.count_nonzero(self.trig_pattern,axis = 1)


        if compute_trigger_patern and np.max(self.trig_pattern) == 0:
            self.compute_trigger_patern()


    def create_output_table(self) : 
        #TODO
        pass

    def write(self,path : str) : 
        #TODO
        pass

    @staticmethod
    def load(path : str) : 
        #TODO
        pass

    

    


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
    def npixels(self) : return self.__reader.camera_config.num_pixels

    @property
    def nsamples(self) : return self.__reader.camera_config.num_samples


    @property
    def geometry(self) : return self.__reader.subarray.tel[WaveformsContainer.TEL_ID].camera

    @property
    def pixels_ids(self) : return self.__reader.camera_config.expected_pixels_id

    @property
    def nevents(self) : return self.__nevents

    @property
    def run_number(self) : return self.__run_number
