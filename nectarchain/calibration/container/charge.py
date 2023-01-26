from argparse import ArgumentError
import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt
import copy
from pathlib import Path
import sys
import os
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from ctapipe.visualization import CameraDisplay
from ctapipe.coordinates import CameraFrame,EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.image.extractor import (FullWaveformSum,
    FixedWindowSum,
    GlobalPeakWindowSum,
    LocalPeakWindowSum,
    SlidingWindowMaxSum,
    NeighborPeakWindowSum,
    BaselineSubtractedNeighborPeakWindowSum,
    TwoPassWindowSum)

from ctapipe_io_nectarcam import NectarCAMEventSource,constants
from ctapipe.io import EventSource, EventSeeker

from astropy.table import Table
from astropy.io import fits

from .waveforms import WaveformsContainer,WaveformsContainers



#from .charge_extractor import *

__all__ = ['ChargeContainer','ChargeContainers']

list_ctapipe_charge_extractor = ["FullWaveformSum",
                        "FixedWindowSum",
                        "GlobalPeakWindowSum",
                        "LocalPeakWindowSum",
                        "SlidingWindowMaxSum",
                        "NeighborPeakWindowSum",
                        "BaselineSubtractedNeighborPeakWindowSum",
                        "TwoPassWindowSum"]

list_nectarchain_charge_extractor = ['gradient_extractor']




class ChargeContainer() : 
    """class used to compute charge from waveforms container"""
    TEL_ID = 0
    CAMERA = CameraGeometry.from_name("NectarCam-003")

    def __init__(self,charge_hg,charge_lg,peak_hg,peak_lg,run_number,pixels_id,nevents,npixels, method = "FullWaveformSum") : 
        self.charge_hg = charge_hg
        self.charge_lg = charge_lg
        self.peak_hg = peak_hg
        self.peak_lg = peak_lg
        self.__run_number = run_number
        self.__pixels_id = pixels_id
        self.__method = method
        self.__nevents = nevents
        self.__npixels = npixels


        self.ucts_timestamp = np.zeros((self.nevents),dtype = np.uint64)
        self.ucts_busy_counter = np.zeros((self.nevents),dtype = np.uint16)
        self.ucts_event_counter = np.zeros((self.nevents),dtype = np.uint16)
        self.event_type = np.zeros((self.nevents),dtype = np.uint8)
        self.event_id = np.zeros((self.nevents),dtype = np.uint16)
        self.trig_pattern_all = np.zeros((self.nevents,self.CAMERA.n_pixels,4),dtype = bool)

    @classmethod
    def from_waveforms(cls,waveformContainer : WaveformsContainer,method : str = "FullWaveformSum",**kwargs) : 
        log.info(f"computing hg charge with {method} method")
        charge_hg,peak_hg = ChargeContainer.compute_charge(waveformContainer,constants.HIGH_GAIN,method,**kwargs)
        charge_hg = np.array(charge_hg,dtype = np.uint16)
        log.info(f"computing lg charge with {method} method")
        charge_lg,peak_lg = ChargeContainer.compute_charge(waveformContainer,constants.LOW_GAIN,method,**kwargs)
        charge_lg = np.array(charge_lg,dtype = np.uint16)

        chargeContainer = cls(charge_hg,charge_lg,peak_hg,peak_lg,waveformContainer.run_number,waveformContainer.pixels_id,waveformContainer.nevents,waveformContainer.npixels ,method)
        
        chargeContainer.ucts_timestamp = waveformContainer.ucts_timestamp
        chargeContainer.ucts_busy_counter = waveformContainer.ucts_busy_counter
        chargeContainer.ucts_event_counter = waveformContainer.ucts_event_counter
        chargeContainer.event_type = waveformContainer.event_type
        chargeContainer.event_id = waveformContainer.event_id
        chargeContainer.trig_pattern_all = waveformContainer.trig_pattern_all

        return chargeContainer 


    def write(self,path : Path,**kwargs) : 
        suffix = kwargs.get("suffix","")
        if suffix != "" : suffix = f"_{suffix}"

        log.info(f"saving in {path}")
        os.makedirs(path,exist_ok = True)

        #table = Table(self.charge_hg)
        #table.meta["pixels_id"] = self.__pixels_id
        #table.write(Path(path)/f"charge_hg_run{self.run_number}.ecsv",format='ascii.ecsv',overwrite=kwargs.get('overwrite',False))
        #
        #table = Table(self.charge_lg)
        #table.meta["pixels_id"] = self.__pixels_id
        #table.write(Path(path)/f"charge_lg_run{self.run_number}.ecsv",format='ascii.ecsv',overwrite=kwargs.get('overwrite',False))
        #
        #table = Table(self.peak_hg)
        #table.meta["pixels_id"] = self.__pixels_id
        #table.write(Path(path)/f"peak_hg_run{self.run_number}.ecsv",format='ascii.ecsv',overwrite=kwargs.get('overwrite',False))
        #
        #table = Table(self.peak_lg)
        #table.meta["pixels_id"] = self.__pixels_id
        #table.write(Path(path)/f"peak_lg_run{self.run_number}.ecsv",format='ascii.ecsv',overwrite=kwargs.get('overwrite',False))

        hdr = fits.Header()
        hdr['RUN'] = self.__run_number
        hdr['NEVENTS'] = self.nevents
        hdr['NPIXELS'] = self.npixels
        hdr['COMMENT'] = f"The charge containeur for run {self.__run_number} with {self.__method} method : primary is the pixels id, then you can find HG charge, LG charge, HG peak and LG peak, 2 last HDU are composed of event properties and trigger patern"

        primary_hdu = fits.PrimaryHDU(self.pixels_id,header=hdr)
        charge_hg_hdu = fits.ImageHDU(self.charge_hg)
        charge_lg_hdu = fits.ImageHDU(self.charge_lg)
        peak_hg_hdu = fits.ImageHDU(self.peak_hg)
        peak_lg_hdu = fits.ImageHDU(self.peak_lg)

        col1 = fits.Column(array = self.event_id, name = "event_id", format = '1I')
        col2 = fits.Column(array = self.event_type, name = "event_type", format = '1I')
        col3 = fits.Column(array = self.ucts_timestamp, name = "ucts_timestamp", format = '1K')
        col4 = fits.Column(array = self.ucts_busy_counter, name = "ucts_busy_counter", format = '1I')
        col5 = fits.Column(array = self.ucts_event_counter, name = "ucts_event_counter", format = '1I')
        col6 = fits.Column(array = self.multiplicity, name = "multiplicity", format = '1I')

        coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6])
        event_properties = fits.BinTableHDU.from_columns(coldefs)

        col1 = fits.Column(array = self.trig_pattern_all, name = "trig_pattern_all", format = f'{4 * self.CAMERA.n_pixels}L',dim = f'({self.CAMERA.n_pixels},4)')
        col2 = fits.Column(array = self.trig_pattern, name = "trig_pattern", format = f'{self.CAMERA.n_pixels}L')
        coldefs = fits.ColDefs([col1, col2])
        trigger_patern = fits.BinTableHDU.from_columns(coldefs)

        hdul = fits.HDUList([primary_hdu, charge_hg_hdu, charge_lg_hdu,peak_hg_hdu,peak_lg_hdu,event_properties,trigger_patern])
        try : 
            hdul.writeto(Path(path)/f"charge_run{self.run_number}{suffix}.fits",overwrite=kwargs.get('overwrite',False))
            log.info(f"charge saved in {Path(path)}/charge_run{self.run_number}{suffix}.fits")
        except OSError as e : 
            log.warning(e)
        except Exception as e :
            log.error(e,exc_info = True)
            raise e




    @staticmethod
    def from_file(path : Path,run_number : int,**kwargs) : 
        log.info(f"loading in {path} run number {run_number}")
        
        #table = Table.read(Path(path)/f"charge_hg_run{run_number}.ecsv")
        #pixels_id = table.meta['pixels_id']
        #charge_hg = np.array([table[colname] for colname in table.colnames]).T
        #
        #table = Table.read(Path(path)/f"charge_lg_run{run_number}.ecsv")
        #charge_lg = np.array([table[colname] for colname in table.colnames]).T
        #
        #table = Table.read(Path(path)/f"peak_hg_run{run_number}.ecsv")
        #peak_hg = np.array([table[colname] for colname in table.colnames]).T
        #
        #table = Table.read(Path(path)/f"peak_lg_run{run_number}.ecsv")
        #peak_lg = np.array([table[colname] for colname in table.colnames]).T
        #
        hdul = fits.open(Path(path)/f"charge_run{run_number}.fits")
        pixels_id = hdul[0].data
        nevents = hdul[0].header['NEVENTS'] 
        npixels = hdul[0].header['NPIXELS'] 
        charge_hg = hdul[1].data
        charge_lg = hdul[2].data
        peak_hg = hdul[3].data
        peak_lg = hdul[4].data


        cls = ChargeContainer(charge_hg,charge_lg,peak_hg,peak_lg,run_number,pixels_id,nevents,npixels)

        cls.event_id = hdul[5].data["event_id"]
        cls.event_type = hdul[5].data["event_type"]
        cls.ucts_timestamp = hdul[5].data["ucts_timestamp"]
        cls.ucts_busy_counter = hdul[5].data["ucts_busy_counter"]
        cls.ucts_event_counter = hdul[5].data["ucts_event_counter"]

        table_trigger = hdul[6].data
        cls.trig_pattern_all = table_trigger["trig_pattern_all"]

        return cls

        
    @staticmethod 
    def compute_charge(waveformContainer : WaveformsContainer,channel : int,method : str = "FullWaveformSum" ,**kwargs) : 
        if not(method in list_ctapipe_charge_extractor or method in list_nectarchain_charge_extractor) :
            raise ArgumentError(f"method must be in {list_ctapipe_charge_extractor}")
        ImageExtractor = eval(method)(waveformContainer.subarray,**kwargs)
        if channel == constants.HIGH_GAIN:
            return ImageExtractor(waveformContainer.wfs_hg,waveformContainer.TEL_ID,channel)
        elif channel == constants.LOW_GAIN:
            return ImageExtractor(waveformContainer.wfs_lg,waveformContainer.TEL_ID,channel)
        else :
            raise ArgumentError("channel must be 0 or 1")

    def histo_hg(self,n_bins : int = 1000,autoscale : bool = True) -> np.ndarray:
        mask_broken_pix = np.array((self.charge_hg == self.charge_hg.mean(axis = 0)).mean(axis=0),dtype = bool)
        log.debug(f"there are {mask_broken_pix.sum()} broken pixels (charge stays at same level for each events)")
        
        if autoscale : 
            all_range = np.arange(np.uint16(np.min(self.charge_hg.T[~mask_broken_pix].T)) - 0.5,np.uint16(np.max(self.charge_hg.T[~mask_broken_pix].T)) + 1.5,1)
            hist_ma = ma.masked_array(np.zeros((self.charge_hg.shape[1],all_range.shape[0]),dtype = np.uint16), mask=np.zeros((self.charge_hg.shape[1],all_range.shape[0]),dtype = bool))
            charge_ma = ma.masked_array(np.zeros((self.charge_hg.shape[1],all_range.shape[0])), mask=np.zeros((self.charge_hg.shape[1],all_range.shape[0]),dtype = bool))

            new_data_mask = np.array([np.logical_or(charge_ma.mask.T[i],mask_broken_pix) for i in range(charge_ma.shape[1])])
            charge_ma.mask = new_data_mask
            hist_ma.mask = new_data_mask

            
            for i in range(self.charge_hg.shape[1]) :
                log.debug(f'computing charge histogram for pixel {i}')
                if mask_broken_pix[i] :
                    log.debug('This pixel is broken, I mask this one')
                    hist_ma.mask[i] = np.ones(hist_ma.mask[i].shape)
                    charge_ma.mask[i] = np.ones(charge_ma.mask[i].shape)
                else : 
                    log.debug("this pixel is not broken, let's continue computation")
                    hist,charge = np.histogram(self.charge_hg.T[i],bins=np.arange(np.uint16(np.min(self.charge_hg.T[i])) - 1, np.uint16(np.max(self.charge_hg.T[i])) + 2,1))
                    charge_edges = np.array([np.mean(charge[i:i+2],axis = 0) for i in range(charge.shape[0]-1)]) 
                    mask = (all_range >= charge_edges[0]) * (all_range <= charge_edges[-1])

                    #MASK THE DATA
                    hist_ma.mask[i] = np.logical_or(~mask, hist_ma.mask[i])
                    charge_ma.mask[i] = np.logical_or(~mask, charge_ma.mask[i])

                    #FILL THE DATA
                    hist_ma.data[i][mask] = hist
                    charge_ma.data[i] = all_range
            
            return ma.masked_array((hist_ma,charge_ma))
            

        else : 
            hist = np.array([np.histogram(self.charge_hg.T[i],bins=n_bins)[0] for i in range(self.charge_hg.shape[1])])
            charge = np.array([np.histogram(self.charge_hg.T[i],bins=n_bins)[1] for i in range(self.charge_hg.shape[1])])
            charge_edges = np.array([np.mean(charge.T[i:i+2],axis = 0) for i in range(charge.shape[1]-1)]).T
            
            return np.array((hist,charge_edges))

    def histo_lg(self,n_bins: int = 1000,autoscale : bool = True) -> np.ndarray:
        mask_broken_pix = np.array((self.charge_lg == self.charge_lg.mean(axis = 0)).mean(axis=0),dtype = bool)
        log.debug(f"there are {mask_broken_pix.sum()} broken pixels (charge stays at same level for each events)")
        
        if autoscale : 
            all_range = np.arange(np.uint16(np.min(self.charge_lg.T[~mask_broken_pix].T)) - 0.5,np.uint16(np.max(self.charge_lg.T[~mask_broken_pix].T)) + 1.5,1)
            hist_ma = ma.masked_array(np.zeros((self.charge_lg.shape[1],all_range.shape[0]),dtype = np.uint16), mask=np.zeros((self.charge_lg.shape[1],all_range.shape[0]),dtype = bool))
            charge_ma = ma.masked_array(np.zeros((self.charge_lg.shape[1],all_range.shape[0])), mask=np.zeros((self.charge_lg.shape[1],all_range.shape[0]),dtype = bool))

            new_data_mask = np.array([np.logical_or(charge_ma.mask.T[i],mask_broken_pix) for i in range(charge_ma.shape[1])])
            charge_ma.mask = new_data_mask
            hist_ma.mask = new_data_mask

            
            for i in range(self.charge_lg.shape[1]) :
                log.debug(f'computing charge histogram for pixel {i}')
                if mask_broken_pix[i] :
                    log.debug('This pixel is broken, I mask this one')
                    hist_ma.mask[i] = np.ones(hist_ma.mask[i].shape)
                    charge_ma.mask[i] = np.ones(charge_ma.mask[i].shape)
                else : 
                    log.debug("this pixel is not broken, let's continue computation")
                    hist,charge = np.histogram(self.charge_lg.T[i],bins=np.arange(np.uint16(np.min(self.charge_lg.T[i])) - 1, np.uint16(np.max(self.charge_lg.T[i])) + 2,1))
                    charge_edges = np.array([np.mean(charge[i:i+2],axis = 0) for i in range(charge.shape[0]-1)]) 
                    mask = (all_range >= charge_edges[0]) * (all_range <= charge_edges[-1])

                    #MASK THE DATA
                    hist_ma.mask[i] = np.logical_or(~mask, hist_ma.mask[i])
                    charge_ma.mask[i] = np.logical_or(~mask, charge_ma.mask[i])

                    #FILL THE DATA
                    hist_ma.data[i][mask] = hist
                    charge_ma.data[i] = all_range

        else : 
            hist = np.array([np.histogram(self.charge_lg.T[i],bins=n_bins)[0] for i in range(self.charge_lg.shape[1])])
            charge = np.array([np.histogram(self.charge_lg.T[i],bins=n_bins)[1] for i in range(self.charge_lg.shape[1])])
            charge_edges = np.array([np.mean(charge.T[i:i+2],axis = 0) for i in range(charge.shape[1]-1)]).T

        return np.array((hist,charge_edges))

    @property
    def run_number(self) : return self.__run_number

    @property
    def pixels_id(self) : return self.__pixels_id

    @property
    def npixels(self) : return self.__npixels

    @property
    def nevents(self) : return self.__nevents


    #physical properties
    @property
    def multiplicity(self) :  return np.uint16(np.count_nonzero(self.trig_pattern,axis = 1))

    @property
    def trig_pattern(self) :  return self.trig_pattern_all.any(axis = 1)




class ChargeContainers() : 
    def __init__(self, *args, **kwargs) : 
        self.chargeContainers = []
        self.__nChargeContainer = 0

    @classmethod
    def from_waveforms(cls,waveformContainers : WaveformsContainers,**kwargs) : 
        chargeContainers = cls()
        for i in range(waveformContainers.nWaveformsContainer) : 
            chargeContainers.append(ChargeContainer.from_waveforms(waveformContainers.waveformsContainer[i]))
        return chargeContainers

    def write(self,path : str, **kwargs) : 
        for i in range(self.__nChargeContainer) : 
            self.chargeContainers[i].write(path,suffix = f"{i:04d}" ,**kwargs)

    @property
    def nChargeContainer(self) : return self.__nChargeContainer

    @property
    def nevents(self) : 
        return np.sum([self.chargeContainers[i].nevents for i in range(self.__nChargeContainer)])

    def append(self,chargeContainer : ChargeContainer) : 
        self.chargeContainers.append(chargeContainer)
        self.__nChargeContainer += 1