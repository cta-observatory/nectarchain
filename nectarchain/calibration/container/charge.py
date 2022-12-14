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

from ctapipe_io_nectarcam import NectarCAMEventSource
from ctapipe.io import EventSource, EventSeeker

from astropy.table import Table
from astropy.io import fits

from .waveforms import WaveformsContainer

__all__ = ['ChargeContainer']

list_ctapipe_charge_extractor = ["FullWaveformSum",
                        "FixedWindowSum",
                        "GlobalPeakWindowSum",
                        "LocalPeakWindowSum",
                        "SlidingWindowMaxSum",
                        "NeighborPeakWindowSum",
                        "BaselineSubtractedNeighborPeakWindowSum",
                        "TwoPassWindowSum"]



class ChargeContainer() : 
    def __init__(self,charge_hg,charge_lg,peak_hg,peak_lg,run_number,pixels_id,method = "FullWaveformSum") : 
        self.charge_hg = charge_hg
        self.charge_lg = charge_lg
        self.peak_hg = peak_hg
        self.peak_lg = peak_lg
        self.__run_number = run_number
        self.__pixels_id = pixels_id
        self.__method = method

    @classmethod
    def from_waveform(cls,waveformContainer : WaveformsContainer,method : str = "FullWaveformSum",**kwargs) : 
        log.info(f"computing hg charge with {method} method")
        charge_hg,peak_hg = ChargeContainer.compute_charge(waveformContainer,0,method,**kwargs)
        charge_hg = np.array(charge_hg,dtype = np.uint16)
        log.info(f"computing lg charge with {method} method")
        charge_lg,peak_lg = ChargeContainer.compute_charge(waveformContainer,1,method,**kwargs)
        charge_lg = np.array(charge_lg,dtype = np.uint16)

        return cls(charge_hg,charge_lg,peak_hg,peak_lg,waveformContainer.run_number,waveformContainer.pixels_ids,method)

    def write(self,path : Path,**kwargs) : 
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
        hdr['COMMENT'] = f"The charge containeur for run {self.__run_number} with {self.__method} method : primary is the pixels id, then you can find HG charge, LG charge, HG peak and LG peak"

        primary_hdu = fits.PrimaryHDU(self.pixels_id,header=hdr)
        charge_hg_hdu = fits.ImageHDU(self.charge_hg)
        charge_lg_hdu = fits.ImageHDU(self.charge_lg)
        peak_hg_hdu = fits.ImageHDU(self.peak_hg)
        peak_lg_hdu = fits.ImageHDU(self.peak_lg)

        hdul = fits.HDUList([primary_hdu, charge_hg_hdu, charge_lg_hdu,peak_hg_hdu,peak_lg_hdu])
        try : 
            hdul.writeto(Path(path)/f"charge_run{self.run_number}.fits",overwrite=kwargs.get('overwrite',False))
        except OSError as e : 
            log.warning(e)
        except Exception as e :
            log.error(e,exc_info = True)
            raise e




    @classmethod
    def from_file(cls,path : Path,run_number : int,**kwargs) : 
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
        charge_hg = hdul[1].data
        charge_lg = hdul[2].data
        peak_hg = hdul[3].data
        peak_lg = hdul[4].data


        return cls(charge_hg,charge_lg,peak_hg,peak_lg,run_number,pixels_id)

        
    @staticmethod 
    def compute_charge(waveformContainer : WaveformsContainer,channel : int,method : str = "FullWaveformSum" ,**kwargs) : 
        if not(method in list_ctapipe_charge_extractor) :
            raise ArgumentError(f"method must be in {list_ctapipe_charge_extractor}")
        ImageExtractor = eval(method)(waveformContainer.reader.subarray)
        if channel == 0:
            return ImageExtractor(waveformContainer.wfs_hg,waveformContainer.TEL_ID,channel)
        elif channel == 1:
            return ImageExtractor(waveformContainer.wfs_lg,waveformContainer.TEL_ID,channel)
        else :
            raise ArgumentError("channel must be 0 or 1")

    def histo_hg(self,n_bins : int = 1000,autoscale : bool = False) -> np.ndarray:
        if autoscale : 
            all_range = np.arange(np.int16(np.min(self.charge_hg)) + 0.5,np.int16(np.max(self.charge_hg)) + 0.5,1)
            hist_ma = ma.masked_array(np.empty((self.charge_hg.shape[1],all_range.shape[0]),dtype = np.int16), mask=np.zeros((self.charge_hg.shape[1],all_range.shape[0]),dtype = bool))
            charge_ma = ma.masked_array(np.empty((self.charge_hg.shape[1],all_range.shape[0])), mask=np.zeros((self.charge_hg.shape[1],all_range.shape[0]),dtype = bool))
            
            for i in range(self.charge_hg.shape[1]) :
                hist,charge = np.histogram(self.charge_hg.T[i],bins=np.arange(np.int16(np.min(self.charge_hg.T[i])),np.int16(np.max(self.charge_hg.T[i])) + 1,1))
                charge_edges = np.array([np.mean(charge[i:i+2],axis = 0) for i in range(charge.shape[0]-1)]) 
                mask = (all_range >= charge_edges[0]) * (all_range <= charge_edges[-1])

                #MASK THE DATA
                hist_ma.mask[i] = ~mask
                charge_ma.mask[i] = ~mask

                #FILL THE DATA
                hist_ma.data[i][mask] = hist
                charge_ma.data[i] = all_range
            
            return ma.masked_array((hist_ma,charge_ma))
            

        else : 
            hist = np.array([np.histogram(self.charge_hg.T[i],bins=n_bins)[0] for i in range(self.charge_hg.shape[1])])
            charge = np.array([np.histogram(self.charge_hg.T[i],bins=n_bins)[1] for i in range(self.charge_hg.shape[1])])
            charge_edges = np.array([np.mean(charge.T[i:i+2],axis = 0) for i in range(charge.shape[1]-1)]).T
            
            return np.array((hist,charge_edges))

    def histo_lg(self,n_bins: int = 1000,autoscale : bool = False) -> np.ndarray:
        if autoscale : 
            all_range = np.arange(np.int16(np.min(self.charge_lg)) + 0.5,np.int16(np.max(self.charge_lg)) + 0.5,1)
            hist_ma = ma.masked_array(np.empty((self.charge_lg.shape[1],all_range.shape[0]),dtype = np.int16), mask=np.zeros((self.charge_lg.shape[1],all_range.shape[0]),dtype = bool))
            charge_ma = ma.masked_array(np.empty((self.charge_lg.shape[1],all_range.shape[0])), mask=np.zeros((self.charge_lg.shape[1],all_range.shape[0]),dtype = bool))
            
            for i in range(self.charge_lg.shape[1]) :
                hist,charge = np.histogram(self.charge_lg.T[i],bins=np.arange(np.int16(np.min(self.charge_lg.T[i])),np.int16(np.max(self.charge_lg.T[i])) + 1,1))
                charge_edges = np.array([np.mean(charge[i:i+2],axis = 0) for i in range(charge.shape[0]-1)]) 
                mask = (all_range >= charge_edges[0]) * (all_range <= charge_edges[-1])

                #MASK THE DATA
                hist_ma.mask[i] = ~mask
                charge_ma.mask[i] = ~mask

                #FILL THE DATA
                hist_ma.data[i][mask] = hist
                charge_ma.data[i] = all_range
            
            return ma.masked_array((hist_ma,charge_ma)) 

        else : 
            hist = np.array([np.histogram(self.charge_lg.T[i],bins=n_bins)[0] for i in range(self.charge_lg.shape[1])])
            charge = np.array([np.histogram(self.charge_lg.T[i],bins=n_bins)[1] for i in range(self.charge_lg.shape[1])])
            charge_edges = np.array([np.mean(charge.T[i:i+2],axis = 0) for i in range(charge.shape[1]-1)]).T

        return np.array((hist,charge_edges))

    @property
    def run_number(self) : return self.__run_number

    @property
    def pixels_id(self) : return self.__pixels_id
