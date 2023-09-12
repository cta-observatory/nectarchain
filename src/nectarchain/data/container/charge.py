from argparse import ArgumentError
import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt
import copy
from pathlib import Path
import glob
import time
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

from numba import guvectorize, float64, int64, bool_

from .waveforms import WaveformsContainer,WaveformsContainers
from ...makers.extractor.utils import CtapipeExtractor



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


@guvectorize(
[
    (int64[:], float64[:], bool_, bool_[:], int64[:]),
],

"(s),(n),()->(n),(n)",
nopython=True,
cache=True,
)
def make_histo(charge, all_range, mask_broken_pix, _mask, hist_ma_data):
    """compute histogram of charge with numba

    Args:
        charge (np.ndarray(pixels,nevents)): charge
        all_range (np.ndarray(nbins)): charge range
        mask_broken_pix (np.ndarray(pixels)): mask on broxen pixels
        _mask (np.ndarray(pixels,nbins)): mask
        hist_ma_data (np.ndarray(pixels,nbins)): histogram
    """
    #print(f"charge.shape = {charge.shape[0]}")
    #print(f"_mask.shape = {_mask.shape[0]}")
    #print(f"_mask[0] = {_mask[0]}")
    #print(f"hist_ma_data[0] = {hist_ma_data[0]}")
    #print(f"mask_broken_pix = {mask_broken_pix}")

    if not(mask_broken_pix) :
        #print("this pixel is not broken, let's continue computation")
        hist,_charge = np.histogram(charge,bins=np.arange(np.uint16(np.min(charge)) - 1, np.uint16(np.max(charge)) + 2,1))
        #print(f"hist.shape[0] = {hist.shape[0]}")
        #print(f"charge.shape[0] = {_charge.shape[0]}")
        charge_edges = np.array([np.mean(_charge[i:i+2]) for i in range(_charge.shape[0]-1)])
        #print(f"charge_edges.shape[0] = {charge_edges.shape[0]}")
        mask = (all_range >= charge_edges[0]) * (all_range <= charge_edges[-1])
        #print(f"all_range = {int(all_range[0])}-{int(all_range[-1])}")
        #print(f"charge_edges[0] = {int(charge_edges[0])}")
        #print(f"charge_edges[-1] = {int(charge_edges[-1])}")
        #print(f"mask[0] = {mask[0]}")
        #print(f"mask[-1] = {mask[-1]}")

        #MASK THE DATA
        #print(f"mask.shape = {mask.shape[0]}")
        _mask[:] = ~mask
        #print(f"_mask[0] = {_mask[0]}")
        #print(f"_mask[-1] = {_mask[-1]}")
        #FILL THE DATA
        hist_ma_data[mask] = hist
        #print("work done")
    else : 
        #print("this pixel is broken, skipped")
        pass
        
class ChargeContainer() : 
    """class used to compute charge from waveforms container"""
    TEL_ID = 0
    CAMERA = CameraGeometry.from_name("NectarCam-003")

    def __init__(self,charge_hg,charge_lg,peak_hg,peak_lg,run_number,pixels_id,nevents,npixels, method = "FullWaveformSum") : 
        self.charge_hg = charge_hg
        self.charge_lg = charge_lg
        self.peak_hg = peak_hg
        self.peak_lg = peak_lg
        self._run_number = run_number
        self._pixels_id = pixels_id
        self._method = method
        self._nevents = nevents
        self._npixels = npixels


        self.ucts_timestamp = np.zeros((self.nevents),dtype = np.uint64)
        self.ucts_busy_counter = np.zeros((self.nevents),dtype = np.uint16)
        self.ucts_event_counter = np.zeros((self.nevents),dtype = np.uint16)
        self.event_type = np.zeros((self.nevents),dtype = np.uint8)
        self.event_id = np.zeros((self.nevents),dtype = np.uint16)
        self.trig_pattern_all = np.zeros((self.nevents,self.CAMERA.n_pixels,4),dtype = bool)


    @classmethod
    def from_waveforms(cls,waveformContainer : WaveformsContainer,method : str = "FullWaveformSum",**kwargs) : 
        """ create a new ChargeContainer from a WaveformsContainer
        Args:
            waveformContainer (WaveformsContainer): the waveforms
            method (str, optional): Ctapipe ImageExtractor method. Defaults to "FullWaveformSum".

        Returns:
            chargeContainer : the ChargeContainer instance
        """
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
        """method to write in an output FITS file the ChargeContainer.

        Args:
            path (str): the directory where you want to save data
        """
        suffix = kwargs.get("suffix","")
        if suffix != "" : suffix = f"_{suffix}"

        log.info(f"saving in {path}")
        os.makedirs(path,exist_ok = True)

        #table = Table(self.charge_hg)
        #table.meta["pixels_id"] = self._pixels_id
        #table.write(Path(path)/f"charge_hg_run{self.run_number}.ecsv",format='ascii.ecsv',overwrite=kwargs.get('overwrite',False))
        #
        #table = Table(self.charge_lg)
        #table.meta["pixels_id"] = self._pixels_id
        #table.write(Path(path)/f"charge_lg_run{self.run_number}.ecsv",format='ascii.ecsv',overwrite=kwargs.get('overwrite',False))
        #
        #table = Table(self.peak_hg)
        #table.meta["pixels_id"] = self._pixels_id
        #table.write(Path(path)/f"peak_hg_run{self.run_number}.ecsv",format='ascii.ecsv',overwrite=kwargs.get('overwrite',False))
        #
        #table = Table(self.peak_lg)
        #table.meta["pixels_id"] = self._pixels_id
        #table.write(Path(path)/f"peak_lg_run{self.run_number}.ecsv",format='ascii.ecsv',overwrite=kwargs.get('overwrite',False))

        hdr = fits.Header()
        hdr['RUN'] = self._run_number
        hdr['NEVENTS'] = self.nevents
        hdr['NPIXELS'] = self.npixels
        hdr['COMMENT'] = f"The charge containeur for run {self._run_number} with {self._method} method : primary is the pixels id, then you can find HG charge, LG charge, HG peak and LG peak, 2 last HDU are composed of event properties and trigger patern"

        primary_hdu = fits.PrimaryHDU(self.pixels_id,header=hdr)
        charge_hg_hdu = fits.ImageHDU(self.charge_hg,name = "HG charge")
        charge_lg_hdu = fits.ImageHDU(self.charge_lg,name = "LG charge")
        peak_hg_hdu = fits.ImageHDU(self.peak_hg, name = 'HG peak time')
        peak_lg_hdu = fits.ImageHDU(self.peak_lg, name = 'LG peak time')

        col1 = fits.Column(array = self.event_id, name = "event_id", format = '1I')
        col2 = fits.Column(array = self.event_type, name = "event_type", format = '1I')
        col3 = fits.Column(array = self.ucts_timestamp, name = "ucts_timestamp", format = '1K')
        col4 = fits.Column(array = self.ucts_busy_counter, name = "ucts_busy_counter", format = '1I')
        col5 = fits.Column(array = self.ucts_event_counter, name = "ucts_event_counter", format = '1I')
        col6 = fits.Column(array = self.multiplicity, name = "multiplicity", format = '1I')

        coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6])
        event_properties = fits.BinTableHDU.from_columns(coldefs, name = 'event properties')

        col1 = fits.Column(array = self.trig_pattern_all, name = "trig_pattern_all", format = f'{4 * self.CAMERA.n_pixels}L',dim = f'({self.CAMERA.n_pixels},4)')
        col2 = fits.Column(array = self.trig_pattern, name = "trig_pattern", format = f'{self.CAMERA.n_pixels}L')
        coldefs = fits.ColDefs([col1, col2])
        trigger_patern = fits.BinTableHDU.from_columns(coldefs, name = 'trigger patern')

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
        """load ChargeContainer from FITS file previously written with ChargeContainer.write() method  
        
        Args:
            path (str): path of the FITS file
            run_number (int) : the run number

        Returns:
            ChargeContainer: ChargeContainer instance
        """
        if kwargs.get("explicit_filename",False) : 
            filename = kwargs.get("explicit_filename")
            log.info(f"loading {filename}")
        else : 
            log.info(f"loading in {path} run number {run_number}")
            filename = Path(path)/f"charge_run{run_number}.fits"
        
        with fits.open(filename) as hdul :
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
        """compute charge from waveforms 

        Args:
            waveformContainer (WaveformsContainer): the waveforms
            channel (int): channel you want to compute charges
            method (str, optional): ctapipe Image Extractor method method. Defaults to "FullWaveformSum".

        Raises:
            ArgumentError: extraction method unknown
            ArgumentError: channel unknown

        Returns:
            output of the extractor called on waveforms
        """
        
        if not(method in list_ctapipe_charge_extractor or method in list_nectarchain_charge_extractor) :
            raise ArgumentError(f"method must be in {list_ctapipe_charge_extractor}")

        extractor_kwargs = {}
        for key in eval(method).class_own_traits().keys() :
            if key in kwargs.keys() :
                extractor_kwargs[key] = kwargs[key]

        if "apply_integration_correction" in eval(method).class_own_traits().keys() : #to change the default behavior of ctapipe extractor
            extractor_kwargs["apply_integration_correction"] = kwargs.get("apply_integration_correction",False)

        log.debug(f"Extracting charges with method {method} and extractor_kwargs {extractor_kwargs}")
        ImageExtractor = eval(method)(waveformContainer.subarray,**extractor_kwargs)

        if channel == constants.HIGH_GAIN:
            out = np.array([CtaPipeExtractor.get_image_peak_time(ImageExtractor(waveformContainer.wfs_hg[i],waveformContainer.TEL_ID,channel,waveformContainer.broken_pixels_hg)) for i in range(len(waveformContainer.wfs_hg))]).transpose(1,0,2)
            return out[0],out[1]
        elif channel == constants.LOW_GAIN:
            out = np.array([CtaPipeExtractor.get_image_peak_time(ImageExtractor(waveformContainer.wfs_lg[i],waveformContainer.TEL_ID,channel,waveformContainer.broken_pixels_lg)) for i in range(len(waveformContainer.wfs_lg))]).transpose(1,0,2)
            return out[0],out[1]
        else :
            raise ArgumentError(f"channel must be {constants.LOW_GAIN} or {constants.HIGH_GAIN}")

    def histo_hg(self,n_bins : int = 1000,autoscale : bool = True) -> np.ndarray:
        """method to compute histogram of HG channel
        Numba is used to compute histograms in vectorized way

        Args:
            n_bins (int, optional): number of bins in charge (ADC counts). Defaults to 1000.
            autoscale (bool, optional): auto detect number of bins by pixels (bin witdh = 1 ADC). Defaults to True.

        Returns:
            np.ndarray: masked array of charge histograms (histo,charge)
        """
        mask_broken_pix = np.array((self.charge_hg == self.charge_hg.mean(axis = 0)).mean(axis=0),dtype = bool)
        log.debug(f"there are {mask_broken_pix.sum()} broken pixels (charge stays at same level for each events)")
        
        if autoscale : 
            all_range = np.arange(np.uint16(np.min(self.charge_hg.T[~mask_broken_pix].T)) - 0.5,np.uint16(np.max(self.charge_hg.T[~mask_broken_pix].T)) + 1.5,1)
            #hist_ma = ma.masked_array(np.zeros((self.charge_hg.shape[1],all_range.shape[0]),dtype = np.uint16), mask=np.zeros((self.charge_hg.shape[1],all_range.shape[0]),dtype = bool))
            charge_ma = ma.masked_array((all_range.reshape(all_range.shape[0],1) @ np.ones((1,self.charge_hg.shape[1]))).T, mask=np.zeros((self.charge_hg.shape[1],all_range.shape[0]),dtype = bool))

            broxen_pixels_mask = np.array([mask_broken_pix for i in range(charge_ma.shape[1])]).T
            #hist_ma.mask = new_data_mask.T
            start = time.time()
            _mask, hist_ma_data = make_histo(self.charge_hg.T, all_range, mask_broken_pix)#, charge_ma.data, charge_ma.mask, hist_ma.data, hist_ma.mask)
            charge_ma.mask = np.logical_or(_mask,broxen_pixels_mask)
            hist_ma =  ma.masked_array(hist_ma_data,mask = charge_ma.mask)
            log.debug(f"histogram hg computation time : {time.time() - start} sec")          
            
            return ma.masked_array((hist_ma,charge_ma))
            
        else : 
            hist = np.array([np.histogram(self.charge_hg.T[i],bins=n_bins)[0] for i in range(self.charge_hg.shape[1])])
            charge = np.array([np.histogram(self.charge_hg.T[i],bins=n_bins)[1] for i in range(self.charge_hg.shape[1])])
            charge_edges = np.array([np.mean(charge.T[i:i+2],axis = 0) for i in range(charge.shape[1]-1)]).T
            
            return np.array((hist,charge_edges))

    def histo_lg(self,n_bins: int = 1000,autoscale : bool = True) -> np.ndarray:
        """method to compute histogram of LG channel
        Numba is used to compute histograms in vectorized way

        Args:
            n_bins (int, optional): number of bins in charge (ADC counts). Defaults to 1000.
            autoscale (bool, optional): auto detect number of bins by pixels (bin witdh = 1 ADC). Defaults to True.

        Returns:
            np.ndarray: masked array of charge histograms (histo,charge)
        """
        mask_broken_pix = np.array((self.charge_lg == self.charge_lg.mean(axis = 0)).mean(axis=0),dtype = bool)
        log.debug(f"there are {mask_broken_pix.sum()} broken pixels (charge stays at same level for each events)")
        
        if autoscale : 
            all_range = np.arange(np.uint16(np.min(self.charge_lg.T[~mask_broken_pix].T)) - 0.5,np.uint16(np.max(self.charge_lg.T[~mask_broken_pix].T)) + 1.5,1)
            charge_ma = ma.masked_array((all_range.reshape(all_range.shape[0],1) @ np.ones((1,self.charge_lg.shape[1]))).T, mask=np.zeros((self.charge_lg.shape[1],all_range.shape[0]),dtype = bool))

            broxen_pixels_mask = np.array([mask_broken_pix for i in range(charge_ma.shape[1])]).T
            #hist_ma.mask = new_data_mask.T
            start = time.time()
            _mask, hist_ma_data = make_histo(self.charge_lg.T, all_range, mask_broken_pix)#, charge_ma.data, charge_ma.mask, hist_ma.data, hist_ma.mask)
            charge_ma.mask = np.logical_or(_mask,broxen_pixels_mask)
            hist_ma =  ma.masked_array(hist_ma_data,mask = charge_ma.mask)
            log.debug(f"histogram lg computation time : {time.time() - start} sec")  

            return ma.masked_array((hist_ma,charge_ma))

        else : 
            hist = np.array([np.histogram(self.charge_lg.T[i],bins=n_bins)[0] for i in range(self.charge_lg.shape[1])])
            charge = np.array([np.histogram(self.charge_lg.T[i],bins=n_bins)[1] for i in range(self.charge_lg.shape[1])])
            charge_edges = np.array([np.mean(charge.T[i:i+2],axis = 0) for i in range(charge.shape[1]-1)]).T

            return np.array((hist,charge_edges))

    def select_charge_hg(self,pixel_id : np.ndarray) : 
        """method to extract charge HG from a list of pixel ids 
        The output is the charge HG with a shape following the size of the input pixel_id argument
        Pixel in pixel_id which are not present in the ChargeContaineur pixels_id are skipped 

        Args:
            pixel_id (np.ndarray): array of pixel ids you want to extract the charge 

        Returns:
            (np.ndarray): charge array in the order of specified pixel_id
        """
        mask_contain_pixels_id = np.array([pixel in self.pixels_id for pixel in pixel_id],dtype = bool)
        for pixel in pixel_id[~mask_contain_pixels_id] : log.warning(f"You asked for pixel_id {pixel} but it is not present in this ChargeContainer, skip this one")
        return np.array([self.charge_hg.T[np.where(self.pixels_id == pixel)[0][0]] for pixel in pixel_id[mask_contain_pixels_id]]).T

    def select_charge_lg(self,pixel_id : np.ndarray) : 
        """method to extract charge LG from a list of pixel ids 
        The output is the charge LG with a shape following the size of the input pixel_id argument
        Pixel in pixel_id which are not present in the ChargeContaineur pixels_id are skipped 

        Args:
            pixel_id (np.ndarray): array of pixel ids you want to extract the charge 

        Returns:
            (np.ndarray): charge array in the order of specified pixel_id
        """
        mask_contain_pixels_id = np.array([pixel in self.pixels_id for pixel in pixel_id],dtype = bool)
        for pixel in pixel_id[~mask_contain_pixels_id] : log.warning(f"You asked for pixel_id {pixel} but it is not present in this ChargeContainer, skip this one")
        return np.array([self.charge_lg.T[np.where(self.pixels_id == pixel)[0][0]] for pixel in pixel_id[mask_contain_pixels_id]]).T

    def sort(self, method = 'event_id') : 
        if method == 'event_id' :
            log.info('sorting ChargeContaineur with event_id')
            index = np.argsort(self.event_id)
            self.ucts_timestamp = self.ucts_timestamp[index]
            self.ucts_busy_counter = self.ucts_busy_counter[index]
            self.ucts_event_counter = self.ucts_event_counter[index]
            self.event_type = self.event_type[index]
            self.event_id = self.event_id[index] 
            self.trig_pattern_all = self.trig_pattern_all[index]
            self.charge_hg = self.charge_hg[index] 
            self.charge_lg = self.charge_lg[index] 
            self.peak_hg = self.peak_hg[index]
            self.peak_lg = self.peak_lg[index]
        else : 
            raise ArgumentError(f"{method} is not a valid method for sorting")

    @property
    def run_number(self) : return self._run_number

    @property
    def pixels_id(self) : return self._pixels_id

    @property
    def npixels(self) : return self._npixels

    @property
    def nevents(self) : return self._nevents

    @property
    def method(self) : return self._method


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
        """create ChargeContainers from waveformContainers

        Args:
            waveformContainers (WaveformsContainers)

        Returns:
            chargeContainers (ChargeContainers)
        """
        chargeContainers = cls()
        for i in range(waveformContainers.nWaveformsContainer) : 
            chargeContainers.append(ChargeContainer.from_waveforms(waveformContainers.waveformsContainer[i],**kwargs))
        return chargeContainers

    def write(self,path : str, **kwargs) : 
        """write each ChargeContainer in ChargeContainers on disk 

        Args:
            path (str): path where data are saved
        """
        for i in range(self.__nChargeContainer) : 
            self.chargeContainers[i].write(path,suffix = f"{i:04d}" ,**kwargs)

    @staticmethod
    def from_file(path : Path,run_number : int,**kwargs) :
        """load ChargeContainers from FITS file previously written with ChargeContainers.write() method 
        This method will search all the fits files corresponding to {path}/charge_run{run_number}_*.fits scheme
        
        Args:
            path (str): path with name of the FITS file without .fits extension
            run_number (int) : the run number

        Returns:
            ChargeContainers: ChargeContainers instance
        """
        log.info(f"loading from {path}/charge_run{run_number}_*.fits")
        files = glob.glob(f"{path}/charge_run{run_number}_*.fits")
        if len(files) == 0 : 
            e = FileNotFoundError(f"no files found corresponding to {path}_*.fits")
            log.error(e)
            raise e
        else : 
            cls = ChargeContainers.__new__(ChargeContainers)
            cls.chargeContainers = []
            cls.__nchargeContainers = len(files)
            for file in files : 
                cls.chargeContainers.append(ChargeContainer.from_file(path,run_number,explicit_filename = file))
            return cls

    @property
    def nChargeContainer(self) : 
        """getter giving the number of chargeContainer into the ChargeContainers instance

        Returns:
            int: the number of chargeContainer
        """
        return self.__nChargeContainer

    @property
    def nevents(self) : 
        """number of events into the whole ChargesContainers

        Returns:
            int: number of events
        """
        return np.sum([self.chargeContainers[i].nevents for i in range(self.__nChargeContainer)])

    def append(self,chargeContainer : ChargeContainer) :
        """method to stack a ChargeContainer into the ChargeContainers

        Args:
            chargeContainer (ChargeContainer): the data to be stacked into
        """
        self.chargeContainers.append(chargeContainer)
        self.__nChargeContainer += 1


    def merge(self) -> ChargeContainer : 
        """method to merge a ChargeContainers into one single ChargeContainer

        Returns:
            ChargeContainer: the merged object
        """
        cls  = ChargeContainer.__new__(ChargeContainer)
        cls.charge_hg = np.concatenate([chargecontainer.charge_hg for chargecontainer in self.chargeContainers],axis = 0) 
        cls.charge_lg = np.concatenate([chargecontainer.charge_lg for chargecontainer in self.chargeContainers],axis = 0) 
        cls.peak_hg = np.concatenate([chargecontainer.peak_hg for chargecontainer in self.chargeContainers],axis = 0) 
        cls.peak_lg = np.concatenate([chargecontainer.peak_lg for chargecontainer in self.chargeContainers],axis = 0) 

        if np.all([chargecontainer.run_number == self.chargeContainers[0].run_number for chargecontainer in self.chargeContainers]) : 
            cls._run_number = self.chargeContainers[0].run_number
        if np.all([chargecontainer.pixels_id == self.chargeContainers[0].pixels_id for chargecontainer in self.chargeContainers]) : 
            cls._pixels_id = self.chargeContainers[0].pixels_id
        if np.all([chargecontainer.method == self.chargeContainers[0].method for chargecontainer in self.chargeContainers]):
            cls._method = self.chargeContainers[0].method
        cls._nevents = np.sum([chargecontainer.nevents for chargecontainer in self.chargeContainers])
        if np.all([chargecontainer.npixels == self.chargeContainers[0].npixels for chargecontainer in self.chargeContainers]):
            cls._npixels = self.chargeContainers[0].npixels


        cls.ucts_timestamp = np.concatenate([chargecontainer.ucts_timestamp for chargecontainer in self.chargeContainers ])
        cls.ucts_busy_counter = np.concatenate([chargecontainer.ucts_busy_counter for chargecontainer in self.chargeContainers ])
        cls.ucts_event_counter = np.concatenate([chargecontainer.ucts_event_counter for chargecontainer in self.chargeContainers ])
        cls.event_type = np.concatenate([chargecontainer.event_type for chargecontainer in self.chargeContainers ])
        cls.event_id = np.concatenate([chargecontainer.event_id for chargecontainer in self.chargeContainers ])
        cls.trig_pattern_all = np.concatenate([chargecontainer.trig_pattern_all for chargecontainer in self.chargeContainers ],axis = 0)
        
        cls.sort()

        return cls