import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

import os
from ctapipe.containers import Container,Field
from ctapipe.io import TableWriter,TableLoader
from ctapipe.instrument.subarray import SubarrayDescription
import numpy as np

from pathlib import Path

from enum import Enum

from tqdm import tqdm

from astropy.io import fits
from astropy.table import QTable,Column,Table
import astropy.units as u
from abc import ABC

class WaveformsContainer(Container):
    """
    A container that holds information about waveforms from a specific run.

    Attributes:
        run_number (int): The run number associated with the waveforms.
        nevents (int): The number of events.
        npixels (int): The number of pixels.
        nsamples (int): The number of samples in the waveforms.
        subarray (str): The name of the subarray.
        camera (str): The name of the camera.
        pixels_id (np.ndarray): An array of pixel IDs.
        wfs_hg (np.ndarray): An array of high gain waveforms.
        wfs_lg (np.ndarray): An array of low gain waveforms.
        broken_pixels_hg (np.ndarray): An array of high gain broken pixels.
        broken_pixels_lg (np.ndarray): An array of low gain broken pixels.
        ucts_timestamp (np.ndarray): An array of events' UCTS timestamps.
        ucts_busy_counter (np.ndarray): An array of UCTS busy counters.
        ucts_event_counter (np.ndarray): An array of UCTS event counters.
        event_type (np.ndarray): An array of trigger event types.
        event_id (np.ndarray): An array of event IDs.
        trig_pattern_all (np.ndarray): An array of trigger patterns.
        trig_pattern (np.ndarray): An array of reduced trigger patterns.
        multiplicity (np.ndarray): An array of events' multiplicities.
    """

    run_number = Field(
        type=int,
        description="run number associated to the waveforms",
    )
    nevents = Field(
        type=int,
        description="number of events",
    )
    npixels = Field(
        type=int,
        description="number of effective pixels",
    )
    nsamples = Field(
        type=int,
        description="number of samples in the waveforms",
    )
    subarray = Field(
        type=SubarrayDescription,
        description="The subarray  description"
    )
    camera = Field(
        type=str,
        description="camera name",
    )
    pixels_id = Field(
        type=np.ndarray,
        description="pixel ids"
    )
    wfs_hg = Field(
        type=np.ndarray,
        description="high gain waveforms"
    )
    wfs_lg = Field(
        type=np.ndarray,
        description="low gain waveforms"
    )
    broken_pixels_hg = Field(
        type=np.ndarray,
        description="high gain broken pixels"
    )
    broken_pixels_lg = Field(
        type=np.ndarray,
        description="low gain broken pixels"
    )
    ucts_timestamp = Field(
        type=np.ndarray,
        description="events ucts timestamp"
    )
    ucts_busy_counter = Field(
        type=np.ndarray,
        description="ucts busy counter"
    )
    ucts_event_counter = Field(
        type=np.ndarray,
        description="ucts event counter"
    )
    event_type = Field(
        type=np.ndarray,
        description="trigger event type"
    )
    event_id = Field(
        type=np.ndarray,
        description="event ids"
    )
    trig_pattern_all = Field(
        type=np.ndarray,
        description="trigger pattern"
    )
    trig_pattern = Field(
        type=np.ndarray,
        description="reduced trigger pattern"
    )
    multiplicity = Field(
        type=np.ndarray,
        description="events multiplicity"
    )

    
class WaveformsContainerIO(ABC) : 

    @staticmethod
    def write(path : str, containers : WaveformsContainer, **kwargs):
        '''method to write in an output FITS file the WaveformsContainer. Two files are created, one FITS representing the data
        and one HDF5 file representing the subarray configuration

        Args:
            path (str): the directory where you want to save data
        '''
        suffix = kwargs.get("suffix","")
        if suffix != "" : suffix = f"_{suffix}"

        log.info(f"saving in {path}")
        os.makedirs(path,exist_ok = True)

        hdr = fits.Header()
        hdr['RUN'] = containers.run_number
        hdr['NEVENTS'] = containers.nevents
        hdr['NPIXELS'] = containers.npixels
        hdr['NSAMPLES'] = containers.nsamples
        hdr['SUBARRAY'] = containers.subarray.name
        hdr['CAMERA'] = containers.camera


        containers.subarray.to_hdf(f"{Path(path)}/subarray_run{containers.run_number}{suffix}.hdf5",overwrite=kwargs.get('overwrite',False))



        hdr['COMMENT'] = f"The waveforms containeur for run {containers.run_number} : primary is the pixels id, 2nd HDU : high gain waveforms, 3rd HDU : low gain waveforms, 4th HDU :  event properties and 5th HDU trigger paterns."

        primary_hdu = fits.PrimaryHDU(containers.pixels_id,header=hdr)

        wfs_hg_hdu = fits.ImageHDU(containers.wfs_hg,name = "HG Waveforms")
        wfs_lg_hdu = fits.ImageHDU(containers.wfs_lg,name = "LG Waveforms")

        col1 = fits.Column(array = containers.broken_pixels_hg, name = "HG broken pixels", format = f'{containers.broken_pixels_hg.shape[1]}L')
        col2 = fits.Column(array = containers.broken_pixels_lg, name = "LG broken pixels", format = f'{containers.broken_pixels_lg.shape[1]}L')
        coldefs = fits.ColDefs([col1, col2])
        broken_pixels = fits.BinTableHDU.from_columns(coldefs,name = 'trigger patern')

        col1 = fits.Column(array = containers.event_id, name = "event_id", format = '1J')
        col2 = fits.Column(array = containers.event_type, name = "event_type", format = '1I')
        col3 = fits.Column(array = containers.ucts_timestamp, name = "ucts_timestamp", format = '1K')
        col4 = fits.Column(array = containers.ucts_busy_counter, name = "ucts_busy_counter", format = '1J')
        col5 = fits.Column(array = containers.ucts_event_counter, name = "ucts_event_counter", format = '1J')
        col6 = fits.Column(array = containers.multiplicity, name = "multiplicity", format = '1I')

        coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6])
        event_properties = fits.BinTableHDU.from_columns(coldefs,name = 'event properties')

        col1 = fits.Column(array = containers.trig_pattern_all, name = "trig_pattern_all", format = f'{4 * containers.trig_pattern_all.shape[1]}L',dim = f'({containers.trig_pattern_all.shape[1]},4)')
        col2 = fits.Column(array = containers.trig_pattern, name = "trig_pattern", format = f'{containers.trig_pattern_all.shape[1]}L')
        coldefs = fits.ColDefs([col1, col2])
        trigger_patern = fits.BinTableHDU.from_columns(coldefs,name = 'trigger patern')

        hdul = fits.HDUList([primary_hdu, wfs_hg_hdu, wfs_lg_hdu,broken_pixels,event_properties,trigger_patern])
        try : 
            hdul.writeto(Path(path)/f"waveforms_run{containers.run_number}{suffix}.fits",overwrite=kwargs.get('overwrite',False))
            log.info(f"run saved in {Path(path)}/waveforms_run{containers.run_number}{suffix}.fits")
        except OSError as e : 
            log.warning(e)
        except Exception as e :
            log.error(e,exc_info = True)
            raise e
        
    @staticmethod
    def load(path : str) : 
        '''load WaveformsContainer from FITS file previously written with WaveformsContainer.write() method
        Note : 2 files are loaded, the FITS one representing the waveforms data and a HDF5 file representing the subarray configuration. 
        This second file has to be next to the FITS file.  
        
        Args:
            path (str): path of the FITS file

        Returns:
            WaveformsContainer: WaveformsContainer instance
        '''
        log.info(f"loading from {path}")
        with fits.open(Path(path)) as hdul : 
            containers = WaveformsContainer()

            containers.run_number = hdul[0].header['RUN'] 
            containers.nevents = hdul[0].header['NEVENTS'] 
            containers.npixels = hdul[0].header['NPIXELS'] 
            containers.nsamples = hdul[0].header['NSAMPLES'] 
            containers.camera = hdul[0].header['CAMERA'] 


            containers.subarray = SubarrayDescription.from_hdf(Path(path.replace('waveforms_','subarray_').replace('fits','hdf5')))


            containers.pixels_id = hdul[0].data
            containers.wfs_hg = hdul[1].data
            containers.wfs_lg = hdul[2].data
            
            broken_pixels = hdul[3].data
            containers.broken_pixels_hg = broken_pixels["HG broken pixels"]
            containers.broken_pixels_lg = broken_pixels["LG broken pixels"]



            table_prop = hdul[4].data
            containers.event_id = table_prop["event_id"]
            containers.event_type = table_prop["event_type"]
            containers.ucts_timestamp = table_prop["ucts_timestamp"]
            containers.ucts_busy_counter = table_prop["ucts_busy_counter"]
            containers.ucts_event_counter = table_prop["ucts_event_counter"]
            containers.multiplicity = table_prop["multiplicity"]

            table_trigger = hdul[5].data
            containers.trig_pattern_all = table_trigger["trig_pattern_all"]
            containers.trig_pattern = table_trigger["trig_pattern"]

        return containers