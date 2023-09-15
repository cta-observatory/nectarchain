import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

import os

from ctapipe.instrument.subarray import SubarrayDescription
import numpy as np

from pathlib import Path

from enum import Enum

from tqdm import tqdm
from ctapipe.containers import Field
from astropy.io import fits
from astropy.table import QTable,Column,Table
import astropy.units as u
from abc import ABC

from .core import ArrayDataContainer

class WaveformsContainer(ArrayDataContainer):
    """
    A container that holds information about waveforms from a specific run.

    Fields:
        nsamples (int): The number of samples in the waveforms.
        subarray (SubarrayDescription): The subarray description instance.
        wfs_hg (np.ndarray): An array of high gain waveforms.
        wfs_lg (np.ndarray): An array of low gain waveforms.
    """

    nsamples = Field(
        type=int,
        description="number of samples in the waveforms",
    )
    subarray = Field(
        type=SubarrayDescription,
        description="The subarray  description"
    )
    wfs_hg = Field(
        type=np.ndarray,
        description="high gain waveforms"
    )
    wfs_lg = Field(
        type=np.ndarray,
        description="low gain waveforms"
    )

    

class WaveformsContainerIO(ABC) : 
    """
    The `WaveformsContainerIO` class provides methods for writing and loading `WaveformsContainer` instances to/from FITS files. It also includes a method for writing the subarray configuration to an HDF5 file.

    Example Usage:
        # Writing a WaveformsContainer instance to a FITS file
        container = WaveformsContainer()
        # ... populate the container with data ...
        WaveformsContainerIO.write('/path/to/output', container)

        # Loading a WaveformsContainer instance from a FITS file
        container = WaveformsContainerIO.load('/path/to/input.fits')

    Main functionalities:
    - Writing a `WaveformsContainer` instance to a FITS file, including the waveforms data and metadata.
    - Loading a `WaveformsContainer` instance from a FITS file, including the waveforms data and metadata.
    - Writing the subarray configuration to an HDF5 file.

    Methods:
    - `write(path: str, containers: WaveformsContainer, **kwargs) -> None`: Writes a `WaveformsContainer` instance to a FITS file. The method also creates an HDF5 file representing the subarray configuration.
    - `load(path: str) -> WaveformsContainer`: Loads a `WaveformsContainer` instance from a FITS file. The method also loads the subarray configuration from the corresponding HDF5 file.

    Fields:
    None.
    """

    @staticmethod
    def write(path : str, containers : WaveformsContainer, **kwargs) -> None:
        '''Write the WaveformsContainer data to an output FITS file.

        This method creates two files: one FITS file representing the waveform data and one HDF5 file representing the subarray configuration.

        Args:
            path (str): The directory where you want to save the data.
            containers (WaveformsContainer): The WaveformsContainer instance containing the data to be saved.
            **kwargs: Additional keyword arguments for customization (optional).

        Keyword Args:
            suffix (str, optional): A suffix to append to the output file names (default is '').
            overwrite (bool, optional): If True, overwrite the output files if they already exist (default is False).
        Returns:
            None: This method does not return any value.
        Raises:
            OSError: If there is an error while writing the FITS file.
            Exception: If there is any other exception during the writing process.
        Example:
            waveformsContainer = WaveformsContainer()
            WaveformsContainerIO.write(path, waveformsContainer, suffix="v1", overwrite=True)
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
    def load(path : str, run_number : int, **kwargs) -> WaveformsContainer: 
        '''Load a WaveformsContainer from a FITS file previously written with WaveformsContainerIO.write() method.

        Note: Two files are loaded—the FITS file representing the waveform data and an HDF5 file representing the subarray configuration. 
        The HDF5 file should be located next to the FITS file.

        Args:
            path (str): The path to the FITS file containing the waveform data.
            **kwargs: Additional keyword arguments.
                explicit_filename (str): If provided, the explicit filename to load.
        Returns:
            WaveformsContainer: A WaveformsContainer instance loaded from the specified file.
        Example:
            waveformsContainer = WaveformsContainerIO.load(path, run_number)
        '''
        if kwargs.get("explicit_filename", False):
            filename = kwargs.get("explicit_filename")
            log.info(f"Loading {filename}")
        else:
            log.info(f"Loading in {path} run number {run_number}")
            filename = Path(path) / f"waveforms_run{run_number}.fits"

        log.info(f"loading from {path}")
        with fits.open(filename) as hdul : 
            containers = WaveformsContainer()

            containers.run_number = hdul[0].header['RUN'] 
            containers.nevents = hdul[0].header['NEVENTS'] 
            containers.npixels = hdul[0].header['NPIXELS'] 
            containers.nsamples = hdul[0].header['NSAMPLES'] 
            containers.camera = hdul[0].header['CAMERA'] 


            containers.subarray = SubarrayDescription.from_hdf(Path(filename._str.replace('waveforms_','subarray_').replace('fits','hdf5')))


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