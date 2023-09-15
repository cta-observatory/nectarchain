import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

import numpy as np
from ctapipe.containers import Field
from abc import ABC
import os
from pathlib import Path
from astropy.io import fits

from .core import ArrayDataContainer

class ChargesContainer(ArrayDataContainer):

    charges_hg = Field( 
        type = np.ndarray,
        description = 'The high gain charges')
    charges_lg = Field( 
        type = np.ndarray,
        description = 'The low gain charges')
    peak_hg = Field( 
        type = np.ndarray,
        description = 'The high gain peak time')
    peak_lg = Field( 
        type = np.ndarray,
        description = 'The low gain peak time')
    method = Field( 
        type = str,
        description = 'The charge extraction method used')


class ChargesContainerIO(ABC) : 
    def write(path : Path, containers : ChargesContainer,**kwargs) : 
        """method to write in an output FITS file the ChargeContainer.

        Args:
            path (str): the directory where you want to save data
        """
        suffix = kwargs.get("suffix","")
        if suffix != "" : suffix = f"_{suffix}"

        log.info(f"saving in {path}")
        os.makedirs(path,exist_ok = True)

        hdr = fits.Header()
        hdr['RUN'] = containers.run_number
        hdr['NEVENTS'] = containers.nevents
        hdr['NPIXELS'] = containers.npixels
        hdr['METHOD'] = containers.method
        hdr['CAMERA'] = containers.camera
        

        hdr['COMMENT'] = f"The charge containeur for run {containers.run_number} with {containers.method} method : primary is the pixels id, then you can find HG charge, LG charge, HG peak and LG peak, 2 last HDU are composed of event properties and trigger patern"

        primary_hdu = fits.PrimaryHDU(containers.pixels_id,header=hdr)
        charge_hg_hdu = fits.ImageHDU(containers.charges_hg,name = "HG charge")
        charge_lg_hdu = fits.ImageHDU(containers.charges_lg,name = "LG charge")
        peak_hg_hdu = fits.ImageHDU(containers.peak_hg, name = 'HG peak time')
        peak_lg_hdu = fits.ImageHDU(containers.peak_lg, name = 'LG peak time')

        col1 = fits.Column(array = containers.broken_pixels_hg, name = "HG broken pixels", format = f'{containers.broken_pixels_hg.shape[1]}L')
        col2 = fits.Column(array = containers.broken_pixels_lg, name = "LG broken pixels", format = f'{containers.broken_pixels_lg.shape[1]}L')
        coldefs = fits.ColDefs([col1, col2])
        broken_pixels = fits.BinTableHDU.from_columns(coldefs,name = 'trigger patern')

        col1 = fits.Column(array = containers.event_id, name = "event_id", format = '1I')
        col2 = fits.Column(array = containers.event_type, name = "event_type", format = '1I')
        col3 = fits.Column(array = containers.ucts_timestamp, name = "ucts_timestamp", format = '1K')
        col4 = fits.Column(array = containers.ucts_busy_counter, name = "ucts_busy_counter", format = '1I')
        col5 = fits.Column(array = containers.ucts_event_counter, name = "ucts_event_counter", format = '1I')
        col6 = fits.Column(array = containers.multiplicity, name = "multiplicity", format = '1I')

        coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6])
        event_properties = fits.BinTableHDU.from_columns(coldefs, name = 'event properties')

        col1 = fits.Column(array = containers.trig_pattern_all, name = "trig_pattern_all", format = f'{4 * containers.trig_pattern_all.shape[1]}L',dim = f'({ containers.trig_pattern_all.shape[1]},4)')
        col2 = fits.Column(array = containers.trig_pattern, name = "trig_pattern", format = f'{containers.trig_pattern_all.shape[1]}L')
        coldefs = fits.ColDefs([col1, col2])
        trigger_patern = fits.BinTableHDU.from_columns(coldefs, name = 'trigger patern')

        hdul = fits.HDUList([primary_hdu, charge_hg_hdu, charge_lg_hdu,peak_hg_hdu,peak_lg_hdu,broken_pixels,event_properties,trigger_patern])
        try : 
            hdul.writeto(Path(path)/f"charge_run{containers.run_number}{suffix}.fits",overwrite=kwargs.get('overwrite',False))
            log.info(f"charge saved in {Path(path)}/charge_run{containers.run_number}{suffix}.fits")
        except OSError as e : 
            log.warning(e)
        except Exception as e :
            log.error(e,exc_info = True)
            raise e

    def load(path : Path,run_number : int,**kwargs) : 
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
            containers = ChargesContainer()
            containers.run_number = hdul[0].header['RUN'] 
            containers.nevents = hdul[0].header['NEVENTS'] 
            containers.npixels = hdul[0].header['NPIXELS'] 
            containers.method = hdul[0].header['METHOD'] 
            containers.camera = hdul[0].header['CAMERA'] 

            containers.pixels_id = hdul[0].data
            containers.charges_hg = hdul[1].data
            containers.charges_lg = hdul[2].data
            containers.peak_hg = hdul[3].data
            containers.peak_lg = hdul[4].data
            
            broken_pixels = hdul[5].data
            containers.broken_pixels_hg = broken_pixels["HG broken pixels"]
            containers.broken_pixels_lg = broken_pixels["LG broken pixels"]



            table_prop = hdul[6].data
            containers.event_id = table_prop["event_id"]
            containers.event_type = table_prop["event_type"]
            containers.ucts_timestamp = table_prop["ucts_timestamp"]
            containers.ucts_busy_counter = table_prop["ucts_busy_counter"]
            containers.ucts_event_counter = table_prop["ucts_event_counter"]
            containers.multiplicity = table_prop["multiplicity"]

            table_trigger = hdul[7].data
            containers.trig_pattern_all = table_trigger["trig_pattern_all"]
            containers.trig_pattern = table_trigger["trig_pattern"]
        return containers
