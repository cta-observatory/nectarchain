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
    """
    A container that holds information about charges from a specific run.

    Fields:
      charges_hg (np.ndarray): An array of high gain charges.
      charges_lg (np.ndarray): An array of low gain charges.
      peak_hg (np.ndarray): An array of high gain peak time.
      peak_lg (np.ndarray): An array of low gain peak time.
      method (str): The charge extraction method used.
    """

    charges_hg = Field(
        type=np.ndarray,
        description='The high gain charges'
    )
    charges_lg = Field(
        type=np.ndarray,
        description='The low gain charges'
    )
    peak_hg = Field(
        type=np.ndarray,
        description='The high gain peak time'
    )
    peak_lg = Field(
        type=np.ndarray,
        description='The low gain peak time'
    )
    method = Field(
        type=str,
        description='The charge extraction method used'
    )

class ChargesContainerIO(ABC) : 
    """
    The `ChargesContainerIO` class provides methods for writing and loading `ChargesContainer` instances to/from FITS files.

    Example Usage:
        # Writing a ChargesContainer instance to a FITS file
        chargesContainer = ChargesContainer()
        ChargesContainerIO.write(path, chargesContainer)

        # Loading a ChargesContainer instance from a FITS file
        chargesContainer = ChargesContainerIO.load(path, run_number)

    Main functionalities:
    - Writing a `ChargesContainer` instance to a FITS file.
    - Loading a `ChargesContainer` instance from a FITS file.

    Methods:
    - `write(path: Path, container: ChargesContainer, **kwargs) -> None`: Writes a `ChargesContainer` instance to a FITS file. The method takes a file path and the `ChargesContainer` instance as input. Additional keyword arguments can be provided to customize the file name and overwrite behavior.
    - `load(path: Path, run_number: int, **kwargs) -> ChargesContainer`: Loads a `ChargesContainer` instance from a FITS file. The method takes a file path and the run number as input. Additional keyword arguments can be provided to specify an explicit file name.

    Fields:
    The `ChargesContainerIO` class does not have any fields.
    """
    @staticmethod
    def write(path : Path, container : ChargesContainer,**kwargs) -> None: 
        """Write a ChargesContainer instance to a FITS file.
        Args:
            path (str): The directory where the FITS file will be saved.
            container (ChargesContainer): The ChargesContainer instance to be written to the FITS file.
            **kwargs: Additional keyword arguments for customization.
        Keyword Args:
            suffix (str): A suffix to be added to the file name (default: "").
            overwrite (bool): Whether to overwrite the file if it already exists (default: False).
        Returns:
            None: This method does not return any value.
        Raises:
            OSError: If there is an error while writing the FITS file.
            Exception: If there is any other exception during the writing process.
        Example:
            chargesContainer = ChargesContainer()
            ChargesContainerIO.write(path, chargesContainer, suffix="v1", overwrite=True)
        """
        suffix = kwargs.get("suffix","")
        if suffix != "" : suffix = f"_{suffix}"
        log.info(f"saving in {path}")
        os.makedirs(path,exist_ok = True)
        hdr = fits.Header()
        hdr['RUN'] = container.run_number
        hdr['NEVENTS'] = container.nevents
        hdr['NPIXELS'] = container.npixels
        hdr['METHOD'] = container.method
        hdr['CAMERA'] = container.camera
        hdr['COMMENT'] = f"The charge containeur for run {container.run_number} with {container.method} method : primary is the pixels id, then you can find HG charge, LG charge, HG peak and LG peak, 2 last HDU are composed of event properties and trigger patern"
        
        primary_hdu = fits.PrimaryHDU(container.pixels_id,header=hdr)
        charge_hg_hdu = fits.ImageHDU(container.charges_hg,name = "HG charge")
        charge_lg_hdu = fits.ImageHDU(container.charges_lg,name = "LG charge")
        peak_hg_hdu = fits.ImageHDU(container.peak_hg, name = 'HG peak time')
        peak_lg_hdu = fits.ImageHDU(container.peak_lg, name = 'LG peak time')

        col1 = fits.Column(array = container.broken_pixels_hg, name = "HG broken pixels", format = f'{container.broken_pixels_hg.shape[1]}L')
        col2 = fits.Column(array = container.broken_pixels_lg, name = "LG broken pixels", format = f'{container.broken_pixels_lg.shape[1]}L')
        coldefs = fits.ColDefs([col1, col2])
        broken_pixels = fits.BinTableHDU.from_columns(coldefs,name = 'trigger patern')

        col1 = fits.Column(array = container.event_id, name = "event_id", format = '1I')
        col2 = fits.Column(array = container.event_type, name = "event_type", format = '1I')
        col3 = fits.Column(array = container.ucts_timestamp, name = "ucts_timestamp", format = '1K')
        col4 = fits.Column(array = container.ucts_busy_counter, name = "ucts_busy_counter", format = '1I')
        col5 = fits.Column(array = container.ucts_event_counter, name = "ucts_event_counter", format = '1I')
        col6 = fits.Column(array = container.multiplicity, name = "multiplicity", format = '1I')
        coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6])
        event_properties = fits.BinTableHDU.from_columns(coldefs, name = 'event properties')
        
        col1 = fits.Column(array = container.trig_pattern_all, name = "trig_pattern_all", format = f'{4 * container.trig_pattern_all.shape[1]}L',dim = f'({ container.trig_pattern_all.shape[1]},4)')
        col2 = fits.Column(array = container.trig_pattern, name = "trig_pattern", format = f'{container.trig_pattern_all.shape[1]}L')
        coldefs = fits.ColDefs([col1, col2])
        trigger_patern = fits.BinTableHDU.from_columns(coldefs, name = 'trigger patern')
        
        hdul = fits.HDUList([primary_hdu, charge_hg_hdu, charge_lg_hdu,peak_hg_hdu,peak_lg_hdu,broken_pixels,event_properties,trigger_patern])
        try : 
            hdul.writeto(Path(path)/f"charge_run{container.run_number}{suffix}.fits",overwrite=kwargs.get('overwrite',False))
            log.info(f"charge saved in {Path(path)}/charge_run{container.run_number}{suffix}.fits")
        except OSError as e : 
            log.warning(e)
        except Exception as e :
            log.error(e,exc_info = True)
            raise e

    @staticmethod
    def load(path: str, run_number: int, **kwargs) -> ChargesContainer:
        """Load a ChargesContainer instance from a FITS file.
        This method opens a FITS file and retrieves the necessary data to create a ChargesContainer instance.
        The FITS file should have been previously written using the write method of the ChargesContainerIO class.
        Args:
            path (str): The path of the FITS file.
            run_number (int): The run number.
            **kwargs: Additional keyword arguments.
                explicit_filename (str): If provided, the explicit filename to load.
        Returns:
            ChargesContainer: The loaded ChargesContainer instance.
        Example:
            chargesContainer = ChargesContainerIO.load(path, run_number)
        """
        if kwargs.get("explicit_filename", False):
            filename = kwargs.get("explicit_filename")
            log.info(f"Loading {filename}")
        else:
            log.info(f"Loading in {path} run number {run_number}")
            filename = Path(path) / f"charge_run{run_number}.fits"

        with fits.open(filename) as hdul:
            container = ChargesContainer()
            container.run_number = hdul[0].header['RUN']
            container.nevents = hdul[0].header['NEVENTS']
            container.npixels = hdul[0].header['NPIXELS']
            container.method = hdul[0].header['METHOD']
            container.camera = hdul[0].header['CAMERA']
            container.pixels_id = hdul[0].data
            container.charges_hg = hdul[1].data
            container.charges_lg = hdul[2].data
            container.peak_hg = hdul[3].data
            container.peak_lg = hdul[4].data
            broken_pixels = hdul[5].data
            container.broken_pixels_hg = broken_pixels["HG broken pixels"]
            container.broken_pixels_lg = broken_pixels["LG broken pixels"]
            table_prop = hdul[6].data
            container.event_id = table_prop["event_id"]
            container.event_type = table_prop["event_type"]
            container.ucts_timestamp = table_prop["ucts_timestamp"]
            container.ucts_busy_counter = table_prop["ucts_busy_counter"]
            container.ucts_event_counter = table_prop["ucts_event_counter"]
            container.multiplicity = table_prop["multiplicity"]
            table_trigger = hdul[7].data
            container.trig_pattern_all = table_trigger["trig_pattern_all"]
            container.trig_pattern = table_trigger["trig_pattern"]
        return container
