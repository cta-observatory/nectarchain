import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

import numpy as np
from ctapipe.containers import Field

from .core import ArrayDataContainer

class ChargesContainer(ArrayDataContainer):

    charge_hg = Field( 
        type = np.ndarray,
        description = 'The high gain charge')
    charge_lg = Field( 
        type = np.ndarray,
        description = 'The low gain charge')
    peak_hg = Field( 
        type = np.ndarray,
        description = 'The high gain peak time')
    peak_lg = Field( 
        type = np.ndarray,
        description = 'The low gain peak time')
    _method = Field( 
        type = str,
        description = 'The charge extraction method')


'''

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
'''