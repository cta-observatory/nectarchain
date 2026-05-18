import logging

import numpy as np
from astropy.io import fits
from astropy.table import Table

__all__ = ["DQMSummary"]

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class DQMSummary:
    def __init__(self, r0=False):
        log.debug("Processor 0")
        self.Samp = None
        self.Pix = None
        self.r0 = r0
        self.tel_id = None

    def define_for_run(self, reader1):
        self.tel_id = reader1.subarray.tel_ids[0]

        # we just need to access the first event
        evt1 = next(iter(reader1))
        if self.r0:
            self.Samp = evt1.r0.tel[self.tel_id].waveform.shape[-1]
            self.Pix = evt1.r0.tel[self.tel_id].waveform.shape[-2]
        else:
            self.Samp = evt1.r1.tel[self.tel_id].waveform.shape[-1]
            self.Pix = evt1.r1.tel[self.tel_id].waveform.shape[-2]
        return self.Pix, self.Samp

    def configure_for_run(self):
        log.debug("Processor 1")

    def process_event(self, evt, noped):
        log.debug("Processor 2")

    def finish_run(self, M, M_ped, counter_evt, counter_ped):
        log.debug("Processor 3")

    def get_results(self):
        log.debug("Processor 4")

    def plot_results(
        self, name, fig_path, k, M, M_ped, Mean_M_overPix, Mean_M_ped_overPix
    ):
        log.debug("Processor 5")

    @staticmethod
    def _create_hdu(name, content):
        # Convert to numpy array
        arr = np.asarray(content)

        # Ensure consistent dtype (convert to float64 or int64 as appropriate)
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)
        elif np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int64)
        elif arr.dtype == object:
            # Try to infer a numeric type for object arrays
            try:
                arr = arr.astype(np.float64)
            except (ValueError, TypeError):
                # If conversion fails, flatten and use BinTableHDU
                arr_1d = arr.flatten()
                data = Table()
                data[name] = arr_1d
                hdu = fits.BinTableHDU(data)
                hdu.name = name
                return hdu

        # Choose HDU type based on dimensionality
        if arr.ndim <= 1:
            # Use BinTableHDU for 0D and 1D arrays
            if arr.ndim == 0:
                arr = arr.reshape(1)  # Convert scalar to 1D
            data = Table()
            data[name] = arr
            hdu = fits.BinTableHDU(data)
        else:
            # Use ImageHDU for multi-dimensional arrays
            hdu = fits.ImageHDU(arr)

        hdu.name = name
        return hdu

    def write_all_results(self, path, DICT):
        hdulist = fits.HDUList()
        for i, j in DICT.items():
            for name, content in j.items():
                try:
                    hdu = self._create_hdu(name, content)
                    hdulist.append(hdu)
                except TypeError as e:
                    log.warning(
                        f"Caught {type(e).__name__}, skipping {name}. Details: {e}"
                    )
                    pass

        output_filename = path + "_Results.fits"
        log.info(f"Saving DQM results in {output_filename}")
        hdulist.writeto(output_filename, overwrite=True)
        hdulist.info()
