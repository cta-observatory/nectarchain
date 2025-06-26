import logging

import numpy as np
from astropy.io import fits
from astropy.table import Table

__all__ = ["DQMSummary"]

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class DQMSummary:
    def __init__(self):
        log.debug("Processor 0")

    def DefineForRun(self, reader1):
        for i, evt1 in enumerate(reader1):
            self.FirstReader = reader1
            self.Samp = len(evt1.r0.tel[0].waveform[0][0])
            self.Pix = len(evt1.r0.tel[0].waveform[0])
        return self.Pix, self.Samp

    def ConfigureForRun(self):
        log.debug("Processor 1")

    def ProcessEvent(self, evt, noped):
        log.debug("Processor 2")

    def FinishRun(self, M, M_ped, counter_evt, counter_ped):
        log.debug("Processor 3")

    def GetResults(self):
        log.debug("Processor 4")

    def PlotResults(
        self, name, FigPath, k, M, M_ped, Mean_M_overPix, Mean_M_ped_overPix
    ):
        log.debug("Processor 5")

    @staticmethod
    def _create_hdu(name, content):
        data = Table()
        try:
            data[name] = content
        except TypeError:
            try:
                data = Table(content)
            except ValueError:
                # We may have caught just a single float value, try to pack it into
                # the FITS output
                content = np.array([content])
                data = Table(content)
        hdu = fits.BinTableHDU(data)
        hdu.name = name
        return hdu

    def WriteAllResults(self, path, DICT):
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
