from astropy.io import fits
from astropy.table import Table

__all__ = ["DQMSummary"]


class DQMSummary:
    def __init__(self):
        print("Processor 0")

    def DefineForRun(self, reader1):
        for i, evt1 in enumerate(reader1):
            self.FirstReader = reader1
            self.Samp = len(evt1.r0.tel[0].waveform[0][0])
            self.Pix = len(evt1.r0.tel[0].waveform[0])
        return self.Pix, self.Samp

    def ConfigureForRun(self):
        print("Processor 1")

    def ProcessEvent(self, evt, noped):
        print("Processor 2")

    def FinishRun(self, M, M_ped, counter_evt, counter_ped):
        print("Processor 3")

    def GetResults(self):
        print("Processor 4")

    def PlotResults(
        self, name, FigPath, k, M, M_ped, Mean_M_overPix, Mean_M_ped_overPix
    ):
        print("Processor 5")

    @staticmethod
    def _create_hdu(name, content):
        data = Table()
        data[name] = content
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
                    print(f"DEBUG JPL: Got error {e}, skipping {name}")
                    pass

        FileName = path + "_Results.fits"
        print(FileName)
        hdulist.writeto(FileName, overwrite=True)
        hdulist.info()
