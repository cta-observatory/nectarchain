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

    def WriteAllResults(self, path, DICT):
        data2 = Table()
        data1 = Table()
        data0 = Table()
        data = Table()
        hdu, hdu0, hdu1, hdu2 = None, None, None, None
        hdulist = fits.HDUList()
        for i, j in DICT.items():
            if (i == "Results_TriggerStatistics"):
                for n2, m2 in j.items():
                    data2[n2] = m2
                hdu2 = fits.BinTableHDU(data2)
                hdu2.name = "Trigger"

            elif (i == "Results_MeanWaveForms_HighGain") or (
                i == "Results_MeanWaveForms_LowGain"
            ):
                for n1, m1 in j.items():
                    data1[n1] = m1
                hdu1 = fits.BinTableHDU(data1)
                hdu1.name = "MWF"    

            elif (i == "Results_PixelTimeline_HighGain") or (i == "Results_PixelTimeline_LowGain"):
                for n0, m0 in j.items():
                    data0[n0] = m0 
                hdu0 = fits.BinTableHDU(data0)
                hdu0.name = "BPX"

            else:
                for n, m in j.items():
                    data[n] = m
                hdu = fits.BinTableHDU(data)
                hdu.name = "Camera"
        if hdu2:
            hdulist.append(hdu2)
        else:
            print("No trigger statistics requests")
        if hdu1:
            hdulist.append(hdu1)
        else:
            print("No MWF studies requests")
        if:
            hdulist.append(hdu0)
        else:
            print("No Pixel Timeline studies requests")
        if:
            hdulist.append(hdu)
        else:
            print("No Camera studies requests")

        
        FileName = path + '_Results.fits'
        print(FileName)
        hdulist.writeto(FileName, overwrite=True)
        return None
