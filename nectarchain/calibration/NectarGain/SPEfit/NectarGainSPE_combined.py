from ..utils.error import DifferentPixelsID

from .NectarGainSPE_singlerun import NectarGainSPESingle, NectarGainSPESingleSignalStd
from nectarchain.calibration.container import ChargeContainer


class NectarGainSPESingleHHV(NectarGainSPESingle):
    """class to perform fit of the 1400V signal and pedestal"""


class NectarGainSPESingleCombined(NectarGainSPESingle):
    """class to perform fit of the 1400V and 1000V signal and pedestal"""

class NectarGainSPECombinedNoPed(NectarGainSPESingle):
    """class to perform fit of the 1400V and 1000V signal"""
    
    def __init__(self,signalHHV : ChargeContainer, signal : ChargeContainer, parameters_file = None, parameters_file_HHV = None,**kwargs) :
        self.nectarGainHHV = NectarGainSPESingleSignalStd(signalHHV)
        self.nectarGain = NectarGainSPESingleSignalStd(signal)


    def Chi2(self,pixel : int):
        def _Chi2(pp,resolution,mean,meanHHV,n,pedestal,pedestalHHV,pedestalWidth,luminosity) :            
            return self.nectarGainHHV._Chi2(pp,resolution,meanHHV,n,pedestalHHV,pedestalWidth,luminosity) + self.nectarGain._Chi2(pp,resolution,mean,n,pedestal,pedestalWidth,luminosity)
        return _Chi2


    @staticmethod
    def from_prefitted_HHV(SPEfitHHV, signalHV : ChargeContainer, parameters_file_HV = None) :
        pass

