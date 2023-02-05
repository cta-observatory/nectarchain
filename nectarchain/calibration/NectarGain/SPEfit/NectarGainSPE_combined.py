import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from tqdm import tqdm
import numpy as np
from astropy.table import Column
import astropy.units as u
from datetime import date
import random
import matplotlib.pyplot as plt
import os
from pathlib import Path

from iminuit import Minuit

from ..utils.error import DifferentPixelsID

from .NectarGainSPE_singlerun import NectarGainSPESingle, NectarGainSPESingleSignalStd
from nectarchain.calibration.container import ChargeContainer
from .NectarGainSPE import NectarGainSPE
from .utils import UtilsMinuit,Gain,MPE2,weight_gaussian

__all__ = ["NectarGainSPECombinedNoPed"]

class NectarGainSPESingleHHV(NectarGainSPESingle):
    """class to perform fit of the 1400V signal and pedestal"""

class NectarGainSPESingleCombined(NectarGainSPESingle):
    """class to perform fit of the 1400V and 1000V signal and pedestal"""

class NectarGainSPECombinedNoPed(NectarGainSPE):
    """class to perform fit of the 1400V and 1000V signal"""
    __parameters_file = 'parameters_signal_combined.yaml'
    
    def __init__(self,signalHHV : ChargeContainer, signal : ChargeContainer, same_luminosity : bool = True, parameters_file = None, parameters_file_HHV = None,**kwargs) :
        super().__init__(**kwargs)
        
        self.nectarGainHHV = NectarGainSPESingleSignalStd(signalHHV, parameters_file = parameters_file_HHV)
        self.nectarGain = NectarGainSPESingleSignalStd(signal, parameters_file = parameters_file)

        if self.nectarGainHHV.npixels != self.nectarGain.npixels : 
            e = Exception("not same number of pixels in HHV and std run")
            log.error(e,exc_info=True)
            raise e
        else : 
            self.__npixels = self.nectarGainHHV.npixels

        if not((self.nectarGainHHV.pixels_id == self.nectarGain.pixels_id).all()) : 
            e = Exception("not same pixels_id in HHV and std run")
            log.error(e,exc_info=True)
            raise e
        else : 
            self.__pixels_id = self.nectarGainHHV.pixels_id

        self.__gain = np.empty((self.__npixels,3))
        self.__gainHHV = np.empty((self.__npixels,3))

        #shared parameters
        self.__pp = self.nectarGainHHV.pp
        self.__resolution = self.nectarGainHHV.resolution
        self.__n = self.nectarGainHHV.n
        self.__pedestalWidth = self.nectarGainHHV.pedestalWidth
        self._parameters.append(self.__pp)
        self._parameters.append(self.__resolution)
        self._parameters.append(self.__n)
        self._parameters.append(self.__pedestalWidth)
        if same_luminosity :
            self.__luminosity = self.nectarGainHHV.luminosity
            self._parameters.append(self.__luminosity)

        #others
        if not(same_luminosity) :
            self.__luminosity = self.nectarGain.luminosity
            self.__luminosityHHV = self.nectarGain.luminosity
            self.__luminosityHHV.name = "luminosityHHV"
            self._parameters.append(self.__luminosity)
            self._parameters.append(self.__luminosityHHV)

        self.__meanHHV = self.nectarGainHHV.mean
        self.__meanHHV.name = "meanHHV"
        self.__mean = self.nectarGain.mean
        self.__pedestalHHV = self.nectarGainHHV.pedestal
        self.__pedestalHHV.name = "pedestalHHV"
        self.__pedestal = self.nectarGain.pedestal
        self._parameters.append(self.__meanHHV)
        self._parameters.append(self.__mean)
        self._parameters.append(self.__pedestalHHV)
        self._parameters.append(self.__pedestal)

        self.read_param_from_yaml(NectarGainSPECombinedNoPed.__parameters_file)

        log.info(self._parameters)

        self._make_minuit_parameters()

        self.create_output_table()

    def create_output_table(self) :
        self._output_table.meta['npixel'] = self.npixels
        self._output_table.meta['comments'] = f'Produced with NectarGain, Credit : CTA NectarCam {date.today().strftime("%B %d, %Y")}'

        self._output_table.add_column(Column(np.zeros((self.npixels),dtype = bool),"is_valid",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(self.__pixels_id,"pixel",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels),dtype = np.float64),"gain",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels,2),dtype = np.float64),"gain_error",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels),dtype = np.float64),"gainHHV",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels,2),dtype = np.float64),"gainHHV_error",unit = u.dimensionless_unscaled))

        for parameter in self._parameters.parameters : 
            self._output_table.add_column(Column(np.empty((self.npixels),dtype = np.float64),parameter.name,unit = parameter.unit))
            self._output_table.add_column(Column(np.empty((self.npixels),dtype = np.float64),f'{parameter.name}_error',unit = parameter.unit))

    def Chi2(self,pixel : int):
        def _Chi2(resolution,mean,meanHHV,pedestal,pedestalHHV,pedestalWidth,luminosity) :            
            return self.nectarGainHHV.Chi2(pixel)(resolution,meanHHV,pedestalHHV,pedestalWidth,luminosity) + self.nectarGain.Chi2(pixel)(resolution,mean,pedestal,pedestalWidth,luminosity)
        return _Chi2

    def save(self,path,**kwargs) :
        path = Path(path)
        os.makedirs(path,exist_ok = True)
        log.info(f'data saved in {path}')
        self._output_table.write(f"{path}/output_table.ecsv", format='ascii.ecsv',overwrite = kwargs.get("overwrite",False))
    
    def run(self,pixel : int = None,**kwargs):
        if pixel is None : 
            for i in tqdm(range(self.npixels)) :
                if self.nectarGain.charge.mask[i].all() or self.nectarGainHHV.charge.mask[i].all() : 
                    log.info(f'do not run fit on pixel {i} (pixel_id = {self.__pixels_id[i]}), it seems to be a broken pixel from charge computation')
                else  :
                    log.info(f"running SPE fit for pixel {i} (pixel_id = {self.__pixels_id[i]})")
                    self._run_obs(i,**kwargs)
        else : 
            if not(isinstance(pixel,np.ndarray)) :
                pixels = np.asarray([pixel],dtype = np.int16)
            else : 
                pixels = pixel
            for pixel in tqdm(pixels) : 
                if pixel >= self.npixels : 
                    e = Exception(f"pixel must be < {self.npixels}")
                    log.error(e,exc_info=True)
                    raise e
                else :
                    if self.nectarGain.charge.mask[i].all() or self.nectarGainHHV.charge.mask[i].all() : 
                        log.info(f'do not run fit on pixel {i} (pixel_id = {self.__pixels_id[i]}), it seems to be a broken pixel from charge computation')
                    else  :
                        log.info(f"running SPE fit for pixel {pixel} (pixel_id = {self.__pixels_id[pixel]})")
                        self._run_obs(pixel,**kwargs)
        return 0

    def _run_obs(self,pixel,**kwargs) : 
        self._update_parameters_prefit(pixel)
        fit = Minuit(self.Chi2(pixel),**self._minuitParameters['values'])
        UtilsMinuit.set_minuit_parameters_limits_and_errors(fit,self._minuitParameters)
        log.info(f"Initial value of Likelihood = {self.Chi2(pixel)(**self._minuitParameters['values'])}")
        log.info(f"Initial parameters value : {fit.values}")
        #log.debug(self.Chi2(pixel)(0.5,500,14600,50,1))

        if log.getEffectiveLevel() == logging.ERROR :
            fit.print_level = 0
        if log.getEffectiveLevel() == logging.WARNING :
            fit.print_level = 1
        if log.getEffectiveLevel() == logging.INFO :
            fit.print_level = 2
        if log.getEffectiveLevel() == logging.DEBUG :
            fit.print_level = 3
            
        fit.strategy = 2
        fit.throw_nan = True
        try : 
            log.info('migrad execution')
            fit.migrad(ncall=super(NectarGainSPECombinedNoPed,self)._Ncall)
            fit.hesse()
            valid = fit.valid
        except RuntimeError as e : 
            log.warning(e,exc_info = True)
            log.info("change method : re-try with simplex")
            fit.reset()
            fit.throw_nan = True
            try :
                log.info('simplex execution')
                fit.simplex(ncall=super(NectarGainSPECombinedNoPed,self)._Ncall)
                fit.hesse()
                valid = fit.valid
            except Exception as e :
                log.error(e,exc_info = True)
                log.warning(f"skip pixel {pixel} (pixel_id : {self.pixels_id})")
                valid = False

        except Exception as e :
            log.error(e,exc_info = True)
            raise e

        if valid : 
            log.info(f"fitted value : {fit.values}")
            log.info(f"fitted errors : {fit.errors}")
            self._update_parameters_postfit(fit)

            self.__gain[pixel,0] = Gain(self.__pp.value,self.__resolution.value,self.__mean.value,self.__n.value)
            stat_gain = np.array([Gain(self.__pp.value,random.gauss(self.__resolution.value, self.__resolution.error),random.gauss(self.__mean.value, self.__mean.error),self.__n.value) for i in range(1000)])
            self.__gain[pixel,1] = self.__gain[pixel,0] - np.quantile(stat_gain,0.16)
            self.__gain[pixel,2] = np.quantile(stat_gain,0.84) - self.__gain[pixel,0]

            self.__gainHHV[pixel,0] = Gain(self.__pp.value,self.__resolution.value,self.__meanHHV.value,self.__n.value)
            stat_gain = np.array([Gain(self.__pp.value,random.gauss(self.__resolution.value, self.__resolution.error),random.gauss(self.__meanHHV.value, self.__meanHHV.error),self.__n.value) for i in range(1000)])
            self.__gainHHV[pixel,1] = self.__gainHHV[pixel,0] - np.quantile(stat_gain,0.16)
            self.__gainHHV[pixel,2] = np.quantile(stat_gain,0.84) - self.__gainHHV[pixel,0]

            self.fill_table(pixel,valid)
            log.info(f"Reconstructed gain is {self.__gain[pixel,0] - self.__gain[pixel,1]:.2f} < {self.__gain[pixel,0]:.2f} < {self.__gain[pixel,0] + self.__gain[pixel,2]:.2f}")
            self._output_table['gain'][pixel] = self.__gain[pixel,0] 
            self._output_table['gain_error'][pixel][0] = self.__gain[pixel,1] 
            self._output_table['gain_error'][pixel][1] = self.__gain[pixel,2] 
            log.info(f"Reconstructed gainHHV is {self.__gainHHV[pixel,0] - self.__gainHHV[pixel,1]:.2f} < {self.__gainHHV[pixel,0]:.2f} < {self.__gainHHV[pixel,0] + self.__gainHHV[pixel,2]:.2f}")
            self._output_table['gainHHV'][pixel] = self.__gainHHV[pixel,0] 
            self._output_table['gainHHV_error'][pixel][0] = self.__gainHHV[pixel,1] 
            self._output_table['gainHHV_error'][pixel][1] = self.__gainHHV[pixel,2] 

            if kwargs.get('figpath',0) != 0 :
                fig,ax = plt.subplots(1,2,figsize=(16, 6))
                ax[0].errorbar(self.nectarGain.charge[pixel],self.nectarGain.histo[pixel],np.sqrt(self.nectarGain.histo[pixel]),zorder=0,fmt=".",label = "data")
                ax[0].plot(self.nectarGain.charge[pixel],
                    np.trapz(self.nectarGain.histo[pixel],self.nectarGain.charge[pixel])*MPE2(self.nectarGain.charge[pixel],self.__pp.value,self.__resolution.value,self.__mean.value,self.__n.value,self.__pedestal.value,self.__pedestalWidth.value,self.__luminosity.value),
                    zorder=1,
                    linewidth=2,
                    label = f"SPE model fit \n gain : {self.__gain[pixel,0] - self.__gain[pixel,1]:.2f} < {self.__gain[pixel,0]:.2f} < {self.__gain[pixel,0] + self.__gain[pixel,2]:.2f} ADC/pe")
                ax[0].set_xlabel("Charge (ADC)", size=15)
                ax[0].set_ylabel("Events", size=15)
                ax[0].set_title(f"SPE fit pixel : {pixel} (pixel id : {self.pixels_id[pixel]})")
                ax[0].legend(fontsize=12)

                ax[1].errorbar(self.nectarGainHHV.charge[pixel],self.nectarGainHHV.histo[pixel],np.sqrt(self.nectarGainHHV.histo[pixel]),zorder=0,fmt=".",label = "data")
                ax[1].plot(self.nectarGainHHV.charge[pixel],
                    np.trapz(self.nectarGainHHV.histo[pixel],self.nectarGainHHV.charge[pixel])*MPE2(self.nectarGainHHV.charge[pixel],self.__pp.value,self.__resolution.value,self.__meanHHV.value,self.__n.value,self.__pedestalHHV.value,self.__pedestalWidth.value,self.__luminosity.value),
                    zorder=1,
                    linewidth=2,
                    label = f"SPE model fit \n gainHHV : {self.__gainHHV[pixel,0] - self.__gainHHV[pixel,1]:.2f} < {self.__gainHHV[pixel,0]:.2f} < {self.__gainHHV[pixel,0] + self.__gainHHV[pixel,2]:.2f} ADC/pe")
                ax[1].set_xlabel("Charge (ADC)", size=15)
                ax[1].set_ylabel("Events", size=15)
                ax[1].set_title(f"SPE fit pixel : {pixel} (pixel id : {self.pixels_id[pixel]})")
                ax[1].legend(fontsize=12)

                os.makedirs(kwargs.get('figpath'),exist_ok = True)
                fig.savefig(f"{kwargs.get('figpath')}/fit_SPE_pixel{pixel}.pdf")
                fig.clf()
                plt.close(fig)
                del fig,ax
        else : 
            log.warning(f"fit {pixel} is not valid")
            self.fill_table(pixel,valid)

    def _update_parameters_prefit(self,pixel) : 

        coeff,var_matrix =  NectarGainSPE._get_parameters_gaussian_fit(self.nectarGain.charge, self.nectarGain.histo, pixel)
        self.__pedestal.value = coeff[1]
        self.__pedestal.min = coeff[1] - coeff[2]
        self.__pedestal.max = coeff[1] + coeff[2]
        self._minuitParameters['values']['pedestal'] = self.__pedestal.value
        self._minuitParameters['limit_pedestal'] = (self.__pedestal.min,self.__pedestal.max)

        coeff,var_matrix =  NectarGainSPE._get_parameters_gaussian_fit(self.nectarGain.charge, self.nectarGain.histo, pixel,"_HHV")
        self.__pedestalHHV.value = coeff[1]
        self.__pedestalHHV.min = coeff[1] - coeff[2]
        self.__pedestalHHV.max = coeff[1] + coeff[2]
        self._minuitParameters['values']['pedestalHHV'] = self.__pedestalHHV.value
        self._minuitParameters['limit_pedestalHHV'] = (self.__pedestalHHV.min,self.__pedestalHHV.max)

    def NG_Likelihood_Chi2(cls,**kwargs) : pass

    #run properties
    @property
    def pixels_id(self)  :  return self.__pixels_id  
    @property
    def npixels(self) : return self.__npixels

