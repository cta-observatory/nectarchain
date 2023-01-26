import math
import numpy as np
from scipy import optimize, interpolate
from matplotlib import pyplot as plt
from scipy import signal
from scipy.special import gammainc
from iminuit import Minuit
import random
import astropy.units as u
from astropy.table import QTable,Column
import astropy.units as u
import yaml
import os
from datetime import date
from pathlib import Path
from tqdm import tqdm

import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

import copy

from .parameters import Parameters,Parameter
from ...container import ChargeContainer
from .utils import UtilsMinuit,MPE2,Gain,gaussian
from .NectarGainSPE import NectarGainSPE

from abc import ABC, abstractclassmethod, abstractmethod

__all__ = ["NectarGainSPESingleSignalStd","NectarGainSPESingleSignal","NectarGainSPESinglePed","NectarGainSPESingleSignalfromHHVFit"]



class NectarGainSPESingle(NectarGainSPE):
    _Ncall = 4000000

    def __init__(self,signal : ChargeContainer,**kwargs) : 
        log.info("initialisation of the SPE fit instance")
        super().__init__(**kwargs)

        histo = signal.histo_hg(autoscale = True)
        #access data
        self.__charge = histo[1]
        self.__histo = histo[0]
        self.__mask_fitted_pixel = np.zeros((self.__charge.shape[0]),dtype = bool)
        self.__pixels_id = signal.pixels_id

        self.__pedestal = Parameter(name = "pedestal",
                                value = (np.min(self.__charge) + np.sum(self.__charge * self.__histo)/(np.sum(self.__histo)))/2,
                                min = np.min(self.__charge),
                                max = np.sum(self.__charge*self.__histo)/np.sum(self.__histo),
                                unit = u.dimensionless_unscaled)
        

        self._parameters.append(self.__pedestal)
        
    
    def create_output_table(self) :
        self._output_table.meta['npixel'] = self.npixels
        self._output_table.meta['comments'] = f'Produced with NectarGain, Credit : CTA NectarCam {date.today().strftime("%B %d, %Y")}'

        self._output_table.add_column(Column(np.zeros((self.npixels),dtype = bool),"is_valid",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(self.__pixels_id,"pixel",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels),dtype = np.float64),"gain",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels,2),dtype = np.float64),"gain_error",unit = u.dimensionless_unscaled))

        for parameter in self._parameters.parameters : 
            self._output_table.add_column(Column(np.empty((self.npixels),dtype = np.float64),parameter.name,unit = parameter.unit))
            self._output_table.add_column(Column(np.empty((self.npixels),dtype = np.float64),f'{parameter.name}_error',unit = parameter.unit))


        



        
    def run(self,pixel : int = None,multiproc = False, **kwargs):
        
        if pixel is None : 
                for i in tqdm(range(self.npixels)) :
                    if self.charge.mask[i].all() : 
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
                    if self.charge.mask[pixel].all() : 
                        log.info(f'do not run fit on pixel {i} (pixel_id = {self.__pixels_id[i]}), it seems to be a broken pixel from charge computation')
                    else : 
                        log.info(f"running SPE fit for pixel {pixel} (pixel_id = {self.__pixels_id[pixel]})")
                        self._run_obs(pixel,**kwargs)
        return 0

    def save(self,path,**kwargs) : 
        path = Path(path)
        os.makedirs(path,exist_ok = True)
        log.info(f'data saved in {path}')
        self._output_table.write(f"{path}/output_table.ecsv", format='ascii.ecsv',overwrite = kwargs.get("overwrite",False))

    def _update_parameters_postfit(self,m : Minuit) : 
        for i,name in enumerate(m.parameters) : 
            tmp = self._parameters[name]
            if tmp != [] : 
                tmp.value = m.values[i]
                tmp.error = m.errors[i]

    def _update_parameters_prefit(self,pixel) : 
        self.__pedestal.value = (np.min(self.__charge[pixel]) + np.sum(self.__charge[pixel] * self.__histo[pixel])/(np.sum(self.__histo[pixel])))/2
        self.__pedestal.min = np.min(self.__charge[pixel])
        self.__pedestal.max = np.sum(self.__charge[pixel]*self.__histo[pixel])/np.sum(self.__histo[pixel])
        self._minuitParameters['values']['pedestal'] = self.__pedestal.value
        self._minuitParameters['limit_pedestal'] = (self.__pedestal.min,self.__pedestal.max)


    @abstractclassmethod
    def NG_Likelihood_Chi2(cls,**kwargs) : pass


    @property
    def charge(self) : return self.__charge
    @property
    def histo(self) : return self.__histo

    @property
    def pedestal(self) : return self.__pedestal


    #run properties
    @property
    def pixels_id(self)  :  return self.__pixels_id  
    @property
    def npixels(self) : return self.__charge.shape[0]


class NectarGainSPESingleSignal(NectarGainSPESingle):
    """class to perform fit of the SPE signal with all free parameters"""
    __parameters_file = 'parameters_signal.yaml'
    #def __new__(cls) : 
    #    print("NectarGainSPESingleSignal is not instanciable")
    #    return 0
    def __init__(self,signal : ChargeContainer,parameters_file = None,**kwargs) : 
        self.__old_lum = None
        self.__old_ntotalPE = None
        super().__init__(signal,**kwargs)
        self.__gain = np.empty((self.charge.shape[0],3))
        #if parameters file is provided
        if parameters_file is None :
            parameters_file = NectarGainSPESingleSignal.__parameters_file
        with open(f"{os.path.dirname(os.path.abspath(__file__))}/{parameters_file}") as parameters :
            param = yaml.safe_load(parameters) 
            self.__pp = Parameter(name = "pp",
                                value = param["pp"]["value"],
                                min = param["pp"].get("min",np.nan),
                                max = param["pp"].get("max",np.nan),
                                unit = param["pp"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self.__pp)
            
            self.__luminosity = Parameter(name = "luminosity",
                                value = param["luminosity"]["value"],
                                min = param["luminosity"].get("min",np.nan),
                                max = param["luminosity"].get("max",np.nan),
                                unit = param["luminosity"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self.__luminosity)
            
            self.__resolution = Parameter(name = "resolution",
                                value = param["resolution"]["value"],
                                min = param["resolution"].get("min",np.nan),
                                max = param["resolution"].get("max",np.nan),
                                unit = param["resolution"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self.__resolution)
            
            self.__mean = Parameter(name = "mean",
                                value = param["mean"]["value"],
                                min = param["mean"].get("min",0),
                                max = param["mean"].get("max",np.nan),
                                unit = param["mean"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self.__mean)
            
            self.__n = Parameter(name = "n",
                                value = param["n"]["value"],
                                min = param["n"].get("min",np.nan),
                                max = param["n"].get("max",np.nan),
                                unit = param["n"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self.__n)
            
            self.__pedestalWidth = Parameter(name = "pedestalWidth",
                                value = param["pedestalWidth"]["value"],
                                min = param["pedestalWidth"].get("min",np.nan),
                                max = param["pedestalWidth"].get("max",np.nan),
                                unit = param["pedestalWidth"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self.__pedestalWidth)
        
        self._make_minuit_parameters()

        self.create_output_table()


        


    def _run_obs(self,pixel,prescan = False,**kwargs) : 
        self._update_parameters_prefit(pixel)
        fit = Minuit(self.Chi2(pixel),**self._minuitParameters['values'])
        UtilsMinuit.set_minuit_parameters_limits_and_errors(fit,self._minuitParameters)
        log.info(f"Initial value of Likelihood = {self.Chi2(pixel)(**self._minuitParameters['values'])}")
        log.info(f"Initial parameters value : {fit.values}")
        #log.debug(self.Chi2(pixel)(0.5,500,14600,50,1))

        fit.print_level = 3
        fit.strategy = 2
        fit.throw_nan = True
        if prescan : 
            log.info("let's do a fisrt brut force scan") 
            fit.scan()
        try : 
            log.info('migrad execution')
            fit.migrad(ncall=super(NectarGainSPESingleSignal,self)._Ncall)
            fit.hesse()
            valid = fit.valid
        except RuntimeError as e : 
            log.warning(e,exc_info = True)
            log.info("change method : re-try with simplex")
            fit.reset()
            fit.throw_nan = True
            if prescan : 
                log.info("let's do a fisrt brut force scan") 
                fit.scan()
            try :
                log.info('simplex execution')
                fit.simplex(ncall=super(NectarGainSPESingleSignal,self)._Ncall)
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

            self.fill_table(pixel,valid)
            log.info(f"Reconstructed gain is {self.__gain[pixel,0] - self.__gain[pixel,1]:.2f} < {self.__gain[pixel,0]:.2f} < {self.__gain[pixel,0] + self.__gain[pixel,2]:.2f}")
            self._output_table['gain'][pixel] = self.__gain[pixel,0] 
            self._output_table['gain_error'][pixel][0] = self.__gain[pixel,1] 
            self._output_table['gain_error'][pixel][1] = self.__gain[pixel,2] 



            if kwargs.get('figpath',0) != 0 :
                fig,ax = plt.subplots(1,1,figsize=(8, 6))
                ax.errorbar(self.charge[pixel],self.histo[pixel],np.sqrt(self.histo[pixel]),zorder=0,fmt=".",label = "data")
                ax.plot(self.charge[pixel],
                    np.trapz(self.histo[pixel],self.charge[pixel])*MPE2(self.charge[pixel],self.__pp.value,self.__resolution.value,self.__mean.value,self.__n.value,self.pedestal.value,self.__pedestalWidth.value,self.__luminosity.value),
                    zorder=1,
                    linewidth=2,
                    label = f"SPE model fit \n gain : {self.__gain[pixel,0] - self.__gain[pixel,1]:.2f} < {self.__gain[pixel,0]:.2f} < {self.__gain[pixel,0] + self.__gain[pixel,2]:.2f} ADC/pe")
                ax.set_xlabel("Charge (ADC)", size=15)
                ax.set_ylabel("Events", size=15)
                ax.set_title(f"SPE fit pixel : {pixel} (pixel id : {self.pixels_id[pixel]})")
                ax.legend(fontsize=15)
                os.makedirs(kwargs.get('figpath'),exist_ok = True)
                fig.savefig(f"{kwargs.get('figpath')}/fit_SPE_pixel{pixel}.pdf")
                fig.clf()
                del fig,ax
                plt.close('all')
        else : 
            log.warning(f"fit {pixel} is not valid")
            self.fill_table(pixel,valid)

    @classmethod
    def NG_Likelihood_Chi2(cls,pp,res,mu2,n,muped,sigped,lum,charge,histo,**kwargs):
        pdf = MPE2(charge,pp,res,mu2,n,muped,sigped,lum,**kwargs)
        #log.debug(f"pdf : {np.sum(pdf)}")
        Ntot = np.sum(histo)
        #log.debug(f'Ntot : {Ntot}')
        mask = histo > 0
        Lik = np.sum(((pdf*Ntot-histo)[mask])**2/histo[mask]) #2 times faster
        return Lik
    
    def Chi2(self,pixel : int):
        def _Chi2(pp,resolution,mean,n,pedestal,pedestalWidth,luminosity) :
            #assert not(np.isnan(pp) or np.isnan(resolution) or np.isnan(mean) or np.isnan(n) or np.isnan(pedestal) or np.isnan(pedestalWidth) or np.isnan(luminosity))
            if self.__old_lum != luminosity :
                for i in range(1000):
                    if (gammainc(i+1,luminosity) < 1e-5):
                        self.__old_ntotalPE = i
                        break
                self.__old_lum = luminosity
            kwargs = {"ntotalPE" : self.__old_ntotalPE}

            return self.NG_Likelihood_Chi2(pp,resolution,mean,n,pedestal,pedestalWidth,luminosity,self.charge[pixel],self.histo[pixel],**kwargs)
        return _Chi2
    
    #fit parameters
    @property
    def pp(self) : return self.__pp  
    @property
    def luminosity(self) : return self.__luminosity
    @property
    def mean(self) : return self.__mean      
    @property
    def n(self) : return self.__n
    @property
    def resolution(self) : return self.__resolution
    @property
    def pedestalWidth(self) : return self.__pedestalWidth
    @property
    def gain(self) : return self.__gain
    


    #intern parameters
    @property
    def _old_lum(self) : return self.__old_lum
    @_old_lum.setter
    def _old_lum(self,value) : self.__old_lum = value
    @property
    def _old_ntotalPE(self) : return self.__old_ntotalPE
    @_old_ntotalPE.setter
    def _old_ntotalPE(self,value) : self.__old_ntotalPE = value
    
    
    
class NectarGainSPESingleSignalStd(NectarGainSPESingleSignal):
    """class to perform fit of the SPE signal with n and pp fixed"""
    __parameters_file = 'parameters_signalStd.yaml'
    
    #to heavy
    #class Chi2() :
    #    def __init__(self,pixel : int,NectarGainSPESingleSignalStd) : 
    #        self._pixel = pixel
    #        self.NectarGainSPESingleSignalStd = NectarGainSPESingleSignalStd
    #    def __call__(self,resolution,mean,pedestal,pedestalWidth,luminosity):
    #        return super(NectarGainSPESingleSignalStd,self).NG_Likelihood_Chi2(self.NectarGainSPESingleSignalStd.pp.value,resolution,mean,self.NectarGainSPESingleSignalStd.n.value,pedestal,pedestalWidth,luminosity,self.NectarGainSPESingleSignalStd.charge[self._pixel],self.NectarGainSPESingleSignalStd.histo[self._pixel])
            


    def __init__(self,signal : ChargeContainer,parameters_file = None,**kwargs):
        if parameters_file is None : 
            parameters_file = NectarGainSPESingleSignalStd.__parameters_file
        super().__init__(signal,parameters_file,**kwargs)

        self.__fix_parameters()

        self._make_minuit_parameters()

    def __fix_parameters(self) : 
        """this method should be used to fix n and pp if this hypothesis is valid
        """
        log.info("updating parameters by fixing pp and n")
        self.pp.frozen = True
        self.n.frozen = True

    def Chi2(self,pixel : int):
        def _Chi2(resolution,mean,pedestal,pedestalWidth,luminosity) :
            if self._old_lum != luminosity :
                for i in range(1000):
                    if (gammainc(i+1,luminosity) < 1e-5):
                        self._old_ntotalPE = i
                        break
                self._old_lum = luminosity
            kwargs = {"ntotalPE" : self._old_ntotalPE}

            return self.NG_Likelihood_Chi2(self.pp.value,resolution,mean,self.n.value,pedestal,pedestalWidth,luminosity,self.charge[pixel],self.histo[pixel],**kwargs)
        return _Chi2


class NectarGainSPESingleSignalfromHHVFit(NectarGainSPESingleSignal):
    """class to perform fit of the SPE signal at nominal voltage from fitted data obtained with 1400V run
    Thus, n, pp and res are fixed"""
    __parameters_file = 'parameters_signal_fromHHVFit.yaml'

    def __init__(self,signal : ChargeContainer, nectarGainSPEresult, same_luminosity : bool = True, parameters_file = None,**kwargs):
        if parameters_file is None : 
            parameters_file = NectarGainSPESingleSignalfromHHVFit.__parameters_file
        super().__init__(signal,parameters_file,**kwargs)

        self.__fix_parameters(same_luminosity)

        self.__same_luminosity = same_luminosity

        self.__nectarGainSPEresult = QTable.read(nectarGainSPEresult,format = "ascii.ecsv")

        self._make_minuit_parameters()

    def __fix_parameters(self, same_luminosity : bool) : 
        """this method should be used to fix n, pp and res
        """
        log.info("updating parameters by fixing pp, n and res")
        self.pp.frozen = True
        self.n.frozen = True
        self.resolution.frozen = True
        if same_luminosity : 
            self.luminosity.frozen = True
        

    def Chi2(self,pixel : int):
        if self.__same_luminosity : 
            def _Chi2(mean,pedestal,pedestalWidth) :
                if self._old_lum != self.__nectarGainSPEresult[self.__pixel_index(pixel)]['luminosity'].value :
                    for i in range(1000):
                        if (gammainc(i+1,self.__nectarGainSPEresult[self.__pixel_index(pixel)]['luminosity'].value) < 1e-5):
                            self._old_ntotalPE = i
                            break
                    self._old_lum = self.__nectarGainSPEresult[self.__pixel_index(pixel)]['luminosity'].value
                kwargs = {"ntotalPE" : self._old_ntotalPE}

                return self.NG_Likelihood_Chi2(self.__nectarGainSPEresult[self.__pixel_index(pixel)]['pp'].value,self.__nectarGainSPEresult[self.__pixel_index(pixel)]['resolution'].value,mean,self.__nectarGainSPEresult[self.__pixel_index(pixel)]['n'].value,pedestal,pedestalWidth,self.__nectarGainSPEresult[self.__pixel_index(pixel)]['luminosity'],self.charge[pixel],self.histo[pixel],**kwargs)
            return _Chi2
        else : 
            def _Chi2(mean,pedestal,pedestalWidth,luminosity) :
                if self._old_lum != luminosity :
                    for i in range(1000):
                        if (gammainc(i+1,luminosity) < 1e-5):
                            self._old_ntotalPE = i
                            break
                    self._old_lum = luminosity
                kwargs = {"ntotalPE" : self._old_ntotalPE}
                return self.NG_Likelihood_Chi2(self.__nectarGainSPEresult[pixel]['pp'].value,self.__nectarGainSPEresult[pixel]['resolution'].value,mean,self.__nectarGainSPEresult[pixel]['n'].value,pedestal,pedestalWidth,luminosity,self.charge[pixel],self.histo[pixel],**kwargs)
            return _Chi2

    def _run_obs(self,pixel,prescan = False,**kwargs) : 
        if self.__nectarGainSPEresult[pixel]['is_valid'].value : 
            super()._run_obs(pixel,prescan,**kwargs)
        else :
            log.warning(f"fit {pixel} is not valid")
            self.fill_table(pixel,False)
    
    
    def _update_parameters_prefit(self,pixel) : 
        coeff,var_matrix =  NectarGainSPE._get_parameters_gaussian_fit(self.charge,self.histo,pixel,'_nominal')
        self.pedestal.value = coeff[1]
        self.pedestal.min = coeff[1] - coeff[2]
        self.pedestal.max = coeff[1] + coeff[2]
        self._minuitParameters['values']['pedestal'] = self.pedestal.value
        self._minuitParameters['limit_pedestal'] = (self.pedestal.min,self.pedestal.max)

        self.resolution.value = self.__nectarGainSPEresult[self.__pixel_index(pixel)]['resolution'].value
        self.resolution.error = self.__nectarGainSPEresult[self.__pixel_index(pixel)]['resolution_error'].value

        self.pp.value = self.__nectarGainSPEresult[self.__pixel_index(pixel)]['pp'].value
        self.pp.error = self.__nectarGainSPEresult[self.__pixel_index(pixel)]['pp_error'].value

        self.n.value = self.__nectarGainSPEresult[self.__pixel_index(pixel)]['n'].value
        self.n.error = self.__nectarGainSPEresult[self.__pixel_index(pixel)]['n_error'].value

        if self.__same_luminosity : 
            self.luminosity.value = self.__nectarGainSPEresult[self.__pixel_index(pixel)]['luminosity'].value
            self.luminosity.error = self.__nectarGainSPEresult[self.__pixel_index(pixel)]['luminosity_error'].value


    def __pixel_index(self,pixel) : 
        return np.argmax(self._nectarGainSPEresult['pixel'] == self.pixels_id[pixel])

    @property
    def _nectarGainSPEresult(self) : return self.__nectarGainSPEresult



        


class NectarGainSPESinglePed(NectarGainSPESingle):
    """class to perform fit of the pedestal"""

    __parameters_file = 'parameters_ped.yaml'
    #def __new__(cls) : 
    #    print("NectarGainSPESingleSignal is not instanciable")
    #    return 0
    def __init__(self,signal : ChargeContainer,parameters_file = None,**kwargs) : 
        super().__init__(signal,**kwargs)
        self.__pedestalFitted = np.empty((self.npixels,2))
        #if parameters file is provided
        if parameters_file is None :
            parameters_file = NectarGainSPESinglePed.__parameters_file
        with open(f"{os.path.dirname(os.path.abspath(__file__))}/{parameters_file}") as parameters :
            param = yaml.safe_load(parameters) 
            self.__pedestalWidth = Parameter(name = "pedestalWidth",
                                value = param["pedestalWidth"]["value"],
                                min = param["pedestalWidth"].get("min",np.nan),
                                max = param["pedestalWidth"].get("max",np.nan),
                                unit = param["pedestalWidth"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self.__pedestalWidth)

        self.create_output_table()

        self._output_table.remove_column('gain')
        self._output_table.remove_column('gain_error')


        self._make_minuit_parameters()


    def _run_obs(self,pixel,prescan = False,**kwargs) : 
        valid = False
        self._update_parameters_prefit(pixel)
        fit = Minuit(self.Chi2(pixel),**self._minuitParameters['values'])
        UtilsMinuit.set_minuit_parameters_limits_and_errors(fit,self._minuitParameters)
        log.info(f"Initial value of Likelihood = {self.Chi2(pixel)(**self._minuitParameters['values'])}")
        log.info(f"Initial parameters value : {fit.values}")
        #log.debug(self.Chi2(pixel)(0.5,500,14600,50,1))

        fit.print_level = 2
        fit.strategy = 2
        fit.throw_nan = True
        if prescan : 
            log.info("let's do a fisrt brut force scan") 
            fit.scan()
        try : 
            log.info('migrad execution')
            fit.migrad(ncall=super(NectarGainSPESinglePed,self)._Ncall)
            fit.hesse()
            valid = fit.valid
        except RuntimeError as e : 
            log.warning(e,exc_info = True)
            log.info("change method : re-try with simplex")
            fit.reset()
            fit.throw_nan = True
            if prescan : 
                log.info("let's do a fisrt brut force scan") 
                fit.scan()
            try :
                log.info('simplex execution')
                fit.simplex(ncall=super(NectarGainSPESinglePed,self)._Ncall)
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
            self.__pedestalFitted[pixel,0] = self.pedestal.value
            self.__pedestalFitted[pixel,1] = self.pedestal.error

            log.info(f"pedestal is {self.__pedestalFitted[pixel,0]:.2f} +/- {self.__pedestalFitted[pixel,1]:.2f}")

            self.fill_table(pixel,valid)

            if kwargs.get('figpath',0) != 0 :
                fig,ax = plt.subplots(1,1,figsize=(8, 6))
                ax.errorbar(self.charge[pixel],self.histo[pixel],np.sqrt(self.histo[pixel]),zorder=0,fmt=".",label = "data")
                ax.plot(self.charge[pixel],
                    np.trapz(self.histo[pixel],self.charge[pixel])*gaussian(self.charge[pixel],self.pedestal.value,self.__pedestalWidth.value),
                    zorder=1,
                    linewidth=2,
                    label = f"MPE model fit \n Pedestal = {round(self.__pedestalFitted[pixel,0]):.2f} +/-  {round(self.__pedestalFitted[pixel,1],2):.2f} ADC/pe")
                ax.set_xlabel("Charge (ADC)", size=15)
                ax.set_ylabel("Events", size=15)
                ax.set_title(f"Pedestal fit pixel : {pixel} (pixel id : {self.pixels_id[pixel]})")
                ax.legend(fontsize=15)
                os.makedirs(kwargs.get('figpath'),exist_ok = True)
                fig.savefig(f"{kwargs.get('figpath')}/fit_Ped_pixel{pixel}.pdf")
                fig.clf()
                del fig,ax
                plt.close('all')
        else : 
            log.warning(f"fit {pixel} is not valid")
            self.fill_table(pixel,valid)

    @classmethod
    def NG_Likelihood_Chi2(cls,muped,sigped,charge,histo,**kwargs):
        Lik = 0
        Ntot = np.sum(histo)
        mask = histo > 0
        Lik = np.sum((((gaussian(charge,muped,sigped)*Ntot - histo)[mask])**2)/histo[mask])
        return Lik
    
    def Chi2(self,pixel : int):
        def _Chi2(pedestal,pedestalWidth) :
            return self.NG_Likelihood_Chi2(pedestal,pedestalWidth,self.charge[pixel],self.histo[pixel])
        return _Chi2
    

    @property
    def pedestalWidth(self) : return self.__pedestalWidth

    




    #useless now (with __fix__parameters)
    #def Chi2Fixed(self) :
    #    return self.NG_LikelihoodSignal_Chi2(pp,res,mu2,n,muped,sigped,lum,self.chargeSignal,self.histoSignal)




        
#class NectarGainSPESingleStd(NectarGainSPESingle):
#    """class to perform fit of the 1000V signal and pedestal"""

#class NectarGainSPESingleSignalHHV(NectarGainSPESingle):
#    """class to perform fit of the 1400V signal"""

#class NectarGainSPESinglePedestalHHV(NectarGainSPESingle):
#    """class to perform fit of the 1400V pedestal"""


