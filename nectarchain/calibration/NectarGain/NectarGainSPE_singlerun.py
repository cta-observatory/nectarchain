import math
import numpy as np
from scipy import optimize, interpolate
from matplotlib import pyplot as plt
from scipy import signal
from scipy.special import gammainc
from iminuit import Minuit
import random
import astropy.units as u
import yaml
import os

import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

import copy

from .parameters import Parameters,Parameter
from ..container import ChargeContainer
from .utils import UtilsMinuit,MPE2,Gain,gaussian

from abc import ABC, abstractclassmethod, abstractmethod

__all__ = ["NectarGainSPESingleSignalStd","NectarGainSPESingleSignal","NectarGainSPESinglePed"]

class NectarGainSPESingle(ABC):
    @abstractmethod
    def __init__(self,signal : ChargeContainer,**kwargs) : 
        histo = signal.histo_hg(n_bins=1000)
        #access data
        self._charge = histo[1]
        self._histo = histo[0]
        self._mask_fitted_pixel = np.zeros((self._charge.shape[0]),dtype = bool)
        
        #set parameters value for fit
        self._parameters = Parameters()
        self._pedestal = Parameter(name = "pedestal",
                                value = (np.min(self._charge) + np.sum(self._charge * self._histo)/(np.sum(self._histo)))/2,
                                min = np.min(self._charge),
                                max = np.sum(self._charge*self._histo)/np.sum(self._histo),
                                unit = u.dimensionless_unscaled)
        

        self._parameters.append(self._pedestal)
        self._minuitParameters = UtilsMinuit.make_minuit_par_kwargs(self._parameters.unfrozen)

    def _make_minuit_parameters(self) : 
        if log.getEffectiveLevel() == logging.DEBUG:
            for parameter in self._parameters.parameters : 
                log.debug(parameter)
        #create minuit parameters
        self._minuitParameters = UtilsMinuit.make_minuit_par_kwargs(self._parameters.unfrozen)

        
    def run(self,pixel : int = None,**kwargs):
        if pixel is None : 
            for i in range(self._charge.shape[0]) :
                self._run_obs(i,**kwargs)
        else : 
            if pixel >= self._charge.shape[0] : 
                e = Exception(f"pixel must be < {self._charge.shape[0]}")
                log.error(e,exc_info=True)
                raise e
            else :
                self._run_obs(pixel,**kwargs)
        return 0

    def _update_parameters_postfit(self,m : Minuit) : 
        for i,name in enumerate(m.parameters) : 
            tmp = self._parameters[name]
            if tmp != [] : 
                tmp.value = m.values[i]
                tmp.error = m.errors[i]

    def _update_parameters_prefit(self,pixel) : 
        self._pedestal.value = (np.min(self._charge[pixel]) + np.sum(self._charge[pixel] * self._histo[pixel])/(np.sum(self._histo[pixel])))/2
        self._pedestal.min = np.min(self._charge[pixel])
        self._pedestal.max = np.sum(self._charge[pixel]*self._histo[pixel])/np.sum(self._histo[pixel])
        self._minuitParameters['values']['pedestal'] = self._pedestal.value
        self._minuitParameters['limit_pedestal'] = (self._pedestal.min,self._pedestal.max)

        
    @abstractclassmethod
    def NG_Likelihood_Chi2(cls,**kwargs) : pass
    @abstractmethod
    def Chi2(self,**kwargs) : pass
    @abstractmethod
    def _run_obs(self,pixel,**kwargs) : pass

    @property
    def charge(self) : return self._charge
    @property
    def histo(self) : return self._histo
    @property
    def parameters(self) : return copy.deepcopy(self._parameters)
    @property
    def pedestal(self) : return self._pedestal


class NectarGainSPESingleSignal(NectarGainSPESingle):
    """class to perform fit of the SPE signal with all free parameters"""
    __parameters_file = 'parameters_signal.yaml'
    #def __new__(cls) : 
    #    print("NectarGainSPESingleSignal is not instanciable")
    #    return 0
    def __init__(self,signal : ChargeContainer,parameters_file = None,**kwargs) : 
        self._old_lum = None
        self._old_ntotalPE = None
        super().__init__(signal,**kwargs)
        self._gain = np.empty((self._charge.shape[0],2))
        #if parameters file is provided
        if parameters_file is None :
            parameters_file = NectarGainSPESingleSignal.__parameters_file
        with open(f"{os.path.dirname(os.path.abspath(__file__))}/{parameters_file}") as parameters :
            param = yaml.safe_load(parameters) 
            self._pp = Parameter(name = "pp",
                                value = param["pp"]["value"],
                                min = param["pp"].get("min",np.nan),
                                max = param["pp"].get("max",np.nan),
                                unit = param["pp"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self._pp)
            
            self._luminosity = Parameter(name = "luminosity",
                                value = param["luminosity"]["value"],
                                min = param["luminosity"].get("min",np.nan),
                                max = param["luminosity"].get("max",np.nan),
                                unit = param["luminosity"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self._luminosity)
            
            self._resolution = Parameter(name = "resolution",
                                value = param["resolution"]["value"],
                                min = param["resolution"].get("min",np.nan),
                                max = param["resolution"].get("max",np.nan),
                                unit = param["resolution"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self._resolution)
            
            self._mean = Parameter(name = "mean",
                                value = param["mean"]["value"],
                                min = param["mean"].get("min",np.nan),
                                max = param["mean"].get("max",np.nan),
                                unit = param["mean"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self._mean)
            
            self._n = Parameter(name = "n",
                                value = param["n"]["value"],
                                min = param["n"].get("min",np.nan),
                                max = param["n"].get("max",np.nan),
                                unit = param["n"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self._n)
            
            self._pedestalWidth = Parameter(name = "pedestalWidth",
                                value = param["pedestalWidth"]["value"],
                                min = param["pedestalWidth"].get("min",np.nan),
                                max = param["pedestalWidth"].get("max",np.nan),
                                unit = param["pedestalWidth"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self._pedestalWidth)
        self._make_minuit_parameters()

        


    def _run_obs(self,pixel,**kwargs) : 
        self._update_parameters_prefit(pixel)
        fit = Minuit(self.Chi2(pixel),**self._minuitParameters['values'])
        UtilsMinuit.set_minuit_parameters_limits_and_errors(fit,self._minuitParameters)
        log.info(f"Initial value of Likelihood = {self.Chi2(pixel)(**self._minuitParameters['values'])}")
        log.info(f"Initial parameters value : {fit.values}")
        #log.debug(self.Chi2(pixel)(0.5,500,14600,50,1))

        fit.print_level = 2
        fit.strategy = 2
        fit.migrad(ncall=4000000)
        fit.hesse()
        log.info(f"fitted value : {fit.values}")
        log.info(f"fitted errors : {fit.errors}")
        self._update_parameters_postfit(fit)
        self._gain[pixel,0] = Gain(self._pp.value,self._resolution.value,self._mean.value,self._n.value)
        self._gain[pixel,1] = np.std([Gain(self._pp.value,random.gauss(self._resolution.value, self._resolution.error),random.gauss(self._mean.value, self._mean.error),self._n.value) for i in range(1000)])

        log.info(f"Reconstructed gain is {self._gain[pixel,0]} +/- {self._gain[pixel,1]}")

        if kwargs.get('figpath',0) != 0 :
            fig,ax = plt.subplots(1,1,figsize=(8, 6))
            ax.errorbar(self._charge[pixel],self._histo[pixel],np.sqrt(self._histo[pixel]),zorder=0,fmt=".",label = "data")
            ax.plot(self._charge[pixel],
                np.trapz(self._histo[pixel],self._charge[pixel])*MPE2(self._charge[pixel],self._pp.value,self._resolution.value,self._mean.value,self._n.value,self._pedestal.value,self._pedestalWidth.value,self._luminosity.value),
                zorder=1,
                linewidth=2,
                label = f"MPE model fit \n gain = {round(self._gain[pixel,0])} +/-  {round(self._gain[pixel,1],2)} ADC/pe")
            ax.set_xlabel("Charge (ADC)", size=15)
            ax.set_ylabel("Events", size=15)
            ax.set_title(f"SPE fit pixel : {pixel}")
            ax.legend(fontsize=15)
            os.makedirs(kwargs.get('figpath'),exist_ok = True)
            fig.savefig(f"{kwargs.get('figpath')}/fit_SPE_pixel{pixel}.pdf")
            fig.clf()
            del fig,ax
            plt.close('all')

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
        def _Chi2(resolution,mean,pedestal,pedestalWidth,luminosity) :
            if self._old_lum != luminosity :
                for i in range(1000):
                    if (gammainc(i+1,luminosity) < 1e-5):
                        self._old_ntotalPE = i
                        break
                self._old_lum = luminosity
            kwargs = {"ntotalPE" : self._old_ntotalPE}

            return self.NG_Likelihood_Chi2(self._pp.value,resolution,mean,self._n.value,pedestal,pedestalWidth,luminosity,self._charge[pixel],self._histo[pixel],**kwargs)
        return _Chi2
    
    @property
    def pp(self) : return self._pp  
    @property
    def luminosity(self) : return self._luminosity
    @property
    def mean(self) : return self._mean      
    @property
    def n(self) : return self._n
    @property
    def pedestalWidth(self) : return self._pedestalWidth
    @property
    def gain(self) : return self._gain
    
    
    
class NectarGainSPESingleSignalStd(NectarGainSPESingleSignal):
    """class to perform fit of the SPE signal with n' and p fixed"""
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
        log.info("updating parameters with fixing pp and n")
        self._pp.frozen = True
        self._n.frozen = True

    def Chi2(self,pixel : int):
        def _Chi2(pp,resolution,mean,n,pedestal,pedestalWidth,luminosity) :
            if self._old_lum != luminosity :
                for i in range(1000):
                    if (gammainc(i+1,luminosity) < 1e-5):
                        self._old_ntotalPE = i
                        break
                self._old_lum = luminosity
            kwargs = {"ntotalPE" : self._old_ntotalPE}

            return self.NG_Likelihood_Chi2(pp,resolution,mean,n,pedestal,pedestalWidth,luminosity,self._charge[pixel],self._histo[pixel],**kwargs)
        return _Chi2
    


class NectarGainSPESinglePed(NectarGainSPESingle):
    """class to perform fit of the pedestal"""

    __parameters_file = 'parameters_ped.yaml'
    #def __new__(cls) : 
    #    print("NectarGainSPESingleSignal is not instanciable")
    #    return 0
    def __init__(self,signal : ChargeContainer,parameters_file = None,**kwargs) : 
        super().__init__(signal,**kwargs)
        self._pedestalFitted = np.empty((self._charge.shape[0],2))
        #if parameters file is provided
        if parameters_file is None :
            parameters_file = NectarGainSPESinglePed.__parameters_file
        with open(f"{os.path.dirname(os.path.abspath(__file__))}/{parameters_file}") as parameters :
            param = yaml.safe_load(parameters) 
            self._pedestalWidth = Parameter(name = "pedestalWidth",
                                value = param["pedestalWidth"]["value"],
                                min = param["pedestalWidth"].get("min",np.nan),
                                max = param["pedestalWidth"].get("max",np.nan),
                                unit = param["pedestalWidth"].get("unit",u.dimensionless_unscaled))
            self._parameters.append(self._pedestalWidth)

        self._make_minuit_parameters()


    def _run_obs(self,pixel,**kwargs) : 
        self._update_parameters_prefit(pixel)
        fit = Minuit(self.Chi2(pixel),**self._minuitParameters['values'])
        UtilsMinuit.set_minuit_parameters_limits_and_errors(fit,self._minuitParameters)
        log.info(f"Initial value of Likelihood = {self.Chi2(pixel)(**self._minuitParameters['values'])}")
        log.info(f"Initial parameters value : {fit.values}")
        #log.debug(self.Chi2(pixel)(0.5,500,14600,50,1))

        fit.print_level = 2
        fit.strategy = 2
        fit.migrad(ncall=4000000)
        fit.hesse()
        log.info(f"fitted value : {fit.values}")
        log.info(f"fitted errors : {fit.errors}")
        self._update_parameters_postfit(fit)
        self._pedestalFitted[pixel,0] = self._pedestal.value
        self._pedestalFitted[pixel,1] = self._pedestal.error

        log.info(f"pedestal is {self._pedestalFitted[pixel,0]} +/- {self._pedestalFitted[pixel,1]}")

        if kwargs.get('figpath',0) != 0 :
            fig,ax = plt.subplots(1,1,figsize=(8, 6))
            ax.errorbar(self._charge[pixel],self._histo[pixel],np.sqrt(self._histo[pixel]),zorder=0,fmt=".",label = "data")
            ax.plot(self._charge[pixel],
                np.trapz(self._histo[pixel],self._charge[pixel])*gaussian(self._charge[pixel],self._pedestal.value,self._pedestalWidth.value),
                zorder=1,
                linewidth=2,
                label = f"MPE model fit \n gain = {round(self._pedestalFitted[pixel,0])} +/-  {round(self._pedestalFitted[pixel,1],2)} ADC/pe")
            ax.set_xlabel("Charge (ADC)", size=15)
            ax.set_ylabel("Events", size=15)
            ax.set_title(f"Pedestal fit pixel : {pixel}")
            ax.legend(fontsize=15)
            os.makedirs(kwargs.get('figpath'),exist_ok = True)
            fig.savefig(f"{kwargs.get('figpath')}/fit_Ped_pixel{pixel}.pdf")
            fig.clf()
            del fig,ax
            plt.close('all')

    @classmethod
    def NG_Likelihood_Chi2(cls,muped,sigped,charge,histo,**kwargs):
        Lik = 0
        Ntot = np.sum(histo)
        mask = histo > 0
        Lik = np.sum((((gaussian(charge,muped,sigped)*Ntot - histo)[mask])**2)/histo[mask])
        return Lik
    
    def Chi2(self,pixel : int):
        def _Chi2(pedestal,pedestalWidth) :
            return self.NG_Likelihood_Chi2(pedestal,pedestalWidth,self._charge[pixel],self._histo[pixel])
        return _Chi2
    

    @property
    def pedestalWidth(self) : return self._pedestalWidth

    




    #useless now (with __fix_parameters)
    #def Chi2Fixed(self) :
    #    return self.NG_LikelihoodSignal_Chi2(pp,res,mu2,n,muped,sigped,lum,self.chargeSignal,self.histoSignal)




        
#class NectarGainSPESingleStd(NectarGainSPESingle):
#    """class to perform fit of the 1000V signal and pedestal"""

#class NectarGainSPESingleSignalHHV(NectarGainSPESingle):
#    """class to perform fit of the 1400V signal"""

#class NectarGainSPESinglePedestalHHV(NectarGainSPESingle):
#    """class to perform fit of the 1400V pedestal"""


