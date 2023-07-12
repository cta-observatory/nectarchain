import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

import copy
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba

from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

from abc import ABC, abstractclassmethod, abstractmethod


from iminuit import Minuit
from astropy.table import QTable

import pandas as pd

from .parameters import Parameters
from .utils import UtilsMinuit,weight_gaussian,Statistics

__all__ = ['NectarGainSPE']

class NectarGainSPE(ABC) :
    _Ncall = 4000000
    _Windows_lenght = 40
    _Order = 2

    def __init__(self) :
        #set parameters value for fit
        self.__parameters = Parameters()
                
        #output
        self._output_table = QTable()
        #self.create_output_table() #need to be done in the child class __init__

    def fill_table(self,pixel : int, valid : bool,ndof : int,likelihood : float) : 
        self._output_table['is_valid'][pixel] = valid
        for parameter in self._parameters.parameters : 
            self._output_table[parameter.name][pixel] = parameter.value
            self._output_table[f'{parameter.name}_error'][pixel] = parameter.error
        if valid : 
            self._output_table['pvalue'][pixel] = Statistics.chi2_pvalue(ndof,likelihood)
            self._output_table['likelihood'][pixel] = likelihood
        else : 
            self._output_table['pvalue'][pixel] = 0
            self._output_table['likelihood'][pixel] = 0


    def make_table(self,dictionary):
        self._output_table = QTable.from_pandas(pd.DataFrame.from_dict(dictionary))

    def _make_minuit_parameters(self) : 
        if log.getEffectiveLevel() == logging.DEBUG:
            for parameter in self._parameters.parameters : 
                log.debug(parameter)
        #create minuit parameters
        self.__minuitParameters = UtilsMinuit.make_minuit_par_kwargs(self.__parameters.unfrozen)

    #ONLY KEEP STATIC METHOD NOW
    #def _update_parameters_postfit(self,m : Minuit) : 
    #    for i,name in enumerate(m.parameters) : 
    #        tmp = self.__parameters[name]
    #        if tmp != [] : 
    #            tmp.value = m.values[i]
    #            tmp.error = m.errors[i]

    @staticmethod
    def _update_parameters_postfit(m : Minuit,parameters : Parameters) :
        for i,name in enumerate(m.parameters) : 
            tmp = parameters[name]
            if tmp != [] : 
                tmp.value = m.values[i]
                tmp.error = m.errors[i]

    @staticmethod
    def _make_output_dict_obs(m : Minuit,valid,pixels_id,parameters : Parameters,ndof : int) :
        __class__._update_parameters_postfit(m,parameters)
        output = {"is_valid" : valid, "pixel" : pixels_id}
        for parameter in parameters.parameters : 
            output[parameter.name] = parameter.value 
            output[f"{parameter.name}_error"] = parameter.error 

        output['likelihood'] = m.fval
        output['pvalue'] = Statistics.chi2_pvalue(ndof,m.fval)
        return output

    def read_param_from_yaml(self,parameters_file) :
        with open(f"{os.path.dirname(os.path.abspath(__file__))}/{parameters_file}") as parameters :
            param = yaml.safe_load(parameters) 
            for i,name in enumerate(self.__parameters.parnames) :
                dico = param.get(name,False)
                if dico :
                    self._parameters.parameters[i].value = dico.get('value')
                    self._parameters.parameters[i].min = dico.get("min",np.nan)
                    self._parameters.parameters[i].max = dico.get("max",np.nan)

    @staticmethod
    def _get_mean_gaussian_fit(charge_in, histo_in ,extension = ""):
        charge = charge_in.data[~histo_in.mask]
        histo = histo_in.data[~histo_in.mask]

        windows_lenght = NectarGainSPE._Windows_lenght
        order = NectarGainSPE._Order
        histo_smoothed = savgol_filter(histo, windows_lenght, order)

        peaks = find_peaks(histo_smoothed,10)
        peak_max = np.argmax(histo_smoothed[peaks[0]])
        peak_pos,peak_value = charge[peaks[0][peak_max]], histo[peaks[0][peak_max]]

        coeff, var_matrix = curve_fit(weight_gaussian, charge[:peaks[0][peak_max]], histo_smoothed[:peaks[0][peak_max]],p0 = [peak_value,peak_pos,1])

        #nosw find SPE peak excluding pedestal data
        mask = charge > coeff[1]+3*coeff[2]
        peaks_mean = find_peaks(histo_smoothed[mask])
        
        peak_max_mean = np.argmax(histo_smoothed[mask][peaks_mean[0]])
        peak_pos_mean,peak_value_mean = charge[mask][peaks_mean[0][peak_max_mean]], histo_smoothed[mask][peaks_mean[0][peak_max_mean]]

        mask = (charge > ((coeff[1]+peak_pos_mean)/2)) * (charge < (peak_pos_mean + (peak_pos_mean-coeff[1])/2))
        coeff_mean, var_matrix = curve_fit(weight_gaussian, charge[mask], histo_smoothed[mask],p0 = [peak_value_mean,peak_pos_mean,1])

        if log.getEffectiveLevel() == logging.DEBUG :
            log.debug('plotting figures with prefit parameters computation') 
            fig,ax = plt.subplots(1,1,figsize = (5,5))
            ax.errorbar(charge,histo,np.sqrt(histo),zorder=0,fmt=".",label = "data")
            ax.plot(charge,histo_smoothed,label = f'smoothed data with savgol filter (windows lenght : {windows_lenght}, order : {order})')
            ax.plot(charge,weight_gaussian(charge,coeff_mean[0],coeff_mean[1],coeff_mean[2]),label = 'gaussian fit of the SPE')
            ax.vlines(coeff_mean[1],0,peak_value,label = f'mean initial value = {coeff_mean[1] - coeff[1]:.0f}',color = "red")
            ax.add_patch(Rectangle((coeff_mean[1]-coeff_mean[2], 0), 2 * coeff_mean[2], peak_value_mean,fc=to_rgba('red', 0.5)))
            ax.set_xlim([peak_pos - 500,None])
            ax.set_xlabel("Charge (ADC)", size=15)
            ax.set_ylabel("Events", size=15)
            ax.legend(fontsize=7)
            os.makedirs(f"{os.environ.get('NECTARCHAIN_LOG')}/{os.getpid()}/figures/",exist_ok=True)
            fig.savefig(f"{os.environ.get('NECTARCHAIN_LOG')}/{os.getpid()}/figures/initialization_mean_pixel{extension}_{os.getpid()}.pdf")
            fig.clf()
            plt.close(fig)
            del fig,ax
        return coeff_mean,var_matrix

    @staticmethod
    def _get_pedestal_gaussian_fit(charge_in, histo_in ,extension = "") :
        #x = np.linspace(nectargain.charge[pixel].min(),nectargain.charge[pixel].max(),int(nectargain.charge[pixel].max()-nectargain.charge[pixel].min()))
        #interp = interp1d(nectargain.charge[pixel],nectargain.histo[pixel])
        charge = charge_in.data[~histo_in.mask]
        histo = histo_in.data[~histo_in.mask]

        windows_lenght = NectarGainSPE._Windows_lenght
        order = NectarGainSPE._Order
        histo_smoothed = savgol_filter(histo, windows_lenght, order)

        peaks = find_peaks(histo_smoothed,10)
        peak_max = np.argmax(histo_smoothed[peaks[0]])
        peak_pos,peak_value = charge[peaks[0][peak_max]], histo[peaks[0][peak_max]]

        coeff, var_matrix = curve_fit(weight_gaussian, charge[:peaks[0][peak_max]], histo_smoothed[:peaks[0][peak_max]],p0 = [peak_value,peak_pos,1])

        if log.getEffectiveLevel() == logging.DEBUG :
            log.debug('plotting figures with prefit parameters computation') 
            fig,ax = plt.subplots(1,1,figsize = (5,5))
            ax.errorbar(charge,histo,np.sqrt(histo),zorder=0,fmt=".",label = "data")
            ax.plot(charge,histo_smoothed,label = f'smoothed data with savgol filter (windows lenght : {windows_lenght}, order : {order})')
            ax.plot(charge,weight_gaussian(charge,coeff[0],coeff[1],coeff[2]),label = 'gaussian fit of the pedestal, left tail only')
            ax.set_xlim([peak_pos - 500,None])
            ax.vlines(coeff[1],0,peak_value,label = f'pedestal initial value = {coeff[1]:.0f}',color = 'red')
            ax.add_patch(Rectangle((coeff[1]-coeff[2], 0), 2 * coeff[2], peak_value,fc=to_rgba('red', 0.5)))
            ax.set_xlabel("Charge (ADC)", size=15)
            ax.set_ylabel("Events", size=15)
            ax.legend(fontsize=7)
            os.makedirs(f"{os.environ.get('NECTARCHAIN_LOG')}/{os.getpid()}/figures/",exist_ok=True)
            fig.savefig(f"{os.environ.get('NECTARCHAIN_LOG')}/{os.getpid()}/figures/initialization_pedestal_pixel{extension}_{os.getpid()}.pdf")
            fig.clf()
            plt.close(fig)
            del fig,ax
        return coeff,var_matrix

    @abstractmethod
    def create_output_table(self) : pass

    @abstractmethod
    def save(self,path,**kwargs) : pass
    
    @abstractmethod
    def run(self,pixel : int = None,**kwargs): pass
    @abstractmethod
    def _run_obs(self,pixel : int,**kwargs) : pass
    @classmethod
    @abstractmethod
    def _run_obs_static(cls,it : int, funct, parameters : Parameters, pixels_id : int, charge : np.ndarray, histo : np.ndarray, **kwargs) : pass
    
    #@abstractmethod
    #def _update_parameters_prefit(self,pixel : int) : pass
    @classmethod
    @abstractmethod
    def _update_parameters_prefit_static(cls, it : int, parameters : Parameters, charge : np.ndarray, histo : np.ndarray,**kwargs) : pass

    
    @abstractmethod
    def Chi2(self,**kwargs) : pass
    @abstractmethod
    def Chi2_static(self,**kwargs) : pass


    @property
    def parameters(self) : return copy.deepcopy(self.__parameters)
    @property
    def _parameters(self) : return self.__parameters
    @property
    def _minuitParameters(self) : return self.__minuitParameters