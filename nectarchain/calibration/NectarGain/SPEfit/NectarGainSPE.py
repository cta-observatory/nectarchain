import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

import copy
import numpy as np
from datetime import date
import os
import yaml
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

from abc import ABC, abstractclassmethod, abstractmethod


from iminuit import Minuit
from astropy.table import QTable,Column
import astropy.units as u


from .parameters import Parameters
from .utils import UtilsMinuit,weight_gaussian

__all__ = ['NectarGainSPE']

class NectarGainSPE(ABC) :
    _Ncall = 4000000
    def __init__(self) :
        #set parameters value for fit
        self.__parameters = Parameters()
                
        #output
        self._output_table = QTable()
        #self.create_output_table() #need to be done in the child class __init__

    def fill_table(self,pixel : int, valid : bool) : 
        self._output_table['is_valid'][pixel] = valid
        for parameter in self._parameters.parameters : 
            self._output_table[parameter.name][pixel] = parameter.value
            self._output_table[f'{parameter.name}_error'][pixel] = parameter.error
    

    def _make_minuit_parameters(self) : 
        if log.getEffectiveLevel() == logging.DEBUG:
            for parameter in self._parameters.parameters : 
                log.debug(parameter)
        #create minuit parameters
        self.__minuitParameters = UtilsMinuit.make_minuit_par_kwargs(self.__parameters.unfrozen)

    def _update_parameters_postfit(self,m : Minuit) : 
        for i,name in enumerate(m.parameters) : 
            tmp = self.__parameters[name]
            if tmp != [] : 
                tmp.value = m.values[i]
                tmp.error = m.errors[i]

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
    def _get_parameters_gaussian_fit(charge_in, histo_in ,pixel : int,extension = "") :
        #x = np.linspace(nectargain.charge[pixel].min(),nectargain.charge[pixel].max(),int(nectargain.charge[pixel].max()-nectargain.charge[pixel].min()))
        #interp = interp1d(nectargain.charge[pixel],nectargain.histo[pixel])
        charge = charge_in[pixel].data[~histo_in[pixel].mask]
        histo = histo_in[pixel].data[~histo_in[pixel].mask]

        windows_lenght = 80
        order = 2
        histo_smoothed = savgol_filter(histo, windows_lenght, order)

        peaks = find_peaks(histo_smoothed,10)
        peak_max = np.argmax(histo[peaks[0]])
        peak_pos,peak_value = charge[peaks[0][peak_max]], histo[peaks[0][peak_max]]

        coeff, var_matrix = curve_fit(weight_gaussian, charge[:peaks[0][peak_max]], histo[:peaks[0][peak_max]],p0 = [peak_value,peak_pos,1])

        if log.getEffectiveLevel() == logging.DEBUG :
            log.debug('plotting figures with prefit parameters computation') 
            fig,ax = plt.subplots(1,1,figsize = (8,8))
            ax.errorbar(charge,histo,np.sqrt(histo),zorder=0,fmt=".",label = "data")
            ax.plot(charge,histo_smoothed,label = f'smoothed data with savgol filter (windows lenght : {windows_lenght}, order : {order})')
            ax.plot(charge,weight_gaussian(charge,coeff[0],coeff[1],coeff[2]),label = 'gaussian fit of the pedestal, left tail only')
            ax.vlines(peak_pos,0,peak_value,label = 'pedestal initial value')
            ax.set_xlabel("Charge (ADC)", size=15)
            ax.set_ylabel("Events", size=15)
            ax.legend(fontsize=15)
            os.makedirs(f"{os.environ.get('NECTARCHAIN_LOG')}/figures/",exist_ok=True)
            fig.savefig(f"{os.environ.get('NECTARCHAIN_LOG')}/figures/initialization_pixel{pixel}{extension}_{os.getpid()}.pdf")
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
    def _run_obs(self,pixel,**kwargs) : pass
    
    @abstractmethod
    def _update_parameters_prefit(self,pixel) : pass

    
    @abstractmethod
    def Chi2(self,**kwargs) : pass


    @property
    def parameters(self) : return copy.deepcopy(self.__parameters)
    @property
    def _parameters(self) : return self.__parameters
    @property
    def _minuitParameters(self) : return self.__minuitParameters