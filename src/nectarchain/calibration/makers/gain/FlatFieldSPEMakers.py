import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

import numpy as np
import astropy.units as u
from astropy.table import Column,QTable

import copy
import os
import yaml
import time

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba

import numpy as np

from iminuit import Minuit

from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.special import gammainc

from multiprocessing import  Pool

from inspect import signature

from numba import njit, prange

from .gainMakers import GainMaker

from ...container import ChargeContainer

from .parameters import Parameter, Parameters

from .utils import UtilsMinuit,weight_gaussian,Statistics,MPE2


__all__ = ["FlatFieldSingleHHVSPEMaker","FlatFieldSingleHHVStdSPEMaker"]


class FlatFieldSPEMaker(GainMaker) : 
    _Windows_lenght = 40
    _Order = 2

#constructors
    def __init__(self,*args,**kwargs) : 
        super().__init__(*args,**kwargs)
        self.__parameters = Parameters()

#getters and setters
    @property
    def npixels(self) : return len(self._pixels_id)
    @property
    def parameters(self) : return copy.deepcopy(self.__parameters)
    @property
    def _parameters(self) : return self.__parameters

#methods
    def read_param_from_yaml(self,parameters_file,only_update = False) :
        with open(f"{os.path.dirname(os.path.abspath(__file__))}/{parameters_file}") as parameters :
            param = yaml.safe_load(parameters) 
            if only_update : 
                for i,name in enumerate(self.__parameters.parnames) :
                    dico = param.get(name,False)
                    if dico :
                        self._parameters.parameters[i].value = dico.get('value')
                        self._parameters.parameters[i].min = dico.get("min",np.nan)
                        self._parameters.parameters[i].max = dico.get("max",np.nan)
            else : 
                for name,dico in param.items() :
                    setattr(self,
                            f"__{name}",
                            Parameter(
                                name = name,
                                value = dico["value"],
                                min = dico.get("min",np.nan),
                                max = dico.get("max",np.nan),
                                unit = dico.get("unit",u.dimensionless_unscaled)
                                ))
                    self._parameters.append(eval(f"self.__{name}"))

    @staticmethod
    def _update_parameters(parameters,charge,counts,**kwargs) : 
        coeff_ped,coeff_mean = __class__._get_mean_gaussian_fit(charge,counts,**kwargs)
        pedestal = parameters['pedestal']
        pedestal.value = coeff_ped[1]
        pedestal.min = coeff_ped[1] - coeff_ped[2]
        pedestal.max = coeff_ped[1] + coeff_ped[2]
        log.debug(f"pedestal updated : {pedestal}")
        pedestalWidth = parameters["pedestalWidth"]
        pedestalWidth.value = pedestal.max - pedestal.value
        pedestalWidth.max = 3 * pedestalWidth.value
        log.debug(f"pedestalWidth updated : {pedestalWidth.value}")
        try : 
            if (coeff_mean[1] - pedestal.value < 0) or ((coeff_mean[1] - coeff_mean[2]) - pedestal.max < 0) : raise Exception("mean gaussian fit not good")
            mean = parameters['mean']
            mean.value = coeff_mean[1] - pedestal.value
            mean.min = (coeff_mean[1] - coeff_mean[2]) - pedestal.max
            mean.max = (coeff_mean[1] + coeff_mean[2]) - pedestal.min
            log.debug(f"mean updated : {mean}")
        except Exception as e :
            log.warning(e,exc_info=True)
            log.warning("mean parameters limits and starting value not changed")
        return parameters

    @staticmethod
    def _get_mean_gaussian_fit(charge, counts ,extension = "",**kwargs):
        #charge = charge_in.data[~histo_in.mask]
        #histo = histo_in.data[~histo_in.mask]
        windows_lenght = __class__._Windows_lenght
        order = __class__._Order
        histo_smoothed = savgol_filter(counts, windows_lenght, order)
        peaks = find_peaks(histo_smoothed,10)
        peak_max = np.argmax(histo_smoothed[peaks[0]])
        peak_pos,peak_value = charge[peaks[0][peak_max]], counts[peaks[0][peak_max]]
        coeff, _ = curve_fit(weight_gaussian, charge[:peaks[0][peak_max]], histo_smoothed[:peaks[0][peak_max]],p0 = [peak_value,peak_pos,1])
        if log.getEffectiveLevel() == logging.DEBUG and kwargs.get("display",False) :
            log.debug('plotting figures with prefit parameters computation') 
            fig,ax = plt.subplots(1,1,figsize = (5,5))
            ax.errorbar(charge,counts,np.sqrt(counts),zorder=0,fmt=".",label = "data")
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
        #nosw find SPE peak excluding pedestal data
        mask = charge > coeff[1]+3*coeff[2]
        peaks_mean = find_peaks(histo_smoothed[mask])
    
        peak_max_mean = np.argmax(histo_smoothed[mask][peaks_mean[0]])
        peak_pos_mean,peak_value_mean = charge[mask][peaks_mean[0][peak_max_mean]], histo_smoothed[mask][peaks_mean[0][peak_max_mean]]
        mask = (charge > ((coeff[1]+peak_pos_mean)/2)) * (charge < (peak_pos_mean + (peak_pos_mean-coeff[1])/2))
        coeff_mean, _ = curve_fit(weight_gaussian, charge[mask], histo_smoothed[mask],p0 = [peak_value_mean,peak_pos_mean,1])
        if log.getEffectiveLevel() == logging.DEBUG and kwargs.get("display",False) :
            log.debug('plotting figures with prefit parameters computation') 
            fig,ax = plt.subplots(1,1,figsize = (5,5))
            ax.errorbar(charge,counts,np.sqrt(counts),zorder=0,fmt=".",label = "data")
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
    
        return coeff, coeff_mean

    def _update_table_from_parameters(self) : 
        for param in self._parameters.parameters : 
            if not(param.name in self._results.colnames) : 
                self._results.add_column(Column(data = np.empty((self.npixels),dtype = np.float64),name = param.name,unit = param.unit))
                self._results.add_column(Column(data = np.empty((self.npixels),dtype = np.float64),name = f"{param.name}_error",unit = param.unit))



class FlatFieldSingleHHVSPEMaker(FlatFieldSPEMaker) : 
    """class to perform fit of the SPE signal with all free parameters"""

    __parameters_file = 'parameters_signal.yaml'
    __fit_array = None
    __reduced_name = "FlatFieldSingleSPE"
    __nproc_default = 8
    __chunksize_default = 1

#constructors
    def __init__(self,charge,counts,*args,**kwargs) : 
        super().__init__(*args,**kwargs)
        self.__charge = charge
        self.__counts = counts
        self.__mask_fitted_pixel = np.zeros((self.__charge.shape[0]),dtype = bool)



        self.__pedestal = Parameter(name = "pedestal",
                                value = (np.min(self.__charge) + np.sum(self.__charge * self.__counts)/(np.sum(self.__counts)))/2,
                                min = np.min(self.__charge),
                                max = np.sum(self.__charge*self.__counts)/np.sum(self.__counts),
                                unit = u.dimensionless_unscaled)
        

        self._parameters.append(self.__pedestal)
        
        self.read_param_from_yaml(kwargs.get('parameters_file',self.__parameters_file))

        self._update_table_from_parameters()

        self._results.add_column(Column(np.zeros((self.npixels),dtype = np.float64),"likelihood",unit = u.dimensionless_unscaled))
        self._results.add_column(Column(np.zeros((self.npixels),dtype = np.float64),"pvalue",unit = u.dimensionless_unscaled))

    @classmethod
    def create_from_chargeContainer(cls, signal : ChargeContainer,**kwargs) : 
        histo = signal.histo_hg(autoscale = True)
        return cls(charge = histo[1],counts = histo[0],pixels_id = signal.pixels_id,**kwargs)
    
#getters and setters
    @property
    def charge(self) : return copy.deepcopy(self.__charge)
    @property
    def _charge(self) : return self.__charge

    @property
    def counts(self) : return copy.deepcopy(self.__counts)
    @property
    def _counts(self) : return self.__counts

#I/O method
    def save(self,path,**kwargs) : 
        path = Path(path)
        os.makedirs(path,exist_ok = True)
        log.info(f'data saved in {path}')
        self._results.write(f"{path}/results_{self.__reduced_name}.ecsv", format='ascii.ecsv',overwrite = kwargs.get("overwrite",False))

#methods
    def _fill_results_table_from_dict(self,dico,pixels_id) : 
        chi2_sig = signature(__class__.cost(self._charge,self._counts))
        for i in range(len(pixels_id)) : 
            values = dico[i].get(f"values_{i}",None)
            errors = dico[i].get(f"errors_{i}",None) 
            if not((values is None) or (errors is None)) : 
                index = np.argmax(self._results["pixels_id"] == pixels_id[i])
                if len(values) != len(chi2_sig.parameters) : 
                    e = Exception("the size out the minuit output parameters values array does not fit the signature of the minimized cost function")
                    log.error(e,exc_info=True)
                    raise e
                for j,key in enumerate(chi2_sig.parameters) : 
                    self._results[key][index] = values[j]
                    self._results[f"{key}_error"][index] = errors[j]
                    if key == 'mean' : 
                        self._gain[index] = values[j]
                        self._results[f"gain_error"][index] = [errors[j],errors[j]]
                        self._results[f"gain"][index] = values[j]
                self._results['is_valid'][index] = True
                self._results["likelihood"][index] = __class__.__fit_array[i].fcn(__class__.__fit_array[i].values)
                ndof = self._counts.data[index][~self._counts.mask[index]].shape[0] - __class__.__fit_array[i].nfit
                self._results["pvalue"][index] = Statistics.chi2_pvalue(ndof,__class__.__fit_array[i].fcn(__class__.__fit_array[i].values))

    @staticmethod
    def _NG_Likelihood_Chi2(pp,res,mu2,n,muped,sigped,lum,charge,counts,**kwargs) : 
        pdf = MPE2(charge,pp,res,mu2,n,muped,sigped,lum,**kwargs)
        #log.debug(f"pdf : {np.sum(pdf)}")
        Ntot = np.sum(counts)
        #log.debug(f'Ntot : {Ntot}')
        mask = counts > 0
        Lik = np.sum(((pdf*Ntot-counts)[mask])**2/counts[mask]) #2 times faster
        return Lik
    
    @staticmethod
    def cost(charge,counts) : 
        def Chi2(pedestal,pp,luminosity,resolution,mean,n,pedestalWidth) :
            #assert not(np.isnan(pp) or np.isnan(resolution) or np.isnan(mean) or np.isnan(n) or np.isnan(pedestal) or np.isnan(pedestalWidth) or np.isnan(luminosity))
            for i in range(1000):
                if (gammainc(i+1,luminosity) < 1e-5):
                    ntotalPE = i
                    break
            kwargs = {"ntotalPE" : ntotalPE}
            return __class__._NG_Likelihood_Chi2(pp,resolution,mean,n,pedestal,pedestalWidth,luminosity,charge,counts,**kwargs)
        return Chi2

    #@njit(parallel=True,nopython = True)
    def _make_fit_array_from_parameters(self, pixels_id = None, **kwargs) : 
        if pixels_id is None : 
            npix = self.npixels
            pixels_id = self.pixels_id
        else : 
            npix = len(pixels_id)

        fit_array = np.empty((npix),dtype = np.object_)

        for _id in pixels_id : 
        #for j in prange(len(pixels_id)) : 
        #    _id = pixels_id[j]
            i = np.where(self.pixels_id == _id)[0][0]
            parameters = __class__._update_parameters(self.parameters,self._charge[i].data[~self._charge[i].mask],self._counts[i].data[~self._charge[i].mask],pixel_id=_id,**kwargs)
            minuitParameters = UtilsMinuit.make_minuit_par_kwargs(parameters)
            minuit_kwargs = {parname : minuitParameters['values'][parname] for parname in minuitParameters['values']}
            log.info(f'creation of fit instance for pixel : {_id}')
            fit_array[i] = Minuit(__class__.cost(self._charge[i].data[~self._charge[i].mask],self._counts[i].data[~self._charge[i].mask]),**minuit_kwargs)
            log.debug('fit created')
            fit_array[i].errordef = Minuit.LIKELIHOOD
            fit_array[i].strategy = 0
            fit_array[i].tol = 1e40
            fit_array[i].print_level = 1
            fit_array[i].throw_nan = True
            UtilsMinuit.set_minuit_parameters_limits_and_errors(fit_array[i],minuitParameters)
            log.debug(fit_array[i].values)
            log.debug(fit_array[i].limits)
            log.debug(fit_array[i].fixed)
        return fit_array

    @staticmethod
    def run_fit(i) : 
        log.info("Starting")
        __class__.__fit_array[i].migrad()
        __class__.__fit_array[i].hesse()
        _values = np.array([params.value for params in __class__.__fit_array[i].params])
        _errors = np.array([params.error for params in __class__.__fit_array[i].params])
        log.info("Finished")
        return {f"values_{i}" : _values, f"errors_{i}" : _errors}

    def make(self,
             pixels_id = None,
             multiproc = True, 
             display = True,
             **kwargs) : 
        log.info("running maker")
        log.info('checking asked pixels id')
        if pixels_id is None : 
            npix = self.npixels
        else : 
            log.debug('checking that asked pixels id are in data')
            pixels_id = np.asarray(pixels_id)
            mask = np.array([_id in self.pixels_id for _id in pixels_id],dtype = bool)
            if False in mask : 
                log.debug(f"The following pixels are not in data : {pixels_id[~mask]}")
                pixels_id = pixels_id[mask]
            npix = len(pixels_id)

        if npix == 0 : 
            log.warning('The asked pixels id are all out of the data')
            return None
        else : 
            log.info("creation of the fits instance array")
            __class__.__fit_array = self._make_fit_array_from_parameters(
                                    pixels_id = pixels_id
                                    )

            log.info("running fits")
            if multiproc : 
                nproc = kwargs.get("nproc",__class__.__nproc_default)
                chunksize = kwargs.get("chunksize",max(__class__.__chunksize_default,npix//(nproc*10)))
                log.info(f"pooling with nproc {nproc}, chunksize {chunksize}")

                t = time.time()
                with Pool(nproc) as pool: 
                    result = pool.starmap_async(__class__.run_fit, 
                    [(i,) for i in range(npix)],
                    chunksize=chunksize)
                    result.wait()
                try : 
                    res = result.get()
                except Exception as e : 
                    log.error(e,exc_info=True)
                    raise e
                log.debug(res)
                log.info(f'time for multiproc with starmap_async execution is {time.time() - t:.2e} sec')
            else : 
                log.info("running in mono-cpu")
                t = time.time()
                res = [__class__.run_fit(i) for i in range(npix)]
                log.debug(res)
                log.info(f'time for singleproc execution is {time.time() - t:.2e} sec')

            log.info("filling result table from fits results")
            self._fill_results_table_from_dict(res,pixels_id)

            output = copy.copy(__class__.__fit_array)
            __class__.__fit_array = None

            if display : 
                log.info("plotting")
                self.display(pixels_id,**kwargs)

            return output


    def plot_single(pixel_id,charge,counts,pp,resolution,gain,gain_error,n,pedestal,pedestalWidth,luminosity,likelihood) : 
        fig,ax = plt.subplots(1,1,figsize=(8, 8))
        ax.errorbar(charge,counts,np.sqrt(counts),zorder=0,fmt=".",label = "data")
        ax.plot(charge,
            np.trapz(counts,charge)*MPE2(
                                    charge,
                                    pp, 
                                    resolution, 
                                    gain, 
                                    n, 
                                    pedestal, 
                                    pedestalWidth, 
                                    luminosity,
            ),
            zorder=1,
            linewidth=2,
            label = f"SPE model fit \n gain : {gain - gain_error:.2f} < {gain:.2f} < {gain + gain_error:.2f} ADC/pe,\n likelihood :  {likelihood:.2f}")
        ax.set_xlabel("Charge (ADC)", size=15)
        ax.set_ylabel("Events", size=15)
        ax.set_title(f"SPE fit pixel id : {pixel_id}")
        ax.set_xlim([pedestal - 6 * pedestalWidth, None])
        ax.legend(fontsize=18)
        return fig,ax

    def display(self,pixels_id,**kwargs) : 
        figpath = kwargs.get('figpath',f"/tmp/NectarGain_pid{os.getpid()}")
        os.makedirs(figpath,exist_ok = True)
        for _id in pixels_id :
            index = np.argmax(self._results['pixels_id'] == _id) 
            fig,ax = __class__.plot_single(
                _id,
                self._charge[index],
                self._counts[index],
                self._results['pp'][index].value, 
                self._results['resolution'][index].value, 
                self._results['gain'][index].value, 
                self._results['gain_error'][index].value.mean(),
                self._results['n'][index].value, 
                self._results['pedestal'][index].value, 
                self._results['pedestalWidth'][index].value, 
                self._results['luminosity'][index].value,
                self._results['likelihood'][index],
            )
            fig.savefig(f"{figpath}/fit_SPE_pixel{_id}.pdf")
            fig.clf()
            plt.close(fig)
            del fig,ax



class FlatFieldSingleHHVStdSPEMaker(FlatFieldSingleHHVSPEMaker):
    """class to perform fit of the SPE signal with n and pp fixed"""
    __parameters_file = 'parameters_signalStd.yaml'
    
#constructors
    def __init__(self,charge,counts,*args,**kwargs) : 
        super().__init__(charge,counts,*args,**kwargs)
        self.__fix_parameters()

#methods
    def __fix_parameters(self) : 
        """this method should be used to fix n and pp
        """
        log.info("updating parameters by fixing pp and n")
        pp = self._parameters["pp"]
        pp.frozen = True
        n = self._parameters["n"]
        n.frozen = True
  


class FlatFieldSingleNominalSPEMaker(FlatFieldSingleHHVSPEMaker):
    """class to perform fit of the SPE signal at nominal voltage from fitted data obtained with 1400V run
    Thus, n, pp and res are fixed"""
    __parameters_file = 'parameters_signal_fromHHVFit.yaml'

#constructors
    def __init__(self, charge, counts, nectarGainSPEresult : str, same_luminosity : bool = True, *args, **kwargs):
        super().__init__(charge, counts, *args, **kwargs)
        self.__fix_parameters(same_luminosity)
        self.__same_luminosity = same_luminosity
        self.__nectarGainSPEresult = self._read_SPEresult(nectarGainSPEresult)

#getters and setters
    @property
    def nectarGainSPEresult(self) : return copy.deepcopy(self.__nectarGainSPEresult)

    @property
    def same_luminosity(self) : return copy.deepcopy(self.__same_luminosity)

#methods
    def _read_SPEresult(self,nectarGainSPEresult : str) : 
        table = QTable.read(nectarGainSPEresult,format = "ascii.ecsv")
        argsort = []
        mask = []
        for _id in self._pixels_id : 
            if _id in table['pixels_id'] : 
                argsort.append(np.where(_id==table['pixels_id'])[0][0])
                mask.append(True)
            else : 
                mask.append(True)
        self._pixels_id = self._pixels_id[np.array(mask)]
        return table[np.array(argsort)]

    def __fix_parameters(self, same_luminosity : bool) : 
        """this method should be used to fix n, pp and res
        """
        log.info("updating parameters by fixing pp, n and res")
        pp = self._parameters["pp"]
        pp.frozen = True
        n = self._parameters["n"]
        n.frozen = True
        resolution = self._parameters["resolution"]
        resolution.frozen = True
        if same_luminosity : 
            luminosity = self._parameters["luminosity"]
            luminosity.frozen = True

    def _make_fit_array_from_parameters(self, pixels_id = None, **kwargs) : 
        return super()._make_fit_array_from_parameters(self, pixels_id = pixels_id, nectarGainSPEresult = self.__nectarGainSPEresult, **kwargs)

    @staticmethod
    def _update_parameters(parameters,charge,counts,pixel_id,nectarGainSPEresult,**kwargs) : 
        param = super()._update_parameters(parameters,charge,counts,**kwargs)
        luminosity = param["luminosity"]
        resolution = param["resolution"]
        pp = param["pp"]
        n = param["n"]
        
        index = np.where(pixel_id == nectarGainSPEresult["pixels_id"])[0][0]

        resolution.value = nectarGainSPEresult[index]["resolution"].value
        pp.value = nectarGainSPEresult[index]["pp"].value
        n.value = nectarGainSPEresult[index]["n"].value

        if luminosity.frozen : 
            luminosity.value = nectarGainSPEresult[index]["luminosity"].value
        return param