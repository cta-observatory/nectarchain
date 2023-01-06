import math
import numpy as np
from scipy import optimize, interpolate
from scipy.stats import linregress
from matplotlib import pyplot as plt
from scipy import signal
from iminuit import Minuit
import random
import astropy.units as u
from astropy.visualization import quantity_support, time_support
from astropy.table import QTable,Column
import astropy.units as u
import yaml
import os
from datetime import date
from pathlib import Path



import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

import copy

from ...container import ChargeContainer,WaveformsContainer

from ..utils.error import DifferentPixelsID

from abc import ABC, abstractclassmethod, abstractmethod

__all__ = ["PhotoStatGainFFandPed"]

class PhotoStatGain(ABC):

    def _readFF(self,FFRun,maxevents: int = None,**kwargs) :
        log.info('reading FF data')
        method = kwargs.get('method','std')
        if isinstance(FFRun,int) : 
            try : 
                self.FFcharge = ChargeContainer.from_file(f"{os.environ['NECTARCAMDATA']}/charges/{method}",FFRun)
                log.info(f'charges have ever been computed for FF run {FFRun}')
            except Exception as e : 
                log.info(f'loading waveforms for FF run {FFRun}')
                FFwaveforms = WaveformsContainer(FFRun,maxevents)
                FFwaveforms.load_wfs()

                if method != 'std' : 
                    log.info(f'computing charge for FF run {FFRun} with following method : {method}')
                    self.FFcharge = ChargeContainer.from_waveforms(FFwaveforms,method = method)
                else :
                    log.info(f'computing charge for FF run {FFRun} with std method')
                    self.FFcharge = ChargeContainer.from_waveforms(FFwaveforms)

                log.debug('writting on disk charge for further works')
                os.makedirs(f"{os.environ['NECTARCAMDATA']}/charges/{method}",exist_ok = True)
                self.FFcharge.write(f"{os.environ['NECTARCAMDATA']}/charges/{method}",overwrite = True)
        
        elif isinstance(FFRun,ChargeContainer):
            self.FFcharge = FFRun
        else : 
            e =  TypeError("FFRun must be int or ChargeContainer")
            log.error(e,exc_info = True)
            raise e

    def _readPed(self,PedRun,maxevents: int = None,**kwargs) :
        log.info('reading Ped data')
        method = kwargs.get('method','std')
        if isinstance(PedRun,int) : 
            try : 
                self.Pedcharge = ChargeContainer.from_file(f"{os.environ['NECTARCAMDATA']}/charges/{method}",PedRun)
                log.info(f'charges have ever been computed for Ped run {PedRun}')
            except Exception as e : 
                log.info(f'loading waveforms for Ped run {PedRun}')
                Pedwaveforms = WaveformsContainer(PedRun,maxevents)
                Pedwaveforms.load_wfs()

                if method != 'std' : 
                    log.info(f'computing charge for Ped run {PedRun} with following method : {method}')
                    self.Pedcharge = ChargeContainer.from_waveforms(Pedwaveforms,method = method)
                else :
                    log.info(f'computing charge for Ped run {PedRun} with std method')
                    self.Pedcharge = ChargeContainer.from_waveforms(Pedwaveforms)

                log.debug('writting on disk charge for further works')
                os.makedirs(f"{os.environ['NECTARCAMDATA']}/charges/{method}",exist_ok = True)
                self.Pedcharge.write(f"{os.environ['NECTARCAMDATA']}/charges/{method}",overwrite = True)
        
        elif isinstance(PedRun,ChargeContainer):
            self.Pedcharge = PedRun
        else : 
            e =  TypeError("PedRun must be int or ChargeContainer")
            log.error(e,exc_info = True)
            raise e

    def _readSPE(self,SPEresults) : 
        log.info(f'reading SPE resolution from {SPEresults}')
        table = QTable.read(SPEresults)
        self.SPEResolution = table['resolution']
        self.SPEGain = table['gain']
        self.SPEGain_error = table['gain_error']
        self._SPE_pixels_id = table['pixel'].value

    

    def create_output_table(self) :
        self._output_table = QTable()
        self._output_table.meta['npixel'] = self.npixels
        self._output_table.meta['comments'] = f'Produced with NectarGain, Credit : CTA NectarCam {date.today().strftime("%B %d, %Y")}'

        self._output_table.add_column(Column(self.pixels_id,"pixel",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels),dtype = np.float64),"high gain",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels,2),dtype = np.float64),"high gain error",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels),dtype = np.float64),"low gain",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels,2),dtype = np.float64),"low gain error",unit = u.dimensionless_unscaled))



    def run(self,**kwargs):
        log.info('running photo statistic method')

        self._output_table["high gain"] = self.gainHG
        self._output_table["low gain"] = self.gainLG

    def save(self,path,**kwargs) : 
        path = Path(path)
        os.makedirs(path,exist_ok = True)
        log.info(f'data saved in {path}')
        self._output_table.write(f"{path}/output_table.ecsv", format='ascii.ecsv',overwrite = kwargs.get("overwrite",False))

    def plot_correlation(self) : 
        mask = (self._output_table["high gain"]>0) * (self.SPEGain>0)
        a, b, r, p_value, std_err = linregress(self._output_table["high gain"][mask], self.SPEGain[mask],'greater')
        x = np.linspace(self._output_table["high gain"][mask].min(),self._output_table["high gain"][mask].max(),1000)
        y = lambda x: a * x + b 
        with quantity_support() : 
            fig,ax = plt.subplots(1,1,figsize=(8, 6))
            ax.scatter(self._output_table["high gain"],self.SPEGain,marker =".")
            ax.plot(x,y(x),color = 'red', label = f"linear fit,\n a = {a:.2e},\n b = {b:.2e},\n r = {r:.2e},\n p_value = {p_value:.2e},\n std_err = {std_err:.2e}")
            ax.set_xlabel("Gain Photo stat", size=15)
            ax.set_ylabel("Gain SPE fit", size=15)
            ax.set_xlim(xmin = 0)
            ax.set_ylim(ymin = 0)

            ax.legend(fontsize=15)
            return fig




    @property
    def npixels(self) : return self._pixels_id.shape[0]

    @property
    def pixels_id(self) : return self._pixels_id




    @property
    def sigmaPedHG(self) : return np.std(self.Pedcharge.charge_hg,axis = 0)

    @property
    def sigmaChargeHG(self) : return np.std(self.FFcharge.charge_hg - self.meanPedHG,axis = 0)

    @property
    def meanPedHG(self) : return np.mean(self.Pedcharge.charge_hg,axis = 0)

    @property
    def meanChargeHG(self) : return np.mean(self.FFcharge.charge_hg - self.meanPedHG,axis = 0)

    @property
    def BHG(self) : 
        min_events = np.min((self.FFcharge.charge_hg.shape[0],self.Pedcharge.charge_hg.shape[0]))
        upper = (np.power(self.FFcharge.charge_hg.mean(axis = 1)[:min_events] - self.Pedcharge.charge_hg.mean(axis = 1)[:min_events] - self.meanChargeHG.mean(),2)).mean(axis = 0)
        lower =  np.power(self.meanChargeHG,2)
        return np.sqrt(upper/lower)

    @property
    def gainHG(self) : return ((np.power(self.sigmaChargeHG,2) - np.power(self.sigmaPedHG,2) - np.power(self.BHG * self.meanChargeHG,2))
                                /(self.meanChargeHG * (1 + np.power(self.SPEResolution,2))))
    

    @property
    def sigmaPedLG(self) : return np.std(self.Pedcharge.charge_lg,axis = 0)

    @property
    def sigmaChargeLG(self) : return np.std(self.FFcharge.charge_lg - self.meanPedLG,axis = 0)

    @property
    def meanPedLG(self) : return np.mean(self.Pedcharge.charge_lg,axis = 0)

    @property
    def meanChargeLG(self) : return np.mean(self.FFcharge.charge_lg - self.meanPedLG,axis = 0)

    @property
    def BLG(self) : 
        min_events = np.min((self.FFcharge.charge_lg.shape[0],self.Pedcharge.charge_lg.shape[0]))
        upper = (np.power(self.FFcharge.charge_lg.mean(axis = 1)[:min_events] - self.Pedcharge.charge_lg.mean(axis = 1)[:min_events] - self.meanChargeLG.mean(),2)).mean(axis = 0)
        lower =  np.power(self.meanChargeLG,2)
        return np.sqrt(upper/lower)

    @property
    def gainLG(self) : return ((np.power(self.sigmaChargeLG,2) - np.power(self.sigmaPedLG,2) - np.power(self.BLG * self.meanChargeLG,2))
                                /(self.meanChargeLG * (1 + np.power(self.SPEResolution,2))))






class PhotoStatGainFFandPed(PhotoStatGain):
    def __init__(self, FFRun, PedRun, SPEresults : str, maxevents : int = None, **kwargs) : 
        self._readFF(FFRun,maxevents,**kwargs)
        self._readPed(PedRun,maxevents,**kwargs)

        if self.FFcharge.charge_hg.shape[1] != self.Pedcharge.charge_hg.shape[1] : 
            e = Exception("Ped run and FF run must have the same number of pixels")
            log.error(e,exc_info = True)
            raise e

        self._readSPE(SPEresults)

        if (self.FFcharge.pixels_id != self.Pedcharge.pixels_id).any() or (self.FFcharge.pixels_id != self._SPE_pixels_id).any() or (self.Pedcharge.pixels_id != self._SPE_pixels_id).any() : 
            e = DifferentPixelsID("Ped run, FF run and SPE run need to have same pixels id")
            log.error(e,exc_info = True)
            raise e
        else : 
            self._pixels_id = self.FFcharge.pixels_id

        self.create_output_table()









    


class PhotoStatGainFF(PhotoStatGain):
    def __init__(self, FFRun, PedRun, SPEresults : str, maxevents : int = None, **kwargs) : 
        self._readFF(FFRun,maxevents,**kwargs)

        self._readSPE(SPEresults)

        if self.FFcharge.pixels_id != self._SPE_pixels_id : 
            e = DifferentPixelsID("Ped run, FF run and SPE run need to have same pixels id")
            log.error(e,exc_info = True)
            raise e
        else : 
            self._pixels_id = self.FFcharge.pixels_id

        self.create_output_table()