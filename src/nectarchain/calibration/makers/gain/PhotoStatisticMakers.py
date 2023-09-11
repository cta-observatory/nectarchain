import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers


import numpy as np
from scipy.stats import linregress
from matplotlib import pyplot as plt
import astropy.units as u
from astropy.visualization import quantity_support
from astropy.table import QTable,Column
import os
from datetime import date
from pathlib import Path
import copy

from ctapipe_io_nectarcam import constants

from ...container import ChargeContainer

from .gainMakers import GainMaker


__all__ = ["PhotoStatisticMaker"]

class PhotoStatisticMaker(GainMaker):
    _reduced_name = "PhotoStatistic"

#constructors
    def __init__(self,
                 FFcharge_hg,
                 FFcharge_lg,
                 Pedcharge_hg,
                 Pedcharge_lg,
                 coefCharge_FF_Ped,
                 SPE_resolution,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(*args,**kwargs)

        self.__coefCharge_FF_Ped = coefCharge_FF_Ped
        
        self.__FFcharge_hg = FFcharge_hg
        self.__FFcharge_lg = FFcharge_lg

        self.__Pedcharge_hg = Pedcharge_hg
        self.__Pedcharge_lg = Pedcharge_lg

        if isinstance(SPE_resolution,np.ndarray) and len(SPE_resolution) == self.npixels : 
            self.__SPE_resolution = SPE_resolution
        elif isinstance(SPE_resolution,list) and len(SPE_resolution) == self.npixels : 
            self.__SPE_resolution = np.array(SPE_resolution)
        elif isinstance(SPE_resolution,float) :
            self.__SPE_resolution = SPE_resolution * np.ones((self.npixels)) 
        else : 
            e = TypeError("SPE_resolution must be a float, a numpy.ndarray or list instance")
            raise e

        self.__check_shape()


    @classmethod
    def create_from_chargeContainer(cls, 
                                    FFcharge : ChargeContainer, 
                                    Pedcharge : ChargeContainer, 
                                    coefCharge_FF_Ped, 
                                    SPE_resolution, 
                                    **kwargs) : 
        if isinstance(SPE_resolution , str) or isinstance(SPE_resolution , Path) : 
            SPE_resolution,SPE_pixels_id = __class__.__readSPE(SPE_resolution)
        else : 
            SPE_pixels_id = None

        kwargs_init =  __class__.__get_charges_FF_Ped_reshaped(FFcharge,
                                                            Pedcharge,
                                                            SPE_resolution,
                                                            SPE_pixels_id)

        kwargs.update(kwargs_init)
        return cls(coefCharge_FF_Ped = coefCharge_FF_Ped, **kwargs)

    @classmethod
    def create_from_run_numbers(cls, FFrun : int, Pedrun : int, SPE_resolution : str, **kwargs) : 
        FFkwargs = __class__.__readFF(FFrun, **kwargs)
        Pedkwargs = __class__.__readPed(Pedrun, **kwargs)
        kwargs.update(FFkwargs)
        kwargs.update(Pedkwargs)
        return cls.create_from_chargeContainer(SPE_resolution = SPE_resolution, **kwargs)

#methods
    @staticmethod
    def __readSPE(SPEresults) : 
        log.info(f'reading SPE resolution from {SPEresults}')
        table = QTable.read(SPEresults)
        table.sort('pixels_id')
        return table['resolution'][table['is_valid']].value,table['pixels_id'][table['is_valid']].value

    @staticmethod
    def __get_charges_FF_Ped_reshaped( FFcharge : ChargeContainer, Pedcharge : ChargeContainer, SPE_resolution, SPE_pixels_id) : 
        log.info("reshape of SPE, Ped and FF data with intersection of pixel ids")
        out = {}

        FFped_intersection =  np.intersect1d(Pedcharge.pixels_id,FFcharge.pixels_id)
        if not(SPE_pixels_id is None) : 
            SPEFFPed_intersection = np.intersect1d(FFped_intersection,SPE_pixels_id)
            mask_SPE = np.array([SPE_pixels_id[i] in SPEFFPed_intersection for i in range(len(SPE_pixels_id))],dtype = bool)
            out["SPE_resolution"] = SPE_resolution[mask_SPE]

        out["pixels_id"] = SPEFFPed_intersection
        out["FFcharge_hg"] = FFcharge.select_charge_hg(SPEFFPed_intersection)
        out["FFcharge_lg"] = FFcharge.select_charge_lg(SPEFFPed_intersection)
        out["Pedcharge_hg"] = Pedcharge.select_charge_hg(SPEFFPed_intersection)
        out["Pedcharge_lg"] = Pedcharge.select_charge_lg(SPEFFPed_intersection)
        
        log.info(f"data have {len(SPEFFPed_intersection)} pixels in common")
        return out

    @staticmethod
    def __readFF(FFRun,**kwargs) :
        log.info('reading FF data')
        method = kwargs.get('method','FullWaveformSum')
        FFchargeExtractorWindowLength = kwargs.get('FFchargeExtractorWindowLength',None)
        if method != 'FullWaveformSum' :
            if FFchargeExtractorWindowLength is None : 
                e = Exception(f"we have to specify FFchargeExtractorWindowLength argument if charge extractor method is not FullwaveformSum")
                log.error(e,exc_info=True)
                raise e
            else : 
                coefCharge_FF_Ped = FFchargeExtractorWindowLength / constants.N_SAMPLES
        else : 
            coefCharge_FF_Ped = 1
        if isinstance(FFRun,int) : 
            try : 
                FFcharge = ChargeContainer.from_file(f"{os.environ['NECTARCAMDATA']}/charges/{method}",FFRun)
                log.info(f'charges have ever been computed for FF run {FFRun}')
            except Exception as e : 
                log.error("charge have not been yet computed")
                raise e
        else : 
            e =  TypeError("FFRun must be int or ChargeContainer")
            log.error(e,exc_info = True)
            raise e
        return {"FFcharge" : FFcharge, "coefCharge_FF_Ped" : coefCharge_FF_Ped}

    @staticmethod
    def __readPed(PedRun,**kwargs) :
        log.info('reading Ped data')
        method = 'FullWaveformSum'#kwargs.get('method','std')
        if isinstance(PedRun,int) : 
            try : 
                Pedcharge = ChargeContainer.from_file(f"{os.environ['NECTARCAMDATA']}/charges/{method}",PedRun)
                log.info(f'charges have ever been computed for Ped run {PedRun}')
            except Exception as e : 
                log.error("charge have not been yet computed")
                raise e
        else : 
            e =  TypeError("PedRun must be int or ChargeContainer")
            log.error(e,exc_info = True)
            raise e
        return {"Pedcharge" : Pedcharge}

    def __check_shape(self) -> None: 
        try : 
            self.__FFcharge_hg[0] * self.__FFcharge_lg[0] * self.__Pedcharge_hg[0] * self.__Pedcharge_lg[0] * self.__SPE_resolution * self._pixels_id
        except Exception as e : 
            log.error(e,exc_info = True)
            raise e

    def make(self,**kwargs)-> None:
        log.info('running photo statistic method')
        self._results["high_gain"] = self.gainHG
        self._results["low_gain"] = self.gainLG
        #self._results["is_valid"] = self._SPEvalid


    def plot_correlation(photoStat_gain,SPE_gain) : 
        mask = (photoStat_gain>20) * (SPE_gain>0) * (photoStat_gain<80)
        a, b, r, p_value, std_err = linregress(photoStat_gain[mask], SPE_gain[mask],'greater')
        x = np.linspace(photoStat_gain[mask].min(),photoStat_gain[mask].max(),1000)
        y = lambda x: a * x + b 
        with quantity_support() : 
            fig,ax = plt.subplots(1,1,figsize=(8, 6))
            ax.scatter(photoStat_gain[mask],SPE_gain[mask],marker =".")
            ax.plot(x,y(x),color = 'red', label = f"linear fit,\n a = {a:.2e},\n b = {b:.2e},\n r = {r:.2e},\n p_value = {p_value:.2e},\n std_err = {std_err:.2e}")
            ax.plot(x,x,color = 'black',label = "y = x")
            ax.set_xlabel("Gain Photo stat (ADC)", size=15)
            ax.set_ylabel("Gain SPE fit (ADC)", size=15)
            #ax.set_xlim(xmin = 0)
            #ax.set_ylim(ymin = 0)
            ax.legend(fontsize=15)
        return fig

#getters and setters
    @property
    def SPE_resolution(self) : return copy.deepcopy(self.__SPE_resolution)

    @property
    def sigmaPedHG(self) : return np.std(self.__Pedcharge_hg ,axis = 0) * np.sqrt(self.__coefCharge_FF_Ped)

    @property
    def sigmaChargeHG(self) : return np.std(self.__FFcharge_hg - self.meanPedHG, axis = 0)

    @property
    def meanPedHG(self) : return np.mean(self.__Pedcharge_hg ,axis = 0) * self.__coefCharge_FF_Ped

    @property
    def meanChargeHG(self) : return np.mean(self.__FFcharge_hg  - self.meanPedHG, axis = 0)

    @property
    def BHG(self) : 
        min_events = np.min((self.__FFcharge_hg.shape[0],self.__Pedcharge_hg.shape[0]))
        upper = (np.power(self.__FFcharge_hg.mean(axis = 1)[:min_events] - self.__Pedcharge_hg.mean(axis = 1)[:min_events] * self.__coefCharge_FF_Ped - self.meanChargeHG.mean(),2)).mean(axis = 0)
        lower =  np.power(self.meanChargeHG.mean(),2)#np.power(self.meanChargeHG,2)#np.power(self.meanChargeHG.mean(),2)
        return np.sqrt(upper/lower)

    @property
    def gainHG(self) : 
        return ((np.power(self.sigmaChargeHG,2) - np.power(self.sigmaPedHG,2) - np.power(self.BHG * self.meanChargeHG,2))
                                /(self.meanChargeHG * (1 + np.power(self.SPE_resolution,2))))
    

    @property
    def sigmaPedLG(self) : return np.std(self.__Pedcharge_lg ,axis = 0) * np.sqrt(self.__coefCharge_FF_Ped)

    @property
    def sigmaChargeLG(self) : return np.std(self.__FFcharge_lg  - self.meanPedLG,axis = 0)

    @property
    def meanPedLG(self) : return np.mean(self.__Pedcharge_lg,axis = 0) * self.__coefCharge_FF_Ped

    @property
    def meanChargeLG(self) : return np.mean(self.__FFcharge_lg - self.meanPedLG,axis = 0)

    @property
    def BLG(self) : 
        min_events = np.min((self.__FFcharge_lg.shape[0],self.__Pedcharge_lg.shape[0]))
        upper = (np.power(self.__FFcharge_lg.mean(axis = 1)[:min_events] - self.__Pedcharge_lg.mean(axis = 1)[:min_events] * self.__coefCharge_FF_Ped - self.meanChargeLG.mean(),2)).mean(axis = 0)
        lower =  np.power(self.meanChargeLG.mean(),2) #np.power(self.meanChargeLG,2) #np.power(self.meanChargeLG.mean(),2)
        return np.sqrt(upper/lower)

    @property
    def gainLG(self) : return ((np.power(self.sigmaChargeLG,2) - np.power(self.sigmaPedLG,2) - np.power(self.BLG * self.meanChargeLG,2))
                                /(self.meanChargeLG * (1 + np.power(self.SPE_resolution,2))))

