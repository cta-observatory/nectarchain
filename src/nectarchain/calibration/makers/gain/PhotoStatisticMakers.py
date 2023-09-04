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

from . import GainMaker

from ctapipe_io_nectarcam import constants

from ...container import ChargeContainer


__all__ = ["PhotoStatGainFFandPed"]

class PhotoStatGain(ABC):

    def _readFF(self,FFRun,maxevents: int = None,**kwargs) :
        log.info('reading FF data')
        method = kwargs.get('method','FullWaveformSum')
        FFchargeExtractorWindowLength = kwargs.get('FFchargeExtractorWindowLength',None)
        if method != 'FullWaveformSum' :
            if FFchargeExtractorWindowLength is None : 
                e = Exception(f"we have to specify FFchargeExtractorWindowLength argument if charge extractor method is not FullwaveformSum")
                log.error(e,exc_info=True)
                raise e
            else : 
                self.__coefCharge_FF_Ped = FFchargeExtractorWindowLength / constants.N_SAMPLES
        else : 
            self.__coefCharge_FF_Ped = 1
        if isinstance(FFRun,int) : 
            try : 
                self.FFcharge = ChargeContainer.from_file(f"{os.environ['NECTARCAMDATA']}/charges/{method}",FFRun)
                log.info(f'charges have ever been computed for FF run {FFRun}')
            except Exception as e : 
                log.error("charge have not been yet computed")
                raise e

                #log.info(f'loading waveforms for FF run {FFRun}')
                #FFwaveforms = WaveformsContainer(FFRun,maxevents)
                #FFwaveforms.load_wfs()
                #if method != 'std' : 
                #    log.info(f'computing charge for FF run {FFRun} with following method : {method}')
                #    self.FFcharge = ChargeContainer.from_waveforms(FFwaveforms,method = method)
                #else :
                #    log.info(f'computing charge for FF run {FFRun} with std method')
                #    self.FFcharge = ChargeContainer.from_waveforms(FFwaveforms)
                #log.debug('writting on disk charge for further works')
                #os.makedirs(f"{os.environ['NECTARCAMDATA']}/charges/{method}",exist_ok = True)
                #self.FFcharge.write(f"{os.environ['NECTARCAMDATA']}/charges/{method}",overwrite = True)
        
        elif isinstance(FFRun,ChargeContainer):
            self.FFcharge = FFRun
        else : 
            e =  TypeError("FFRun must be int or ChargeContainer")
            log.error(e,exc_info = True)
            raise e

    def _readPed(self,PedRun,maxevents: int = None,**kwargs) :
        log.info('reading Ped data')
        method = 'FullWaveformSum'#kwargs.get('method','std')
        if isinstance(PedRun,int) : 
            try : 
                self.Pedcharge = ChargeContainer.from_file(f"{os.environ['NECTARCAMDATA']}/charges/{method}",PedRun)
                log.info(f'charges have ever been computed for Ped run {PedRun}')
            except Exception as e : 
                log.error("charge have not been yet computed")
                raise e

                #log.info(f'loading waveforms for Ped run {PedRun}')
                #Pedwaveforms = WaveformsContainer(PedRun,maxevents)
                #Pedwaveforms.load_wfs()
                #if method != 'std' : 
                #    log.info(f'computing charge for Ped run {PedRun} with following method : {method}')
                #    self.Pedcharge = ChargeContainer.from_waveforms(Pedwaveforms,method = method)
                #else :
                #    log.info(f'computing charge for Ped run {PedRun} with std method')
                #    self.Pedcharge = ChargeContainer.from_waveforms(Pedwaveforms)
                #log.debug('writting on disk charge for further works')
                #os.makedirs(f"{os.environ['NECTARCAMDATA']}/charges/{method}",exist_ok = True)
                #self.Pedcharge.write(f"{os.environ['NECTARCAMDATA']}/charges/{method}",overwrite = True)
        
        elif isinstance(PedRun,ChargeContainer):
            self.Pedcharge = PedRun
        else : 
            e =  TypeError("PedRun must be int or ChargeContainer")
            log.error(e,exc_info = True)
            raise e

    def _readSPE(self,SPEresults) : 
        log.info(f'reading SPE resolution from {SPEresults}')
        table = QTable.read(SPEresults)
        table.sort('pixel')
        self.SPEResolution = table['resolution']
        self.SPEGain = table['gain']
        self.SPEGain_error = table['gain_error']
        self._SPEvalid = table['is_valid']
        self._SPE_pixels_id = np.array(table['pixel'].value,dtype = np.uint16)

    def _reshape_all(self) : 
        log.info("reshape of SPE, Ped and FF data with intersection of pixel ids")
        FFped_intersection =  np.intersect1d(self.Pedcharge.pixels_id,self.FFcharge.pixels_id)
        SPEFFPed_intersection = np.intersect1d(FFped_intersection,self._SPE_pixels_id[self._SPEvalid])
        self._pixels_id = SPEFFPed_intersection
        log.info(f"data have {len(self._pixels_id)} pixels in common")

        self._FFcharge_hg = self.FFcharge.select_charge_hg(SPEFFPed_intersection)
        self._FFcharge_lg = self.FFcharge.select_charge_lg(SPEFFPed_intersection)

        self._Pedcharge_hg = self.Pedcharge.select_charge_hg(SPEFFPed_intersection)
        self._Pedcharge_lg = self.Pedcharge.select_charge_lg(SPEFFPed_intersection)
        
        #self._mask_FF = np.array([self.FFcharge.pixels_id[i] in SPEFFPed_intersection for i in range(self.FFcharge.npixels)],dtype = bool)
        #self._mask_Ped = np.array([self.Pedcharge.pixels_id[i] in SPEFFPed_intersection for i in range(self.Pedcharge.npixels)],dtype = bool)
        self._mask_SPE = np.array([self._SPE_pixels_id[i] in SPEFFPed_intersection for i in range(len(self._SPE_pixels_id))],dtype = bool)

    

    def create_output_table(self) :
        self._output_table = QTable()
        self._output_table.meta['npixel'] = self.npixels
        self._output_table.meta['comments'] = f'Produced with NectarGain, Credit : CTA NectarCam {date.today().strftime("%B %d, %Y")}'

        self._output_table.add_column(Column(np.ones((self.npixels),dtype = bool),"is_valid",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(self.pixels_id,"pixel",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels),dtype = np.float64),"high gain",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels,2),dtype = np.float64),"high gain error",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels),dtype = np.float64),"low gain",unit = u.dimensionless_unscaled))
        self._output_table.add_column(Column(np.empty((self.npixels,2),dtype = np.float64),"low gain error",unit = u.dimensionless_unscaled))

    def run(self,**kwargs):
        log.info('running photo statistic method')

        self._output_table["high gain"] = self.gainHG
        self._output_table["low gain"] = self.gainLG
        #self._output_table["is_valid"] = self._SPEvalid

    def save(self,path,**kwargs) : 
        path = Path(path)
        os.makedirs(path,exist_ok = True)
        log.info(f'data saved in {path}')
        self._output_table.write(f"{path}/output_table.ecsv", format='ascii.ecsv',overwrite = kwargs.get("overwrite",False))

    def plot_correlation(self) : 
        mask = (self._output_table["high gain"]>20) * (self.SPEGain[self._mask_SPE]>0) * (self._output_table["high gain"]<80) * self._output_table['is_valid']
        a, b, r, p_value, std_err = linregress(self._output_table["high gain"][mask], self.SPEGain[self._mask_SPE][mask],'greater')
        x = np.linspace(self._output_table["high gain"][mask].min(),self._output_table["high gain"][mask].max(),1000)
        y = lambda x: a * x + b 
        with quantity_support() : 
            fig,ax = plt.subplots(1,1,figsize=(8, 6))
            ax.scatter(self._output_table["high gain"][mask],self.SPEGain[self._mask_SPE][mask],marker =".")
            ax.plot(x,y(x),color = 'red', label = f"linear fit,\n a = {a:.2e},\n b = {b:.2e},\n r = {r:.2e},\n p_value = {p_value:.2e},\n std_err = {std_err:.2e}")
            ax.plot(x,x,color = 'black',label = "y = x")
            ax.set_xlabel("Gain Photo stat (ADC)", size=15)
            ax.set_ylabel("Gain SPE fit (ADC)", size=15)
            #ax.set_xlim(xmin = 0)
            #ax.set_ylim(ymin = 0)

            ax.legend(fontsize=15)
            return fig

    @property
    def npixels(self) : return self._pixels_id.shape[0]

    @property
    def pixels_id(self) : return self._pixels_id

    @property
    def sigmaPedHG(self) : return np.std(self._Pedcharge_hg ,axis = 0) * np.sqrt(self.__coefCharge_FF_Ped)

    @property
    def sigmaChargeHG(self) : return np.std(self._FFcharge_hg - self.meanPedHG, axis = 0)

    @property
    def meanPedHG(self) : return np.mean(self._Pedcharge_hg ,axis = 0) * self.__coefCharge_FF_Ped

    @property
    def meanChargeHG(self) : return np.mean(self._FFcharge_hg  - self.meanPedHG, axis = 0)

    @property
    def BHG(self) : 
        min_events = np.min((self._FFcharge_hg.shape[0],self._Pedcharge_hg.shape[0]))
        upper = (np.power(self._FFcharge_hg.mean(axis = 1)[:min_events] - self._Pedcharge_hg.mean(axis = 1)[:min_events] * self.__coefCharge_FF_Ped - self.meanChargeHG.mean(),2)).mean(axis = 0)
        lower =  np.power(self.meanChargeHG.mean(),2)#np.power(self.meanChargeHG,2)#np.power(self.meanChargeHG.mean(),2)
        return np.sqrt(upper/lower)

    @property
    def gainHG(self) : 
        return ((np.power(self.sigmaChargeHG,2) - np.power(self.sigmaPedHG,2) - np.power(self.BHG * self.meanChargeHG,2))
                                /(self.meanChargeHG * (1 + np.power(self.SPEResolution[self._mask_SPE],2))))
    

    @property
    def sigmaPedLG(self) : return np.std(self._Pedcharge_lg ,axis = 0) * np.sqrt(self.__coefCharge_FF_Ped)

    @property
    def sigmaChargeLG(self) : return np.std(self._FFcharge_lg  - self.meanPedLG,axis = 0)

    @property
    def meanPedLG(self) : return np.mean(self._Pedcharge_lg,axis = 0) * self.__coefCharge_FF_Ped

    @property
    def meanChargeLG(self) : return np.mean(self._FFcharge_lg - self.meanPedLG,axis = 0)

    @property
    def BLG(self) : 
        min_events = np.min((self._FFcharge_lg.shape[0],self._Pedcharge_lg.shape[0]))
        upper = (np.power(self._FFcharge_lg.mean(axis = 1)[:min_events] - self._Pedcharge_lg.mean(axis = 1)[:min_events] * self.__coefCharge_FF_Ped - self.meanChargeLG.mean(),2)).mean(axis = 0)
        lower =  np.power(self.meanChargeLG.mean(),2) #np.power(self.meanChargeLG,2) #np.power(self.meanChargeLG.mean(),2)
        return np.sqrt(upper/lower)

    @property
    def gainLG(self) : return ((np.power(self.sigmaChargeLG,2) - np.power(self.sigmaPedLG,2) - np.power(self.BLG * self.meanChargeLG,2))
                                /(self.meanChargeLG * (1 + np.power(self.SPEResolution[self._mask_SPE],2))))



class PhotoStatGainFFandPed(PhotoStatGain):
    def __init__(self, FFRun, PedRun, SPEresults : str, maxevents : int = None, **kwargs) : 
        self._readFF(FFRun,maxevents,**kwargs)
        self._readPed(PedRun,maxevents,**kwargs)

        """
        if self.FFcharge.charge_hg.shape[1] != self.Pedcharge.charge_hg.shape[1] : 
            e = Exception("Ped run and FF run must have the same number of pixels")
            log.error(e,exc_info = True)
            raise e
        """

        self._readSPE(SPEresults)
        ##need to implement reshape of SPE results with FF and Ped pixels ids 
        self._reshape_all()

        """
        if (self.FFcharge.pixels_id.shape[0] != self._SPE_pixels_id.shape[0]) : 
            e = Exception("Ped run and FF run must have the same number of pixels as SPE fit results")
            log.error(e,exc_info = True)
            raise e

        if (self.FFcharge.pixels_id != self.Pedcharge.pixels_id).any() or (self.FFcharge.pixels_id != self._SPE_pixels_id).any() or (self.Pedcharge.pixels_id != self._SPE_pixels_id).any() : 
            e = DifferentPixelsID("Ped run, FF run and SPE run need to have same pixels id")
            log.error(e,exc_info = True)
            raise e
        else : 
            self._pixels_id = self.FFcharge.pixels_id
        """
        self.create_output_table()



class PhotoStatGainFF(PhotoStatGain):
    def __init__(self, FFRun, PedRun, SPEresults : str, maxevents : int = None, **kwargs) : 
        e = NotImplementedError("PhotoStatGainFF is not yet implemented")
        log.error(e, exc_info = True)
        raise e
        self._readFF(FFRun,maxevents,**kwargs)

        self._readSPE(SPEresults)
        
        """
        if self.FFcharge.pixels_id != self._SPE_pixels_id : 
            e = DifferentPixelsID("Ped run, FF run and SPE run need to have same pixels id")
            log.error(e,exc_info = True)
            raise e
        else : 
            self._pixels_id = self.FFcharge.pixels_id
        """

        self._reshape_all()

        self.create_output_table()

class PhotoStatisticMaker(GainMaker) :
