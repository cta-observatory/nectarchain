from argparse import ArgumentError
import numpy as np
from matplotlib import pyplot as plt
import copy
from pathlib import Path
import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from ctapipe.visualization import CameraDisplay
from ctapipe.coordinates import CameraFrame,EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.image.extractor import (FullWaveformSum,
    FixedWindowSum,
    GlobalPeakWindowSum,
    LocalPeakWindowSum,
    SlidingWindowMaxSum,
    NeighborPeakWindowSum,
    BaselineSubtractedNeighborPeakWindowSum,
    TwoPassWindowSum)

from ctapipe_io_nectarcam import NectarCAMEventSource
from ctapipe.io import EventSource, EventSeeker

from astropy.table import Table


from .waveforms import WaveformsContainer

__all__ = ['ChargeContainer']

list_ctapipe_charge_extractor = ["FullWaveformSum",
                        "FixedWindowSum",
                        "GlobalPeakWindowSum",
                        "LocalPeakWindowSum",
                        "SlidingWindowMaxSum",
                        "NeighborPeakWindowSum",
                        "BaselineSubtractedNeighborPeakWindowSum",
                        "TwoPassWindowSum"]



class ChargeContainer() : 
    def __init__(self,charge_hg,charge_lg,peak_hg,peak_lg,run_number) : 
        self.charge_hg = charge_hg
        self.charge_lg = charge_lg
        self.peak_hg = peak_hg
        self.peak_lg = peak_lg
        self.__run_number = run_number

    @classmethod
    def from_waveform(cls,waveformContainer : WaveformsContainer,method : str = "FullWaveformSum",**kwargs) : 
        log.info(f"computing hg charge with {method} method")
        charge_hg,peak_hg = ChargeContainer.compute_charge(waveformContainer,0,method,**kwargs)
        log.info(f"computing lg charge with {method} method")
        charge_lg,peak_lg = ChargeContainer.compute_charge(waveformContainer,1,method,**kwargs)
        return cls(charge_hg,charge_lg,peak_hg,peak_lg,waveformContainer.run_number)

    def write(self,path : Path,**kwargs) : 
        log.info(f"saving in {path}")
        table = Table(self.charge_hg)
        table.write(Path(path)/f"charge_hg_run{self.run_number}.csv",format='csv',overwrite=kwargs.get('overwrite',False))
        table = Table(self.charge_lg)
        table.write(Path(path)/f"charge_lg_run{self.run_number}.csv",format='csv',overwrite=kwargs.get('overwrite',False))
        table = Table(self.peak_hg)
        table.write(Path(path)/f"peak_hg_run{self.run_number}.csv",format='csv',overwrite=kwargs.get('overwrite',False))
        table = Table(self.peak_lg)
        table.write(Path(path)/f"peak_lg_run{self.run_number}.csv",format='csv',overwrite=kwargs.get('overwrite',False))

    @classmethod
    def from_file(cls,path : Path,run_number : int,**kwargs) : 
        log.info(f"loading in {path} run number {run_number}")
        table = Table.read(Path(path)/f"charge_hg_run{run_number}.csv")
        charge_hg = np.array([table[colname] for colname in table.colnames]).T
        table = Table.read(Path(path)/f"charge_lg_run{run_number}.csv")
        charge_lg = np.array([table[colname] for colname in table.colnames]).T
        table = Table.read(Path(path)/f"peak_hg_run{run_number}.csv")
        peak_hg = np.array([table[colname] for colname in table.colnames]).T
        table = Table.read(Path(path)/f"peak_lg_run{run_number}.csv")
        peak_lg = np.array([table[colname] for colname in table.colnames]).T
        return cls(charge_hg,charge_lg,peak_hg,peak_lg,run_number)

        
    @staticmethod 
    def compute_charge(waveformContainer : WaveformsContainer,channel : int,method : str = "FullWaveformSum" ,**kwargs) : 
        if not(method in list_ctapipe_charge_extractor) :
            raise ArgumentError(f"method must be in {list_ctapipe_charge_extractor}")
        ImageExtractor = eval(method)(waveformContainer.reader.subarray)
        if channel == 0:
            return ImageExtractor(waveformContainer.wfs_hg,waveformContainer.TEL_ID,channel)
        elif channel == 1:
            return ImageExtractor(waveformContainer.wfs_lg,waveformContainer.TEL_ID,channel)
        else :
            raise ArgumentError("channel must be 0 or 1")

    def histo_hg(self,n_bins : int = 1000) -> np.ndarray:
        hist = np.array([np.histogram(self.charge_hg.T[i],bins=n_bins)[0] for i in range(self.charge_hg.shape[1])])
        charge = np.array([np.histogram(self.charge_hg.T[i],bins=n_bins)[1] for i in range(self.charge_hg.shape[1])])
        charge_edges = np.array([np.mean(charge.T[i:i+1],axis = 0) for i in range(charge.shape[1]-1)]).T
        return np.array((hist,charge_edges))

    def histo_lg(self,n_bins: int = 1000) -> np.ndarray:
        hist = np.array([np.histogram(self.charge_lg.T[i],bins=n_bins)[0] for i in range(self.charge_lg.shape[1])])
        charge = np.array([np.histogram(self.charge_lg.T[i],bins=n_bins)[1] for i in range(self.charge_lg.shape[1])])
        charge_edges = np.array([np.mean(charge.T[i:i+1],axis = 0) for i in range(charge.shape[1]-1)]).T
        return np.array((hist,charge_edges))

    @property
    def run_number(self) : return self.__run_number
