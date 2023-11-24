import logging
import typing as t

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np
from astropy.table import Column
import copy

from ctapipe_io_nectarcam.containers import NectarCAMDataContainer

from ctapipe.core.traits import Unicode,Integer,Bool,List,classes_with_traits,Dict,ComponentNameList,Path
from ctapipe.core.component import Component

from ...data.container import merge_map_ArrayDataContainer,SPEfitContainer
from .chargesComponent import ChargesComponent
from .gainComponent import GainNectarCAMComponent
from .chargesComponent import ChargesComponent
from .spe import SPEHHVStdalgorithm,SPEHHValgorithm,SPECombinedalgorithm
from ...utils import ComponentUtils

__all__ = ["FlatFieldSingleHHVSPENectarCAMComponent","FlatFieldSingleHHVSPEStdNectarCAMComponent","FlatFieldCombinedSPEStdNectarCAMComponent"]

class FlatFieldSingleHHVSPENectarCAMComponent(GainNectarCAMComponent):
    SPEfitalgorithm = Unicode("SPEHHValgorithm",
                              help = "The Spe fit method to be use",
                              read_only = True,
    ).tag(config = True)

    SubComponents = copy.deepcopy(GainNectarCAMComponent.SubComponents)
    SubComponents.default_value = ["ChargesComponent",f"{SPEfitalgorithm.default_value}"]
    SubComponents.read_only = True



    #Windows_lenght = Integer(40,
    #                        read_only = True,
    #                        help = "The windows leght used for the savgol filter algorithm",
    #).tag(config = True)
    #
    #Order = Integer(2,
    #                read_only = True,
    #                help = "The order of the polynome used in the savgol filter algorithm",
    #).tag(config = True)

    asked_pixels_id = List(default_value=None,
                           allow_none = True,
                           help = "The pixels id where we want to perform the SPE fit",
    ).tag(config = True)
    
    #nproc = Integer(8,
    #                help = "The Number of cpu used for SPE fit",
    #).tag(config = True)
#
    #chunksize = Integer(1,
    #                help = "The chunk size for multi-processing",
    #).tag(config = True)
#
    #multiproc = Bool(True,
    #                help = "flag to active multi-processing",
    #).tag(config = True)

    #method = Unicode(default_value = "FullWaveformSum",
    #                 help = "the charge extraction method",
#
    #                 ).tag(config = True)
    #
    #extractor_kwargs = Dict(default_value = {},
    #                        help = "The kwargs to be pass to the charge extractor method",
    #                        ).tag(config = True)

    

    # constructor
    def __init__(self, subarray, config=None, parent=None,*args, **kwargs) -> None:
        chargesComponent_kwargs = {}
        self._SPEfitalgorithm_kwargs = {}
        other_kwargs = {}
        chargesComponent_configurable_traits = ComponentUtils.get_configurable_traits(ChargesComponent)
        SPEfitalgorithm_configurable_traits = ComponentUtils.get_configurable_traits(eval(self.SPEfitalgorithm))

        for key in kwargs.keys() : 
            if key in chargesComponent_configurable_traits.keys() : 
                chargesComponent_kwargs[key] = kwargs[key]
            elif key in SPEfitalgorithm_configurable_traits.keys() : 
                self._SPEfitalgorithm_kwargs[key] = kwargs[key]
            else :
                other_kwargs[key] = kwargs[key]

        super().__init__(subarray = subarray,config = config, parent = parent,*args, **other_kwargs)
        self.chargesComponent = ChargesComponent(subarray = subarray,config = config, parent = parent,*args, **chargesComponent_kwargs)
        self._chargesContainers = None


    def __call__(
            self,
            event : NectarCAMDataContainer, 
            *args, 
            **kwargs
        ):
        self.chargesComponent(event = event, *args, **kwargs) 


    def finish(self,*args,**kwargs) : 
        is_empty = False
        if self._chargesContainers is None : 
            self._chargesContainers = self.chargesComponent.finish(*args,**kwargs)
            if len(self._chargesContainers.containers.keys()) != 0 : 
                is_empty = False
                self._chargesContainers = merge_map_ArrayDataContainer(self._chargesContainers)
            else : 
                log.warning("empty chargesContainer in output")
                is_empty = True
        if not(is_empty) : 
            spe_fit = eval(self.SPEfitalgorithm).create_from_chargesContainer(self._chargesContainers,parent = self,**self._SPEfitalgorithm_kwargs)
            fit_output = spe_fit.run(pixels_id = self.asked_pixels_id, *args, **kwargs)
            conv_rate = np.sum(spe_fit.results.is_valid)/len(self.asked_pixels_id)
            self.log.info(f"convergence rate : {conv_rate}")
            return spe_fit.results
        else : 
            return None






class FlatFieldSingleHHVSPEStdNectarCAMComponent(FlatFieldSingleHHVSPENectarCAMComponent):
    SPEfitalgorithm = Unicode("SPEHHVStdalgorithm",
                              help = "The Spe fit method to be use",
                              read_only = True,
    ).tag(config = True)
    
    SubComponents = copy.deepcopy(GainNectarCAMComponent.SubComponents)
    SubComponents.default_value = ["ChargesComponent",f"{SPEfitalgorithm.default_value}"]
    SubComponents.read_only = True


class FlatFieldCombinedSPEStdNectarCAMComponent(FlatFieldSingleHHVSPEStdNectarCAMComponent) :
    SPEfitalgorithm = Unicode("SPECombinedalgorithm",
                              help = "The Spe fit method to be use",
                              read_only = True,
    ).tag(config = True)

    
    SubComponents = copy.deepcopy(GainNectarCAMComponent.SubComponents)
    SubComponents.default_value = ["ChargesComponent",f"{SPEfitalgorithm.default_value}"]
    SubComponents.read_only = True

    def __init__(self, subarray, config=None, parent=None,*args, **kwargs) -> None:
        super().__init__(subarray = subarray,config = config, parent = parent,*args,**kwargs)
    

