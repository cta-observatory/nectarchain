import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

import yaml
import os
import copy
import numpy as np

import astropy.units as u
from astropy.units import UnitsError

__all__ = ["Parameter","Parameters"]

class Parameter() :
    def __init__(self, name, value, min = np.nan, max = np.nan, error = np.nan, unit = u.dimensionless_unscaled, frozen : bool = False):
        self.__name = name
        self.__value = value
        self.__error = error
        self.__min = min
        self.__max = max
        self.__unit = unit
        self.__frozen = frozen

    @classmethod
    def from_instance(cls,parameter):
        return cls(parameter.name,parameter.value,parameter.min,parameter.max,parameter.unit,parameter.frozen)

    def __str__(self):
        return f"name : {self.__name}, value : {self.__value}, error : {self.__error}, unit : {self.__unit}, min : {self.__min}, max : {self.__max},frozen : {self.__frozen}"


    @property
    def name(self) : return self.__name
    @name.setter
    def name(self,value) : self.__name = value

    @property
    def value(self) : return self.__value
    @value.setter
    def value(self,value) : self.__value = value

    @property
    def min(self):  return self.__min
    @min.setter
    def min(self,value) : self.__min = value
    
    @property
    def max(self):  return self.__max
    @max.setter
    def max(self,value) : self.__max = value

    @property
    def unit(self) : return self.__unit
    @unit.setter
    def unit(self,value) : self.__unit = value

    @property
    def error(self) : return self.__error
    @error.setter
    def error(self,value) : self.__error = value

    @property
    def frozen(self) : return self.__frozen
    @frozen.setter
    def frozen(self,value : bool) : self.__frozen = value

class Parameters() : 
    def __init__(self,parameters_liste : list = []) -> None:
        self.__parameters = copy.deepcopy(parameters_liste)
        
    
    def append(self,parameter : Parameter) :
        self.__parameters.append(parameter)

    def __getitem__(self,key) : 
        for parameter in self.__parameters : 
            if parameter.name == key : 
                return parameter
        return []
    
    def __str__(self):
        string=""
        for parameter in self.__parameters :
            string += str(parameter)+"\n"
        return string
    

    @property
    def parameters(self) : return self.__parameters

    @property
    def size(self) : return len(self.__parameters)

    @property
    def parnames(self) : return [parameter.name for parameter in self.__parameters]

    @property
    def parvalues(self) : return [parameter.value for parameter in self.__parameters]

    @property
    def unfrozen(self) : 
        parameters = Parameters()
        for parameter in self.__parameters : 
            if not(parameter.frozen) :
                parameters.append(parameter) 
        return parameters
