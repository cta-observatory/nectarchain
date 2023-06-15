#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code that implement Welford's algorithm for stats computation

This is inspired by the implementation done at https://github.com/a-mitani/welford
"""
import numpy as np
import warnings


class Stats:
    """class Stats
     Accumulator object for Welfords online / parallel variance algorithm.

    Attributes:
    """

    def __init__(self,shape=(1,)):
        """__init__

        Initialize with an optional data. 
        For the calculation efficiency, Welford's method is not used on the initialization process.

        """
        # Initialize instance attributes
        self.__shape = shape
        self.__count = np.zeros( shape, dtype=int )
        self.__m     = np.zeros( shape, dtype=float )
        self.__s     = np.zeros( shape, dtype=float )
        self.__min   = np.full( shape, np.inf )
        self.__max   = np.full( shape, -np.inf )
                 
    @property
    def shape(self):
        return self.__shape
    
    @property
    def count(self):
        return self.__count

    @property
    def mean(self):
        return self.__m

    @property
    def variance(self):
        return self.__getvars(ddof=1)

    @property
    def stddev(self):
        return np.sqrt(self.__getvars(ddof=1))

    @property
    def std(self):
        return self.stddev

    @property
    def min(self):
        return self.__min
    
    @property
    def max(self):
        return self.__max
    
    def get_lowcount_mask(self,mincount=3):
        return self.__count<mincount
    
    
    def add(self, element, validmask = None):
        """
        Add entry. If mask is given, it will only update the entry from mask
        """

        # Welford's algorithm
        if validmask is None:
            self.__count += 1
            delta = element - self.__m
            self.__m += delta / self.__count
            self.__s += delta * (element - self.__m)
            self.__min = np.minimum( self.__min, element )
            self.__max = np.maximum( self.__max, element )
        else:
            #print(validmask)
            #print(validmask.shape)
            self.__count[validmask] += 1
            #print("here 1")
            delta = element[validmask] - self.__m[validmask]
            #print("here 2")
            self.__m[validmask] += delta / self.__count[validmask]
            #print("here 3")
            self.__s[validmask] += delta * (element[validmask] - self.__m[validmask])
            #print("here 4")
            self.__min[validmask] = np.minimum( self.__min, element )[validmask]
            #print("here 5")
            self.__max[validmask] = np.maximum( self.__max, element )[validmask]
            #print("here 6")


    def merge(self, other):
        """Merge this accumulator with another one."""
        if self.__shape != other.__shape:
            raise ValueError(f"Trying to merge from a different shape this: {self.__shape}, given: {other.__shape}")        
        
        #with warnings.catch_warnings():
        count = self.__count + other.__count
        delta = self.__m - other.__m
        delta2 = delta * delta
        mean = (self.__count * self.__m + other.__count * other.__m) / count
        s = self.__s + other.__s + delta2 * (self.__count * other.__count) / count

        self.__count = count
        self.__m = mean
        self.__s = s
    

    def __getvars(self, ddof):
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore", category=RuntimeWarning)
        variance = self.__s / (self.__count - ddof)
        variance[ self.get_lowcount_mask(3) ] = np.nan
        return variance
    

class CameraStats(Stats):
    """class CameraStats
        Accumulator object for Welfords online / parallel variance algorithm, specialized for camera info
    """ 

    def __init__(self,shape=(2,1855), *args, **kwargs):
        super().__init__(shape,*args,**kwargs)



class CameraSampleStats(Stats):
    """class CameraSampleStats
        Accumulator object for Welfords online / parallel variance algorithm, specialized for trace info
    """ 

    def __init__(self,shape=(2,1855,60), *args, **kwargs):
        super().__init__(shape,*args,**kwargs)
