#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code that implement Welford's algorithm for stats computation

This is inspired by the implementation done at https://github.com/a-mitani/welford
"""
import numpy as np
import warnings

from numba import njit, prange


@njit(parallel=False)
def fast_add(rcount,rm,rs,rmin,rmax,element,validmask):
    nbins = rcount.shape[0]
    mask_exist = validmask is not None
    for b in prange(nbins):
        if mask_exist and not validmask[b]:
            continue
        rcount[b] += 1 #self.__count[validmask] += 1
        el = element[b]
        delta = el - rm[b]  # delta = element[validmask] - self.__m[validmask]
        rm[b] += delta/rcount[b] # self.__m[validmask] += delta / self.__count[validmask]
        rs[b] += delta * ( el - rm[b] ) # self.__s[validmask] += delta * (element[validmask] - self.__m[validmask])
        rmin[b] = element[b] if element[b]<rmin[b] else rmin[b]
        rmax[b] = element[b] if element[b]>rmax[b] else rmax[b]
#        rmin[b] = min(rmin[b],el) # self.__min[validmask] = np.minimum( self.__min, element )[validmask]
#        rmax[b] = max(rmax[b],el) # self.__max[validmask] = np.maximum( self.__max, element )[validmask]

        

class Stats:
    """class Stats
     Accumulator object for Welfords online / parallel variance algorithm.

    Attributes:
    """

    def __init__(self,shape=(1,),use_jit=False):
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
        self.use_jit = use_jit 

    def __str__(self):

        infos = list()
        infos.append( f'Shape:   {self.shape}' )
        infos.append( f'Entries: {self.count}' )
        infos.append( f'Mean:    {self.mean}'  )
        infos.append( f'Std Dev: {self.std}'   )
        infos.append( f'Error:   {self.error}' )
        infos.append( f'Min:     {self.min}'   )
        infos.append( f'Max:     {self.max}'   )
        
        str = "#### Statistics:"
        for i in infos:
            str += f'\t{i}\n'
        return str            
        

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
    

    @property
    def error(self):
        """
        Return approximate gaussian error (neglecting student coeffs)
        """
        return self.stddev/np.sqrt(self.count)
    
    def add(self, element, validmask = None):
        """
        Add entry. If mask is given, it will only update the entry from mask
        """

        # Welford's algorithm
        if self.use_jit:
            fast_add(self.__count.ravel(),self.__m.ravel(),self.__s.ravel(),self.__min.ravel(),self.__max.ravel(),element.ravel(), validmask.ravel() if validmask is not None else None)
        elif validmask is None:
            self.__count += 1
            delta = element - self.__m
            self.__m += delta / self.__count
            self.__s += delta * (element - self.__m)
            self.__min = np.minimum( self.__min, element )
            self.__max = np.maximum( self.__max, element )
        else:
            self.__count[validmask] += 1
            delta = element[validmask] - self.__m[validmask]
            self.__m[validmask] += delta / self.__count[validmask]
            self.__s[validmask] += delta * (element[validmask] - self.__m[validmask])
            self.__min[validmask] = np.minimum( self.__min, element )[validmask]
            self.__max[validmask] = np.maximum( self.__max, element )[validmask]


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
    
        self.__min = np.minimum(self.__min, other.__min)
        self.__max = np.maximum(self.__max, other.__max)

    def __getvars(self, ddof):
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore", category=RuntimeWarning)
        variance = self.__s / (self.__count - ddof)
        variance[ self.get_lowcount_mask(1+ddof) ] = np.nan
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


class Covariance:
    """class compute covariance on the fly
    """

    def __init__(self):
        self._xy_stat = Stats()
        self._x_stat = Stats()
        self._y_stat = Stats()
    
    def add(self,x,y):
        self._xy_stat.add(x*y)
        self._x_stat.add(x)
        self._y_stat.add(y)

    @property
    def xy_stat(self):
        return self._xy_stat
    @property
    def x_stat(self):
        return self._x_stat
    @property
    def y_stat(self):
        return self._y_stat
    
    #@property
    def get_covariance(self):
        N = self._xy_stat.count
        Nm1 = N-1
        mean_xy = self._xy_stat.mean
        mean_x = self._x_stat.mean
        mean_y = self._y_stat.mean
        return (N/Nm1)*(mean_xy - mean_x*mean_y) if N>0 else 0.
    
    
    def get_correlation_coefficient(self):
        cov_xy   = self.get_covariance()
        stddev_x = self._x_stat.stddev
        stddev_y = self._y_stat.stddev
        return cov_xy/(stddev_x*stddev_y)
    
    def get_alpha(self):
        cov_xy = self.get_covariance()
        var_x = self._x_stat.variance
        return cov_xy/var_x if var_x!=0. else 0.
    
    def get_beta(self):
        mean_x = self._x_stat.mean
        mean_y = self._y_stat.mean
        return mean_y - self.get_alpha()*mean_x    
    
