#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code that implement Welford's algorithm for stats computation

This is inspired by the implementation done at https://github.com/a-mitani/welford
"""
from copy import deepcopy

import numpy as np
from ctapipe_io_nectarcam import constants as nc


class Stats:
    """class Stats
    Accumulator object for Welfords online / parallel variance algorithm.


    Examples
    --------
    Example with only one variable
    >>> from nectarchain.utils.stats import Stats
    >>> s = Stats()
    >>> s.add(1)
    >>> s.add(2)
    >>> s.add(3)
    >>> s.mean
    2
    >>> s.std
    1
    """

    def __init__(self, shape=(1,)):
        """__init__

        Initialize with an optional data.
        For the calculation efficiency, Welford's method is not used on the
        initialization process.

        """
        # Initialize instance attributes
        self._shape = shape
        self._count = np.zeros(shape, dtype=int)
        self._m = np.zeros(shape, dtype=float)
        self._s = np.zeros(shape, dtype=float)
        self._min = np.full(shape, np.inf)
        self._max = np.full(shape, -np.inf)

    def __str__(self):
        """Return a formatted string of the current statistics."""
        infos = ""
        infos += f"mean: {self.mean}" + "\n"
        infos += f"std: {self.stddev}" + "\n"
        infos += f"min: {self.min}" + "\n"
        infos += f"max: {self.max}" + "\n"
        infos += f"count: {self.count}" + "\n"
        infos += f"shape: {self.shape}"
        return infos

    def __repr__(self):
        """Return the string representation of the statistics."""
        return self.__str__()

    def copy(self):
        """Create a deep copy of this Stats object.

        Returns
        -------
        Stats
            A deep copy.
        """
        return deepcopy(self)

    def __add__(self, other):
        """Merge two Stats objects and return a new copy.

        Parameters
        ----------
        other : Stats
            Another Stats object to merge.

        Returns
        -------
        Stats
            The merged copy.
        """
        r = self.copy()
        r.merge(other)
        return r

    def __iadd__(self, other):
        """Merge another Stats object in-place.

        Parameters
        ----------
        other : Stats
            Another Stats object to merge.

        Returns
        -------
        Stats
            Self after merging.
        """
        self.merge(other)
        return self

    @property
    def shape(self):
        """tuple: Shape of the accumulator arrays."""
        return self._shape

    @property
    def count(self):
        """np.ndarray: Number of samples per element."""
        return self._count

    @property
    def mean(self):
        """np.ndarray: Mean value per element."""
        return self._m

    @property
    def variance(self):
        """np.ndarray: Variance per element (sample variance, ddof=1)."""
        return self._getvars(ddof=1)

    @property
    def stddev(self):
        """np.ndarray: Standard deviation per element (ddof=1)."""
        return np.sqrt(self._getvars(ddof=1))

    @property
    def std(self):
        """np.ndarray: Alias for stddev."""
        return self.stddev

    @property
    def min(self):
        """np.ndarray: Minimum value per element."""
        return self._min

    @property
    def max(self):
        """np.ndarray: Maximum value per element."""
        return self._max

    def get_lowcount_mask(self, mincount=3):
        """Get a boolean mask for elements with low sample counts.

        Parameters
        ----------
        mincount : int, optional
            Minimum required count.

        Returns
        -------
        np.ndarray
            Boolean mask where count < mincount.
        """
        return self._count < mincount

    def add(self, element, validmask=None):
        """
        Add entry. If mask is given, it will only update the entry from mask
        element

        Parameters
        ----------
        element : np.array
            array of element to added to the stat object (must be similar shape as
            the Stats object)
        validmask : np.array
            array that indicate which value to use. Only element entry where
            validmask is True will be added. It must be a boolean array of the same
            shape as element

        """

        # Welford's algorithm
        if validmask is None:
            self._count += 1
            delta = element - self._m
            self._m += delta / self._count
            self._s += delta * (element - self._m)
            self._min = np.minimum(self._min, element)
            self._max = np.maximum(self._max, element)
        else:
            self._count[validmask] += 1
            delta = element[validmask] - self._m[validmask]
            self._m[validmask] += delta / self._count[validmask]
            self._s[validmask] += delta * (element[validmask] - self._m[validmask])
            self._min[validmask] = np.minimum(self._min, element)[validmask]
            self._max[validmask] = np.maximum(self._max, element)[validmask]

    def merge(self, other):
        """Merge this accumulator with another one.

        Parameters
        ----------
        other: nectarchain.utils.stats.Stats
            Another object of the same type which you want to combined the statistics
        """
        if self._shape != other._shape:
            raise ValueError(
                f"Trying to merge from a different shape this: {self._shape}, "
                f"given: {other._shape}"
            )

        # with warnings.catch_warnings():
        count = self._count + other._count
        delta = self._m - other._m
        delta2 = delta * delta
        mean = (self._count * self._m + other._count * other._m) / count
        s = self._s + other._s + delta2 * (self._count * other._count) / count

        self._count = count
        self._m = mean
        self._s = s

        self._min = np.minimum(self._min, other._min)
        self._max = np.maximum(self._max, other._max)

    def _getvars(self, ddof):
        """Compute variance using Welford's online algorithm.

        Parameters
        ----------
        ddof : int
            Delta degrees of freedom.

        Returns
        -------
        np.ndarray
            Variance array with low-count entries set to NaN.
        """
        # with warnings.catch_warnings():
        #    warnings.simplefilter("ignore", category=RuntimeWarning)
        variance = self._s / (self._count - ddof)
        variance[self.get_lowcount_mask(1 + ddof)] = np.nan
        return variance


class CameraStats(Stats):
    """class CameraStats
    Accumulator object for Welfords online / parallel variance algorithm,
    specialized for camera info
    """

    def __init__(self, shape=(nc.N_GAINS, nc.N_PIXELS), *args, **kwargs):
        """Initialize CameraStats.

        Parameters
        ----------
        shape : tuple, optional
            Shape of the accumulator. Defaults to (n_gains, n_pixels).
        *args
            Additional positional arguments passed to Stats.
        **kwargs
            Additional keyword arguments passed to Stats.
        """
        super().__init__(shape, *args, **kwargs)


class CameraSampleStats(Stats):
    """class CameraSampleStats
    Accumulator object for Welfords online / parallel variance algorithm,
    specialized for trace info

    Examples
    --------
    Cumulating the rawdata from a run to get the average waveform::

    >>> from nectarchain.utils.stats import CameraSampleStats
    >>> from ctapipe_io_nectarcam import NectarCAMEventSource
    >>> reader = NectarCAMEventSource(input_url='NectarCAM.Run4560.00??.fits.fz')
    >>> s = CameraSampleStats()
    >>> for event in reader:
    >>>     s.add(event.r0.tel[0].waveform,
    >>>           validmask=~evt.mon.tel[0].pixel_status.hardware_failing_pixels )
    >>> print(s.mean)
    """

    def __init__(self, shape=(nc.N_GAINS, nc.N_PIXELS, nc.N_SAMPLES), *args, **kwargs):
        """Initialize CameraSampleStats.

        Parameters
        ----------
        shape : tuple, optional
            Shape of the accumulator. Defaults to (n_gains, n_pixels, n_samples).
        *args
            Additional positional arguments passed to Stats.
        **kwargs
            Additional keyword arguments passed to Stats.
        """
        super().__init__(shape, *args, **kwargs)
