import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import numpy as np
from ctapipe.containers import Field

from .core import NectarCAMContainer

__all__ = ["GainContainer", "SPEfitContainer"]


class GainContainer(NectarCAMContainer):
    """
    Class representing a GainContainer.

    This class is a subclass of NectarCAMContainer and provides additional fields and methods specific to gain calibration data.

    Attributes:
        is_valid (np.ndarray): Array of booleans indicating the validity of each gain value.
        high_gain (np.ndarray): Array of high gain values.
        low_gain (np.ndarray): Array of low gain values.
        pixels_id (np.ndarray): Array of pixel IDs.

    Methods:
        from_hdf5(cls, path): Class method to read a GainContainer from an HDF5 file.
            Parameters:
                path (str or Path): The path to the HDF5 file.

            Yields:
                GainContainer: The container from the data in the HDF5 file.
    """

    is_valid = Field(type=np.ndarray, dtype=bool, ndim=1, description="is_valid")
    high_gain = Field(
        type=np.ndarray, dtype=np.float64, ndim=2, description="high gain"
    )
    low_gain = Field(type=np.ndarray, dtype=np.float64, ndim=2, description="low gain")
    pixels_id = Field(type=np.ndarray, dtype=np.uint16, ndim=1, description="pixel ids")

    @classmethod
    def from_hdf5(cls, path):
        """Class method to read a GainContainer from an HDF5 file.

        Args:
           path (str or Path): The path to the HDF5 file.

        Yields:
            GainContainer: The container from the data in the HDF5 file.
        """
        return super(__class__, cls)._container_from_hdf5(path, container_class=cls)


class SPEfitContainer(GainContainer):
    """
    Class representing a SPEfitContainer.

    This class is a subclass of GainContainer and provides additional fields specific to single photoelectron (SPE) fit data.

    Attributes:
        likelihood (np.ndarray): Array of likelihood values.
        p_value (np.ndarray): Array of p-values.
        pedestal (np.ndarray): Array of pedestal values.
        pedestalWidth (np.ndarray): Array of pedestal widths.
        resolution (np.ndarray): Array of resolution values.
        luminosity (np.ndarray): Array of luminosity values.
        mean (np.ndarray): Array of mean values.
        n (np.ndarray): Array of n values.
        pp (np.ndarray): Array of pp values.
    """

    likelihood = Field(
        type=np.ndarray, dtype=np.float64, ndim=1, description="likelihood"
    )
    p_value = Field(type=np.ndarray, dtype=np.float64, ndim=1, description="p_value")
    pedestal = Field(type=np.ndarray, dtype=np.float64, ndim=2, description="pedestal")
    pedestalWidth = Field(
        type=np.ndarray, dtype=np.float64, ndim=2, description="pedestalWidth"
    )
    resolution = Field(
        type=np.ndarray, dtype=np.float64, ndim=2, description="resolution"
    )
    luminosity = Field(
        type=np.ndarray, dtype=np.float64, ndim=2, description="luminosity"
    )
    mean = Field(type=np.ndarray, dtype=np.float64, ndim=2, description="mean")
    n = Field(type=np.ndarray, dtype=np.float64, ndim=2, description="n")
    pp = Field(type=np.ndarray, dtype=np.float64, ndim=2, description="pp")
