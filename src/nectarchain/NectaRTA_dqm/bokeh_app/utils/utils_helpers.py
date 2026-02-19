# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
This module stores the Bokeh webpage utility helpers for the RTA of NectarCAM.
"""

# imports
import logging
import numpy as np

__all__ = ["get_hillas_parameters", "hdf5Proxy", "hdf5GroupProxy"]

logger = logging.getLogger(__name__)


def get_hillas_parameters(file, parameterkeys, parameter_parentkeys, run_index=-1):
    """Get the Hillas parameters from the file.

    Parameters
    ----------
    file : hdf5 file
        File to retrieve the data.
    parameterkeys : dict
        Dictionnary of parameter keys to retrieve
        the Hillas parameters from ``file``.
    parameter_parentkeys: string
        Parent key for the parameters in the dictionary
    run_index : int, optional
        A file is constituted of multiple events,
        select an event in the stored file.
        Default is -1, resulting in the latest event of the run.

    Returns
    -------
    x : float
        Position of the center of the Hillas ellipse on the x axis.
    y : float
        Position of the center of the Hillas ellipse on the y axis.
    width : float
        Width of the Hillas ellipse.
    length : float
        Length of the Hillas ellipse.
    angle : float
        Angle between the mahor axis of the Hillas ellipse and the x axis.

    """

    try:
        x = file[parameter_parentkeys][parameterkeys["hillas_x_key"]][run_index]
        y = file[parameter_parentkeys][parameterkeys["hillas_y_key"]][run_index]
        width = file[parameter_parentkeys][parameterkeys["hillas_length_key"]][
            run_index
        ]
        height = file[parameter_parentkeys][parameterkeys["hillas_width_key"]][
            run_index
        ]
        angle = file[parameter_parentkeys][parameterkeys["hillas_phi_key"]][run_index]
        return x, y, width, height, angle
    except Exception as e:
        logger.warning("Failed to retrieve Hillas parameters:", e)


# hdf5 file handling for time sorting


def _leaf_paths_hdf5(file):
    """List unique leaf paths of the HDF5 tree.

    Parameters
    ----------
    file : HDF5 file
        hdf5 file to find the data.

    Returns
    -------
    leaf_paths : list
        Unique leaf paths of the file.

    """

    paths = []
    file.visititems(lambda name, obj: paths.append(name))
    leaf_paths = [
        p
        for p in paths
        if not any(other != p and other.startswith(p + "/") for other in set(paths))
    ]
    return leaf_paths


class hdf5GroupProxy(dict):
    """Proxy for an easier handling of the HDF5 group
    from ``h5py._hl.dataset.Dataset``.

    Parameters
    ----------
    input_data : HDF5 dataset
        HDF5 group to imitate.

    Returns
    -------
    out : hdf5GroupProxy
        Proxy of the dataset to imitate.

    Attributes
    ----------
    ogroup : HDF5 dataset
        Initial HDF5 dataset to imitate.
    shape : tuple of ints
        Length of the attributes in the proxy.
    childkeys : list of strings
        Keys of the proxy.

    Methods
    -------
    __init__(self, input_data, /)
        Instantiate the proxy from ``input_data``.

    mask(self, indexes, /)
        Mask the data according to the ``indexes``.

    sort_from_key(self, key, increasing=True, /)
        Sort the data by increasing/decreasing order from ``key`` values.

    """

    def __init__(self, input_data):
        """Instantiate the proxy of HDF5 group.

        Parameters
        ----------
        input_data : HDF5 dataset
            HDF5 group to imitate.

        Returns
        -------
        out : None

        """

        self.ogroup = input_data
        self.shape = input_data.shape

        if input_data.dtype.names is not None:
            self.childkeys = list(input_data.dtype.names)
            for key in self.childkeys:
                self[key] = np.asarray(input_data[key])
        else:
            self.childkeys = ["values"]
            self[self.childkeys[0]] = np.asarray(input_data)

    def mask(self, indexes):
        """Mask the data in ``self`` according to ``indexes``.

        Parameters
        ----------
        indexes : array_like
            Array of indexes to mask the data.

        Returns
        -------
        out : None

        """

        indexes = np.asarray(indexes)
        for key in self.childkeys:
            self[key] = self[key][indexes]

    def sort_from_key(self, key, increasing=True, axis=-1):
        """Sort the data in increasing/decreasing order of data in ``key``.

        Parameters
        ----------
        key : string
            Key to find the data to use for sorting.
        increasing : bool, optional
            If ``True``, sort by increasing order of the values given by ``key``.
            Sort them by decreasing order otherwise.
            Default is ``True``.
        axis : int or None, optional
            Axis along which to sort.
            The default is -1 (the last axis).
            If None, the flattened array is used.

        Returns
        -------
        out : None

        """

        increase = 1 if increasing else -1
        indexes = np.argsort(self[key], axis=axis)[::increase]
        self.mask(indexes)


class hdf5Proxy(dict):
    """Proxy for an easier handling of the HDF5 file
    from ``h5py._hl.files.File``.

    Parameters
    ----------
    input_file : HDF5 file
        HDF5 file to imitate.

    Returns
    -------
    out : hdf5Proxy
        Proxy of the file to imitate.

    Attributes
    ----------
    ofile : HDF5 file
        Initial HDF5 file to imitate.
    filename : string
        Filepath of the initial HDF5 file.
    parentkeys : list of strings
        Keys of the proxy.
    shape : int
        Length of the keys of the proxy.

    Methods
    -------
    __init__(self, input_data, /)
        Instantiate the proxy from ``input_file``.

    """

    def __init__(self, input_file):
        """Instantiate the proxy of the HDF5 file.

        Parameters
        ----------
        input_file : HDF5 file
            HDF5 file to imitate.

        Returns
        -------
        out : None

        """

        self.ofile = input_file
        self.filename = input_file.filename
        self.parentkeys = _leaf_paths_hdf5(input_file)
        self.shape = len(self.parentkeys)

        for parentkey in self.parentkeys:
            self[parentkey] = hdf5GroupProxy(input_file[parentkey])
