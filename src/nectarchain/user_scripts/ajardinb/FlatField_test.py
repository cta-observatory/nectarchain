import os
import pathlib

import matplotlib.pyplot as plt

# %%
import numpy as np
from ctapipe.containers import Field
from ctapipe.core import Component
from ctapipe.core.traits import ComponentNameList, Integer, List, Dict, Tuple
from ctapipe.io import HDF5TableReader
from ctapipe_io_nectarcam import constants
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.coordinates import EngineeringCameraFrame

from nectarchain.data.container import (
    ArrayDataContainer,
    NectarCAMContainer,
    TriggerMapContainer,
)
from ctapipe.containers import EventType
from nectarchain.makers import EventsLoopNectarCAMCalibrationTool
from nectarchain.makers.component import ArrayDataComponent, NectarCAMComponent
from nectarchain.utils import ComponentUtils

from ctapipe.io import EventSource, EventSeeker
from ctapipe_io_nectarcam import NectarCAMEventSource
from astropy import units as u
from astropy.time import Time

from scipy import signal
from scipy.signal import find_peaks
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.stats import chi2, norm
import scipy
