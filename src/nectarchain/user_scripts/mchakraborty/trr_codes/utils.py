import argparse
import os
import pickle
import sys

# from ctapipe.visualization import CameraDisplay
# from ctapipe.instrument import CameraGeometry
# from ctapipe_io_nectarcam import NectarCAMEventSource, constants
# from ctapipe.coordinates import CameraFrame, EngineeringCameraFrame
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ctapipe.io import read_table
from lmfit.models import Model

try:
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dateutil.parser import ParserError, parse
    from matplotlib import dates
    from tqdm import tqdm

except ImportError as err:
    print(err)


def get_bad_pixels_list():
    # List of modules and pixels to be rejected
    df = pd.read_json("bad_pix_module.json")
    modules_list = np.array(df.bad_module[0])
    pix_list = np.array(df.bad_pixel[0])

    pix_nos = np.arange(7)

    print("module_to_pix ", modules_list)
    module_to_pix = modules_list[:, None] * 7 + pix_nos
    combined = np.concatenate([pix_list, module_to_pix.ravel()])

    bad_pix_list = np.unique(combined)

    return bad_pix_list


def get_adc_to_pe(temperature):
    temp = np.array([-10, -5, 0, 5, 10, 14, 20, 25])
    runs = np.array([6853, 6775, 6718, 6589, 7191, 7000, 7123, 7066])

    if temperature in temp:
        run_no = runs[temp == temperature][0]
        gain_file_name = "gain_spe/gain_spe/FlatFieldSPENominalStdNectarCAM_run{}_maxevents1000_LocalPeakWindowSum_window_shift_4_window_width_16.h5".format(
            run_no
        )
        # print(gain_file_name)

        with h5py.File(gain_file_name, "r") as f:

            def print_hdf5_structure(name, obj):
                print(name)  # Prints the full path of each dataset/group

            f.visititems(print_hdf5_structure)

            try:
                toto = read_table(gain_file_name, path="/data/SPEfitContainer_0")

            except:
                toto = read_table(gain_file_name, path="/data/PhotosatatfitContainer_0")
                method = "Photostat"

            data = {
                "is_valid": toto["is_valid"][0],
                "high_gain_lw": [x[0] for x in toto["high_gain"][0]],
                "high_gain": [x[1] for x in toto["high_gain"][0]],
                "high_gain_up": [x[-1] for x in toto["high_gain"][0]],
                "pedestal_lw": [x[0] for x in toto["pedestal"][0]],
                "pedestal": [x[1] for x in toto["pedestal"][0]],
                "pedestal_up": [x[-1] for x in toto["pedestal"][0]],
                "pixels_id": toto["pixels_id"][0],
                # 'luminosity': toto['luminosity']
            }
            # print(data["high_gain_lw"])
            adc_to_pe = data["high_gain_lw"]
    else:
        adc_to_pe = 58.0

    return adc_to_pe


def get_ff_coeff(temperature, ff_model):
    temp = np.array([0, -5, 14, 25, 20, 10])
    runs = np.array([6672, 6729, 6954, 7020, 7077, 7144])

    if temperature in temp:
        run_no = runs[temp == temperature][0]
    else:
        return 1

    df = pd.read_csv("FF_coeff/FF_calibration_run{}.dat".format(run_no), sep=r"\s+")

    if ff_model == 1:
        ff = np.array(df["FF_coef_independent_way"])
    else:
        ff = np.array(df["FF_coef_model_way"])
    return ff


# =====HV Cut to be added===================
