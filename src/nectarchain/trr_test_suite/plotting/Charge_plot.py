#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Charge normalization analysis script:
- Reads pedestal maps depending on (temperature, FF voltage, NSB intensity)
- Subtracts baseline dynamically for each run
- Applies FF correction, SPE gain normalization, and generates per-pixel maps
- Produces camera-average vs temperature plots
(all configurations on the same figure)
- Produces per-pixel slope maps per (Voltage, NSB)
- Excludes bad modules from calculations and plots
"""

import os
from collections import defaultdict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

# from DBHandler2 import DBInfos, to_datetime
from scipy.stats import linregress

# ================================
# INPUTS / PATHS / METADATA
# ================================

temp_map = {
    25: [
        7022,
        7028,
        7049,
        7037,
        7025,
        7058,
        7043,
        7026,
        7024,
        7029,
        7038,
        7047,
        7056,
        7021,
        7031,
        7040,
        7033,
        7042,
        7051,
        7060,
        7035,
        7044,
        7053,
        7062,
        7046,
        7055,
        7064,
        7030,
        7039,
        7048,
        7057,
        7023,
        7032,
        7041,
        7050,
        7059,
        7034,
        7052,
        7061,
        7027,
        7036,
        7045,
        7054,
        7063,
        7020,
    ],
    20: [
        7079,
        7085,
        7106,
        7094,
        7082,
        7115,
        7100,
        7083,
        7081,
        7086,
        7095,
        7104,
        7113,
        7078,
        7088,
        7097,
        7090,
        7099,
        7108,
        7117,
        7092,
        7101,
        7110,
        7119,
        7103,
        7112,
        7121,
        7087,
        7096,
        7105,
        7114,
        7080,
        7089,
        7098,
        7107,
        7116,
        7091,
        7109,
        7118,
        7084,
        7093,
        7102,
        7111,
        7120,
        7077,
    ],
    14: [
        6956,
        6962,
        6983,
        6971,
        6959,
        6992,
        6977,
        6960,
        6958,
        6963,
        6972,
        6981,
        6990,
        6955,
        6965,
        6974,
        6967,
        6976,
        6985,
        6994,
        6969,
        6978,
        6987,
        6996,
        6980,
        6989,
        6998,
        6964,
        6973,
        6982,
        6991,
        6957,
        6966,
        6975,
        6984,
        6993,
        6968,
        6986,
        6995,
        6961,
        6970,
        6979,
        6988,
        6997,
        6954,
    ],
    10: [
        7146,
        7152,
        7173,
        7161,
        7149,
        7183,
        7167,
        7150,
        7148,
        7153,
        7162,
        7171,
        7180,
        7145,
        7155,
        7164,
        7157,
        7166,
        7175,
        7185,
        7159,
        7168,
        7177,
        7187,
        7170,
        7179,
        7189,
        7154,
        7163,
        7172,
        7181,
        7147,
        7156,
        7165,
        7174,
        7184,
        7158,
        7176,
        7186,
        7151,
        7160,
        7169,
        7178,
        7188,
        7144,
    ],
    5: [
        6545,
        6551,
        6572,
        6560,
        6548,
        6581,
        6566,
        6549,
        6547,
        6552,
        6561,
        6570,
        6579,
        6544,
        6554,
        6563,
        6556,
        6565,
        6574,
        6583,
        6558,
        6567,
        6576,
        6585,
        6569,
        6578,
        6587,
        6553,
        6562,
        6571,
        6580,
        6546,
        6555,
        6564,
        6573,
        6582,
        6557,
        6575,
        6584,
        6550,
        6559,
        6568,
        6577,
        6586,
        6543,
    ],
    0: [
        6674,
        6680,
        6701,
        6689,
        6677,
        6710,
        6695,
        6678,
        6676,
        6681,
        6690,
        6699,
        6708,
        6673,
        6683,
        6692,
        6685,
        6694,
        6703,
        6712,
        6687,
        6696,
        6705,
        6714,
        6698,
        6707,
        6716,
        6682,
        6691,
        6700,
        6709,
        6675,
        6684,
        6693,
        6702,
        6711,
        6686,
        6704,
        6713,
        6679,
        6688,
        6697,
        6706,
        6715,
        6672,
    ],
    -5: [
        6731,
        6737,
        6758,
        6746,
        6734,
        6767,
        6752,
        6735,
        6733,
        6738,
        6747,
        6756,
        6765,
        6730,
        6740,
        6749,
        6742,
        6751,
        6760,
        6769,
        6744,
        6753,
        6762,
        6771,
        6755,
        6764,
        6773,
        6739,
        6748,
        6757,
        6766,
        6732,
        6741,
        6750,
        6759,
        6768,
        6743,
        6761,
        6770,
        6736,
        6745,
        6754,
        6763,
        6772,
        6729,
    ],
}

gain_map = {25: 7066, 20: 7123, 14: 7000, 10: 7191, 5: 6589, 0: 6718, -5: 6775}

ff_map = {25: 7020, 20: 7077, 14: 6954, 10: 7144, 5: 6674, 0: 6672, -5: 6729}

# Voltages and NSB grouped by temperature (same order as runs in temp_map)
vvalff_map = {
    25: [
        10,
        16,
        10,
        16,
        13,
        10,
        13,
        14,
        12,
        8,
        8,
        8,
        8,
        9,
        10,
        10,
        12,
        12,
        12,
        12,
        14,
        14,
        14,
        14,
        16,
        16,
        16,
        9,
        9,
        9,
        9,
        11,
        11,
        11,
        11,
        11,
        13,
        13,
        13,
        15,
        15,
        15,
        15,
        15,
        8,
    ],
    20: [
        10,
        16,
        10,
        16,
        13,
        10,
        13,
        14,
        12,
        8,
        8,
        8,
        8,
        9,
        10,
        10,
        12,
        12,
        12,
        12,
        14,
        14,
        14,
        14,
        16,
        16,
        16,
        9,
        9,
        9,
        9,
        11,
        11,
        11,
        11,
        11,
        13,
        13,
        13,
        15,
        15,
        15,
        15,
        15,
        8,
    ],
    14: [
        10,
        16,
        10,
        16,
        13,
        10,
        13,
        14,
        12,
        8,
        8,
        8,
        8,
        9,
        10,
        10,
        12,
        12,
        12,
        12,
        14,
        14,
        14,
        14,
        16,
        16,
        16,
        9,
        9,
        9,
        9,
        11,
        11,
        11,
        11,
        11,
        13,
        13,
        13,
        15,
        15,
        15,
        15,
        15,
        8,
    ],
    10: [
        10,
        16,
        10,
        16,
        13,
        10,
        13,
        14,
        12,
        8,
        8,
        8,
        8,
        9,
        10,
        10,
        12,
        12,
        12,
        12,
        14,
        14,
        14,
        14,
        16,
        16,
        16,
        9,
        9,
        9,
        9,
        11,
        11,
        11,
        11,
        11,
        13,
        13,
        13,
        15,
        15,
        15,
        15,
        15,
        8,
    ],
    5: [
        10,
        16,
        10,
        16,
        13,
        10,
        13,
        14,
        12,
        8,
        8,
        8,
        8,
        9,
        10,
        10,
        12,
        12,
        12,
        12,
        14,
        14,
        14,
        14,
        16,
        16,
        16,
        9,
        9,
        9,
        9,
        11,
        11,
        11,
        11,
        11,
        13,
        13,
        13,
        15,
        15,
        15,
        15,
        15,
        8,
    ],
    0: [
        10,
        16,
        10,
        16,
        13,
        10,
        13,
        14,
        12,
        8,
        8,
        8,
        8,
        9,
        10,
        10,
        12,
        12,
        12,
        12,
        14,
        14,
        14,
        14,
        16,
        16,
        16,
        9,
        9,
        9,
        9,
        11,
        11,
        11,
        11,
        11,
        13,
        13,
        13,
        15,
        15,
        15,
        15,
        15,
        8,
    ],
    -5: [
        10,
        16,
        10,
        16,
        13,
        10,
        13,
        14,
        12,
        8,
        8,
        8,
        8,
        9,
        10,
        10,
        12,
        12,
        12,
        12,
        14,
        14,
        14,
        14,
        16,
        16,
        16,
        9,
        9,
        9,
        9,
        11,
        11,
        11,
        11,
        11,
        13,
        13,
        13,
        15,
        15,
        15,
        15,
        15,
        8,
    ],
}

ivalnsb_map = {
    25: [
        0,
        0,
        39.8,
        10.6,
        0,
        78.8,
        20.4,
        0,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
    ],
    20: [
        0,
        0,
        39.8,
        10.6,
        0,
        78.8,
        20.4,
        0,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
    ],
    14: [
        0,
        0,
        39.8,
        10.6,
        0,
        78.8,
        20.4,
        0,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
    ],
    10: [
        0,
        0,
        39.8,
        10.6,
        0,
        78.8,
        20.4,
        0,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
    ],
    5: [
        0,
        0,
        39.8,
        10.6,
        0,
        78.8,
        20.4,
        0,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
    ],
    0: [
        0,
        0,
        39.8,
        10.6,
        0,
        78.8,
        20.4,
        0,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
    ],
    -5: [
        0,
        0,
        39.8,
        10.6,
        0,
        78.8,
        20.4,
        0,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        20.4,
        39.8,
        78.8,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        10.6,
        39.8,
        78.8,
        0,
        10.6,
        20.4,
        39.8,
        78.8,
        0,
    ],
}

dirname = "/Volumes/Untitled/runs/charges/"
gain_path = "/Users/hashkar/Desktop/20221108/SPEfit/thermal_gain"
pedestal_file = "/Users/hashkar/Desktop/20221108/pedestals/Baseline_Thermal_Tests.npz"
ff_dir = "/Users/hashkar/Desktop/20221108/FF/"
outdir = "./charge_comp_output"
os.makedirs(outdir, exist_ok=True)

METHOD = "LocalPeakWindowSum"
WINDOW = 16
SHIFT = 4
NSIGMA_BADPIX = 5

camera_geom = CameraGeometry.from_name("NectarCam").transform_to(
    EngineeringCameraFrame()
)


path = "/Users/hashkar/Desktop/20221108/runs"
db_data_path = "/Users/hashkar/Desktop/20221108/runs"


# ================================
# BAD MODULES DEFINITION
# ================================
BAD_MODULE_IDS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    18,
    30,
    43,
    57,
    72,
    88,
    105,
    123,
    158,
    175,
    191,
    206,
    220,
    233,
    245,
    256,
    264,
    263,
    262,
    261,
    260,
    259,
    258,
    257,
    246,
    234,
    221,
    207,
    192,
    58,
    44,
    31,
    19,
    8,
]


BAD_PIXELS_GAIN = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    1106,
    1107,
    1108,
    1109,
    1110,
    1111,
    1112,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    1225,
    1226,
    1227,
    1228,
    1229,
    1230,
    1231,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    296,
    301,
    302,
    303,
    304,
    305,
    306,
    307,
    308,
    309,
    310,
    311,
    312,
    313,
    314,
    1337,
    1338,
    1339,
    1340,
    1341,
    1342,
    1343,
    1344,
    1345,
    1346,
    1349,
    1347,
    1348,
    1350,
    399,
    400,
    401,
    402,
    403,
    404,
    405,
    406,
    407,
    408,
    409,
    410,
    411,
    412,
    1442,
    1443,
    1444,
    1445,
    1446,
    1447,
    1448,
    1449,
    1450,
    1451,
    1452,
    1453,
    1454,
    1455,
    504,
    505,
    506,
    507,
    508,
    509,
    510,
    1540,
    1541,
    1542,
    1543,
    1544,
    1545,
    1546,
    1547,
    1548,
    1549,
    1550,
    1551,
    1552,
    1553,
    1631,
    1632,
    1633,
    1634,
    1635,
    1636,
    1637,
    1638,
    1639,
    616,
    617,
    618,
    619,
    620,
    621,
    622,
    1640,
    1641,
    1642,
    1643,
    1644,
    1715,
    1716,
    1717,
    1718,
    1719,
    1720,
    1721,
    1722,
    1723,
    1724,
    1725,
    1726,
    1727,
    1728,
    735,
    736,
    737,
    738,
    739,
    740,
    741,
    1792,
    1793,
    1794,
    1795,
    1796,
    1797,
    1798,
    1799,
    1800,
    1801,
    1802,
    1803,
    1804,
    1805,
    1806,
    1807,
    1808,
    1809,
    1810,
    1811,
    1812,
    1813,
    1814,
    1815,
    1816,
    1817,
    1818,
    1819,
    1820,
    1821,
    1822,
    1823,
    1824,
    1825,
    1826,
    1827,
    1828,
    1829,
    1830,
    1831,
    1832,
    1833,
    1834,
    1835,
    1836,
    1837,
    1838,
    1839,
    1840,
    1841,
    1842,
    1843,
    1844,
    1845,
    1846,
    1847,
    1848,
    1849,
    1850,
    1851,
    1852,
    1853,
    1854,
    861,
    862,
    863,
    864,
    865,
    866,
    867,
]


def get_bad_pixels_from_modules(bad_module_ids):
    """
    Convert bad module IDs to bad pixel IDs.
    NectarCAM has 7 pixels per module:
    module M contains pixels [M*7, M*7+1, ..., M*7+6]
    """
    bad_pixels = set()
    for mod_id in bad_module_ids:
        for pixel_offset in range(7):
            bad_pixels.add(mod_id * 7 + pixel_offset)
    return bad_pixels


# Get all bad pixel IDs from bad modules
BAD_PIXELS_FROM_MODULES = get_bad_pixels_from_modules(BAD_MODULE_IDS)
print(f"Total bad pixels from bad modules: {len(BAD_PIXELS_FROM_MODULES)}")
bad_ids = None


def get_bad_hv_pixels_db(
    run, path, db_data_path, hv_tolerance=4.0, telid=0, verbose=False
):
    """
    Identify pixels for which |measured HV - target HV| > hv_tolerance.

    Parameters
    ----------
    run : int
        Run number.
    path : str
        Path to rawdata (needed by DBInfos.init_from_run).
    db_data_path : str
        Path to sqlite monitoring database.
    hv_tolerance : float
        Allowed HV deviation relative to target HV (default: 4 V).
    telid : int
        Telescope ID.
    verbose : bool
        Print detailed information.

    Returns
    -------
    set
        Pixel IDs failing the HV criterion.
    """
    """
    # Load DBInfos
    dbinfos = DBInfos.init_from_run(run, path=path, dbpath=db_data_path, verbose=False)
    dbinfos.connect("monitoring_drawer_temperatures", "monitoring_channel_voltages")

    try:
        hv_measured = (
            dbinfos.tel[telid]
            .monitoring_channel_voltages
            .voltage
            .datas
        )

        hv_target = (
            dbinfos.tel[telid]
            .monitoring_channel_voltages
            .target_voltage
            .datas
        )  # input here the target voltage

    except Exception as e:
        print(f"Error retrieving HV data for run {run}, telid {telid}: {e}")
        return set()

    # Mean values per pixel (ignore obviously wrong measurement frames)
    hv_measured_mean = np.nanmean(hv_measured, where=hv_measured > 400, axis=-1)
    hv_target_mean   = np.nanmean(hv_target,  where=hv_target > 400,   axis=-1)

    # Apply Vincent's condition: |measured - target| > tolerance
    deviation = np.abs(hv_measured_mean - hv_target_mean)
    bad_pixels = set(np.where(deviation > hv_tolerance)[0])

    if verbose:
        print(
            f"Run {run}: {len(bad_pixels)} bad pixels "
            f"(|HV_meas - HV_target| > {hv_tolerance} V)"
        )
        print("Bad pixel IDs:", bad_pixels)
    """
    bad_pixels = {
        50,
        310,
        353,
        412,
        638,
        737,
        742,
        793,
        827,
        864,
        866,
        1354,
        1530,
        1702,
        1841,
        1842,
    }
    return bad_pixels


# ================================
# HELPER FUNCTIONS
# ================================
def load_charge_file(filename, dataset_name="FLATFIELD"):
    with h5py.File(filename, "r") as f:
        print("filename:", filename)
        container = f.get("data/ChargesContainer_0")
        dataset = container.get(dataset_name)
        dataset0 = dataset[0]
        pixels_id = dataset0["pixels_id"]
        charges_hg = dataset0["charges_hg"].squeeze()
    return np.array(pixels_id), np.array(charges_hg)


def read_gain_file(spe_run, window, shift, method, dirname):
    filename = (
        f"FlatFieldSPENominalStdNectarCAM_run{spe_run}_"
        f"maxevents5000_{method}_window_shift_{shift}_"
        f"window_width_{window}.h5"
    )

    spe_filename = os.path.join(dirname, filename)

    with h5py.File(spe_filename, "r") as f:
        data = f["/data/SPEfitContainer_0"]
        raw_high_gain = data["high_gain"][0]
        gains = [
            float(entry[0]) if hasattr(entry, "__getitem__") else float(entry)
            for entry in raw_high_gain
        ]
        pixels_id = np.array(data["pixels_id"][0], dtype=int)
    return pixels_id, np.array(gains, dtype=float)


def detect_bad_pixels(pixels_id, charges, nsigma=5):
    vals = np.array(charges, dtype=float)
    mean_val = np.nanmean(vals)
    std_val = np.nanstd(vals)
    bad_mask = np.abs(vals - mean_val) > nsigma * std_val
    return set(np.array(pixels_id)[bad_mask])


def compute_slopes_and_stats(df_pixels, runs, temperatures):
    pixel_gains = df_pixels.pivot_table(
        index="Pixel", columns="Run", values="Gain"
    ).reindex(columns=runs)
    pivot_pixel_ids = pixel_gains.index.values
    pixel_gains_array = pixel_gains.values
    slopes, mean_gain, std_gain = [], [], []
    for i in range(pixel_gains_array.shape[0]):
        gains = pixel_gains_array[i, :]
        mask = ~np.isnan(gains)
        x = np.array(temperatures)[mask]
        y = gains[mask]
        slope = linregress(x, y).slope if len(x) > 1 and np.ptp(x) > 0 else np.nan
        slopes.append(slope)
        mean_gain.append(np.nanmean(y))
        std_gain.append(np.nanstd(y))
    return (
        pivot_pixel_ids,
        np.array(slopes),
        np.array(mean_gain),
        np.array(std_gain),
        pixel_gains_array,
    )


def plot_camera(values, pixel_ids, camera_geom, title, fig_path, cmap="viridis"):
    """Plot camera display, automatically masking bad pixels from modules"""
    # Filter out bad pixels from modules
    good_mask = np.array([pid not in bad_ids for pid in pixel_ids])
    filtered_pixel_ids = pixel_ids[good_mask]
    filtered_values = np.array(values)[good_mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = CameraDisplay(
        geometry=camera_geom[filtered_pixel_ids],
        image=np.array(filtered_values, dtype=float),
        cmap=cmap,
        ax=ax,
    )
    disp.add_colorbar(label=title)
    plt.title(title)
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_pixel_histogram(values, pixel_ids, title, fig_path, bins=50, xlabel="Value"):
    """
    Plot histogram of per-pixel values, excluding bad pixels.

    Parameters
    ----------
    values : array-like
        Per-pixel values (same order as pixel_ids)
    pixel_ids : array-like
        Pixel IDs corresponding to values
    title : str
        Plot title
    fig_path : str
        Output figure path
    bins : int
        Number of histogram bins
    xlabel : str
        X-axis label
    """

    # Exclude bad pixels
    good_mask = np.array([pid not in bad_ids for pid in pixel_ids])
    good_values = np.array(values, dtype=float)[good_mask]

    # Remove NaNs
    good_values = good_values[np.isfinite(good_values)]

    plt.figure(figsize=(7, 5))
    plt.hist(good_values, bins=bins, histtype="stepfilled", alpha=0.7)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("Number of pixels", fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.tick_params(axis="both", which="minor", labelsize=12)
    # plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def load_ff_file(temp, ff_map, ff_dir):
    ff_run = ff_map.get(temp)
    if ff_run is None:
        raise ValueError(f"No FF run defined for temp {temp}")
    ff_filename = os.path.join(ff_dir, f"FF_calibration_run{ff_run}.dat")
    df_ff = pd.read_csv(ff_filename, delim_whitespace=True)
    ff_dict = dict(zip(df_ff["pixel_id"], df_ff["FF_coef_independent_way"]))
    return ff_dict


# ================================
# LOAD PEDESTALS
# ================================
ped_data = np.load(pedestal_file)
ped_temperature = ped_data["temperature"]
ped_baseline = ped_data["baseline"]
ped_pixel_ids = ped_data["pixel_ids"]
ped_vvalff = ped_data["vvalff"]
ped_ivalnsb = ped_data["ivalnsb"]

# ================================
# PROCESSING
# ================================
avg_charges_config = defaultdict(list)
all_pixel_records = []

for temp, runs in temp_map.items():
    voltages = vvalff_map[temp]
    nsbs = ivalnsb_map[temp]

    for i, run in enumerate(runs):
        print(run)
        Voo = voltages[i]
        Inn = nsbs[i]

        # pedestal
        t_idx = int(np.argmin(np.abs(ped_temperature - temp)))
        iff = int(np.argmin(np.abs(ped_vvalff - Voo)))
        jnsb = int(np.argmin(np.abs(ped_ivalnsb - Inn)))
        ped_vals = ped_baseline[t_idx, iff, jnsb, :]

        ped_dict = dict(zip(ped_pixel_ids, ped_vals))

        # FF correction
        try:
            ff_dict = load_ff_file(temp, ff_map, ff_dir)
        except Exception:
            ff_dict = {pid: 1.0 for pid in ped_pixel_ids}

        # SPE gain
        gain_run = gain_map[temp]
        try:
            spe_pixels, spe_gains = read_gain_file(
                gain_run, WINDOW, SHIFT, METHOD, gain_path
            )
        except Exception:
            continue
        pixel_to_gain = dict(zip(spe_pixels, spe_gains))

        # load charges
        charge_filename = os.path.join(
            dirname,
            (
                f"ChargesNectarCAMCalibration_run{run}_"
                f"maxevents5000_{METHOD}_window_shift_{SHIFT}_"
                f"window_width_{WINDOW}.h5"
            ),
        )

        if not os.path.exists(charge_filename):
            continue
        pixels_id, charges_hg = load_charge_file(charge_filename)

        # Check if there are multiple events (i.e., charges_hg is 2D)
        if charges_hg.ndim > 1:
            charges_hg = np.mean(charges_hg, axis=0)
        else:
            charges_hg = charges_hg

        print(temp, run)
        print(np.mean(charges_hg))

        # pedestal subtraction + FF + SPE normalization
        charges_pedcorr = np.array(
            [
                charges_hg[i] - ped_dict.get(pid, np.nan) * WINDOW
                for i, pid in enumerate(pixels_id)
            ]
        )
        charges_ffcorr = np.array(
            [
                charges_pedcorr[i] * ff_dict.get(pid, 1.0)
                for i, pid in enumerate(pixels_id)
            ]
        )
        charges_normalized = np.array(
            [
                charges_ffcorr[i] / pixel_to_gain.get(pid, np.nan)
                for i, pid in enumerate(pixels_id)
            ]
        )

        # 1. Get HV-bad pixels
        hv_bad_pixels = get_bad_hv_pixels_db(
            run,
            path=path,
            db_data_path=db_data_path,
            hv_tolerance=4.0,
            telid=0,
            verbose=True,
        )

        # 2. Get module-bad pixels
        module_bad_pixels = set(BAD_PIXELS_FROM_MODULES)
        gain_bad_pixels = set(BAD_PIXELS_GAIN)

        # Combine all non-statistical bad pixels
        non_statistical_bad = hv_bad_pixels.union(module_bad_pixels).union(
            gain_bad_pixels
        )

        print(f"Fixed bad pixels (HV + modules + gain): {len(non_statistical_bad)}")

        # 3. Select ONLY the remaining good pixels
        remaining_pixel_ids = [p for p in pixels_id if p not in non_statistical_bad]
        remaining_charges = [
            charges_normalized[i]
            for i, p in enumerate(pixels_id)
            if p not in non_statistical_bad
        ]

        # 4. Compute statistical bad pixels only on remaining ones
        stat_bad_relative = detect_bad_pixels(
            remaining_pixel_ids, remaining_charges, nsigma=NSIGMA_BADPIX
        )

        # Map the relative indices back to pixel IDs
        stat_bad_pixels = {remaining_pixel_ids[i] for i in stat_bad_relative}

        print(f"Statistical bad pixels (after exclusions): {len(stat_bad_pixels)}")

        # 5. Final union of ALL bad pixels
        bad_ids = non_statistical_bad.union(stat_bad_pixels)

        print(f"Total bad pixels: {len(bad_ids)}")

        # store per-pixel (excluding bad pixels)
        for pid, charge in zip(pixels_id, charges_normalized):
            if pid not in bad_ids:
                all_pixel_records.append(
                    {
                        "Run": run,
                        "Temperature": temp,
                        "Pixel": int(pid),
                        "V": Voo,
                        "NSB": Inn,
                        "Charge": float(charge),
                    }
                )

        # store camera average (excluding bad pixels)
        good_charges = [
            g for pid, g in zip(pixels_id, charges_normalized) if pid not in bad_ids
        ]
        cam_avg = np.nanmean(good_charges)
        cam_std = np.nanstd(good_charges)
        avg_charges_config[(Voo, Inn)].append(
            {
                "Run": run,
                "Temperature": temp,
                "CameraAvgCharge": cam_avg,
                "CameraStd": cam_std,
            }
        )

df_all_pixels = pd.DataFrame(all_pixel_records)

# ================================
# PLOT CAMERA AVG VS TEMP ALL ON SAME FIGURE
# ================================
plt.figure(figsize=(20, 12))
for (V, I), records in avg_charges_config.items():
    df_cfg = pd.DataFrame(records).sort_values("Temperature")

    # compute mean and standard error instead of std
    cam_avgs = df_cfg["CameraAvgCharge"].values
    # Count only good pixels for SEM calculation
    n_good_pixels = len(ped_pixel_ids) - len(bad_ids)
    cam_errs = np.array(
        [g / np.sqrt(n_good_pixels) for g in df_cfg["CameraStd"].values]
    )

    plt.errorbar(
        df_cfg["Temperature"],
        cam_avgs,
        yerr=cam_errs,
        fmt="o-",
        capsize=3,
        label=f"V={V} V, NSB={I} mA",
    )

plt.xlabel("Temperature (°C)", fontsize=16)
plt.ylabel("Camera Average Normalized Charge (p.e.)", fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=14)
plt.tick_params(axis="both", which="minor", labelsize=12)

# plt.title(f"Camera Average (excluding {len(BAD_MODULE_IDS)} bad modules)")
handles, labels = plt.gca().get_legend_handles_labels()
sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
sorted_labels, sorted_handles = zip(*sorted_handles_labels)
plt.grid(True)
# Use multiple columns if there are many items
plt.legend(
    sorted_handles,
    sorted_labels,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=16,  # smaller font
    ncol=2,  # number of columns in the legend
    frameon=True,  # optional: draw frame
    borderaxespad=0.5,  # padding
)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "CameraAvgCharge_AllConfigs_SEM.png"), dpi=150)
plt.show()
plt.close()

# ================================
# PER-PIXEL SLOPE VS TEMP PER (V,NSB)
# ================================
for (V, I), subset in df_all_pixels.groupby(["V", "NSB"]):
    subset_runs = subset["Run"].unique()
    subset_temps = [
        next(temp for temp, runs in temp_map.items() if run in runs)
        for run in subset_runs
    ]

    pivot_pixel_ids, slopes, mean_charge, std_charge, _ = compute_slopes_and_stats(
        subset.rename(columns={"Charge": "Gain"}),
        runs=subset_runs,
        temperatures=subset_temps,
    )

    method_outdir = os.path.join(outdir, f"Slopes_V{V}_NSB{I}")
    os.makedirs(method_outdir, exist_ok=True)

    plot_camera(
        slopes,
        pivot_pixel_ids,
        camera_geom,
        title=f"Per-Pixel Slope vs Temp (V={V}V, NSB={I}) [p.e./°C]",
        fig_path=os.path.join(method_outdir, f"Slopes_V{V}_NSB{I}.png"),
        cmap="coolwarm",
    )

    plot_pixel_histogram(
        values=slopes,
        pixel_ids=pivot_pixel_ids,
        title=f"Pixel slope distribution (V={V}V, NSB={I})",
        fig_path=os.path.join(method_outdir, f"SlopeHist_V{V}_NSB{I}.png"),
        bins=60,
        xlabel="Slope [p.e./°C]",
    )


# ================================
# PLOT AVERAGE PEDESTALS PER TEMPERATURE
# ================================
avg_ped_per_temp = []
std_ped_per_temp = []
temps_sorted = sorted(temp_map.keys())

for temp in temps_sorted:
    runs = temp_map[temp]
    voltages = vvalff_map[temp]
    nsbs = ivalnsb_map[temp]

    ped_vals_all = []
    for i, run in enumerate(runs):
        Voo = voltages[i]
        Inn = nsbs[i]

        # get indices for pedestal arrays
        t_idx = int(np.argmin(np.abs(ped_temperature - temp)))
        iff = int(np.argmin(np.abs(ped_vvalff - Voo)))
        jnsb = int(np.argmin(np.abs(ped_ivalnsb - Inn)))
        ped_vals = ped_baseline[t_idx, iff, jnsb, :]

        # Filter out bad pixels from modules
        good_pixel_mask = np.array([pid not in bad_ids for pid in ped_pixel_ids])
        ped_vals_good = ped_vals[good_pixel_mask]

        ped_vals_all.append(ped_vals_good)

    # flatten all good pixel pedestals for this temperature across runs
    ped_vals_all = np.concatenate(ped_vals_all)
    avg_ped_per_temp.append(np.mean(ped_vals_all))
    std_ped_per_temp.append(np.std(ped_vals_all))

# plot
plt.figure(figsize=(8, 5))
plt.errorbar(temps_sorted, avg_ped_per_temp, yerr=std_ped_per_temp, fmt="o-", capsize=4)
plt.xlabel("Temperature (°C)")
plt.ylabel("Average Pedestal (ADC)")
plt.title(
    f"Camera Average Pedestal vs Temperature "
    f"(excluding {len(BAD_MODULE_IDS)} bad modules)"
)

plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "AvgPedestal_vs_Temperature.png"), dpi=150)
plt.show()

print("\n=== Summary ===")
print(f"Bad modules excluded: {len(BAD_MODULE_IDS)}")
print(f"Bad pixels excluded: {len(BAD_PIXELS_FROM_MODULES)}")
print(f"Total pixels analyzed: {len(ped_pixel_ids) - len(BAD_PIXELS_FROM_MODULES)}")
