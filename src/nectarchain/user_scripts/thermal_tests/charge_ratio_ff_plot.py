import matplotlib.pyplot as plt
import numpy as np
import tables
from astropy.io import fits
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

# OPTIONS ARE 2: PED/CHARGE/PEDSTD .... and 1:AVERAGE/STD  -> if 2:PEDSTD change name of files to _ped
name_entity2 = "Charge"
name_entity = "average"

temperature = [15, 5, 0, -5]  # should be taken from logs

divide = [58, 58, 58, 58]
divide = np.array(divide)


fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, sharex=True, figsize=(8, 8), gridspec_kw={"height_ratios": [2, 1, 1]}
)  # Make the first one bigger

axis_font = {"fontname": "Arial", "size": "16"}


charge_laser_high = []
charge_laser_high_std = []
charge_laser_low = []
charge_laser_low_std = []

charge_ff_high = []
charge_ff_high_std = []
charge_ff_low = []
charge_ff_low_std = []

FF_MEAN_HG = []
FF_MEAN_LG = []


dirname = "./output/charge/output/"

# Open the fits file
for i in [3630, 3714, 3764, 3798]:
    with fits.open(
        "%sNectarCAM_Run%s/NectarCAM_Run%s_calib/NectarCAM_Run%s_Results.fits"
        % (dirname, i, i, i)
    ) as hdulist1:
        hdu = hdulist1["CHARGE-INTEGRATION-PED-ALL-AVERAGE-HIGH-GAIN"]
        column_data = hdu.data.field(0)

        charge_av = np.mean(column_data)
        print(charge_av)

        charge_std = np.std(column_data)

        charge_laser_high.append(charge_av)
        charge_laser_high_std.append(charge_std)
ax1.plot(
    temperature,
    charge_laser_high / divide,
    marker="o",
    color="cyan",
    label="Laser High Gain",
)


# Open the fits file
for i in [3630, 3714, 3764, 3798]:
    with fits.open(
        "%sNectarCAM_Run%s/NectarCAM_Run%s_calib/NectarCAM_Run%s_Results.fits"
        % (dirname, i, i, i)
    ) as hdulist1:
        hdu = hdulist1["CHARGE-INTEGRATION-PED-ALL-AVERAGE-LOW-GAIN"]
        column_data = hdu.data.field(0)

        charge_av = np.mean(column_data)
        print(charge_av)

        charge_std = np.std(column_data)

        charge_laser_low.append(charge_av)
        charge_laser_low_std.append(charge_std)
ax1.plot(
    temperature,
    charge_laser_low / divide,
    marker="o",
    color="magenta",
    label="Laser Low Gain",
)

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

ax2.plot(
    temperature,
    (np.array(charge_laser_high) / np.array(charge_laser_low)),
    marker="o",
    color="orange",
    label="Laser",
)
# ax2.plot(temperature, (np.array(charge_ff_high) / np.array(charge_ff_low)), marker='s', color = 'purple', label = "FF")


############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################


# read FF values
Runs = [4940, 4940, 4940, 4940]
for i in Runs:
    outputfile = (
        "/Users/hashkar/Desktop/nectarchain/src/nectarchain/user_scripts/thermal_tests/output/FF/FlatFieldTests/2FF_%s.h5"
        % str(i)
    )
    h5file = tables.open_file(outputfile)

    # for result in h5file.root.__members__:
    #    table = h5file.root[result]["FlatFieldContainer_0"][0]
    #    ff = table["FF_coef"]

    result = h5file.root.__members__[0]
    table = h5file.root[result]["FlatFieldContainer_0"][0]
    ff = table["FF_coef"]
    pixel = 0
    HG = 0
    LG = 1

    ff_pix_hg = ff[:, HG, pixel]
    mean_ff_pix_hg = np.mean(ff_pix_hg, axis=0)
    std_ff_pix_hg = np.std(ff_pix_hg, axis=0)

    ff_pix_lg = ff[:, LG, pixel]
    mean_ff_pix_lg = np.mean(ff_pix_lg, axis=0)
    std_ff_pix_lg = np.std(ff_pix_lg, axis=0)

    FF_MEAN_HG.append(mean_ff_pix_hg)
    print(FF_MEAN_HG)

    FF_MEAN_LG.append(mean_ff_pix_lg)
    print(FF_MEAN_LG)


ax3.plot(temperature, FF_MEAN_HG, marker="o", color="cyan", label="FF High Gain")
ax3.plot(temperature, FF_MEAN_LG, marker="o", color="magenta", label="FF Low Gain")


############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################


# plt.yscale("log")

ax1.set_ylabel("%s %s (p.e.)" % (name_entity2, name_entity), **axis_font)
ax2.set_ylabel("HG/LG ratio", **axis_font)
ax3.set_ylabel("FF Coeff", **axis_font)

plt.xlabel("Temperature", **axis_font)
ax1.legend(prop={"size": 16})
ax2.legend(prop={"size": 16})
ax3.legend(prop={"size": 16})

ax1.grid()
ax2.grid()
ax3.grid()
fig.savefig(
    "temp_%s_%s_all_intensity.png" % (name_entity2, name_entity),
    dpi=300,
    bbox_inches="tight",
)
plt.show()
############################################################################################################
