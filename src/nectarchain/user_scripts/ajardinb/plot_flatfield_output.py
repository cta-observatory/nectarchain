import os

import matplotlib.pyplot as plt
import numpy as np
import tables

# Define the global environment variable NECTARCAMDATA (folder where are the runs)
run_number = 4940
os.environ["NECTARCAMDATA"] = "./20231222/FlatFieldTests"

outputfile = os.environ["NECTARCAMDATA"] + "/2FF_{}.h5".format(run_number)

h5file = tables.open_file(outputfile)

# for result in h5file.root.__members__:
#    table = h5file.root[result]["FlatFieldContainer_0"][0]
#    ff = table["FF_coef"]

result = h5file.root.__members__[0]
table = h5file.root[result]["FlatFieldContainer_0"][0]
ff = table["FF_coef"]

# Histogramm of FF coef for one pixel, one gain chanel
pix = 0
HG = 0
ff_pix = ff[:, HG, pix]
mean_ff_pix = np.mean(ff_pix, axis=0)
std_ff_pix = np.std(ff_pix, axis=0)

fig = plt.figure(figsize=(5, 4))
plt.hist(
    ff_pix,
    20,
    label="pixel %s \n$\mu$=%0.3f, \nstd=%0.3f" % (pix, mean_ff_pix, std_ff_pix),
)
plt.xlabel("FF coefficient (HG)")
plt.legend()
plt.savefig(os.environ["NECTARCAMDATA"] + "/run{}_FFpix{}.png".format(run_number, pix))


# Histogramm of the mean FF coef for all pixels of the camera, one gain chanel
ff_pix = ff[:, HG, :]
mean_ff_pix = np.mean(ff_pix, axis=0, where=np.isinf(ff_pix) == False)

mean_ff_cam = np.mean(mean_ff_pix, axis=0, where=np.isnan(mean_ff_pix) == False)
std_ff_cam = np.std(mean_ff_pix, axis=0, where=np.isnan(mean_ff_pix) == False)

fig = plt.figure(figsize=(5, 4))
plt.hist(mean_ff_pix, 50, label="$\mu$=%0.3f, \nstd=%0.3f" % (mean_ff_cam, std_ff_cam))
plt.axvline(mean_ff_cam, ls="--")

plt.xlabel("mean FF coefficient for all pixels (HG)")
plt.legend()
plt.savefig(os.environ["NECTARCAMDATA"] + "/run{}_FFcam.png".format(run_number))
