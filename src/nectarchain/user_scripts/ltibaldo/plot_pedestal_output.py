import os

import matplotlib.pyplot as plt
import numpy as np
import tables

filename = os.environ["NECTARCAMDATA"] + "/runs/pedestal_3938.h5"
pixel_id = 132
sample = 13
nsamples = 60

container_name = "NectarCAMPedestalContainer"

h5file = tables.open_file(filename)

# arrays to store slice results
pedestals = np.zeros([len(h5file.root.__members__) - 1, nsamples])
pedestals_std = np.zeros([len(h5file.root.__members__) - 1, nsamples])
tmin = np.array([])
tmax = np.array([])

# fill results for plotting
i = 0
for result in h5file.root.__members__:
    table = h5file.root[result]["NectarCAMPedestalContainer"][0]
    wf = table["pedestal_mean_hg"][table["pixels_id"] == pixel_id][0]
    std = table["pedestal_std_hg"][table["pixels_id"] == pixel_id][0]
    if result == "data_combined":
        pedestal_combined = wf
        pedestal_std = std
    else:
        pedestals[i] = wf
        pedestals_std[i] = std
        tmin = np.append(tmin, table["ucts_timestamp_min"])
        tmax = np.append(tmax, table["ucts_timestamp_max"])
        i += 1

tmean = 0.5 * (tmin + tmax)

fig1 = plt.figure()
ax1 = plt.axes()
ax1.set_title(f"Pixel {pixel_id}")
ax1.set_xlabel("sample (ns)")
ax1.set_ylabel("pedestal (ADC counts)")
for s in range(len(pedestals)):
    ax1.plot(pedestals[s], color="0.5", alpha=0.2)
ax1.plot(pedestal_combined, color="r")

fig2 = plt.figure()
ax2 = plt.axes()
ax2.set_title(f"Pixel {pixel_id} Sample {sample}")
ax2.set_xlabel("UCTS timestamp")
ax2.set_ylabel("pedestal (ADC counts)")
ax2.errorbar(
    tmean,
    pedestals[:, sample],
    xerr=[tmean - tmin, tmax - tmean],
    yerr=[pedestals_std[:, sample]],
    fmt="o",
    color="k",
    capsize=0.0,
)
ax2.axhspan(
    pedestal_combined[sample] - pedestal_std[sample],
    pedestal_combined[sample] + pedestal_std[sample],
    facecolor="r",
    alpha=0.2,
    zorder=0,
)

plt.show()
