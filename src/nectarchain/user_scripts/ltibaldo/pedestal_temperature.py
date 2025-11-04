import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tables

# Imports from ctapipe
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe_io_nectarcam.constants import N_PIXELS
from scipy import stats

sns.set(style="whitegrid")

rms_threshold = 47

data = [
    {"T": -5, "runs": np.arange(6882, 6891)},
    {"T": 0, "runs": np.arange(6672, 6681)},
    {"T": 5, "runs": np.arange(6543, 6552)},
    {"T": 10, "runs": np.arange(7144, 7153)},
    {"T": 14, "runs": np.arange(6954, 6963)},
    {"T": 20, "runs": np.arange(7077, 7086)},
    {"T": 25, "runs": np.arange(7020, 7029)},
]

pixel_display = [100, 144, 663, 1058, 491, 631, 656, 701, 756, 757, 1612, 1629]

fill_value = np.nan
slopes_ped_hg = np.full(N_PIXELS, fill_value=fill_value)
slopes_rms_hg = np.full(N_PIXELS, fill_value=fill_value)
slopes_ped_lg = np.full(N_PIXELS, fill_value=fill_value)
slopes_rms_lg = np.full(N_PIXELS, fill_value=fill_value)

removed_pixels = 0
flagged_pixels = 0

for pixel_id in np.arange(N_PIXELS):
    print(f"Working on pixel {pixel_id}")
    # fill panda dataframe
    temperatures = []
    channels = []
    avgpeds = []
    pedsrms = []

    for dataset in data:
        for run_number in dataset["runs"]:
            run_number = int(run_number)
            filename = (
                os.environ["NECTARCAMDATA"] + f"/runs/pedestal_cfilt3s_{run_number}.h5"
            )
            h5file = tables.open_file(filename)
            table = h5file.root["data_combined"]["NectarCAMPedestalContainer_0"][0]
            for channel in ["hg", "lg"]:
                pedestal = table[f"pedestal_mean_{channel}"][
                    table["pixels_id"] == pixel_id
                ][0]
                rms = table[f"pedestal_charge_std_{channel}"][
                    table["pixels_id"] == pixel_id
                ][0]
                avgped = np.average(pedestal)
                combrms = rms
                temperatures.append(dataset["T"])
                channels.append(channel)
                avgpeds.append(avgped)
                pedsrms.append(combrms)
            h5file.close()

    if np.any(np.isnan(avgpeds)) or np.any(np.isnan(combrms)):
        flagged_pixels += 1
        print("Bad pixel")
    elif np.any(np.array(pedsrms) > rms_threshold):  # filter on NSB OFF
        removed_pixels += 1
        print("Removed pixel")
    else:
        df = pd.DataFrame(
            {
                "T (deg)": temperatures,
                "channel": channels,
                "pedestal (ADC)": avgpeds,
                "pedestal width (ADC)": pedsrms,
            }
        )

        for k, q in enumerate(["pedestal (ADC)", "pedestal width (ADC)"]):
            if pixel_id in pixel_display:
                lm = sns.lmplot(
                    x="T (deg)",
                    y=q,
                    hue="channel",
                    data=df,
                    scatter_kws={"alpha": 0.0},
                    legend=False,
                )

                ax = lm.axes[0][0]
                fig = ax.get_figure()
                fig.subplots_adjust(top=0.9)
                # clean legend to avoid duplication
                for _artist in ax.lines + ax.collections + ax.patches + ax.images:
                    _artist.set_label(s=None)

                sns.violinplot(
                    x="T (deg)",
                    y=q,
                    hue="channel",
                    split=True,
                    native_scale=True,
                    data=df,
                    ax=ax,
                )

            # extract slope and intercept for display and further usage
            colors = sns.color_palette()
            for s, channel in enumerate(["hg", "lg"]):
                filtered_df = df[df["channel"] == channel]
                slope, intercept, r_value, pv, se = stats.linregress(
                    filtered_df["T (deg)"],
                    filtered_df[q],
                )
                if k == 0 and s == 0:
                    slopes_ped_hg[pixel_id] = slope
                elif k == 0 and s == 1:
                    slopes_ped_lg[pixel_id] = slope
                elif k == 1 and s == 0:
                    slopes_rms_hg[pixel_id] = slope
                elif k == 1 and s == 1:
                    slopes_rms_lg[pixel_id] = slope
                if pixel_id in pixel_display:
                    ax.annotate(
                        f"y = {slope:.4f} T + {intercept:.4f}",
                        (0.05, 0.85 - s * 0.05),
                        color=colors[s],
                        xycoords="axes fraction",
                    )

            if pixel_id in pixel_display:
                ax.set_title(f"Pixel {pixel_id}, NSB OFF")
                fig.savefig(
                    f"{os.environ['NECTARCHAIN_FIGURES']}/pixel{pixel_id}_{k}.png"
                )
                del fig
                plt.close()

print(f"{flagged_pixels} pixels were flagged as bad by the pedestal estimation tool")
print(
    f"{removed_pixels} pixels were removed because they had a pedestal RMS "
    f"exceeding {rms_threshold}"
)

camera = CameraGeometry.from_name("NectarCam-003")
camera = camera.transform_to(EngineeringCameraFrame())

for k in range(2):  # quantity to treat
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(wspace=0.35, hspace=0.3)
    for s in range(2):  # HG and LG channels
        if k == 0 and s == 0:
            slopes = slopes_ped_hg
        elif k == 0 and s == 1:
            slopes = slopes_ped_lg
        elif k == 1 and s == 0:
            slopes = slopes_rms_hg
        elif k == 1 and s == 1:
            slopes = slopes_rms_lg
        # histograms
        ax = plt.subplot(2, 2, s + 1)
        plt.hist(slopes, bins="auto", histtype="step")
        if k == 1:
            ax.set_yscale("log")
        if k == 0 and s == 0:
            ax.set_title("Pedestal slope (ADC/deg), hg, NSB OFF")
        elif k == 0 and s == 1:
            ax.set_title("Pedestal slope (ADC/deg), lg, NSB OFF")
        elif k == 1 and s == 0:
            ax.set_title("Pedestal width slope (ADC/deg), hg, NSB OFF")
        elif k == 1 and s == 1:
            ax.set_title("Pedestal width slope (ADC/deg), lg, NSB OFF")
        plt.axvline(np.nanmean(slopes), linestyle=":")
        plt.axvline(np.nanmean(slopes) - np.nanstd(slopes), linestyle=":")
        plt.axvline(np.nanmean(slopes) + np.nanstd(slopes), linestyle=":")
        ax.annotate(
            r"mean: {:.3f} $\pm$ {:.3f}".format(np.nanmean(slopes), np.nanstd(slopes)),
            (0.05, 0.85 - s * 0.05),
            xycoords="axes fraction",
        )
        # camera display
        ax = plt.subplot(2, 2, s + 3)
        disp = CameraDisplay(camera)
        disp.image = slopes
        if k == 0:
            disp.set_limits_minmax(-0.3, 0.3)
        elif k == 1:
            disp.set_limits_minmax(-0.07, 0.07)
        disp.cmap = "Spectral_r"
        disp.add_colorbar()
        disp.update()
        ax.set_title("")
    fig.savefig(f"{os.environ['NECTARCHAIN_FIGURES']}/camera_{k}.png")
