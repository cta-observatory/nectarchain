import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tables
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from scipy import stats

sns.set(style="whitegrid")

rms_threshold = 47

pedfiles = [  # OK
    {
        "T": -5,
        "data": [
            {"NSB": 0, "runs": np.arange(6882, 6891)},
            {"NSB": 10.6, "runs": np.arange(6891, 6900)},
            {"NSB": 20.4, "runs": np.arange(6900, 6909)},
            {"NSB": 39.8, "runs": np.arange(6909, 6918)},
            {"NSB": 78.8, "runs": np.arange(6918, 6927)},
        ],
    },
    {
        "T": 0,
        "data": [  # OK
            {"NSB": 0, "runs": np.arange(6672, 6681)},
            {"NSB": 10.6, "runs": np.arange(6681, 6690)},
            {"NSB": 20.4, "runs": np.arange(6690, 6699)},
            {"NSB": 39.8, "runs": np.arange(6699, 6708)},
            {"NSB": 78.8, "runs": np.arange(6708, 6717)},
        ],
    },
    {
        "T": 5,
        "data": [  # OK
            {"NSB": 0, "runs": np.arange(6543, 6552)},
            {"NSB": 10.6, "runs": np.arange(6552, 6561)},
            {"NSB": 20.4, "runs": np.arange(6661, 6570)},
            {"NSB": 39.8, "runs": np.arange(6570, 6579)},
            {"NSB": 78.8, "runs": np.arange(6579, 6588)},
        ],
    },
    {
        "T": 10,
        "data": [  # OK
            {"NSB": 0, "runs": np.arange(7144, 7153)},
            {"NSB": 10.6, "runs": np.arange(7153, 7162)},
            {"NSB": 20.4, "runs": np.arange(7162, 7171)},
            {"NSB": 39.8, "runs": np.arange(7171, 7180)},
            {"NSB": 78.8, "runs": np.append([7180], np.arange(7182, 7190))},
        ],
    },
    {
        "T": 14,
        "data": [  # OK
            {"NSB": 0, "runs": np.arange(6954, 6963)},
            {"NSB": 10.6, "runs": np.arange(6963, 6972)},
            {"NSB": 20.4, "runs": np.arange(6972, 6981)},
            {"NSB": 39.8, "runs": np.arange(6981, 6990)},
            {"NSB": 78.8, "runs": np.arange(6990, 6999)},
        ],
    },
    {
        "T": 20,
        "data": [  # OK
            {"NSB": 0, "runs": np.arange(7077, 7086)},
            {"NSB": 10.6, "runs": np.arange(7086, 7095)},
            {"NSB": 20.4, "runs": np.arange(7095, 7104)},
            {"NSB": 39.8, "runs": np.arange(7104, 7112)},
            {"NSB": 78.8, "runs": np.arange(7113, 7122)},
        ],
    },
    {
        "T": 25,
        "data": [  # OK
            {"NSB": 0, "runs": np.arange(7020, 7029)},
            {"NSB": 10.6, "runs": np.arange(7029, 7038)},
            {"NSB": 20.4, "runs": np.arange(7038, 7047)},
            {"NSB": 39.8, "runs": np.arange(7047, 7056)},
            {"NSB": 78.8, "runs": np.arange(7056, 7064)},
        ],
    },
]

outfigroot = os.environ["FIGDIR"]
pixel_display = [100, 144, 240, 723, 816, 1034, 1516]
fill_value = np.nan

pixel_process = np.arange(1855)
# pixel_process = pixel_display

slopes_width_hg = np.full([len(pedfiles), 1855], fill_value=fill_value)
slopes_width_lg = np.full([len(pedfiles), 1855], fill_value=fill_value)

for itemp, uberset in enumerate(pedfiles):
    temp = uberset["T"]
    print("Analyzing data for temperature {}".format(temp))

    # create output directory if needed
    outdir = os.path.join(outfigroot, "NSB_T{}deg".format(temp))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    slopes_ped_hg = np.full(1855, fill_value=fill_value)
    slopes_rms_hg = np.full(1855, fill_value=fill_value)
    slopes_ped_lg = np.full(1855, fill_value=fill_value)
    slopes_rms_lg = np.full(1855, fill_value=fill_value)

    removed_pixels = 0
    flagged_pixels = 0

    for pixel_id in pixel_process:
        print("Working on pixel {}".format(pixel_id))
        # fill panda dataframe
        nsbs = []
        channels = []
        avgpeds = []
        pedsrms = []

        for dataset in uberset["data"]:
            for run_number in dataset["runs"]:
                run_number = int(run_number)
                filename = os.environ[
                    "NECTARCAMDATA"
                ] + "/runs/pedestal_cfilt3s_{}.h5".format(run_number)
                h5file = tables.open_file(filename)
                table = h5file.root["data_combined"]["NectarCAMPedestalContainer_0"][0]
                for channel in ["hg", "lg"]:
                    pedestal = table["pedestal_mean_{}".format(channel)][
                        table["pixels_id"] == pixel_id
                    ][0]
                    rms = table["pedestal_charge_std_{}".format(channel)][
                        table["pixels_id"] == pixel_id
                    ][0]
                    avgped = np.average(pedestal)
                    combrms = rms
                    nsbs.append(dataset["NSB"])
                    channels.append(channel)
                    avgpeds.append(avgped)
                    pedsrms.append(combrms)
                h5file.close()

        if np.any(np.isnan(avgpeds)) or np.any(np.isnan(combrms)):
            flagged_pixels += 1
            print("Bad pixel")
        elif np.any(
            np.array(pedsrms)[np.array(nsbs) == 0.0] > rms_threshold
        ):  # filter on NSB OFF
            removed_pixels += 1
            print("Removed pixel")
        else:
            df = pd.DataFrame(
                {
                    "I_NSB (mA)": nsbs,
                    "channel": channels,
                    "pedestal (ADC)": avgpeds,
                    "pedestal width^2 (ADC^2)": np.power(pedsrms, 2),
                }
            )

            for k, q in enumerate(["pedestal (ADC)", "pedestal width^2 (ADC^2)"]):
                if pixel_id in pixel_display:
                    lm = sns.lmplot(
                        x="I_NSB (mA)",
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
                        x="I_NSB (mA)",
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
                        filtered_df["I_NSB (mA)"],
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
                            "y = {:.4f} I + {:.4f}".format(slope, intercept),
                            (0.05, 0.85 - s * 0.05),
                            color=colors[s],
                            xycoords="axes fraction",
                        )

                if pixel_id in pixel_display:
                    ax.set_title("Pixel {}, T {} deg".format(pixel_id, temp))
                    fig.savefig("{}/pixel_{}_{}.png".format(outdir, pixel_id, k))
                    del fig
                    plt.close()

    print(
        "{} pixels were flagged as bad by the pedestal estimation tool".format(
            flagged_pixels
        )
    )
    print(
        "{} pixels were removed because they had a pedestal RMS exceeding {}".format(
            removed_pixels, rms_threshold
        )
    )

    slopes_width_hg[itemp] = slopes_rms_hg
    slopes_width_lg[itemp] = slopes_rms_lg

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
                ax.set_title("Pedestal slope (ADC/mA), hg, T {} deg".format(temp))
            elif k == 0 and s == 1:
                ax.set_title("Pedestal slope (ADC/mA), lg, T {} deg".format(temp))
            elif k == 1 and s == 0:
                ax.set_title(
                    "Pedestal width slope (ADC^2/mA), hg, T {} deg".format(temp)
                )
            elif k == 1 and s == 1:
                ax.set_title(
                    "Pedestal width slope (ADC^2/mA), lg, T {} deg".format(temp)
                )
            plt.axvline(np.nanmean(slopes), linestyle=":")
            plt.axvline(np.nanmean(slopes) - np.nanstd(slopes), linestyle=":")
            plt.axvline(np.nanmean(slopes) + np.nanstd(slopes), linestyle=":")
            ax.annotate(
                r"mean: {:.3f} $\pm$ {:.3f}".format(
                    np.nanmean(slopes), np.nanstd(slopes)
                ),
                (0.05, 0.85 - s * 0.05),
                xycoords="axes fraction",
            )
            # camera display
            ax = plt.subplot(2, 2, s + 3)
            disp = CameraDisplay(camera)
            disp.image = slopes
            # if k == 0:
            #     disp.set_limits_minmax(-0.3, 0.3)
            # elif k == 1:
            #     disp.set_limits_minmax(-0.07, 0.07)
            disp.cmap = "Spectral_r"
            disp.add_colorbar()
            disp.update()
            ax.set_title("")
        fig.savefig("{}/camera_{}.png".format(outdir, k))

# Ped width^2-I slope as a function of temp
# create output directory if needed
outdir = os.path.join(outfigroot, "NSB_temperature")
if not os.path.exists(outdir):
    os.makedirs(outdir)

slopes_hg = np.full(1855, fill_value=fill_value)
slopes_lg = np.full(1855, fill_value=fill_value)
x = [data["T"] for data in pedfiles]

print("Working on temperature variations")
bad_pixels_hg = 0
bad_pixels_lg = 0
for pixel_id in pixel_process:
    print("Working on pixel {}".format(pixel_id))

    for s in range(2):  # channels
        isgood = True
        if s == 0:
            y = slopes_width_hg[:, pixel_id]
            label = "hg"
            if np.any(np.isnan(y)):
                isgood = False
                bad_pixels_hg += 1
        elif s == 1:
            y = slopes_width_lg[:, pixel_id]
            label = "lg"
            if np.any(np.isnan(y)):
                isgood = False
                bad_pixels_lg += 1

        if isgood:
            slope, intercept, r_value, pv, se = stats.linregress(x, y)

            if s == 0:
                slopes_hg[pixel_id] = slope
            elif s == 1:
                slopes_lg[pixel_id] = slope

            if pixel_id in pixel_display:
                df = pd.DataFrame(
                    {
                        "T (deg)": x,
                        "pedestal width^2-I slope (ADC^2/mA)": y,
                    }
                )
                lm = sns.lmplot(
                    x="T (deg)",
                    y="pedestal width^2-I slope (ADC^2/mA)",
                    data=df,
                    scatter_kws={"alpha": 0.5},
                    legend=False,
                )

                ax = lm.axes[0][0]
                fig = ax.get_figure()
                fig.subplots_adjust(top=0.9)

                ax.annotate(
                    "y = {:.4f} T + {:.4f}".format(slope, intercept),
                    (0.05, 0.85 - s * 0.05),
                    xycoords="axes fraction",
                )
                ax.set_title("Pixel {}, ".format(pixel_id, label))
                fig.savefig("{}/pixel_{}_{}.png".format(outdir, pixel_id, label))
                del fig
                plt.close()

print("Bad pixels: HG {}, LG {}".format(bad_pixels_hg, bad_pixels_lg))

camera = CameraGeometry.from_name("NectarCam-003")
camera = camera.transform_to(EngineeringCameraFrame())

fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(wspace=0.35, hspace=0.3)
for s in range(2):  # HG and LG channels
    if s == 0:
        slopes = slopes_hg
    elif s == 1:
        slopes = slopes_lg
    # histograms
    ax = plt.subplot(2, 2, s + 1)
    plt.hist(slopes, bins="auto", histtype="step")
    if s == 0:
        ax.set_title("Pedestal width slope (ADC^2/mA/deg), hg")
    elif k == 1 and s == 1:
        ax.set_title("Pedestal width slope (ADC^2/mA/deg), lg")
    ax.set_yscale("log")
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
    if s == 0:
        disp.set_limits_minmax(-30.0, 10.0)
    if s == 1:
        disp.set_limits_minmax(-0.15, 0.15)
    disp.cmap = "Spectral_r"
    disp.add_colorbar()
    disp.update()
    ax.set_title("")
fig.savefig("{}/camera.png".format(outdir))
