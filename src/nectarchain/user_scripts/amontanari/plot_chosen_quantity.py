import argparse
import json
import os
import re
import sys
from glob import glob

import numpy as np
from astropy.io import fits
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.io import EventSource
from ctapipe.visualization import CameraDisplay
from matplotlib import pyplot as plt

NOTINDISPLAY = [
    "TRIGGER-.*",
    "PED-INTEGRATION-.*",
    "START-TIMES",
    "WF-.*",
    ".*PIXTIMELINE-.*",
]
TEST_PATTERN = "(?:% s)" % "|".join(NOTINDISPLAY)

labels_path = os.path.join("../../dqm/bokeh_app", "data", "labels.json")


def select_quantity_to_plot(hdul, index_for_bad_pixel):
    chosen_quantity_index = input(
        "Please type the index of the quantity "
        "(or several indexes separated by a blank space) you wish to plot, "
        "exactly as it appeared in the list: "
    ).split()
    chosen_quantity_index = [int(idx) for idx in chosen_quantity_index]
    if len(chosen_quantity_index) > 1:
        quantities_to_plot = []
        for idx in chosen_quantity_index:
            quantity_to_plot = hdul[idx].data
            quantity_name = hdul[idx].header["EXTNAME"]
            quantity_to_plot = np.array(
                [np.nan_to_num(value[0], nan=0.0) for value in quantity_to_plot]
            )
            quantity_to_plot, mask_bad_pixels = mask_bad_pixels_on_quantity(
                quantity_to_plot, quantity_name, hdul, index_for_bad_pixel
            )
            quantities_to_plot.append(
                (quantity_name, quantity_to_plot, mask_bad_pixels)
            )
    else:
        quantity_to_plot = hdul[chosen_quantity_index].data
        quantity_name = hdul[chosen_quantity_index].header["EXTNAME"]
        quantity_to_plot = np.array(
            [np.nan_to_num(value[0], nan=0.0) for value in quantity_to_plot]
        )
        quantity_to_plot, mask_bad_pixels = mask_bad_pixels_on_quantity(
            quantity_to_plot, quantity_name, hdul, index_for_bad_pixel
        )
        quantities_to_plot = [(quantity_name, quantity_to_plot, mask_bad_pixels)]

    return quantities_to_plot


def mask_bad_pixels_on_quantity(
    quantity_to_plot, quantity_name, hdul, index_for_bad_pixel
):
    if "CAMERA-BADPIX-" in quantity_name:
        mask_bad_pixels = quantity_to_plot > 1
        quantity_to_plot[mask_bad_pixels] = 1.0
    else:
        try:
            bad_pixels = hdul[index_for_bad_pixel].data
            bad_pixels = np.array(
                [np.nan_to_num(value[0], nan=0.0) for value in bad_pixels]
            )
        except KeyError:
            print("Bad pixels not found in the fits file")
        mask_bad_pixels = bad_pixels >= 1.0
        # removing from the plot the drawers with the bad pixels
        quantity_to_plot[mask_bad_pixels] = 0.0

    return quantity_to_plot, mask_bad_pixels


def make_camera_display(
    chosen_run, quantity_image, quantity_name, mask_bad_pixels=None
):
    with open(labels_path, "r", encoding="utf-8") as file:
        colorbar_labels = json.load(file)["colorbar_labels_camera_display"]

    try:
        colorbar_label = colorbar_labels[quantity_name]
    except KeyError:
        colorbar_label = ""

    if "CAMERA-TEMPERATURE" in quantity_name:
        colorbar_label = r"$^{\circ} C$"

    chosen_path = (
        f"../../../../../nectar_cam_data/runs/NectarCAM.Run{chosen_run}.0000.fits.fz"
    )
    reader = EventSource(input_url=chosen_path, max_events=1)

    geometry = reader.subarray.tel[0].camera.geometry.transform_to(
        EngineeringCameraFrame()
    )
    display = CameraDisplay(geometry=geometry)
    display.image = quantity_image
    display.cmap = plt.cm.coolwarm

    display.axes.text(2.2, -0.8, colorbar_label, fontsize=12, rotation=270)
    display.add_colorbar()
    if "CAMERA-BADPIX-" not in quantity_name:
        assert (
            mask_bad_pixels is not None,
            "mask_bad_pixels is needed to exclude the drawers"
            + "with bad pixels from the colorbar range",
        )
        display.colorbar.norm.vmin = np.min(quantity_image[~mask_bad_pixels])
        display.colorbar.norm.vmax = np.max(quantity_image[~mask_bad_pixels])

    plt.title(quantity_name)

    plt.savefig(
        "./plots/" + quantity_name.lower() + f"_nectarchain_dqm_run{chosen_run}.png",
        metadata={"Creator": sys.argv[0]},
        dpi=700,
    )

    plt.show()


def make_timeline():
    pass


def make_histogram():
    pass


def main(path_to_data):
    path_to_output = path_to_data + "output/"
    if not os.path.exists(path_to_output):
        raise OSError("The output path does not exists. Have you run the DQM?")

    list_available_paths = glob(path_to_output + "*")
    list_available_runs = [
        int(path_.split("NectarCAM_Run")[1]) for path_ in list_available_paths
    ]
    print("These are the runs for which the output is already available:")
    for ii, run in enumerate(list_available_runs):
        print(ii, run)
    chosen_run_index = int(
        input(
            "Please type the index of the run you wish to explore, "
            "exactly as it appeared in the list: "
        )
    )
    chosen_path = list_available_paths[chosen_run_index]

    run_path = chosen_path.split("output/")[1]
    fits_result_path = f"{chosen_path}/{run_path}_calib/{run_path}_Results.fits"

    print("These are the quantities you can plot:")
    with fits.open(fits_result_path) as hdul:
        hdul.info()
        for ii in range(len(hdul)):
            if ii != 0:
                print(ii, hdul[ii].header["EXTNAME"])
                if (
                    hdul[ii].header["EXTNAME"]
                    == "CAMERA-BADPIX-PED-PHY-OVEREVENTS-HIGH-GAIN"
                ):
                    index_for_bad_pixel = ii

        quantities_to_plot = select_quantity_to_plot(hdul, index_for_bad_pixel)

    for quantity_name, quantity_to_plot, mask_bad_pixels in quantities_to_plot:
        if not re.match(TEST_PATTERN, quantity_name):
            make_camera_display(
                chosen_run=list_available_runs[chosen_run_index],
                quantity_image=quantity_to_plot,
                quantity_name=quantity_name,
                mask_bad_pixels=mask_bad_pixels,
            )
        elif re.match("(?:.*PIXTIMELINE-.*)", quantity_name):
            make_timeline()
        else:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple tool to check the content "
        "of the FITS files after running the DQM"
    )

    parser.add_argument(
        "--path",
        type=str,
        # this should be set to the $NECTARCAMDATA,
        # but I am lazy and the absolute pattern does not always work,
        # so just setting a default for now
        default="../../../../../nectar_cam_data/",
        help="Path to the directory containing the output of the start_dqm.py script",
    )

    args = parser.parse_args()

    main(path_to_data=args.path)
