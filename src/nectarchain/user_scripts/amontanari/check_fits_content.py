import argparse
import os
from glob import glob

from astropy.io import fits


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

    with fits.open(fits_result_path) as hdul:
        hdul.info()
        for ii in range(len(hdul)):
            print("-" * 30)
            print(hdul[ii].header)
            print(hdul[ii].data)


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
