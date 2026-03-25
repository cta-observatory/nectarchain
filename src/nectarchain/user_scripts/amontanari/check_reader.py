import argparse
import os
from glob import glob

import numpy as np
from ctapipe.io import EventSource


def main(path_to_data):
    path_to_runs = path_to_data + "runs/"
    if not os.path.exists(path_to_runs):
        raise OSError("The output path does not exists. Have you downloaded the runs?")

    list_available_paths = glob(path_to_runs + "*")
    list_available_paths = [path_ for path_ in list_available_paths if "fits" in path_]
    chosen_path = list_available_paths[116]

    reader = EventSource(input_url=chosen_path, max_events=1)

    print("Now printing everything")

    for event in reader:
        print("\n" + "#" * 40)
        print(event.keys())
        for key in event.keys():
            print("--" * 20)
            print(f"Printing {key}")
            print(event[key])


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="Simple tool to check the content "
        "of the Hillas parameters for a chosen run number"
    )

    parser.add_argument(
        "--path",
        type=str,
        default="../../../../../nectar_cam_data/",
        help="Path to the directory containing the runs taken with NectarCAM",
    )

    args = parser.parse_args()

    main(path_to_data=args.path)
