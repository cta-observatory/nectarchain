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
    list_available_runs = [
        path_.split("NectarCAM.Run")[1].split(".fits")[0]
        for path_ in list_available_paths
    ]
    print("These are the runs for which the output is already available:")
    for ii, run in enumerate(list_available_runs):
        print(ii, run)
    chosen_run_index = int(
        input(
            "Please type the index of the run you wish to explore, exactly as it appeared in the list: "
        )
    )
    chosen_path = list_available_paths[chosen_run_index]

    reader = EventSource(input_url=chosen_path, max_events=40000)
    timestamps_ucts = []

    for event in reader:
        timestamp = event.nectarcam.tel[0].evt.ucts_timestamp
        timestamps_ucts.append(timestamp)

    delta_timestamps_ucts = (
        np.array(
            [
                timestamps_ucts[ii] - timestamps_ucts[ii - 1]
                for ii in range(1, len(timestamps_ucts))
            ]
        )
        * 1e-3
    )

    print(np.min(delta_timestamps_ucts))


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
