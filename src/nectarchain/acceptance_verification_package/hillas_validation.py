import argparse
import copy
import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from nectarchain.utils.constants import ALLOWED_CAMERAS

# TODO
# this is just a placeholder that makes a plot for test purposes
# the name is such that in the future it can be replaced
# by the script to validate the distribution of Hillas parameters

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    filename=f"{os.environ.get('NECTARCHAIN_LOG', '/tmp')}/{os.getpid()}/"
    f"{Path(__file__).stem}_{os.getpid()}.log",
    handlers=[logging.getLogger("__main__").handlers],
)
log = logging.getLogger(__name__)

plt.style.use(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../utils/plot_style.mpltstyle"
    )
)


def get_args():
    """Parses command-line arguments for the Hillas parameters validation script.

    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="""To be filled""")
    parser.add_argument(
        "-r",
        "--run_file",
        type=str,
        help="Run file path and name",
        required=False,
        default="",
    )
    parser.add_argument(
        "-c",
        "--camera",
        choices=ALLOWED_CAMERAS,
        default=[camera for camera in ALLOWED_CAMERAS if "QM" in camera][0],
        help="Process data for a specific NectarCAM camera.",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory",
        default=f"{os.environ.get('NECTARCHAIN_FIGURES', f'/tmp/{os.getpid()}')}",
    )
    parser.add_argument(
        "--temp_output", help="Temporary output directory for GUI", default=None
    )
    parser.add_argument(
        "-l",
        "--log",
        help="log level",
        default="info",
        type=str,
    )

    return parser


def main():
    parser = get_args()
    args = parser.parse_args()
    log.setLevel(args.log.upper())

    kwargs = copy.deepcopy(vars(args))
    kwargs.pop("camera")
    camera = args.camera

    output_dir = os.path.join(
        os.path.abspath(args.output),
        f"trr_camera_{camera}/{Path(__file__).stem}",
    )
    os.makedirs(output_dir, exist_ok=True)
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

    log.info(f"Running the script with arguments: {kwargs}")

    # Drop arguments from the script after they are parsed, for the GUI to work properly
    sys.argv = sys.argv[:1]

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    fig_name = "hillas"
    plot_path = os.path.join(output_dir, f"{fig_name}.png")
    plt.savefig(plot_path)

    if temp_output:
        with open(os.path.join(temp_output, f"plot_{fig_name}.pkl"), "wb") as f:
            pickle.dump(fig, f)

    plt.close("all")


if __name__ == "__main__":
    main()
