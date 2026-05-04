import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt

from nectarchain.utils.constants import ALLOWED_CAMERAS

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
    """Parses command-line arguments for the deadtime test script.

    Returns
    -------
    parser : argparse.ArgumentParser
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Deadtime tests B-TEL-1260 & B-TEL-1270. \n"
        + "According to the nectarchain component interface, you have to set a\
            NECTARCAMDATA environment variable in the folder where you have the data\
                from your runs or where you want them to be downloaded.\n"
        + "You have to provide a run file path and name, and, optionally, a \
            corresponding camera tag and an output directory to save the final plot.\n"
        + "If the data is not in NECTARCAMDATA, the files will be downloaded through \
            DIRAC.\n"
        + "You can optionally specify the number of events to be processed \
            (default 8000).\n",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-r",
        "--run_file",
        type=str,
        help="Run file path and name",
        required=False,
        default="",
    )
    parser.add_argument(
        "-e",
        "--evts",
        type=int,
        help="Number of events to process from each run.",
        required=False,
        default=8000,
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


if __name__ == "__main__":
    main()
