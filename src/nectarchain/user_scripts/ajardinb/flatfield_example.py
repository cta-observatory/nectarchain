import argparse
import logging
import os
import pathlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ctapipe_io_nectarcam import constants

from nectarchain.makers.calibration import FlatfieldNectarCAMCalibrationTool
from nectarchain.utils.constants import ALLOWED_CAMERAS

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

# Option and argument parser
parser = argparse.ArgumentParser(
    description="Give a run number and the location of the runs folder (NECTARCAMDATA)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--camera",
    choices=ALLOWED_CAMERAS,
    default=[camera for camera in ALLOWED_CAMERAS if "QM" in camera][0],
    help="""Process data for a specific NectarCAM camera.
Default: NectarCAMQM (Qualification Model).""",
    type=str,
)
parser.add_argument(
    "-r",
    "--run",
    default=None,
    help="Flatfield run number",
    type=int,
)
parser.add_argument("--log", default="info", help="output verbosity", type=str)

args = parser.parse_args()

if args.run is None:
    log.error(f"At least one run number should be provided (use -r).")
    sys.exit(1)

log.setLevel(args.log.upper())


def get_gain(output_from_FlatFieldComponent):
    """
    Calculate the gain using the ratio of the variance and the mean amplitude per pixel

    Args:
        output_from_FlatFieldComponent: output from the FlatFieldComponent

    Returns:
        gain: list of gain for each pixel
    """

    amp_int_per_pix_per_event = (
        output_from_FlatFieldComponent.amp_int_per_pix_per_event[:, :, :]
    )
    amp_int_per_pix_mean = np.mean(amp_int_per_pix_per_event, axis=0)
    amp_int_per_pix_var = np.var(amp_int_per_pix_per_event, axis=0)
    gain = list(
        np.divide(
            amp_int_per_pix_var, amp_int_per_pix_mean, where=amp_int_per_pix_mean != 0.0
        )
    )
    return gain


def get_hi_lo_ratio(output_from_FlatFieldComponent):
    """
    Calculate the high gain to low gain ratio

    Args:
        output_from_FlatFieldComponent: output from the FlatFieldComponent

    Returns:
        hi_lo_ratio: list of hi/lo ratio for each pixel
    """

    gain = get_gain(output_from_FlatFieldComponent)
    hi_lo_ratio = gain[constants.HIGH_GAIN] / gain[constants.LOW_GAIN]
    return hi_lo_ratio


def get_bad_pixels(output_from_FlatFieldComponent):
    """
    Identify bad pixels

    Args:
        output_from_FlatFieldComponent: output from the FlatFieldComponent

    Returns:
        all_bad_pix: list of bad pixels
    """

    bad_pix = []

    n_event = len(output_from_FlatFieldComponent.FF_coef[:, 0, 0])
    step = 100
    n_step = round(n_event / step)

    hi_lo = get_hi_lo_ratio(output_from_FlatFieldComponent)

    amp_int_per_pix_per_event = FlatFieldOutput.amp_int_per_pix_per_event[:, :, :]
    mean_amp_int_per_pix = np.mean(amp_int_per_pix_per_event, axis=0)
    mean_amp = np.mean(mean_amp_int_per_pix, axis=1)
    std_amp = np.std(mean_amp_int_per_pix, axis=1)

    for p in range(0, constants.N_PIXELS):
        # pixel with hi/lo ratio to small or to high (+/- 5 times the mean hi/lo ratio)
        if (hi_lo[p] < np.mean(hi_lo) - (5 * np.std(hi_lo))) or (
            hi_lo[p] > np.mean(hi_lo) + (5 * np.std(hi_lo))
        ):
            bad_pix.append(p)

        amp_int_per_pix_per_event = (
            output_from_FlatFieldComponent.amp_int_per_pix_per_event[:, :, p]
        )
        mean_amp_int_per_pix = np.mean(amp_int_per_pix_per_event, axis=0)

        for G in [constants.HIGH_GAIN, constants.LOW_GAIN]:
            # pixels with too low amplitude
            if mean_amp_int_per_pix[G] < (mean_amp[G] - 10 * std_amp[G]):
                bad_pix.append(p)

            # pixels with unstable flat-field coefficient
            FF_coef = output_from_FlatFieldComponent.FF_coef[:, G, p]
            mean_FF_per_pix = np.mean(FF_coef, axis=0)
            std_FF_per_pix = np.std(FF_coef, axis=0)

            for e in range(0, round(n_step)):
                x_block = np.linspace(e * step, (e + 1) * step, step)
                FF_coef_mean_per_block = np.mean(
                    output_from_FlatFieldComponent.FF_coef[
                        e * step : (e + 1) * step, G, p
                    ],
                    axis=0,
                )
                FF_coef_std_per_block = np.std(
                    output_from_FlatFieldComponent.FF_coef[
                        e * step : (e + 1) * step, G, p
                    ],
                    axis=0,
                )

                if (
                    FF_coef_mean_per_block < mean_FF_per_pix - FF_coef_std_per_block
                ) or (FF_coef_mean_per_block > mean_FF_per_pix + FF_coef_std_per_block):
                    bad_pix.append(p)

    all_bad_pix = list(set(bad_pix))
    all_bad_pix.sort()

    return all_bad_pix


log.info("\n *** First pass with default gain and hi/lo values *** \n")

# default gain array
gain_default = 58.0
hi_lo_ratio_default = 13.0
gain_array = list(np.ones(shape=(constants.N_GAINS, constants.N_PIXELS)))
gain_array[0] = gain_array[0] * gain_default
gain_array[1] = gain_array[1] * gain_default / hi_lo_ratio_default

# empty list of bad pixels
bad_pixels_array = list([])
run_number = args.run
max_events = 10000
window_width = 12
window_shift = 5
outfile = Path(os.environ.get("NECTARCAMDATA"), f"FlatFieldOutput/1FF_{run_number}.h5")

# Initial call
tool = FlatfieldNectarCAMCalibrationTool(
    progress_bar=True,
    camera=args.camera,
    run_number=run_number,
    max_events=max_events,
    log_level=args.log.upper(),
    charge_extraction_method=None,  # None, "LocalPeakWindowSum", "GlobalPeakWindowSum"
    charge_integration_correction=False,
    window_width=window_width,
    window_shift=window_shift,
    overwrite=True,
    gain=gain_array,
    bad_pix=bad_pixels_array,
    output_path=outfile,
)

tool.initialize()
tool.setup()

tool.start()
FlatFieldOutput = tool.finish(return_output_component=True)[0]

if len(FlatFieldOutput.event_id) == 0:
    log.error(f"There are no flatfield events")
    sys.exit(1)

log.debug("\n\tIntermediate output file %s" % outfile)

log.info(
    "\n *** Second pass with updates gain and hi/lo values and \
taking into account bad pixels *** \n"
)

outfile = Path(os.environ.get("NECTARCAMDATA"), f"FlatFieldOutput/2FF_{run_number}.h5")

# Second call with updated gain aray and bad pixels
tool = FlatfieldNectarCAMCalibrationTool(
    progress_bar=True,
    camera=args.camera,
    run_number=run_number,
    max_events=max_events,
    log_level=args.log.upper(),
    charge_extraction_method=None,  # None, "LocalPeakWindowSum", "GlobalPeakWindowSum"
    charge_integration_correction=False,
    window_width=window_width,
    window_shift=window_shift,
    overwrite=True,
    gain=get_gain(FlatFieldOutput),
    bad_pix=get_bad_pixels(FlatFieldOutput),
    output_path=outfile,
)

tool.initialize()
tool.setup()

tool.start()
FlatFieldOutput = tool.finish(return_output_component=True)[0]

log.info("\n\tFinal output file %s \n" % outfile)

# Another option would be to use only one tool and make the gain calculation and
# identification of bad pixels in the finish fonction of the component (to be tested)
