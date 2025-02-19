import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from ctapipe_io_nectarcam import constants

from nectarchain.makers.calibration import FlatfieldNectarCAMCalibrationTool

# Define the global environment variable NECTARCAMDATA (folder where are the runs)
os.environ["NECTARCAMDATA"] = "../output/FF/"


def get_gain(output_from_preFlatFieldComponent):
    """
    Calculate the gain using the ratio of the variance and the mean amplitude per pixel

    Args:
        output_from_preFlatFieldComponent: output from the preFlatFieldComponent

    Returns:
        gain: list of gain for each pixel
    """

    amp_int_per_pix_per_event = (
        output_from_preFlatFieldComponent.amp_int_per_pix_per_event[:, :, :]
    )
    amp_int_per_pix_mean = np.mean(amp_int_per_pix_per_event, axis=0)
    amp_int_per_pix_var = np.var(amp_int_per_pix_per_event, axis=0)
    gain = list(
        np.divide(
            amp_int_per_pix_var, amp_int_per_pix_mean, where=amp_int_per_pix_mean != 0.0
        )
    )
    return gain


def get_hi_lo_ratio(output_from_preFlatFieldComponent):
    """
    Calculate the high gain to low gain ratio

    Args:
        output_from_preFlatFieldComponent: output from the preFlatFieldComponent

    Returns:
        hi_lo_ratio: list of hi/lo ratio for each pixel
    """

    gain = get_gain(output_from_preFlatFieldComponent)
    hi_lo_ratio = gain[constants.HIGH_GAIN] / gain[constants.LOW_GAIN]
    return hi_lo_ratio


def get_bad_pixels(output_from_preFlatFieldComponent):
    """
    Identify bad pixels

    Args:
        output_from_preFlatFieldComponent: output from the preFlatFieldComponent

    Returns:
        all_bad_pix: list of bad pixels
    """

    bad_pix = []

    n_event = len(output_from_preFlatFieldComponent.FF_coef[:, 0, 0])
    step = 100
    n_step = round(n_event / step)

    hi_lo = get_hi_lo_ratio(output_from_preFlatFieldComponent)

    amp_int_per_pix_per_event = preFlatFieldOutput.amp_int_per_pix_per_event[:, :, :]
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
            output_from_preFlatFieldComponent.amp_int_per_pix_per_event[:, :, p]
        )
        mean_amp_int_per_pix = np.mean(amp_int_per_pix_per_event, axis=0)

        for G in [constants.HIGH_GAIN, constants.LOW_GAIN]:
            # pixels with too low amplitude
            if mean_amp_int_per_pix[G] < (mean_amp[G] - 10 * std_amp[G]):
                bad_pix.append(p)

            # pixels with unstable flat-field coefficient
            FF_coef = output_from_preFlatFieldComponent.FF_coef[:, G, p]
            mean_FF_per_pix = np.mean(FF_coef, axis=0)
            std_FF_per_pix = np.std(FF_coef, axis=0)

            for e in range(0, round(n_step)):
                x_block = np.linspace(e * step, (e + 1) * step, step)
                FF_coef_mean_per_block = np.mean(
                    output_from_preFlatFieldComponent.FF_coef[
                        e * step : (e + 1) * step, G, p
                    ],
                    axis=0,
                )
                FF_coef_std_per_block = np.std(
                    output_from_preFlatFieldComponent.FF_coef[
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


# defalut gain array
gain_default = 58.0
hi_lo_ratio_default = 13.0
gain_array = list(np.ones(shape=(constants.N_GAINS, constants.N_PIXELS)))
gain_array[0] = gain_array[0] * gain_default
gain_array[1] = gain_array[1] * gain_default / hi_lo_ratio_default

# empty list of bad pixels
bad_pixels_array = list([])


parser = argparse.ArgumentParser(
    prog="flatfield_example.py",
    description=f"compute FF\
        method",
)

# run numbers
parser.add_argument(
    "-r", "--run_number", nargs="+", default=[], help="run(s) list", type=int
)

parser.add_argument(
    "-m",
    "--max_events",
    nargs="+",
    # default=[],
    help="max events to be load",
    type=int,
)


args = parser.parse_args()
run_number = args.run_number
if len(args.run_number) == 1:
    run_number = args.run_number[0]  # Extract the single value
elif len(args.run_number) > 1:
    run_number = args.run_number  # Keep as a list if multiple values
else:
    run_number = 4940  # Handle the case where no value is given

max_events = args.max_events
# run_number = 4940
# max_events = 10000
window_width = 12
outfile = os.environ["NECTARCAMDATA"] + "/FlatFieldTests/1FF_{}.h5".format(run_number)

# Initial call
tool = FlatfieldNectarCAMCalibrationTool(
    progress_bar=True,
    run_number=run_number,
    max_events=max_events,
    log_level=20,
    window_width=window_width,
    overwrite=True,
    gain=gain_array,
    bad_pix=bad_pixels_array,
    output_path=outfile,
)

tool.initialize()
tool.setup()

tool.start()
preFlatFieldOutput = tool.finish(return_output_component=True)[0]

outfile = os.environ["NECTARCAMDATA"] + "/FlatFieldTests/2FF_{}.h5".format(run_number)

# Second call with updated gain aray and bad pixels
tool = FlatfieldNectarCAMCalibrationTool(
    progress_bar=True,
    run_number=run_number,
    max_events=max_events,
    log_level=20,
    window_width=window_width,
    overwrite=True,
    gain=get_gain(preFlatFieldOutput),
    bad_pix=get_bad_pixels(preFlatFieldOutput),
    output_path=outfile,
)

tool.initialize()
tool.setup()

tool.start()
FlatFieldOutput = tool.finish(return_output_component=True)[0]

# Another option would be to use only one tool and make the gain calculation and
# identification of bad pixels in the finish fonction of the component (to be tested)
