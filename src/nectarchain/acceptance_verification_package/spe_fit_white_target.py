import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ctapipe.containers import EventType, Field
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.core.traits import ComponentNameList
from ctapipe.image import tailcuts_clean
from ctapipe.instrument import CameraGeometry
from ctapipe.io import HDF5TableWriter
from ctapipe_io_nectarcam import N_PIXELS
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer

from nectarchain.data.container import ChargesContainers, NectarCAMContainer
from nectarchain.makers import (
    DelimiterLoopNectarCAMCalibrationTool,
    EventsLoopNectarCAMCalibrationTool,
)
from nectarchain.makers.component import NectarCAMComponent
from nectarchain.makers.component.spe.spe_algorithm import SPEnominalalgorithm
from nectarchain.utils.constants import ALLOWED_CAMERAS

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.getLogger("__main__").handlers],
)
log = logging.getLogger(__name__)
"""
try:
    plt.style.use(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "../utils/plot_style_new.mpltstyle",
        )
    )
except FileNotFoundError as e:
    raise e
"""

pixel_ids_block_combined = []

ToM_avg_waveform = []  # default settings


def get_args():
    """Parses command-line arguments for the linearity test script.

    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Gain calibration for white target source \n"
        + "According to the nectarchain component interface, \
            you have to set a NECTARCAMDATA environment variable\
                in the folder where you have the data from your runs\
                    or where you want them to be downloaded.\n"
        + "If the data is not in NECTARCAMDATA, the files will be downloaded through\
            DIRAC.\n For the purposes of testing this script, default data is from the\
                runs used for this test in the TRR document.\n"
    )
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        help="run number",
        required=False,
        default=886,
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
        help="Output base directory",
        required=False,
        default=f"{os.environ.get('NECTARCHAIN_FIGURES', f'/tmp/{os.getpid()}')}",
    )

    parser.add_argument(
        "--intermediate_file",
        type=bool,
        default=False,
        help="Store intermediate file with all values and parameters",
    )
    parser.add_argument(
        "--temp_output", help="Temporary output directory for GUI", default=None
    )
    parser.add_argument(
        "--averaging_method",
        choices=[
            "linear",
            "best_fit",
        ],
        default="best_fit",
        help="Method for weighted average.\
            linear - weights are 1/error"
        "best_mean - Statistics book by Luca Lista recommends 1/error^2"
        "from chi sq minimisation",
    )

    return parser


class BlockIdentifierComponent(NectarCAMComponent):
    def __init__(self, subarray, config=None, parent=None, *args, **kwargs):
        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **kwargs
        )
        self.__amplitude__ = []
        self.__pixels_id_block__ = []
        self.__wfs__ = []
        # To determine ToM from average waveform

    def __call__(self, event: NectarCAMDataContainer, *args, **kwargs):
        if event.trigger.event_type == EventType.SINGLE_PE:
            self.__amplitude__.append(
                np.max(event.r0.tel[self.tel_id].waveform[0], axis=1)
            )

            if len(self.__wfs__) < 100:
                self.__wfs__.append(event.r0.tel[self.tel_id].waveform[0])

        elif event.trigger.event_type == EventType.UNKNOWN:
            amp_rms = np.std(self.__amplitude__, axis=0)

            camera = CameraGeometry.from_name("NectarCam-003").transform_to(
                EngineeringCameraFrame()
            )
            cleanmask = tailcuts_clean(
                camera,
                amp_rms,
                picture_thresh=4,
                boundary_thresh=2,
                min_number_picture_neighbors=3,
            )
            amp_rms[~cleanmask] = 0
            pix = np.where(amp_rms > 3)

            self.__amplitude__ = []
            self.__pixels_id_block__.append(pix)

    def finish(self):
        pixel_ids_block_combined.append(self.__pixels_id_block__)
        mean_waveform = np.mean(self.__wfs__, axis=0)

        max_position = np.argmax(mean_waveform, axis=1)[self.__pixels_id_block__[0]]

        max_position = np.ma.masked_array(
            max_position, mask=(max_position < 10) | (max_position > 45)
        )
        ToM_avg_waveform.append(int(np.mean(max_position)))

        del self.__wfs__


class BlockIdentificationTool(EventsLoopNectarCAMCalibrationTool):
    name = "BlockIdentification"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["BlockIdentifierComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def finish(self, *args, **kwargs):
        output = super().finish(return_output_component=True, *args, **kwargs)
        log.info(f"{output}")


class GainContainer(NectarCAMContainer):
    gain = Field(default=None, description="Weighted gain of each pixel")

    gain_error = Field(default=None, description="Gain error of each pixel")


class WhiteTargetTestTool(DelimiterLoopNectarCAMCalibrationTool):
    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["ChargesComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def finish(self, *args, **kwargs):
        output = super().finish(return_output_component=True, *args, **kwargs)

        results_list = []

        for i in range(2, len(pixel_ids_block_combined[0])):
            charge_container = next(
                ChargesContainers.from_hdf5(self.output_path, slice_index=i)
            )

            charge_container1 = charge_container.containers[EventType.SINGLE_PE]

            spe_fit_run = SPEnominalalgorithm.create_from_chargesContainer(
                signal=charge_container1
            )

            log.debug(f"{i}, {pixel_ids_block_combined[0][i-1][0]}")
            output = spe_fit_run.run(
                pixels_id=np.array(pixel_ids_block_combined[0][i - 1][0])
            )

            is_valid = [r["is_valid"] for r in output]

            for (
                pixel_id,
                gain,
                gain_error,
                pp,
                n,
                pedestal,
                pedestalWidth,
                luminosity,
                resolution,
                likelihood,
                ifit_status,
            ) in zip(
                pixel_ids_block_combined[0][i - 1][0],
                spe_fit_run.results.high_gain[:, 0][
                    pixel_ids_block_combined[0][i - 1][0]
                ],
                spe_fit_run.results.high_gain[:, 1][
                    pixel_ids_block_combined[0][i - 1][0]
                ],
                spe_fit_run.results.pp[:, 0][pixel_ids_block_combined[0][i - 1][0]],
                spe_fit_run.results.n[:, 0][pixel_ids_block_combined[0][i - 1][0]],
                spe_fit_run.results.pedestal[:, 0][
                    pixel_ids_block_combined[0][i - 1][0]
                ],
                spe_fit_run.results.pedestalWidth[:, 0][
                    pixel_ids_block_combined[0][i - 1][0]
                ],
                spe_fit_run.results.luminosity[:, 0][
                    pixel_ids_block_combined[0][i - 1][0]
                ],
                spe_fit_run.results.resolution[:, 0][
                    pixel_ids_block_combined[0][i - 1][0]
                ],
                spe_fit_run.results.likelihood[pixel_ids_block_combined[0][i - 1][0]],
                is_valid,
            ):
                results_list.append(
                    {
                        "slice_number": i,
                        "pixel_id": int(pixel_id),
                        "gain_value": float(gain),
                        "gain_error": float(gain_error),
                        "pp": pp,
                        "n": n,
                        "pedestal": pedestal,
                        "pedestalWidth": pedestalWidth,
                        "luminosity": luminosity,
                        "resolution": resolution,
                        "likelihood": likelihood,
                        "fit status": ifit_status,
                    }
                )

        return results_list

        # spe_fit_run.run(pixels_id=pixel_ids_block_combined[i][0])


def main():
    """
    The `main()` function is the entry point of the spe_fit_white_target. It parses \
            the command-line arguments, processes the calibration run,\
                  taken using aiv/nectarpy/spe_white_target_scan.py\
            The run takes SPE runs seperated by UNKNOWN event type\
              for each position of white target\
    1. Parses the command-line arguments using the `get_args()` function,\
          which sets up\
            the argument parser and handles the input parameters.\
    2. Processes the run and produces the\
          spe gains for each position\
    3. Produce an output h5 file having gain and gain_error\
        computed using weighted average of gain values\
        calculated at each position\
    """
    start_time = time.time()
    parser = get_args()
    args = parser.parse_args()

    run = args.run
    camera = args.camera

    output_dir = os.path.join(
        os.path.abspath(args.output),
        f"av_camera_{camera}/{Path(__file__).stem}",
    )
    os.makedirs(output_dir, exist_ok=True)
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

    log.debug(f"Output directory: {output_dir}")
    log.debug(f"Temporary output file: {temp_output}")

    sys.argv = sys.argv[:1]

    log.info(f"PROCESSING RUN {run} {output_dir}")
    output_file_name = Path(
        f"{output_dir}/SPE_WhiteTarget_TestTool_run{str(run)}_window8.h5"
    )

    # constants for this code
    window_shift = 4
    window_width = 8
    method = "FixedWindowSum"

    tool = BlockIdentificationTool(
        progress_bar=True,
        run_number=run,
        overwrite=True,
    )
    tool.initialize()
    tool.setup()
    tool.start()
    # pixel_ids_block_combined, ToM_avg_waveform = tool.finish()
    tool.finish()
    log.debug(f"Block identifier , {pixel_ids_block_combined}, {ToM_avg_waveform[0]}")

    tool = WhiteTargetTestTool(
        progress_bar=True,
        run_number=run,
        method=method,
        extractor_kwargs={
            "window_width": window_width,
            "window_shift": window_shift,
            "peak_index": ToM_avg_waveform[0],
        },
        overwrite=True,
    )

    tool.initialize()
    tool.setup()
    tool.start()
    results = tool.finish()
    df = pd.DataFrame(results)

    if args.intermediate_file:
        output_file_csv_name = os.path.join(
            output_dir, f"spe_white_target_gain_results_{args.averaging_method}.csv"
        )
        df.to_csv(output_file_csv_name, index=False, sep="\t")

    # Computing weighted average#####
    # To get one gain value and error per pixel#

    weighted_gain = np.zeros(N_PIXELS)
    weighted_gain_error = np.zeros(N_PIXELS)

    for ipix in range(N_PIXELS):
        slice_pix = df[df["pixel_id"] == ipix]

        gain_error = slice_pix["gain_error"].values

        gain_values = slice_pix["gain_value"].values

        likelihood = slice_pix["likelihood"].values

        if len(gain_values) == 0 or np.sum(gain_error) == 0:
            continue

        gain_error = np.ma.masked_array(gain_error, gain_error == 0)
        gain_values = np.ma.masked_array(gain_values, gain_error == 0)

        # Cuts before computing weighted average
        gain_values = np.ma.masked_array(gain_values, likelihood > 10**4)
        gain_error = np.ma.masked_array(gain_error, likelihood > 10**4)
        gain_values = np.ma.masked_array(gain_values, gain_values > 150)
        gain_error = np.ma.masked_array(gain_error, gain_values > 150)

        if args.averaging_method == "linear":
            weights = 1.0 / (gain_error)
        else:
            weights = 1.0 / (gain_error * gain_error)

        weights[np.where(gain_error == 0)] = 0

        if np.sum(weights) == 0:
            continue

        # weighted_gain[ipix] = gain_values[np.where(gain_error==np.min(gain_error))]

        weighted_gain[ipix] = np.average(gain_values, weights=weights)
        weighted_gain_error[ipix] = np.sqrt(1.0 / np.sum(weights))

    gain_container = GainContainer()
    gain_container.gain = weighted_gain
    gain_container.gain_error = weighted_gain_error

    # Write to HDF5

    output_file_name = os.path.join(
        output_dir, f"gain_values_white_target_{args.averaging_method}.h5"
    )
    with HDF5TableWriter(filename=output_file_name, group_name="data") as writer:
        writer.write("gain", gain_container)

    log.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
