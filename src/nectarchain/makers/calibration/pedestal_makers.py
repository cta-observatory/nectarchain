import logging
import os
import pathlib

import numpy as np
import tables
from ctapipe.core.traits import ComponentNameList
from ctapipe_io_nectarcam.constants import HIGH_GAIN, LOW_GAIN, N_GAINS

from ...data.container import NectarCAMPedestalContainer
from ..component import NectarCAMComponent
from .core import NectarCAMCalibrationTool

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


__all__ = ["PedestalNectarCAMCalibrationTool"]


class PedestalNectarCAMCalibrationTool(NectarCAMCalibrationTool):
    name = "PedestalNectarCAMCalibrationTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["PedestalEstimationComponent"],
        help="List of Component names to be applied, the order will be respected",
    ).tag(config=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_output_path(self):
        """
        Initialize output path
        """

        if self.events_per_slice is None:
            ext = ".h5"
        else:
            ext = f"_sliced{self.events_per_slice}.h5"
        if self.max_events is None:
            filename = f"{self.name}_run{self.run_number}{ext}"
        else:
            filename = (
                f"{self.name}_run{self.run_number}_maxevents{self.max_events}{ext}"
            )

        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA', '/tmp')}/PedestalEstimation/{filename}"
        )

    def _combine_results(self):
        """
        Method that combines sliced results to reduce memory load
        Can only be called after the file with the sliced results has been saved to disk
        """

        # re-open results
        pedestalContainers = next(
            NectarCAMPedestalContainers.from_hdf5(self.output_path)
        )
        # Loop over sliced results to fill the combined results
        if "data_combined" in pedestalContainers.containers.keys():
            log.error("Trying to combine results that already contain combined data")
        self.log.info("Combine sliced results")
        for i, (_, pedestalContainer) in enumerate(
            pedestalContainers.containers.items()
        ):
            if i == 0:
                # initialize fields for the combined results based on first slice
                nsamples = pedestalContainer.nsamples
                nevents = np.zeros(len(pedestalContainer.nevents))
                pixels_id = pedestalContainer.pixels_id
                ucts_timestamp_min = pedestalContainer.ucts_timestamp_min
                ucts_timestamp_max = pedestalContainer.ucts_timestamp_max
                pedestal_mean_hg = np.zeros(
                    np.shape(pedestalContainer.pedestal_mean_hg)
                )
                pedestal_mean_lg = np.zeros(
                    np.shape(pedestalContainer.pedestal_mean_lg)
                )
                pedestal_std_hg = np.zeros(np.shape(pedestalContainer.pedestal_std_hg))
                pedestal_std_lg = np.zeros(np.shape(pedestalContainer.pedestal_std_lg))
            else:
                # otherwise consider the overall time interval
                ucts_timestamp_min = np.minimum(
                    ucts_timestamp_min, pedestalContainer.ucts_timestamp_min
                )
                ucts_timestamp_max = np.maximum(
                    ucts_timestamp_max, pedestalContainer.ucts_timestamp_max
                )

            # calculate final values of mean and std
            pedestal_mean_hg /= nevents[:, np.newaxis]
            pedestal_mean_lg /= nevents[:, np.newaxis]
            pedestal_std_hg /= nevents[:, np.newaxis]
            pedestal_std_hg = np.sqrt(pedestal_std_hg)
            pedestal_std_lg /= nevents[:, np.newaxis]
            pedestal_std_lg = np.sqrt(pedestal_std_lg)

        # flag bad pixels in overall results based on same criteria as for individual
        # slides
        # reconstitute dictionary with cumulated results consistently with
        # PedestalComponent
        ped_stats = {}
        array_shape = np.append([N_GAINS], np.shape(pedestal_mean_hg))
        for statistic in ["mean", "std"]:
            ped_stat = np.zeros(array_shape)
            if statistic == "mean":
                ped_stat[HIGH_GAIN] = pedestal_mean_hg
                ped_stat[LOW_GAIN] = pedestal_mean_lg
            elif statistic == "std":
                ped_stat[HIGH_GAIN] = pedestal_std_hg
                ped_stat[LOW_GAIN] = pedestal_std_lg
            # Store the result in the dictionary
            ped_stats[statistic] = ped_stat
        # use flagging method from PedestalComponent
        pixel_mask = self.components[0].flag_bad_pixels(ped_stats, nevents)

        output = NectarCAMPedestalContainer(
            nsamples=nsamples,
            nevents=nevents,
            pixels_id=pixels_id,
            ucts_timestamp_min=ucts_timestamp_min,
            ucts_timestamp_max=ucts_timestamp_max,
            pedestal_mean_hg=pedestal_mean_hg,
            pedestal_mean_lg=pedestal_mean_lg,
            pedestal_std_hg=pedestal_std_hg,
            pedestal_std_lg=pedestal_std_lg,
            pixel_mask=pixel_mask,
        )

        return output

    def finish(self, return_output_component=False, *args, **kwargs):
        """
        Redefines finish method to combine sliced results
        """

        self.log.info("finishing Tool")

        # finish components
        output = self._finish_components(*args, **kwargs)

        # close  writer
        self.writer.close()

        # Check if there are slices
        if self.events_per_slice is None:
            # If not nothing to do
            pass
        else:
            # combine results
            output = self._combine_results()
            # add combined results to output
            # re-initialise writer to store combined results
            self._init_writer(sliced=True, group_name="data_combined")
            # add combined results to writer
            self._write_container(output)
            self.writer.close()

        self.log.info("Shutting down.")
        if return_output_component:
            return output
