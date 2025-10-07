import logging
import os
import pathlib

import numpy as np
import tables
from ctapipe.core.traits import ComponentNameList
from ctapipe_io_nectarcam.constants import HIGH_GAIN, LOW_GAIN, N_GAINS

from ...data.container import NectarCAMPedestalContainer, NectarCAMPedestalContainers
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
        Combines sliced results efficiently in one pass
        using an online mean/variance algorithm
        """
        already_combined = False
        with tables.open_file(self.output_path, mode="r") as f:
            keys = list(f.root._v_children)
            if "data_combined" in keys:
                log.error(
                    "Trying to combine results that already contain combined data"
                )
                already_combined = True

        # re-open results
        if already_combined:
            pedestalContainers = NectarCAMPedestalContainers.from_hdf5(
                self.output_path,
                slice_index="combined",
            )
        else:
            pedestalContainers = NectarCAMPedestalContainers.from_hdf5(self.output_path)

        # Loop over sliced results to fill the combined results
        self.log.info("Combine sliced results")

        first = True
        for _pedestalContainer in pedestalContainers:
            pedestalContainer = list(_pedestalContainer.containers.values())[0]

            # usable pixel mask
            usable_pixels = pedestalContainer.pixel_mask == 0
            usable_pixels = np.logical_and(usable_pixels[0], usable_pixels[1])

            nevents_i = pedestalContainer.nevents * usable_pixels
            mean_hg_i = pedestalContainer.pedestal_mean_hg
            mean_lg_i = pedestalContainer.pedestal_mean_lg
            std_hg_i = pedestalContainer.pedestal_std_hg
            std_lg_i = pedestalContainer.pedestal_std_lg

            if first:
                nsamples = pedestalContainer.nsamples
                pixels_id = pedestalContainer.pixels_id
                ucts_timestamp_min = pedestalContainer.ucts_timestamp_min
                ucts_timestamp_max = pedestalContainer.ucts_timestamp_max

                nevents = np.zeros_like(nevents_i)
                mean_hg = np.zeros_like(mean_hg_i)
                mean_lg = np.zeros_like(mean_lg_i)
                M2_hg = np.zeros_like(mean_hg_i)
                M2_lg = np.zeros_like(mean_lg_i)
                first = False

            # update timestamps
            ucts_timestamp_min = np.minimum(
                ucts_timestamp_min, pedestalContainer.ucts_timestamp_min
            )
            ucts_timestamp_max = np.maximum(
                ucts_timestamp_max, pedestalContainer.ucts_timestamp_max
            )

            old_nevents = nevents.copy()
            nevents += nevents_i

            # delta between means
            delta_hg = mean_hg_i - mean_hg
            delta_lg = mean_lg_i - mean_lg

            # update mean (weighted)
            mean_hg += delta_hg * (nevents_i[:, np.newaxis] / nevents[:, np.newaxis])
            mean_lg += delta_lg * (nevents_i[:, np.newaxis] / nevents[:, np.newaxis])

            # update M2 (sum of squares of differences)
            M2_hg += std_hg_i**2 * (nevents_i[:, np.newaxis] - 1) + delta_hg**2 * (
                old_nevents[:, np.newaxis]
                * nevents_i[:, np.newaxis]
                / nevents[:, np.newaxis]
            )
            M2_lg += std_lg_i**2 * (nevents_i[:, np.newaxis] - 1) + delta_lg**2 * (
                old_nevents[:, np.newaxis]
                * nevents_i[:, np.newaxis]
                / nevents[:, np.newaxis]
            )

        # finalize std
        std_hg = np.sqrt(M2_hg / (nevents[:, np.newaxis] - 1))
        std_lg = np.sqrt(M2_lg / (nevents[:, np.newaxis] - 1))

        # flag bad pixels in overall results based on same criteria as for individual
        ped_stats = {}
        array_shape = np.append([N_GAINS], np.shape(mean_hg))
        for statistic in ["mean", "std"]:
            ped_stat = np.zeros(array_shape)
            if statistic == "mean":
                ped_stat[HIGH_GAIN] = mean_hg
                ped_stat[LOW_GAIN] = mean_lg
            elif statistic == "std":
                ped_stat[HIGH_GAIN] = std_hg
                ped_stat[LOW_GAIN] = std_lg
            # Store the result in the dictionary
            ped_stats[statistic] = ped_stat
        # use flagging method from PedestalComponent
        pixel_mask = self.components[0].flag_bad_pixels(ped_stats, nevents)

        # reconstitute dictionary with cumulated results consistently with
        # PedestalComponent
        output = NectarCAMPedestalContainer(
            nsamples=nsamples,
            nevents=nevents,
            pixels_id=pixels_id,
            ucts_timestamp_min=ucts_timestamp_min,
            ucts_timestamp_max=ucts_timestamp_max,
            pedestal_mean_hg=mean_hg,
            pedestal_mean_lg=mean_lg,
            pedestal_std_hg=std_hg,
            pedestal_std_lg=std_lg,
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
