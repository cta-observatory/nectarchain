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

    @staticmethod
    def mean_std_multisample(nevents, means, stds):
        """
        Method that calculates means and std of the combination of multiple subsamples.
        Works for both:

            - pedestal data (means/stds shaped (n_pixels, n_samples))
            - charge data (means/stds shaped (n_pixels,))

        Parameters
        ----------
        nevents : list of `~numpy.ndarray`
            Number of events for each sample (per pixel)
        means : list of `~numpy.ndarray`
            Mean values
        stds : list of `~numpy.ndarray`
            Std values

        Returns
        -------
        mean : `~numpy.ndarray`
            Mean values of combined sample
        std : `~numpy.ndarray`
            Std values of combined sample
        nevent : `~numpy.ndarray`
            Number of events of combined sample (per pixel)
        """

        # convert lists to numpy arrays
        # axis 0 corresponds to the subsamples
        nevents = np.array(nevents)
        means = np.array(means)
        stds = np.array(stds)

        total_nevents = np.sum(nevents, axis=0)

        # Handle both 1D and 2D cases cleanly
        if means.ndim == 3:
            # (n_subsamples, n_pixels, n_samples)
            nevents_expanded = nevents[:, :, np.newaxis]
            total_nevents_expanded = total_nevents[:, np.newaxis]
        elif means.ndim == 2:
            # (n_subsamples, n_pixels)
            nevents_expanded = nevents
            total_nevents_expanded = total_nevents
        else:
            log.error("Unexpected shape for means array")

        mean = np.sum(nevents_expanded * means, axis=0) / total_nevents_expanded
        num = np.sum(
            (nevents_expanded - 1) * stds**2 + nevents_expanded * (means - mean) ** 2,
            axis=0,
        )
        std = np.sqrt(num / (total_nevents_expanded - 1))

        return mean, std, total_nevents

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
        nevents_list = []
        mean_lists = {"ped_hg": [], "ped_lg": [], "charge_hg": [], "charge_lg": []}
        std_lists = {"ped_hg": [], "ped_lg": [], "charge_hg": [], "charge_lg": []}
        first = True
        for _pedestalContainer in pedestalContainers:
            pedestalContainer = list(_pedestalContainer.containers.values())[0]

            # usable pixel mask
            usable_pixels = pedestalContainer.pixel_mask == 0
            usable_pixels = np.logical_and(usable_pixels[0], usable_pixels[1])

            nevents_list.append(pedestalContainer.nevents * usable_pixels)
            mean_lists["ped_hg"].append(pedestalContainer.pedestal_mean_hg)
            mean_lists["ped_lg"].append(pedestalContainer.pedestal_mean_lg)
            std_lists["ped_hg"].append(pedestalContainer.pedestal_std_hg)
            std_lists["ped_lg"].append(pedestalContainer.pedestal_std_lg)
            mean_lists["charge_hg"].append(pedestalContainer.pedestal_charge_mean_hg)
            mean_lists["charge_lg"].append(pedestalContainer.pedestal_charge_mean_lg)
            std_lists["charge_hg"].append(pedestalContainer.pedestal_charge_std_hg)
            std_lists["charge_lg"].append(pedestalContainer.pedestal_charge_std_lg)

            if first:
                nsamples = pedestalContainer.nsamples
                pixels_id = pedestalContainer.pixels_id
                ucts_timestamp_min = pedestalContainer.ucts_timestamp_min
                ucts_timestamp_max = pedestalContainer.ucts_timestamp_max
                first = False

            # update timestamps
            ucts_timestamp_min = np.minimum(
                ucts_timestamp_min, pedestalContainer.ucts_timestamp_min
            )
            ucts_timestamp_max = np.maximum(
                ucts_timestamp_max, pedestalContainer.ucts_timestamp_max
            )

        # Compute combined stats for both gains
        results = {}
        for q in ["ped_hg", "ped_lg", "charge_hg", "charge_lg"]:
            mean, std, nevents = self.mean_std_multisample(
                nevents_list, mean_lists[q], std_lists[q]
            )
            results[q] = {"mean": mean, "std": std}

        mean_hg, std_hg = results["ped_hg"]["mean"], results["ped_hg"]["std"]
        mean_lg, std_lg = results["ped_lg"]["mean"], results["ped_lg"]["std"]
        charge_mean_hg, charge_std_hg = (
            results["charge_hg"]["mean"],
            results["charge_hg"]["std"],
        )
        charge_mean_lg, charge_std_lg = (
            results["charge_lg"]["mean"],
            results["charge_lg"]["std"],
        )

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
            pedestal_charge_mean_hg=charge_mean_hg,
            pedestal_charge_mean_lg=charge_mean_lg,
            pedestal_charge_std_hg=charge_std_hg,
            pedestal_charge_std_lg=charge_std_lg,
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
