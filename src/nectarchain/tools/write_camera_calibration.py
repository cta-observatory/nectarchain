"""
Extract flat field coefficients from flasher data files.
"""
import numpy as np
from ctapipe.calib.camera.flatfield import FlatFieldCalculator
from ctapipe.calib.camera.pedestals import PedestalCalculator
from ctapipe.core import Provenance, Tool, traits
from ctapipe.image import ImageExtractor
from ctapipe.io import EventSource, HDF5TableWriter
from ctapipe.io.containers import (
    FlatFieldContainer,
    PedestalContainer,
    WaveformCalibrationContainer,
)
from traitlets import Dict, Float, List, Unicode


class CalibrationHDF5Writer(Tool):
    """
    Tool that generates a HDF5 file with camera calibration coefficients.
    This is just an example on how the monitoring containers can be filled using
    the calibration Components in calib/camera.
    This example is based on an input file with interleaved pedestal and flat-field
    events.

    For getting help run:
    python calc_camera_calibration.py --help

    """

    name = "CalibrationHDF5Writer"
    description = "Generate a HDF5 file with camera calibration coefficients"

    output_file = Unicode("calibration.hdf5", help="Name of the output file").tag(
        config=True
    )

    minimum_charge = Float(
        800, help="Temporary cut on charge to eliminate events without led signal"
    ).tag(config=True)

    pedestal_product = traits.enum_trait(
        PedestalCalculator, default="PedestalIntegrator"
    )

    flatfield_product = traits.enum_trait(
        FlatFieldCalculator, default="FlasherFlatFieldCalculator"
    )

    aliases = Dict(
        dict(
            input_file="EventSource.input_url",
            max_events="EventSource.max_events",
            flatfield_product="CalibrationHDF5Writer.flatfield_product",
            pedestal_product="CalibrationHDF5Writer.pedestal_product",
        )
    )

    classes = List(
        [
            EventSource,
            FlatFieldCalculator,
            FlatFieldContainer,
            PedestalCalculator,
            PedestalContainer,
            WaveformCalibrationContainer,
        ]
        + traits.classes_with_traits(ImageExtractor)
        + traits.classes_with_traits(FlatFieldCalculator)
        + traits.classes_with_traits(PedestalCalculator)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Tool that generates a HDF5 file with camera calibration coefficients.
        Input file must contain interleaved pedestal and flat-field events.

        For getting help, run:
        python calc_camera_calibration.py --help
        """
        self.eventsource = None
        self.flatfield = None
        self.pedestal = None
        self.container = None
        self.writer = None
        self.tel_id = None

    def setup(self):
        kwargs = dict(parent=self)
        self.eventsource = EventSource.from_config(**kwargs)

        self.flatfield = FlatFieldCalculator.from_name(self.flatfield_product, **kwargs)
        self.pedestal = PedestalCalculator.from_name(self.pedestal_product, **kwargs)

        msg = "tel_id not the same for all calibration components"
        assert self.pedestal.tel_id == self.flatfield.tel_id, msg

        self.tel_id = self.flatfield.tel_id

        group_name = "tel_" + str(self.tel_id)

        self.writer = HDF5TableWriter(
            filename=self.output_file, group_name=group_name, overwrite=True
        )

    def start(self):
        """Calibration coefficient calculator"""

        ped_initialized = False
        ff_initialized = False

        for count, event in enumerate(self.eventsource):
            # get link to monitoring containers
            if count == 0:
                ped_data = event.mon.tel[self.tel_id].pedestal
                ff_data = event.mon.tel[self.tel_id].flatfield
                status_data = event.mon.tel[self.tel_id].pixel_status
                calib_data = event.mon.tel[self.tel_id].calibration
                ids = event.nectarcam.tel[self.tel_id].svc.pixel_ids

            # if pedestal
            if event.r1.tel[self.tel_id].trigger_type == 32:
                if self.pedestal.calculate_pedestals(event):
                    self.log.debug(
                        f"new pedestal at event n. {count+1}, id {event.r0.event_id}"
                    )

                    # update pedestal mask
                    status_data.pedestal_failing_pixels = np.logical_or(
                        ped_data.charge_median_outliers, ped_data.charge_std_outliers
                    )

                    if not ped_initialized:
                        # save the config, to be retrieved as data.meta['config']
                        ped_data.meta["config"] = self.config
                        ped_initialized = True
                    else:
                        self.log.debug("write pedestal data")
                        # write only after a first event (used to initialize the mask)
                        self.writer.write("pedestal", ped_data)

            # consider flat field events only after first pedestal event
            elif (
                event.r1.tel[self.tel_id].trigger_type == 5
                or event.r1.tel[self.tel_id].trigger_type == 4
            ) and (
                ped_initialized
                and np.median(
                    np.sum(event.r1.tel[self.tel_id].waveform[0, ids], axis=1)
                )
                > self.minimum_charge
            ):
                if self.flatfield.calculate_relative_gain(event):
                    self.log.debug(
                        f"new flatfield at event n. {count+1}, id {event.r0.event_id}"
                    )

                    # update the flatfield mask
                    status_data.flatfield_failing_pixels = np.logical_or(
                        ff_data.charge_median_outliers, ff_data.time_median_outliers
                    )

                    # mask from pedestal and flat-fleid data
                    monitoring_unusable_pixels = np.logical_or(
                        status_data.pedestal_failing_pixels,
                        status_data.flatfield_failing_pixels,
                    )

                    # calibration unusable pixels are an OR of all maskes
                    calib_data.unusable_pixels = np.logical_or(
                        monitoring_unusable_pixels, status_data.hardware_failing_pixels
                    )

                    # Extract calibration coefficients with F-factor method
                    # Assume fix F2 factor, F2=1+Var(gain)/Mean(Gain)**2 must be
                    # known from elsewhere
                    F2 = 1.1

                    # calculate photon-electrons
                    n_pe = (
                        F2
                        * (ff_data.charge_median - ped_data.charge_median) ** 2
                        / (ff_data.charge_std**2 - ped_data.charge_std**2)
                    )

                    # fill WaveformCalibrationContainer (this is an example)
                    calib_data.time = ff_data.sample_time
                    calib_data.time_range = ff_data.sample_time_range
                    calib_data.n_pe = n_pe
                    calib_data.dc_to_pe = n_pe / ff_data.charge_median
                    calib_data.time_correction = -ff_data.relative_time_median

                    ped_extractor_name = self.config.get("PedestalCalculator").get(
                        "charge_product"
                    )
                    ped_width = self.config.get(ped_extractor_name).get("window_width")
                    calib_data.pedestal_per_sample = ped_data.charge_median / ped_width

                    # save the config, to be retrieved as data.meta['config']
                    if not ff_initialized:
                        calib_data.meta["config"] = self.config
                        ff_initialized = True
                    else:
                        # write only after a first event (used to initialize the mask)
                        self.log.debug("write flatfield data")
                        self.writer.write("flatfield", ff_data)
                        self.log.debug("write pixel_status data")
                        self.writer.write("pixel_status", status_data)
                        self.log.debug("write calibration data")
                        self.writer.write("calibration", calib_data)

    def finish(self):
        Provenance().add_output_file(self.output_file, role="mon.tel.calibration")
        self.writer.close()


def main():
    exe = CalibrationHDF5Writer()

    exe.run()


if __name__ == "__main__":
    main()
