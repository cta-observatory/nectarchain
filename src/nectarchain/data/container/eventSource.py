import logging
import struct

import numpy as np
from ctapipe.containers import EventType
from ctapipe_io_nectarcam import (
    NectarCAMDataContainer,
    NectarCAMEventSource,
    TriggerBits,
    time_from_unix_tai_ns,
)
from ctapipe_io_nectarcam.anyarray_dtypes import (
    CDTS_AFTER_37201_DTYPE,
    CDTS_BEFORE_37201_DTYPE,
    TIB_DTYPE,
)
from ctapipe_io_nectarcam.constants import N_PIXELS
from ctapipe_io_nectarcam.containers import NectarCAMEventContainer

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


__all__ = ["LightNectarCAMEventSource"]


def fill_nectarcam_event_container_from_zfile(self, array_event, event):
    """Fill the NectarCAM event container from the zfile event data.

    Parameters:
    - array_event: The NectarCAMDataContainer object to fill with event data.
    - event: The event data from the zfile.

    Returns:
    - None

    This function fills the NectarCAM event container in the NectarCAMDataContainer
    object with the event data from the zfile. It unpacks the necessary data from the
    event and assigns it to the corresponding fields in the event container.

    The function performs the following steps:
    1. Assigns the tel_id to the local variable.
    2. Creates a new NectarCAMEventContainer object and assigns it to the
    event_container field of the NectarCAMDataContainer object.
    3. Assigns the extdevices_presence field of the event to the extdevices_presence
    field of the event_container.
    4. Assigns the counters field of the event to the counters field of the
    event_container.
    5. Unpacks the TIB data from the event and assigns it to the corresponding fields in
    the event_container.
    6. Unpacks the CDTS data from the event and assigns it to the corresponding fields
    in the event_container.
    7. Calls the unpack_feb_data function to unpack the FEB counters and trigger pattern
    from the event and assign them to the corresponding fields in the event_container.
    """

    tel_id = self.tel_id
    event_container = NectarCAMEventContainer()
    array_event.nectarcam.tel[tel_id].evt = event_container
    event_container.extdevices_presence = event.nectarcam.extdevices_presence
    event_container.counters = event.nectarcam.counters
    # unpack TIB data
    unpacked_tib = event.nectarcam.tib_data.view(TIB_DTYPE)[0]
    event_container.tib_masked_trigger = unpacked_tib[4]
    # unpack CDTS data
    is_old_cdts = len(event.nectarcam.cdts_data) < 36
    if is_old_cdts:
        unpacked_cdts = event.nectarcam.cdts_data.view(CDTS_BEFORE_37201_DTYPE)[0]
        event_container.ucts_event_counter = unpacked_cdts[0]
        event_container.ucts_timestamp = unpacked_cdts[3]
        event_container.ucts_trigger_type = unpacked_cdts[5]
    else:
        unpacked_cdts = event.nectarcam.cdts_data.view(CDTS_AFTER_37201_DTYPE)[0]
        event_container.ucts_timestamp = unpacked_cdts[0]
        event_container.ucts_event_counter = unpacked_cdts[2]
        event_container.ucts_busy_counter = unpacked_cdts[3]
        event_container.ucts_trigger_type = unpacked_cdts[6]
        # Unpack FEB counters and trigger pattern
        self.unpack_feb_data(event_container, event)


def unpack_feb_data(self, event_container, event):
    """Unpack FEB counters and trigger pattern."""
    # Deduce data format version
    bytes_per_module = (
        len(event.nectarcam.counters) // self.nectarcam_service.num_modules
    )
    # Remain compatible with data before addition of trigger pattern
    module_fmt = "IHHIBBBBBBBB" if bytes_per_module > 16 else "IHHIBBBB"
    n_fields = len(module_fmt)
    rec_fmt = "=" + module_fmt * self.nectarcam_service.num_modules
    # Unpack
    unpacked_feb = struct.unpack(rec_fmt, event.nectarcam.counters)
    # Initialize field containers
    if bytes_per_module > 16:
        n_patterns = 4
        event_container.trigger_pattern = np.zeros(
            shape=(n_patterns, N_PIXELS), dtype=bool
        )

    for module_idx, module_id in enumerate(self.nectarcam_service.module_ids):
        offset = module_id * 7
        if bytes_per_module > 16:
            field_id = 8
            # Decode trigger pattern
            for pattern_id in range(n_patterns):
                value = unpacked_feb[n_fields * module_idx + field_id + pattern_id]
                module_pattern = [
                    int(digit) for digit in reversed(bin(value)[2:].zfill(7))
                ]
                event_container.trigger_pattern[
                    pattern_id, offset : offset + 7
                ] = module_pattern


def fill_trigger_info(self, array_event):
    """Fill the trigger information for a given event.

    Parameters:
        array_event (NectarCAMEventContainer): The NectarCAMEventContainer object to
        fill with trigger information.

    Returns:
        None

    Raises:
        None
    """

    tel_id = self.tel_id
    nectarcam = array_event.nectarcam.tel[tel_id]
    tib_available = nectarcam.evt.extdevices_presence & 1
    ucts_available = nectarcam.evt.extdevices_presence & 2
    # fill trigger time using UCTS timestamp
    trigger = array_event.trigger
    trigger_time = nectarcam.evt.ucts_timestamp
    trigger_time = time_from_unix_tai_ns(trigger_time)
    trigger.time = trigger_time
    trigger.tels_with_trigger = [tel_id]
    trigger.tel[tel_id].time = trigger.time
    # decide which source to use, if both are available,
    # the option decides, if not, fallback to the avilable source
    # if no source available, warn and do not fill trigger info
    if tib_available and ucts_available:
        if self.default_trigger_type == "ucts":
            trigger_bits = nectarcam.evt.ucts_trigger_type
        else:
            trigger_bits = nectarcam.evt.tib_masked_trigger
    elif tib_available:
        trigger_bits = nectarcam.evt.tib_masked_trigger
    elif ucts_available:
        trigger_bits = nectarcam.evt.ucts_trigger_type
    else:
        self.log.warning("No trigger info available.")
        trigger.event_type = EventType.UNKNOWN
        return
    if (
        ucts_available
        and nectarcam.evt.ucts_trigger_type == 42  # TODO check if it's correct
        and self.default_trigger_type == "ucts"
    ):
        self.log.warning(
            "Event with UCTS trigger_type 42 found."
            " Probably means unreliable or shifted UCTS data."
            ' Consider switching to TIB using `default_trigger_type="tib"`'
        )
    # first bit mono trigger, second stereo.
    # If *only* those two are set, we assume it's a physics event
    # for all other we only check if the flag is present
    if (trigger_bits & TriggerBits.PHYSICS) and not (trigger_bits & TriggerBits.OTHER):
        trigger.event_type = EventType.SUBARRAY
    elif trigger_bits & TriggerBits.CALIBRATION:
        trigger.event_type = EventType.FLATFIELD
    elif trigger_bits & TriggerBits.PEDESTAL:
        trigger.event_type = EventType.SKY_PEDESTAL
    elif trigger_bits & TriggerBits.SINGLE_PE:
        trigger.event_type = EventType.SINGLE_PE
    else:
        self.log.warning(
            f"Event {array_event.index.event_id} has unknown event type, trigger: "
            f"{trigger_bits:08b}"
        )
        trigger.event_type = EventType.UNKNOWN


class LightNectarCAMEventSource(NectarCAMEventSource):
    """LightNectarCAMEventSource is a subclass of NectarCAMEventSource that
    provides a generator for iterating over NectarCAM events.

    This implementation of the NectarCAMEventSource is much lighter than the one within
    ctapipe_io_nectarcam, only the fields interesting for nectarchain are kept.

    Parameters
    ----------
    input_url : str
        The input URL of the data source.
    max_events : int
        The maximum number of events to process.
    tel_id : int
        The telescope ID.
    nectarcam_service : NectarCAMService
        The service container for NectarCAM.
    trigger_information : bool
        Flag indicating whether to fill trigger information in the event container.
    obs_ids : list
        The list of observation IDs.
    multi_file : MultiFileReader
        The multi-file reader for reading the data source.
    r0_r1_calibrator : R0R1Calibrator
        The calibrator for R0 to R1 conversion.
    calibrate_flatfields_and_pedestals : bool
        Flag indicating whether to calibrate flatfield and pedestal events.
    """

    def _generator(self):
        """The generator function that yields NectarCAMDataContainer objects
        representing each event.

        Yields
        ------
        NectarCAMDataContainer :
            The NectarCAMDataContainer object representing each event.

        Raises
        ------
        None
        """
        # container for NectarCAM data
        array_event = NectarCAMDataContainer()
        array_event.meta["input_url"] = self.input_url
        array_event.meta["max_events"] = self.max_events
        array_event.meta["origin"] = "NectarCAM"

        # also add service container to the event section
        array_event.nectarcam.tel[self.tel_id].svc = self.nectarcam_service

        # initialize general monitoring container
        self.initialize_mon_container(array_event)

        # loop on events
        for count, event in enumerate(self.multi_file):
            array_event.count = count
            array_event.index.event_id = event.event_id
            array_event.index.obs_id = self.obs_ids[0]

            # fill R0/R1 data
            self.fill_r0r1_container(array_event, event)
            # fill specific NectarCAM event data
            self.fill_nectarcam_event_container_from_zfile(array_event, event)

            if self.trigger_information:
                self.fill_trigger_info(array_event)

            yield array_event
