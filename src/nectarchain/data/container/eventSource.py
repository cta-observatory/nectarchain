import logging
import sys

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


import struct
import numpy as np

from ctapipe_io_nectarcam import NectarCAMEventSource,NectarCAMDataContainer,time_from_unix_tai_ns,TriggerBits
from ctapipe_io_nectarcam.containers import NectarCAMEventContainer
from ctapipe_io_nectarcam.anyarray_dtypes import (
    CDTS_AFTER_37201_DTYPE,
    CDTS_BEFORE_37201_DTYPE,
    TIB_DTYPE,
)
from ctapipe_io_nectarcam.constants import (
    N_GAINS, N_PIXELS, N_SAMPLES
)
from ctapipe.containers import EventType

__all__ = ["LightNectarCAMEventSource"]

#DONE event.trigger.event_type
#DONE event.index.event_id
#DONE event.nectarcam.tel[__class__.TEL_ID.default_value].evt.ucts_timestamp
#DONE event.nectarcam.tel[__class__.TEL_ID.default_value].evt.ucts_busy_counter
#DONE event.nectarcam.tel[__class__.TEL_ID.default_value].evt.ucts_event_counter
#DONE event.nectarcam.tel[__class__.TEL_ID.default_value].evt.trigger_pattern
#DONE event.r0.tel[0].waveform



def fill_nectarcam_event_container_from_zfile(self,array_event, event) :
    tel_id = self.tel_id
    event_container = NectarCAMEventContainer()
    array_event.nectarcam.tel[tel_id].evt = event_container
    #event_container.configuration_id = event.configuration_id
    #event_container.event_id = event.event_id
    #event_container.tel_event_id = event.tel_event_id
    #event_container.pixel_status = event.pixel_status
    #event_container.ped_id = event.ped_id
    #event_container.module_status = event.nectarcam.module_status
    event_container.extdevices_presence = event.nectarcam.extdevices_presence
    #event_container.swat_data = event.nectarcam.swat_data
    event_container.counters = event.nectarcam.counters
    ## unpack TIB data
    unpacked_tib = event.nectarcam.tib_data.view(TIB_DTYPE)[0]
    #event_container.tib_event_counter = unpacked_tib[0]
    #event_container.tib_pps_counter = unpacked_tib[1]
    #event_container.tib_tenMHz_counter = unpacked_tib[2]
    #event_container.tib_stereo_pattern = unpacked_tib[3]
    event_container.tib_masked_trigger = unpacked_tib[4]
    # unpack CDTS data
    is_old_cdts = len(event.nectarcam.cdts_data) < 36
    if is_old_cdts:
        unpacked_cdts = event.nectarcam.cdts_data.view(CDTS_BEFORE_37201_DTYPE)[0]
        event_container.ucts_event_counter = unpacked_cdts[0]
        #event_container.ucts_pps_counter = unpacked_cdts[1]
        #event_container.ucts_clock_counter = unpacked_cdts[2]
        event_container.ucts_timestamp = unpacked_cdts[3]
        #event_container.ucts_camera_timestamp = unpacked_cdts[4]
        event_container.ucts_trigger_type = unpacked_cdts[5]
        #event_container.ucts_white_rabbit_status = unpacked_cdts[6]
    else:
        unpacked_cdts = event.nectarcam.cdts_data.view(CDTS_AFTER_37201_DTYPE)[0]
        event_container.ucts_timestamp = unpacked_cdts[0]
        #event_container.ucts_address = unpacked_cdts[1]  # new
        event_container.ucts_event_counter = unpacked_cdts[2]
        event_container.ucts_busy_counter = unpacked_cdts[3]  # new
        #event_container.ucts_pps_counter = unpacked_cdts[4]
        #event_container.ucts_clock_counter = unpacked_cdts[5]
        event_container.ucts_trigger_type = unpacked_cdts[6]
        #event_container.ucts_white_rabbit_status = unpacked_cdts[7]
        #event_container.ucts_stereo_pattern = unpacked_cdts[8]  # new
        #event_container.ucts_num_in_bunch = unpacked_cdts[9]  # new
        #event_container.cdts_version = unpacked_cdts[10]  # new
        # Unpack FEB counters and trigger pattern
        self.unpack_feb_data(event_container, event)

def unpack_feb_data(self, event_container, event):
    '''Unpack FEB counters and trigger pattern'''
    # Deduce data format version
    bytes_per_module = len(
        event.nectarcam.counters) // self.camera_config.nectarcam.num_modules
    # Remain compatible with data before addition of trigger pattern
    module_fmt = 'IHHIBBBBBBBB' if bytes_per_module > 16 else 'IHHIBBBB'
    n_fields = len(module_fmt)
    rec_fmt = '=' + module_fmt * self.camera_config.nectarcam.num_modules
    # Unpack
    unpacked_feb = struct.unpack(rec_fmt, event.nectarcam.counters)
    # Initialize field containers
    #n_camera_modules = N_PIXELS // 7
    #event_container.feb_abs_event_id = np.zeros(shape=(n_camera_modules,), dtype=np.uint32)
    #event_container.feb_event_id = np.zeros(shape=(n_camera_modules,), dtype=np.uint16)
    #event_container.feb_pps_cnt = np.zeros(shape=(n_camera_modules,), dtype=np.uint16)
    #event_container.feb_ts1 = np.zeros(shape=(n_camera_modules,), dtype=np.uint32)
    #event_container.feb_ts2_trig = np.zeros(shape=(n_camera_modules,), dtype=np.int16)
    #event_container.feb_ts2_pps = np.zeros(shape=(n_camera_modules,), dtype=np.int16)
    if bytes_per_module > 16:
        n_patterns = 4
        event_container.trigger_pattern = np.zeros(shape=(n_patterns, N_PIXELS),
                                                   dtype=bool)
    # Unpack absolute event ID
    #event_container.feb_abs_event_id[
    #    self.camera_config.nectarcam.expected_modules_id] = unpacked_feb[0::n_fields]
    ## Unpack PPS counter
    #event_container.feb_pps_cnt[
    #    self.camera_config.nectarcam.expected_modules_id] = unpacked_feb[1::n_fields]
    ## Unpack relative event ID
    #event_container.feb_event_id[
    #    self.camera_config.nectarcam.expected_modules_id] = unpacked_feb[2::n_fields]
    ## Unpack TS1 counter
    #event_container.feb_ts1[
    #    self.camera_config.nectarcam.expected_modules_id] = unpacked_feb[3::n_fields]
    ## Unpack TS2 counters
    #ts2_decimal = lambda bits: bits - (1 << 8) if bits & 0x80 != 0 else bits
    #ts2_decimal_vec = np.vectorize(ts2_decimal)
    #event_container.feb_ts2_trig[
    #    self.camera_config.nectarcam.expected_modules_id] = ts2_decimal_vec(
    #    unpacked_feb[4::n_fields])
    #event_container.feb_ts2_pps[
    #    self.camera_config.nectarcam.expected_modules_id] = ts2_decimal_vec(
    #    unpacked_feb[5::n_fields])
    # Loop over modules
    
    
    for module_idx, module_id in enumerate(
            self.camera_config.nectarcam.expected_modules_id):
        offset = module_id * 7
        if bytes_per_module > 16:
            field_id = 8
            # Decode trigger pattern
            for pattern_id in range(n_patterns):
                value = unpacked_feb[n_fields * module_idx + field_id + pattern_id]
                module_pattern = [int(digit) for digit in
                                  reversed(bin(value)[2:].zfill(7))]
                event_container.trigger_pattern[pattern_id,
                offset:offset + 7] = module_pattern
    
    
    # Unpack native charge
    #if len(event.nectarcam.charges_gain1) > 0:
    #    event_container.native_charge = np.zeros(shape=(N_GAINS, N_PIXELS),
    #                                             dtype=np.uint16)
    #    rec_fmt = '=' + 'H' * self.camera_config.num_pixels
    #    for gain_id in range(N_GAINS):
    #        unpacked_charge = struct.unpack(rec_fmt, getattr(event.nectarcam,
    #                                                         f'charges_gain{gain_id + 1}'))
    #        event_container.native_charge[
    #            gain_id, self.camera_config.expected_pixels_id] = unpacked_charge

def fill_trigger_info(self, array_event):
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
        if self.default_trigger_type == 'ucts':
            trigger_bits = nectarcam.evt.ucts_trigger_type
        else:
            trigger_bits = nectarcam.evt.tib_masked_trigger
    elif tib_available:
        trigger_bits = nectarcam.evt.tib_masked_trigger
    elif ucts_available:
        trigger_bits = nectarcam.evt.ucts_trigger_type
    else:
        self.log.warning('No trigger info available.')
        trigger.event_type = EventType.UNKNOWN
        return
    if (
            ucts_available
            and nectarcam.evt.ucts_trigger_type == 42  # TODO check if it's correct
            and self.default_trigger_type == "ucts"
    ):
        self.log.warning(
            'Event with UCTS trigger_type 42 found.'
            ' Probably means unreliable or shifted UCTS data.'
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
            f'Event {array_event.index.event_id} has unknown event type, trigger: {trigger_bits:08b}')
        trigger.event_type = EventType.UNKNOWN


class LightNectarCAMEventSource(NectarCAMEventSource):
    def _generator(self):

        # container for NectarCAM data
        array_event = NectarCAMDataContainer()
        array_event.meta['input_url'] = self.input_url
        array_event.meta['max_events'] = self.max_events
        array_event.meta['origin'] = 'NectarCAM'

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
            # fill specific NectarCAM event data
            self.fill_nectarcam_event_container_from_zfile(array_event, event)

            if self.trigger_information:
                self.fill_trigger_info(array_event)

            # fill general monitoring data
            #self.fill_mon_container_from_zfile(array_event, event)
#
            ## gain select and calibrate to pe
            #if self.r0_r1_calibrator.calibration_path is not None:
            #    # skip flatfield and pedestal events if asked
            #    if (
            #            self.calibrate_flatfields_and_pedestals
            #            or array_event.trigger.event_type not in {EventType.FLATFIELD,
            #                                                      EventType.SKY_PEDESTAL}
            #    ):
            #        self.r0_r1_calibrator.calibrate(array_event)

            yield array_event