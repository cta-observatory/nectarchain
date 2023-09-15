from nectarchain.makers import ChargesMaker,WaveformsMaker
from nectarchain.data.container import ChargesContainer,ChargesContainerIO
from ctapipe.containers import EventType
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',level = logging.DEBUG)
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

class TestChargesMaker:
    run_number = 3938
    max_events = 100

    def test_instance(self) : 
        chargesMaker = ChargesMaker(run_number = TestChargesMaker.run_number,
                                max_events = TestChargesMaker.max_events)
        assert isinstance(chargesMaker,ChargesMaker)

    def test_shape_valid(self):
        chargesMaker = ChargesMaker(run_number = TestChargesMaker.run_number,
                        max_events = TestChargesMaker.max_events)
        chargesContainer = chargesMaker.make()[0]

        assert chargesContainer.nevents <= TestChargesMaker.max_events
        assert chargesContainer.run_number == TestChargesMaker.run_number
        assert chargesContainer.ucts_timestamp.shape == (chargesContainer.nevents,) 
        assert chargesContainer.ucts_busy_counter.shape == (chargesContainer.nevents,)
        assert chargesContainer.ucts_event_counter.shape == (chargesContainer.nevents,) 
        assert chargesContainer.event_type.shape == (chargesContainer.nevents,) 
        assert chargesContainer.event_id.shape == (chargesContainer.nevents,) 
        assert chargesContainer.trig_pattern_all.shape[0] == chargesContainer.nevents
        assert chargesContainer.trig_pattern_all.shape[2] == 4

        assert chargesContainer.trig_pattern.shape[0] == chargesContainer.nevents
        assert chargesContainer.multiplicity.shape == (chargesContainer.nevents,)

        assert chargesContainer.charges_hg.mean() != 0
        assert chargesContainer.charges_lg.mean() != 0
        assert chargesContainer.peak_hg.mean() != 0
        assert chargesContainer.peak_lg.mean() != 0
        assert chargesContainer.charges_hg.shape == (chargesContainer.nevents,chargesContainer.npixels)
        assert chargesContainer.charges_lg.shape == (chargesContainer.nevents,chargesContainer.npixels)
        assert chargesContainer.peak_hg.shape == (chargesContainer.nevents,chargesContainer.npixels)
        assert chargesContainer.peak_lg.shape == (chargesContainer.nevents,chargesContainer.npixels)

        assert chargesContainer.broken_pixels_hg.shape == (chargesContainer.nevents,chargesContainer.npixels)
        assert chargesContainer.broken_pixels_lg.shape == (chargesContainer.nevents,chargesContainer.npixels)

    def test_make_restart_eventsource(self) : 
        chargesMaker = ChargesMaker(run_number = TestChargesMaker.run_number,
                                max_events = TestChargesMaker.max_events)
        chargesContainer_list = chargesMaker.make(restart_from_begining = True)
        assert isinstance(chargesContainer_list[0],ChargesContainer)

    def test_make_LocalPeakWindowSum(self) : 
        chargesMaker = ChargesMaker(run_number = TestChargesMaker.run_number,
                                max_events = TestChargesMaker.max_events)
        chargesContainer_list = chargesMaker.make(method = "LocalPeakWindowSum",window_shift = -4, window_length = 16)
        assert isinstance(chargesContainer_list[0],ChargesContainer)
        
    def test_all_multiple_trigger(self) : 
        trigger1 = EventType.FLATFIELD
        trigger2 = EventType.SKY_PEDESTAL
        chargesMaker = ChargesMaker(run_number = TestChargesMaker.run_number,
                                max_events = TestChargesMaker.max_events)
        chargesContainer_list = chargesMaker.make(trigger_type = [trigger1,trigger2])
        for chargesContainer in chargesContainer_list : 
            assert isinstance(chargesContainer,ChargesContainer)
        
        
    def test_all_trigger_None(self) : 
        chargesMaker = ChargesMaker(run_number = TestChargesMaker.run_number,
                                max_events = TestChargesMaker.max_events)
        chargesContainer_list = chargesMaker.make()
        assert isinstance(chargesContainer_list[0],ChargesContainer)

    def test_create_from_waveforms(self)  : 
        waveformsMaker = WaveformsMaker(run_number = TestChargesMaker.run_number,
                                max_events = TestChargesMaker.max_events)
        waveformsContainer_list = waveformsMaker.make()
        chargesContainer = ChargesMaker.create_from_waveforms(waveformsContainer_list[0],method = "LocalPeakWindowSum",window_shift = -4, window_length = 16)
        assert isinstance(chargesContainer,ChargesContainer)

    def test_select_charges(self) : 
        chargesMaker = ChargesMaker(run_number = TestChargesMaker.run_number,
                                max_events = TestChargesMaker.max_events)
        chargesContainer_list = chargesMaker.make()
        pixel_id = np.array([3,67,87])
        assert isinstance(ChargesMaker.select_charges_hg(chargesContainer_list[0],pixel_id),np.ndarray)
        assert isinstance(ChargesMaker.select_charges_lg(chargesContainer_list[0],pixel_id),np.ndarray)

    def test_histo(self) : 
        chargesMaker = ChargesMaker(run_number = TestChargesMaker.run_number,
                                max_events = TestChargesMaker.max_events)
        chargesContainer_list = chargesMaker.make()
        histo = ChargesMaker.histo_hg(chargesContainer_list[0])
        assert isinstance(histo,np.ndarray)
        assert histo.mean() != 0
        assert histo.shape[0] == 2
        assert histo.shape[1] == chargesContainer_list[0].npixels
        histo = ChargesMaker.histo_lg(chargesContainer_list[0])
        assert isinstance(histo,np.ndarray)
        assert histo.mean() != 0
        assert histo.shape[0] == 2
        assert histo.shape[1] == chargesContainer_list[0].npixels


    def test_sort_ChargesContainer(self) :
        chargesMaker = ChargesMaker(run_number = TestChargesMaker.run_number,
                        max_events = TestChargesMaker.max_events)
        chargesContainer_list = chargesMaker.make()
        sortWfs = ChargesMaker.sort(chargesContainer_list[0],method = 'event_id')
        assert np.array_equal(sortWfs.event_id ,np.sort(chargesContainer_list[0].event_id))

    def test_write_load_container(self) : 
        chargesMaker = ChargesMaker(run_number = TestChargesMaker.run_number,
                                max_events = TestChargesMaker.max_events)
        chargesContainer_list = chargesMaker.make()
        ChargesContainerIO.write("/tmp/test_charge_container/",chargesContainer_list[0],overwrite = True)
        loaded_charge = ChargesContainerIO.load(f"/tmp/test_charge_container",run_number = TestChargesMaker.run_number)
        assert np.array_equal(chargesContainer_list[0].charges_hg,loaded_charge.charges_hg)

if __name__ == '__main__' : 
    import logging
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',level = logging.DEBUG)
    log = logging.getLogger(__name__)
    log.handlers = logging.getLogger('__main__').handlers
    TestChargesMaker().test_write_load_container()