from nectarchain.makers.waveformsMakers import WaveformsMaker
from nectarchain.data.container import WaveformsContainer,WaveformsContainerIO
from ctapipe.containers import EventType
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',level = logging.DEBUG)
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

class TestWaveformsMaker:
    run_number = 3938
    max_events = 1000

    def test_instance(self) : 
        waveformsMaker = WaveformsMaker(run_number = TestWaveformsMaker.run_number,
                                max_events = TestWaveformsMaker.max_events)
        assert isinstance(waveformsMaker,WaveformsMaker)
        
    def test_all_multiple_trigger(self) : 
        trigger1 = EventType.FLATFIELD
        trigger2 = EventType.SKY_PEDESTAL
        waveformsMaker = WaveformsMaker(run_number = TestWaveformsMaker.run_number,
                                max_events = TestWaveformsMaker.max_events)
        waveformsContainer_list = waveformsMaker.make(trigger_type = [trigger1,trigger2],restart_from_begining = True)
        for waveformsContainer in waveformsContainer_list : 
            assert isinstance(waveformsContainer,WaveformsContainer)
        
        
    def test_all_trigger_None(self) : 
        waveformsMaker = WaveformsMaker(run_number = TestWaveformsMaker.run_number,
                                max_events = TestWaveformsMaker.max_events)
        waveformsContainer_list = waveformsMaker.make()
        assert isinstance(waveformsContainer_list[0],WaveformsContainer)

    def test_select_waveforms_hg(self) : 
        waveformsMaker = WaveformsMaker(run_number = TestWaveformsMaker.run_number,
                                max_events = TestWaveformsMaker.max_events)
        waveformsContainer_list = waveformsMaker.make()
        pixel_id = np.array([3,67,87])
        WaveformsMaker.select_waveforms_hg(waveformsContainer_list[0],pixel_id)

    def test_sort_WaveformsContainer(self) :
        waveformsMaker = WaveformsMaker(run_number = TestWaveformsMaker.run_number,
                        max_events = TestWaveformsMaker.max_events)
        waveformsContainer_list = waveformsMaker.make()
        sortWfs = WaveformsMaker.sort(waveformsContainer_list[0],method = 'event_id')
        assert np.array_equal(sortWfs.event_id ,np.sort(waveformsContainer_list[0].event_id))

    def test_write_load_container(self) : 
            waveformsMaker = WaveformsMaker(run_number = TestWaveformsMaker.run_number,
                                    max_events = TestWaveformsMaker.max_events)
            waveformsContainer_list = waveformsMaker.make()
            WaveformsContainerIO.write("/tmp/test_wfs_container/",waveformsContainer_list[0],overwrite = True)
            loaded_wfs = WaveformsContainerIO.load(f"/tmp/test_wfs_container/waveforms_run{TestWaveformsMaker.run_number}.fits")

if __name__ == '__main__' : 
    import logging
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',level = logging.DEBUG)
    log = logging.getLogger(__name__)
    log.handlers = logging.getLogger('__main__').handlers
    TestWaveformsMaker().test_write_load_container()