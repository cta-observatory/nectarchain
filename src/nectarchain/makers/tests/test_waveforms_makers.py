# import logging
#
# import numpy as np
# import pytest
# from ctapipe.containers import EventType
#
# from nectarchain.data.container import WaveformsContainer, WaveformsContainerIO
# from nectarchain.makers.waveforms_makers import WaveformsMaker
#
# logging.basicConfig(
#     format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.DEBUG
# )
# log = logging.getLogger(__name__)
# log.handlers = logging.getLogger("__main__").handlers
#
#
# @pytest.disable()
# class TestWaveformsMaker:
#     run_number = 3938
#     max_events = 100
#
#     def test_instance(self):
#         waveformsMaker = WaveformsMaker(
#             run_number=TestWaveformsMaker.run_number,
#             max_events=TestWaveformsMaker.max_events,
#         )
#         assert isinstance(waveformsMaker, WaveformsMaker)
#
#     def test_shape_valid(self):
#         waveformsMaker = WaveformsMaker(
#             run_number=TestWaveformsMaker.run_number,
#             max_events=TestWaveformsMaker.max_events,
#         )
#         waveformsContainer = waveformsMaker.make()[0]
#
#         assert waveformsContainer.nevents <= TestWaveformsMaker.max_events
#         assert waveformsContainer.run_number == TestWaveformsMaker.run_number
#          assert (waveformsContainer.ucts_timestamp.shape ==
#                  (waveformsContainer.nevents,))
#         assert waveformsContainer.ucts_busy_counter.shape == (
#             waveformsContainer.nevents,
#         )
#         assert waveformsContainer.ucts_event_counter.shape == (
#             waveformsContainer.nevents,
#         )
#         assert waveformsContainer.event_type.shape == (waveformsContainer.nevents,)
#         assert waveformsContainer.event_id.shape == (waveformsContainer.nevents,)
#         assert (
#             waveformsContainer.trig_pattern_all.shape[0] == waveformsContainer.nevents
#         )
#         assert waveformsContainer.trig_pattern_all.shape[2] == 4
#
#         assert waveformsContainer.trig_pattern.shape[0] == waveformsContainer.nevents
#         assert waveformsContainer.multiplicity.shape == (waveformsContainer.nevents,)
#
#         assert waveformsContainer.wfs_hg.mean() != 0
#         assert waveformsContainer.wfs_lg.mean() != 0
#         assert waveformsContainer.wfs_hg.shape == (
#             waveformsContainer.nevents,
#             waveformsContainer.npixels,
#             waveformsContainer.nsamples,
#         )
#         assert waveformsContainer.wfs_lg.shape == (
#             waveformsContainer.nevents,
#             waveformsContainer.npixels,
#             waveformsContainer.nsamples,
#         )
#         assert waveformsContainer.broken_pixels_hg.shape == (
#             waveformsContainer.nevents,
#             waveformsContainer.npixels,
#         )
#         assert waveformsContainer.broken_pixels_lg.shape == (
#             waveformsContainer.nevents,
#             waveformsContainer.npixels,
#         )
#
#     def test_all_multiple_trigger(self):
#         trigger1 = EventType.FLATFIELD
#         trigger2 = EventType.SKY_PEDESTAL
#         waveformsMaker = WaveformsMaker(
#             run_number=TestWaveformsMaker.run_number,
#             max_events=TestWaveformsMaker.max_events,
#         )
#         waveformsContainer_list = waveformsMaker.make(
#             trigger_type=[trigger1, trigger2], restart_from_begining=True
#         )
#         for waveformsContainer in waveformsContainer_list:
#             assert isinstance(waveformsContainer, WaveformsContainer)
#             assert waveformsContainer.wfs_hg.mean() != 0
#
#     def test_all_trigger_None(self):
#         waveformsMaker = WaveformsMaker(
#             run_number=TestWaveformsMaker.run_number,
#             max_events=TestWaveformsMaker.max_events,
#         )
#         waveformsContainer_list = waveformsMaker.make()
#         assert isinstance(waveformsContainer_list[0], WaveformsContainer)
#
#     def test_select_waveforms_hg(self):
#         waveformsMaker = WaveformsMaker(
#             run_number=TestWaveformsMaker.run_number,
#             max_events=TestWaveformsMaker.max_events,
#         )
#         waveformsContainer_list = waveformsMaker.make()
#         pixel_id = np.array([3, 67, 87])
#         assert isinstance(
#             WaveformsMaker.select_waveforms_hg(waveformsContainer_list[0], pixel_id),
#             np.ndarray,
#         )
#         assert isinstance(
#             WaveformsMaker.select_waveforms_lg(waveformsContainer_list[0], pixel_id),
#             np.ndarray,
#         )
#
#     def test_sort_WaveformsContainer(self):
#         waveformsMaker = WaveformsMaker(
#             run_number=TestWaveformsMaker.run_number,
#             max_events=TestWaveformsMaker.max_events,
#         )
#         waveformsContainer_list = waveformsMaker.make()
#         sortWfs = WaveformsMaker.sort(waveformsContainer_list[0], method="event_id")
#         assert np.array_equal(
#             sortWfs.event_id, np.sort(waveformsContainer_list[0].event_id)
#         )
#
#     def test_write_load_container(self):
#         waveformsMaker = WaveformsMaker(
#             run_number=TestWaveformsMaker.run_number,
#             max_events=TestWaveformsMaker.max_events,
#         )
#         waveformsContainer_list = waveformsMaker.make()
#         WaveformsContainerIO.write(
#             "/tmp/test_wfs_container/", waveformsContainer_list[0], overwrite=True
#         )
#         loaded_wfs = WaveformsContainerIO.load(
#             "/tmp/test_wfs_container", TestWaveformsMaker.run_number
#         )
#         assert np.array_equal(waveformsContainer_list[0].wfs_hg, loaded_wfs.wfs_hg)
#
#     def test_create_from_events_list(self):
#         waveformsMaker = WaveformsMaker(
#             run_number=TestWaveformsMaker.run_number,
#             max_events=TestWaveformsMaker.max_events,
#         )
#         events_list = []
#         for i, event in enumerate(waveformsMaker._reader):
#             events_list.append(event)
#         waveformsContainer = WaveformsMaker.create_from_events_list(
#             events_list,
#             waveformsMaker.run_number,
#             waveformsMaker.npixels,
#             waveformsMaker.nsamples,
#             waveformsMaker.subarray,
#             waveformsMaker.pixels_id,
#         )
#         assert isinstance(waveformsContainer, WaveformsContainer)
#
#
# if __name__ == "__main__":
#     import logging
#
#     logging.basicConfig(
#         format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.DEBUG
#     )
#     log = logging.getLogger(__name__)
#     log.handlers = logging.getLogger("__main__").handlers
