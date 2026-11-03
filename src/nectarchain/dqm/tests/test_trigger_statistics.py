import numpy as np
from astropy.table import Table
from ctapipe.containers import EventType
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam import LightNectarCAMEventSource as EventSource
from ctapipe_io_nectarcam.constants import HIGH_GAIN
from traitlets.config import Config

from nectarchain.dqm.trigger_statistics import TriggerStatistics


class TestTriggerStatistics:
    """Comprehensive tests for TriggerStatistics processor."""

    @classmethod
    def setup_class(cls):
        """Setup test fixtures once for all tests."""
        cls.paths = [
            get_dataset_path("NectarCAM.Run5288.0001.fits.fz"),
            get_dataset_path("NectarCAM.Run4717.0000.fits.fz"),
        ]

        cls.config = Config(
            dict(
                NectarCAMEventSource=dict(
                    NectarCAMR0Corrections=dict(
                        calibration_path=None,
                        apply_flatfield=False,
                        select_gain=False,
                    )
                )
            )
        )

    def test_define_for_run(self):
        """Test define_for_run method extracts correct geometry."""
        reader1 = EventSource(input_url=self.paths[0], config=self.config, max_events=1)

        Pix, Samp = TriggerStatistics(HIGH_GAIN, r0=True).define_for_run(reader1)

        # Verify geometry is extracted correctly
        assert Pix == 1855  # NectarCAM 1855 pixels
        assert Samp == 60  # 60 samples
        assert Pix + Samp == 1915

    def test_configure_for_run(self):
        """Test configure_for_run method sets up processor correctly."""
        reader1 = EventSource(input_url=self.paths[0], config=self.config, max_events=1)
        tel_id = reader1.subarray.tel_ids[0]

        processor = TriggerStatistics(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader1)

        processor.configure_for_run(self.paths[0], Pix, Samp, reader1)

        assert processor.Pix == 1855
        assert processor.Samp == 60
        assert processor.tel_id == tel_id

    def test_process_event(self):
        """Test process_event method processes events correctly."""
        reader = EventSource(input_url=self.paths[0], config=self.config, max_events=5)

        processor = TriggerStatistics(HIGH_GAIN, r0=True)
        Pix, Samp = processor.define_for_run(reader)
        processor.configure_for_run(self.paths[0], Pix, Samp, reader)

        evt_count = 0
        for evt in reader:
            processor.process_event(evt, noped=False)
            evt_count += 1

        assert evt_count == 5
        assert len(processor.event_type) == 5
        assert len(processor.event_times) == 5
        assert len(processor.event_id) == 5
        assert len(processor.run_times) == 5

    def test_finish_run(self):
        """Test finish_run method computes statistics correctly."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=5)

            processor = TriggerStatistics(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()

            run_start1 = processor.run_times[
                processor.event_id == np.min(processor.event_id)
            ]
            run_start = processor.event_times[
                processor.event_id == np.min(processor.event_id)
            ]
            run_end = np.max(processor.event_times)

            is_wrong = processor.event_times < processor.run_start
            event_wrong_times = processor.event_times[is_wrong]

            valid = processor.event_times >= processor.run_start
            event_id = processor.event_id[valid]
            event_times = processor.event_times[valid]
            event_type = processor.event_type[valid]

            _ped_mask = processor.event_type == EventType.SKY_PEDESTAL.value
            _phy_mask = processor.event_type == EventType.SUBARRAY.value
            # TODO: add ids and event time selection for
            # other event types, e.g., dark pedestals
            _other_mask = ~processor._ped_mask & ~processor._phy_mask

            assert processor.triggers == np.unique(processor.event_type)
            assert isinstance(processor.event_id, np.ndarray)
            assert isinstance(processor.run_times, np.ndarray)
            assert isinstance(processor.event_times, np.ndarray)
            assert isinstance(processor.event_type, np.ndarray)
            assert processor.run_start1 == run_start1
            assert processor.run_start == run_start
            assert processor.run_end == run_end

            if len(processor.event_wrong_times) > 0:
                assert processor.event_wrong_times == event_wrong_times

            assert np.array_equal(processor.event_id, event_id)
            assert np.array_equal(processor.event_times, event_times)
            assert np.array_equal(processor.event_type, event_type)
            assert np.array_equal(processor._phy_mask, _phy_mask)
            assert np.array_equal(processor._ped_mask, _ped_mask)
            assert np.array_equal(processor._other_mask, _other_mask)

    def test_get_results(self):
        """Test get_results method returns proper dictionary structure."""
        for path in self.paths:
            reader = EventSource(input_url=path, config=self.config, max_events=100)

            processor = TriggerStatistics(HIGH_GAIN, r0=True)
            Pix, Samp = processor.define_for_run(reader)
            processor.configure_for_run(path, Pix, Samp, reader)

            for evt in reader:
                processor.process_event(evt, noped=False)

            processor.finish_run()

            _ped_mask = processor.event_type == EventType.SKY_PEDESTAL.value
            _phy_mask = processor.event_type == EventType.SUBARRAY.value
            # TODO: add ids and event time selection for
            # other event types, e.g., dark pedestals
            _other_mask = ~processor._ped_mask & ~processor._phy_mask

            is_wrong = processor.event_times < processor.run_start
            event_wrong_times = processor.event_times[is_wrong]

            results = processor.get_results()

            assert isinstance(results, dict)

            triggers = np.unique(processor.event_type)

            assert np.array_equal(results["TRIGGER-TYPES"], triggers)

            assert isinstance(results["TRIGGER-STATISTICS"], dict)
            assert isinstance(results["TRIGGER-EVENTS-ALL"], Table)
            if _phy_mask.sum() > 0:
                assert isinstance(results["TRIGGER-EVENTS-PHY"], Table)
            if _ped_mask.sum() > 0:
                assert isinstance(results["TRIGGER-EVENTS-PED"], Table)
            if _other_mask.sum() > 0:
                assert isinstance(results["TRIGGER-EVENTS-OTHERS"], Table)
            if len(event_wrong_times) > 0:
                assert isinstance(results["TRIGGER-EVENTS-WRONG"], Table)
            assert isinstance(results["START-TIMES"], dict)

    def test_plot_results(self):
        """Test plot_results method generates plots correctly."""
        import tempfile

        from matplotlib.figure import Figure

        for path in self.paths:
            with tempfile.TemporaryDirectory() as tmpdir:
                reader = EventSource(input_url=path, config=self.config, max_events=3)

                processor = TriggerStatistics(HIGH_GAIN, r0=True)
                Pix, Samp = processor.define_for_run(reader)
                processor.configure_for_run(path, Pix, Samp, reader)

                for evt in reader:
                    processor.process_event(evt, noped=False)

                processor.finish_run()

                fig_dict, fig_names_dict = processor.plot_results("test", tmpdir)

                assert isinstance(fig_dict, dict)
                assert isinstance(fig_names_dict, dict)

                assert "TRIGGER-STATISTICS" in fig_dict
                assert "EVENT-TIME" in fig_dict
                assert "EVENT-ID" in fig_dict

                assert isinstance(fig_dict["TRIGGER-STATISTICS"], Figure)
                assert isinstance(fig_dict["EVENT-TIME"], Figure)
                assert isinstance(fig_dict["EVENT-ID"], Figure)

                assert "TRIGGER-STATISTICS" in fig_names_dict
                assert "EVENT-TIME" in fig_names_dict
                assert "EVENT-ID" in fig_names_dict

                # Verify paths are absolute and contain the temp directory
                assert tmpdir in fig_names_dict["TRIGGER-STATISTICS"]
                assert tmpdir in fig_names_dict["EVENT-TIME"]
                assert tmpdir in fig_names_dict["EVENT-ID"]

                # Verify filenames match expected pattern
                assert (
                    "test_Trigger_Statistics.png"
                    in fig_names_dict["TRIGGER-STATISTICS"]
                )
                assert "test_Event_rate.png" in fig_names_dict["EVENT-TIME"]
                assert "test_Event_IDs.png" in fig_names_dict["EVENT-ID"]
