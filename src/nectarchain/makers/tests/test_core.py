import logging
from ctapipe.utils import get_dataset_path
from nectarchain.makers.core import (
    BaseNectarCAMCalibrationTool,
    EventsLoopNectarCAMCalibrationTool,
    DelimiterLoopNectarCAMCalibrationTool
    )
from ctapipe.containers import EventType
from nectarchain.data import NectarCAMContainer
from nectarchain.makers.component import NectarCAMComponent
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from ctapipe_io_nectarcam import LightNectarCAMEventSource
import os
import pathlib
import pytest
from unittest.mock import patch
import numpy as np
import traitlets
from ctapipe.core.container import Container
from nectarchain.data.container.core import (
    NectarCAMContainer, 
    TriggerMapContainer,
)
from ctapipe.containers import TriggerContainer
from unittest.mock import MagicMock


logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.DEBUG
)
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class TestBaseNectarCAMCalibrationTool:
    RUN_NUMBER = 3938
    RUN_FILE = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
    
    def test_load_run(self):
        eventsource = BaseNectarCAMCalibrationTool.load_run(run_number = self.RUN_NUMBER,max_events=1,run_file = self.RUN_FILE)
        assert isinstance(eventsource,LightNectarCAMEventSource)

class MockComponent(NectarCAMComponent):
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, event, *args, **kwargs):
        pass
    def start(self):
        pass
    def finish(self):
        return [NectarCAMContainer()]

class TestEventsLoopNectarCAMCalibrationTool(TestBaseNectarCAMCalibrationTool):
    MAX_EVENTS = 10
    EVENTS_PER_SLICE = 8
    

    
    @pytest.fixture
    def tool_instance(self):
        return EventsLoopNectarCAMCalibrationTool(
            run_number=self.RUN_NUMBER,
            )
    @pytest.fixture
    def tool_instance_run_file(self):
        return EventsLoopNectarCAMCalibrationTool(
            run_number=self.RUN_NUMBER,
            run_file=self.RUN_FILE,
            )
    
    def test_init_output_path(self, tool_instance):
        expected_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA', '/tmp')}/runs/EventsLoopNectarCAMCalibration_run{self.RUN_NUMBER}.h5"
        )
        assert tool_instance.output_path == expected_path
        assert tool_instance.run_number == self.RUN_NUMBER
        assert tool_instance.max_events is None
        assert tool_instance.run_file is None
        assert tool_instance.name == "EventsLoopNectarCAMCalibration"
        assert tool_instance.events_per_slice is None
        
    def test_init_with_output_path(self):
        custom_path = pathlib.Path("/custom/path/output.h5")
        tool_instance = EventsLoopNectarCAMCalibrationTool(run_number=self.RUN_NUMBER, output_path=custom_path)
        assert tool_instance.output_path == custom_path
        
    def test_init_with_max_events(self):
        tool_instance = EventsLoopNectarCAMCalibrationTool(run_number=self.RUN_NUMBER, max_events = 10)
        assert tool_instance.max_events == self.MAX_EVENTS
        
    def test_init_with_events_per_slice(self):
        tool_instance = EventsLoopNectarCAMCalibrationTool(run_number=self.RUN_NUMBER, events_per_slice = 4)
        assert tool_instance.events_per_slice == self.EVENTS_PER_SLICE
        
    def test_init_with_run_file(self):
        tool_instance = EventsLoopNectarCAMCalibrationTool(run_number=self.RUN_NUMBER, run_file = self.RUN_FILE)
        assert tool_instance.run_file == self.RUN_FILE

    def test_load_eventsource(self, tool_instance_run_file):
        tool_instance_run_file._load_eventsource()
        assert isinstance(tool_instance_run_file.event_source, LightNectarCAMEventSource)
        assert tool_instance_run_file.event_source.input_url == self.RUN_FILE
    def test_load_eventsource_max_events(self, tool_instance_run_file):
        tool_instance_run_file.max_events = self.MAX_EVENTS
        tool_instance_run_file._load_eventsource()
        assert isinstance(tool_instance_run_file.event_source, LightNectarCAMEventSource)
        assert tool_instance_run_file.event_source.input_url == self.RUN_FILE
        assert tool_instance_run_file.event_source.max_events == self.MAX_EVENTS

    @patch("nectarchain.makers.core.HDF5TableWriter")
    @patch("nectarchain.makers.core.os.remove")
    @patch("nectarchain.makers.core.os.makedirs")
    def test_init_writer_full_mode(self, mock_makedirs, mock_remove, mock_writer, tool_instance):
        tool_instance.overwrite = True
        tool_instance._init_writer(sliced=False)
        mock_remove.assert_called_once_with(tool_instance.output_path)
        mock_makedirs.assert_called_once_with(tool_instance.output_path.parent, exist_ok=True)
        mock_writer.assert_called_once_with(
            filename=tool_instance.output_path,
            parent=tool_instance,
            mode="w",
            group_name="data",
        )
        
        
        
    @patch("nectarchain.makers.core.HDF5TableWriter")
    @patch("nectarchain.makers.core.os.remove")
    @patch("nectarchain.makers.core.os.makedirs")
    def test_init_writer_sliced_mode(self, mock_makedirs, mock_remove, mock_writer, tool_instance):
        tool_instance.overwrite = True
        tool_instance._init_writer(sliced=True, slice_index=1)
        mock_remove.assert_not_called()
        mock_makedirs.assert_called_once_with(tool_instance.output_path.parent, exist_ok=True)
        mock_writer.assert_called_once_with(
            filename=tool_instance.output_path,
            parent=tool_instance,
            mode="a",
            group_name="data_1",
        )
    @patch("nectarchain.makers.core.HDF5TableWriter")
    @patch("nectarchain.makers.core.os.remove")
    @patch("nectarchain.makers.core.os.makedirs")
    def test_init_writer_overwrite_false(self, mock_makedirs, mock_remove, mock_writer, tool_instance):
        tool_instance.overwrite = False
        with patch("nectarchain.makers.core.os.path.exists", return_value=True):
            with pytest.raises(Exception):
                tool_instance._init_writer(sliced=False)
        mock_remove.assert_not_called()
        mock_makedirs.assert_not_called()
        mock_writer.assert_not_called()
        
    def test_setup_eventsource(self, tool_instance_run_file):
        tool_instance_run_file._setup_eventsource()
        
        assert tool_instance_run_file._npixels == tool_instance_run_file.event_source.nectarcam_service.num_pixels
        assert np.all(tool_instance_run_file._pixels_id == tool_instance_run_file.event_source.nectarcam_service.pixel_ids)
        assert isinstance(tool_instance_run_file.event_source,LightNectarCAMEventSource)
        
        
    @patch("nectarchain.makers.core.ComponentUtils.get_class_name_from_ComponentName", return_value="ValidComponentClass")
    @patch("nectarchain.makers.core.ComponentUtils.get_configurable_traits", return_value={'trait1': 'value1'})
    def test_get_provided_component_kwargs(self,mock_get_class_name,mock_get_valid_component,tool_instance):
        tool_instance.trait1 = "value1"
        output_component_kwargs = tool_instance._get_provided_component_kwargs('componentName')
        assert output_component_kwargs == {"trait1": "value1"}
        
        
    @patch("nectarchain.makers.core.Component")
    @patch("nectarchain.makers.core.get_valid_component", return_value=["WaveformsComponent"])
    @patch("nectarchain.makers.core.ComponentUtils.get_class_name_from_ComponentName", return_value="WaveformsComponentClass")
    @patch("nectarchain.makers.core.ComponentUtils.get_configurable_traits", return_value={"trait1": "value1"})
    def test_setup_components(self, mock_get_configurable_traits, mock_get_class_name, mock_get_valid_component, mock_component, tool_instance_run_file):
        with pytest.raises(traitlets.traitlets.TraitError):
            tool_instance_run_file.componentsList = ["ValidComponent"]
        tool_instance_run_file.componentsList = ["WaveformsComponent"]
        tool_instance_run_file.trait1 = "value1"
        tool_instance_run_file._setup_eventsource()
        tool_instance_run_file._setup_components()
        mock_get_valid_component.assert_called_once()
        mock_get_class_name.assert_called_once_with("WaveformsComponent")
        mock_get_configurable_traits.assert_called_once_with("WaveformsComponentClass")
        mock_component.from_name.assert_called_once_with(
            "WaveformsComponent",
            subarray=tool_instance_run_file.event_source.subarray,
            parent=tool_instance_run_file,
            trait1="value1"
        )
        assert len(tool_instance_run_file.components) == 1
        assert tool_instance_run_file.components[0] == mock_component.from_name.return_value
        
        
    @patch("nectarchain.makers.core.os.remove")
    @patch("nectarchain.makers.core.EventsLoopNectarCAMCalibrationTool._setup_eventsource")
    @patch("nectarchain.makers.core.EventsLoopNectarCAMCalibrationTool._setup_components")
    @patch("nectarchain.makers.core.EventsLoopNectarCAMCalibrationTool._init_writer")
    def test_setup(self, mock_init_writer, mock_setup_components, mock_setup_eventsource, mock_remove, tool_instance_run_file):
        tool_instance_run_file.overwrite = True
        tool_instance_run_file.output_path = pathlib.Path("/tmp/test_output.h5")
        
        with patch("nectarchain.makers.core.pathlib.Path.exists", return_value=True):
            tool_instance_run_file.setup()
        
        mock_setup_eventsource.assert_called_once()
        mock_setup_components.assert_called_once()
        mock_remove.assert_called_once_with(tool_instance_run_file.output_path)
        mock_init_writer.assert_called_once_with(sliced=False, slice_index=1)
        assert tool_instance_run_file._n_traited_events == 0


    def test_setup_run_number_not_set(self, tool_instance):
        tool_instance.run_number = -1
        with pytest.raises(Exception, match="run_number need to be set up"):
            tool_instance.setup()
            
    def test_split_run(self,tool_instance):
        event = NectarCAMDataContainer()
        assert not(tool_instance.split_run(n_events_in_slice = 6,event=event))
        tool_instance.events_per_slice = 4
        assert tool_instance.split_run(n_events_in_slice = 6,event = event)
        assert not(tool_instance.split_run(n_events_in_slice =2,event = event))

    @patch("nectarchain.makers.core.Component")
    def test_start(self,mock_component,tool_instance_run_file):
        tool_instance_run_file.overwrite = True
        tool_instance_run_file.setup()
        n_events = len(tool_instance_run_file.event_source)
        tool_instance_run_file.components = [mock_component.from_name.return_value]
        tool_instance_run_file.start()
        assert tool_instance_run_file._n_traited_events == n_events
        
    @patch("nectarchain.makers.core.Component")
    def test_start_n_events(self,mock_component,tool_instance_run_file):
        tool_instance_run_file.overwrite = True
        tool_instance_run_file.setup()
        tool_instance_run_file.components = [mock_component.from_name.return_value]
        tool_instance_run_file.start(n_events = 10)
        assert tool_instance_run_file._n_traited_events == 10
        
    @patch("nectarchain.makers.core.Component")
    @patch("nectarchain.makers.core.EventsLoopNectarCAMCalibrationTool._finish_components")
    @patch("nectarchain.makers.core.EventsLoopNectarCAMCalibrationTool._setup_components")
    def test_start_sliced(self,mock_setup_components, mock_finish_components,mock_component,tool_instance_run_file):
        tool_instance_run_file.overwrite = True
        tool_instance_run_file.events_per_slice = self.EVENTS_PER_SLICE
        tool_instance_run_file.setup()
        n_events = len(tool_instance_run_file.event_source)
        tool_instance_run_file.components = [MockComponent()]
        tool_instance_run_file.start()
        assert mock_finish_components.call_count == n_events//self.EVENTS_PER_SLICE
        assert mock_setup_components.call_count == n_events//self.EVENTS_PER_SLICE + 1
    
    @patch("nectarchain.makers.core.Component")
    @patch("nectarchain.makers.core.EventsLoopNectarCAMCalibrationTool._finish_components")
    def test_finish(self, mock_finish_components, mock_component, tool_instance_run_file):
        tool_instance_run_file.overwrite = True
        tool_instance_run_file.setup()
        tool_instance_run_file.components = [mock_component.from_name.return_value]
        mock_finish_components.return_value = ["output"]
        
        output = tool_instance_run_file.finish()
        
        mock_finish_components.assert_called_once()
        assert output is None
        assert tool_instance_run_file.writer.h5file.isopen == 0
    @patch("nectarchain.makers.core.Component")
    @patch("nectarchain.makers.core.EventsLoopNectarCAMCalibrationTool._finish_components")
    def test_finish_with_output(self, mock_finish_components, mock_component, tool_instance_run_file):
        tool_instance_run_file.overwrite = True
        tool_instance_run_file.setup()
        tool_instance_run_file.components = [mock_component.from_name.return_value]
        
        output = tool_instance_run_file.finish(return_output_component=True)
        
        mock_finish_components.assert_called_once()
        assert output is not None
        assert tool_instance_run_file.writer.h5file.isopen == 0
    @patch("nectarchain.makers.core.EventsLoopNectarCAMCalibrationTool._finish_components")
    def test_finish_components(self, mock_finish_components, tool_instance_run_file):
        tool_instance_run_file.overwrite = True
        tool_instance_run_file.setup()
        tool_instance_run_file.components = [MockComponent()]
        
        output = tool_instance_run_file._finish_components()
        
        assert mock_finish_components.called_with([MockComponent().finish()], 0)

    @patch("nectarchain.makers.core.HDF5TableWriter")
    def test_write_container_with_nectarcam_container(self, mock_writer, tool_instance_run_file):
        tool_instance_run_file.writer = mock_writer
        container = MagicMock(spec=NectarCAMContainer)
        container.validate = MagicMock()
        tool_instance_run_file._write_container(container, index_component=0)
        container.validate.assert_called_once()
        mock_writer.write.assert_called_once_with(
            table_name=f"{container.__class__.__name__}_0",
            containers=container,
        )
        
    @patch("nectarchain.makers.core.HDF5TableWriter")
    def test_write_container_with_triggermap_container(self, mock_writer, tool_instance_run_file):
        tool_instance_run_file.writer = mock_writer
        container = MagicMock(spec=TriggerMapContainer)
        container.validate = MagicMock()
        container.containers = {
            EventType.FLATFIELD: MagicMock(spec=Container),
            EventType.UNKNOWN: MagicMock(spec=Container),
        }
        tool_instance_run_file._write_container(container, index_component=0)
        container.validate.assert_called_once()
        mock_writer.write.assert_any_call(
            table_name=f"{container.containers[EventType.FLATFIELD].__class__.__name__}_0/{EventType.FLATFIELD.name}",
            containers=container.containers[EventType.FLATFIELD],
        )
        mock_writer.write.assert_any_call(
            table_name=f"{container.containers[EventType.UNKNOWN].__class__.__name__}_0/{EventType.UNKNOWN.name}",
            containers=container.containers[EventType.UNKNOWN],
        )
        
    @patch("nectarchain.makers.core.HDF5TableWriter")
    def test_write_container_with_invalid_container(self, mock_writer, tool_instance_run_file):
        tool_instance_run_file.writer = mock_writer
        container = MagicMock(spec=Container)
        container.validate = MagicMock()
        with pytest.raises(TypeError, match="component output must be an instance of TriggerMapContainer or NectarCAMContainer"):
            tool_instance_run_file._write_container(container, index_component=0)
        container.validate.assert_called_once()
        mock_writer.write.assert_not_called()
        
            
    @patch("nectarchain.makers.core.HDF5TableWriter")
    def test_write_container_with_generic_exception(self, mock_writer, tool_instance_run_file):
        tool_instance_run_file.writer = mock_writer
        container = MagicMock(spec=NectarCAMContainer)
        container.validate = MagicMock(side_effect=Exception("Generic error"))
        with patch.object(tool_instance_run_file.log, 'error') as mock_log_error:
            with pytest.raises(Exception, match="Generic error"):
                tool_instance_run_file._write_container(container, index_component=0)
            container.validate.assert_called_once()
            mock_writer.write.assert_not_called()
            mock_log_error.assert_called_with("Generic error", exc_info=True)
            
    

class TestDelimiterLoopNectarCAMCalibrationTool:
    def test_split_run(self) :
        tool = DelimiterLoopNectarCAMCalibrationTool()
        event = NectarCAMDataContainer(trigger = TriggerContainer(event_type = EventType.FLATFIELD))
        assert not(tool.split_run(event = event))
        event = NectarCAMDataContainer(trigger = TriggerContainer(event_type = EventType.UNKNOWN))
        assert tool.split_run(event = event)


