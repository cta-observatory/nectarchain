import numpy as np

# bokeh imports
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, Select, TabPanel, Tabs
from bokeh.plotting import curdoc, figure

# ctapipe imports
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ZODB import DB

geom = CameraGeometry.from_name("NectarCam-003")
geom = geom.transform_to(EngineeringCameraFrame())

test_dict = {
    "run1": {
        "START-TIMES": {
            "Run start time": np.array([1609459200]),  # 2021-01-01 00:00:00 UTC
            "First event": np.array([1609459260]),  # 2021-01-01 00:01:00 UTC
            "Last event": np.array([1609462800]),  # 2021-01-01 01:00:00 UTC
        },
        "mykey1": {
            "mysubkey1": np.random.normal(size=geom.n_pixels),
            "mysubkey2": np.random.normal(size=geom.n_pixels),
            "FOOPIXTIMELINE-HIGH": np.random.normal(size=1000),
        },
        "mykey2": {
            "mysubkey1": np.random.normal(size=geom.n_pixels),
            "mysubkey2": np.random.normal(size=geom.n_pixels),
            "FOOPIXTIMELINE-HIGH": np.random.normal(size=1000),
        },
        # BADPIX keys for testing
        "CAMERA-BADPIX-PED-PHY-OVEREVENTS-HIGH-GAIN": {
            "CAMERA-BadPix-PED-PHY-OverEVENTS-HIGH-GAIN": np.array(
                [0, 1, 2] + [0] * (geom.n_pixels - 3)
            ),
        },
        "CAMERA-BADPIX-PHY-OVEREVENTS-HIGH-GAIN": {
            "CAMERA-BadPix-PHY-OverEVENTS-HIGH-GAIN": np.array(
                [0, 1, 2] + [0] * (geom.n_pixels - 3)
            ),
        },
        # TODO: these two entries may actually need update
        # considering the conversion to UTC time
        # in the dedicated maker function
        "TRIGGER-EVENTS-PHY": {
            "Timestamps": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "IDs": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        },
        "TRIGGER-EVENTS-PED": {
            "Timestamps": np.array([2, 1, 3, 4, 5, 7, 6, 8, 9, 10]),
            "IDs": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        },
    }
}
# Renders the second image incomplete
test_dict["run1"]["mykey2"]["mysubkey2"][10:20] = np.nan


def test_set_bad_pixels_cap_value():
    from nectarchain.dqm.bokeh_app.app_hooks import set_bad_pixels_cap_value

    arr = np.array([0.0, 1.2, 0.0, 2.0])
    capped = set_bad_pixels_cap_value(arr.copy())
    assert np.all(capped <= 1.0)
    assert capped[1] == 1.0
    assert capped[3] == 1.0


def test_get_bad_pixels_position():
    from nectarchain.dqm.bokeh_app.app_hooks import get_bad_pixels_position

    source = test_dict["run1"]
    image_shape = (geom.n_pixels,)
    mask_high, mask_low = get_bad_pixels_position(source, image_shape)
    assert mask_high.shape == image_shape
    assert mask_low.shape == image_shape
    assert mask_high[1]
    assert mask_high[2]
    assert not mask_high[0]
    assert mask_low[1]
    assert mask_low[2]
    assert not mask_low[0]


def test_make_camera_displays():
    from nectarchain.dqm.bokeh_app.app_hooks import make_camera_displays
    from nectarchain.dqm.bokeh_app.extract_data import categorize_source_data

    for runid in list(test_dict.keys()):
        categorized = categorize_source_data(test_dict[runid])
        make_camera_displays(
            camera_displays_data=categorized["camera_displays"], runid=runid
        )


def test_make_timelines():
    from nectarchain.dqm.bokeh_app.app_hooks import make_timelines
    from nectarchain.dqm.bokeh_app.extract_data import categorize_source_data

    for runid in list(test_dict.keys()):
        categorized = categorize_source_data(test_dict[runid])
        make_timelines(timelines_data=categorized["timelines"], runid=runid)


def test_make_waveforms():
    from nectarchain.dqm.bokeh_app.app_hooks import make_waveforms
    from nectarchain.dqm.bokeh_app.extract_data import categorize_source_data

    for runid in list(test_dict.keys()):
        categorized = categorize_source_data(test_dict[runid])
        make_waveforms(waveforms_data=categorized["waveforms"], runid=runid)


def test_make_trigger_timestamps_vs_ids():
    from nectarchain.dqm.bokeh_app.app_hooks import make_trigger_timestamps_vs_ids
    from nectarchain.dqm.bokeh_app.extract_data import categorize_source_data

    for runid in list(test_dict.keys()):
        categorized = categorize_source_data(test_dict[runid])
        make_trigger_timestamps_vs_ids(
            trigger_events_data=categorized["trigger_events"], runid=runid
        )


def test_get_run_ids_for_camera():
    from nectarchain.dqm.bokeh_app.app_hooks import get_run_ids_for_camera

    db = DB(None)
    conn = db.open()
    root = conn.root()

    # Test data with multiple cameras and runs
    test_keys = [
        "NectarCAM1_Run1000",
        "NectarCAM1_Run1001",
        "NectarCAM2_Run1000",
        "NectarCAM2_Run1002",
        "NectarCAM3_Run1003",
    ]

    for key in test_keys:
        root[key] = {"data": "dummy"}

    # Test extracting run ids for camera 1
    run_ids_cam01 = get_run_ids_for_camera(root, "1")
    assert len(run_ids_cam01) == 2
    assert "NectarCAM1_Run1000" in run_ids_cam01
    assert "NectarCAM1_Run1001" in run_ids_cam01

    # Test extracting run ids for camera 2
    run_ids_cam02 = get_run_ids_for_camera(root, "2")
    assert len(run_ids_cam02) == 2
    assert "NectarCAM2_Run1000" in run_ids_cam02
    assert "NectarCAM2_Run1002" in run_ids_cam02

    # Test extracting run ids for camera 3
    run_ids_cam03 = get_run_ids_for_camera(root, "3")
    assert len(run_ids_cam03) == 1
    assert "NectarCAM3_Run1003" in run_ids_cam03

    # Test with camera code that has no matches
    run_ids_cam99 = get_run_ids_for_camera(root, "99")
    assert len(run_ids_cam99) == 0
    assert isinstance(run_ids_cam99, list)


def test_get_available_cameras_from_db_keys():
    from nectarchain.dqm.bokeh_app.app_hooks import get_available_cameras_from_db_keys

    db = DB(None)
    conn = db.open()
    root = conn.root()

    # Test data with various key types
    test_keys = [
        "NectarCAM1_Run1000",
        "NectarCAM1_Run1001",
        "NectarCAM2_Run1000",
        "NectarCAM3_Run1003",
    ]

    for key in test_keys:
        root[key] = {"data": "dummy"}

    # Test that available cameras are correctly extracted
    available_cameras = get_available_cameras_from_db_keys(root)

    # Should return a set
    assert isinstance(available_cameras, set)

    # Should contain only the cameras, not the filtered keys
    assert "1" in available_cameras
    assert "2" in available_cameras
    assert "3" in available_cameras

    # Should have exactly 3 cameras
    assert len(available_cameras) == 3


def test_bokeh(tmp_path):
    from nectarchain.dqm.bokeh_app.app_hooks import (
        get_rundata,
        make_camera_displays,
        make_timelines,
        make_trigger_timestamps_vs_ids,
        make_waveforms,
    )
    from nectarchain.dqm.bokeh_app.extract_data import categorize_source_data

    db = DB(None)
    conn = db.open()
    root = conn.root()
    runids = sorted(list(test_dict.keys()))

    # Fill in-memory DB
    for runid in runids:
        root[runid] = test_dict[runid]

    runid = runids[-1]
    run_select = Select(value=runid, title="NectarCAM run number", options=runids)

    source = get_rundata(root, run_select.value)
    categorized = categorize_source_data(source)
    displays = make_camera_displays(categorized["camera_displays"], runid)
    timelines = make_timelines(categorized["timelines"], runid)
    waveforms = make_waveforms(categorized["waveforms"], runid)
    trig_timestamps = make_trigger_timestamps_vs_ids(
        categorized["trigger_events"], runid
    )

    camera_displays = [
        column(
            [
                displays[parentkey][childkey][1],  # RangeSlider
                row(
                    displays[parentkey][childkey][0].figure,
                    displays[parentkey][childkey][2],
                    displays[parentkey][childkey][3],
                ),
            ]
        )
        if len(displays[parentkey][childkey]) == 4
        else displays[parentkey][childkey][0].figure
        for parentkey in displays.keys()
        for childkey in displays[parentkey].keys()
    ]

    list_timelines = [
        timelines[parentkey][childkey]
        for parentkey in timelines.keys()
        for childkey in timelines[parentkey].keys()
    ]

    list_waveforms = [
        waveforms[parentkey][childkey]
        for parentkey in waveforms.keys()
        for childkey in waveforms[parentkey].keys()
    ]

    list_trig_timestamps = [
        trig_timestamps[parentkey] for parentkey in trig_timestamps.keys()
    ]

    layout_camera_displays = column(
        camera_displays,
        sizing_mode="scale_width",
    )

    layout_timelines = column(
        list_timelines,
        sizing_mode="scale_width",
    )

    layout_waveforms = column(
        list_waveforms,
        sizing_mode="scale_width",
    )

    layout_trig_timestamps = column(
        list_trig_timestamps,
        sizing_mode="scale_width",
    )

    # Create different tabs
    tab_camera_displays = TabPanel(
        child=layout_camera_displays, title="Camera displays"
    )
    tab_timelines = TabPanel(child=layout_timelines, title="Timelines")

    tab_waveforms = TabPanel(child=layout_waveforms, title="Waveforms")

    tab_trig_timestamps = TabPanel(
        child=layout_trig_timestamps, title="Trigger Timestamps"
    )

    # Combine panels into tabs
    tabs = Tabs(
        tabs=[tab_camera_displays, tab_timelines, tab_waveforms, tab_trig_timestamps],
        sizing_mode="scale_width",
    )

    controls = row(run_select)

    page_layout = column([controls, tabs], sizing_mode="scale_width")

    curdoc().add_root(page_layout)
    curdoc().title = "NectarCAM Data Quality Monitoring web app"

    output_path = tmp_path / "test.html"
    output_file(output_path)
    save(curdoc(), filename=output_path)


def test_compile_hover_tool():
    from ctapipe.visualization.bokeh import CameraDisplay

    from nectarchain.dqm.bokeh_app.app_hooks import compile_hover_tool

    display = CameraDisplay(geometry=geom)
    display.image = np.random.normal(size=geom.n_pixels)

    result = compile_hover_tool(display, geom)

    # Verify the function returns a display object
    assert isinstance(result, CameraDisplay)

    # Verify datasource was populated with expected columns
    datasource_data = result.datasource.data
    expected_keys = [
        "pix_id",
        "pix_x",
        "pix_y",
        "cluster_n",
        "pix_id_in_cluster",
        "image",
    ]
    for key in expected_keys:
        assert key in datasource_data, f"Expected key '{key}' not found in datasource"

    # Verify data shapes and content
    assert len(datasource_data["pix_id"]) == geom.n_pixels
    assert len(datasource_data["pix_x"]) == geom.n_pixels
    assert len(datasource_data["pix_y"]) == geom.n_pixels
    assert len(datasource_data["cluster_n"]) == geom.n_pixels
    assert len(datasource_data["pix_id_in_cluster"]) == geom.n_pixels
    assert len(datasource_data["image"]) == geom.n_pixels

    # Verify HoverTool was added to the figure
    hover_tools = [tool for tool in result.figure.tools if isinstance(tool, HoverTool)]
    assert len(hover_tools) > 0, "No HoverTool found in figure"

    # Find the custom HoverTool we added
    custom_hover_tool = None
    for tool in hover_tools:
        if len(tool.tooltips) == 6:  # Our custom HoverTool has 6 tooltips
            custom_hover_tool = tool
            break

    assert (
        custom_hover_tool is not None
    ), "Custom HoverTool with expected tooltips not found"

    # Verify HoverTool has the expected tooltips
    tooltip_fields = [tooltip[0] for tooltip in custom_hover_tool.tooltips]
    expected_tooltip_fields = [
        "pix id",
        "pix # in cluster",
        "cluster #",
        "pix x pos",
        "pix y pos",
        "value",
    ]
    assert tooltip_fields == expected_tooltip_fields


def test_compile_hover_tool_val_vs_id():
    from nectarchain.dqm.bokeh_app.app_hooks import compile_hover_tool_val_vs_id

    fig = figure()
    data_source = ColumnDataSource(
        data=dict(pix_id=np.arange(10), value=np.random.normal(size=10))
    )
    scatter = fig.scatter(x="pix_id", y="value", source=data_source)

    result = compile_hover_tool_val_vs_id(pixel_data=scatter, figure=fig)

    assert result is fig

    # Verify HoverTool was added to the figure
    hover_tools = [tool for tool in result.tools if isinstance(tool, HoverTool)]
    assert len(hover_tools) > 0, "No HoverTool found in figure"

    # Find the custom HoverTool we added
    custom_hover_tool = None
    for tool in hover_tools:
        if len(tool.tooltips) == 1:
            custom_hover_tool = tool
            break

    assert (
        custom_hover_tool is not None
    ), "Custom HoverTool with expected tooltip not found"

    # Verify the tooltip is correct
    tooltip_label = custom_hover_tool.tooltips[0][0]
    tooltip_value = custom_hover_tool.tooltips[0][1]
    assert tooltip_label == "(pix_id, value)"
    assert tooltip_value == "(@pix_id, @value)"

    # Verify the scatter plot is in the renderers
    assert scatter in custom_hover_tool.renderers


def test_camera_display_initialization():
    """Test CameraDisplayNectarCAM class initialization"""
    from nectarchain.dqm.bokeh_app.app_hooks import CameraDisplayNectarCAM

    # Create a CameraDisplayNectarCAM instance
    display = CameraDisplayNectarCAM(geometry=geom)

    # Verify that it inherits from CameraDisplay
    from ctapipe.visualization.bokeh import CameraDisplay

    assert isinstance(display, CameraDisplay)

    # Verify that all custom attributes are initialized
    assert hasattr(display, "colorbar_label")
    assert hasattr(display, "figure_title")
    assert hasattr(display, "selected_pixel_waveform")
    assert hasattr(display, "camera_displays_data")
    assert hasattr(display, "parent_key")
    assert hasattr(display, "child_key")
    assert hasattr(display, "waveforms_data")

    # Verify initial values
    assert display.colorbar_label == ""
    assert display.figure_title == ""
    assert display.camera_displays_data is None
    assert display.parent_key is None
    assert display.child_key is None
    assert display.waveforms_data is None

    # Verify that selected_pixel_waveform is a bokeh row
    from bokeh.models.layouts import Row

    assert isinstance(display.selected_pixel_waveform, Row)


def test_camera_display_pixel_picker_callback_no_waveforms():
    """Test pixel_picker_callback when no waveforms_data is available"""
    from nectarchain.dqm.bokeh_app.app_hooks import CameraDisplayNectarCAM

    # Create a CameraDisplayNectarCAM instance
    display = CameraDisplayNectarCAM(geometry=geom)
    display.image = np.random.normal(size=geom.n_pixels)
    display.figure_title = "Test Display"
    display.colorbar_label = "Test Units"

    # Set waveforms_data to None to test the no data case
    display.waveforms_data = None

    # Mock the logger to capture the warning message
    from unittest.mock import patch

    with patch("nectarchain.dqm.bokeh_app.app_hooks.logger") as mock_logger:
        # Call the pixel picker callback with some pixel indices
        display.pixel_picker_callback("indices", [], [0, 1, 2])

        # Verify that warning was logged
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0]
        assert "No waveform data found to add to the display" in warning_call[0]


def test_camera_display_pixel_picker_callback_with_waveforms():
    """Test pixel_picker_callback with waveforms_data available"""
    from bokeh.plotting import figure

    from nectarchain.dqm.bokeh_app.app_hooks import CameraDisplayNectarCAM

    # Create a CameraDisplayNectarCAM instance
    display = CameraDisplayNectarCAM(geometry=geom)
    display.image = np.random.normal(size=geom.n_pixels)
    display.figure_title = "Test Display"
    display.colorbar_label = "Test Units"

    # Create mock waveforms data
    n_samples = 100
    n_pixels = geom.n_pixels

    # Create average waveforms
    avg_waveforms_dict = {
        "WF-PHY-AVERAGE-PIX-HIGH-GAIN": {
            "WF-PHY-AVERAGE-PIX-HIGH-GAIN": np.random.normal(size=n_samples)
        }
    }

    # Create all waveforms
    all_waveforms_dict = {
        "WF-PHY-HIGH-GAIN": {
            "WF-PHY-HIGH-GAIN": np.random.normal(size=(n_pixels, n_samples))
        }
    }

    display.waveforms_data = {"average": avg_waveforms_dict, "all": all_waveforms_dict}

    # Mock the labels.json file for y-axis labels
    import json
    from unittest.mock import mock_open, patch

    mock_labels_data = {
        "y_axis_labels_waveforms": {"WF-PHY-AVERAGE-PIX-HIGH-GAIN": "Test Label"}
    }

    # Mock the logger to capture info messages
    with patch("nectarchain.dqm.bokeh_app.app_hooks.logger") as mock_logger, patch(
        "builtins.open", mock_open(read_data=json.dumps(mock_labels_data))
    ):
        # Call the pixel picker callback with a pixel index
        display.pixel_picker_callback("indices", [], [0])

        # Verify that info messages were logged
        assert (
            mock_logger.info.call_count >= 2
        )  # At least one for pixel value, one for adding to display

        # Check the selected_pixel_waveform children were populated
        assert len(display.selected_pixel_waveform.children) > 0

        # Verify that the first child is a figure
        first_child = display.selected_pixel_waveform.children[0]
        assert isinstance(first_child, figure)

        # Verify the figure has the expected properties
        assert "selected pixel 0" in first_child.title.text


def test_camera_display_pixel_picker_callback_mixed_waveforms():
    """Test pixel_picker_callback with mixed PHY and PED waveforms"""
    from nectarchain.dqm.bokeh_app.app_hooks import CameraDisplayNectarCAM

    # Create a CameraDisplayNectarCAM instance
    display = CameraDisplayNectarCAM(geometry=geom)
    display.image = np.random.normal(size=geom.n_pixels)
    display.figure_title = "Test Display"
    display.colorbar_label = "Test Units"

    # Create mock waveforms data with both PHY and PED
    n_samples = 50
    n_pixels = geom.n_pixels

    avg_waveforms_dict = {
        "WF-PHY-AVERAGE-PIX-HIGH-GAIN": {
            "WF-PHY-AVERAGE-PIX-HIGH-GAIN": np.random.normal(size=n_samples)
        },
        "WF-PED-AVERAGE-PIX-HIGH-GAIN": {
            "WF-PED-AVERAGE-PIX-HIGH-GAIN": np.random.normal(size=n_samples)
        },
    }

    all_waveforms_dict = {
        "WF-PHY-HIGH-GAIN": {
            "WF-PHY-HIGH-GAIN": np.random.normal(size=(n_pixels, n_samples))
        },
        "WF-PED-HIGH-GAIN": {
            "WF-PED-HIGH-GAIN": np.random.normal(size=(n_pixels, n_samples))
        },
    }

    display.waveforms_data = {"average": avg_waveforms_dict, "all": all_waveforms_dict}

    # Mock the labels.json file
    import json
    from unittest.mock import mock_open, patch

    mock_labels_data = {
        "y_axis_labels_waveforms": {
            "WF-PHY-AVERAGE-PIX-HIGH-GAIN": "Physics Label",
            "WF-PED-AVERAGE-PIX-HIGH-GAIN": "Pedestal Label",
        }
    }

    with patch("nectarchain.dqm.bokeh_app.app_hooks.logger"), patch(
        "builtins.open", mock_open(read_data=json.dumps(mock_labels_data))
    ):
        # Call the pixel picker callback
        display.pixel_picker_callback("indices", [], [5])

        # Check that waveforms for both PHY and PED were created
        assert len(display.selected_pixel_waveform.children) == 2

        # Verify the titles contain the expected keys
        titles = [
            child.title.text for child in display.selected_pixel_waveform.children
        ]
        assert any("WF-PHY" in title for title in titles)
        assert any("WF-PED" in title for title in titles)
