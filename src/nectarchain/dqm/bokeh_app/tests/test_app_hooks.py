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

    for runid in list(test_dict.keys()):
        make_camera_displays(source=test_dict[runid], runid=runid)


def test_get_run_times():
    from datetime import datetime

    from nectarchain.dqm.bokeh_app.app_hooks import get_run_times

    # Create a test source dict with START-TIMES data
    # Using np.array to simulate the structure expected by the function
    run_start_time_ts = 1609459200  # 2021-01-01 00:00:00 UTC
    first_event_time_ts = 1609459260  # 2021-01-01 00:01:00 UTC
    last_event_time_ts = 1609462800  # 2021-01-01 01:00:00 UTC

    source_with_times = {
        "START-TIMES": {
            "Run start time": np.array([run_start_time_ts]),
            "First event": np.array([first_event_time_ts]),
            "Last event": np.array([last_event_time_ts]),
        }
    }

    run_start_time_dt, first_event_time_dt, last_event_time_dt = get_run_times(
        source_with_times
    )

    # Verify the returned strings are in the correct format
    assert isinstance(run_start_time_dt, str)
    assert isinstance(first_event_time_dt, str)
    assert isinstance(last_event_time_dt, str)

    # Verify the format is YYYY-MM-DD HH:MM:SS
    expected_format = "%Y-%m-%d %H:%M:%S"
    datetime.strptime(run_start_time_dt, expected_format)
    datetime.strptime(first_event_time_dt, expected_format)
    datetime.strptime(last_event_time_dt, expected_format)

    # Verify the values match the input timestamps
    assert run_start_time_dt == "2021-01-01 00:00:00"
    assert first_event_time_dt == "2021-01-01 00:01:00"
    assert last_event_time_dt == "2021-01-01 01:00:00"


def test_make_timelines():
    from nectarchain.dqm.bokeh_app.app_hooks import make_timelines

    for runid in list(test_dict.keys()):
        make_timelines(source=test_dict[runid], runid=runid)


def test_make_waveforms():
    from nectarchain.dqm.bokeh_app.app_hooks import make_waveforms

    for runid in list(test_dict.keys()):
        make_waveforms(source=test_dict[runid], runid=runid)


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
        make_waveforms,
    )

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
    displays = make_camera_displays(source, runid)
    timelines = make_timelines(source, runid)
    waveforms = make_waveforms(source, runid)

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

    # Create different tabs
    tab_camera_displays = TabPanel(
        child=layout_camera_displays, title="Camera displays"
    )
    tab_timelines = TabPanel(child=layout_timelines, title="Timelines")

    tab_waveforms = TabPanel(child=layout_waveforms, title="Waveforms")

    # Combine panels into tabs
    tabs = Tabs(
        tabs=[tab_camera_displays, tab_timelines, tab_waveforms],
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
