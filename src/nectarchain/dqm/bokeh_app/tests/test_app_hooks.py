import numpy as np

# bokeh imports
from bokeh.io import output_file, save
from bokeh.layouts import column, gridplot, row
from bokeh.models import Select, TabPanel, Tabs
from bokeh.plotting import curdoc

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
    displays = make_camera_displays(source=source, runid=runid)
    timelines = make_timelines(source, runid)

    ncols = 3
    camera_displays = [
        displays[parentkey][childkey].figure
        for parentkey in displays.keys()
        for childkey in displays[parentkey].keys()
    ]
    list_timelines = [
        timelines[parentkey][childkey]
        for parentkey in timelines.keys()
        for childkey in timelines[parentkey].keys()
    ]

    layout_camera_displays = gridplot(
        camera_displays,
        ncols=ncols,
    )

    layout_timelines = gridplot(
        list_timelines,
        ncols=2,
    )
    # Create different tabs
    tab_camera_displays = TabPanel(
        child=layout_camera_displays, title="Camera displays"
    )
    tab_timelines = TabPanel(child=layout_timelines, title="Timelines")

    # Combine panels into tabs
    tabs = Tabs(
        tabs=[tab_camera_displays, tab_timelines],
    )

    controls = row(run_select)

    page_layout = column([controls, tabs], sizing_mode="scale_width")

    curdoc().add_root(page_layout)
    curdoc().title = "NectarCAM Data Quality Monitoring web app"

    output_path = tmp_path / "test.html"
    output_file(output_path)
    save(curdoc(), filename=output_path)
