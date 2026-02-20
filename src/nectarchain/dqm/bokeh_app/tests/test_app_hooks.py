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
    assert mask_high[1] == True
    assert mask_high[2] == True
    assert mask_high[0] == False
    assert mask_low[1] == True
    assert mask_low[2] == True
    assert mask_low[0] == False


def test_make_camera_displays():
    from nectarchain.dqm.bokeh_app.app_hooks import make_camera_displays

    for runid in list(test_dict.keys()):
        make_camera_displays(source=test_dict[runid], runid=runid)


def test_make_timelines():
    from nectarchain.dqm.bokeh_app.app_hooks import make_timelines

    for runid in list(test_dict.keys()):
        make_timelines(source=test_dict[runid], runid=runid)


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
