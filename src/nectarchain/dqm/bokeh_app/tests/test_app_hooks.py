import numpy as np

# bokeh imports
from bokeh.io import output_file, save
from bokeh.layouts import layout
from bokeh.models import Select
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
        },
        "mykey2": {
            "mysubkey1": np.random.normal(size=geom.n_pixels),
            "mysubkey2": np.random.normal(size=geom.n_pixels),
        },
    }
}
# Renders the second image incomplete
test_dict["run1"]["mykey2"]["mysubkey2"][10:20] = np.nan


def test_make_camera_displays():
    from nectarchain.dqm.bokeh_app.app_hooks import make_camera_displays

    for runid in list(test_dict.keys()):
        make_camera_displays(test_dict, test_dict[runid], runid)


def test_bokeh(tmp_path):
    from nectarchain.dqm.bokeh_app.app_hooks import get_rundata, make_camera_displays

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
    displays = make_camera_displays(root, source, runid)

    ncols = 3
    plots = [
        displays[parentkey][childkey].figure
        for parentkey in displays.keys()
        for childkey in displays[parentkey].keys()
    ]
    curdoc().add_root(
        layout(
            [[[plots[x : x + ncols] for x in range(0, len(plots), ncols)]]],
            sizing_mode="scale_width",
        )
    )
    curdoc().title = "NectarCAM Data Quality Monitoring web app"

    output_path = tmp_path / "test.html"
    output_file(output_path)
    save(curdoc(), filename=output_path)
