def get_allowed_cameras():
    prefix = "NectarCAM"
    allowed_cameras = [f"{prefix}" + "QM"]
    allowed_cameras.extend([f"{prefix + str(i)}" for i in range(2, 10)])
    return allowed_cameras


ALLOWED_CAMERAS = get_allowed_cameras()

PEDESTAL_DEFAULT = 250.0
GAIN_DEFAULT = 58.0
HILO_DEFAULT = 13.0
FLATFIELD_DEFAULT = 1.0

GAIN_LINEAR_RANGE = [10, 200]  # p.e.

GROUP_NAMES_PEDESTAL = ["data", "data_combined"]
