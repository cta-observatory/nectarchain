def get_allowed_cameras():
    prefix = "NectarCAM"
    allowed_cameras = [f"{prefix}" + "QM"]
    allowed_cameras.extend([f"{prefix + str(i)}" for i in range(2, 10)])
    return allowed_cameras


ALLOWED_CAMERAS = get_allowed_cameras()

GAIN_DEFAULT = 58.0
HILO_DEFAULT = 13.0
