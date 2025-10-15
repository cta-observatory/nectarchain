from ..data.container import GainContainer, SPEfitContainer
from ..makers.calibration import flatfield_makers, gain, pedestal_makers

PEDESTAL_CALIBRATION_TOOLS = {
    name: getattr(pedestal_makers, name) for name in pedestal_makers.__all__
}
GAIN_CALIBRATION_TOOLS = {name: getattr(gain, name) for name in gain.__all__}
FLATFIELD_CALIBRATION_TOOLS = {
    name: getattr(flatfield_makers, name) for name in flatfield_makers.__all__
}

GAIN_CONTAINER_CLASSES = [GainContainer, SPEfitContainer]

GAIN_DEFAULT = 58.0
HILO_DEFAULT = 13.0
