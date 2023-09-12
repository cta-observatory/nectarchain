import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from ctapipe.containers import DL1CameraContainer 

class CtapipeExtractor():
    def get_image_peak_time(cameraContainer : DL1CameraContainer) : 
        return cameraContainer.image, cameraContainer.peak_time
