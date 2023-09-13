import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from ctapipe.containers import DL1CameraContainer 

class CtapipeExtractor():
    """
    A class to extract the image and peak time from a DL1CameraContainer object.
    """

    def get_image_peak_time(cameraContainer):
        """
        Extracts the image and peak time from a DL1CameraContainer object.

        Parameters:
        cameraContainer (DL1CameraContainer): The DL1CameraContainer object to extract the image and peak time from.

        Returns:
        tuple: A tuple containing the image and peak time values from the container.
        """
        return cameraContainer.image, cameraContainer.peak_time
