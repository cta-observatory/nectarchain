import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

from ctapipe.containers import DL1CameraContainer


class CtapipeExtractor:
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
    
    def get_extractor_kwargs_str(extractor_kwargs) : 
        if len(extractor_kwargs) == 0 :
            str_extractor_kwargs = ""
        else :
            extractor_kwargs_list = [f"{key}_{value}" for key,value in extractor_kwargs.items()]
            str_extractor_kwargs = extractor_kwargs_list[0]
            for item in extractor_kwargs_list[1:] : 
                str_extractor_kwargs += f"_{item}"
        return str_extractor_kwargs
