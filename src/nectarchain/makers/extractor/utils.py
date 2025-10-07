import importlib
import logging

from ctapipe.containers import DL1CameraContainer

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class CtapipeExtractor:
    """
    A class to extract the image and peak time from a DL1CameraContainer object.
    """

    def get_image_peak_time(cameraContainer: DL1CameraContainer):
        """
        Extracts the image and peak time from a DL1CameraContainer object.

        Parameters:
        cameraContainer (DL1CameraContainer): The DL1CameraContainer object to extract
        the image and peak time from.

        Returns:
        tuple: A tuple containing the image and peak time values from the container.
        """
        return cameraContainer.image, cameraContainer.peak_time

    def get_extractor_kwargs_str(method: str, extractor_kwargs: dict):
        ctapipe_extractor_module = importlib.import_module("ctapipe.image.extractor")
        extractor = getattr(ctapipe_extractor_module, method)
        str_extractor_kwargs = ""
        for trait_name, trait in extractor.class_own_traits().items():
            if trait_name in extractor_kwargs:
                if trait.default()[0][2] != extractor_kwargs[trait_name]:
                    str_extractor_kwargs += (
                        f"{trait_name}_{extractor_kwargs[trait_name]}_"
                    )
        if len(str_extractor_kwargs) > 0:
            str_extractor_kwargs = f"{str_extractor_kwargs[:-1]}"
        else:
            str_extractor_kwargs = "default"
        return str_extractor_kwargs
