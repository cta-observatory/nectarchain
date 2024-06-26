import logging
from abc import ABC

from ctapipe.visualization import CameraDisplay
from matplotlib import pyplot as plt

from ..data.container import ArrayDataContainer, ChargesContainer, WaveformsContainer

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import numpy as np


class ContainerDisplay(ABC):
    @staticmethod
    def display(container: ArrayDataContainer, evt, geometry, cmap="gnuplot2"):
        """plot camera display for HIGH GAIN channel

        Args:
            evt (int): event index
            cmap (str, optional): colormap. Defaults to 'gnuplot2'.

        Returns:
            CameraDisplay: thoe cameraDisplay plot
        """
        if isinstance(container, ChargesContainer):
            image = container.charges_hg
            pixels_id = container.pixels_id
        elif isinstance(container, WaveformsContainer):
            image = container.wfs_hg.mean(axis=2)
            pixels_id = container.pixels_id
        else:
            log.error(
                "container can't be displayed, must be a ChargesContainer or a WaveformsContainer"
            )
            raise Exception(
                "container can't be displayed, must be a ChargesContainer or a WaveformsContainer"
            )

        highlighten_pixels = np.array([], dtype=int)
        if geometry.pix_id.value.shape[0] != image.shape[1]:
            mask = np.array([_id in pixels_id for _id in geometry.pix_id.value])
            missing_pixels = np.array(geometry.pix_id.value[~mask], dtype=int)

            missing_values = np.empty((image.shape[0], missing_pixels.shape[0]))
            missing_values.fill(np.nan)
            highlighten_pixels = np.concatenate((highlighten_pixels, missing_pixels))

            image = np.concatenate((missing_values, image), axis=1)
            pixels_id = np.concatenate((missing_pixels, pixels_id))
        sort_index = [np.where(pixels_id == pix)[0][0] for pix in geometry.pix_id.value]
        image = image.T[sort_index].T

        disp = CameraDisplay(geometry=geometry, image=image[evt], cmap=cmap)
        disp.highlight_pixels(highlighten_pixels, color="r", linewidth=2)
        disp.add_colorbar(label="ADC")
        return {"disp": disp, "highlighten_pixels": highlighten_pixels}

    @staticmethod
    def plot_waveform(waveformsContainer: WaveformsContainer, evt, **kwargs):
        """plot the waveform of the evt in the HIGH GAIN channel

        Args:
            evt (int): the event index

        Returns:
            tuple: the figure and axes
        """
        if "figure" in kwargs.keys() and "ax" in kwargs.keys():
            fig = kwargs.get("figure")
            ax = kwargs.get("ax")
        else:
            fig, ax = plt.subplots(1, 1)
        ax.plot(waveformsContainer.wfs_hg[evt].T)
        return fig, ax
