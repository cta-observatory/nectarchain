import logging
from abc import ABC

from ctapipe.visualization import CameraDisplay
from matplotlib import pyplot as plt

from ..data.container import ArrayDataContainer, ChargesContainer, WaveformsContainer

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


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
        elif isinstance(container, WaveformsContainer):
            image = container.wfs_hg.sum(axis=2)
        else:
            log.warning("container can't be displayed")
        disp = CameraDisplay(geometry=geometry, image=image[evt], cmap=cmap)
        disp.add_colorbar()
        return disp

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
