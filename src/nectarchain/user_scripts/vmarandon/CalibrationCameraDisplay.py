try:
    import numpy as np
    from ctapipe.visualization import CameraDisplay
except ImportError as err:
    print(err)
    raise SystemExit


class CalibrationCameraDisplay(CameraDisplay):
    class PixelsEdgesDisplayInfos:
        def __init__(self, nPixels):
            self.linewidth = np.zeros(nPixels)
            self.alpha = np.zeros(nPixels)
            self.edgecolor = np.full(nPixels, fill_value="black", dtype=object)

        @property
        def shape(self):
            return self.linewidth.shape

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clickfunc = None
        try:
            n_pixels = self.geom.n_pixels
        except AttributeError as err:
            n_pixels = (
                1855  # Fallback to this one as this class is likely for NectarCAM
            )
        self.pixel_edges_infos = CalibrationCameraDisplay.PixelsEdgesDisplayInfos(
            n_pixels
        )

    def highlight_pixels(self, pixels, color="g", linewidth=1, alpha=0.75):
        """
        Highlight the given pixels with a colored line around them.
        Can highlight different pixels with different colors

        Parameters
        ----------
        pixels : index-like
            The pixels to highlight.
            Can either be a list or array of integers or a
            boolean mask of length number of pixels
        color: a matplotlib conform color
            the color for the pixel highlighting
        linewidth: float
            linewidth of the highlighting in points
        alpha: 0 <= alpha <= 1
            The transparency
        """
        if self.image.shape != self.pixel_edges_infos.shape:
            print(
                "WARNING> Inconsistent shape between internal and given image --> all previous highlights will be lost"
            )
            self.pixel_edges_infos = CalibrationCameraDisplay.PixelsEdgesDisplayInfos(
                self.image.shape[0]
            )

        self.pixel_edges_infos.linewidth[pixels] = linewidth
        self.pixel_edges_infos.alpha[pixels] = alpha
        self.pixel_edges_infos.edgecolor[pixels] = color

        self.pixel_highlighting.set_linewidth(self.pixel_edges_infos.linewidth)
        self.pixel_highlighting.set_alpha(self.pixel_edges_infos.alpha)
        self.pixel_highlighting.set_edgecolor(self.pixel_edges_infos.edgecolor)

        self._update()

    def set_function(self, func_name):
        self.clickfunc = func_name

    def on_pixel_clicked(self, pix_id):
        if self.clickfunc is not None:
            self.clickfunc(pix_id)
