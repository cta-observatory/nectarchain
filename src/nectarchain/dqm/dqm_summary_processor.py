import logging

import numpy as np
from astropy.io import fits
from astropy.table import Table

__all__ = ["DQMSummary"]

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class DQMSummary:
    """Base class for DQM summary processors.

    Provides the interface for per-run initialisation, per-event processing,
    result aggregation, plotting, and FITS serialisation of DQM outputs.
    """

    def __init__(self, r0=False):
        """Initialise the summary processor.

        Parameters
        ----------
        r0 : bool, optional
            Whether to use R0 waveforms (skip R1 corrections).
        """
        log.debug("Processor 0")
        self.Samp = None
        self.Pix = None
        self.r0 = r0
        self.tel_id = None

    def define_for_run(self, reader1):
        """Extract camera geometry from the first event of a run.

        Parameters
        ----------
        reader1 : ctapipe_io_nectarcam.LightNectarCAMEventSource
            Event source whose first event is used to retrieve the number
            of pixels and samples.

        Returns
        -------
        tuple of (int, int)
            Number of pixels and number of samples.
        """
        self.tel_id = reader1.subarray.tel_ids[0]

        # we just need to access the first event
        evt1 = next(iter(reader1))
        if self.r0:
            self.Samp = evt1.r0.tel[self.tel_id].waveform.shape[-1]
            self.Pix = evt1.r0.tel[self.tel_id].waveform.shape[-2]
        else:
            self.Samp = evt1.r1.tel[self.tel_id].waveform.shape[-1]
            self.Pix = evt1.r1.tel[self.tel_id].waveform.shape[-2]
        return self.Pix, self.Samp

    def configure_for_run(self):
        """Configure the processor for a new run.

        Subclasses may override this to perform additional setup.
        """
        log.debug("Processor 1")

    def process_event(self, evt, noped):
        """Process a single event.

        Parameters
        ----------
        evt : NectarCAMDataContainer
            The event to process.
        noped : bool
            Whether pedestal subtraction should be skipped.
        """
        log.debug("Processor 2")

    def finish_run(self, M, M_ped, counter_evt, counter_ped):
        """Finalise per-run accumulations.

        Parameters
        ----------
        M : np.ndarray
            Accumulated physics-event data.
        M_ped : np.ndarray
            Accumulated pedestal-event data.
        counter_evt : int
            Number of physics events processed.
        counter_ped : int
            Number of pedestal events processed.
        """
        log.debug("Processor 3")

    def get_results(self):
        """Return aggregated results as a dictionary.

        Returns
        -------
        dict
            Dictionary of result names to computed arrays or tables.
        """
        log.debug("Processor 4")

    def plot_results(
        self, name, fig_path, k, M, M_ped, Mean_M_overPix, Mean_M_ped_overPix
    ):
        """Generate and save summary plots.

        Parameters
        ----------
        name : str
            Base name for the plot files.
        fig_path : str
            Directory where plots should be saved.
        k : int
            Gain channel index.
        M : np.ndarray
            Physics-event accumulations.
        M_ped : np.ndarray
            Pedestal-event accumulations.
        Mean_M_overPix : np.ndarray
            Physics mean over pixels.
        Mean_M_ped_overPix : np.ndarray
            Pedestal mean over pixels.
        """
        log.debug("Processor 5")

    @staticmethod
    def _create_hdu(name, content):
        """Create a FITS HDU from a content object.

        Parameters
        ----------
        name : str
            Name for the HDU extension.
        content : np.ndarray or dict or Table or float
            Data to store.  Arrays with ``ndim <= 1`` are stored in a
            binary table; higher-dimensional arrays become image HDUs.
            Dicts and Tables are stored as binary tables.

        Returns
        -------
        fits.BinTableHDU or fits.ImageHDU
            The constructed HDU.
        """
        data = Table()

        if isinstance(content, np.ndarray):
            arr = np.asarray(content)
            # Ensure consistent dtype (convert to float64 or int64 as appropriate)
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float64)
            elif np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.int64)
            elif np.issubdtype(arr.dtype, np.object_):
                arr = arr.astype(np.float64)

            # for 1- and 0-dimensional arrays
            if arr.ndim <= 1:
                if arr.ndim == 0:
                    arr = arr.reshape(1)  # Convert scalar to 1D
                data[name] = arr
                hdu = fits.BinTableHDU(data)
            else:
                hdu = fits.ImageHDU(arr)

        # sometimes content can be a dict, as from trigger_statistics
        elif isinstance(content, dict):
            try:
                data[name] = content
            except Exception as e:
                log.warning(f"Caught {type(e).__name__}. Details: {e}")
                data = Table(content)

            hdu = fits.BinTableHDU(data)

        # Handle astropy Table objects directly
        elif isinstance(content, Table):
            # Content is already a Table, use it as the HDU data
            hdu = fits.BinTableHDU(content, name=name)

        # content can be a float, need to be recast to an array
        else:
            data = Table(np.array([content]))
            hdu = fits.BinTableHDU(data)

        hdu.name = name
        return hdu

    def write_all_results(self, path, DICT):
        """Write all DQM results to a multi-extension FITS file.

        Parameters
        ----------
        path : str
            Output path prefix (``_Results.fits`` is appended).
        DICT : dict
            Nested dictionary of result-name / content pairs.
        """
        hdulist = fits.HDUList()
        for i, j in DICT.items():
            for name, content in j.items():
                try:
                    hdu = self._create_hdu(name, content)
                    hdulist.append(hdu)
                except TypeError as e:
                    log.warning(
                        f"Caught {type(e).__name__}, skipping {name}. Details: {e}"
                    )
                    pass

        output_filename = path + "_Results.fits"
        log.info(f"Saving DQM results in {output_filename}")
        hdulist.writeto(output_filename, overwrite=True)
        hdulist.info()
