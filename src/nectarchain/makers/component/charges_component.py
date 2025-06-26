import copy
import logging
import time
from argparse import ArgumentError

import numpy as np
import numpy.ma as ma
from ctapipe.containers import EventType
from ctapipe.core.traits import Dict, Unicode
from ctapipe.image.extractor import FixedWindowSum  # noqa: F401
from ctapipe.image.extractor import FullWaveformSum  # noqa: F401
from ctapipe.image.extractor import GlobalPeakWindowSum  # noqa: F401
from ctapipe.image.extractor import LocalPeakWindowSum  # noqa: F401
from ctapipe.image.extractor import NeighborPeakWindowSum  # noqa: F401
from ctapipe.image.extractor import SlidingWindowMaxSum  # noqa: F401
from ctapipe.image.extractor import TwoPassWindowSum  # noqa: F401
from ctapipe.image.extractor import (  # noqa: F401
    BaselineSubtractedNeighborPeakWindowSum,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe_io_nectarcam import constants
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from numba import bool_, float64, guvectorize, int64

from ...data.container import (
    ChargesContainer,
    ChargesContainers,
    WaveformsContainer,
    WaveformsContainers,
)
from ..extractor.utils import CtapipeExtractor
from .core import ArrayDataComponent

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

__all__ = ["ChargesComponent"]

list_ctapipe_charge_extractor = [
    "FullWaveformSum",
    "FixedWindowSum",
    "GlobalPeakWindowSum",
    "LocalPeakWindowSum",
    "SlidingWindowMaxSum",
    "NeighborPeakWindowSum",
    "BaselineSubtractedNeighborPeakWindowSum",
    "TwoPassWindowSum",
]


list_nectarchain_charge_extractor = ["gradient_extractor"]


@guvectorize(
    [
        (int64[:], float64[:], bool_, bool_[:], int64[:]),
    ],
    "(s),(n),()->(n),(n)",
    nopython=True,
    cache=True,
)
def make_histo(charge, all_range, mask_broken_pix, _mask, hist_ma_data):
    """Compute histogram of charge with numba.

    Parameters
    ----------
        charge (np.ndarray(pixels,nevents)): charge
        all_range (np.ndarray(nbins)): charge range
        mask_broken_pix (np.ndarray(pixels)): mask on broxen pixels
        _mask (np.ndarray(pixels,nbins)): mask
        hist_ma_data (np.ndarray(pixels,nbins)): histogram
    """
    # print(f"charge.shape = {charge.shape[0]}")
    # print(f"_mask.shape = {_mask.shape[0]}")
    # print(f"_mask[0] = {_mask[0]}")
    # print(f"hist_ma_data[0] = {hist_ma_data[0]}")
    # print(f"mask_broken_pix = {mask_broken_pix}")

    if not (mask_broken_pix):
        # print("this pixel is not broken, let's continue computation")
        hist, _charge = np.histogram(
            charge,
            bins=np.arange(
                np.uint16(np.min(charge)) - 1, np.uint16(np.max(charge)) + 2, 1
            ),
        )
        # print(f"hist.shape[0] = {hist.shape[0]}")
        # print(f"charge.shape[0] = {_charge.shape[0]}")
        charge_edges = np.array(
            [np.mean(_charge[i : i + 2]) for i in range(_charge.shape[0] - 1)]
        )
        # print(f"charge_edges.shape[0] = {charge_edges.shape[0]}")
        mask = (all_range >= charge_edges[0]) * (all_range <= charge_edges[-1])
        # print(f"all_range = {int(all_range[0])}-{int(all_range[-1])}")
        # print(f"charge_edges[0] = {int(charge_edges[0])}")
        # print(f"charge_edges[-1] = {int(charge_edges[-1])}")
        # print(f"mask[0] = {mask[0]}")
        # print(f"mask[-1] = {mask[-1]}")

        # MASK THE DATA
        # print(f"mask.shape = {mask.shape[0]}")
        _mask[:] = ~mask
        # print(f"_mask[0] = {_mask[0]}")
        # print(f"_mask[-1] = {_mask[-1]}")
        # FILL THE DATA
        hist_ma_data[mask] = hist
        # print("work done")
    else:
        # print("this pixel is broken, skipped")
        pass


class ChargesComponent(ArrayDataComponent):
    method = Unicode(
        default_value="FullWaveformSum",
        help="the charge extraction method",
    ).tag(config=True)

    extractor_kwargs = Dict(
        default_value={},
        help="The kwargs to be pass to the charge extractor method",
    ).tag(config=True)

    SubComponents = copy.deepcopy(ArrayDataComponent.SubComponents)
    SubComponents.read_only = True

    def __init__(self, subarray, config=None, parent=None, *args, **kwargs):
        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **kwargs
        )

        self.__charges_hg = {}
        self.__charges_lg = {}
        self.__peak_hg = {}
        self.__peak_lg = {}

    def _init_trigger_type(self, trigger_type: EventType, **kwargs):
        """Initializes the ChargesMaker based on the trigger type.

        Parameters
        ----------
            trigger_type (EventType): The type of trigger.
            kwargs: Additional keyword arguments.

        Returns
        -------
            None
        """
        super()._init_trigger_type(trigger_type, **kwargs)
        name = __class__._get_name_trigger(trigger_type)
        log.info(f"initialization of the ChargesMaker following trigger type : {name}")
        self.__charges_hg[f"{name}"] = []
        self.__charges_lg[f"{name}"] = []
        self.__peak_hg[f"{name}"] = []
        self.__peak_lg[f"{name}"] = []

    def __call__(
        self,
        event: NectarCAMDataContainer,
        *args,
        **kwargs,
    ):
        wfs_hg_tmp, wfs_lg_tmp = super(ChargesComponent, self).__call__(
            event=event, return_wfs=True, *args, **kwargs
        )
        name = __class__._get_name_trigger(event.trigger.event_type)

        broken_pixels_hg, broken_pixels_lg = __class__._compute_broken_pixels_event(
            event, self._pixels_id
        )

        imageExtractor = __class__._get_imageExtractor(
            self.method, self.subarray, **self.extractor_kwargs
        )

        __image = CtapipeExtractor.get_image_peak_time(
            imageExtractor(
                np.array([wfs_hg_tmp, wfs_lg_tmp]),
                self.TEL_ID,
                None,
                np.array([broken_pixels_hg, broken_pixels_lg]),
            )
        )
        self.__charges_hg[f"{name}"].append(__image[0][0])
        self.__peak_hg[f"{name}"].append(__image[1][0])

        self.__charges_lg[f"{name}"].append(__image[0][1])
        self.__peak_lg[f"{name}"].append(__image[1][1])

    @staticmethod
    def _get_extractor_kwargs_from_method_and_kwargs(method: str, kwargs: dict):
        extractor_kwargs = {}
        for key in eval(method).class_own_traits().keys():
            if key in kwargs.keys():
                extractor_kwargs[key] = kwargs[key]
        if (
            "apply_integration_correction" in eval(method).class_own_traits().keys()
        ):  # to change the default behavior of ctapipe extractor
            extractor_kwargs["apply_integration_correction"] = kwargs.get(
                "apply_integration_correction", False
            )
        return extractor_kwargs

    @staticmethod
    def _get_imageExtractor(method: str, subarray: SubarrayDescription, **kwargs):
        """Create an instance of a charge extraction method based on the provided method
        name and subarray description.

        Parameters
        ----------
        method : str
            The name of the charge extraction method.
        subarray : SubarrayDescription
            The description of the subarray.
        ``**kwargs`` : dict
            Additional keyword arguments for the charge extraction method.

        Returns
        -------
        imageExtractor:
            An instance of the charge extraction method specified by `method` with the
            provided subarray description and keyword arguments.
        """
        if not (
            method in list_ctapipe_charge_extractor
            or method in list_nectarchain_charge_extractor
        ):
            raise ArgumentError(
                None,
                f"method must be in {list_ctapipe_charge_extractor} or "
                f"{list_nectarchain_charge_extractor}",
            )
        extractor_kwargs = __class__._get_extractor_kwargs_from_method_and_kwargs(
            method=method, kwargs=kwargs
        )
        log.debug(
            f"Extracting charges with method {method} and extractor_kwargs "
            f"{extractor_kwargs}"
        )
        imageExtractor = eval(method)(subarray, **extractor_kwargs)
        return imageExtractor

    def finish(self, *args, **kwargs):
        """Create an output container for the specified trigger type and method.

        Parameters
        ----------
            trigger_type (EventType): The type of trigger.
            method (str): The name of the charge extraction method.
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.

        Returns
        -------
            list: A list of ChargesContainer objects.
        """
        output = ChargesContainers()
        for i, trigger in enumerate(self.trigger_list):
            chargesContainer = ChargesContainer(
                run_number=ChargesContainer.fields["run_number"].type(self._run_number),
                npixels=ChargesContainer.fields["npixels"].type(self._npixels),
                camera=self.CAMERA_NAME,
                pixels_id=ChargesContainer.fields["pixels_id"].dtype.type(
                    self._pixels_id
                ),
                method=self.method,
                nevents=self.nevents(trigger),
                charges_hg=self.charges_hg(trigger),
                charges_lg=self.charges_lg(trigger),
                peak_hg=self.peak_hg(trigger),
                peak_lg=self.peak_lg(trigger),
                broken_pixels_hg=self.broken_pixels_hg(trigger),
                broken_pixels_lg=self.broken_pixels_lg(trigger),
                ucts_timestamp=self.ucts_timestamp(trigger),
                ucts_busy_counter=self.ucts_busy_counter(trigger),
                ucts_event_counter=self.ucts_event_counter(trigger),
                event_type=self.event_type(trigger),
                event_id=self.event_id(trigger),
                trig_pattern_all=self.trig_pattern_all(trigger),
                trig_pattern=self.trig_pattern(trigger),
                multiplicity=self.multiplicity(trigger),
            )
            output.containers[trigger] = chargesContainer
        return output

    @staticmethod
    def sort(chargesContainer: ChargesContainer, method: str = "event_id"):
        """Sorts the charges in a ChargesContainer object based on the specified method.

        Parameters
        ----------
        chargesContainer : ChargesContainer
            The ChargesContainer object to be sorted.
        method : str, optional
            The sorting method. Defaults to 'event_id'.

        Returns
        -------
        ChargesContainer:
            A new ChargesContainer object with the charges sorted based on the specified
            method.

        Raises
        ------
        ArgumentError:
            If the specified method is not valid.
        """
        output = ChargesContainer(
            run_number=chargesContainer.run_number,
            npixels=chargesContainer.npixels,
            camera=chargesContainer.camera,
            pixels_id=chargesContainer.pixels_id,
            nevents=chargesContainer.nevents,
            method=chargesContainer.method,
        )
        if method == "event_id":
            index = np.argsort(chargesContainer.event_id)
            for field in chargesContainer.keys():
                if not (
                    field
                    in [
                        "run_number",
                        "npixels",
                        "camera",
                        "pixels_id",
                        "nevents",
                        "method",
                    ]
                ):
                    output[field] = chargesContainer[field][index]
        else:
            raise ArgumentError(
                None, message=f"{method} is not a valid method for sorting"
            )
        return output

    @staticmethod
    def select_charges_hg(chargesContainer: ChargesContainer, pixel_id: np.ndarray):
        """Selects the charges from the ChargesContainer object for the given pixel_id
        and returns the result transposed.

        Parameters
        ----------
        chargesContainer : ChargesContainer
            The ChargesContainer object.
        pixel_id : np.ndarray
            An array of pixel IDs.

        Returns
        -------
        res : np.ndarray
            The selected charges from the ChargesContainer object for the given
            ``pixel_id``, transposed.
        """
        res = __class__.select_container_array_field(
            container=chargesContainer, pixel_id=pixel_id, field="charges_hg"
        )
        res = res.transpose(1, 0)
        return res

    @staticmethod
    def select_charges_lg(chargesContainer: ChargesContainer, pixel_id: np.ndarray):
        """Selects the charges from the ChargesContainer object for the given pixel_id
        and returns the result transposed.

        Parameters
        ----------
        chargesContainer : ChargesContainer
            The ChargesContainer object.
        pixel_id : np.ndarray
            An array of pixel IDs.

        Returns
        -------
        res : np.ndarray
            The selected charges from the ChargesContainer object for the given
            ``pixel_id``, transposed.
        """
        res = __class__.select_container_array_field(
            container=chargesContainer, pixel_id=pixel_id, field="charges_lg"
        )
        res = res.transpose(1, 0)
        return res

    def charges_hg(self, trigger: EventType):
        """Returns the charges for a specific trigger type as a NumPy array of unsigned
        16-bit integers.

        Parameters
        ----------
        trigger : EventType
            The specific trigger type.

        Returns
        -------
        : np.ndarray
            The charges for the specific trigger type.
        """
        return np.array(
            self.__charges_hg[__class__._get_name_trigger(trigger)],
            dtype=ChargesContainer.fields["charges_hg"].dtype,
        )

    def charges_lg(self, trigger: EventType):
        """Returns the charges for a specific trigger type as a NumPy array of unsigned
        16-bit integers.

        Parameters
        ----------
        trigger : EventType
            The specific trigger type.

        Returns
        -------
        : np.ndarray
            The charges for the specific trigger type.
        """
        return np.array(
            self.__charges_lg[__class__._get_name_trigger(trigger)],
            dtype=ChargesContainer.fields["charges_lg"].dtype,
        )

    def peak_hg(self, trigger: EventType):
        """Returns the peak charges for a specific trigger type as a NumPy array of
        unsigned 16-bit integers.

        Parameters
        ----------
        trigger : EventType
            The specific trigger type.

        Returns
        -------
        : np.ndarray
            The peak charges for the specific trigger type.
        """
        return np.array(
            self.__peak_hg[__class__._get_name_trigger(trigger)],
            dtype=ChargesContainer.fields["peak_hg"].dtype,
        )

    def peak_lg(self, trigger: EventType):
        """Returns the peak charges for a specific trigger type as a NumPy array of
        unsigned 16-bit integers.

        Parameters
        ----------
        trigger : EventType
            The specific trigger type.

        Returns
        -------
        : np.ndarray
            The peak charges for the specific trigger type.
        """
        return np.array(
            self.__peak_lg[__class__._get_name_trigger(trigger)],
            dtype=ChargesContainer.fields["peak_lg"].dtype,
        )

    @staticmethod
    def _create_from_waveforms_looping_eventType(
        waveformsContainers: WaveformsContainers, **kwargs
    ):
        chargesContainers = ChargesContainers()
        for key in waveformsContainers.containers.keys():
            chargesContainers.containers[key] = ChargesComponent.create_from_waveforms(
                waveformsContainer=waveformsContainers.containers[key], **kwargs
            )
        return chargesContainers

    @staticmethod
    def create_from_waveforms(
        waveformsContainer: WaveformsContainer,
        subarray=SubarrayDescription,
        method: str = "FullWaveformSum",
        **kwargs,
    ) -> ChargesContainer:
        """Create a ChargesContainer object from waveforms using the specified charge
        extraction method.

        Parameters
        ----------
        waveformsContainer : WaveformsContainer
            The waveforms container object.
        method : str, optional
            The charge extraction method to use (default is ``FullWaveformSum``).
        kwargs
            Additional keyword arguments to pass to the charge extraction method.

        Returns
        -------
        chargesContainer : ChargesContainer
            The charges container object containing the computed charges and peak times.
        """
        chargesContainer = ChargesContainer()
        for field in waveformsContainer.keys():
            if not (field in ["nsamples", "wfs_hg", "wfs_lg"]):
                chargesContainer[field] = waveformsContainer[field]
        log.info(f"computing hg charge with {method} method")
        charges_hg, peak_hg = __class__.compute_charges(
            waveformsContainer=waveformsContainer,
            channel=constants.HIGH_GAIN,
            subarray=subarray,
            method=method,
            **kwargs,
        )
        log.info(f"computing lg charge with {method} method")
        charges_lg, peak_lg = __class__.compute_charges(
            waveformsContainer=waveformsContainer,
            channel=constants.LOW_GAIN,
            subarray=subarray,
            method=method,
            **kwargs,
        )
        chargesContainer.charges_hg = charges_hg
        chargesContainer.charges_lg = charges_lg
        chargesContainer.peak_hg = peak_hg
        chargesContainer.peak_lg = peak_lg
        chargesContainer.method = method
        return chargesContainer

    @staticmethod
    def compute_charges(
        waveformsContainer: WaveformsContainer,
        channel: int,
        subarray: SubarrayDescription,
        method: str = "FullWaveformSum",
        tel_id: int = None,
        **kwargs,
    ):
        """Compute charge from waveforms.

        Parameters
        ----------
        waveformsContainer : WaveformsContainer
            The waveforms container object.
        subarray : SubarrayDescription
            The subarray to work on.
        channel : int
            The channel to compute charges for.
        method : str, optional
            The charge extraction method to use (default is ``FullWaveformSum``).
        tel_id : int, optional
            The telescope ID on which charges will be computed.
        kwargs
            Additional keyword arguments to pass to the charge extraction method.

        Raises
        ------
        ArgumentError
            If the extraction method is unknown.
        ArgumentError
            If the channel is unknown.

        Returns
        -------
        : tuple
            A tuple containing the computed charges and peak times.
        """
        # import is here for fix issue with pytest
        # (TypeError :  inference is not possible with python <3.9 (Numba conflict bc
        # there is no inference...))
        from ..extractor.utils import CtapipeExtractor

        if tel_id is None:
            tel_id = __class__.TEL_ID.default_value

        # by setting selected_gain_channel to None, we can now pass to ctapipe extractor
        # waveforms in (n_ch, n_pix, n_samples)
        # then out hase shape (n_events, 2, n_ch, n_pi)
        if channel == constants.HIGH_GAIN:
            gain_label = "hg"
            waveforms = waveformsContainer.wfs_hg
            broken_pixels = waveformsContainer.broken_pixels_hg
        elif channel == constants.LOW_GAIN:
            gain_label = "lg"
            waveforms = waveformsContainer.wfs_lg
            broken_pixels = waveformsContainer.broken_pixels_lg
        else:
            raise ArgumentError(
                None,
                f"channel must be {constants.LOW_GAIN} or {constants.HIGH_GAIN}",
            )

        imageExtractor = __class__._get_imageExtractor(
            method=method, subarray=subarray, **kwargs
        )
        out = np.array(
            [
                CtapipeExtractor.get_image_peak_time(
                    imageExtractor(
                        waveforms=waveforms,
                        tel_id=tel_id,
                        selected_gain_channel=None,
                        broken_pixels=broken_pixels,
                    )
                )
            ]
        )

        return ChargesContainer.fields[f"charges_{gain_label}"].dtype.type(
            out[:, 0, channel, :]
        ), ChargesContainer.fields[f"peak_{gain_label}"].dtype.type(
            out[:, 1, channel, :]
        )

    @staticmethod
    def histo_hg(
        chargesContainer: ChargesContainer, n_bins: int = 1000, autoscale: bool = True
    ) -> ma.masked_array:
        """Computes histogram of high gain charges from a ChargesContainer object.

        Parameters
        ----------
        chargesContainer : ChargesContainer
            A ChargesContainer object that holds information about charges from a
            specific run.
        n_bins : int, optional
            The number of bins in the charge histogram. Defaults to 1000.
        autoscale : bool, optional
            Whether to automatically detect the number of bins based on the pixel data.
            Defaults to True.

        Returns
        -------
        : ma.masked_array
            A masked array representing the charge histogram, where each row corresponds
            to an event and each column corresponds to a bin in the histogram.
        """
        return __class__._histo(
            chargesContainer=chargesContainer,
            field="charges_hg",
            n_bins=n_bins,
            autoscale=autoscale,
        )

    @staticmethod
    def histo_lg(
        chargesContainer: ChargesContainer, n_bins: int = 1000, autoscale: bool = True
    ) -> ma.masked_array:
        """Computes histogram of low gain charges from a ChargesContainer object.

        Parameters
        ----------
        chargesContainer : ChargesContainer
            A ChargesContainer object that holds information about charges from a
            specific run.
        n_bins : int, optional
            The number of bins in the charge histogram. Defaults to 1000.
        autoscale : bool, optional
            Whether to automatically detect the number of bins based on the pixel
            data. Defaults to True.

        Returns
        -------
        : ma.masked_array
            A masked array representing the charge histogram, where each row corresponds
            to an event and each column corresponds to a bin in the histogram.
        """
        return __class__._histo(
            chargesContainer=chargesContainer,
            field="charges_lg",
            n_bins=n_bins,
            autoscale=autoscale,
        )

    @staticmethod
    def _histo(
        chargesContainer: ChargesContainer,
        field: str,
        n_bins: int = 1000,
        autoscale: bool = True,
    ) -> ma.masked_array:
        """Computes histogram of charges for a given field from a ChargesContainer
        object. Numba is used to compute histograms in a vectorized way.

        Parameters
        ----------
        chargesContainer : ChargesContainer
            A ChargesContainer object that holds information about charges from a
            specific run.
        field : str
            The field name for which the histogram is computed.
        n_bins : int, optional
            The number of bins in the charge histogram. Defaults to 1000.
        autoscale : bool, optional
            Whether to automatically detect the number of bins based on the pixel data.
            Defaults to True.

        Returns
        -------
        : ma.masked_array
            A masked array representing the charge histogram, where each row
            corresponds to an event and each column corresponds to a bin in the
            histogram.
        """
        mask_broken_pix = np.array(
            (chargesContainer[field] == chargesContainer[field].mean(axis=0)).mean(
                axis=0
            ),
            dtype=bool,
        )
        log.debug(
            f"there are {mask_broken_pix.sum()} broken pixels (charge stays at same "
            f"level for each events)"
        )

        if autoscale:
            all_range = np.arange(
                np.uint16(np.min(chargesContainer[field].T[~mask_broken_pix].T)) - 0.5,
                np.uint16(np.max(chargesContainer[field].T[~mask_broken_pix].T)) + 1.5,
                1,
            )
            charge_ma = ma.masked_array(
                (
                    all_range.reshape(all_range.shape[0], 1)
                    @ np.ones((1, chargesContainer[field].shape[1]))
                ).T,
                mask=np.zeros(
                    (chargesContainer[field].shape[1], all_range.shape[0]), dtype=bool
                ),
            )
            broxen_pixels_mask = np.array(
                [mask_broken_pix for i in range(charge_ma.shape[1])]
            ).T
            start = time.time()
            _mask, hist_ma_data = make_histo(
                chargesContainer[field].T, all_range, mask_broken_pix
            )
            charge_ma.mask = np.logical_or(_mask, broxen_pixels_mask)
            hist_ma = ma.masked_array(hist_ma_data, mask=charge_ma.mask)
            log.debug(f"histogram hg computation time : {time.time() - start} sec")

            return ma.masked_array((hist_ma, charge_ma))

        else:
            hist = np.array(
                [
                    np.histogram(chargesContainer[field].T[i], bins=n_bins)[0]
                    for i in range(chargesContainer[field].shape[1])
                ]
            )
            charge = np.array(
                [
                    np.histogram(chargesContainer[field].T[i], bins=n_bins)[1]
                    for i in range(chargesContainer[field].shape[1])
                ]
            )
            charge_edges = np.array(
                [
                    np.mean(charge.T[i : i + 2], axis=0)
                    for i in range(charge.shape[1] - 1)
                ]
            ).T

            return np.array((hist, charge_edges))

    @property
    def _charges_hg(self):
        """Returns the charges_hg attribute.

        Returns:
            np.ndarray: The charges_hg attribute.
        """
        return copy.deepcopy(self.__charges_hg)

    @property
    def _charges_lg(self):
        """Returns the charges_lg attribute.

        Returns:
            np.ndarray: The charges_lg attribute.
        """
        return copy.deepcopy(self.__charges_lg)

    @property
    def _peak_hg(self):
        """Returns the peak_hg attribute.

        Returns:
            np.ndarray: The peak_hg attribute.
        """
        return copy.deepcopy(self.__peak_hg)

    @property
    def _peak_lg(self):
        """Returns the peak_lg attribute.

        Returns:
            np.ndarray: The peak_lg attribute.
        """
        return copy.deepcopy(self.__peak_lg)
