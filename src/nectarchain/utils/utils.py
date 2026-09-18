import importlib
import logging
import math
from typing import Sequence, Type, Union

import numpy as np
from ctapipe.core.component import Component
from ctapipe.core.traits import Path
from ctapipe_io_nectarcam import constants
from iminuit import Minuit
from scipy import interpolate, signal
from scipy.special import gammainc
from scipy.stats import chi2, norm

from ..data.container.core import NectarCAMContainer

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class ComponentUtils:
    """Utility methods for working with ctapipe Components."""

    @staticmethod
    def is_in_non_abstract_subclasses(
        component: Component, motherClass="NectarCAMComponent"
    ):
        """Check if a component is an instance of or equals a non-abstract subclass.

        Parameters
        ----------
        component : Component
            The component to check.
        motherClass : str, optional
            Name of the parent class.

        Returns
        -------
        bool
            True if the component matches.
        """
        from nectarchain.makers.component.core import NectarCAMComponent  # noqa: F401

        # module = importlib.import_module(f'nectarchain.makers.component.core')
        is_in = False
        if isinstance(component, eval(f"{motherClass}")):
            is_in = True
        else:
            for _, value in eval(motherClass).non_abstract_subclasses().items():
                is_in = np.logical_or(is_in, component == value)
        return is_in

    @staticmethod
    def get_specific_traits(component: Component):
        """Get the specific traits of a component, including sub-components.

        Parameters
        ----------
        component : Component
            The component whose traits to retrieve.

        Returns
        -------
        dict
            Dictionary of trait names and values.
        """
        importlib.import_module(f"{component.__module__}")
        traits_dict = component.class_traits()
        if ComponentUtils.is_in_non_abstract_subclasses(
            component, "NectarCAMComponent"
        ) and not (component.SubComponents.default_value is None):
            for component_name in component.SubComponents.default_value:  # CPT
                _class = getattr(
                    importlib.import_module("nectarchain.makers.component"),
                    component_name,
                )
                traits_dict.update(ComponentUtils.get_specific_traits(_class))
        traits_dict.pop("config", True)
        traits_dict.pop("parent", True)
        return traits_dict

    @staticmethod
    def get_configurable_traits(component: Component):
        """Get the configurable (non-read-only) traits of a component.

        Parameters
        ----------
        component : Component
            The component whose configurable traits to retrieve.

        Returns
        -------
        dict
            Dictionary of configurable trait names and values.
        """
        traits_dict = ComponentUtils.get_specific_traits(component)
        output_traits_dict = traits_dict.copy()
        for key, item in traits_dict.items():
            if item.read_only:
                output_traits_dict.pop(key)
        return output_traits_dict

    @staticmethod
    def get_class_name_from_ComponentName(componentName: str):
        """Get the class corresponding to a component name.

        Parameters
        ----------
        componentName : str
            Name of the component.

        Returns
        -------
        type
            The component class.

        Raises
        ------
        ValueError
            If the component name is not a valid NectarCAMComponent child.
        """
        from nectarchain.makers.component.core import NectarCAMComponent

        for class_name, _class in NectarCAMComponent.non_abstract_subclasses().items():
            if componentName in class_name:
                return _class

        raise ValueError(
            "componentName is not a valid component, this component is not known as a "
            "child of NectarCAMComponent"
        )


class ContainerUtils:
    """Utility methods for working with NectarCAMContainer objects."""

    @staticmethod
    def add_missing_pixels_to_container(
        container: NectarCAMContainer, pad_value=np.nan
    ):
        """
        Pads fields of `~nectarchain.data.container.core.NectarCAMContainer` with
        missing pixels (due to e.g. an incomplete camera) with chosen value.
        For boolean arrays related to pixel status, zero/one-padding is applied
        appriopriately.
        """

        # Make sure the container has `pixels_id` values
        try:
            pixels_id_input = container.pixels_id
        except Exception as e:
            raise ValueError(f"{container} has no field named `pixels_id`: {e}")

        # Do nothing if there are no missing pixels
        if len(pixels_id_input) == constants.N_PIXELS:
            log.info("Input container already contains data for all pixels!")
            return

        log.info(
            f"Input container contains data for "
            f"{len(pixels_id_input)}/{constants.N_PIXELS} pixels, "
            f"will add missing pixels and fill missing-pixel data with {pad_value}"
        )

        log.debug(f"Original container: {container}")

        for name, field in zip(container.keys(), container.values()):
            # Update the pixels_id with the full camera
            if name == "pixels_id":
                setattr(
                    container,
                    "pixels_id",
                    constants.PIXEL_INDEX.astype(pixels_id_input.dtype),
                )
            elif isinstance(field, np.ndarray):
                # Find pixel axis if there is one
                pixel_axis = None
                for i, dim in enumerate(field.shape):
                    if dim == len(pixels_id_input):
                        pixel_axis = i
                        break
                if pixel_axis is None:
                    continue

                # Reshape fields to full camera with NaN values for missing pixels
                # For fields related to pixel status, apply zero/one-padding
                shape_new_field = list(field.shape)
                shape_new_field[pixel_axis] = constants.N_PIXELS
                # Pixel status in NectarCAMPedestalContainer, FlatFieldContainer
                # NOTE: `bad_pixels` does not have a `pixel_axis` so nothing happens
                if name in ["pixel_mask", "bad_pixels"]:
                    new_field = np.ones(shape_new_field, dtype=field.dtype)
                # Pixel status in GainContainer
                elif name in ["is_valid"]:
                    new_field = np.zeros(shape_new_field, dtype=field.dtype)
                else:
                    new_field = np.full(shape_new_field, pad_value, dtype=field.dtype)

                # Copy data in slices so that the correct axis is zero/one-padded
                # Also sorts the arrays in terms of `PIXEL_INDEX`
                pixel_pos = np.searchsorted(constants.PIXEL_INDEX, pixels_id_input)
                slc = [slice(None)] * field.ndim
                slc[pixel_axis] = pixel_pos
                new_field[tuple(slc)] = field

                # Update the container
                setattr(container, name, new_field)

        log.debug(f"Updated container: {container}")

        return

    @staticmethod
    def get_container_from_hdf5(
        path: Path,
        container_classes: Union[
            Type[NectarCAMContainer], Sequence[Type[NectarCAMContainer]]
        ],
        group_names: Union[str, Sequence[str]] = "data",
    ) -> NectarCAMContainer:
        """Wrapper function for `NectarCAMContainer.from_hdf5`.

        Loads a container for a given `path`, allowing to scan for:
            - different NectarCAMContainer subclasses;
            - different group names.

        For example, you may want to load the output of a gain calibration tool, but
        you don't know whether it is a `SPEFitContainer` or `GainContainer`.

        Or you may want to load a `NectarCAMPedestalContainer`, which can either have
        the group name "data" or "data_combined" depending on the slicing applied.

        Parameters
        ----------
        path : Path
            Path to the HDF5 file.
        container_classes : Type[NectarCAMContainer] or sequence of such types
            One or more container subclasses to attempt loading.
        group_names : str or sequence of str, optional
            Group names to try inside the HDF5 structure. Default is "data".

        Returns
        -------
        NectarCAMContainer
            The successfully loaded container.

        Raises
        ------
        TypeError
            If `container_classes` contains items that are not NectarCAMContainer
            subclasses.
        ValueError
            If no container can be loaded from the file using any class/group
            combination.
        """

        # Normalize inputs lists
        if isinstance(container_classes, type):
            container_classes = [container_classes]
        if isinstance(group_names, (str, bytes)):
            group_names = [group_names]

        # Validate input classes
        for _cls in container_classes:
            if not issubclass(_cls, NectarCAMContainer):
                raise TypeError(
                    f"{_cls.__name__} is not a subclass of "
                    f"{NectarCAMContainer.__name__}"
                )

        exceptions = []

        for _cls in container_classes:
            for group_name in group_names:
                try:
                    container = next(_cls.from_hdf5(path, group_name=group_name))
                    log.info(f"Loaded {_cls.__name__} from {path}")
                    return container
                except Exception as e:
                    log.debug(
                        f"Failed to load {_cls.__name__} with `group_name` = "
                        f"'{group_name}': {e}"
                    )
                    log.debug("Adding exception to `exceptions` list")
                    exceptions.append((_cls.__name__, group_name, str(e)))

        # Raise an error if no container could be loaded
        raise ValueError(
            f"Failed to load any container from {path}. Tried: {exceptions}"
        )


class multiprocessing:
    """Namespace for multiprocessing utility functions."""

    @staticmethod
    def custom_error_callback(error):
        """Log an error from a multiprocessing worker.

        Parameters
        ----------
        error : Exception
            The error raised by the worker.
        """
        log.error(f"Got an error: {error}")
        log.error(error, exc_info=True)


class Statistics:
    """Statistical utility functions."""

    @staticmethod
    def chi2_pvalue(ndof: int, likelihood: float):
        """Compute the p-value from a chi-squared distribution.

        Parameters
        ----------
        ndof : int
            Number of degrees of freedom.
        likelihood : float
            The likelihood value.

        Returns
        -------
        float
            The p-value.
        """
        return 1 - chi2(df=ndof).cdf(likelihood)


class UtilsMinuit:
    """Utility functions for creating and configuring Minuit instances."""

    @staticmethod
    def make_minuit_par_kwargs(parameters):
        """Create *Parameter Keyword Arguments* for the `Minuit` constructor.

        updated for Minuit >2.0
        """
        names = parameters.parnames
        kwargs = {"names": names, "values": {}}

        for parameter in parameters.parameters:
            kwargs["values"][parameter.name] = parameter.value
            min_ = None if np.isnan(parameter.min) else parameter.min
            max_ = None if np.isnan(parameter.max) else parameter.max
            error = 0.1 if np.isnan(parameter.error) else parameter.error
            kwargs[f"limit_{parameter.name}"] = (min_, max_)
            kwargs[f"error_{parameter.name}"] = error
            if parameter.frozen:
                kwargs[f"fix_{parameter.name}"] = True
        return kwargs

    @staticmethod
    def set_minuit_parameters_limits_and_errors(m: Minuit, parameters: dict):
        """Function to set minuit parameter limits and errors with Minuit >2.0

        Args:
            m (Minuit): a Minuit instance
            parameters (dict): dict containing parameters names, limits errors and
            values.
        """
        for name in parameters["names"]:
            m.limits[name] = parameters[f"limit_{name}"]
            m.errors[name] = parameters[f"error_{name}"]
            if parameters.get(f"fix_{name}", False):
                m.fixed[name] = True


# Useful functions for the fit
def gaussian(x, mu, sig):
    """Gaussian (normal) probability density function.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate the PDF.
    mu : float
        Mean of the Gaussian.
    sig : float
        Standard deviation of the Gaussian.

    Returns
    -------
    array-like
        PDF values.
    """

    # return (1./(sig*np.sqrt(2*math.pi))) *
    # np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return norm.pdf(x, loc=mu, scale=sig)


def weight_gaussian(x, N, mu, sig):
    """Weighted Gaussian (normal) PDF scaled by N.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate.
    N : float
        Scaling factor.
    mu : float
        Mean of the Gaussian.
    sig : float
        Standard deviation of the Gaussian.

    Returns
    -------
    array-like
        Scaled PDF values.
    """
    return N * gaussian(x, mu, sig)


def doubleGauss(x, sig1, mu2, sig2, p):
    """Double Gaussian model with a low- and a high-charge component.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate.
    sig1 : float
        Width of the low-charge Gaussian.
    mu2 : float
        Mean of the high-charge Gaussian.
    sig2 : float
        Width of the high-charge Gaussian.
    p : float
        Proportion of the low-charge component.

    Returns
    -------
    array-like
        Evaluated double Gaussian.
    """
    return p * 2 * gaussian(x, 0, sig1) + (1 - p) * gaussian(x, mu2, sig2)


def PMax(r):
    """p_{max} in equation 6 in Caroff et al. (2019)

    Args:
        r (float): SPE resolution

    Returns:
        float : p_{max}
    """
    if r > np.sqrt((np.pi - 2) / 2):
        pmax = np.pi / (2 * (r**2 + 1))
    else:
        pmax = np.pi * r**2 / (np.pi * r**2 + np.pi - 2 * r**2 - 2)
    return pmax


def ax(p, res):
    """a in equation 4 in Caroff et al. (2019)

    Args:
        p (float): proportion of the low charge component (2 gaussians model)
        res (float): SPE resolution

    Returns:
        float : a
    """
    return (2 / np.pi) * p**2 - p / (res**2 + 1)


def bx(p, mu2):
    """b in equation 4 in Caroff et al. (2019)

    Args:
        p (float): proportion of the low charge component (2 gaussians model)
        mu2 (float): position of the high charge Gaussian

    Returns:
        float : b
    """
    return np.sqrt(2 / np.pi) * 2 * p * (1 - p) * mu2


def cx(sig2, mu2, res, p):
    """c in equation 4 in Caroff et al. (2019)
    Note : There is a typo in the article 1-p**2 -> (1-p)**2

    Args:
        sig2 (float): width of the high charge Gaussian
        mu2 (float): position of the high charge Gaussian
        res (float): SPE resolution
        p (float): proportion of the low charge component (2 gaussians model)

    Returns:
        float : c
    """
    return (1 - p) ** 2 * mu2**2 - (1 - p) * (sig2**2 + mu2**2) / (res**2 + 1)


def delta(p, res, sig2, mu2):
    """well known delta in 2nd order polynom

    Args:
        p (_type_): _description_
        res (_type_): _description_
        sig2 (_type_): _description_
        mu2 (_type_): _description_

    Returns:
        float : b**2 - 4*a*c
    """
    return bx(p, mu2) * bx(p, mu2) - 4 * ax(p, res) * cx(sig2, mu2, res, p)


def ParamU(p, r):
    """d in equation 6 in Caroff et al. (2019)

    Args:
        p (float): proportion of the low charge component (2 gaussians model)
        r (float): SPE resolution

    Returns:
        float : d
    """
    return (8 * (1 - p) ** 2 * p**2) / np.pi - 4 * (
        2 * p**2 / np.pi - p / (r**2 + 1)
    ) * ((1 - p) ** 2 - (1 - p) / (r**2 + 1))


def ParamS(p, r):
    """e in equation 6 in Caroff et al. (2019)

    Args:
        p (float): proportion of the low charge component (2 gaussians model)
        r (float): SPE resolution

    Returns:
        float : e
    """
    e = (4 * (2 * p**2 / np.pi - p / (r**2 + 1)) * (1 - p)) / (r**2 + 1)
    return e


def SigMin(p, res, mu2):
    """sigma_{high,min} in equation 6 in Caroff et al. (2019)

    Args:
        p (float): proportion of the low charge component (2 gaussians model)
        res (float): SPE resolution
        mu2 (float): position of the high charge Gaussian

    Returns:
        float : sigma_{high,min}
    """
    return mu2 * np.sqrt(
        (-ParamU(p, res) + (bx(p, mu2) ** 2 / mu2**2)) / (ParamS(p, res))
    )


def SigMax(p, res, mu2):
    """sigma_{high,min} in equation 6 in Caroff et al. (2019)

    Args:
        p (float): proportion of the low charge component (2 gaussians model)
        res (float): SPE resolution
        mu2 (float): position of the high charge Gaussian

    Returns:
        float : sigma_{high,min}
    """
    temp = (-ParamU(p, res)) / (ParamS(p, res))
    if temp < 0:
        err = ValueError("-d/e must be < 0")
        log.error(err, exc_info=True)
        raise err
    else:
        return mu2 * np.sqrt(temp)


def sigma1(p, res, sig2, mu2):
    """sigma_{low} in equation 5 in Caroff et al. (2019)

    Args:
        sig2 (float): width of the high charge Gaussian
        mu2 (float): position of the high charge Gaussian
        res (float): SPE resolution
        p (float): proportion of the low charge component (2 gaussians model)

    Returns:
        float : sigma_{low}
    """
    return (-bx(p, mu2) + np.sqrt(delta(p, res, sig2, mu2))) / (2 * ax(p, res))


def sigma2(n, p, res, mu2):
    """sigma_{high} in equation 7 in Caroff et al. (2019)

    Args:
        n (float): parameter n in equation
        mu2 (float): position of the high charge Gaussian
        res (float): SPE resolution
        p (float): proportion of the low charge component (2 gaussians model)

    Returns:
        float : sigma_{high}
    """
    if (-ParamU(p, res) + (bx(p, mu2) ** 2 / mu2**2)) / (ParamS(p, res)) > 0:
        return SigMin(p, res, mu2) + n * (SigMax(p, res, mu2) - SigMin(p, res, mu2))
    else:
        return n * SigMax(p, res, mu2)

    # The real final model callign all the above for luminosity (lum) + PED, wil return
    # probability of number of Spe


def MPE2(x, pp, res, mu2, n, muped, sigped, lum, **kwargs):
    """Multi-photo-electron (MPE) model summing nPEPDF contributions.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate.
    pp : float
        Parameter p' for the constrained double Gaussian.
    res : float
        SPE resolution.
    mu2 : float
        Position of the high-charge Gaussian.
    n : float
        Parameter n for sigma2.
    muped : float
        Pedestal mean.
    sigped : float
        Pedestal width.
    lum : float
        Luminosity parameter (mean number of photo-electrons).
    **kwargs
        Additional keyword arguments. If ``ntotalPE`` is provided, it
        specifies the maximum number of photo-electrons to sum.

    Returns
    -------
    array-like
        Evaluated MPE model.
    """
    log.debug(
        f"pp = {pp}, res = {res}, mu2 = {mu2}, n = {n}, muped = {muped}, "
        f"sigped = {sigped}, lum = {lum}"
    )
    f = 0
    ntotalPE = kwargs.get("ntotalPE", 0)
    if ntotalPE == 0:
        # about 1sec
        for i in range(1000):
            if gammainc(i + 1, lum) < 1e-5:
                ntotalPE = i
                break
    # print(ntotalPE)
    # about 8 sec, 1 sec by nPEPDF call
    # for i in range(ntotalPE):
    #    f = f + ((lum**i)/math.factorial(i)) * np.exp(-lum) *
    #    nPEPDF(x,pp,res,mu2,n,muped,sigped,i,int(mu2*ntotalPE+10*mu2))

    f = np.sum(
        [
            (lum**i)
            / math.factorial(i)
            * np.exp(-lum)
            * nPEPDF(
                x, pp, res, mu2, n, muped, sigped, i, int(mu2 * ntotalPE + 10 * mu2)
            )
            for i in range(ntotalPE)
        ],
        axis=0,
    )  # 10 % faster
    return f


# Fnal model shape/function (for one SPE)
def doubleGaussConstrained(x, pp, res, mu2, n):
    """Constrained double Gaussian model for a single SPE.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate.
    pp : float
        Parameter p' controlling the low-charge proportion.
    res : float
        SPE resolution.
    mu2 : float
        Position of the high-charge Gaussian.
    n : float
        Parameter n controlling sigma2.

    Returns
    -------
    array-like
        Evaluated constrained double Gaussian.
    """
    p = pp * PMax(res)
    sig2 = sigma2(n, p, res, mu2)
    sig1 = sigma1(p, res, sig2, mu2)
    return doubleGauss(x, sig1, mu2, sig2, p)


# Get the gain from the parameters model
def Gain(pp, res, mu2, n):
    """analytic gain computatuon

    Args:
        mu2 (float): position of the high charge Gaussian
        res (float): SPE resolution
        pp (float): p' in equation 7 in Caroff et al. (2019)
        n (float): n in equation 7 in Caroff et al. (2019)

    Returns:
        float : gain
    """
    p = pp * PMax(res)
    sig2 = sigma2(n, p, res, mu2)
    return (1 - p) * mu2 + 2 * p * sigma1(p, res, sig2, mu2) / np.sqrt(2 * np.pi)


def nPEPDF(x, pp, res, mu2, n, muped, sigped, nph, size_charge):
    """Compute the PDF for n photo-electrons.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate the PDF.
    pp : float
        Parameter p' for the constrained double Gaussian.
    res : float
        SPE resolution.
    mu2 : float
        Position of the high-charge Gaussian.
    n : float
        Parameter n for sigma2.
    muped : float
        Pedestal mean.
    sigped : float
        Pedestal width.
    nph : int
        Number of photo-electrons.
    size_charge : int
        Size of the charge range for the evaluation grid.

    Returns
    -------
    array-like
        Normalized PDF values.
    """
    allrange = np.linspace(-1 * size_charge, size_charge, size_charge * 2)
    spe = []
    # about 2 sec this is the main pb
    # for i in range(len(allrange)):
    #    if (allrange[i]>=0):
    #        spe.append(doubleGaussConstrained(allrange[i],pp,res,mu2,n))
    #    else:
    #        spe.append(0)

    spe = doubleGaussConstrained(allrange, pp, res, mu2, n) * (
        allrange >= 0 * np.ones(allrange.shape)
    )  # 100 times faster

    # ~ plt.plot(allrange,spe)
    # npe = semi_gaussian(allrange, muped, sigped)
    npe = gaussian(allrange, 0, sigped)
    # ~ plt.plot(allrange,npe)
    # ~ plt.show()
    for i in range(nph):
        # npe = np.convolve(npe,spe,"same")
        npe = signal.fftconvolve(npe, spe, "same")
    # ~ plt.plot(allrange,npe)
    # ~ plt.show()
    fff = interpolate.UnivariateSpline(allrange, npe, ext=1, k=3, s=0)
    norm = np.trapezoid(fff(allrange), allrange)
    return fff(x - muped) / norm
