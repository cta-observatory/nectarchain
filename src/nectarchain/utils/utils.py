import importlib
import logging
import math

import numpy as np
from ctapipe.core.component import Component
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
    @staticmethod
    def is_in_non_abstract_subclasses(
        component: Component, motherClass="NectarCAMComponent"
    ):
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
        traits_dict = ComponentUtils.get_specific_traits(component)
        output_traits_dict = traits_dict.copy()
        for key, item in traits_dict.items():
            if item.read_only:
                output_traits_dict.pop(key)
        return output_traits_dict

    @staticmethod
    def get_class_name_from_ComponentName(componentName: str):
        from nectarchain.makers.component.core import NectarCAMComponent

        for class_name, _class in NectarCAMComponent.non_abstract_subclasses().items():
            if componentName in class_name:
                return _class

        raise ValueError(
            "componentName is not a valid component, this component is not known as a "
            "child of NectarCAMComponent"
        )


class ContainerUtils:
    @staticmethod
    def add_missing_pixels_to_container(container: NectarCAMContainer):
        """
        Zero-pads fields of `~nectarchain.data.container.core.NectarCAMContainer` with
        missing pixels (due to e.g. an incomplete camera). For boolean arrays related to
        pixel status, zero/one-padding is applied appriopriately.
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
            "will add missing pixels and fill missing-pixel data with NaN values"
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
                if name in ["pixel_mask", "bad_pixels"]:
                    new_field = np.ones(shape_new_field, dtype=field.dtype)
                # Pixel status in GainContainer
                elif name in ["is_valid"]:
                    new_field = np.zeros(shape_new_field, dtype=field.dtype)
                else:
                    new_field = np.full(shape_new_field, np.nan, dtype=field.dtype)

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


class multiprocessing:
    @staticmethod
    def custom_error_callback(error):
        log.error(f"Got an error: {error}")
        log.error(error, exc_info=True)


class Statistics:
    @staticmethod
    def chi2_pvalue(ndof: int, likelihood: float):
        return 1 - chi2(df=ndof).cdf(likelihood)


class UtilsMinuit:
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
    # return (1./(sig*np.sqrt(2*math.pi))) *
    # np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return norm.pdf(x, loc=mu, scale=sig)


def weight_gaussian(x, N, mu, sig):
    return N * gaussian(x, mu, sig)


def doubleGauss(x, sig1, mu2, sig2, p):
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
    norm = np.trapz(fff(allrange), allrange)
    return fff(x - muped) / norm
