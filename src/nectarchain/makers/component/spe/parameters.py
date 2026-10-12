import copy
import logging

import astropy.units as u
import numpy as np

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


__all__ = ["Parameter", "Parameters"]


class Parameter:
    """A single named parameter with value, bounds, error, unit, and frozen status.

    Parameters
    ----------
    name : str
        Parameter name.
    value : object
        Parameter value.
    min : float, optional
        Lower bound.
    max : float, optional
        Upper bound.
    error : float, optional
        Uncertainty on the value.
    unit : astropy.units.Unit, optional
        Physical unit.
    frozen : bool, optional
        Whether the parameter is fixed during fitting.
    """

    def __init__(
        self,
        name: str,
        value,
        min=np.nan,
        max=np.nan,
        error=np.nan,
        unit=u.dimensionless_unscaled,
        frozen: bool = False,
    ) -> None:
        """Initialise a Parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        value : object
            Parameter value.
        min : float, optional
            Lower bound.
        max : float, optional
            Upper bound.
        error : float, optional
            Uncertainty on the value.
        unit : astropy.units.Unit, optional
            Physical unit.
        frozen : bool, optional
            Whether the parameter is fixed during fitting.
        """
        self.__name = name
        self.__value = value
        self.__error = error
        self.__min = min
        self.__max = max
        self.__unit = unit
        self.__frozen = frozen

    @classmethod
    def from_instance(cls, parameter):
        """Create a new Parameter from an existing Parameter instance.

        Parameters
        ----------
        parameter : Parameter
            The instance to copy.

        Returns
        -------
        Parameter
            A new Parameter with the same attributes.
        """
        return cls(
            name=parameter.name,
            value=parameter.value,
            min=parameter.min,
            max=parameter.max,
            error=parameter.error,
            unit=parameter.unit,
            frozen=parameter.frozen,
        )

    def __str__(self):
        """Return a formatted string representation of the parameter.

        Returns
        -------
        str
            Parameter description.
        """
        return (
            f"name : {self.__name}, value : {self.__value}, error : {self.__error},"
            f"unit : {self.__unit}, min : {self.__min}, max : {self.__max},"
            f"frozen : {self.__frozen}"
        )

    @property
    def name(self):
        """The parameter name.

        Returns
        -------
        str
        """
        return self.__name

    @name.setter
    def name(self, value: str):
        """Set the parameter name.

        Parameters
        ----------
        value : str
        """
        self.__name = value

    @property
    def value(self):
        """The parameter value.

        Returns
        -------
        object
        """
        return self.__value

    @value.setter
    def value(self, value):
        """Set the parameter value.

        Parameters
        ----------
        value : object
        """
        self.__value = value

    @property
    def min(self):
        """The lower bound.

        Returns
        -------
        float
        """
        return self.__min

    @min.setter
    def min(self, value):
        """Set the lower bound.

        Parameters
        ----------
        value : float
        """
        self.__min = value

    @property
    def max(self):
        """The upper bound.

        Returns
        -------
        float
        """
        return self.__max

    @max.setter
    def max(self, value):
        """Set the upper bound.

        Parameters
        ----------
        value : float
        """
        self.__max = value

    @property
    def unit(self):
        """The physical unit.

        Returns
        -------
        astropy.units.Unit
        """
        return self.__unit

    @unit.setter
    def unit(self, value: u.Unit):
        """Set the physical unit.

        Parameters
        ----------
        value : astropy.units.Unit
        """
        self.__unit = value

    @property
    def error(self):
        """The uncertainty on the value.

        Returns
        -------
        float
        """
        return self.__error

    @error.setter
    def error(self, value):
        """Set the uncertainty on the value.

        Parameters
        ----------
        value : float
        """
        self.__error = value

    @property
    def frozen(self):
        """Whether the parameter is fixed during fitting.

        Returns
        -------
        bool
        """
        return self.__frozen

    @frozen.setter
    def frozen(self, value: bool):
        """Set whether the parameter is fixed during fitting.

        Parameters
        ----------
        value : bool
        """
        self.__frozen = value


class Parameters:
    """A list-like container for Parameter objects.

    Parameters
    ----------
    parameters_liste : list of Parameter, optional
        Initial list of parameters.
    """

    def __init__(self, parameters_liste: list = []) -> None:
        """Initialise the Parameters container.

        Parameters
        ----------
        parameters_liste : list of Parameter, optional
            Initial list of parameters.
        """
        self.__parameters = copy.deepcopy(parameters_liste)

    def append(self, parameter: Parameter) -> None:
        """Append a Parameter to the container.

        Parameters
        ----------
        parameter : Parameter
            The parameter to add.
        """
        self.__parameters.append(parameter)

    def __getitem__(self, key: str):
        """Retrieve a Parameter by name.

        Parameters
        ----------
        key : str
            The parameter name.

        Returns
        -------
        Parameter or list
            The matching Parameter, or an empty list if not found.
        """
        for parameter in self.__parameters:
            if parameter.name == key:
                return parameter
        return []

    def __str__(self):
        """Return a multi-line string representation of all parameters.

        Returns
        -------
        str
        """
        string = ""
        for parameter in self.__parameters:
            string += str(parameter) + "\n"
        return string

    @property
    def parameters(self):
        """The internal list of Parameter objects.

        Returns
        -------
        list of Parameter
        """
        return self.__parameters

    @property
    def size(self):
        """The number of parameters.

        Returns
        -------
        int
        """
        return len(self.__parameters)

    @property
    def parnames(self):
        """The names of all parameters.

        Returns
        -------
        list of str
        """
        return [parameter.name for parameter in self.__parameters]

    @property
    def parvalues(self):
        """The values of all parameters.

        Returns
        -------
        list
        """
        return [parameter.value for parameter in self.__parameters]

    @property
    def unfrozen(self):
        """A new Parameters container with only the non-frozen parameters.

        Returns
        -------
        Parameters
        """
        parameters = Parameters()
        for parameter in self.__parameters:
            if not (parameter.frozen):
                parameters.append(parameter)
        return parameters
