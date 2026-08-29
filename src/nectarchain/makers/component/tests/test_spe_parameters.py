import astropy.units as u
import numpy as np

from nectarchain.makers.component.spe.parameters import Parameter, Parameters


class TestParameter:
    def test_create(self):
        p = Parameter(name="test", value=42.0)
        assert p.name == "test"
        assert p.value == 42.0

    def test_defaults(self):
        p = Parameter(name="x", value=1.0)
        assert np.isnan(p.min)
        assert np.isnan(p.max)
        assert np.isnan(p.error)
        assert p.unit == u.dimensionless_unscaled
        assert p.frozen is False

    def test_setters(self):
        p = Parameter(name="x", value=1.0)
        p.value = 99.0
        p.min = 0.0
        p.max = 100.0
        p.error = 1.5
        p.frozen = True
        assert p.value == 99.0
        assert p.min == 0.0
        assert p.max == 100.0
        assert p.error == 1.5
        assert p.frozen is True

    def test_from_instance(self):
        p1 = Parameter(name="a", value=1.0, min=0.0, max=2.0, frozen=True)
        p2 = Parameter.from_instance(p1)
        assert p2.name == "a"
        assert p2.value == 1.0
        assert p2.min == 0.0
        assert p2.max == 2.0
        assert p2.frozen is True

    def test_str(self):
        p = Parameter(name="p", value=3.0)
        s = str(p)
        assert "name : p" in s
        assert "value : 3.0" in s

    def test_unit_setter(self):
        p = Parameter(name="p", value=1.0)
        p.unit = u.meter
        assert p.unit == u.meter


class TestParameters:
    def test_append_and_getitem(self):
        params = Parameters()
        p = Parameter(name="alpha", value=1.0)
        params.append(p)
        assert params["alpha"] is p

    def test_getitem_missing(self):
        params = Parameters()
        assert params["missing"] == []

    def test_size(self):
        params = Parameters()
        assert params.size == 0
        params.append(Parameter(name="a", value=1.0))
        assert params.size == 1

    def test_parnames(self):
        params = Parameters()
        params.append(Parameter(name="a", value=1))
        params.append(Parameter(name="b", value=2))
        assert params.parnames == ["a", "b"]

    def test_parvalues(self):
        params = Parameters()
        params.append(Parameter(name="a", value=1.0))
        params.append(Parameter(name="b", value=2.0))
        assert params.parvalues == [1.0, 2.0]

    def test_unfrozen(self):
        params = Parameters()
        params.append(Parameter(name="a", value=1.0, frozen=False))
        params.append(Parameter(name="b", value=2.0, frozen=True))
        uf = params.unfrozen
        assert uf.size == 1
        assert uf["a"].name == "a"

    def test_str(self):
        params = Parameters()
        params.append(Parameter(name="x", value=1.0))
        s = str(params)
        assert "name : x" in s

    def test_parameters_property(self):
        params = Parameters()
        p = Parameter(name="p", value=0.5)
        params.append(p)
        assert params.parameters == [p]

    def test_constructor_with_list(self):
        p = Parameter(name="p", value=1.0)
        params = Parameters([p])
        assert params.size == 1
