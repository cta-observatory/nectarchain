import pytest


class TestGainNectarCAMComponent:
    def test_cannot_instantiate_abstract(self):
        from nectarchain.makers.component.gain_component import GainNectarCAMComponent

        with pytest.raises(TypeError):
            GainNectarCAMComponent()
