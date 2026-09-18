import pytest

from nectarchain.utils.error import (
    DifferentPixelsID,
    MeanValueError,
    PedestalValueError,
    TooMuchFileException,
)


class TestTooMuchFileException:
    def test_raise(self):
        with pytest.raises(TooMuchFileException):
            raise TooMuchFileException()

    def test_is_exception(self):
        assert issubclass(TooMuchFileException, Exception)


class TestDifferentPixelsID:
    def test_message(self):
        with pytest.raises(DifferentPixelsID) as exc:
            raise DifferentPixelsID("mismatch")
        assert exc.value.message == "mismatch"


class TestPedestalValueError:
    def test_raise(self):
        with pytest.raises(PedestalValueError) as exc:
            raise PedestalValueError("bad pedestal")
        assert exc.value.message == "bad pedestal"
        assert issubclass(PedestalValueError, ValueError)


class TestMeanValueError:
    def test_raise(self):
        with pytest.raises(MeanValueError) as exc:
            raise MeanValueError("bad mean")
        assert exc.value.message == "bad mean"
        assert issubclass(MeanValueError, ValueError)
