import pytest


def test_stats_simple():
    import numpy as np

    from nectarchain.utils.stats import Stats

    s = Stats()
    s.add(1)
    s.add(2)
    s.add(3)

    np.testing.assert_allclose(s.mean, np.array([2.0]))
    np.testing.assert_allclose(s.std, np.array([1.0]))
    np.testing.assert_allclose(s.max, np.array([3.0]))
    np.testing.assert_allclose(s.min, np.array([1.0]))
    np.testing.assert_allclose(s.count, np.array([3]))


def test_stats_camera():
    import numpy as np

    from nectarchain.utils.stats import Stats

    s = Stats(shape=(5))
    s.add(
        np.array([0, 0, 0, 0, 0]),
        validmask=np.array([False, False, False, False, False]),
    )
    s.add(np.array([0, 1, 2, 3, 4]), validmask=np.array([True, True, True, True, True]))
    s.add(
        np.array([1, 2, 3, 4, 5]), validmask=np.array([True, True, True, False, False])
    )
    s.add(np.array([2, 3, 4, 5, 6]), validmask=None)
    s.add(
        np.array([3, 4, 5, 6, 7]),
        validmask=np.array([False, False, False, False, False]),
    )

    np.testing.assert_allclose(s.count, np.array([3, 3, 3, 2, 2]))
    np.testing.assert_allclose(s.mean, np.array([1, 2, 3, 4, 5]))
    np.testing.assert_allclose(s.variance, np.array([1, 1, 1, 2, 2]))
    np.testing.assert_allclose(s.min, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(s.max, np.array([2, 3, 4, 5, 6]))


def test_stats_copy():
    from nectarchain.utils.stats import CameraSampleStats, CameraStats, Stats

    a = Stats()
    assert id(a) != id(a.copy())

    a = CameraStats()
    assert id(a) != id(a.copy())

    a = CameraSampleStats()
    assert id(a) != id(a.copy())


def test_stats_lowcounts():
    import numpy as np

    from nectarchain.utils.stats import Stats

    s = Stats(shape=(5))
    s.add(
        np.array([0, 0, 0, 0, 0]), validmask=np.array([True, True, True, True, False])
    )
    s.add(np.array([1, 1, 1, 1, 1]))
    s.add(np.array([2, 2, 2, 2, 2]))

    np.testing.assert_array_equal(
        s.get_lowcount_mask(3), np.array([False, False, False, False, True])
    )


def test_stats_merge():
    import numpy as np

    from nectarchain.utils.stats import Stats

    s = Stats(shape=(5))
    s.add(
        np.array([0, 0, 0, 0, 0]),
        validmask=np.array([False, False, False, False, False]),
    )
    s.add(np.array([0, 1, 2, 3, 4]), validmask=np.array([True, True, True, True, True]))

    s2 = Stats(shape=(5))
    s2.add(
        np.array([1, 2, 3, 4, 5]), validmask=np.array([True, True, True, False, False])
    )
    s2.add(np.array([2, 3, 4, 5, 6]), validmask=None)
    s2.add(
        np.array([3, 4, 5, 6, 7]),
        validmask=np.array([False, False, False, False, False]),
    )

    s.merge(s2)

    np.testing.assert_allclose(s.count, np.array([3, 3, 3, 2, 2]))
    np.testing.assert_allclose(s.mean, np.array([1, 2, 3, 4, 5]))
    np.testing.assert_allclose(s.variance, np.array([1, 1, 1, 2, 2]))
    np.testing.assert_allclose(s.min, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(s.max, np.array([2, 3, 4, 5, 6]))


def test_stats_merge2():
    import numpy as np

    from nectarchain.utils.stats import Stats

    s = Stats(shape=(5))
    s.add(
        np.array([0, 0, 0, 0, 0]),
        validmask=np.array([False, False, False, False, False]),
    )
    s.add(np.array([0, 1, 2, 3, 4]), validmask=np.array([True, True, True, True, True]))

    s2 = Stats(shape=(5))
    s2.add(
        np.array([1, 2, 3, 4, 5]), validmask=np.array([True, True, True, False, False])
    )
    s2.add(np.array([2, 3, 4, 5, 6]), validmask=None)
    s2.add(
        np.array([3, 4, 5, 6, 7]),
        validmask=np.array([False, False, False, False, False]),
    )

    s += s2

    np.testing.assert_allclose(s.count, np.array([3, 3, 3, 2, 2]))
    np.testing.assert_allclose(s.mean, np.array([1, 2, 3, 4, 5]))
    np.testing.assert_allclose(s.variance, np.array([1, 1, 1, 2, 2]))
    np.testing.assert_allclose(s.min, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(s.max, np.array([2, 3, 4, 5, 6]))


def test_stats_merge3():
    import numpy as np

    from nectarchain.utils.stats import Stats

    s1 = Stats(shape=(5))
    s1.add(
        np.array([0, 0, 0, 0, 0]),
        validmask=np.array([False, False, False, False, False]),
    )
    s1.add(
        np.array([0, 1, 2, 3, 4]), validmask=np.array([True, True, True, True, True])
    )

    s2 = Stats(shape=(5))
    s2.add(
        np.array([1, 2, 3, 4, 5]), validmask=np.array([True, True, True, False, False])
    )
    s2.add(np.array([2, 3, 4, 5, 6]), validmask=None)
    s2.add(
        np.array([3, 4, 5, 6, 7]),
        validmask=np.array([False, False, False, False, False]),
    )

    s = s1 + s2

    assert id(s) != id(s1)
    assert id(s) != id(s2)

    np.testing.assert_allclose(s.count, np.array([3, 3, 3, 2, 2]))
    np.testing.assert_allclose(s.mean, np.array([1, 2, 3, 4, 5]))
    np.testing.assert_allclose(s.variance, np.array([1, 1, 1, 2, 2]))
    np.testing.assert_allclose(s.min, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(s.max, np.array([2, 3, 4, 5, 6]))


def test_stats_shape():
    from ctapipe_io_nectarcam import constants as nc

    from nectarchain.utils.stats import CameraSampleStats, CameraStats, Stats

    s = Stats()
    assert s.shape == (1,)

    s = CameraStats()
    assert s.shape == (nc.N_GAINS, nc.N_PIXELS)

    s = CameraSampleStats()
    assert s.shape == (nc.N_GAINS, nc.N_PIXELS, nc.N_SAMPLES)


def test_stats_print():
    from nectarchain.utils.stats import Stats

    s = Stats()
    s.add(1)
    s.add(2)
    s.add(3)

    assert (
        s.__str__()
        == "mean: [2.]\nstd: [1.]\nmin: [1.]\nmax: [3.]\ncount: [3]\nshape: (1,)"
    )
    assert s.__repr__() == s.__str__()


def test_stats_badmerge():
    from nectarchain.utils.stats import Stats

    s = Stats()
    s.add(1)

    s2 = Stats(5)
    s2.add([1, 2, 3, 4, 5])

    with pytest.raises(
        ValueError, match="Trying to merge from a different shape this:.*"
    ):
        s.merge(s2)
