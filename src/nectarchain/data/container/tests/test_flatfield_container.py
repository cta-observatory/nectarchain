import numpy as np

from nectarchain.data.container import FlatFieldContainer


class TestFlatFieldContainer:
    def test_create(self):
        c = FlatFieldContainer(
            run_number=np.uint16(42),
            npixels=np.uint16(10),
            pixels_id=np.arange(1, 11, dtype=np.uint16),
            ucts_timestamp=np.array([100, 200], dtype=np.uint64),
            event_type=np.array([0, 1], dtype=np.uint8),
            event_id=np.array([0, 1], dtype=np.uint32),
            amp_int_per_pix_per_event=np.zeros((2, 10, 2), dtype=np.float32),
            eff_coef=np.zeros((2, 10, 2), dtype=np.float32),
            bad_pixels=np.array([], dtype=np.uint16),
        )
        c.validate()
        assert c.run_number == 42 and c.npixels == 10
        assert c.amp_int_per_pix_per_event.shape == (2, 10, 2)

    def test_bad_pixels(self):
        c = FlatFieldContainer(
            run_number=np.uint16(1),
            npixels=np.uint16(3),
            pixels_id=np.array([1, 2, 3], dtype=np.uint16),
            ucts_timestamp=np.array([0], dtype=np.uint64),
            event_type=np.array([0], dtype=np.uint8),
            event_id=np.array([0], dtype=np.uint32),
            amp_int_per_pix_per_event=np.zeros((1, 3, 2), dtype=np.float32),
            eff_coef=np.zeros((1, 3, 2), dtype=np.float32),
            bad_pixels=np.array([5], dtype=np.uint16),
        )
        c.validate()
        assert c.bad_pixels.tolist() == [5]
