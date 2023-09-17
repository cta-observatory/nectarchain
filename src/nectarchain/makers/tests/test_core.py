import logging

from nectarchain.data.container import ChargesContainer
from nectarchain.makers import ArrayDataMaker, ChargesMaker, WaveformsMaker

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.DEBUG
)
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class TestArrayDataMaker:
    run_number = 3938
    max_events = 100

    def test_merge(self):
        chargesMaker = ChargesMaker(
            run_number=TestArrayDataMaker.run_number,
            max_events=TestArrayDataMaker.max_events,
        )
        charges_1 = chargesMaker.make()
        chargesMaker_2 = ChargesMaker(
            run_number=TestArrayDataMaker.run_number,
            max_events=TestArrayDataMaker.max_events,
        )
        charges_2 = chargesMaker_2.make()

        merged = ArrayDataMaker.merge(charges_1, charges_2)
        assert isinstance(merged, ChargesContainer)

    def test_merge_different_container(self):
        chargesMaker = ChargesMaker(
            run_number=TestArrayDataMaker.run_number,
            max_events=TestArrayDataMaker.max_events,
        )
        charges_1 = chargesMaker.make()
        wfsMaker_2 = WaveformsMaker(
            run_number=TestArrayDataMaker.run_number,
            max_events=TestArrayDataMaker.max_events,
        )
        wfs_2 = wfsMaker_2.make()
        merged = ArrayDataMaker.merge(charges_1, wfs_2)
