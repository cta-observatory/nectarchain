import logging
import os
import pathlib
from multiprocessing import Pool

import numpy as np

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

from nectarchain.makers.calibration import PedestalNectarCAMCalibrationTool

events_per_slice = 500
nthreads = 30

run_list = np.concatenate(
    (
        # np.arange(6882, 6891),
        # np.arange(6672, 6681),
        # np.arange(7144, 7153),
        # np.arange(6954, 6963),
        # np.arange(7020, 7029),
        # np.arange(6543, 6552),
        # np.arange(7077, 7086),
        # np.arange(7153, 7181),
        # np.arange(7182, 7190),
        # np.arange(6891, 6927),
        # np.arange(6681, 6717),
        # np.arange(6552, 6588),
        # np.arange(6963, 6999),
        # np.arange(7086, 7122),
        # np.arange(7029, 7065),
        np.arange(6552, 6588),
        np.arange(7086, 7110),
    )
)


def process_run(run_number):
    outfile = os.environ["NECTARCAMDATA"] + "/runs/pedestal_cfilt3s_{}.h5".format(
        run_number
    )
    tool = PedestalNectarCAMCalibrationTool(
        progress_bar=True,
        run_number=run_number,
        max_events=1999,
        events_per_slice=events_per_slice,
        log_level=0,
        output_path=outfile,
        overwrite=True,
        filter_method="ChargeDistributionFilter",
        charge_sigma_low_thr=3.0,
        charge_sigma_high_thr=3.0,
    )

    tool.initialize()
    tool.setup()

    tool.start()
    tool.finish()


args = [int(x) for x in run_list]
pool = Pool(processes=nthreads)
pool.map(process_run, args)
pool.close()
pool.join()
