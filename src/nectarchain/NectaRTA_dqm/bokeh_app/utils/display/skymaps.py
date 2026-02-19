# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
This module stores the Bokeh webpage skymap maker for the RTA of NectarCAM.
"""


# imports
import logging
logger = logging.getLogger(__name__)

# Bokeh imports
from bokeh.models import (
    Div,
    TabPanel,
)


__all__ = ["make_tab_skymaps"]


def make_tab_skymaps():
    """Waiting for DL2 and DL3 to write this part"""
    return TabPanel(
        child=Div(text="TBD - Waiting for DL2 & DL3"),
        title="Skymaps"
    )