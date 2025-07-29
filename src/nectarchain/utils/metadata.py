"""
Functions to write metadata in output h5/fits files.
Taken from `lstcam_calib` with minor adaptations for NectarCAM.
"""

import logging

from ctapipe.core import Container, Field, Provenance
from ctapipe.io import metadata as meta

log = logging.getLogger(__name__)

__all__ = [
    "MetaData",
    "get_ctapipe_metadata",
    "get_local_metadata",
    "add_metadata_to_hdu",
]

PROV = Provenance()


class MetaData(Container):
    """Some metadata."""

    NECTARCHAIN_VERSION = Field(None, "Version of nectarchain")
    CTAPIPE_VERSION = Field(None, "Version of ctapipe")
    CTAPIPE_IO_NECTARCAM_VERSION = Field(None, "Version of ctapipe_io_nectarcam")
    TEL_ID = Field(None, "Telescope id")
    PROV_LOG = Field(None, "Provenance log file")
    RUN_START = Field(None, "Start time of first run used")


def get_ctapipe_metadata(
    description="Cat-A calibration coefficients", data_category="A", site="CTA-North"
):
    """Fill reference metadata following ctapipe standard."""
    from .. import __version__ as nectarchain_version

    activity = PROV.current_activity
    activity_meta = meta.Activity.from_provenance(activity.provenance)
    activity_meta.software_name = "nectarchain"
    activity_meta.software_version = nectarchain_version

    reference = meta.Reference(
        contact=meta.Contact(organization="MST Consortium"),
        product=meta.Product(
            description=description,
            data_levels=["R1"],
            data_category=data_category,
            data_association="Telescope",
            data_model_name="Unofficial monitoring R1",
            data_model_version="1.0",
        ),
        instrument=meta.Instrument(site=site, class_="Camera", type_="MST"),
        process=meta.Process(type_="Observation", subtype="Calibration"),
        activity=activity_meta,
    )

    return reference


def get_local_metadata(tel_id, provenance_log, run_start):
    """Get local metadata container."""
    from ctapipe import __version__ as ctapipe_version
    from ctapipe_io_nectarcam import __version__ as ctapipe_io_nectarcam_version

    from .. import __version__ as nectarchain_version

    metadata = MetaData()
    metadata.NECTARCHAIN_VERSION = nectarchain_version
    metadata.CTAPIPE_VERSION = ctapipe_version
    metadata.CTAPIPE_IO_NECTARCAM_VERSION = ctapipe_io_nectarcam_version
    metadata.TEL_ID = tel_id
    metadata.PROV_LOG = provenance_log
    metadata.RUN_START = run_start

    return metadata


def add_metadata_to_hdu(metadata, hdu):
    """
    Write metadata dictionary to a fits HDU .

    Parameters
    ----------
    metadata: dictionary of metadata
    hdu: fits HDU where to add metadata
    """
    for k, item in metadata.items():
        if len(k) <= 8 or "HIERARCH" in k:
            hdu.header[k] = metadata[k]
        else:
            hdu.header[f"HIERARCH {k}"] = metadata[k]
