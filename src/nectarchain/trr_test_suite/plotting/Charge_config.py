import json
import os


# Load metadata from external JSON file
def load_metadata(filepath="Charge_metadata.json"):
    """Load configuration metadata from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


# Load the metadata
metadata = load_metadata()

# Access the data (convert string keys to integers where needed)
temp_map = {int(k): v for k, v in metadata["temp_map"].items()}
gain_map = {int(k): v for k, v in metadata["gain_map"].items()}
ff_map = {int(k): v for k, v in metadata["ff_map"].items()}
vvalff_map = {int(k): v for k, v in metadata["vvalff_map"].items()}
ivalnsb_map = {int(k): v for k, v in metadata["ivalnsb_map"].items()}

# Bad pixels and modules
BAD_MODULE_IDS = metadata["bad_module_ids"]
BAD_PIXELS_GAIN = metadata["bad_pixels_gain"]
BAD_PIXELS_HV = metadata["bad_pixels_HV"]

# Environment-based paths
BASE = os.environ["NECTARCAMDATA"]  # will raise KeyError if not set
BASEFIGURE = os.environ["NECTARCHAIN_FIGURES"]


dirname = f"{BASE}/runs/charges"
gain_path = f"{BASE}/SPEfit/thermal_gain"
pedestal_file = f"{BASE}/pedestals/Baseline_Thermal_Tests.npz"
ff_dir = f"{BASE}/FF"
path = f"{BASE}/runs"
db_data_path = path

outdir = f"{BASEFIGURE}/charge_comp_output"
