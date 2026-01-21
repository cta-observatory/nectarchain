import json
import os

BASE = os.environ["NECTARCAMDATA"]  # your base path
BASEFIGURE = os.environ["NECTARCHAIN_FIGURES"]


metadata_file = "ff_metadata.json"  # path to your JSON file

outdir = f"{BASEFIGURE}/FFplots"
dirname = f"{BASE}/FF"

with open(metadata_file, "r") as f:
    metadata = json.load(f)

# now extract the variables
Runs = metadata["Runs"]
temp_map = metadata["temp_map"]
NSB_map = metadata["NSB_map"]

BAD_MODULE_IDS = metadata["bad_module_ids"]
BAD_PIXELS = set(metadata["bad_pixels"])


# ----------------- Helper functions ----------------- #
def categorize_run1(run):
    return NSB_map.get(run, "Unknown")


def categorize_run2(run):
    return temp_map.get(run, None)
