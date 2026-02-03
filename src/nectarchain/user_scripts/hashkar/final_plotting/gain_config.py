import json
import os

# ----------------------
# NECTARCAMDATA paths
# ----------------------
dirname = os.path.join(os.environ["NECTARCAMDATA"], "SPEfit")
dirname2 = os.path.join(os.environ["NECTARCAMDATA"], "PhotoStat")
outdir = os.path.join(os.environ["NECTARCHAIN_FIGURES"], "Gain_output")
path = os.path.join(os.environ["NECTARCAMDATA"], "runs")
db_data_path = os.path.join(os.environ["NECTARCAMDATA"], "runs")


# ----------------------
# Load runs from JSON
# ----------------------
json_file = os.path.join(os.path.dirname(__file__), "metadata/gain_metadata.json")
with open(json_file, "r") as f:
    config = json.load(f)

Runs = config["Runs"]
temp_map = {int(k): v for k, v in config["temp_map"].items()}
SPE_runs = config["SPE_runs"]
Photostat_runs = config["Photostat_runs"]

# Bad modules and bad pixels
BAD_MODULE_IDS = config["bad_module_ids"]
BAD_PIXELS_HV = set(config["bad_pixels"])
