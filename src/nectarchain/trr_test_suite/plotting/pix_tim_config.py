import json
import logging
import os
import subprocess

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ==========================
#      LOAD METADATA
# ==========================
def load_metadata(filepath="pix_tim_metadata.json"):
    """Load configuration metadata from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


# Load metadata
metadata = load_metadata()

# Convert string keys to integers
temperature_map = {int(k): v for k, v in metadata["temperature_map"].items()}
voltage_map = {int(k): v for k, v in metadata["voltage_map"].items()}
nsb_map = {int(k): v for k, v in metadata["nsb_map"].items()}
temp_order = metadata["temp_order"]

# ==========================
#      READ DATA
# ==========================
BASE = os.environ["NECTARCAMDATA"]  # will raise KeyError if not set
file_path = os.path.join(BASE, "pix_tim/pix_time_uncer_data.txt")

# Check if file exists and provide helpful error message
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Data file not found at: {file_path}\n"
        f"NECTARCAMDATA is set to: {BASE}\n"
        f"Please verify the file path or adjust the relative path in the config."
    )

try:
    df = pd.read_csv(file_path, header=0, sep="\t")
except UnicodeDecodeError:
    # Try with different encoding or check if it's the wrong file
    print(f"Error: Cannot read {file_path} as text file.")
    print("This might be a binary file. Please check the file type.")
    # Try to determine file type

    try:
        result = subprocess.run(
            ["file", file_path],
            capture_output=True,
            text=True,
            check=True,
        )
        logging.info(f"File type: {result.stdout.strip()}")
    except Exception as e:
        logging.error(
            "Failed to determine file type for %s: %s",
            file_path,
            e,
        )
        raise

# Map Run to Temperature, Voltage, and NSB
df["Temperature"] = df["Run"].map(temperature_map)
df["Voltage"] = df["Run"].map(voltage_map)
df["NSB"] = df["Run"].map(nsb_map)

# Drop rows without temperature info
df = df.dropna(subset=["Temperature"])

# Convert to categorical with proper order
df["Temperature"] = pd.Categorical(
    df["Temperature"], categories=temp_order, ordered=True
)
df["Voltage"] = df["Voltage"].astype(str)
df["NSB"] = df["NSB"].astype(str)


FIGURE_BASE = os.environ["NECTARCHAIN_FIGURES"]  # will raise KeyError if not set
output_dir = os.path.join(FIGURE_BASE, "pixel_timing_output")
