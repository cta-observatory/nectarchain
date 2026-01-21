import os

import matplotlib.pyplot as plt
import numpy as np
from pix_tim_config import df, output_dir
from Utils import pe2photons, photons2pe

plt.style.use("../../utils/plot_style.mpltstyle")

# ==========================
#       PLOTTING (ONE FIGURE)
# ==========================
os.makedirs(output_dir, exist_ok=True)
fig, axx = plt.subplots(figsize=(10, 7), constrained_layout=True)

# Get unique values for grouping
temperatures = df["Temperature"].cat.categories
nsb_values = sorted(df["NSB"].unique())

# Define color map and line styles
colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures)))
line_styles = {
    "0": "-",
    "10.6": "--",
    "20.4": "-.",
    "39.8": ":",
    "78.8": (0, (3, 1, 1, 1)),  # dashdot with custom pattern
}

# Plot each temperature and NSB combination
for i, temp in enumerate(temperatures):
    for nsb in nsb_values:
        subset = df[(df["Temperature"] == temp) & (df["NSB"] == nsb)]
        if not subset.empty:
            label = f"{temp}°C, {nsb} mA"
            axx.plot(
                subset["photons_spline"],
                subset["Mean_RMS"],
                color=colors[i],
                linestyle=line_styles.get(nsb, "-"),
                marker="o",
                markersize=4,
                linewidth=2,
                label=label,
                alpha=0.8,
            )

# Horizontal reference lines
axx.axhline(1, ls="--", color="C4", alpha=0.6, label="_nolegend_")
axx.axhline(1 / np.sqrt(12), ls="--", color="gray", alpha=0.7, label="_nolegend_")

# Set log scale for x-axis and limits
axx.set_xscale("log")
axx.set_xlim(left=50)
axx.set_ylim(0, 2.7)
axx.set_xlabel("Illumination charge [photons]", fontsize=16)
axx.set_ylabel("Mean RMS per pixel [ns]", fontsize=16)
axx.tick_params(axis="both", which="major", labelsize=14)
axx.tick_params(axis="both", which="minor", labelsize=12)

# Compact legend with smaller font
legend = axx.legend(
    title="Temp / NSB",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
    fontsize=8,
    title_fontsize=9,
    handlelength=1.5,
    handleheight=0.8,
    labelspacing=0.3,
    borderpad=0.4,
    ncol=2,  # Use 2 columns to make it more compact
)

# Shaded region for CTAO requirement
axx.axvspan(20, 1000, alpha=0.1, color="C4")

axx.text(
    51.5,
    1.04,
    "CTAO requirement",
    color="C4",
    fontsize=14,
    horizontalalignment="left",
    verticalalignment="center",
)

for x in [40, 200]:
    axx.annotate(
        "",
        xy=(x, 0.9),
        xytext=(x, 0.995),
        arrowprops=dict(color="C4", alpha=0.7, lw=2, arrowstyle="->"),
    )

# Secondary axis in p.e.
secax = axx.secondary_xaxis("top", functions=(pe2photons, photons2pe))
secax.set_xlabel("Illumination charge [p.e.]", labelpad=7, fontsize=16)

plt.tight_layout()

output_path = os.path.join(output_dir, "mean_rms_vs_photons.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()
