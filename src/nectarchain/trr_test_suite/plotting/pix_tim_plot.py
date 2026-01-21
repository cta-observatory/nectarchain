import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pix_tim_config import df
from Utils import pe2photons, photons2pe

plt.style.use("../../utils/plot_style.mpltstyle")


# ==========================
#       PLOTTING (ONE FIGURE)
# ==========================
fig, axx = plt.subplots(figsize=(10, 7), constrained_layout=True)

# Use hue=Temperature, style=NSB, and size/markers for Voltage if desired
sns.lineplot(
    data=df,
    x="photons_spline",
    y="Mean_RMS",
    hue="Temperature",
    style="NSB",  # Different line styles for different NSB
    markers=True,  # Show markers
    dashes=False,  # Solid lines for clarity
    palette="viridis",
    linewidth=2,
)

# Horizontal reference lines
plt.axhline(1, ls="--", color="C4", alpha=0.6, label="_nolegend_")
plt.axhline(1 / np.sqrt(12), ls="--", color="gray", alpha=0.7, label="_nolegend_")

# Set log scale for x-axis and limits
plt.xscale("log")
plt.xlim(left=50)
plt.ylim(0, 2.7)
plt.xlabel("Illumination charge [photons]", fontsize=16)
plt.ylabel("Mean RMS per pixel [ns]", fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=14)
plt.tick_params(axis="both", which="minor", labelsize=12)

# Legend for Temperature and NSB
plt.legend(
    title="Temperature [°C] / NSB [mA]",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
    prop={"size": 12},
    handlelength=1.2,
)


plt.axvspan(20, 1000, alpha=0.1, color="C4")

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

# plt.legend(frameon=True, prop={"size": 12}, loc="upper right", handlelength=1.2)
plt.xlabel("Illumination charge [photons]")
plt.ylabel("Mean RMS per pixel [ns]")
plt.xscale("log")
plt.ylim(0, 2.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# secondary axis in p.e.
secax = axx.secondary_xaxis("top", functions=(pe2photons, photons2pe))
secax.set_xlabel("Illumination charge [p.e.]", labelpad=7, fontsize=16)

plt.tight_layout()
plt.savefig("mean_rms_vs_photons.png", dpi=300, bbox_inches="tight")

plt.show()
