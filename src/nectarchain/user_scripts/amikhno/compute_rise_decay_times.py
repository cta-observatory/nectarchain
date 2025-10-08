import argparse
import os

import numpy as np
import numpy.ma as ma
from ctapipe.containers import EventType
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry

# ctapipe modules
from ctapipe.visualization import CameraDisplay
from ctapipe_io_nectarcam import NectarCAMEventSource
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(
    description="Run computation of mean parameters of waveform"
)

parser.add_argument("-r", "--run-number", required=True, help="Run number")
parser.add_argument(
    "-n",
    "--pixel-number",
    type=int,
    required=True,
    help="Pixel number",
)
parser.add_argument(
    "-p",
    "--run-path",
    default=f'{os.environ.get("NECTARCAMDATA", "").strip()}',
    help="Path to run file",
)
args = parser.parse_args()

# --- Assign other variables ---
run_number = args.run_number
pixel_number = args.pixel_number
run_path = args.run_path + f"/runs/NectarCAM.Run{run_number}.0000.fits.fz"


def extract_times(mean_waveform, total_time=60):
    """
    Compute rise time, decay time, and time of maximum
    from a mean waveform (1D array).
    """
    n_samples = len(mean_waveform)
    time = np.linspace(0, total_time, n_samples)

    baseline = np.mean(mean_waveform[:10])
    max_val = np.max(mean_waveform)
    max_idx = np.argmax(mean_waveform)
    t_max = time[max_idx]

    height = max_val - baseline
    if height <= 0:
        return np.nan, np.nan, np.nan  # invalid waveform

    threshold_10 = baseline + 0.10 * height
    threshold_90 = baseline + 0.90 * height

    try:
        # --- RISE (10% to 90%) ---
        t10_rise, t90_rise = np.nan, np.nan
        for i in range(1, n_samples):
            if mean_waveform[i - 1] < threshold_10 <= mean_waveform[i]:
                frac = (threshold_10 - mean_waveform[i - 1]) / (
                    mean_waveform[i] - mean_waveform[i - 1]
                )
                t10_rise = time[i - 1] + frac * (time[i] - time[i - 1])
                break
        for i in range(1, n_samples):
            if mean_waveform[i - 1] < threshold_90 <= mean_waveform[i]:
                frac = (threshold_90 - mean_waveform[i - 1]) / (
                    mean_waveform[i] - mean_waveform[i - 1]
                )
                t90_rise = time[i - 1] + frac * (time[i] - time[i - 1])
                break

        # --- DECAY (90% to 10%) ---
        t90_decay, t10_decay = np.nan, np.nan
        for i in range(n_samples - 1, 0, -1):
            if mean_waveform[i - 1] >= threshold_90 > mean_waveform[i]:
                frac = (mean_waveform[i - 1] - threshold_90) / (
                    mean_waveform[i - 1] - mean_waveform[i]
                )
                t90_decay = time[i - 1] + frac * (time[i] - time[i - 1])
                break
        for i in range(n_samples - 1, 0, -1):
            if mean_waveform[i - 1] >= threshold_10 > mean_waveform[i]:
                frac = (mean_waveform[i - 1] - threshold_10) / (
                    mean_waveform[i - 1] - mean_waveform[i]
                )
                t10_decay = time[i - 1] + frac * (time[i] - time[i - 1])
                break

        rise_time = (
            t90_rise - t10_rise
            if (np.isfinite(t90_rise) and np.isfinite(t10_rise))
            else np.nan
        )
        decay_time = (
            t10_decay - t90_decay
            if (np.isfinite(t90_decay) and np.isfinite(t10_decay))
            else np.nan
        )

        return rise_time, decay_time, t_max

    except Exception:
        return np.nan, np.nan, np.nan


def plot_waveform_param(mean_signals, pix_id, rise_times, decay_times, t_max_values):
    # ---- Choose pixel to plot ----
    pixel_id = pix_id
    print(f"dtype pix_id {type(pix_id)}")  # change this to any pixel index (0–1854)

    # Get waveform and results for this pixel
    mean_waveform = mean_signals[pixel_id]
    t_max = t_max_values[pixel_id]

    n_samples = len(mean_waveform)
    time = np.linspace(0, 60, n_samples)

    # --- recompute thresholds and crossing times for this pixel (needed for markers) ---
    baseline = np.mean(mean_waveform[:10])
    max_val = np.max(mean_waveform)
    height = max_val - baseline
    threshold_10 = baseline + 0.10 * height
    threshold_90 = baseline + 0.90 * height

    # Helper function: find crossing
    def find_crossing(waveform, t, threshold, direction="rise"):
        if direction == "rise":
            for i in range(1, len(waveform)):
                if waveform[i - 1] < threshold <= waveform[i]:
                    frac = (threshold - waveform[i - 1]) / (
                        waveform[i] - waveform[i - 1]
                    )
                    return t[i - 1] + frac * (t[i] - t[i - 1])
        elif direction == "decay":
            for i in range(len(waveform) - 1, 0, -1):
                if waveform[i - 1] >= threshold > waveform[i]:
                    frac = (waveform[i - 1] - threshold) / (
                        waveform[i - 1] - waveform[i]
                    )
                    return t[i - 1] + frac * (t[i] - t[i - 1])
        return np.nan

    t10_rise = find_crossing(mean_waveform, time, threshold_10, "rise")
    t90_rise = find_crossing(mean_waveform, time, threshold_90, "rise")
    t90_decay = find_crossing(mean_waveform, time, threshold_90, "decay")
    t10_decay = find_crossing(mean_waveform, time, threshold_10, "decay")

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(
        time, mean_waveform, label=f"Pixel {pixel_id} mean waveform", color="royalblue"
    )

    # Threshold lines
    plt.axhline(threshold_10, color="gray", linestyle=":", label="10% level")
    plt.axhline(threshold_90, color="gray", linestyle="--", label="90% level")

    # Rise interval markers
    if not np.isnan(t10_rise) and not np.isnan(t90_rise):
        plt.axvline(t10_rise, color="green", linestyle="--")
        plt.axvline(t90_rise, color="green", linestyle="--")
        x_rise_center = 0.5 * (t10_rise + t90_rise)
        y_text = mean_waveform.min() - 0.05 * (
            mean_waveform.max() - mean_waveform.min()
        )
        plt.text(
            x_rise_center,
            y_text,
            f"{t90_rise - t10_rise:.2f} ns",
            color="green",
            ha="right",
            va="bottom",
            fontsize=16,
        )

    # Decay interval markers
    if not np.isnan(t90_decay) and not np.isnan(t10_decay):
        plt.axvline(t90_decay, color="red", linestyle="--")
        plt.axvline(t10_decay, color="red", linestyle="--")
        x_decay_center = 0.5 * (t90_decay + t10_decay)
        y_text = mean_waveform.min() - 0.12 * (
            mean_waveform.max() - mean_waveform.min()
        )
        plt.text(
            x_decay_center,
            y_text,
            f"{t10_decay - t90_decay:.2f} ns",
            color="red",
            ha="left",
            va="bottom",
            fontsize=16,
        )

    # Mark maximum
    plt.axvline(
        t_max, color="orange", linestyle="-", label=f"Time of max = {t_max:.2f} ns"
    )
    plt.scatter([t_max], [max_val], color="orange", zorder=5)

    # Labels & layout
    plt.xlabel("Time (ns)", fontsize=14)
    plt.ylabel("Mean signal amplitude", fontsize=14)
    plt.title(f"Pixel {pixel_id}: waveform with rise/decay markers", fontsize=16)
    plt.legend(fontsize=12, loc="best")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"pixel_{pixel_id}_waveform_run_{run_number}.pdf", format="pdf", dpi=300
    )
    plt.show()


def plot_over_camera(rise_times, decay_times, t_max_values):
    # rise time
    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot()
    disp1 = CameraDisplay(camera, show_frame=False, ax=ax)
    mask_raise = rise_times > 5
    disp1.image = ma.array(rise_times, mask=mask_raise)
    disp1.add_colorbar()
    disp1.set_limits_minmax(0, 4.5)

    # Set colorbar label for subplot 1
    cbar1 = fig.axes[-1]
    cbar1.set_ylabel("Time (ns)", rotation=90, labelpad=15, fontsize=16)
    cbar1.tick_params(labelsize=11)
    # Increase tick label size on colorbar
    # # Axis labels
    ax.set_xlabel("x (m)", fontsize=14)
    ax.set_ylabel("y (m)", fontsize=14)
    # Tick label size
    ax.tick_params(axis="both", which="major", labelsize=11)
    # Title
    plt.title("Mean rise time per pixel", fontsize=16)
    plt.savefig(f"Raise_time_over_camera_run_{run_number}.pdf", format="pdf", dpi=300)

    # decay time
    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot()
    disp1 = CameraDisplay(camera, show_frame=False, ax=ax)
    mask_decay = decay_times > 7
    disp1.image = ma.array(decay_times, mask=mask_decay)
    disp1.add_colorbar()
    disp1.set_limits_minmax(0, 5.5)
    # Set colorbar label for subplot 1
    cbar1 = fig.axes[-1]
    cbar1.set_ylabel("Time (ns)", rotation=90, labelpad=15, fontsize=16)
    cbar1.tick_params(labelsize=11)  # Increase tick label size on colorbar
    # Axis labels
    ax.set_xlabel("x (m)", fontsize=14)
    ax.set_ylabel("y (m)", fontsize=14)
    # Tick label size
    ax.tick_params(axis="both", which="major", labelsize=11)
    # Title
    plt.title("Mean decay time per pixel", fontsize=16)
    plt.savefig(f"Decay_time_over_camera_run_{run_number}.pdf", format="pdf", dpi=300)

    # time of maximum

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot()
    disp1 = CameraDisplay(camera, show_frame=False, ax=ax)
    disp1.image = t_max_values
    disp1.add_colorbar()

    # Set colorbar label for subplot 1
    cbar1 = fig.axes[-1]
    cbar1.set_ylabel("Time (ns)", rotation=90, labelpad=15, fontsize=16)
    cbar1.tick_params(labelsize=11)  # Increase tick label size on colorbar
    # Axis labels
    ax.set_xlabel("x (m)", fontsize=14)
    ax.set_ylabel("y (m)", fontsize=14)
    # Tick label size
    ax.tick_params(axis="both", which="major", labelsize=11)
    # Title
    plt.title("Mean peak time per pixel", fontsize=16)
    plt.savefig(f"Peak_time_over_camera_run_{run_number}.pdf", format="pdf", dpi=300)


# main
if __name__ == "__main__":
    camera = CameraGeometry.from_name("NectarCam-003").transform_to(
        EngineeringCameraFrame()
    )
    tel_id = 0
    waveforms_all = []
    # load run

    reader = NectarCAMEventSource(input_url=run_path, max_events=100)

    for event in reader:
        if event.trigger.event_type == EventType.FLATFIELD:
            broken_pixels = event.mon.tel[0].pixel_status.hardware_failing_pixels[0]
            waveform = event.r0.tel[tel_id].waveform[0][~broken_pixels]
            waveforms_all.append(waveform)

    waveforms_array = np.array(waveforms_all)

    # ---- Loop over all pixels ----
    # mean_signals shape = (1855, 60)
    rise_times = np.zeros(1855)
    decay_times = np.zeros(1855)
    t_max_values = np.zeros(1855)

    mean_signal_per_pixel = np.mean(waveforms_array, axis=0)
    print(f"shape of the mean signal per pix {mean_signal_per_pixel.shape[0]}")

    for pix in range(1855):
        rise, decay, t_max = extract_times(mean_signal_per_pixel[pix])
        rise_times[pix] = rise
        decay_times[pix] = decay
        t_max_values[pix] = t_max

    print("Rise times shape:", rise_times.shape)  # (1855,)
    print("Decay times shape:", decay_times.shape)  # (1855,)
    print("T_max shape:", t_max_values.shape)  # (1855,)

    plot_waveform_param(
        mean_signal_per_pixel, pixel_number, rise_times, decay_times, t_max_values
    )
    plot_over_camera(rise_times, decay_times, t_max_values)

    print("[INFO]: Done")
