"""
Script showing how to generate movies with the waveform evolution.

To be used with https://github.com/cta-observatory/ctapipe,
as well as https://github.com/cta-observatory/ctapipe_io_nectarcam, which includes
ctapipe.io.nectarcameventsource, to read NectarCAM data, pending
https://github.com/cta-observatory/ctapipe_io_nectarcam/pull/2

protozfitsreader should be installed (see
https://github.com/cta-sst-1m/protozfitsreader) because
ctapipe.io.nectarcameventsource depends on it to read zfits files.

"""

import matplotlib.pylab as plt
import numpy as np
from ctapipe.image import HillasParameterizationError, hillas_parameters, tailcuts_clean
from ctapipe.io import EventSeeker, event_source
from ctapipe.visualization import CameraDisplay
from matplotlib.animation import FuncAnimation

ped_path = "ped/NectarCAM.Run1247.0000.fits.fz"

num_events = 10

path = "obs/NectarCAM.Run1250.0000.fits.fz"
reader = event_source(input_url=path, max_events=num_events)
seeker = EventSeeker(reader)

runid = path.split("NectarCAM.Run")[1].split(".")[0]

delay = 100  # ms
fps = 1000.0 / delay  # frame per second
hi, lo = 0, 1  # channel index in data
tel = 0  # NectarCAM QM telescope ID
adc_to_pe = 58.0  # theoretical value
winsize_start, winsize_end = (
    6,
    10,
)  # quick rise, long tail: start winsize_start before max, ends winsize_end after
# max, in ns
tailcuts = [5, 10]
neighbors = 3  # number of neighboring pixels to include in Hillas cleaning

perform_calib = True
clean = True
verbose = True
save = False

cmap = "gnuplot2"

currentevent = seeker[0]
num_samples = seeker[0].nectarcam.tel[tel].svc.num_samples
trigtype = 1


def get_pedestals(path):
    wfs = []
    maxev = 500
    for i, ev in enumerate(event_source(path, max_events=maxev)):
        wfs.append(ev.r0.tel[tel].waveform)
    wfs = np.array(wfs)  # evt, gain, pixels, samples
    return wfs.mean(axis=0)  # gain, pixels, samples


peds_hi = get_pedestals(ped_path)[hi]  # pixels, samples; all in high gain


def get_window(event, gain):
    w = event.r0.tel[tel].waveform[gain]
    max = np.max(w[w.argmax(axis=0)])
    imax = np.where(w.T == max)[0][0]
    if imax - winsize_start < 0:
        istart = 0
        iend = winsize_start + winsize_end
    elif imax + winsize_end > num_samples - 1:
        istart = num_samples - 1 - winsize_start - winsize_end
        iend = num_samples - 1
    else:
        # quick rise, long tail, offset time window
        istart = imax - winsize_start
        iend = imax + winsize_end
    if iend > num_samples - 1:
        iend = num_samples - 1
    if istart < 0:
        istart = 0
    return istart, iend


counter = get_window(currentevent, hi)[0]
hillasdone = False
sw = []


def animation():
    fig = plt.figure(num="NectarCAM events display", figsize=(14, 10))
    fig.suptitle("CT{}, run {}".format(tel, runid))

    ax_hi_raw = fig.add_subplot(221)
    ax_hi_charge = fig.add_subplot(222)
    ax_hi_wf = fig.add_subplot(223)
    ax_hi_plot_wf = fig.add_subplot(224)

    camgeom = seeker[0].inst.subarray.tel[tel].camera

    disp_hi_raw = CameraDisplay(camgeom, ax=ax_hi_raw, autoupdate=True)
    disp_hi_raw.cmap = cmap
    disp_hi_raw.add_colorbar(ax=ax_hi_raw)

    disp_hi_charge = CameraDisplay(camgeom, ax=ax_hi_charge, autoupdate=True)
    disp_hi_charge.cmap = cmap
    disp_hi_charge.add_colorbar(ax=ax_hi_charge)

    disp_hi_wf = CameraDisplay(camgeom, ax=ax_hi_wf, autoupdate=True)
    disp_hi_wf.cmap = cmap
    disp_hi_wf.add_colorbar(ax=ax_hi_wf)

    ax_hi_plot_wf.set_title("High gain, wave form (ADC)")
    ax_hi_plot_wf.set_xlabel("Time (ns)")
    ax_hi_plot_wf.set_ylabel("ADC summed over all pixels")

    def update(frames):
        global counter, currentevent, hillasdone, sw, ped_hi
        event = currentevent

        hiistart, hiiend = get_window(event, hi)

        if counter >= hiiend or event.r0.tel[tel].trigger_type != trigtype:
            event = next(iter(seeker))  # get next event
            currentevent = event
            disp_hi_charge.clear_overlays()
            hiistart, hiiend = get_window(event, hi)
            hillasdone = False  # reset Hillas reco flag
            counter = hiistart

        w = event.r0.tel[tel].waveform

        ax_hi_raw.set_title(
            "High gain, raw data (ADC), event {}".format(event.r0.event_id)
        )

        hiw2 = w[hi].T[hiistart:hiiend]  # select time window
        wf_hi_max = np.amax(hiw2)

        if verbose:
            print(
                "INFO High gain: event id {}, counter {}, max(waveform)={}, window "
                "{}-{} ns".format(
                    event.r0.event_id, counter, w[hi].max(), hiistart, hiiend
                )
            )

        ax_hi_charge.set_title(
            "High gain, charge (PE), event {}, window {}-{} ns".format(
                event.r0.event_id, hiistart, hiiend
            )
        )

        image_hi_raw = hiw2.sum(axis=0)

        if perform_calib:
            # Very rough calibration
            image_hi_charge = ((w[hi] - peds_hi).T[hiistart:hiiend].T / adc_to_pe).sum(
                axis=1
            )

            if clean and not hillasdone:
                # Cleaning
                cleanmask_hi = tailcuts_clean(
                    camgeom,
                    image_hi_charge,
                    picture_thresh=tailcuts[1],
                    boundary_thresh=tailcuts[0],
                    min_number_picture_neighbors=neighbors,
                )
                image_hi_charge[cleanmask_hi == 0] = 0

                # Hillas reco
                try:
                    hillas_param_hi = hillas_parameters(camgeom, image_hi_charge)
                    disp_hi_charge.overlay_moments(
                        hillas_param_hi,
                        with_label=False,
                        color="red",
                        alpha=0.7,
                        linewidth=2,
                        linestyle="dashed",
                    )
                    disp_hi_charge.highlight_pixels(
                        cleanmask_hi, color="white", alpha=0.3, linewidth=2
                    )

                    sw.append(hillas_param_hi.width.value / hillas_param_hi.intensity)
                except HillasParameterizationError:
                    disp_hi_charge.clear_overlays()
                    disp_hi_charge.axes.figure.canvas.draw()
                    pass
                hillasdone = True

            charge_hi = image_hi_charge.sum()
            if verbose:
                print("   charge hi = {} pe".format(charge_hi))

        disp_hi_raw.image = image_hi_raw
        disp_hi_raw.set_limits_percent(95)
        disp_hi_raw.axes.figure.canvas.draw()

        disp_hi_charge.image = image_hi_charge
        disp_hi_charge.set_limits_percent(95)
        disp_hi_charge.axes.figure.canvas.draw()

        disp_hi_wf.image = hiw2[counter - hiistart]
        disp_hi_wf.set_limits_minmax(0, wf_hi_max)
        disp_hi_wf.axes.figure.canvas.draw()

        ax_hi_wf.set_title(
            "High gain, wave form (ADC), event {}, time {} ns".format(
                event.r0.event_id, counter
            )
        )
        ax_hi_plot_wf.plot(w[hi].sum(axis=0)[0:counter])
        ax_hi_plot_wf.figure.canvas.draw()

        counter += 1  # beurk...
        return [
            ax_hi_wf,
            ax_hi_plot_wf,
            ax_hi_raw,
            ax_hi_charge,
        ]

    try:
        frames = num_events * (winsize_start + winsize_end)
    except Exception:
        frames = None
    anim = FuncAnimation(
        fig, update, repeat=False, interval=delay, frames=frames, blit=(not save)
    )
    if save:
        anim.save(
            filename=path.replace(".fits.fz", "_highGain.mp4"),
            fps=fps,
            extra_args=["-vcodec", "libx264"],
        )
    plt.show()


def main():
    global sw
    animation()


if __name__ == "__main__":
    main()
