try:
    import argparse
    import copy

    # from multiprocessing import Pool
    import pickle
    import sys
    import time

    # from multiprocessing.dummy import Pool as ThreadPool
    from multiprocessing import Pool

    import numpy as np
    import numpy.ma as ma
    from ctapipe.containers import EventType
    from ctapipe.visualization import CameraDisplay
    from DataUtils import CountEventTypes, GetLongRunTimeEdges, GetTotalEventCounts
    from FileHandler import DataReader, GetNectarCamEvents
    from FitUtils import JPT2FitFunction
    from iminuit import Minuit
    from IPython import embed
    from matplotlib import pyplot as plt
    from scipy.stats import norm, poisson
    from Stats import CameraSampleStats, CameraStats
    from tqdm import tqdm
    from Utils import (
        ConvertTitleToName,
        CustomFormatter,
        GetCamera,
        GetDefaultDataPath,
        IntegrationMethod,
        SignalIntegration,
        save_simple_data,
    )
except ImportError as e:
    print(e)
    raise SystemExit


# %matplotlib tk
# ma.array(t0max[0],mask=css.get_lowcount_mask()[0,:,0])
class FitResult:
    def __init__(self, res, params):
        self.minuit_result = res
        self.fit_parameters = params


class SPEHistograms:
    def __init__(self, bins, vals, x_edges):
        self.bins = bins
        self.vals = vals
        self.x_edges = x_edges


def split_array(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def save_data(datas, fileName):
    """Save info in a file using pickle"""
    with open(fileName, "wb") as file:
        pickle.dump(datas, file)


def FitPart(pixels, charges, sigpeds, peds):
    results = list()
    for p, c, s, ped in tqdm(zip(pixels, charges, sigpeds, peds)):
        res = FitPixel(c, s, ped)
        if res is not None:
            results.append((p,) + res)
    return results


def CreateChargeHistograms(hicharges):
    pix_datas = hicharges[~hicharges.mask]
    if len(pix_datas) == 0:
        return None, None, None
    # bins = np.linspace( np.min(pix_datas)-0.5,np.max(pix_datas)+0.5,int(np.max(pix_datas)-np.min(pix_datas)+2))
    bins = np.linspace(
        pix_datas.min() - 0.5,
        pix_datas.max() + 0.5,
        int(pix_datas.max() - pix_datas.min() + 2),
    )
    vals, x_edges = np.histogram(
        pix_datas.compressed(), bins=bins
    )  # warning np histogram does not take into account maskarray
    return bins, vals, x_edges


def FitPixel(hicharges, sigped=None, ped=None):
    # hicharges : mask array !
    start = time.time()
    # pix_datas = hicharges[ ~hicharges.mask ]
    # bins = np.linspace( np.min(pix_datas)-0.5,np.max(pix_datas)+0.5,int(np.max(pix_datas)-np.min(pix_datas)+2))
    # bins = np.linspace( pix_datas.min()-0.5, pix_datas.max()+0.5, int(pix_datas.max()-pix_datas.min()+2) )
    # vals,x_edges = np.histogram(pix_datas.compressed(),bins=bins) # warning np histogram does not take into account maskarray
    bins, vals, x_edges = CreateChargeHistograms(hicharges)
    if bins is None or len(bins) <= 100:
        # print(f"Pixel: {pix} --> Too few bins... bad pixel ? --> SKIP !")
        return  # continue

    jpt2 = JPT2FitFunction(x_edges, vals)
    amplitude = float(np.sum(vals))
    if ped is None:
        hiPed = x_edges[np.argmax(vals)] + 0.5
    else:
        hiPed = ped

    if sigped is None:
        sigmaHiPed = 14.0
    else:
        sigmaHiPed = sigped

    ns = 1.0
    meanillu = 1.0
    gain = 65.0
    sigmaGain = 0.45 * gain
    start_parameters = [amplitude, hiPed, sigmaHiPed, ns, meanillu, gain, sigmaGain]

    m = Minuit(
        jpt2.Minus2LogLikelihood,
        start_parameters,
        name=("Norm", "Ped", "SigmaPed", "Ns", "Illu", "Gain", "SigmaGain"),
    )

    m.errors["Norm"] = 0.1 * amplitude
    m.errors["Ped"] = 1.0 * sigmaHiPed
    m.errors["SigmaPed"] = 0.2 * sigmaHiPed
    m.errors["Ns"] = 0.1
    m.errors["Illu"] = 0.3
    m.errors["Gain"] = 0.3 * gain
    m.errors["SigmaGain"] = 0.3 * sigmaGain

    m.limits["Norm"] = (0.0, None)
    m.limits["Illu"] = (0.0, 8.0)
    m.limits["Gain"] = (10.0, 400)
    m.limits["SigmaGain"] = (10.0, None)
    m.limits["Ns"] = (0.0, None)
    m.limits["SigmaPed"] = (0.0, None)

    try:
        min_result = m.migrad(20000)
        # if min_result.valid:
        # print("HERE 1")
        # print(min_result.valid)
        # print(min_result)
        # print("HERE 2")
        # min_result.hesse()
        # print("HERE 3")
        # print(min_result)
        # print("HERE 4")
        # min_result.minos()
        # print("HERE 5")
        # print(min_result)
        # print("HERE 6")

        fit_params = m.params

        end = time.time()
        fitTime = end - start
        # pixelFitTime[pix] = end-start
        # pixelFitResult[pix] = FitResult( copy.deepcopy(min_result), copy.deepcopy(fit_params) )
        return FitResult(copy.deepcopy(min_result), copy.deepcopy(fit_params)), fitTime

    except ValueError as err:
        # print(f"Pixel: {pix} Error: {err}")
        pass


def average_minos_error(merror):
    return 0.5 * (np.abs(merror.lower) + np.abs(merror.upper))


def read_t0_file(t0file):
    run_t0 = dict()
    with open(t0file) as f:
        lines = f.readlines()
        for line in lines:
            l = line.split()
            try:
                run = int(l[0])
                t0 = round(float(l[1]))
                run_t0[run] = t0
            except Exception as err:
                print(err)
    return run_t0


def DoSPEFFFit(arglist):
    p = argparse.ArgumentParser(
        description="Perform SPE Fit for FF data",
        epilog="examples:\n" "\t python %(prog)s --run 3750  \n",
        formatter_class=CustomFormatter,
    )

    p.add_argument("--run", dest="run", type=int, help="Run number")
    p.add_argument(
        "--data-path",
        dest="dataPath",
        type=str,
        default=GetDefaultDataPath(),
        help="Path to the rawdata directory. The program will recursively search in all directory for matching rawdata",
    )
    p.add_argument(
        "--njobs", dest="njobs", type=int, default=1, help="Number of CPU to use"
    )
    p.add_argument(
        "--batch",
        dest="batch",
        action="store_true",
        help="run in batch, no plots will be displayed",
    )
    p.add_argument(
        "--t0-file",
        dest="t0file",
        type=str,
        default=None,
        help="file that contain the T0 position to use for each pixel",
    )

    args = p.parse_args(arglist)

    if args.run is None:
        p.print_help()
        return -1

    run = args.run
    path = args.dataPath

    # path = '/Users/vm273425/Programs/NectarCAM/data/'

    data = DataReader(run, path)
    ok = data.Connect("trigger", "r0", "mon")
    if not ok:
        data = GetNectarCamEvents(run, path, applycalib=False)

    nEvents = GetTotalEventCounts(run, path)

    css = CameraSampleStats()

    default_mask = None

    # pixels_in_config = data.camera_config.pixel_id_map
    # embed()
    # pixels_in_config =  data.nectarcam.tel[0].svc.pixel_ids
    # pixels_in_config = np.sort(pixels_in_config)
    # print(pixels_in_config)
    pixels_in_config = None
    # pixel_ids
    pix_t0s = None if args.t0file is None else read_t0_file(args.t0file)

    if pix_t0s is None:
        counter = 0
        nUsedEntries = 40000
        try:
            for evt in tqdm(data, total=min(nEvents, nUsedEntries)):
                if pixels_in_config is None:
                    pixels_in_config = np.sort(
                        evt.nectarcam.tel[0].svc.pixel_ids.copy()
                    )

                if evt.trigger.event_type == EventType.FLATFIELD:
                    wvf = evt.r0.tel[0].waveform
                    hdw_fail = evt.mon.tel[0].pixel_status.hardware_failing_pixels

                    if default_mask is None:
                        default_mask = np.ones(wvf.shape, dtype=bool)

                    hdw_valid_slice = default_mask ^ hdw_fail[:, :, None]
                    css.add(wvf, validmask=hdw_valid_slice)

                    counter += 1
                    if counter >= nUsedEntries:
                        break
        except Exception as err:
            print(err)
        mean_waveform = css.mean

        fig1 = plt.figure()
        pixs = [777, 747, 320, 380, 123, 1727, 427, 74]
        for p in pixs:
            plt.plot(mean_waveform[0, p, :], label=f"{p}")
        plt.grid()
        plt.legend()

        fig2 = plt.figure()
        # pixs = [777,747,320,380,123,1727,427,74]
        pixs = [923, 483, 1573, 1751, 1491, 482, 720]
        for p in pixs:
            plt.plot(mean_waveform[0, p, :], label=f"{p}")
        plt.grid()
        plt.legend()

        t0max = np.argmax(mean_waveform, axis=2)

    else:
        t0max = np.zeros((2, 1855), dtype=int)
        for k, v in pix_t0s.items():
            t0max[0, k] = v
            t0max[1, k] = v

    fig3 = plt.figure()
    cam_tom_hg = CameraDisplay(
        geometry=GetCamera(),
        image=ma.array(t0max[0], mask=css.get_lowcount_mask()[0, :, 0]),
        cmap="coolwarm",
        title="High-Gain Mean ToM",
        allow_pick=True,
    )
    cam_tom_hg.highlight_pixels(range(1855), color="grey")
    cam_tom_hg.add_colorbar()
    # cam_tom_hg.show()

    fig4 = plt.figure()
    histdata = t0max[0]
    bins = np.linspace(-0.05, 50.05, 502)
    good_histdata = histdata[histdata > 0.0]
    _ = plt.hist(good_histdata, bins=bins)
    plt.xlim(np.min(good_histdata) - 1, np.max(good_histdata) + 1)
    plt.title("ToM Average High Gain Waveform")
    plt.grid()

    data = DataReader(run, path)
    ok = data.Connect("trigger", "r0", "mon")
    if not ok:
        data = GetNectarCamEvents(run, path, applycalib=False)

    nEvents = GetTotalEventCounts(run, path)

    camera = GetCamera()

    ped_stats = CameraStats()
    hicharges = list()
    badcharges = list()
    counter = 0

    # data.rewind()

    default_mask = None

    counter = 0
    try:
        for evt in tqdm(data, total=nEvents):
            if pixels_in_config is None:
                pixels_in_config = np.sort(evt.nectarcam.tel[0].svc.pixel_ids.copy())

            if evt.trigger.event_type == EventType.FLATFIELD:
                wvf = evt.r0.tel[0].waveform
                hdw_fail = evt.mon.tel[0].pixel_status.hardware_failing_pixels
                hdw_good = ~hdw_fail

                if default_mask is None:
                    default_mask = np.zeros(wvf.shape, dtype=bool)

                hdw_fail_slice = default_mask | hdw_fail[:, :, None]

                charges, times = SignalIntegration(
                    wvf,
                    exclusion_mask=hdw_fail_slice,
                    peakpositions=t0max,
                    method=IntegrationMethod.USERPEAK,
                    left_bound=5,
                    right_bound=11,
                    camera=camera,
                )

                ped = np.sum(evt.r0.tel[0].waveform[:, :, :16], axis=2)
                ped_stats.add(ped, validmask=hdw_good)

                hicharges.append(charges[0].copy())
                badcharges.append(hdw_fail[0].copy())

                counter += 1
    except Exception as err:
        print(err)
    #        if counter >=1000:
    #            break

    # hicharges = np.array(hicharges)
    hicharges = ma.array(hicharges, mask=badcharges)

    # print(f"DEBUG> {hicharges.dtype}")
    # print(f"DEBUG> {goodcharges.dtype}")

    fig5 = plt.figure()
    cam_pedw_hg = CameraDisplay(
        geometry=GetCamera(),
        image=ma.array(ped_stats.stddev[0], mask=ped_stats.get_lowcount_mask()[0]),
        cmap="coolwarm",
        title="High-Gain Mean Ped Width",
        allow_pick=True,
    )
    cam_pedw_hg.highlight_pixels(range(1855), color="grey")
    cam_pedw_hg.add_colorbar()
    # cam_pedw_hg.show()

    epedwidth = ped_stats.stddev[0]
    eped = ped_stats.mean[0]
    badpedwidth = ped_stats.get_lowcount_mask()[0]

    pixelFitResult = dict()
    pixelFitTime = dict()
    pixelHistograms = dict()

    print("Creating histograms")
    # for pix in tqdm(range(1855)):
    for pix in tqdm(pixels_in_config):
        b, v, x = CreateChargeHistograms(hicharges=hicharges[:, pix])
        pixelHistograms[pix] = SPEHistograms(bins=b, vals=v, x_edges=x)

    save_simple_data(pixelHistograms, f"run_{run}_SPEHistograms.pickle")

    use_multiprocess = args.njobs > 1
    simple_multiprocess = False
    # HiCharges = hicharges
    # BadPedWidth = badpedwidth
    # EPedWidth = epedwidth

    start_fit = time.time()
    print(f"Starting the fit at : {start_fit}")
    if use_multiprocess and simple_multiprocess:
        print("Using the fit by pixel method")
        pixlist = [pix for pix in pixels_in_config]

        with Pool(args.njobs) as p:
            datas = list()
            sigpeds = list()
            peds = list()
            for pix in pixlist:
                # pix_hicharges = hicharges[:,pix]
                datas.append(hicharges[:, pix])
                sigpeds.append(14.0 if badpedwidth[pix] else epedwidth[pix])
                peds.append(None if badpedwidth[pix] else eped[pix])

            print(len(datas))
            results = p.starmap(FitPixel, zip(datas, sigpeds, peds))

            p.close()
            p.join()
            print(len(results))

            for pix in pixlist:
                res = results[pix]
                if res is not None:
                    pixelFitResult[pix] = res[0]
                    pixelFitTime[pix] = res[1]
    elif use_multiprocess and not simple_multiprocess:
        print("Using the fit by part method")
        nPart = args.njobs

        npix2use = 1855
        part_pixels = list(split_array([pix for pix in range(npix2use)], nPart))

        part_charges = list(
            split_array([hicharges[:, pix] for pix in range(npix2use)], nPart)
        )
        part_sigpeds = list(
            split_array(
                [
                    14.0 if badpedwidth[pix] else epedwidth[pix]
                    for pix in range(npix2use)
                ],
                nPart,
            )
        )
        part_peds = list(
            split_array(
                [None if badpedwidth[pix] else eped[pix] for pix in range(npix2use)],
                nPart,
            )
        )

        # embed()
        # for part in part_pixels:
        #    pcharges
        #    for pix in part:
        with Pool(nPart) as p:
            results = p.starmap(
                FitPart, zip(part_pixels, part_charges, part_sigpeds, part_peds)
            )
            for rs in results:
                for res in rs:
                    if res is not None:
                        cur_pixel = res[0]
                        pixelFitResult[cur_pixel] = res[1]
                        pixelFitTime[cur_pixel] = res[2]

    else:
        print("Using the standard approach")
        for pix in tqdm(pixels_in_config):
            sigped = 14.0 if badpedwidth[pix] else epedwidth[pix]
            ped = None if badpedwidth[pix] else eped[pix]
            res = FitPixel(hicharges[:, pix], sigped, ped)
            if res is not None:
                pixelFitResult[pix] = res[0]
                pixelFitTime[pix] = res[1]
            # if pix>=100 :
            #    break

    end_fit = time.time()

    print(f"Finishing the fit at : {end_fit}")
    print(f"Time to do it : {end_fit-start_fit} s")

    print(len(pixelFitResult))

    gains = list()
    for p, f in pixelFitResult.items():
        fit_gain = f.fit_parameters["Gain"].value
        gains.append(fit_gain)

    cam_gain_values = np.zeros(1855)
    for p, f in pixelFitResult.items():
        fit_gain = f.fit_parameters["Gain"].value
        cam_gain_values[p] = fit_gain

    valid_fit = np.zeros(1855, dtype=bool)
    for p, f in pixelFitResult.items():
        valid_fit[p] = f.minuit_result.valid

    fig6 = plt.figure(figsize=(8, 8))
    cam_gain = CameraDisplay(
        geometry=GetCamera(),
        image=ma.array(cam_gain_values, mask=cam_gain_values < 40.0),
        cmap="turbo",
        title=f"High-Gain Gain (VIM Edition)\nRun: {run}",
        show_frame=False,
    )
    cam_gain.highlight_pixels(range(1855), color="grey")
    cam_gain.add_colorbar()
    cam_gain.show()
    # fig.savefig(f'run_{run}_gain_camera.png')
    fig6.savefig(f"run_{run}_FittedGain_VIMEdition.png")

    fig7 = plt.figure(figsize=(8, 8))
    mean_gain = np.mean(gains)
    std_gain = np.std(gains)
    _ = plt.hist(
        gains, bins=60, label=f"$\mu:$ {mean_gain:.2f}\n$\sigma:$ {std_gain:.2f}"
    )
    plt.title(f"Fitted Gain Distribution (Run: {run})")
    plt.xlabel("Gain (ADC per pe)")
    plt.grid()
    plt.legend()
    # fig.savefig(f'run_{run}_gain_distribution.png')
    fig7.savefig(f"run_{run}_FittedGainDistribution_VIMEdition.png")

    cam_illu_values = np.zeros(1855)
    for p, f in pixelFitResult.items():
        fit_illu = f.fit_parameters["Illu"].value
        cam_illu_values[p] = fit_illu

    fig8 = plt.figure(figsize=(8, 8))
    cam_illu = CameraDisplay(
        geometry=GetCamera(),
        image=ma.array(cam_illu_values, mask=~valid_fit),
        cmap="turbo",
        title=f"Mean Illumination (VIM Edition)\nRun: {run}",
        show_frame=False,
    )
    cam_illu.highlight_pixels(range(1855), color="grey")
    cam_illu.add_colorbar()
    cam_illu.show()
    fig8.savefig(f"run_{run}_FittedIllumination_VIMEdition.png")

    cam_ns_values = np.zeros(1855)
    for p, f in pixelFitResult.items():
        fit_ns = f.fit_parameters["Ns"].value
        cam_ns_values[p] = fit_ns

    fig9 = plt.figure(figsize=(8, 8))
    cam_ns = CameraDisplay(
        geometry=GetCamera(),
        image=ma.array(cam_ns_values, mask=~valid_fit),
        cmap="turbo",
        title=f"Ns Parameter (VIM Edition)\nRun: {run}",
        show_frame=False,
    )
    cam_ns.highlight_pixels(range(1855), color="grey")
    cam_ns.add_colorbar()
    cam_ns.show()

    cam_pedw_values = np.zeros(1855)
    for p, f in pixelFitResult.items():
        fit_pedw = f.fit_parameters["SigmaPed"].value
        cam_pedw_values[p] = fit_pedw

    fig9 = plt.figure(figsize=(8, 8))
    cam_pedw = CameraDisplay(
        geometry=GetCamera(),
        image=ma.array(cam_pedw_values, mask=~valid_fit),
        cmap="turbo",
        title=f"Pedestal Width (VIM Edition)\nRun: {run}",
        show_frame=False,
    )
    cam_pedw.highlight_pixels(range(1855), color="grey")
    cam_pedw.add_colorbar()
    cam_pedw.show()
    fig9.savefig(f"run_{run}_FittedPedestalWidth_VIMEdition.png")

    ## Save results :
    outdataname = f"run_{run}_FittedSPEResult.dat"
    outfitname = f"run_{run}_FittedSPEResult.pickle"

    with open(outdataname, "w") as outfile:
        outfile.write(
            f"pix Ped SigmaPed Ns Illu Gain SigmaGain ErrorPed ErrorSigmaPed ErrorNs ErrorIllu ErrorGain ErrorSigmaGain Valid FVal NFcn\n"
        )
        for p, f in pixelFitResult.items():
            # if f.minuit_result.valid:
            ped = f.fit_parameters["Ped"].value
            sigmaped = f.fit_parameters["SigmaPed"].value
            ns = f.fit_parameters["Ns"].value
            illu = f.fit_parameters["Illu"].value
            gain = f.fit_parameters["Gain"].value
            sigmagain = f.fit_parameters["SigmaGain"].value

            # err_ped = average_minos_error( f.merrors['Ped'] )
            # err_sigmaped = average_minos_error( f.merrors['SigmaPed'] )
            # err_ns = average_minos_error( f.merrors['Ns'] )
            # err_illu = average_minos_error( f.merrors["Illu"] )
            # err_gain = average_minos_error( f.merrors['Gain'] )
            # err_sigmagain = average_minos_error( f.merrors['SigmaGain'] )

            err_ped = f.minuit_result.errors["Ped"]
            err_sigmaped = f.minuit_result.errors["SigmaPed"]
            err_ns = f.minuit_result.errors["Ns"]
            err_illu = f.minuit_result.errors["Illu"]
            err_gain = f.minuit_result.errors["Gain"]
            err_sigmagain = f.minuit_result.errors["SigmaGain"]

            outfile.write(
                f"{p} {ped} {sigmaped} {ns} {illu} {gain} {sigmagain} {err_ped} {err_sigmaped} {err_ns} {err_illu} {err_gain} {err_sigmagain} {f.minuit_result.valid} {f.minuit_result.fval} {f.minuit_result.nfcn}\n"
            )

    save_simple_data(pixelFitResult, outfitname)

    # m.errors["Norm"] = 0.1*amplitude
    # m.errors["Ped"] = 1.* sigmaHiPed
    # m.errors["SigmaPed"] = 0.2*sigmaHiPed
    # m.errors["Ns"] = 0.1
    # m.errors["Illu"] = 0.3
    # m.errors["Gain"] = 0.3*gain
    # m.errors["SigmaGain"] = 0.3*sigmaGain

    if not args.batch:
        plt.show()


if __name__ == "__main__":
    DoSPEFFFit(sys.argv[1:])
