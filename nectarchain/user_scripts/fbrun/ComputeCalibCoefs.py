
import sys
from matplotlib import pyplot as plt
import os
import glob
import h5py
import numpy as np


from ctapipe_io_nectarcam import NectarCAMEventSource
from ctapipe_io_nectarcam import constants

plt.rcParams['figure.figsize'] = [13, 9]

################################################################################

FF_LIST_VALS = (0,2,32)

NPREMAX = 4
NPOSTMAX = 10

# Bpx because not in the run -> 1
# Bpx because gain has strange value -> 2
GAINCUT_LOW = 50
GAINCUT_HIGH = 66
# Bpx because HG/LG has strange values -> 4
HGLGCUT_LOW = 12
HGLGCUT_HIGH = 16
# Bpx because charge ratio in pe is too different from one -> 8
QRATIO_DEV = 0.1

################################################################################


class ncam_wfs:

    def __init__(self):
        self.wfs_hg = []
        self.wfs_lg = []
        self.wfs_evttime = []
        self.wfs_triggertype = []
        self.wfs_evtid = []
        self.pixels_ids = []
        self.trig_pattern = []
        self.size = 0
        self.npixels = 0
        self.geometry = 0

    # ff_ttypes = trigger type of the FF events
    def load_wfs(self, nevents, file_name, ped_only=0, ff_only=0, ff_ttypes=0):
        if nevents == 0:
            inputfile_reader = NectarCAMEventSource(input_url=file_name)
            for _ in enumerate(inputfile_reader):
                nevents += 1
        inputfile_reader = NectarCAMEventSource(input_url=file_name, max_events=nevents*10)

        print("N events request = ", nevents)

        self.size = 0  # nevents
        npix = inputfile_reader.camera_config.num_pixels
        self.npixels = npix

        print("N pixels = ", npix)

        # self.wfs_hg = np.zeros((nevents,1855,60))
        # self.wfs_lg = np.zeros((nevents,1855,60))
        self.wfs_evttime = np.zeros(nevents)
        self.wfs_triggertype = np.zeros(nevents)
        # self.wfs_evtid = np.zeros(nevents)
        self.trig_pattern = np.zeros((nevents, constants.N_PIXELS))
        self.pixels_ids = inputfile_reader.camera_config.expected_pixels_id
        self.geometry = inputfile_reader.subarray.tel[0].camera

        self.wfs_evtid = []
        waveforms_hg = []
        waveforms_lg = []

        for i, event in enumerate(inputfile_reader):
            if self.size >= nevents:
                break
            if i % 100 == 0:
                print(i, self.size)
            # print(i,ff_only, event.trigger.event_type.value)

            # Only keep pedestal events
            if ped_only and event.trigger.event_type.value != 2:
                continue

            # Only keep flasher events
            if ff_only and int(event.trigger.event_type.value) != ff_ttypes:
                continue

            self.wfs_evtid.append(event.index.event_id)
            self.wfs_evttime[self.size] = event.nectarcam.tel[0].evt.ucts_timestamp
            self.wfs_triggertype[self.size] = event.trigger.event_type.value

            trig_in = np.zeros((constants.N_PIXELS))
            for slice in range(4):
                trig_in = np.logical_or(trig_in, event.nectarcam.tel[0].evt.trigger_pattern[slice])
            self.trig_pattern[self.size] += trig_in

            # if ff_only and len(np.where(self.trig_pattern[self.size]>0)[0]) < 200: continue

            wfs_hg_tmp = np.zeros((constants.N_PIXELS, constants.N_SAMPLES))
            wfs_lg_tmp = np.zeros((constants.N_PIXELS, constants.N_SAMPLES))
            for pix in self.pixels_ids:  # range(self.npixels):
                wfs_lg_tmp[pix] = event.r0.tel[0].waveform[1, pix]
                wfs_hg_tmp[pix] = event.r0.tel[0].waveform[0, pix]

            waveforms_hg.append(wfs_hg_tmp)
            waveforms_lg.append(wfs_lg_tmp)
            self.size += 1

        self.wfs_hg = np.array(waveforms_hg)
        self.wfs_lg = np.array(waveforms_lg)
        self.wfs_evtid = np.array(self.wfs_evtid)

########################


def average_wfs(nevents, filename, ped_only, ff_only):
    wfs_ped = ncam_wfs()
    wfs_ped.load_wfs(nevents, filename, ped_only, ff_only)
    print(wfs_ped.wfs_triggertype)

    average_lg = np.zeros((constants.N_PIXELS, constants.N_SAMPLES))
    average_hg = np.zeros((constants.N_PIXELS, constants.N_SAMPLES))

    for pix in wfs_ped.pixels_ids:
        average_lg[pix] = sum(wfs_ped.wfs_lg[:, pix]*1./wfs_ped.size)
        average_hg[pix] = sum(wfs_ped.wfs_hg[:, pix]*1./wfs_ped.size)

    return average_lg, average_hg


def subtract_average(wfs, average_hg, average_lg):
    wfs_sub_hg = []
    wfs_sub_lg = []

    print('Subtracting average to waveforms')
    print(f'Test {wfs.wfs_hg.shape} to {average_hg.shape}')

    for i in range(wfs.size):
        if i % 100 == 0:
            print(i)
        wfs_sub_hg.append((wfs.wfs_hg[i, :, :]-average_hg[:]))
        wfs_sub_lg.append((wfs.wfs_lg[i, :, :]-average_lg[:]))

    wfs_sub_hg = np.array(wfs_sub_hg)
    wfs_sub_lg = np.array(wfs_sub_lg)

    return wfs_sub_hg, wfs_sub_lg


def compute_charges(wfs_sub_hg, wfs_sub_lg, n_premax, n_postmax):
    charges_hg = []
    charges_lg = []
    ws = n_premax
    we = n_postmax
    print(f"Compute charges, from max in HG, params : n_premax = {ws}, n_postmax = {we}")
    for ev in range(1, len(wfs_sub_hg)):
        if ev % 100 == 0:
            print(ev)
        tmaxs = wfs_sub_hg[ev].argmax(axis=1)
        charges_hg.append(np.array([wfs_sub_hg[ev, ii, max(0, pmax-ws):pmax+we].sum(axis=0) for ii, pmax in enumerate(tmaxs)]))
        charges_lg.append(np.array([wfs_sub_lg[ev, ii, max(0, pmax-ws):pmax+we].sum(axis=0) for ii, pmax in enumerate(tmaxs)]))

    return np.array(charges_hg), np.array(charges_lg)


def findrun(run_number):
    basepath = os.environ['NECTARCAMDATA']
    list = glob.glob(basepath+'**/*'+str(run_number)+'*.fits.fz', recursive=True)
    fullnamesplit = list[0].split('.')
    print(f"Found file with name starting as: {fullnamesplit[0]}{fullnamesplit[1]}")
    return fullnamesplit[0]+'.'+fullnamesplit[1]+'.000[0-9].fits.fz'


def findrun_firstfile(run_number):
    basepath = os.environ['NECTARCAMDATA']
    list = glob.glob(basepath+'**/*'+str(run_number)+'*.fits.fz', recursive=True)
    fullnamesplit = list[0].split('.')
    print(f"Found file with name starting as: {fullnamesplit[0]}{fullnamesplit[1]}")
    return fullnamesplit[0]+'.'+fullnamesplit[1]+'.0000.fits.fz'


if __name__ == "__main__":

    # Open file and load waveforms
    max_events = int(sys.argv[1])  # 100
    run_number = sys.argv[2]
    ped_run_number = sys.argv[3]
    gain_run_number = sys.argv[4]
    ff_ttypes = int(sys.argv[5])

    if not ff_ttypes in FF_LIST_VALS:
        raise ValueError("Possible FF trigger types are: "+str(FF_LIST_VALS))

    filename = findrun(run_number)  # 3105)

    # Load ncam_waveforms
    wfs = ncam_wfs()
    wfs.load_wfs(max_events, filename, 0, 1, ff_ttypes)

    # Compute Average Ped to subtract
    filename_ped = findrun(ped_run_number)  # filename/#3107
    average_lg, average_hg = average_wfs(1000, filename_ped, 1, 0)
    wfs_sub_hg, wfs_sub_lg = subtract_average(wfs, average_hg, average_lg)

    # Load gain values
    h5f = h5py.File('gains_run_{}.h5'.format(gain_run_number), 'r')
    gains = h5f['gains'][:]

    # Compute charges
    charges_hg, charges_lg = compute_charges(wfs_sub_hg, wfs_sub_lg, NPREMAX, NPOSTMAX)
    print("Shapes: ", charges_hg.shape, charges_lg.shape)
    # HG/LG ratio
    hglg = charges_hg/charges_lg
    hglgm = hglg.mean(0)  # np.ma.where(bpx_tot>0,hglg.mean(0),0)

    # Compute charges in pe
    charges_hg_pe = np.array([charges_hg[:, ii] / gains[ii] for ii in range(constants.N_PIXELS)])
    charges_lg_pe = np.array([charges_lg[:, ii] / gains[ii] * hglgm[ii] for ii in range(constants.N_PIXELS)])
    mean_q_hg_pe = charges_hg_pe.mean(1)
    mean_q_lg_pe = charges_lg_pe.mean(1)
    charges_hg_pe = charges_hg_pe.T
    charges_lg_pe = charges_lg_pe.T

    ########
    # Compute broken/malfunctioning pixels from various estimates
    bpx_tot = np.ones((constants.N_PIXELS))
    bpx_flag = np.zeros((constants.N_PIXELS), dtype=int)

    # Bpx because not in the run
    participating_pix = np.zeros((constants.N_PIXELS))
    for pix in wfs.pixels_ids:
        participating_pix[pix] = 1

    bpx_daq = np.where(participating_pix == 0)

    print(bpx_daq)
    for ii in bpx_daq:
        bpx_tot[ii] = 0
        bpx_flag[ii] = bpx_flag[ii] | (1 << 0)

    # Bpx because gain has strange value
    bpx_gains_m = np.where(gains < GAINCUT_LOW)
    bpx_gains_p = np.where(gains > GAINCUT_HIGH)
    for ii in bpx_gains_m:
        bpx_tot[ii] = 0
        bpx_flag[ii] = bpx_flag[ii] | (1 << 1)
    for ii in bpx_gains_p:
        bpx_tot[ii] = 0
        bpx_flag[ii] = bpx_flag[ii] | (1 << 1)

    # Bpx because HG/LG has strange values
    bpx_hglg_m = np.where(hglgm < HGLGCUT_LOW)
    bpx_hglg_p = np.where(hglgm > HGLGCUT_HIGH)
    for ii in bpx_hglg_m:
        bpx_tot[ii] = 0
        bpx_flag[ii] = bpx_flag[ii] | (1 << 2)
    for ii in bpx_hglg_p:
        bpx_tot[ii] = 0
        bpx_flag[ii] = bpx_flag[ii] | (1 << 2)

    # Bpx because charge ratio is too different from one
    bpx_qratio = np.where(abs(charges_hg_pe.mean(0)/charges_lg_pe.mean(0)-1) > QRATIO_DEV)
    for ii in bpx_qratio:
        bpx_tot[ii] = 0
        bpx_flag[ii] = bpx_flag[ii] | (1 << 3)

    ############################################################################
    # Compute FF coefs, taking bpx into account
    masked_charges_hg_pe = np.ma.array([np.ma.masked_where(bpx_tot == 0, charges_hg_pe[ii]) for ii in range(charges_hg_pe.shape[0])])  # np.ma.where(bpx_tot>0,charges_hg_pe,0)
    masked_charges_lg_pe = np.ma.array([np.ma.masked_where(bpx_tot == 0, charges_lg_pe[ii]) for ii in range(charges_lg_pe.shape[0])])  # np.ma.where(bpx_tot>0,charges_lg_pe,0)

    masked_mean_hg_pe = np.ma.mean(masked_charges_hg_pe, axis=1)
    masked_mean_lg_pe = np.ma.mean(masked_charges_lg_pe, axis=1)

    q_over_mean_hg = np.array([masked_charges_hg_pe[ii]/masked_mean_hg_pe[ii] for ii in range(len(masked_charges_hg_pe))])
    masked_q_over_mean_hg = np.ma.array([np.ma.masked_where(bpx_tot == 0, q_over_mean_hg[ii]) for ii in range(q_over_mean_hg.shape[0])])

    ff_coef = np.ma.mean(masked_q_over_mean_hg, axis=0)
    ff_calib_coef = np.ma.divide(1., ff_coef)

    # ff_coef=np.array([ masked_charges_hg_pe[ii]/masked_mean_hg_pe[ii] for ii in range(len(masked_charges_hg_pe))]).mean(0)
    # ff_coef_lg=np.array([ masked_charges_lg_pe[ii]/masked_mean_lg_pe[ii] for ii in range(len(masked_charges_lg_pe))]).mean(0)
    # ff_calib_coef=np.ma.where(bpx_tot>0,1./ff_coef,1)

    # Write file
    h5calibparams = h5py.File('calibparams_run{}_pedrun{}_gainrun{}.h5'.format(run_number, ped_run_number, gain_run_number), 'w')
    h5calibparams.create_dataset('hglg', data=hglgm)
    h5calibparams.create_dataset('ped_hg', data=average_hg)
    h5calibparams.create_dataset('ped_lg', data=average_lg)
    h5calibparams.create_dataset('bpx', data=bpx_tot)
    h5calibparams.create_dataset('bpx_flag', data=bpx_flag)
    h5calibparams.create_dataset('ff_calib_coefs', data=ff_calib_coef)
    h5calibparams.create_dataset('gains', data=gains)
    h5calibparams.create_dataset('mean_q_hg', data=mean_q_hg_pe)
    h5calibparams.create_dataset('mean_q_lg', data=mean_q_lg_pe)
    h5calibparams.close()
