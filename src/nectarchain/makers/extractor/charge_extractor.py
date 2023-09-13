from ctapipe.image import ImageExtractor
import numpy as np
import logging
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import find_peaks

from numba import guvectorize

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

__all__ = ['gradient_extractor']

class gradient_extractor(ImageExtractor) : 

    
    fixed_window=np.uint8(16)
    height_peak=np.uint8(10)

    def __call__(self, waveforms, telid, selected_gain_channel,substract_ped = False):

        shape = waveforms.shape
        waveforms = waveforms.reshape(shape[0]*shape[1],shape[2])
        
        if substract_ped : 
            log.info('substracting pedestal')
            #calculate pedestal
            ped_mean = np.mean(waveforms[:,:16],axis = 1).reshape(shape[0]*shape[1],1)

            y = waveforms - (ped_mean)@(np.ones(1,shape[2])) # waveforms without pedestal
        else : 
            log.info('do not substract pedestal')
            y = waveforms

        waveforms.reshape(shape[0],shape[1],shape[2])

        peak_time, charge_integral, charge_sum, charge_integral_fixed_window, charge_sum_fixed_window = extract_charge(waveforms, self.height_peak, self.fixed_window)

        return peak_time, charge_integral


@guvectorize(
    [
        '(np.uint16[:], np.uint8, np.uint8, np.uint16[:])',
    ],
    "(n,p,s),(),()->(n,p)",
    nopython=True,
    cache=True,
)
def extract_charge(y,height_peak,fixed_window) : 
    x = np.linspace(0, len(y), len(y))
    xi = np.linspace(0, len(y), 251)
    ius = InterpolatedUnivariateSpline(x, y)
    yi = ius(xi)
    peaks, _ = find_peaks(yi, height=height_peak)
    # find the peak
    if len(peaks)>0:
        # find the max peak
        # max_peak = max(yi[peaks])
        max_peak_index = np.argmax(yi[peaks], axis=0)
        # Check if there is not a peak but a plateaux
        # 1. divide for the maximum to round to the first digit
        # 2. create a new array with normalized and rounded values, yi_rounded
        yi_rounded = np.around(yi[peaks]/max(yi[peaks]),1)
        maxima_peak_index = np.argwhere(yi_rounded == np.amax(yi_rounded))
        if len(maxima_peak_index)>1:
            # saturated event
            max_peak_index = int(np.median(maxima_peak_index))
            # width_peak = 20
        if (xi[peaks[max_peak_index]]>20)&(xi[peaks[max_peak_index]]<40):                                                                     # Search the adaptive integration window
            # calculate total gradients (not used, only for plot)
            yi_grad_tot = np.gradient(yi, 1)
            maxposition = peaks[max_peak_index]
            # calcualte grandients starting from the max peak and going to the left to find the left margin of the window
            yi_left = yi[:maxposition]
            yi_grad_left = np.gradient(yi_left[::-1], 0.9)
            change_grad_pos_left = (np.where(yi_grad_left[:-1] * yi_grad_left[1:] < 0 )[0] +1)[0]
            # calcualte grandients starting from the max peak and going to the right to find the right margin of the window
            yi_right = yi[maxposition:]
            yi_grad_right = np.gradient(yi_right, 0.5)
            change_grad_pos_right = (np.where(yi_grad_right[:-1] * yi_grad_right[1:] < 0 )[0] +1)[0]
            charge_integral = ius.integral(xi[peaks[max_peak_index]-(change_grad_pos_left)], xi[peaks[max_peak_index]+ change_grad_pos_right])
            charge_sum = yi[(peaks[max_peak_index]-(change_grad_pos_left)):peaks[max_peak_index]+change_grad_pos_right].sum()/(change_grad_pos_left+change_grad_pos_right)    # simple sum integration
            adaptive_window = change_grad_pos_right + change_grad_pos_left
            window_right = (fixed_window-6)/2
            window_left = (fixed_window-6)/2 + 6
            charge_integral_fixed_window = ius.integral(xi[peaks[max_peak_index]-(window_left)], xi[peaks[max_peak_index]+window_right])
            charge_sum_fixed_window = yi[(peaks[max_peak_index]-(window_left)):peaks[max_peak_index]+window_right].sum()/(fixed_window)    # simple sum integration
    else:
        log.info('No peak found, maybe it is a pedestal or noisy run!')
    return adaptive_window, charge_integral, charge_sum, charge_integral_fixed_window, charge_sum_fixed_window