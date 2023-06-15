import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass, field
from ctapipe.visualization import CameraDisplay

import matplotlib
matplotlib.use('TkAgg')
from Utils import GetCamera

from IPython import embed

class CalibInfo:
    def __init__(self):
        self.tel = dict()

class TimedInfo:
    def __init__(self,startTime = None, endTime = None):
        self.startTime = startTime
        self.endTime = endTime

class PedestalInfo(TimedInfo):
    def __init__(self,startTime = None, endTime = None):
        super().__init__(startTime,endTime)
        # self.startTime = None
        # self.endTime = None
        self.pos = None
        self.width = None
        self.min = None
        self.max = None
        self.median = None
        self.nEvents = None
        self.datas = None
        # put the list of info per axis ? (because this could be 2d or 3d)

## Add display function
    def ShowTraceInfo(self,pixid,showMinMax=True,entryId=None):

        if entryId is not None:
            suffix = f"\nEntry: {entryId}"
        else:
            suffix = None

        ped_shape = self.pos.shape
        if len(ped_shape) != 3:
            print(f"Can't show trace for pixel {pixid}")
            return
        
        try:
            fig, axs = plt.subplots(nrows=1,ncols=ped_shape[0],figsize=(ped_shape[0]*6,6))

            ## left : Show High Gain Trace, right : Show Low Gain Trace

            x = np.arange( len( self.pos[0,pixid]) ) ## get the number of sample
            color = 'tab:cyan'

            for chan in range(2):
                if chan == 0:
                    chan_str = "HG"
                elif chan==1:
                    chan_str = "LG"
                else:
                    chan_str = "UNKNOWN"
                
                pos = self.pos[chan,pixid]
                width = self.width[chan,pixid]
                err = width/np.sqrt(self.nEvents[chan,pixid])
                min = self.min[chan,pixid]
                max = self.max[chan,pixid]
                
                axs[chan].fill_between(x, pos-width,pos+width,color=color,alpha=0.5,label='Width')
                axs[chan].errorbar( x+0.5, pos, xerr = 0.5*np.ones(len(x)), yerr=err, label=f'Pedestal',color=color )
                if showMinMax:
                    axs[chan].plot(x,min,"",color='b',label="min/max")
                    axs[chan].plot(x,max,"",color='b',label="")


                axs[chan].set_title( f'Pixel {pixid} {chan_str} Average Traces{suffix if suffix is not None else ""}' )
                axs[chan].grid()
                axs[chan].set_xlim(x[0],x[-1])

                #axs[chan].legend()

        except AttributeError as err:
            print(f"Can't show trace for pixel {pixid} [{err}]")
        except IndexError as err:
            print(f"Can't show trace for pixel {pixid} [{err}]")

        plt.show()



    def ShowPedestal(self):

        try:    
            fig, axs = plt.subplots(nrows=2,ncols=2, figsize=(10,10))
            fig.subplots_adjust(right=1)
            ped_shape = self.pos.shape
            print(ped_shape)

            if len(ped_shape) == 3:
                image = self.pos[0].mean(axis=1) # now that I've taken the HG, there is 2 axis : 0 for the pixel id, 1 for the slices
            else:
                image = self.pos[0]

            cam_disp = CalibrationCameraDisplay(geometry=GetCamera(), cmap='turbo',image=image, ax=axs[0][0], allow_pick=True, title='High-Gain Average Pedestal Position')
            cam_disp.add_colorbar()
            cam_disp.set_function(self.ShowTraceInfo)

            if len(ped_shape) == 3:
                image = self.width[0].mean(axis=1) # now that I've taken the HG, there is 2 axis : 0 for the pixel id, 1 for the slices
            else:
                image = self.width[0]

            cam_disp = CalibrationCameraDisplay(geometry=GetCamera(), cmap='turbo',image=image, ax=axs[0][1], allow_pick=True, title='High-Gain Average Pedestal Width')
            cam_disp.add_colorbar()
            cam_disp.set_function(self.ShowTraceInfo)



            if len(ped_shape) == 3:
                image = self.pos[1].mean(axis=1) # now that I've taken the HG, there is 2 axis : 0 for the pixel id, 1 for the slices
            else:
                image = self.pos[1]

            cam_disp = CalibrationCameraDisplay(geometry=GetCamera(), cmap='turbo',image=image, ax=axs[1][0], allow_pick=True, title='Low-Gain Average Pedestal Position')
            cam_disp.add_colorbar()
            cam_disp.set_function(self.ShowTraceInfo)

            if len(ped_shape) == 3:
                image = self.width[1].mean(axis=1) # now that I've taken the HG, there is 2 axis : 0 for the pixel id, 1 for the slices
            else:
                image = self.width[1]

            cam_disp = CalibrationCameraDisplay(geometry=GetCamera(), cmap='turbo',image=image, ax=axs[1][1], allow_pick=True, title='Low-Gain Average Pedestal Width')
            cam_disp.add_colorbar()
            cam_disp.set_function(self.ShowTraceInfo)



        except AttributeError as err:
            print(err)
        except IndexError as err:
            print(err)

        plt.show()





class FlatFieldInfo(TimedInfo):
    def __init__(self,startTime = None, endTime = None):
        super().__init__(startTime,endTime)
        self.ff = None
        self.ff_width = None
        self.tom = None
        self.tom_width = None
        self.var_over_mean = None
        self.charge = None
        self.charge_width = None
        self.min = None
        self.max = None
        self.nEvents = None

















########################################################
#################### OLD STUFF #########################
########################################################



@dataclass(order=True)
class IntegratedRawData:
    charge: int = field( default = 0)
    t0: int = field(default=0, compare=False)


class CalibrationCameraDisplay(CameraDisplay):

    def set_function(self,func_name):
        self.clickfunc = func_name

    def on_pixel_clicked(self, pix_id):
        self.clickfunc(pix_id)


class RawDataCameraDisplay(CameraDisplay):
    def __init__(self,HG=True,wvf=None,*args, **kwargs):
        print("RawDataCamereaDisplay: __init__")
        super().__init__(*args,**kwargs)
        self.use_hg = HG
        self.waveform = wvf
        print("__init__ done !")

    def set_waveform(self,waveform):
        self.waveform = waveform

    def show_trace(self,use_hg, pixel_id):
        if self.waveform is None:
            return
        
        #channel_id = 0 if self.use_hg else 1
        channel_name = "HG" if use_hg else "LG"
        trace = self.waveform[ pixel_id ]

        fig, ax = plt.subplots()
        #np.arange(0,len(trace))
        ax.plot(trace)
        ax.set_title(f'Pixel: {pixel_id} {channel_name} trace')
        ax.set_xlabel('slice number')
        ax.set_ylabel('ADC count')
        plt.show()
            #def set_function(self,)
    def on_pixel_clicked(self,pixel_id):
        #print("HERE")
        self.show_trace(use_hg=self.use_hg, pixel_id=pixel_id)

class FlatFieldInfo:
    def __init__(self,camera):
        self.camera = camera
        self.ff = np.zeros( (2,camera.n_pixels) )
        self.datas = None

    def ShowHGFFDistribution(self,pixid):
        pass
    
    def ShowLGFFDistribution(self,pixid):
        pass








class PedestalInfoOld:
    def __init__(self,camera):
        self.camera = camera
        self.pos = np.zeros( (2,camera.n_pixels) )
        self.width = np.zeros( (2,camera.n_pixels) )
        self.datas = None
        self.t0pos = np.zeros( (2,camera.n_pixels) )
        self.t0width = np.zeros( (2,camera.n_pixels) )
        self.t0datas = None
        self.avg_wvf = None # average waveform
        self.err_wvf = None # error on the average
        self.std_wvf = None # std dev on each slice
        self.all_pos = np.zeros( (2,camera.n_pixels) ) # std dev on the total trace
        self.all_width = np.zeros( (2,camera.n_pixels) ) # std dev on the total trace
        self.all_pos_per_slice = np.zeros( (2,camera.n_pixels) ) # std dev on the total trace
        self.all_width_per_slice = np.zeros( (2,camera.n_pixels) ) # std dev on the total trace
        
    def ShowHGPedestalDistribution(self,pixid):
        fig, ax = plt.subplots()
        datas = self.datas[:,0,pixid]
        min_val = np.min(datas)
        max_val = np.max(datas)
        nbins = int(max_val - min_val + 1)
        ax.hist(self.datas[:,0,pixid],bins=nbins)
        ax.set_xlabel('ADC')
        ax.set_title(f'Pixel {pixid} High Gain Pedestal Distribution\n$\mu$: {np.mean(datas):.2f}, $\sigma$: {np.std(datas):.2f}')
        ax.grid()
        plt.show()
        

    def ShowLGPedestalDistribution(self,pixid):
        fig, ax = plt.subplots()
        datas = self.datas[:,1,pixid]
        min_val = np.min(datas)
        max_val = np.max(datas)
        nbins = int(max_val - min_val + 1)
        ax.hist(self.datas[:,1,pixid],bins=nbins)
        ax.set_xlabel('ADC')
        ax.set_title(f'Pixel {pixid} Low Gain Pedestal Distribution\n$\mu$: {np.mean(datas):.2f}, $\sigma$: {np.std(datas):.2f}')
        ax.grid()
        plt.show()

    def ShowHGPedestalT0Distribution(self,pixid):
        fig, ax = plt.subplots()
        datas = self.t0datas[:,0,pixid]
        min_val = np.min(datas)
        max_val = np.max(datas)
        nbins = int(max_val - min_val + 1)
        ax.hist(self.t0datas[:,0,pixid],bins=nbins)
        ax.set_xlabel('T0')
        ax.set_title(f'Pixel {pixid} High Gain T0 Pedestal Distribution\n$\mu$: {np.mean(datas):.2f}, $\sigma$: {np.std(datas):.2f}')
        ax.grid()
        plt.show()
        

    def ShowLGPedestalT0Distribution(self,pixid):
        fig, ax = plt.subplots()
        datas = self.t0datas[:,1,pixid]
        min_val = np.min(datas)
        max_val = np.max(datas)
        nbins = int(max_val - min_val + 1)
        ax.hist(self.t0datas[:,1,pixid],bins=nbins)
        ax.set_xlabel('ADC')
        ax.set_title(f'Pixel {pixid} Low Gain T0 Pedestal Distribution\n$\mu$: {np.mean(datas):.2f}, $\sigma$: {np.std(datas):.2f}')
        ax.grid()
        plt.show()

    def ShowHGTraceInfo(self,pixid):
        
        fig, axs = plt.subplots(nrows=1,ncols=2)
        #fig, axs = plt.subplots(1, 1, figsize=(15, 10))
        avg_wvf = self.avg_wvf[0,pixid,:]
        err_wvf = self.err_wvf[0,pixid,:]

        x = np.arange(len( avg_wvf) )
        color = 'tab:cyan'

        axs[0].plot( x, avg_wvf, label='Per slice',color=color )
        axs[0].fill_between(x,avg_wvf-err_wvf,avg_wvf+err_wvf,color=color)
        axs[0].plot( x, self.all_pos_per_slice[0,pixid,:], label = 'Over trace',color='tab:orange')
        axs[0].set_title( f'Pixel {pixid} HG Average Traces' )
        axs[0].grid()

        axs[1].plot( x, self.std_wvf[0,pixid,:], label='Per slice',color=color )
        axs[1].plot( x, self.all_width_per_slice[0,pixid,:], label = 'Over trace',color='tab:orange')
        axs[1].set_title( f'Pixel {pixid} HG Average Width Traces' )
        axs[1].grid()

        axs[0].set_xlim(x[0],x[-1])
        axs[1].set_xlim(x[0],x[-1])
        plt.show()       
        # self.avg_wvf = None # average waveform
        # self.err_wvf = None # error on the average
        # self.std_wvf = None # std dev on each slice
        # self.all_pos = np.zeros( (2,camera.n_pixels) ) # std dev on the total trace
        # self.all_width = np.zeros( (2,camera.n_pixels) ) # std dev on the total trace
        # self.all_pos_per_slice = np.zeros( (2,camera.n_pixels) ) # std dev on the total trace
        # self.all_width_per_slice = np.zeros( (2,camera.n_pixels) ) # std dev on the total trace
 

    def ShowLGTraceInfo(self,pixid):
        fig, axs = plt.subplots(nrows=1,ncols=2)
        #fig, axs = plt.subplots(1, 1, figsize=(15, 10))
        avg_wvf = self.avg_wvf[1,pixid,:]
        err_wvf = self.err_wvf[1,pixid,:]

        x = np.arange(len( avg_wvf) )
        color = 'tab:cyan'

        axs[0].plot( x, avg_wvf, label='Per slice',color=color )
        axs[0].fill_between(x,avg_wvf-err_wvf,avg_wvf+err_wvf,color=color)
        axs[0].plot( x, self.all_pos_per_slice[1,pixid,:], label = 'Over trace',color='tab:orange')
        axs[0].set_title( f'Pixel {pixid} LG Average Traces' )
        axs[0].grid()

        axs[1].plot( x, self.std_wvf[1,pixid,:], label='Per slice',color=color )
        axs[1].plot( x, self.all_width_per_slice[1,pixid,:], label = 'Over trace',color='tab:orange')
        axs[1].set_title( f'Pixel {pixid} LG Average Width Traces' )
        axs[1].grid()

        axs[0].set_xlim(x[0],x[-1])
        axs[1].set_xlim(x[0],x[-1])

        plt.show()       
        # fig, ax = plt.subplots()
        # avg_wvf = self.avg_wvf[1,pixid,:]
        # err_wvf = self.err_wvf[1,pixid,:]

        # x = np.arange(len( avg_wvf) )
        # color = 'tab:cyan'

        # ax.plot( x, avg_wvf, label='Average per slice',color=color )
        # ax.fill_between(x,avg_wvf-err_wvf,avg_wvf+err_wvf,color=color)
        # ax.plot( x, self.all_pos_per_slice[1,pixid,:], label = 'Average over trace',color='tab:orange')
        # ax.set_title( f'Pixel {pixid} LG Average Traces' )
        
        # ax.grid()
        # ax.legend()

        # plt.show()


    def DisplayTrace(self):

        expected_t0 = 28.24
        expected_t0width = 17.28
        
        fig1, ax1 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        fig1.subplots_adjust(right=1)
        image1 = self.width[0].copy()
        image1[ image1 == 0. ] = np.nan
        cam_disp1 = CalibrationCameraDisplay(geometry=self.camera, image=image1, ax=ax1, allow_pick=True, title='High-Gain Pedestal Width')
        cam_disp1.add_colorbar()
        cam_disp1.set_function(self.ShowHGTraceInfo)

        fig2, ax2 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        fig2.subplots_adjust(right=1)
        image2 = self.width[1].copy()
        image2[ image2 == 0. ] = np.nan
        cam_disp2 = CalibrationCameraDisplay(geometry=self.camera, image=image2, ax=ax2, allow_pick=True, title='Low-Gain Pedestal Width')
        cam_disp2.add_colorbar()
        cam_disp2.set_function(self.ShowLGTraceInfo)


        fig3, ax3 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig3.subplots_adjust(right=1)
        image3 = self.t0pos[0].copy()
        image3[ image3 == 0. ] = np.nan
        
        image3 = image3-expected_t0

        cam_disp3 = CalibrationCameraDisplay(geometry=self.camera, image=image3, ax=ax3, allow_pick=True, title='High-Gain Pedestal T0 Position - Expectation',cmap='coolwarm') 
        cam_disp3.add_colorbar()
        cam_disp3.set_function(self.ShowHGTraceInfo)
        cam_disp3.set_limits_minmax(-3,3)

        fig4, ax4 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig4.subplots_adjust(right=1)
        image4 = self.t0width[0].copy()
        image4[ image4 == 0. ] = np.nan
        
        image4 = image4-expected_t0width

        cam_disp4 = CalibrationCameraDisplay(geometry=self.camera, image=image4, ax=ax4, allow_pick=True, title='High-Gain Pedestal T0 Standard Deviation - Expectation',cmap='coolwarm') 
        cam_disp4.add_colorbar()
        cam_disp4.set_function(self.ShowHGTraceInfo)
        cam_disp4.set_limits_minmax(-3,3)

        fig5, ax5 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig5.subplots_adjust(right=1)
        image5 = self.t0pos[1].copy()
        image5[ image5 == 0. ] = np.nan

        image5 = image5-expected_t0

        cam_disp5 = CalibrationCameraDisplay(geometry=self.camera, image=image5, ax=ax5, allow_pick=True, title='Low-Gain Pedestal T0 Position - Expectation',cmap='coolwarm') 
        cam_disp5.add_colorbar()
        cam_disp5.set_function(self.ShowLGTraceInfo)
        cam_disp5.set_limits_minmax(-3,3)
        

        fig6, ax6 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig6.subplots_adjust(right=1)
        image6 = self.t0width[1].copy()
        image6[ image6 == 0. ] = np.nan
        
        image6 = image6-expected_t0width

        cam_disp6 = CalibrationCameraDisplay(geometry=self.camera, image=image6, ax=ax6, allow_pick=True, title='Low-Gain Pedestal T0 Standard Deviation - Expectation',cmap='coolwarm') 
        cam_disp6.add_colorbar()
        cam_disp6.set_function(self.ShowLGTraceInfo)
        cam_disp6.set_limits_minmax(-3,3)


        
        fig7, ax7 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig7.subplots_adjust(right=1)
        image7 = ( ( np.sum(np.power(self.std_wvf[0,:,:],2.),axis=1) - 1 )/np.power( self.all_width[0,:],2. ) ) * (self.all_width[0,:] > 0.)

        cam_disp7 = CalibrationCameraDisplay(geometry=self.camera, image=image7, ax=ax7, allow_pick=True, title='High-Gain (Sum Slice Variance - Global Variance)/Global Variance')
        cam_disp7.add_colorbar()
        cam_disp7.set_function(self.ShowHGTraceInfo)



        fig8, ax8 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig8.subplots_adjust(right=1)
        image8 = ( ( np.sum(np.power(self.std_wvf[1,:,:],2.),axis=1) - 1 )/np.power( self.all_width[1,:],2. ) ) * (self.all_width[1,:] > 0.)

        cam_disp8 = CalibrationCameraDisplay(geometry=self.camera, image=image8, ax=ax8, allow_pick=True, title='Low-Gain (Sum Slice Variance - Global Variance)/Global Variance')
        cam_disp8.add_colorbar()
        cam_disp8.set_function(self.ShowLGTraceInfo)


        




        # self.camera = camera
        # self.pos = np.zeros( (2,camera.n_pixels) )
        # self.width = np.zeros( (2,camera.n_pixels) )
        # self.datas = None
        # self.t0pos = np.zeros( (2,camera.n_pixels) )
        # self.t0width = np.zeros( (2,camera.n_pixels) )
        # self.t0datas = None
        # self.avg_wvf = None # average waveform
        # self.err_wvf = None # error on the average
        # self.std_wvf = None # std dev on each slice
        # self.all_pos = np.zeros( (2,camera.n_pixels) ) # std dev on the total trace
        # self.all_width = np.zeros( (2,camera.n_pixels) ) # std dev on the total trace
        # self.all_pos_per_slice = np.zeros( (2,camera.n_pixels) ) # std dev on the total trace
        # self.all_width_per_slice = np.zeros( (2,camera.n_pixels) ) # std dev on the total trace
  


        plt.show()


    def Display(self):
        
        fig1, ax1 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig1.subplots_adjust(right=1)
        image1 = self.pos[0].copy()
        image1[ image1 == 0. ] = np.nan
        cam_disp1 = CalibrationCameraDisplay(geometry=self.camera, image=image1, ax=ax1, allow_pick=True, title='High-Gain Pedestal Position') 
        cam_disp1.add_colorbar()
        cam_disp1.set_function(self.ShowHGPedestalDistribution)

        fig2, ax2 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        fig2.subplots_adjust(right=1)

        image2 = self.width[0].copy()
        image2[ image2 == 0. ] = np.nan
        cam_disp2 = CalibrationCameraDisplay(geometry=self.camera, image=image2, ax=ax2, allow_pick=True, title='High-Gain Pedestal Width') 
        cam_disp2.add_colorbar()
        cam_disp2.set_function(self.ShowHGPedestalDistribution)

        fig3, ax3 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        fig3.subplots_adjust(right=1)

        image3 = self.pos[1].copy()
        image3[ image3 == 0. ] = np.nan
        cam_disp3 = CalibrationCameraDisplay(geometry=self.camera, image=image3, ax=ax3, allow_pick=True, title='Low-Gain Pedestal Position') 
        cam_disp3.add_colorbar()
        cam_disp3.set_function(self.ShowLGPedestalDistribution)

        fig4, ax4 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        fig4.subplots_adjust(right=1)   

        image4 = self.width[1].copy()
        image4[ image4 == 0. ] = np.nan
        cam_disp4 = CalibrationCameraDisplay(geometry=self.camera, image=image4, ax=ax4, allow_pick=True, title='Low-Gain Pedestal Width') 
        cam_disp4.add_colorbar()
        cam_disp4.set_function(self.ShowLGPedestalDistribution)


        fig5, ax5 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig5.subplots_adjust(right=1)
        image5 = self.t0pos[0].copy()
        image5[ image5 == 0. ] = np.nan
        cam_disp5 = CalibrationCameraDisplay(geometry=self.camera, image=image5, ax=ax5, allow_pick=True, title='High-Gain Pedestal T0 Position') 
        cam_disp5.add_colorbar()
        cam_disp5.set_function(self.ShowHGPedestalT0Distribution)

        fig6, ax6 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig6.subplots_adjust(right=1)
        image6 = self.t0width[0].copy()
        image6[ image6 == 0. ] = np.nan
        cam_disp6 = CalibrationCameraDisplay(geometry=self.camera, image=image6, ax=ax6, allow_pick=True, title='High-Gain Pedestal T0 Standard Deviation') 
        cam_disp6.add_colorbar()
        cam_disp6.set_function(self.ShowHGPedestalT0Distribution)
        
        fig7, ax7 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig7.subplots_adjust(right=1)
        image7 = self.t0pos[1].copy()
        image7[ image7 == 0. ] = np.nan
        cam_disp7 = CalibrationCameraDisplay(geometry=self.camera, image=image7, ax=ax7, allow_pick=True, title='Low-Gain Pedestal T0 Position') 
        cam_disp7.add_colorbar()
        cam_disp7.set_function(self.ShowLGPedestalT0Distribution)
        

        fig8, ax8 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig8.subplots_adjust(right=1)
        image8 = self.t0width[1].copy()
        image8[ image8 == 0. ] = np.nan
        cam_disp8 = CalibrationCameraDisplay(geometry=self.camera, image=image8, ax=ax8, allow_pick=True, title='Low-Gain Pedestal T0 Standard Deviation') 
        cam_disp8.add_colorbar()
        cam_disp8.set_function(self.ShowLGPedestalT0Distribution)

        plt.show()


    def DisplayTime(self):
        
        fig1, ax1 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig1.subplots_adjust(right=1)
        image1 = self.t0pos[0].copy()
        image1[ image1 == 0. ] = np.nan
        cam_disp1 = CalibrationCameraDisplay(geometry=self.camera, image=image1, ax=ax1, allow_pick=True, title='High-Gain Pedestal T0 Position') 
        cam_disp1.add_colorbar()
        cam_disp1.set_function(self.ShowHGPedestalT0Distribution)

        fig2, ax2 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig2.subplots_adjust(right=1)
        image2 = self.t0width[0].copy()
        image2[ image2 == 0. ] = np.nan
        cam_disp2 = CalibrationCameraDisplay(geometry=self.camera, image=image2, ax=ax2, allow_pick=True, title='High-Gain Pedestal T0 Standard Deviation') 
        cam_disp2.add_colorbar()
        cam_disp2.set_function(self.ShowHGPedestalT0Distribution)
        
        fig3, ax3 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig3.subplots_adjust(right=1)
        image3 = self.t0pos[1].copy()
        image3[ image3 == 0. ] = np.nan
        cam_disp3 = CalibrationCameraDisplay(geometry=self.camera, image=image3, ax=ax3, allow_pick=True, title='Low-Gain Pedestal T0 Position') 
        cam_disp3.add_colorbar()
        cam_disp3.set_function(self.ShowLGPedestalT0Distribution)
        

        fig4, ax4 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig4.subplots_adjust(right=1)
        image4 = self.t0width[1].copy()
        image4[ image4 == 0. ] = np.nan
        cam_disp4 = CalibrationCameraDisplay(geometry=self.camera, image=image4, ax=ax4, allow_pick=True, title='Low-Gain Pedestal T0 Standard Deviation') 
        cam_disp4.add_colorbar()
        cam_disp4.set_function(self.ShowLGPedestalT0Distribution)
        
        expected_t0 = 28.24
        expected_t0width = 17.28
        
        fig5, ax5 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig5.subplots_adjust(right=1)
        image5 = self.t0pos[0].copy()
        image5[ image5 == 0. ] = np.nan
        
        image5 = image5-expected_t0

        cam_disp5 = CalibrationCameraDisplay(geometry=self.camera, image=image5, ax=ax5, allow_pick=True, title='High-Gain Pedestal T0 Position - Expectation',cmap='coolwarm') 
        cam_disp5.add_colorbar()
        cam_disp5.set_function(self.ShowHGPedestalT0Distribution)
        cam_disp5.set_limits_minmax(-3,3)

        fig6, ax6 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig6.subplots_adjust(right=1)
        image6 = self.t0width[0].copy()
        image6[ image6 == 0. ] = np.nan
        
        image6 = image6-expected_t0width

        cam_disp6 = CalibrationCameraDisplay(geometry=self.camera, image=image6, ax=ax6, allow_pick=True, title='High-Gain Pedestal T0 Standard Deviation - Expectation',cmap='coolwarm') 
        cam_disp6.add_colorbar()
        cam_disp6.set_function(self.ShowHGPedestalT0Distribution)
        cam_disp6.set_limits_minmax(-3,3)

        fig7, ax7 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig7.subplots_adjust(right=1)
        image7 = self.t0pos[1].copy()
        image7[ image7 == 0. ] = np.nan

        image7 = image7-expected_t0

        cam_disp7 = CalibrationCameraDisplay(geometry=self.camera, image=image7, ax=ax7, allow_pick=True, title='Low-Gain Pedestal T0 Position - Expectation',cmap='coolwarm') 
        cam_disp7.add_colorbar()
        cam_disp7.set_function(self.ShowLGPedestalT0Distribution)
        cam_disp7.set_limits_minmax(-3,3)
        

        fig8, ax8 = plt.subplots(nrows=1,ncols=1, figsize=(6,6))

        fig8.subplots_adjust(right=1)
        image8 = self.t0width[1].copy()
        image8[ image8 == 0. ] = np.nan
        
        image8 = image8-expected_t0width

        cam_disp8 = CalibrationCameraDisplay(geometry=self.camera, image=image8, ax=ax8, allow_pick=True, title='Low-Gain Pedestal T0 Standard Deviation - Expectation',cmap='coolwarm') 
        cam_disp8.add_colorbar()
        cam_disp8.set_function(self.ShowLGPedestalT0Distribution)
        cam_disp8.set_limits_minmax(-3,3)

        plt.show()
