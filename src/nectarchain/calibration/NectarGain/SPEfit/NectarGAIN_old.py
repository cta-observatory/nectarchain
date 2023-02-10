import math
import numpy as np
from scipy import optimize, interpolate
from matplotlib import pyplot as plt
from scipy import signal
from scipy.special import gammainc
from iminuit import Minuit
import random




# Gain's class
class NectarSPEGain():
    
    def  __init__(self):
        self.histoPed = []
        self.chargePed = []
        self.histoSPE = []
        self.chargeSPE = []
        self.histoPedHHV = []
        self.chargePedHHV = []
        self.histoSPEHHV = []
        self.chargeSPEHHV = []
        self.dataType = ""
        ###### parameters model #########
        self.pedestalMean = 0
        self.pedestalWidth = 0
        self.pedestalMeanSPE = 0
        self.pedestalWidthSPE = 0
        self.Luminosity = 0
        self.pp = 0
        self.resolution = 0
        self.meanUp = 0
        self.n = 0
        self.pedestalMeanHHV = 0
        self.pedestalWidthHHV = 0
        self.pedestalMeanSPEHHV = 0
        self.pedestalWidthSPEHHV = 0
        self.LuminosityHHV = 0
        self.ppHHV = 0
        self.resolutionHHV = 0
        self.meanUpHHV = 0
        self.nHHV = 0
        self.gain = 0
        
        self.pedestalMeanUp = 0
        self.pedestalWidthUp = 0
        self.pedestalMeanSPEUp = 0
        self.pedestalWidthSPEUp = 0
        self.LuminosityUp = 0
        self.ppUp = 0
        self.resolutionUp = 0
        self.meanUpUp = 0
        self.nUp = 0
        self.pedestalMeanHHVUp = 0
        self.pedestalWidthHHVUp = 0
        self.pedestalMeanSPEHHVUp = 0
        self.pedestalWidthSPEHHVUp = 0
        self.LuminosityHHVUp = 0
        self.ppHHVUp = 0
        self.resolutionHHVUp = 0
        self.meanUpHHVUp = 0
        self.nHHVUp = 0
        self.gainUp = 0
        
        self.pedestalMeanLow = 0
        self.pedestalWidthLow = 0
        self.pedestalMeanSPELow = 0
        self.pedestalWidthSPELow = 0
        self.LuminosityLow = 0
        self.ppLow = 0
        self.resolutionLow = 0
        self.meanUpLow = 0
        self.nLow = 0
        self.pedestalMeanHHVLow = 0
        self.pedestalWidthHHVLow = 0
        self.pedestalMeanSPEHHVLow = 0
        self.pedestalWidthSPEHHVLow = 0
        self.LuminosityHHVLow = 0
        self.ppHHVLow = 0
        self.resolutionHHVLow = 0
        self.meanUpHHVLow = 0
        self.nHHVLow = 0
        self.gainLow = 0
        
    ####### data #######
    
# Fill histos as given by the user
    def FillDataHisto(self,chargeSPE,dataSPE,chargePed = 0,dataPed = 0,chargePedHHV=0,dataPedHHV=0,chargeSPEHHV=0,dataSPEHHV=0):
        self.histoPed = dataPed
        self.chargePed = chargePed
        self.histoSignal = dataSPE
        self.chargeSignal = chargeSPE
        self.histoPedHHV = dataPedHHV
        self.chargePedHHV = chargePedHHV
        self.histoSignalHHV = dataSPEHHV
        self.chargeSignalHHV = chargeSPEHHV
        self.dataType = "histo"

    ####### functions model #######

# Usefull fucntions for the fit
    def gaussian(self,x, mu, sig):
        return (1./(sig*np.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    def doubleGauss(self,x,sig1,mu2,sig2,p):
        return p *2 *self.gaussian(x, 0, sig1) + (1-p) * self.gaussian(x, mu2, sig2)
    
    def PMax(self,r):
        if (np.pi*r**2/(np.pi*r**2 + np.pi - 2*r**2 - 2) <= 1):
            return np.pi*r**2/(np.pi*r**2 + np.pi - 2*r**2 - 2)
        else:
            return 1
    
    def ax(self,p,res):
        return ((2/np.pi)*p**2-p/(res**2+1))
    
    def bx(self,p,mu2):
        return (np.sqrt(2/np.pi)*2*p*(1-p)*mu2)
    
    def cx(self,sig2,mu2,res,p):
        return (1-p)**2*mu2**2 - (1-p)*(sig2**2+mu2**2)/(res**2+1)
    
    def delta(self,p,res,sig2,mu2):
        return self.bx(p,mu2)*self.bx(p,mu2) - 4*self.ax(p,res)*self.cx(sig2,mu2,res,p)
    
    def ParamU(self,p,r):
        return ((8*(1-p)**2*p**2)/np.pi - 4*(2*p**2/np.pi - p/(r**2+1))*((1-p)**2-(1-p)/(r**2+1)))
    
    def ParamS(self,p,r):
        return (4*(2*p**2/np.pi - p/(r**2+1))*(1-p))/(r**2+1)
    
    def SigMin(self,p,res,mu2):
        return mu2*np.sqrt((-self.ParamU(p,res)+(self.bx(p,mu2)**2/mu2**2))/(self.ParamS(p,res)))
    
    def SigMax(self,p,res,mu2):
        return mu2*np.sqrt((-self.ParamU(p,res))/(self.ParamS(p,res)))
    
    def sigma1(self,p,res,sig2,mu2):
        return (-self.bx(p,mu2)+np.sqrt(self.delta(p,res,sig2,mu2)))/(2*self.ax(p,res))
    
    def sigma2(self,n,p,res,mu2):
        if ((-self.ParamU(p,res)+(self.bx(p,mu2)**2/mu2**2))/(self.ParamS(p,res)) > 0):
            return self.SigMin(p,res,mu2)+n*(self.SigMax(p,res,mu2)-self.SigMin(p,res,mu2))
        else:
            return n*self.SigMax(p,res,mu2)
        
# Final model shape/function (for one SPE)
    def doubleGaussConstrained(self,x,pp,res,mu2,n):
        p = pp*self.PMax(res)
        sig2 = self.sigma2(n,p,res,mu2)
        sig1 = self.sigma1(p,res,sig2,mu2)
        return self.doubleGauss(x,sig1,mu2,sig2,p)
        
# Get the gain from the parameters model
    def Gain(self,pp,res,mu2,n):
        p = pp*self.PMax(res)
        sig2 = self.sigma2(n,p,res,mu2)
        return (1-p)*mu2 + 2*p*self.sigma1(p,res,sig2,mu2)/np.sqrt(2*np.pi)
    
    #def nPEPDF(x,pp,res,mu2,n,muped,sigped,nph):
    #    allrange = np.linspace(-1000,1000,2000)
    #    spe = doubleGaussConstrained(allrange,pp,res,mu2,n)
    #    ped = gaussian(allrange, muped, sigped)
    #    for i in range(nph):
    #        npe = np.convolve(spe,ppp,"same")
    
# The PDF model for one/N SPE convoluted with the pedestal (0 SPE --> pedestal, 1 SPE --> 1 SPExPed, etc)
    def nPEPDF(self,x,pp,res,mu2,n,muped,sigped,nph,size_charge):
        allrange = np.linspace(-1 * size_charge,size_charge,size_charge*2)
        spe = []
        for i in range(len(allrange)):
            if (allrange[i]>=0):
                spe.append(self.doubleGaussConstrained(allrange[i],pp,res,mu2,n))
            else:
                spe.append(0)
        # ~ plt.plot(allrange,spe)
        #npe = semi_gaussian(allrange, muped, sigped)
        npe = self.gaussian(allrange, 0, sigped)
        # ~ plt.plot(allrange,npe)
        # ~ plt.show()
        for i in range(nph):
            #npe = np.convolve(npe,spe,"same")
            npe = signal.fftconvolve(npe,spe,"same")
        # ~ plt.plot(allrange,npe)
        # ~ plt.show()
        fff = interpolate.UnivariateSpline(allrange,npe,ext=1,k=3,s=0)
        norm = np.trapz(fff(allrange),allrange)
        return fff(x-muped)/norm
    
# The real final model callign all the above for luminosity (lum) + PED, wil return probability of number of Spe
    def MPE2(self,x,pp,res,mu2,n,muped,sigped,lum):
        f = 0
        ntotalPE = 0
        for i in range(1000):
            if (gammainc(i+1,lum) < 1e-5):
                ntotalPE = i
                break
        #print(ntotalPE)
        for i in range(ntotalPE):
            f = f + ((lum**i)/math.factorial(i)) * np.exp(-lum) * self.nPEPDF(x,pp,res,mu2,n,muped,sigped,i,int(mu2*ntotalPE+10*mu2))
        return f
        
    ####### Likelihood ########
    
# Not tested
    def NG_LikelihoodPedestal_Unbinned(self,mean,sigma,charge):
        Lik = 0
        for i in range(len(events)):
            Lik = Lik-2.*math.log(self.gaussian(charge[i],mean,sigma))
        return Lik
        
    
# Not tested
    def NG_LikelihoodSignal_Unbinned(self,pp,res,mu2,n,muped,sigped,lum,charge,nPrecision):
        MaxCharge = np.maximum(charge)+1
        MinCharge = np.minimum(charge)-1
        ChargeTable = np.linspace(MinCharge,MaxCharge,nPrecision)
        pdf = self.MPE2(ChargeTable,pp,res,mu2,n,muped,sigped,lum)
        pdf_interpolated = interpolate.UnivariateSpline(ChargeTable,npe,ext=1,k=3,s=0)
        Lik = 0
        for i in range(len(events)):
            Lik = Lik-2*math.log(pdf_interpolated(charge[i]))
        return Lik
        
# Chi2 (used now) for the pedestal (for pure pedestal data)
    def NG_LikelihoodPedestal_Chi2(self,mean,sigma,charge,nEvents):
        Lik = 0
        Ntot = np.sum(nEvents)
        for i in range(len(nEvents)):
            if (nEvents[i] > 0):
                Lik = Lik + ((self.gaussian(charge[i],mean,sigma)*Ntot - nEvents[i])**2)/nEvents[i]
        return Lik
        
    
# Chi2 (used now) for signal ie luminosity (SPE)
    def NG_LikelihoodSignal_Chi2(self,pp,res,mu2,n,muped,sigped,lum,charge,nEvents):
        pdf = self.MPE2(charge,pp,res,mu2,n,muped,sigped,lum)
        Ntot = np.sum(nEvents)
        Lik = 0
        for i in range(len(nEvents)):
            if (nEvents[i] > 0):
                Lik = Lik + (pdf[i]*Ntot-nEvents[i])**2/nEvents[i]
        return Lik
    
# To call NG_LikelihoodSignal_Chi2 wit hthe data loaded, will all free paramters of the model (work only at 1000V) but not recommened (very complex to converg) better to use the fix mdoel (below)
# Ideal/final for 1000V
    def Chi2Signal(self,pp,res,mu2,n,muped,sigped,lum):
        return self.NG_LikelihoodSignal_Chi2(pp,res,mu2,n,muped,sigped,lum,self.chargeSignal,self.histoSignal)
    
# To call NG_LikelihoodSignal_Chi2 wit hthe data loaded, not all free parameters (pp and n are fixed (as they are independent from the gain, should be the same for all pixels althogh unchecked), will be usefull for the combined fit with the 1000V
# Paramters fixed after Sami did a avergae on 50 pixels just have an idea and gain time
    def Chi2SignalFixedModel(self,res,mu2,muped,sigped,lum):
        return self.NG_LikelihoodSignal_Chi2(self.pp,res,mu2,self.n,muped,sigped,lum,self.chargeSignal,self.histoSignal)
    
# Chi2 (used now) for the ped at 1000V (alwasy work)
    def Chi2Ped(self,muped,sigped):
        return self.NG_LikelihoodPedestal_Chi2(muped,sigped,self.chargePed,self.histoPed)
    
# Chi2 (used now) for the signal (HHV: high high voltage ie 1400V) at 1400V
# Ideal/final for 1400V
    def Chi2SignalHHV(self,pp,res,mu2,n,muped,sigped,lum):
        return self.NG_LikelihoodSignal_Chi2(pp,res,mu2,n,muped,sigped,lum,self.chargeSignalHHV,self.histoSignalHHV)
    
# Chi2 (used now) for the ped at 1400V (always work)
# Ideal for file with only pedestal/noise (not for us now))
    def Chi2PedHHV(self,muped,sigped):
        return self.NG_LikelihoodPedestal_Chi2(muped,sigped,self.chargePedHHV,self.histoPedHHV)
        
# Chi2 (not tested yet) for the combined pedestal + signal (at 1000V)
# If pp and n are free it will not converge, requires a pedestal run like above, ideal for 1000V file + noise/pedestal file
# Not for now
    def Chi2CombiSignalAndPed(self,pp,res,mu2,n,muped,sigped,lum):
        return self.Chi2Signal(self.pp,res,mu2,self.n,muped,sigped,lum)+self.Chi2Ped(muped,sigped)
        #return self.Chi2Signal(pp,res,mu2,n,muped,sigped,lum)+self.Chi2Ped(muped,sigped)
        
# Chi2 (not tested yet) for the combined pedestal + signal (at 1400V)
# Requires a pedestal file so not for now
    def Chi2CombiSignalAndPedHHV(self,pp,res,mu2,n,muped,sigped,lum):
        return self.Chi2SignalHHV(pp,res,mu2,n,muped,sigped,lum)+self.Chi2PedHHV(muped,sigped)
        
# Chi2 (not tested yet) for the combined pedestal + signal (1000V) + signal (1400V)
# Requires a pedestal file so not for now
# Ideal/final for ped + 1000V + 1400V
    def Chi2AllCombined(self,pp,res,mu2,mu2HHV,n,muped,mupedHHV,sigped,lum,lumHHV):
        return self.Chi2CombiSignalAndPed(pp,res,mu2,n,muped,sigped,lum)+self.Chi2CombiSignalAndPedHHV(pp,res,mu2HHV,n,mupedHHV,sigped,lum)
        
# Chi2 (not tested yet) for the combined signal (1000V) + signal (1400V)
# Ideal/final for 1000V + 1400V
    def Chi2AllNoPedCombined(self,pp,res,mu2,mu2HHV,n,muped,mupedHHV,sigped,lum,lumHHV):
        return self.Chi2Signal(pp,res,mu2,n,muped,sigped,lum)+self.Chi2SignalHHV(pp,res,mu2HHV,n,mupedHHV,sigped,lum)
        
    ####### Compute Start Parameters ######
   
# "Smart" values to start the fit 
    def StartParameters(self):
        self.pedestalMean = (np.min(self.chargePed) + np.sum(self.chargePed*self.histoPed)/np.sum(self.histoPed))/2.
        self.pedestalMeanLow = np.min(self.chargePed)
        self.pedestalMeanUp = np.sum(self.chargePed*self.histoPed)/np.sum(self.histoPed)
        #self.pedestalWidth = np.sqrt(np.sum(self.chargePed**2 * self.histoPed)/np.sum(self.histoPed)-self.pedestalMean**2)
        #self.pedestalWidth = 16
        self.pedestalWidth = 50
        #self.pedestalWidthLow = self.pedestalWidth-3
        self.pedestalWidthLow = 1
        self.pedestalWidthUp = self.pedestalWidth+100
        print("pedestal mean ",self.pedestalMean," width ", self.pedestalWidth)
        self.pedestalMeanSPE = self.pedestalMean
        self.pedestalMeanSPELow = self.pedestalMeanSPE-60
        self.pedestalMeanSPEUp =self.pedestalMeanSPE+60
        self.pedestalWidthSPE = self.pedestalWidth
        self.pedestalWidthSPELow = self.pedestalWidthSPE-10
        self.pedestalWidthSPEUp = self.pedestalWidthSPE+10
        self.Luminosity = 1.
        self.LuminosityLow = 0.01
        #self.LuminosityUp = 2.
        self.LuminosityUp = 5.
        self.pp = 0.3735
        self.resolution = 0.5
        self.resolutionLow = 0.3
        self.resolutionUp = 0.7 
        #self.meanUp = 50.
        self.meanUp = 500.
        #self.meanUpLow = 20.
        self.meanUpLow = 400.
        #self.meanUpUp = 100.
        self.meanUpUp = 600.
        self.n = 0.708
        self.pedestalMeanHHV = self.pedestalMean
        self.pedestalWidthHHV = self.pedestalWidth
        self.pedestalMeanSPEHHV = self.pedestalMean
        self.pedestalWidthSPEHHV = self.pedestalWidth
        self.LuminosityHHV = 1.
        self.ppHHV = 0.3735
        self.resolutionHHV = 0.5
        self.meanUpHHV = 300.
        self.nHHV = 0.708
        
    ####### Fit minuit #######
   
# To fit with iminuit for Signal only, create functions for all the cases above 
 
    def fitSignalOnly(self,ID = 0):
        self.StartParameters()
        parName = ["res","mu2","muped","sigped","lum"]
        parValues = [self.resolution,self.meanUp,self.pedestalMean,self.pedestalWidth,self.Luminosity]
        LimitLow = [self.resolutionLow,self.meanUpLow,self.pedestalMeanLow,self.pedestalWidthLow,self.LuminosityLow]
        LimitUp = [self.resolutionUp,self.meanUpUp,self.pedestalMeanUp,self.pedestalWidthUp,self.LuminosityUp]
        
        parameters = make_minuit_par_kwargs(parValues,parName,LimitLow,LimitUp)
        m = Minuit(self.Chi2SignalFixedModel,**parameters['values'])
        m.print_level = 2
        set_minuit_parameters_limits_and_errors(m,parameters)
        m.strategy = 2
        print(m.values)
        results = m.migrad(ncall=4000000)
        m.hesse()
        print(m.values)
        print(m.errors)
        print("Reconstructed gain is ", self.Gain(self.pp,m.values[0],m.values[1],self.n))
        gainGenerated = []
        for i in range(1000): 
            gainGenerated.append(self.Gain(self.pp,random.gauss(m.values[0], m.errors[0]),random.gauss(m.values[1], m.errors[1]),self.n))
        print("Uncertainty is ", np.std(gainGenerated))
        plt.figure(figsize=(8, 6))
        plt.errorbar(self.chargeSignal,self.histoSignal,np.sqrt(self.histoSignal),zorder=0,fmt=".",label = "data")
        plt.plot(self.chargeSignal,np.trapz(self.histoSignal,self.chargeSignal)*self.MPE2(self.chargeSignal,self.pp,m.values[0],m.values[1],self.n,m.values[2],m.values[3],m.values[4]),zorder=1,linewidth=2,label = "MPE model fit \n gain = "+str(round(self.Gain(self.pp,m.values[0],m.values[1],self.n),2))+" +/- " + str(round(np.std(gainGenerated),2)) + " ADC/pe")
        plt.xticks(size = 15)
        plt.yticks(size = 15)
        plt.xlabel("Charge (ADC)", size=15)
        plt.ylabel("Events", size=15)
        #plt.plot(self.chargeSignal,self.MPE2(self.chargeSignal,self.pp,m.values[0],m.values[1],self.n,m.values[2],m.values[3],m.values[4]),linewidth=2)
        #print(np.trapz(self.MPE2(self.chargeSignal,self.pp,m.values[0],m.values[1],self.n,m.values[2],m.values[3],m.values[4]),self.chargeSignal))
        plt.legend(fontsize=15)
        #plt.show()
        return self.Gain(self.pp,m.values[0],m.values[1],self.n),np.std(gainGenerated),m.values,m.errors
        

# To fit with iminuit for Signal 1400V only with pp and n free paramters, create functions for all the cases above 
    def fitSignalOnly1400V(self):
        self.StartParameters()
	# need to adapt free parameters
        parName = ["pp","res","mu2","n", "muped","sigped","lum"]
        parValues = [self.pp,self.resolution,self.meanUp,self.n,self.pedestalMean,self.pedestalWidth,self.Luminosity]
        LimitLow = [0.2,self.resolutionLow,self.meanUpLow,0.5,self.pedestalMeanLow,self.pedestalWidthLow,self.LuminosityLow]
        LimitUp = [0.8,self.resolutionUp,self.meanUpUp,0.9,self.pedestalMeanUp,self.pedestalWidthUp,self.LuminosityUp]
        
        test = make_minuit_par_kwargs(parValues,parName,LimitLow,LimitUp)
	# the critical lign (here using fixed model)
        m = Minuit(self.Chi2Signal,**test, print_level=2)
        m.get_initial_param_states()
        m.set_strategy(2)
        print(m.values)
        results = m.migrad(ncall=4000000)
        m.hesse()
        print(m.values)
        print(m.errors)
        gain = self.Gain(m.values[0],m.values[1],m.values[2],m.values[3])
        print(f"Reconstructed gain is  {gain}")
        gainGenerated = []
        for i in range(1000): 
            gainGenerated.append(self.Gain(random.gauss(m.values[0],m.errors[0]),random.gauss(m.values[1],m.errors[1]),random.gauss(m.values[2],m.errors[2]),random.gauss(m.values[3],m.errors[3])))
        print(gainGenerated)
        print("Uncertainty is ", np.std(gainGenerated))
        plt.figure(figsize=(8, 6))
        plt.errorbar(self.chargeSignal,self.histoSignal,np.sqrt(self.histoSignal),zorder=0,fmt=".",label = "data")
        plt.plot(self.chargeSignal,np.trapz(self.histoSignal,self.chargeSignal)*self.MPE2(self.chargeSignal,m.values[0],m.values[1],m.values[2],m.values[3],m.values[4],m.values[5],m.values[6]),zorder=1,linewidth=2,label = "MPE model fit \n gain = "+str(round(gain,2))+" +/- " + str(round(np.std(gainGenerated),2)) + " ADC/pe")
        plt.xticks(size = 15)
        plt.yticks(size = 15)
        plt.xlabel("Charge (ADC)", size=15)
        plt.ylabel("Events", size=15)
        #plt.plot(self.chargeSignal,self.MPE2(self.chargeSignal,self.pp,m.values[0],m.values[1],self.n,m.values[2],m.values[3],m.values[4]),linewidth=2)
        #print(np.trapz(self.MPE2(self.chargeSignal,self.pp,m.values[0],m.values[1],self.n,m.values[2],m.values[3],m.values[4]),self.chargeSignal))
        plt.legend(fontsize=15)
        #plt.show()
        return gain,np.std(gainGenerated),m.values,m.errors
        

