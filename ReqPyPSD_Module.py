'''
ReqPyPSD_Module
luis.montejo@upr.edu
Generation of response spectrum compatible records satisfying a minimum power 
spectral density function

===============================================================================
References:
    
Montejo, L.A. 2025. "Generation of Response Spectrum Compatible Records Satisfying 
a Minimum Power Spectral Density Function." Earthquake Engineering and Resilience. 
https://doi.org/10.1002/eer2.70008
    
Montejo, L.A. 2024. "Strong-Motion-Duration-Dependent Power Spectral Density 
Functions Compatible with Design Response Spectra." Geotechnics 4, no. 4: 1048-1064. 
https://doi.org/10.3390/geotechnics4040053

Montejo, L.A. 2021. "Response spectral matching of horizontal ground motion 
components to an orientation-independent spectrum (RotDnn)."
Earthquake Spectra, 37(2), 1127-1144. https://doi.org/10.1177/8755293020970981

===============================================================================

This module contains the Python functions required to generate response spectrum 
compatible records satisfying a prescribed minimum power spectral density function.

* ReqPyPSD: generate response spectrum compatible records satisfying a 
prescribed minimum power spectral density function

* CheckPeriodRange: Verifies the specified matching period

* zumontw: Generates the Suarez-Montejo Wavelet function

* cwtzm: Continuous Wavelet Transform using the Suarez-Montejo wavelet
via convolution in the frequency domain

* getdetails: Generates the detail functions

* basecorr: performs baseline correction

* baselinecorrect: performs baseline correction iteratively calling basecorr

* PGAcorrection: performs localized PGA correction

* logfrequencies: Calculates logarithmically spaced frequencies within 
  a given range

* SignificantDuration: Estimates significant duration and Arias Intensity

* PSDFFTEq: Calculates the power spectral density of earthquake acceleration 
time-series, FFT is normalized by dt, FFT/PSD is calculated over the strong 
motion duration returns the one-sided PSD and a "smoothed" version by taking 
the average over a frequency window width of user defined % of the subject 
frequency.

* log_interp: Performs logarithmic interpolation

* load_PEERNGA_record(filepath): Load record in .at2 format (PEER NGA Databases)

* RSFD_S: Response spectra (operations in the frequency domain)

'''

def ReqPyPSD(s,fs,f_PSA,targetPSA,f_PSD,targetPSD,targetPGA,
             F1PSA=0.2,F2PSA=50,F1PSD=0.3,F2PSD=30,zi=0.05,
             PSA_poly_order=4,PSAPSD_poly_order=4,
             NS=300,nit=30,maxit=1000,plots=1,nameOut='ReqPyOut'):
    '''
    Parameters
    ----------
    s : seed record (acceleration time series in g's)
    fs : seed record sampling frequency [Hz]
    f_PSA : frequencies at which the target spectrum is defined [Hz]
    targetPSA : design/target PSA spectrum [g]
    f_PSD : frequencies at which the target PSD is defined [Hz]
    targetPSD : target power spectral density [Hz]
    targetPGA : target PGA associated with the target PSA spectrum [g]
    F1PSA, F2PSA : Defines the frequency range for matching PSA
                   The default is 0.2 and 50.
    F1PSD,F2PSD : Defines the frequency range for minimum PSD
                   The default is 0.3 and 30
    zi: damping ratio for response spectrum (default 5%)
    PSA_poly_order : order of polynomial for detrending of PSA matched record
                     use -1 for no detrending, default is 4
    PSAPSD_poly_order : order of polynomial for detrending of PSD adjusted record
                     use -1 for no detrending, default is 4
    NS: number of scale values to perform the CWT (default 100)
    nit: number of iterations for matching PSA (default 30)
    maxit: max number of iterations for adjusting PSD (default 1000)
    plots: 1/0 (yes/no, whether plots are generated, default 1)
    nameOut: string used to name the output files
    Returns
    -------
    sc : PSA spectrally matched record
    sca: PSD adjusted record
    '''
    
    
    import numpy as np
    from scipy import signal,integrate
 
    g = 9.81
    
    T1PSA, T2PSA = 1/F2PSA, 1/F1PSA       # period range for matching PSA
    T1PSD, T2PSD = 1/F2PSD, 1/F1PSD       # period range for matching PSD    
    
    dt = 1/fs
    nt = np.size(s)
    if nt%2!=0: s = np.append(s,0); nt+=1 # Adjust time vector for even length
    
    t = np.linspace(0,nt*dt,nt)  # time vector
    
    To = 1/f_PSA                 
    Tsortindex = np.argsort(To)
    To = To[Tsortindex]
    targetPSA = targetPSA[Tsortindex]   # ensures ascending T order in target PSA
       
    Fsortindex = np.argsort(f_PSD)
    f_PSD = f_PSD[Fsortindex]
    targetPSD = targetPSD[Fsortindex]   # ensures ascending F order in target PSD
       
    # modify in wavelet domain:
        
    FF1 = min(4/(nt*dt),0.1); FF2 = 1/(2*dt)     # frequency range for CWT decomposition
       
    T1PSA,T2PSA,FF1 = CheckPeriodRange(T1PSA,T2PSA,To,FF1,FF2) # verifies period range for PSA matching
    
    if T1PSD<T1PSA or T2PSD>T2PSA:
        print('PSD matching range must be within PSA matching period range')
        exit
    
    F1PSA, F2PSA = 1/T1PSA, 1/T2PSA       # period range for matching PSA
    
    # Perform Continuous Wavelet Decomposition:
    
    omega = np.pi; zeta  = 0.05         # wavelet function parameters
    freqs = np.geomspace(FF2,FF1,NS)    # frequencies vector
    T = 1/freqs                         # periods vector
    scales = omega/(2*np.pi*freqs)      # scales vector    
    C = cwtzm(s,fs,scales,omega,zeta)   # performs CWT 
    
    TlocsPSA = np.nonzero((T>=T1PSA)&(T<=T2PSA)) # loacations of the scalees/periods/freqs to match PSA
    TlocsPSD = np.nonzero((T>=T1PSD)&(T<=T2PSD)) # loacations of the scalees/periods/freqs to match PSD
    
    nTlocsPSA = np.size(TlocsPSA)
    
    ds = log_interp(T,To,targetPSA)  # resample target PSA spectrum
    TargetPSD_int = log_interp(freqs, f_PSD, targetPSD) # resample target PSD
    
    print('='*40)
    print('Wavelet decomposition performed')
    print('='*40)
    
    # Generate detail functions:
            
    D,sr = getdetails(t,s,C,scales,omega,zeta) # matrix with the detail
                                               # functions (D) and 
                                               # signal recondtructed (sr)
                                               
          
    print('='*40)
    print('Detail functions generated')
    print('='*40)
    
    # response spectra from the reconstructed signal:
       
    PSAsr,_,_ = RSFD_S(T,sr,zi,dt)
        
    # initial scaling of record:
    
    sf = np.sum(ds[TlocsPSA])/np.sum(PSAsr[TlocsPSA]) # initial scaling factor
    
    sr  = sf * sr; D  = sf * D # scaled reconstructed record and details
    
    ##############################################################################
    ##############################################################################
    
    # 1. match PSA (traditional CWT approach)
    
    ##############################################################################
    ##############################################################################
    
    rmsePSA   = np.zeros((nit+1))  # stores the rms errors
    hPSAbc = np.zeros((NS,nit+1))  # stores the response spectra
    ns = np.zeros((nt, nit+1))     # stores the motions
    hPSAbc[:,0] = sf*PSAsr         # response spectrum of the scaled motion
    ns[:,0] = sr                   # scaled motion
    DN = np.zeros((NS,nt,nit+1))   # 3d array storing the detail functions
    DN[:,:,0] = D                  # detail functons of the scaled record
    
    difPSA = np.abs( hPSAbc[TlocsPSA,0] - ds[TlocsPSA] ) / ds[TlocsPSA]
    rmsePSA[0] = np.linalg.norm(difPSA) / np.sqrt(nTlocsPSA) * 100 # error related to the scaled motion
    
    factorPSA = np.ones((NS,1)) # details modification factors based on PSA ratios
    
    print('='*40)
    print('Matching the target PSA')
    print('='*40)
    print(f'iteration 0 of {nit} rmse = {rmsePSA[0]:.2f}%')
    
    for qq in range(1,nit+1):
        
        factorPSA[TlocsPSA,0] = ds[TlocsPSA]/hPSAbc[TlocsPSA,qq-1]
        DN[:,:,qq] = factorPSA*DN[:,:,qq-1]
        ns[:,qq] = np.trapz(DN[:,:,qq].T,scales)
        hPSAbc[:,qq],_,_ = RSFD_S(T,ns[:,qq],zi,dt)
        difPSA = np.abs( hPSAbc[TlocsPSA,qq] - ds[TlocsPSA] ) / ds[TlocsPSA]
        rmsePSA[qq]  =np.linalg.norm(difPSA) / np.sqrt(nTlocsPSA) * 100
        
        print(f'iteration {qq} of {nit} rmse = {rmsePSA[qq]:.2f}%')
        
    brloc = np.argmin(rmsePSA) # locates min error
    sc = ns[:,brloc]           # compatible record
    Dc = DN[:,:,brloc]         # compatible record details
    
    print('='*40)
    print(f'lowest rmse was {rmsePSA[brloc]:.2f}% at iteration {brloc}')
    print('='*40)
    
    ##############################################################################
    ##############################################################################
    
    # 2. satisfy 0.7*PSD requirement (one-by-one detail modifications only within 
    # the strong motion part Sd5-75)
    
    ##############################################################################
    ##############################################################################
    
    sca = np.copy(sc)   # compatible record adjusted for PSD
    Dca = np.copy(Dc)   # details of the compatible record adjusted for PSD
    
    # PSD for PSA-compatible record:
    _,PSD_sca,PSDavg_sca,freqs_sca,sd_sca,_,t1sca,t2sca = PSDFFTEq(sca*g,fs,alphaw=0.1,duration=(5,75),nFFT='same',basefornFFT = 0, overlap=20, detrend='linear')
    
    MotionPSD_int = log_interp(freqs, freqs_sca, PSDavg_sca) # interpolate to detail frequencies
    ratio = MotionPSD_int/TargetPSD_int # take ratios with target PSD to check 0.7 requirem
    minratio = np.min(ratio[TlocsPSD])
    
    if minratio>=0.7:
        
        print('PSD ratio staisified from PSA matching')
        
    else:
        print('='*40)
        print('Adjustements to satify the minimum required PSD ratio of 0.7')
        print('='*40)
        print(f'iteratiom #: 0, min. ratio: {minratio:.2f}  ')
    
        crit_neg_loc = TlocsPSD[0][0] + np.argmin(ratio[TlocsPSD]) # period/frequency location of critical PSD amplitude
        alphaw = 0.1    # Tukey window paramerer
        locs = np.where((t>=t1sca-dt/2)&(t<=t2sca+dt/2)) # time location to modify detail
        nlocs = np.size(locs)
        windowshort = signal.windows.tukey(nlocs,alphaw)
        window = np.zeros(nt)
        window[locs] = windowshort
    
        cont = 0   # controls the number of iterations
       
        while (MotionPSD_int[crit_neg_loc]<0.7*TargetPSD_int[crit_neg_loc]): 
            
            factorn = (0.71*TargetPSD_int[crit_neg_loc]/MotionPSD_int[crit_neg_loc])**0.5
            factor = factorn*window
    
            Dca[crit_neg_loc,:] = factor*Dca[crit_neg_loc,:]
            sca = np.trapz(Dca.T,scales)
            _,PSD_sca,PSDavg_sca,freqs_sca,sd_sca,_,t1sca,t2sca = PSDFFTEq(sca*g,fs,alphaw=0.1,duration=(5,75),nFFT='same',basefornFFT = 0, overlap=20, detrend='linear')
            
            locs = np.where((t>=t1sca-dt/2)&(t<=t2sca+dt/2)) # time location to modify detail
            nlocs = np.size(locs)
            windowshort = signal.windows.tukey(nlocs,alphaw)
            window = np.zeros(nt)
            window[locs] = windowshort
            
            MotionPSD_int = log_interp(freqs, freqs_sca, PSDavg_sca)
            ratio = MotionPSD_int/TargetPSD_int
            
            minratio = np.min(ratio[TlocsPSD])
            
            print(f'iteratiom #: {cont+1}, detail modified: {T[crit_neg_loc]:.2f}s, factor: {factorn:.2f}, min. ratio: {minratio:.2f}  ')
    
            if minratio>=0.7:
                print('='*40)
                print(f'PSD ratio satisfied at iteration {cont+1}')
                print('='*40)
                break
            
            crit_neg_loc = TlocsPSD[0][0] + np.argmin(ratio[TlocsPSD])
            cont = cont+1
    
            if cont==maxit:
                print('='*40)
                print(f'the max # of iteration was reached, the min PSD ratio is {ratio[crit_neg_loc]:.2f}')
                print('='*40)
                break
    
    ##############################################################################
    ##############################################################################
    
    # 3. Corect PGA locally in the time domain
    
    ##############################################################################
    ##############################################################################
    
    print('='*40)
    print('Performing PGA correction')
    print('='*40)
    sca = PGAcorrection(targetPGA,t,sca,maxit=1000)
    
    ##############################################################################
    ##############################################################################
    
    # 4. final amplitude scaling to ensure 0.9PSA and 0.7PSD
    #    PSD is checked using all Fourier frequencies within the range
    #    PSA is chequed NRC periods (no the wavelet decomposition periods as before)
    
    ##############################################################################
    ##############################################################################
    
    print('='*40)
    print('Performing final amplitude scaling to ensure 0.9PSA and 0.7PSD')
    print('='*40)
    
    
    freqs_NRC_PSA_check = logfrequencies(0.1, fs/2, 100)[::-1]
    T_NRC_PSA_check = 1/freqs_NRC_PSA_check
    nfreqsNRC = np.size(freqs_NRC_PSA_check)
    
    PSAsca,_,_ = RSFD_S(T_NRC_PSA_check,sca,zi,dt)
    _,PSD_sca,PSDavg_sca,freqs_sca,sd_sca,_,t1sca,t2sca = PSDFFTEq(sca*g,fs,alphaw=0.1,duration=(5,75),nFFT='same',basefornFFT = 0, overlap=20, detrend='linear')
    
    TargetPSD_int_NRC_check = log_interp(freqs_sca, f_PSD, targetPSD)
    TargetPSA_int_NRC_check = log_interp(T_NRC_PSA_check,To,targetPSA)
    
    TlocsPSA_NRC_check = np.nonzero((T_NRC_PSA_check>=T1PSA)&(T_NRC_PSA_check<=T2PSA)) # loacations of the scalees/periods/freqs to match PSA
    FlocsPSD_NRC_check = np.nonzero((freqs_sca>=F1PSD)&(freqs_sca<=F2PSD)) # loacations of the scalees/periods/freqs to match PSD
    
    factorPSA = np.ones(np.size(T_NRC_PSA_check))
    factorPSD = np.ones(np.size(freqs_sca))
    factorPSA[TlocsPSA_NRC_check] = 0.9*TargetPSA_int_NRC_check[TlocsPSA_NRC_check]/PSAsca[TlocsPSA_NRC_check]
    factorPSD[FlocsPSD_NRC_check] = (0.7*TargetPSD_int_NRC_check[FlocsPSD_NRC_check]/PSDavg_sca[FlocsPSD_NRC_check])**0.5
    factor = np.max(np.hstack((factorPSA,factorPSD)))
    
    sca = factor*sca
    
    print(f'amplitude factor applied: {factor:.2f}')
    
    ##############################################################################
    ##############################################################################
    
    # 5. performing baseline correction for PSA-PSD compatible record
    #     and PSA-only comatible record
    
    ##############################################################################
    ##############################################################################
    
    print('baseline correction for PSA-PSD compatible record:')
    sca,sca_vel,sca_disp = baselinecorrect(sca,t,porder=PSAPSD_poly_order,imax=80,tol=0.01)
    print('baseline correction for PSA-only compatible record:')
    sc,sc_vel,sc_disp = baselinecorrect(sc,t,porder=PSA_poly_order,imax=80,tol=0.01)
    
    ##############################################################################
    
    
    if plots==1:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['font.size'] = 9
        mpl.rcParams['legend.frameon'] = False
    
        ##############################################################################
        
        # 6. obtain final PSA and PSD for records and generate plots:
        
        ##############################################################################
        ##############################################################################
        ##############################################
        # plot spectra:
        ##############################################
        
        # PSD for scaled record:  
        _,PSD_ss,PSDavg_ss,freqs_ss,sd_s,_,_,_ = PSDFFTEq(sf*s*g,fs,alphaw=0.1,duration=(5,75),nFFT='same',basefornFFT = 0, overlap=20, detrend='linear')
        # PSA for scaled record:
        PSAss,_,_ =  RSFD_S(T_NRC_PSA_check,sf*s,zi,dt)
        
        # PSD for PSA-only comatible record:  
        _,PSD_sc,PSDavg_sc,freqs_sc,sd_sc,_,t1sc,t2sc = PSDFFTEq(sc*g,fs,alphaw=0.1,duration=(5,75),nFFT='same',basefornFFT = 0, overlap=20, detrend='linear')
        # PSA for PSA-only comatible record:
        PSAsc,_,_ =  RSFD_S(T_NRC_PSA_check,sc,zi,dt)
        
        # PSD for PSA-PSD comatible record:  
        _,PSD_sca,PSDavg_sca,freqs_sca,sd_sca,_,t1sca,t2sca = PSDFFTEq(sca*g,fs,alphaw=0.1,duration=(5,75),nFFT='same',basefornFFT = 0, overlap=20, detrend='linear')
        # PSA for PSA-only comatible record:
        PSAsca,_,_ =  RSFD_S(T_NRC_PSA_check,sca,zi,dt)
        
        ratioPSAsc = PSAsc/TargetPSA_int_NRC_check
        ratioPSAsca = PSAsca/TargetPSA_int_NRC_check
        
        ratioPSDsc = PSDavg_sc/TargetPSD_int_NRC_check
        ratioPSDsca = PSDavg_sca/TargetPSD_int_NRC_check
        
        PSAylim = (-0.05,1.03*np.max(np.hstack((1.3*TargetPSA_int_NRC_check,PSAsc,PSAsca))))
        PSDylim = (0.7*np.min(TargetPSD_int_NRC_check),1.1*np.max(np.hstack((PSDavg_sc,PSDavg_sca))))
        
        auxxPSA = [F1PSA,F1PSA,F2PSA,F2PSA,F1PSA]
        auxyPSA = [PSAylim[0],PSAylim[1],PSAylim[1],PSAylim[0],PSAylim[0]]
        
        auxxPSD = [F1PSD,F1PSD,F2PSD,F2PSD,F1PSD]
        auxyPSD = [PSDylim[0],PSDylim[1],PSAylim[1],PSDylim[0],PSDylim[0]]
        
        fig1 = plt.figure(constrained_layout=True,figsize=(6.5,4.6))
        gs = fig1.add_gridspec(3,2)
        
        fig1.add_subplot(gs[0:2,0])
        plt.fill_between( auxxPSA, auxyPSA, color='silver', alpha=0.3)
        plt.semilogx(freqs_NRC_PSA_check,TargetPSA_int_NRC_check,label='target',color='black',lw=2)
        plt.semilogx(freqs_NRC_PSA_check,1.3*TargetPSA_int_NRC_check,'--',label='SRP 3.7.1 limits',color='black',lw=1)
        plt.semilogx(freqs_NRC_PSA_check,0.9*TargetPSA_int_NRC_check,'--',color='black',lw=1)
        #plt.semilogx(freqs_NRC_PSA_check,PSAss,label='scaled',color='cornflowerblue',lw=1)
        plt.semilogx(freqs_NRC_PSA_check,PSAsc,label='PSA matched',color='cornflowerblue',lw=1)
        plt.semilogx(freqs_NRC_PSA_check,PSAsca,label='PSA matched / PSD adjusted',color='salmon',lw=1)
        plt.gca().set_xticklabels([])
        plt.ylim(PSAylim)
        plt.xlim((0.09,110))
        plt.ylabel('PSA [g]')
        
        fig1.add_subplot(gs[2,0])
        plt.semilogx(freqs_NRC_PSA_check,np.ones(nfreqsNRC),color='black',lw=1)
        plt.semilogx(freqs_NRC_PSA_check,1.3*np.ones(nfreqsNRC),'--',color='black',lw=1)
        plt.semilogx(freqs_NRC_PSA_check,0.9*np.ones(nfreqsNRC),'--',color='black',lw=1)
        plt.semilogx(freqs_NRC_PSA_check[TlocsPSA_NRC_check],ratioPSAsc[TlocsPSA_NRC_check],color='cornflowerblue',lw=1)
        plt.semilogx(freqs_NRC_PSA_check[TlocsPSA_NRC_check],ratioPSAsca[TlocsPSA_NRC_check],color='salmon',lw=1)
        plt.xlim((0.09,110))
        plt.xlabel('F [Hz]')
        plt.ylabel('PSA ratio')
        
        fig1.add_subplot(gs[0:2,1])
        plt.fill_between( auxxPSD, auxyPSD, color='silver', alpha=0.3)
        plt.loglog(freqs_ss,TargetPSD_int_NRC_check,color='black',lw=2)
        plt.loglog(freqs_ss,0.7*TargetPSD_int_NRC_check,'--',color='black',lw=1)
        plt.loglog(freqs_sc,PSDavg_sc,color='cornflowerblue',lw=1)
        plt.loglog(freqs_sca,PSDavg_sca,color='salmon',lw=1)
        plt.gca().set_xticklabels([])
        plt.ylim(PSDylim)
        plt.xlim((0.09,110))
        plt.ylabel(r'PSD $[m^2/s^3]$')
        
        fig1.add_subplot(gs[2,1])
        plt.semilogx(freqs_ss,np.ones(np.size(freqs_ss)),color='black',lw=1)
        plt.semilogx(freqs_ss,0.7*np.ones(np.size(freqs_ss)),'--',color='black',lw=1)
        plt.semilogx(freqs_sc[FlocsPSD_NRC_check],ratioPSDsc[FlocsPSD_NRC_check],color='cornflowerblue',lw=1)
        plt.semilogx(freqs_sca[FlocsPSD_NRC_check],ratioPSDsca[FlocsPSD_NRC_check],color='salmon',lw=1)
        plt.xlim((0.09,110))
        plt.xlabel('F [Hz]')
        plt.ylabel('PSD ratio')
        
        fig1.legend(loc='upper center',ncols=4,labelcolor='linecolor')
        fig1.tight_layout()
        fig1.subplots_adjust(top=0.92)
        plt.savefig('Spectra_' + nameOut + '.jpg',dpi=300)
        ##############################################
        # plot time histories:
        ##############################################
        
        s_sd,s_AIcumnorm,s_AI,s_t1,s_t2 = SignificantDuration(sf*s,t,ival=5,fval=75)
        sc_sd,sc_AIcumnorm,sc_AI,sc_t1,sc_t2 = SignificantDuration(sc,t,ival=5,fval=75)
        sca_sd,sca_AIcumnorm,sca_AI,sca_t1,sca_t2 = SignificantDuration(sca,t,ival=5,fval=75)
        
        v = integrate.cumulative_trapezoid(s, t, initial=0)
        d = integrate.cumulative_trapezoid(v, t, initial=0)
        
        s_CAV = integrate.cumulative_trapezoid(np.abs(v), t, initial=0)
        sc_CAV = integrate.cumulative_trapezoid(np.abs(sc_vel), t, initial=0)
        sca_CAV = integrate.cumulative_trapezoid(np.abs(sca_vel), t, initial=0)
        
        alim = 1.05*np.max(np.abs(np.array([sf*s,sc,sca])))
        vlim = 1.05*np.max(np.abs(np.array([sf*v,sc_vel,sca_vel])))
        dlim = 1.05*np.max(np.abs(np.array([sf*d,sc_disp,sca_disp])))
        
        
        plt.figure(figsize=(6.5,6))
        
        plt.subplot(511)
        plt.plot(t,sf*s,linewidth=1,color='darkgray')
        plt.plot(t,sc,linewidth=1,color='cornflowerblue')
        plt.plot(t,sca,linewidth=1,color='salmon')
        plt.ylim(-alim,alim)
        plt.ylabel('a [g]')
        frame1 = plt.gca();frame1.axes.xaxis.set_ticklabels([])
        
        plt.subplot(512)
        plt.plot(t,sf*v,linewidth=1,color='darkgray')
        plt.plot(t,sc_vel,linewidth=1,color='cornflowerblue')
        plt.plot(t,sca_vel,linewidth=1,color='salmon')
        plt.ylim(-vlim,vlim)
        frame1 = plt.gca();frame1.axes.xaxis.set_ticklabels([])
        plt.ylabel('v/g')
        
        plt.subplot(513)
        plt.plot(t,sf*d,linewidth=1,color='darkgray')
        plt.plot(t,sc_disp,linewidth=1,color='cornflowerblue')
        plt.plot(t,sca_disp,linewidth=1,color='salmon')
        plt.ylim(-dlim,dlim)
        frame1 = plt.gca();frame1.axes.xaxis.set_ticklabels([])
        plt.ylabel('d/g')
        
        plt.subplot(514)
        plt.plot(t,sf*s_CAV,linewidth=1,color='darkgray')
        plt.plot(t,sc_CAV,linewidth=1,color='cornflowerblue')
        plt.plot(t,sca_CAV,linewidth=1,color='salmon')
        frame1 = plt.gca();frame1.axes.xaxis.set_ticklabels([])
        plt.ylabel('CAV/g')
        
        plt.subplot(515)
        plt.plot(t,s_AIcumnorm,linewidth=1,color='darkgray',label='scaled')
        plt.plot(t,sc_AIcumnorm,linewidth=1,color='cornflowerblue',label='PSA matched')
        plt.plot(t,sca_AIcumnorm,linewidth=1,color='salmon',label='PSA matced / PSD adjusted')
        plt.ylim(-0.05,1.05)
        plt.ylabel('AI norm.'); plt.xlabel('t [s]')
        
        plt.figlegend(loc='upper center',ncols=3,labelcolor='linecolor')
        plt.tight_layout(h_pad=0.1)
        plt.subplots_adjust(top=0.95)
        plt.savefig('TimeHistories_' + nameOut + '.jpg',dpi=300)
    
    np.savetxt(nameOut+'_SpectrallyMatched.txt',sc,header=f'dt={dt:.4f}')
    np.savetxt(nameOut+'_PSDAdjusted.txt',sca,header=f'dt={dt:.4f}')   
    return sc,sca

def CheckPeriodRange(T1,T2,To,FF1,FF2):
    '''
    CheckPeriodRange - Verifies that the specified matching period 
    range is doable 
    
    input:
        To: vector with the periods at which DS is defined
        T1, T2: define period range for matching 
                (defautl T1=T2=0 matches the whole spectrum)
        FF1, FF2: defines frequency range for CWT decomposition
        
    returns:
        updated values of T1,T2,FF1 if required
                
    '''
    if T1==0 and T2==0:
        T1 = To[0]; T2 = To[-1]
        
    if T1<To[0]: 
        T1 = To[0]
        print('='*40)
        print('warning: initial period for matching')
        print('fails outside the target spectrum')
        print('redefined to %.2f' %T1)
        print('='*40)
        
    if T2>To[-1]:
        T2 = To[-1]
        print('='*40)
        print('warning: final period for matching')
        print('fails outside the target spectrum')
        print('redefined to %.2f s' %T2)
        print('='*40)
    
    if T1<(1/FF2):
        T1 = 1/FF2
        print('='*40)
        print('warning: because of sampling frequency')
        print('limitations in the seed record')
        print('the target spectra can only be matched from %.2f s'%T1)
        print('='*40)
    
    if T2>(1/FF1):
        FF1 = 1/T2   # redefine FF1 to match the whole spectrum
    
    return T1,T2,FF1

def zumontw(t,omega,zeta):
    '''
    zumontw - Generates the Suarez-Montejo Wavelet function
    
    Ref. Suarez, L.E. & Montejo, L.A. Generation of artificial earthquakes 
    via the wavelet transform, Int. Journal of Solids and Structures, 42, 2005
    
    input:
        t     : time vector
        omega : wavelet parameter
        zeta  : wavelet parameter
    
    output:
        wv : wavelet function
    '''
    
    import numpy as np
    wv = np.exp(-zeta*omega*np.abs(t))*np.sin(omega*t)
    return wv

def cwtzm(s,fs,scales,omega,zeta):
    '''
    cwtzm - Continuous Wavelet Transform using the Suarez-Montejo wavelet
    via convolution in the frequency domain

    input:
        s        : input signal (vector)
        fs       : sampling frequency
        scales   : scales at which cwt would be performed 
        omega    : wavelet parameter
        zeta     : wavelet parameter

    output:
        coefs    : wavelet coefficients

    References:
        
        Gaviria, C. A., & Montejo, L. A. (2018). Optimal Wavelet Parameters for 
        System Identification of Civil Engineering Structures. Earthquake Spectra, 
        34(1), 197-216.
        
        Montejo, L.A., Suarez, L.E., 2013. An improved CWT-based algorithm for 
        the generation of spectrum-compatible records. International Journal 
        of Advanced Structural Engineering 5, 1-7.

    '''
    import numpy as np
    from scipy import signal

    nf = np.size(scales)
    dt = 1/fs
    n  = np.size(s)
    t = np.linspace(0,(n-1)*dt,n)
    centertime = np.median(t)

    coefs  = np.zeros((nf,n))
    for k in range(nf):
        wv = zumontw((t-centertime)/scales[k],omega,zeta)/np.sqrt(scales[k])
        coefs[k,:] = signal.fftconvolve(s, wv, mode='same')
        
    return coefs

def getdetails(t,s,C,scales,omega,zeta):
    '''
    getdetails - Generates the detail functions
    
    input:
        t: time vector [s]
        s:  signal being analyzed
        C:  matrix with the coeff. from the CWT
        scales: vector with the scales at which the CWT was performed
        omega, zeta: wavelet parameters
        
    returns:
        D: 2D array with the detail functions
        sr: reconstructed signal        
    '''
    import numpy as np
    from scipy import signal
    
    NS = np.size(scales)
    n = np.size(s)
    D    = np.zeros((NS,n))
    
    centertime = np.median(t)
    
    for k in range(NS):
        wv = zumontw((t-centertime)/scales[k],omega,zeta)  
        D[k,:] = -signal.fftconvolve(C[k,:], wv, mode='same')/(scales[k]**(5/2))
    
    sr = np.trapz(D.T,scales)  # signal reconstructed from the details
    ff = np.max(np.abs(s))/np.max(np.abs(sr))
    sr  = ff * sr
    D   = ff * D
    
    return D,sr

def basecorr(t,xg,CT,porder=-1,imax=80,tol=0.01):
    '''
    performs baseline correction
    
    references:
        
    Wilson, W.L. (2001), Three-Dimensional Static and Dynamic Analysis of Structures: 
    A Physical Approach with Emphasis on Earthquake Engineering, 
    Third Edition, Computers and Structures Inc., Berkeley, California, 2001.
    
    Suarez, L. E., & Montejo, L. A. (2007). Applications of the wavelet transform
    in the generation and analysis of spectrum-compatible records. 
    Structural Engineering and Mechanics, 27(2), 173-197.
    
    input:
        t: time vector [s]
        xg: time history of accelerations
        CT: time for correction [s]
        porder: order of the poynomial to perform initial detrending (default -1, no detrend) 
        imax: maximum number of iterations (default 80)
        tol: tolerance (percent of the max, default 0.01)
    
    return:
        vel: time history of velocities (original record)
        despl: time history of diplacements (original record)
        cxg: baseline-corrected time history of accelerarions
        cvel: baseline-corrected history of velocities
        cdespl: baseline-corrected history of displacements
    '''
    import numpy as np
    from scipy import integrate
    
    if porder>=0:
        pp = np.polynomial.Polynomial.fit(t, xg, deg=porder)
        xg = xg-pp(t) # initial detrend
    
    n = np.size(xg)
    cxg = np.copy(xg)  
    
    vel = integrate.cumulative_trapezoid(xg, t, initial=0)
    despl = integrate.cumulative_trapezoid(vel, t, initial=0)
    dt = t[1]-t[0]
    L  = int(np.ceil(CT/(dt))-1)   
    M  = n-L
        
    for q in range(imax):
        
      dU, ap, an = 0, 0, 0
      dV, vp, vn = 0, 0, 0
      
      for i in range(n-1):
          dU = dU + (t[-1]-t[i+1]) * cxg[i+1] * dt
    
      for i in range(L+1):
          aux = ((L-i)/L)*(t[-1]-t[i]) * cxg[i] * dt
          if aux >= 0:
              ap = ap + aux
          else:
              an = an + aux
    
      alfap = -dU/(2*ap)
      alfan = -dU/(2*an)
    
      for i in range(1,L+1):

          if cxg[i]>0:
              cxg[i] = (1 + alfap*(L-i)/L) * cxg[i]
          else:
              cxg[i] = (1 + alfan*(L-i)/L) * cxg[i]
              
      for i in range(n-1):
          dV = dV + cxg[i+1] * dt
          
      for i in range(M-1,n):
          auxv = ((i + 1 - M)/(n-M))*cxg[i]*dt
          if auxv >= 0:
              vp = vp + auxv
          else:
              vn = vn + auxv

      valfap = -dV/(2*vp)
      valfan = -dV/(2*vn)
    
      for i in range(M-1,n):
         
          if cxg[i]>0:
              cxg[i] = (1 + valfap*((i + 1 - M)/(n-M))) * cxg[i]
          else:
              cxg[i] = (1 + valfan*((i + 1 - M)/(n-M))) * cxg[i]
      
      cvel = integrate.cumulative_trapezoid(cxg, t, initial=0)
      cdespl = integrate.cumulative_trapezoid(cvel, t, initial=0)

      errv = np.abs(cvel[-1]/np.max(np.abs(cvel)))
      errd = np.abs(cdespl[-1]/np.max(np.abs(cdespl)))
    
      if errv <= tol and errd <= tol:
          break
      
    return vel,despl,cxg,cvel,cdespl

def baselinecorrect(sc,t,porder=-1,imax=80,tol=0.01):
    '''
    t: time vector [s]
    sc: time history of accelerations
    porder: order of the poynomial to perform initial detrending (default -1, no detrend) 
    imax: maximum number of iterations (default 80)
    tol: tolerance (percent of the max, default 0.01)

    baselinecorrect - performs baseline correction iteratively 
    calling basecorr
    
    references:
        
    Wilson, W.L. (2001), Three-Dimensional Static and Dynamic Analysis of Structures: 
    A Physical Approach with Emphasis on Earthquake Engineering, 
    Third Edition, Computers and Structures Inc., Berkeley, California, 2001.
    
    Suarez, L. E., & Montejo, L. A. (2007). Applications of the wavelet transform
    in the generation and analysis of spectrum-compatible records. 
    Structural Engineering and Mechanics, 27(2), 173-197.
    
    input:
        sc: uncorrected acceleration time series
        t: time vector
    returns:
        ccs,cvel,cdespl: corrected acc., vel. and disp.
        
    '''
    import numpy as np
    
    CT = np.max(np.array([1,t[-1]/20])) # time to correct
    print(f'using first and last {CT:.1f} seconds for baseline correction')
    vel,despl,ccs,cvel,cdespl = basecorr(t,sc,CT,porder=porder,imax=imax,tol=tol)
    kka = 1; flbc = True
    
    while any(np.isnan(ccs)):
        kka = kka + 1
        CTn = kka*CT
        print(f'using first and last {CTn:.1f} seconds for baseline correction')
        if CTn >= np.median(t):
            print('='*40)
            print('**baseline correction failed**')
            print('='*40)
            flbc = False; ccs = sc; cvel=vel; cdespl=despl
            break
        vel,despl,ccs,cvel,cdespl = basecorr(t,sc,CTn,porder=porder,imax=imax,tol=tol)
    if flbc:
        print('='*40)
        print('**baseline correction was succesful**')
        print('='*40)
        
    return ccs,cvel,cdespl

def PGAcorrection(targetPGA,t,s,maxit=1000):
    '''
    Parameters
    ----------
    targetPGA : target PGA [g]
    t : time vector
    s : acceleration time history [g]
    maxit: max # of iterations

    Returns
    -------
    sc: PGA corrected acceleration time history
    '''
    
    from scipy.signal import find_peaks
    import numpy as np
    
    sc = np.copy(s)
    motionPGA = np.max(np.abs(sc))
    cont=0
    if motionPGA>targetPGA:
        while motionPGA>targetPGA and cont<maxit:
            motionPGAloc = np.argmax(np.abs(sc))
            lamb = targetPGA/motionPGA
            peaks_neg, _ = find_peaks(-np.abs(sc))
            left_p = peaks_neg[np.where(peaks_neg<motionPGAloc)[0][-1]]
            right_p = peaks_neg[np.where(peaks_neg>motionPGAloc)[0][0]]
            mods = [1,1,lamb,1,1]
            tmods = [t[0],t[left_p],t[motionPGAloc],t[right_p],t[-1]]
            mod = np.interp(t,tmods,mods)
            sc = mod*sc
            motionPGA = np.max(np.abs(sc))
            cont=cont+1
    elif motionPGA<targetPGA and cont<maxit:
        while motionPGA<targetPGA:
            motionPGAloc = np.argmax(np.abs(sc))
            lamb = targetPGA/motionPGA
            peaks_neg, _ = find_peaks(-np.abs(sc))
            left_p = peaks_neg[np.where(peaks_neg<motionPGAloc)[0][-1]]
            right_p = peaks_neg[np.where(peaks_neg>motionPGAloc)[0][0]]
            mods = [1,1,lamb,1,1]
            tmods = [t[0],t[left_p],t[motionPGAloc],t[right_p],t[-1]]
            mod = np.interp(t,tmods,mods)
            sc = mod*sc
            motionPGA = np.max(np.abs(sc))
            cont=cont+1
    
    if cont==maxit:
        print('='*40)
        print(f'PGA correction failed, target PGA: {targetPGA:.2f}, current PGA: {motionPGA:.2f}')
        print('='*40)
    else:
        print('='*40)
        print(f'PGA correction sucessful, target PGA: {targetPGA:.2f}, current PGA: {motionPGA:.2f}')
        print('='*40)
        
    return sc

def logfrequencies(start_freq, end_freq, points_per_decade):
    
    '''
    Calculates logarithmically spaced frequencies within a given range.

    Parameters
    ----------
        start_freq (float): The starting frequency in Hz.
        end_freq (float): The ending frequency in Hz.
        points_per_decade (int): The desired number of points per frequency decade.

    Returns
    -------
        numpy.ndarray: An array of logarithmically spaced frequencies.
    '''
    import numpy as np
    
    num_decades = np.log10(end_freq / start_freq)  # Number of decades in the range
    total_points = int(num_decades * points_per_decade + 1)  # Ensure at least 100 points/decade

    # Use logspace to create logarithmically spaced frequencies
    frequencies = np.logspace(np.log10(start_freq), np.log10(end_freq), total_points)

    return frequencies

def SignificantDuration(s,t,ival=5,fval=75):
    '''
    Estimates significant duration and Arias Intensity
    
    Parameters
    ----------
    s : 1d array
        acceleration time-history
    t : 1d array
        time vector
    ival : float, optional
        Initial % of Arias Intensity to estimate significant duration. 
        The default is 5.
    fval :float, optional
        Final % of Arias Intensity to estimate significant duration. 
        The default is 75.

    Returns
    -------
    sd : float
        significant duration
    AIcumnorm : 1d array
        normalized cummulative AI
    AI : float
        Arias Intensity (just the integral, 2*pi/g not included)
    t1 : float
        initial time for sd
    t2 : float
        final time for sd

    '''
    from scipy import integrate
    AIcum = integrate.cumulative_trapezoid(s**2, t, initial=0)
    AI = AIcum[-1]
    AIcumnorm = AIcum/AI
    t_strong = t[(AIcumnorm>=ival/100)&(AIcumnorm<=fval/100)]
    t1, t2 = t_strong[0], t_strong[-1]
    sd = t2-t1
    return sd,AIcumnorm,AI,t1,t2

def PSDFFTEq(so,fs,alphaw=0.1,duration=(5,75),nFFT='nextpow2',basefornFFT = 0, overlap=20, detrend='linear'):
    '''
    Calculates the power spectral densityof earthquake acceleration time-series
    FFT is normalized by dy dt, FFT/PSD is calcualted over the stong motion duration
    returnd the one-sided PSD and a "smoothed" version by taking the average 
    over a frequency window width of user defined % of the subject frequency. 
    
    Parameters
    ----------
    so : 1D array
        acceleration time-series
        
    fs : integer
        sampling frequency
        
    alphaw : Optional, float, tukey window parameter [0 1], defaults to 0.1
             0 -> rectangular, 1 -> Hann
    
    duration: Optional, tuple or None
    
              (a,b) stong motion duration used to defined the portion of the signal 
              used to calculate FFT and PSD.Defined as the duration corresponding 
              to a a%-to-b% rise of the cumulative Arias energy
              
              None: the whole signal is used
              
              The default is (5,75).
        
    nFFT : Optional, number of points to claculate the FFT, options:
        
        
        'nextpow2': zero padding until the mext power of 2 is reached
        
        'same': keep the number of points equal to the number of poitns in
                the signal
                
        An integer:  
        If n is smaller than the length of the input, the input is cropped. 
        If it is larger, the input is padded with zeros. 
    
        Defaults to 'nextpow2'
        
    basefornFFT: Optional, interger 0 or 1, whether nFFT is determined based on
                 the original/total number of datapoints in the signal or based
                 on the strong motion part.
                 
                 0 -> total number, 1 -> strong motion part
                 
                 defaults to 0
        
    overlap : Optional, float
        
        ±% frequency window width to smooth PSD 
        The default is 20.
        
    detrend = None, 'linear' or 'constant' (defaults to linear)
        'linear' (default), the result of a linear least-squares fit to data is subtracted from data. 
        'constant', only the mean of data is subtracted

    Returns
    -------
    mags : One-sided Fourier amplitudes
    PSD  : One-sided power spectral density
    PSDavg : One-sided average power spectral density
    freqs :  Vector with the frequencies
    sd : duration used to calculated FFT/SD
    AI : Arias intensity of the signal (Just the integral, units depend on the
                                        initial signal units, pi/2g is not applied)
    '''
    import numpy as np
    from scipy import signal
    
    no = np.size(so)
    dt = 1/fs
    t = np.linspace(0,(no-1)*dt,no)   # time vector 
      
    if duration==None:
        duration = (0,100)
        
    if len(duration)==2 :
        sd,AIcum,AI,t1,t2 = SignificantDuration(so,t,ival=duration[0],fval=duration[1])
        locs = np.where((t>=t1-dt/2)&(t<=t2+dt/2))
        nlocs = np.size(locs)
        s = so[locs]
        window = signal.windows.tukey(nlocs,alphaw)
        if detrend=='linear':
            s = signal.detrend(s,type='linear')
        elif detrend=='constant':
            s = signal.detrend(s,type='constant')
        elif detrend!=None:
            print('*** error definig detrend in PSDFFTEq function ***')
            return
        s = window*s
        
    else:
        print('*** error definig duration in PSDFFTEq function ***')
        return
    
    if basefornFFT == 0:
        n = no
    else:
        n = nlocs
        
    if nFFT=='nextpow2':
        nFFT = int(2**np.ceil(np.log2(n)))
    elif nFFT=='same':
        nFFT = n
    elif not isinstance(nFFT, int):
        print('*** error definig nFFT in PSDFFTEq function ***')
        return
        
    fres = fs/nFFT; nfrs = int(np.ceil(nFFT/2))
    freqs = fres*np.arange(0,nfrs+1,1)   # vector with frequencies
    
    Fs = np.fft.fft(s,nFFT)
    mags = dt*np.abs(Fs[:nfrs+1])
    
    PSD = 2*mags**2/(2*np.pi*sd)
    
    PSDavg = np.copy(PSD)
    overl = overlap/100
    if overl>0:
        for k in range(1,nfrs-1):
            lim1 = (1-overl)*freqs[k]
            lim2 = (1+overl)*freqs[k]
            
            if freqs[0]>lim1:
                lim1 = freqs[0]
                lim2 = freqs[k]+(freqs[k]-freqs[0])
            if freqs[-1]<lim2:
                lim2 = freqs[-1]
                lim1 = freqs[k]-(freqs[-1]-freqs[k])
                
            locsf = np.where((freqs>=lim1)&(freqs<=lim2))
            PSDavg[k]=np.mean(PSD[locsf])

    
    return mags,PSD,PSDavg,freqs,sd,AI,t1,t2

def log_interp(x, xp, fp):
    import numpy as np
    np.seterr(divide = 'ignore') 
    logx = np.log10(x)
    logxp = np.log10(xp)
    logfp = np.log10(fp)
    return np.power(10.0, np.interp(logx, logxp, logfp))

def load_PEERNGA_record(filepath):
    '''
    Load record in .at2 format (PEER NGA Databases)

    Input:
        filepath : file path for the file to be load
        
    Returns:
    
        acc : vector wit the acceleration time series
        dt : time step
        npts : number of points in record
        eqname : string with year_name_station_component info

    '''

    import numpy as np

    with open(filepath) as fp:
        line = next(fp)
        line = next(fp).split(',')
        year = (line[1].split('/'))[2]
        eqname = (year + '_' + line[0].strip() + '_' + 
                  line[2].strip() + '_comp_' + line[3].strip())
        line = next(fp)
        line = next(fp).split(',')
        npts = int(line[0].split('=')[1])
        dt = float(line[1].split('=')[1].split()[0])
        acc = np.array([p for l in fp for p in l.split()]).astype(float)
    
    return acc,dt,npts,eqname

def RSFD_S(T,s,z,dt):
    '''   
    luis.montejo@upr.edu 
    
    Response spectra (operations in the frequency domain)
    Faster than RSFD as only computes PSA, PSV, SD
    Input:
        T: vector with periods (s)
        s: acceleration time series
        z: damping ratio
        dt: time steps for s
    
    Returns:
        PSA, PSV, SD
    
    '''
    import numpy as np
    from numpy.fft import fft, ifft
    
    pi = np.pi

    npo = np.size(s)
    nT  = np.size(T)
    SD  = np.zeros(nT)

    
    n = int(2**np.ceil(np.log2(npo+10*np.max(T)/dt)))  # add zeros to provide enough quiet time
    fs=1/dt;
    s = np.append(s,np.zeros(n-npo))
    
    fres  = fs/n                            # frequency resolution
    nfrs  = int(np.ceil(n/2))               # number of frequencies
    freqs = fres*np.arange(0,nfrs+1,1)      # vector with frequencies
    ww    = 2*pi*freqs                      # vector with frequencies [rad/s]
    ffts = fft(s);         
    
    m = 1
    for kk in range(nT):
        w = 2*pi/T[kk] ; k=m*w**2; c = 2*z*m*w
        
        H1 = 1       / ( -m*ww**2 + k + 1j*c*ww )  # Transfer function (half) - Receptance
        
        H1 = np.append(H1,np.conj(H1[n//2-1:0:-1]))
        H1[n//2] = np.real(H1[n//2])     # Transfer function (complete) - Receptance
        
        
        CoF1 = H1*ffts   # frequency domain convolution
        d = ifft(CoF1)   # go back to the time domain (displacement)
        SD[kk] = np.max(np.abs(d))
            
    
    PSV = (2*pi/T)* SD
    PSA = (2*pi/T)**2 * SD
    
    return PSA, PSV, SD