'''
Example on the use of ReqPyPSD_Module
luis.montejo@upr.edu

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
'''
import numpy as np
import matplotlib.pyplot as plt
from ReqPyPSD_Module import ReqPyPSD, load_PEERNGA_record
import warnings
warnings.filterwarnings( "ignore")

plt.close('all')

SeedRecordFile = 'RSN6953_DARFIELD_PRPCW.AT2'

targetPSAname = 'WUS_M7.5_R75_Frequencies.txt'       # target spectrum (F,PSA[g])
targetPGA = 1                                        # target PGA [g]
targetPSDname = 'NRC_WUS_M7.5_R75_SD575_12.50s.txt'  # target PSD (F,PSD[m2/s3])

# load target PSA:
TPSA = np.loadtxt(targetPSAname)
F_PSA = TPSA[:, 0]              # original target PSA frequencies
targetPSA = TPSA[:, 1]          # original target PSA
nF_PSA = np.size(F_PSA)

# load target PSD:
TPSD = np.loadtxt(targetPSDname) 
F_PSD = TPSD[:,0]        # original target PSD frequencies
targetPSD = TPSD[:,1]    # original target PSD
nF_PSD = np.size(F_PSD)

# load seed record:
s,dt,nt,eqname = load_PEERNGA_record(SeedRecordFile)  # dt: time step, s: accelertion series
fs = 1/dt                                             # sampling frequency (Hz)

sc,sca = ReqPyPSD(s,fs,F_PSA,targetPSA,F_PSD,targetPSD,targetPGA, 
                  F1PSA=0.2,F2PSA=50,F1PSD=0.3,F2PSD=30,zi=0.05,
                  PSA_poly_order=4,PSAPSD_poly_order=4,
                  NS=300,nit=20,maxit=1000,plots=1,nameOut='ReqPyOut')
