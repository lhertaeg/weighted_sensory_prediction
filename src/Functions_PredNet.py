 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 08:34:59 2021

@author: loreen.hertaeg
"""

# %% import packages

import numpy as np
import os
from numba import njit
from typing import NamedTuple

dtype = np.float32

# %% Classes

class Neurons(NamedTuple):
        
        NCells: list = np.array([140,20,20,20], dtype=np.int32)
        tau_inv_E: dtype = dtype(1.0/60.0)
        tau_inv_I: dtype = dtype(1.0/2.0)
        
        
        
class Network:
    def __init__(self, Neurons, NetPar_PE, mLE, bool_nPE, bool_pPE, neurons_motor):
        
        # Pre-processing
        NE, NP, NS, NV = np.int32(Neurons.NCells)
        N_nPE = sum(bool_nPE)
        N_pPE = sum(bool_pPE)
        
        P_Pred_flag = neurons_motor[NE:(NE+NP)]
        S_Pred_flag = neurons_motor[(NE+NP):(NE+NP+NS)]
        V_Pred_flag = neurons_motor[(NE+NP+NS):]
        
        # PE network connectivity (PE-PE)
        self.wEP = NetPar_PE.wEP
        self.wED = NetPar_PE.wED
        self.wDE = NetPar_PE.wDE
        self.wDS = NetPar_PE.wDS 
        self.wPE = NetPar_PE.wPE
        self.wPP = NetPar_PE.wPP
        self.wPS = NetPar_PE.wPS
        self.wPV = NetPar_PE.wPV
        self.wSE = NetPar_PE.wSE
        self.wSP = NetPar_PE.wSP
        self.wSS = NetPar_PE.wSS
        self.wSV = NetPar_PE.wSV
        self.wVE = NetPar_PE.wVE
        self.wVP = NetPar_PE.wVP
        self.wVS = NetPar_PE.wVS
        self.wVV = NetPar_PE.wVV
        
        # Connectivity between line attractor and PE (L-PE)
        wLE = np.zeros((1,NE), dtype=dtype)
        wLE[0,:] = mLE * bool_pPE/N_pPE - mLE * bool_nPE/N_nPE
        
        wDL = np.zeros((NE,1), dtype=dtype)
        wDL[:,0] = 1
        
        wPL = np.zeros((NP,1), dtype=dtype)
        wPL[:,0] = 1 * P_Pred_flag
        
        wSL = np.zeros((NS,1), dtype=dtype)
        wSL[:,0] = 1 * S_Pred_flag
        
        wVL = np.zeros((NV,1), dtype=dtype)
        wVL[:,0] = 1 * V_Pred_flag
        
        self.wLE = wLE
        self.wEL = np.zeros((NE,1), dtype=dtype)
        self.wDL = wDL 
        self.wPL = wPL 
        self.wSL = wSL 
        self.wVL = wVL 
        
        
class InputStructure:
    def __init__(self, Neurons, InPar):
            
        Xsoma = dtype(InPar.inp_ext_soma)
        Xdend = dtype(InPar.inp_ext_dend) 
        
        NCells = Neurons.NCells
        NE, NP, NS, NV = np.int32(NCells)
       
        # External, fixed inputs
        inp_ext_soma = np.zeros(sum(NCells))
        inp_ext_dend = np.zeros(NCells[0])
        
        inp_ext_soma[:NE] = Xsoma[:NE]
        inp_ext_soma[NE:(NE+NP)] = Xsoma[NE:(NE+NP)]
        inp_ext_soma[(NE+NP):(NE+NP+NS)] = Xsoma[(NE+NP):(NE+NP+NS)]
        inp_ext_soma[(NE+NP+NS):] = Xsoma[(NE+NP+NS):]
        inp_ext_dend = Xdend
        
        self.inp_ext_soma: dtype = inp_ext_soma
        self.inp_ext_dend: dtype = inp_ext_dend 
        
        # Flag to indicate which neurons receive stimulus (visual)
        ind_break = np.cumsum(np.int32(NCells),dtype=np.int32)[:-1]
        neurons_visual_P, neurons_visual_S, neurons_visual_V = np.split(InPar.neurons_visual, ind_break)[1:]

        self.Flag_visual_P = neurons_visual_P
        self.Flag_visual_S = neurons_visual_S
        self.Flag_visual_V = neurons_visual_V
        
        
class Stimulation:
    def __init__(self, stimuli, stimuli_std, SD_individual = dtype(1)):
        
        self.SD = SD_individual
        self.stimuli = dtype(stimuli)
        self.stimuli_std: dtype = dtype(stimuli_std)

        
class Activity_Zero:
    def __init__(self, NeuPar,  L0, r0 = dtype([0,0,0,0,0])):

        NCells = NeuPar.NCells
        Nb = np.cumsum(NCells, dtype=np.int32)
    
        if len(r0)<sum(NCells):
            self.rE0 = np.repeat(r0[0],NCells[0])
            self.rP0 = np.repeat(r0[1],NCells[1])
            self.rS0 = np.repeat(r0[2],NCells[2])
            self.rV0 = np.repeat(r0[3],NCells[3])
            self.rD0 = np.repeat(r0[4],NCells[0])
        else:
            self.rE0, self.rP0, self.rS0, self.rV0 , self.rD0  = np.split(r0,Nb)
        
        self.rL0 = np.array([L0], dtype=dtype)   

        
class Simulation(NamedTuple): 
       
       dt: dtype = dtype(0.1)
       stim_duration: dtype = dtype(1000.0) 


# %% Functions

@njit(cache=True)
def drdt(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, 
         wSS, wSV, wVE, wVP, wVS, wVV, wLE, wEL, wDL, wPL, wSL, wVL, 
         rE, rD, rP, rS, rV, rL, Stim_E, Stim_D, Stim_P, Stim_S, Stim_V): 
    
    # Line attractor neurons
    drL = tau_inv_E * (wLE @ rE)
    
    # PE network
    drE = tau_inv_E * (-rE + wED @ rD + wEP @ rP + wEL @ rL + Stim_E) 
    drD = tau_inv_E * (-rD + wDE @ rE + wDS @ rS + wDL @ rL + Stim_D) 
    drP = tau_inv_I * (-rP + wPE @ rE + wPP @ rP + wPS @ rS + wPV @ rV + wPL @ rL + Stim_P) 
    drS = tau_inv_I * (-rS + wSE @ rE + wSP @ rP + wSS @ rS + wSV @ rV + wSL @ rL + Stim_S) 
    drV = tau_inv_I * (-rV + wVE @ rE + wVP @ rP + wVS @ rS + wVV @ rV + wSL @ rL + Stim_V) 
    
    return drE, drD, drP, drS, drV, drL


def NetworkDynamics(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, 
                    wSS, wSV, wVE, wVP, wVS, wVV, wLE, wEL, wDL, wPL, wSL, wVL, 
                    rE, rD, rP, rS, rV, rL, Stim_E, Stim_D, Stim_P, Stim_S, Stim_V, dt):
    
    rE0 = rE.copy()
    rD0 = rD.copy()
    rP0 = rP.copy()
    rS0 = rS.copy()
    rV0 = rV.copy()
    rL0 = rL.copy()
    
    drE1, drD1, drP1, drS1, drV1, drL1 = drdt(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, 
                                              wSS, wSV, wVE, wVP, wVS, wVV, wLE, wEL, wDL, wPL, wSL, wVL, 
                                              rE0, rD0, rP0, rS0, rV0, rL0, Stim_E, Stim_D, Stim_P, Stim_S, Stim_V)
        
    rE0[:] += dt * drE1
    rD0[:] += dt * drD1
    rP0[:] += dt * drP1
    rS0[:] += dt * drS1
    rV0[:] += dt * drV1
    rL0[:] += dt * drL1
    
    drE2, drD2, drP2, drS2, drV2, drL2 = drdt(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, 
                                              wSS, wSV, wVE, wVP, wVS, wVV, wLE, wEL, wDL, wPL, wSL, wVL, 
                                              rE0, rD0, rP0, rS0, rV0, rL0, Stim_E, Stim_D, Stim_P, Stim_S, Stim_V)
    
    
    rE[:] += dt/2 * (drE1 + drE2)
    rD[:] += dt/2 * (drD1 + drD2)
    rP[:] += dt/2 * (drP1 + drP2) 
    rS[:] += dt/2 * (drS1 + drS2) 
    rV[:] += dt/2 * (drV1 + drV2) 
    rL[:] += dt/2 * (drL1 + drL2)
    
    rE[rE<0] = 0
    rP[rP<0] = 0
    rS[rS<0] = 0
    rV[rV<0] = 0
    rD[rD<0] = 0
    rL[rL<0] = 0

    return


def RunPredNet(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder: str, fln: str = ''):
    
    ### Neuron parameters for PE network
    NCells = NeuPar.NCells
    ind_break = np.cumsum(NCells[1:],dtype=np.int32)[:-1]
    
    tau_inv_E = NeuPar.tau_inv_E
    tau_inv_I = NeuPar.tau_inv_I
    
    ### Network parameters
    wEP = NetPar.wEP
    wED = NetPar.wED
    wDE = NetPar.wDE
    wDS = NetPar.wDS 
    wPE = NetPar.wPE
    wPP = NetPar.wPP
    wPS = NetPar.wPS
    wPV = NetPar.wPV
    wSE = NetPar.wSE
    wSP = NetPar.wSP
    wSS = NetPar.wSS
    wSV = NetPar.wSV
    wVE = NetPar.wVE
    wVP = NetPar.wVP
    wVS = NetPar.wVS
    wVV = NetPar.wVV
    wLE = NetPar.wLE
    wEL = NetPar.wEL
    wDL = NetPar.wDL
    wPL = NetPar.wPL
    wSL = NetPar.wSL
    wVL = NetPar.wVL
    
    ## Stimulation protocol & Inputs
    SD = StimPar.SD
    stimuli = iter(StimPar.stimuli)
    stimuli_std = iter(StimPar.stimuli_std)
    NStim = np.int32(len(StimPar.stimuli))
    
    inp_ext_soma = InPar.inp_ext_soma
    inp_ext_dend = InPar.inp_ext_dend
    
    inp_ext_E = inp_ext_soma[:NCells[0]]
    inp_ext_P, inp_ext_S, inp_ext_V = np.split(inp_ext_soma[NCells[0]:], ind_break)
    inp_ext_D = inp_ext_dend
    
    Flag_visual_P = InPar.Flag_visual_P
    Flag_visual_S = InPar.Flag_visual_S
    Flag_visual_V = InPar.Flag_visual_V
      
    ### Simulation parameters
    dt = SimPar.dt
    stim_duration = SimPar.stim_duration
    num_time_steps = np.int32(stim_duration/dt)    
    
    ### Initial activity levels
    rE = RatePar.rE0
    rD = RatePar.rD0
    rP = RatePar.rP0
    rS = RatePar.rS0
    rV = RatePar.rV0
    rL = RatePar.rL0
    
    ### Initialisation
    Stim_E = np.zeros(NCells[0], dtype=dtype)
    Stim_P = np.zeros(NCells[1], dtype=dtype)
    Stim_S = np.zeros(NCells[2], dtype=dtype)
    Stim_V = np.zeros(NCells[3], dtype=dtype)
    Stim_D = np.zeros(NCells[0], dtype=dtype)
    
    Stim_Mean_E = np.zeros(NCells[0], dtype=dtype)
    Stim_Mean_P = np.zeros(NCells[1], dtype=dtype)
    Stim_Mean_S = np.zeros(NCells[2], dtype=dtype)
    Stim_Mean_V = np.zeros(NCells[3], dtype=dtype)
    Stim_Mean_D = np.zeros(NCells[0], dtype=dtype)
    
    noise_E = np.zeros((NCells[0], num_time_steps), dtype=dtype)
    noise_P = np.zeros((NCells[1], num_time_steps), dtype=dtype)
    noise_S = np.zeros((NCells[2], num_time_steps), dtype=dtype)
    noise_V = np.zeros((NCells[3], num_time_steps), dtype=dtype)
    noise_D = np.zeros((NCells[0], num_time_steps), dtype=dtype)
    noise_stimuli = np.zeros(num_time_steps, dtype=dtype)
    
    ### Path & file for data to be stored
    path = 'Results/Data/' + folder
    if not os.path.exists(path):
        os.mkdir(path)

    #fp1 = open(path +'/Data_StaticPredNet_PE_' + fln + '.dat','w') 
    fp2 = open(path +'/Data_StaticPredNet_P_' + fln + '.dat','w') 
    
    ### Main loop
    for s in range(NStim):
        
        if (s % 20 == 0):
            print('Stimuli', str(s+1), '/', str(NStim))
        
        stim = next(stimuli)
        stim_std = next(stimuli_std)
        noise_stimuli[:] = np.random.normal(0, stim_std, size=num_time_steps)
        
        noise_E[:] = np.random.normal(0,SD,size=(NCells[0], num_time_steps))
        noise_P[:] = np.random.normal(0,SD,size=(NCells[1], num_time_steps))
        noise_S[:] = np.random.normal(0,SD,size=(NCells[2], num_time_steps))
        noise_V[:] = np.random.normal(0,SD,size=(NCells[3], num_time_steps))
        noise_D[:] = np.random.normal(0,SD,size=(NCells[0], num_time_steps))
            
        Stim_Mean_E[:] = inp_ext_E + stim 
        Stim_Mean_P[:] = inp_ext_P + stim * (Flag_visual_P==1)
        Stim_Mean_S[:] = inp_ext_S + stim * (Flag_visual_S==1)
        Stim_Mean_V[:] = inp_ext_V + stim * (Flag_visual_V==1)
        Stim_Mean_D[:] = inp_ext_D
        
        for tstep in range(num_time_steps):

            Stim_E[:] = Stim_Mean_E + noise_E[:, tstep] + noise_stimuli[tstep]
            Stim_D[:] = Stim_Mean_D + noise_D[:, tstep]
            Stim_P[:] = Stim_Mean_P + noise_P[:, tstep] + noise_stimuli[tstep] * (Flag_visual_P==1)
            Stim_S[:] = Stim_Mean_S + noise_S[:, tstep] + noise_stimuli[tstep] * (Flag_visual_S==1)
            Stim_V[:] = Stim_Mean_V + noise_V[:, tstep] + noise_stimuli[tstep] * (Flag_visual_V==1)
            
            NetworkDynamics(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, 
                            wSS, wSV, wVE, wVP, wVS, wVV, wLE, wEL, wDL, wPL, wSL, wVL, 
                            rE, rD, rP, rS, rV, rL, Stim_E, Stim_D, Stim_P, Stim_S, Stim_V, dt)
            
        #     #if (tstep % 10 == 0): if you want to include several time points per stimulus, then this must be uncommented and the part below needs to be intended
            
        # fp1.write("%f" % (s * stim_duration + (tstep+1) * dt))
        # for i in range(NCells[0]):
        #     fp1.write(" %f" % rE[i])
        # # for i in range(NCells[1]):
        # #     fp1.write(" %f" % rP[i])
        # # for i in range(NCells[2]):
        # #     fp1.write(" %f" % rS[i])
        # # for i in range(NCells[3]):
        # #     fp1.write(" %f" % rV[i])
        # # for i in range(NCells[0]):
        # #     fp1.write(" %f" % rD[i])
        # fp1.write("\n")
             
        fp2.write("%f" % (s * stim_duration + (tstep+1) * dt))
        fp2.write(" %f" % rL[0])
        fp2.write("\n") 
                
    #fp1.closed
    fp2.closed
    
    return


##############################################################################
###################    MFN PE circuit with attractor    ######################
##############################################################################


def RateDynamics_MFN(tau_inv_E, tau_inv_I, U, V, W, r, L, Stim, dt):
    
    # Initialise
    r0 = r.copy()
    L0 = L.copy()
    dr1 = np.zeros(len(r), dtype=dtype)
    dr2 = np.zeros(len(r), dtype=dtype)
    dL1 = np.zeros(len(L), dtype=dtype)
    dL2 = np.zeros(len(L), dtype=dtype)
    
    # RK 2nd order
    dL1 = tau_inv_E * (U @ r0)
    dr1 = -r0 + W @ r0 + V @ L0 + Stim
    dr1[:4] *= tau_inv_E 
    dr1[4:] *= tau_inv_I 

    r0[:] += dt * dr1[:]
    L0 += dt * dL1
    
    dL2 = tau_inv_E * (U @ r0)
    dr2 = -r0 + W @ r0 + V @ L0 + Stim
    dr2[:4] *= tau_inv_E 
    dr2[4:] *= tau_inv_I
    
    r += dt/2 * (dr1 + dr2)
    L += dt/2 * (dL1 + dL2)
    
    # Rectify
    r[r<0] = 0

    return


def RunPredNet_MFN(U, V, W, tau_inv, stim_duration, dt, fixed_input, stimuli, stimuli_std, SD, folder: str, fln: str = '', VS=1, VV=0):
    
    tau_inv_E, tau_inv_I  = tau_inv
    neurons_sensory = np.array([1, 1, 0, 0, 1, 0, VS, VV])
    
    N = len(neurons_sensory)
    num_time_steps = np.int32(stim_duration/dt)
    
    r = np.zeros(N, dtype=dtype)
    L = np.zeros(1, dtype=dtype)
    
    path = 'Results/Data/' + folder
    fp = open(path +'/Data_StaticNetwork_MFN_' + str(fln) + '.dat','w')   
    
    # main loop
    for s, stim in enumerate(stimuli):
        
        stim_with_noise = np.random.normal(stim, stimuli_std[s], size=num_time_steps)
        individual_noise = np.random.normal(0, SD, size=(8, num_time_steps))
        
        for tstep in range(num_time_steps):
              
            # Numerical integration of mean-field rate dynamics
            sensory_input = fixed_input + stim_with_noise[tstep] * neurons_sensory + individual_noise[:, tstep]
            RateDynamics_MFN(tau_inv_E, tau_inv_I, U,V, W, r, L, sensory_input, dt)
            
            # write in file
            fp.write("%f" % (s*stim_duration + (tstep+1)*dt))
            for i in range(N):
                fp.write(" %f" % r[i])
            fp.write(" %f" % L)
            fp.write("\n") 
        
    fp.closed
    return
    