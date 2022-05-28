#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 14 2021

@author: loreen.hertaeg
"""

# %% import packages

import numpy as np
import tables
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
    def __init__(self, Neurons, weights_mean, conn_prob, weight_name, temp_name, uni_low: dtype = dtype(0.5), 
                 uni_high: dtype = dtype(1.5), flag_hetero: int = np.int32(1), flag_conn: int = np.int32(1)):
            
            ### Connection probabilty (either all-to-all or specified by conn_prob)
            if flag_conn==0:
                conn_prob = 1*(weights_mean!=0)
            else:
                conn_prob = conn_prob
                
            if not isinstance(uni_high, np.ndarray):
                uni_high = uni_high * np.ones_like(weights_mean)
                
            if not isinstance(uni_low, np.ndarray):
                uni_low = uni_low * np.ones_like(weights_mean)
            
            NUnits = np.insert(Neurons.NCells, 0, Neurons.NCells[0])
            NCon = np.round(conn_prob * NUnits).astype(np.int32)
            self.NCon = NCon
            self.weight_name = weight_name
            self.weights_mean = weights_mean
            
            ### Connection strengths between neuons
            for i in range(25):
                
                m,n = np.unravel_index(i,(5,5))
                Mtx = np.zeros((NUnits[m],NUnits[n]), dtype=dtype)
                
                if NCon[m,n]>0:
                    
                    # establish connections
                    if m==n:
                        for l in range(NUnits[m]):
                            r = np.array([0] * (NUnits[n]-1-NCon[m,n]) + [1] * NCon[m,n], dtype=dtype)
                            np.random.shuffle(r)
                            if flag_conn==1:
                                r = np.insert(r,l,0)         
                            Mtx[l,:] = r            
                    else:
                        for l in range(NUnits[m]):
                            r = np.array([0] * (NUnits[n]-NCon[m,n]) + [1] * NCon[m,n], dtype=dtype)
                            np.random.shuffle(r)      
                            Mtx[l,:] = r 
                    if weight_name[m,n]=='wED':
                        Mtx = np.identity(NUnits[0], dtype=dtype)
                        NCon[m,n] = 1
                        
                    # assign weights
                    if flag_hetero==1:
                        Mtx[:] *= dtype(np.random.uniform(uni_low[m,n] * weights_mean[m,n], uni_high[m,n] * weights_mean[m,n],
                                                          size=(NUnits[m],NUnits[n])))/NCon[m,n]
                    elif flag_hetero==0:
                        Mtx[:] *= weights_mean[m,n]/NCon[m,n]
                
                # Save
                exec('self.' + weight_name[m][n] + ' = Mtx')
                exec('self.' + temp_name[m][n] + ' = (Mtx!=dtype(0))') 
                

class InputStructure:
    def __init__(self, Neurons, Network, VE = dtype(1), VP = dtype(1), VS = dtype(1),
                 VV = dtype(0), r0 = None,  Xternal = None):
            
        # Assign input type to each neuron
        NCells = Neurons.NCells
        Nb = np.cumsum(NCells, dtype=np.int32) 
        NE, NP, NS, NV = NCells
        num_VE, num_VP, num_VS, num_VV  = int(VE*NE), int(VP*NP), int(VS*NS), int(VV*NV) 
    
        vis_E = np.array([1] * num_VE + [0] * (NE - num_VE), dtype=dtype)
        vis_P = np.array([1] * num_VP + [0] * (NP - num_VP), dtype=dtype)
        vis_S = np.array([1] * num_VS + [0] * (NS - num_VS), dtype=dtype)
        vis_V = np.array([1] * num_VV + [0] * (NV - num_VV), dtype=dtype)    
    
        neurons_visual = np.zeros(sum(NCells), dtype=dtype)
        neurons_visual[:Nb[0]] = vis_E
        neurons_visual[Nb[0]:Nb[1]] = vis_P
        neurons_visual[Nb[1]:Nb[2]] = vis_S
        neurons_visual[Nb[2]:] = vis_V
        
        neurons_motor = np.zeros(sum(NCells), dtype=dtype)
        neurons_motor[:Nb[0]] = 1
        neurons_motor[Nb[0]:Nb[1]] = 1 - neurons_visual[Nb[0]:Nb[1]]
        neurons_motor[Nb[1]:Nb[2]] = 1 - neurons_visual[Nb[1]:Nb[2]]
        neurons_motor[Nb[2]:] = 1 - neurons_visual[Nb[2]:]
        
        self.neurons_visual: list = neurons_visual
        self.neurons_motor: list = neurons_motor
        
        # External, fixed inputs
        inp_ext_soma, inp_ext_dend = BackgroundInput(Neurons, Network, r0, Xternal)
        
        self.inp_ext_soma: dtype = inp_ext_soma
        self.inp_ext_dend: dtype = inp_ext_dend
        
        
class Stimulation:
    def __init__(self, stim_high, stim_low = dtype(0), SD = dtype(1), 
                 num_stim = 7, p_phases = dtype([0,1,0]), flg_test = 1):
        
        self.SD: dtype = SD
        
        # Stimulation protocol
        if flg_test==1:
            stimuli = np.array([0,1,0,1,0,1,0]) 
            stim_type = np.array([0,0,0,-1,0,1,0])
            stim_visual = stim_high * ((stimuli==1) & (stim_type>=0)) + stim_low * ((stimuli==1) & (stim_type<0))
            stim_motor = stim_low * ((stimuli==1) & (stim_type>0)) + stim_high * ((stimuli==1) & (stim_type<=0))
        else:
            stimuli_high = np.tile(np.array([0,1],dtype=dtype), num_stim) 
            stimuli_low = np.tile(np.array([0,1],dtype=dtype), num_stim) 
            
            stim_type = np.zeros_like(stimuli_high)
            stim_type[1::2] = np.random.choice([-1,0,1],size=num_stim,p=p_phases) 
            
            stimuli_high[:] = stimuli_high * dtype(np.random.uniform(0, stim_high, size=2*num_stim))
            if stim_low==0:
                stimuli_low[:] = np.zeros(2*num_stim, dtype=dtype)
            else:
                stimuli_low[:] = stimuli_high * dtype(np.random.uniform(stim_low, 1, size=2*num_stim))
            
            stim_visual = stimuli_high * (stim_type>=0) + stimuli_low * (stim_type<0)
            stim_motor = stimuli_high * (stim_type<=0) + stimuli_low * (stim_type>0)
            
        self.stim_type = np.int32(np.sign(stim_visual-stim_motor))
        self.stim_visual: dtype = stim_visual
        self.stim_motor: dtype = stim_motor
        

class Activity_Zero:
    def __init__(self, Neurons, r0 = dtype([0,0,0,0,0])):

        NCells = Neurons.NCells
        Nb = np.cumsum(NCells, dtype=np.int32)
    
        if len(r0)<sum(NCells):
            self.rE0 = np.repeat(r0[0],NCells[0])
            self.rP0 = np.repeat(r0[1],NCells[1])
            self.rS0 = np.repeat(r0[2],NCells[2])
            self.rV0 = np.repeat(r0[3],NCells[3])
            self.rD0 = np.repeat(r0[4],NCells[0])
        else:
            self.rE0, self.rP0, self.rS0, self.rV0 , self.rD0  = np.split(r0,Nb)  
            
            
class Learning: 
    def __init__(self, Neurons, eta_mean, eta_name, rE_target = dtype(0), rD_target = dtype(0),
                 dw_min = -np.inf, dw_max = np.inf, plasticity_approx=1):
        
        NUnits = np.insert(Neurons.NCells, 0, Neurons.NCells[0])
        rho_E = np.full(NUnits[0], rE_target, dtype=dtype)
        rho_D = np.full(NUnits[0], rD_target, dtype=dtype)
        
        plastic_flag = 1*(eta_mean!=0)

        for i in range(25):
            m,n = np.unravel_index(i,(5,5))
            Mtx = np.ones((NUnits[m],NUnits[n]), dtype=dtype)
            Mtx[:] *= eta_mean[m,n]
            exec('self.' + eta_name[m][n] + ' = Mtx') 
        
        self.rho_E: dtype = rho_E
        self.rho_D: dtype = rho_D
        self.dw_min = dw_min
        self.dw_max = dw_max
        self.plasticity_approx = plasticity_approx
        self.plastic_flag = plastic_flag
            
                 
class Simulation(NamedTuple): 
       
       dt: dtype = dtype(0.1)
       stim_duration: dtype = dtype(1000.0)  


class SaveData(NamedTuple):
        
        ijk: list = np.array([0,0,0,0],dtype=np.int32)
        nstep: int = np.int32(100)             
                                            

# %% Functions

@njit(cache=True)
def drdt(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, 
         wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV, 
         wVE, wVP, wVS, wVV, rE, rD, rP, rS, rV,
         StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend): 
    
    drE = tau_inv_E * (-rE + wED @ rD + wEP @ rP + StimSoma_E)
    drD = tau_inv_E * (-rD + wDE @ rE + wDS @ rS + StimDend)
    drP = tau_inv_I * (-rP + wPE @ rE + wPP @ rP + wPS @ rS + wPV @ rV + StimSoma_P)
    drS = tau_inv_I * (-rS + wSE @ rE + wSP @ rP + wSS @ rS + wSV @ rV + StimSoma_S)
    drV = tau_inv_I * (-rV + wVE @ rE + wVP @ rP + wVS @ rS + wVV @ rV + StimSoma_V)
    
    return drE, drD, drP, drS, drV


def RateDynamics(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, 
                 wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV, 
                 wVE, wVP, wVS, wVV,rE, rD, rP, rS, rV,
                 StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend, dt):
    
    rE0 = rE.copy()
    rD0 = rD.copy()
    rP0 = rP.copy()
    rS0 = rS.copy()
    rV0 = rV.copy()
    
    drE1, drD1, drP1, drS1, drV1 = drdt(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, 
                                        wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV, 
                                        wVE, wVP, wVS, wVV,rE0, rD0, rP0, rS0, rV0,
                                        StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend)
    rE0[:] += dt * drE1
    rD0[:] += dt * drD1
    rP0[:] += dt * drP1
    rS0[:] += dt * drS1
    rV0[:] += dt * drV1
    
    drE2, drD2, drP2, drS2, drV2 = drdt(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, 
                                        wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV, 
                                        wVE, wVP, wVS, wVV,rE0, rD0, rP0, rS0, rV0,
                                        StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend)
    
    rE[:] += dt/2 * (drE1 + drE2)
    rD[:] += dt/2 * (drD1 + drD2)
    rP[:] += dt/2 * (drP1 + drP2) 
    rS[:] += dt/2 * (drS1 + drS2) 
    rV[:] += dt/2 * (drV1 + drV2) 
    
    rE[rE<0] = 0
    rP[rP<0] = 0
    rS[rS<0] = 0
    rV[rV<0] = 0
    rD[rD<0] = 0

    return


def RunStaticNetwork(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder: str, fln: str = ''):
    
    ### Neuron parameters
    NCells = NeuPar.NCells
    N = np.int32(sum(NCells))
    nE = NCells[0]
    
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
    
    ## Stimulation protocol & Inputs
    SD = StimPar.SD
    stim_visual = iter(StimPar.stim_visual)
    stim_motor = iter(StimPar.stim_motor)
    NStim = np.int32(len(StimPar.stim_visual))
    
    neurons_visual = InPar.neurons_visual
    neurons_motor = InPar.neurons_motor
    inp_ext_soma = InPar.inp_ext_soma
    inp_ext_dend = InPar.inp_ext_dend
    
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
    
    ### Initialisation
    StimSoma_E = np.zeros(nE, dtype=dtype)
    StimSoma_P = np.zeros(NCells[1], dtype=dtype)
    StimSoma_S = np.zeros(NCells[2], dtype=dtype)
    StimSoma_V = np.zeros(NCells[3], dtype=dtype)
    StimDend = np.zeros(NCells[0], dtype=dtype)
    stim_IN = np.zeros(N-nE, dtype=dtype)

    noise_soma = np.zeros((N, num_time_steps), dtype=dtype)
    noise_dend = np.zeros((NCells[0], num_time_steps), dtype=dtype)
    
    ### Path & file for data to be stored
    path = 'Results/Data/' + folder
    if not os.path.exists(path):
        os.mkdir(path)

    fp = open(path +'/Data_StaticNetwork_' + fln + '.dat','w')   
    
    ### Main loop
    for s in range(NStim):
        
        print('Stimuli', str(s+1), '/', str(NStim))
        
        V = next(stim_visual)
        M = next(stim_motor)
        noise_soma[:] = np.random.normal(0,SD,size=(N, num_time_steps))
        noise_dend[:] = np.random.normal(0,SD,size=(NCells[0], num_time_steps))
        
        for tstep in range(num_time_steps):
            
            stim_IN[:] = V * neurons_visual[nE:] + M * neurons_motor[nE:] + noise_soma[nE:,tstep] + inp_ext_soma[nE:]
            StimSoma_P[:], StimSoma_S[:], StimSoma_V[:] = np.split(stim_IN, ind_break)
            
            StimSoma_E[:] = V * neurons_visual[:nE] + noise_soma[:nE,tstep] + inp_ext_soma[:nE]
            StimDend[:] = M * neurons_motor[:nE] + noise_dend[:,tstep] + inp_ext_dend
            
            RateDynamics(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, 
                         wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV, 
                         wVE, wVP, wVS, wVV, rE, rD, rP, rS, rV,
                         StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend, dt)
            
            fp.write("%f" % (s * stim_duration + (tstep+1) * dt))
            for i in range(NCells[0]):
                fp.write(" %f" % rE[i])
            for i in range(NCells[1]):
                fp.write(" %f" % rP[i])
            for i in range(NCells[2]):
                fp.write(" %f" % rS[i])
            for i in range(NCells[3]):
                fp.write(" %f" % rV[i])
            for i in range(NCells[0]):
                fp.write(" %f" % rD[i])
            fp.write("\n") 
        
    fp.closed
    return


def RunPlasticNetwork(NeuPar, NetPar, InPar, StimPar, SimPar, LearnPar, RatePar, folder, fln, plasticity_type = 1):
    
    ### Neuron parameters
    NCells = NeuPar.NCells
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
  
    TEP = NetPar.TEP
    TDS = NetPar.TDS
    TPE = NetPar.TPE 
    TPS = NetPar.TPS
    TPV = NetPar.TPV 

    ### Stimulation protocol & Inputs
    SD = StimPar.SD
    stim_visual = iter(StimPar.stim_visual)
    stim_motor = iter(StimPar.stim_motor)
    NStim = np.int32(len(StimPar.stim_visual))
    
    neurons_visual = InPar.neurons_visual
    neurons_motor = InPar.neurons_motor
    inp_ext_soma = InPar.inp_ext_soma
    inp_ext_dend = InPar.inp_ext_dend
    
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
    
    ### Learning parameters
    etaEP = LearnPar.etaEP
    etaDS = LearnPar.etaDS
    etaPS = LearnPar.etaPS
    etaPV = LearnPar.etaPV
    rho_E = LearnPar.rho_E
    rho_D = LearnPar.rho_D
    dw_min = LearnPar.dw_min
    dw_max = LearnPar.dw_max
    plasticity_approx = LearnPar.plasticity_approx

    ### Path & file for data to be stored
    path = 'Results/Data/' + folder
    if not os.path.exists(path):
        os.mkdir(path)

    hdf = tables.open_file(path + '/Data_PlasticNetwork_' + fln + '.hdf', 'w')
    atom = tables.Float32Atom()
    hdf_weights = hdf.create_carray(hdf.root, 'weights', atom, (np.int32(NStim)+1,4+1))
        
    ### Main loop
    if plasticity_type==0:
        
        inp_ext_dend = Learning_Target_R(NCells, tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV,
                                         wVE, wVP, wVS, wVV, TEP, TDS, TPE, TPS, TPV, SD, stim_visual, stim_motor, NStim, dt, 
                                         num_time_steps, neurons_visual, neurons_motor, inp_ext_soma, inp_ext_dend, rE, rD, rP, rS, rV, 
                                         etaEP, etaDS, etaPS, etaPV, rho_E, rho_D, dw_min, dw_max, plasticity_approx, hdf, hdf_weights)
        
        InPar.inp_ext_dend = inp_ext_dend
        
    elif plasticity_type==1:
        
        inp_ext_dend = Learning_Target_I(NCells, tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV,
                                         wVE, wVP, wVS, wVV, TEP, TDS, TPE, TPS, TPV, SD, stim_visual, stim_motor, NStim, dt, 
                                         num_time_steps, neurons_visual, neurons_motor, inp_ext_soma, inp_ext_dend, rE, rD, rP, rS, rV, 
                                         etaEP, etaDS, etaPS, etaPV, rho_E, rho_D, dw_min, dw_max, plasticity_approx, hdf, hdf_weights)
        
        InPar.inp_ext_dend = inp_ext_dend
        
    elif plasticity_type==2:
        
        inp_ext_dend = Learning_Hebb(NCells, tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV,
                                     wVE, wVP, wVS, wVV, TEP, TDS, TPE, TPS, TPV, SD, stim_visual, stim_motor, NStim, dt, 
                                     num_time_steps, neurons_visual, neurons_motor, inp_ext_soma, inp_ext_dend, rE, rD, rP, rS, rV, 
                                     etaEP, etaDS, etaPS, etaPV, rho_E, rho_D, dw_min, dw_max, plasticity_approx, hdf, hdf_weights)
        
        InPar.inp_ext_dend = inp_ext_dend
        
    hdf.flush()
    hdf.close()
 
    return 


def Learning_Target_R(NCells, tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV,
                      wVE, wVP, wVS, wVV, TEP, TDS, TPE, TPS, TPV, SD, stim_visual, stim_motor, NStim, dt, 
                      num_time_steps, neurons_visual, neurons_motor, inp_ext_soma, inp_ext_dend, rE, rD, rP, rS, rV, 
                      etaEP, etaDS, etaPS, etaPV, rho_E, rho_D, dw_min, dw_max, plasticity_approx, hdf, hdf_weights):
    
    ### Neuron parameters
    N = np.int32(sum(NCells))
    nE = NCells[0]
    
    ind_break = np.cumsum(NCells[1:],dtype=np.int32)[:-1]
    
    ### Initialisation
    StimSoma_E = np.zeros(nE, dtype=dtype)
    StimSoma_P = np.zeros(NCells[1], dtype=dtype)
    StimSoma_S = np.zeros(NCells[2], dtype=dtype)
    StimSoma_V = np.zeros(NCells[3], dtype=dtype)
    StimDend = np.zeros(NCells[0], dtype=dtype)
    stim_IN = np.zeros(N-nE, dtype=dtype)
    
    MeanWeightVec = np.zeros(4, dtype=dtype) 

    noise_soma = np.zeros((N, num_time_steps), dtype=dtype)
    noise_dend = np.zeros((NCells[0], num_time_steps), dtype=dtype)
    
    ### Training/Learning
    for s in range(NStim):

        V = next(stim_visual)
        M = next(stim_motor)
        noise_soma[:] = np.random.normal(0,SD,size=(N, num_time_steps))
        noise_dend[:] = np.random.normal(0,SD,size=(NCells[0], num_time_steps))
        
        # Steady state rates
        for tstep in range(num_time_steps):
            
            stim_IN[:] = V*neurons_visual[nE:] + M*neurons_motor[nE:] + noise_soma[nE:,tstep] + inp_ext_soma[nE:]
            StimSoma_E[:] = V*neurons_visual[:nE] + noise_soma[:nE,tstep] + inp_ext_soma[:nE]
            StimSoma_P[:], StimSoma_S[:], StimSoma_V[:] = np.split(stim_IN, ind_break)
            StimDend[:] = M*neurons_motor[:nE] + noise_dend[:,tstep] + inp_ext_dend
    
            RateDynamics(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, 
                         wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV, 
                         wVE, wVP, wVS, wVV, rE, rD, rP, rS, rV,
                         StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend, dt)
        
        # Weight update
        MeanWeightVec[:] = WeightDynamics_Target_R(wEP, wDS, wPS, wPV, wPE, TEP, TDS, TPS, TPV, TPE,
                                                   rE, rD, rP, rS, rV, etaEP, etaDS, etaPS, etaPV,
                                                   rho_E, rho_D, dw_min, dw_max, plasticity_approx)
        
        if V==0: # only in BL
            XD = rho_D - wDE @ rE - wDS @ rS 
            inp_ext_dend[abs(rD-rho_D)>1e-1] = XD[abs(rD-rho_D)>1e-1]
        
        # Store data in hdf
        hdf_weights[s,0] = s
        hdf_weights[s,1:] = MeanWeightVec
        
        # Write to file and print progress
        if (s % 50 == 0):
            print('Stimuli', str(s+1), '/', str(NStim))
            hdf.flush()
            
    return inp_ext_dend
            
    
def Learning_Target_I(NCells, tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV,
                      wVE, wVP, wVS, wVV, TEP, TDS, TPE, TPS, TPV, SD, stim_visual, stim_motor, NStim, dt, 
                      num_time_steps, neurons_visual, neurons_motor, inp_ext_soma, inp_ext_dend, rE, rD, rP, rS, rV, 
                      etaEP, etaDS, etaPS, etaPV, rho_E, rho_D, dw_min, dw_max, plasticity_approx, hdf, hdf_weights):
    
    ### Neuron parameters
    N = np.int32(sum(NCells))
    nE = NCells[0]
    
    ind_break = np.cumsum(NCells[1:],dtype=np.int32)[:-1]
    
    ### Initialisation
    StimSoma_E = np.zeros(nE, dtype=dtype)
    StimSoma_P = np.zeros(NCells[1], dtype=dtype)
    StimSoma_S = np.zeros(NCells[2], dtype=dtype)
    StimSoma_V = np.zeros(NCells[3], dtype=dtype)
    StimDend = np.zeros(NCells[0], dtype=dtype)
    stim_IN = np.zeros(N-nE, dtype=dtype)
    
    MeanWeightVec = np.zeros(4, dtype=dtype) 

    noise_soma = np.zeros((N, num_time_steps), dtype=dtype)
    noise_dend = np.zeros((NCells[0], num_time_steps), dtype=dtype)
    
    ### Training/Learning
    for s in range(NStim):

        V = next(stim_visual)
        M = next(stim_motor)
        noise_soma[:] = np.random.normal(0,SD,size=(N, num_time_steps))
        noise_dend[:] = np.random.normal(0,SD,size=(NCells[0], num_time_steps))
        
        # Steady state rates
        for tstep in range(num_time_steps):
            
            stim_IN[:] = V*neurons_visual[nE:] + M*neurons_motor[nE:] + noise_soma[nE:,tstep] + inp_ext_soma[nE:]
            StimSoma_E[:] = V*neurons_visual[:nE] + noise_soma[:nE,tstep] + inp_ext_soma[:nE]
            StimSoma_P[:], StimSoma_S[:], StimSoma_V[:] = np.split(stim_IN, ind_break)
            StimDend[:] = M*neurons_motor[:nE] + noise_dend[:,tstep] + inp_ext_dend
    
            RateDynamics(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, 
                         wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV, 
                         wVE, wVP, wVS, wVV, rE, rD, rP, rS, rV,
                         StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend, dt)
        
        # Weight update
        MeanWeightVec[:] = WeightDynamics_Target_I(wEP, wDS, wPS, wPV, wED, wDE, wPE, TEP, TDS, TPS, 
                                                   TPV, TPE, rE, rD, rP, rS, rV, etaEP, etaDS, etaPS, 
                                                   etaPV, rho_E, rho_D, dw_min, dw_max, 
                                                   plasticity_approx, StimSoma_E, StimDend)
        
        if V==0: # only in BL
            inp_ext_dend = rho_D - wDE @ rE - wDS @ rS
        
        
        # Store data in hdf
        hdf_weights[s,0] = s
        hdf_weights[s,1:] = MeanWeightVec
        
        # Write to file and print progress
        if (s % 50 == 0):
            print('Stimuli', str(s+1), '/', str(NStim))
            hdf.flush()
            
    return inp_ext_dend 


def Learning_Hebb(NCells, tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV,
                  wVE, wVP, wVS, wVV, TEP, TDS, TPE, TPS, TPV, SD, stim_visual, stim_motor, NStim, dt, 
                  num_time_steps, neurons_visual, neurons_motor, inp_ext_soma, inp_ext_dend, rE, rD, rP, rS, rV, 
                  etaEP, etaDS, etaPS, etaPV, rho_E, rho_D, dw_min, dw_max, plasticity_approx, hdf, hdf_weights):
    
    ### Neuron parameters
    N = np.int32(sum(NCells))
    nE = NCells[0]
    
    ind_break = np.cumsum(NCells[1:],dtype=np.int32)[:-1]
    
    ### Initialisation
    StimSoma_E = np.zeros(nE, dtype=dtype)
    StimSoma_P = np.zeros(NCells[1], dtype=dtype)
    StimSoma_S = np.zeros(NCells[2], dtype=dtype)
    StimSoma_V = np.zeros(NCells[3], dtype=dtype)
    StimDend = np.zeros(NCells[0], dtype=dtype)
    stim_IN = np.zeros(N-nE, dtype=dtype)
    
    MeanWeightVec = np.zeros(4, dtype=dtype) 

    noise_soma = np.zeros((N, num_time_steps), dtype=dtype)
    noise_dend = np.zeros((NCells[0], num_time_steps), dtype=dtype)
    
    ### Training/Learning
    for s in range(NStim):

        V = next(stim_visual)
        M = next(stim_motor)
        noise_soma[:] = np.random.normal(0,SD,size=(N, num_time_steps))
        noise_dend[:] = np.random.normal(0,SD,size=(NCells[0], num_time_steps))
        
        # Steady state rates
        for tstep in range(num_time_steps):
            
            stim_IN[:] = V*neurons_visual[nE:] + M*neurons_motor[nE:] + noise_soma[nE:,tstep] + inp_ext_soma[nE:]
            StimSoma_E[:] = V*neurons_visual[:nE] + noise_soma[:nE,tstep] + inp_ext_soma[:nE]
            StimSoma_P[:], StimSoma_S[:], StimSoma_V[:] = np.split(stim_IN, ind_break)
            StimDend[:] = M*neurons_motor[:nE] + noise_dend[:,tstep] + inp_ext_dend
    
            RateDynamics(tau_inv_E, tau_inv_I, wEP, wED, wDS, wDE, 
                         wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV, 
                         wVE, wVP, wVS, wVV, rE, rD, rP, rS, rV,
                         StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend, dt)
        
        # Weight update
        MeanWeightVec[:] = WeightDynamics_Hebb(wEP, wDS, wPS, wPV, wPE, TEP, TDS, TPS, TPV, TPE,
                                               rE, rD, rP, rS, rV, etaEP, etaDS, etaPS, etaPV,
                                               rho_E, rho_D, dw_min, dw_max, plasticity_approx)
        
        if V==0: # only in BL
            XD = rho_D - wDE @ rE - wDS @ rS 
            inp_ext_dend[abs(rD-rho_D)>1e-1] = XD[abs(rD-rho_D)>1e-1]
        
        # Store data in hdf
        hdf_weights[s,0] = s
        hdf_weights[s,1:] = MeanWeightVec
        
        # Write to file and print progress
        if (s % 50 == 0):
            print('Stimuli', str(s+1), '/', str(NStim))
            hdf.flush()
            
    return inp_ext_dend


@njit(cache=True)
def WeightDynamics_Target_R(wEP, wDS, wPS, wPV, wPE, TEP, TDS, TPS, TPV, TPE,
                            rE, rD, rP, rS, rV, etaEP, etaDS, etaPS, etaPV,
                            rho_E, rho_D, dw_min, dw_max, plasticity_approx):

    ### Initialisation
    mean_weights = np.ones(4, dtype=dtype)
    mean_weights.fill(np.nan)
    
    ### Difference to target
    diff_rE = (rE - rho_E)
    diff_rD = (rD - rho_D)
    
    ### Averaged error 
    mean_error = np.zeros(len(rP), dtype=dtype)
    
    if plasticity_approx==0:
        
        # Averaged error for all postsynaptic E's a PV projects to (backprop-like rule)
        UEP = np.outer(diff_rE, np.ones_like(rP, dtype=dtype))
        UEP[...] = np.where(TEP, UEP, np.nan)
        for i in range(len(rP)):
            mean_error[i] = np.nanmean(UEP[:,i]) # numba does not support the second argument
            
    elif plasticity_approx==1:
        # Averaged error for all presynaptic E's a PV receives input from (backprop-like rule)  
        UPE = np.outer(np.ones_like(rP, dtype=dtype), diff_rE)
        UPE[...] = np.where(TPE, UPE, np.nan)
        UPE *= wPE 
        for i in range(len(rP)):
            mean_error[i] = np.nanmean(UPE[i,:])
    
    ### Weight updates
    
    # PV --> soma of E
    dw = etaEP * np.outer(diff_rE,rP)
    dw[...] = np.where(dw>=dw_min, dw, dw_min)
    dw[...] = np.where(dw<=dw_max, dw, dw_max)
    
    wEP -= dw
    wEP *= TEP
    wEP[...] = np.where(wEP<=0,wEP,dtype(0))        
    mean_weights[0] = np.nanmean(np.where(TEP, wEP, np.nan))
    
    # SOM --> dendrites of E
    dw = etaDS * np.outer(diff_rD,rS)
    dw[...] = np.where(dw>=dw_min, dw, dw_min)
    dw[...] = np.where(dw<=dw_max, dw, dw_max)
    
    wDS -= dw
    wDS *= TDS
    wDS[...] = np.where(wDS<=0,wDS,dtype(0)) 
    mean_weights[1] = np.nanmean(np.where(TDS, wDS, np.nan))
        
    # SOM --> PV
    dw = etaPS * np.outer(mean_error,rS)
    dw[...] = np.where(dw>=dw_min, dw, dw_min)
    dw[...] = np.where(dw<=dw_max, dw, dw_max)
    
    wPS += dw 
    wPS *= TPS
    wPS[...] = np.where(wPS<=0,wPS,dtype(0)) 
    mean_weights[2] = np.nanmean(np.where(TPS, wPS, np.nan))
        
    # VIP --> PV
    dw = etaPV * np.outer(mean_error,rV)
    dw[...] = np.where(dw>=dw_min, dw, dw_min)
    dw[...] = np.where(dw<=dw_max, dw, dw_max)
    
    wPV += dw 
    wPV *= TPV
    wPV[...] = np.where(wPV<=0,wPV,dtype(0)) 
    mean_weights[3] = np.nanmean(np.where(TPV, wPV, np.nan))
    
    return mean_weights


@njit(cache=True)
def WeightDynamics_Hebb(wEP, wDS, wPS, wPV, wPE, TEP, TDS, TPS, TPV, TPE,
                        rE, rD, rP, rS, rV, etaEP, etaDS, etaPS, etaPV,
                        rho_E, rho_D, dw_min, dw_max, plasticity_approx):

    ### Initialisation
    mean_weights = np.ones(4, dtype=dtype)
    mean_weights.fill(np.nan)
    
    
    ### Weight updates
    
    # PV --> soma of E
    dw = etaEP * np.outer(rE,rP)
    dw[...] = np.where(dw>=dw_min, dw, dw_min)
    dw[...] = np.where(dw<=dw_max, dw, dw_max)
    
    wEP -= dw # hebbian
    wEP *= TEP
    wEP[...] = np.where(wEP<=0,wEP,dtype(0))        
    mean_weights[0] = np.nanmean(np.where(TEP, wEP, np.nan))
    
    # SOM --> dendrites of E
    dw = etaDS * np.outer(rD,rS)
    dw[...] = np.where(dw>=dw_min, dw, dw_min)
    dw[...] = np.where(dw<=dw_max, dw, dw_max)
    
    wDS -= dw # hebbian
    wDS *= TDS
    wDS[...] = np.where(wDS<=0,wDS,dtype(0)) 
    mean_weights[1] = np.nanmean(np.where(TDS, wDS, np.nan))
        
    # SOM --> PV
    dw = etaPS * np.outer(rP,rS)
    dw[...] = np.where(dw>=dw_min, dw, dw_min)
    dw[...] = np.where(dw<=dw_max, dw, dw_max)
    
    wPS -= dw # anti-hebbian
    wPS *= TPS
    wPS[...] = np.where(wPS<=0,wPS,dtype(0)) 
    mean_weights[2] = np.nanmean(np.where(TPS, wPS, np.nan))
        
    # VIP --> PV
    dw = etaPV * np.outer(rP,rV)
    dw[...] = np.where(dw>=dw_min, dw, dw_min)
    dw[...] = np.where(dw<=dw_max, dw, dw_max)
    
    wPV -= dw  # anti-hebbian
    wPV *= TPV
    wPV[...] = np.where(wPV<=0,wPV,dtype(0)) 
    mean_weights[3] = np.nanmean(np.where(TPV, wPV, np.nan))
    
    return mean_weights


@njit(cache=True)
def WeightDynamics_Target_I(wEP, wDS, wPS, wPV, wED, wDE, wPE, TEP, TDS, TPS, TPV, TPE,
                            rE, rD, rP, rS, rV, etaEP, etaDS, etaPS, etaPV,
                            rho_E, rho_D, dw_min, dw_max, plasticity_approx, 
                            StimSoma_E, StimDend):

    ### Initialisation
    mean_weights = np.ones(4, dtype=dtype)
    mean_weights.fill(np.nan)
    
    ### Difference to target
    IE = wED @ rD + wEP @ rP + StimSoma_E
    ID = wDE @ rE + wDS @ rS + StimDend
    
    diff_IE = (IE - rho_E)
    diff_ID = (ID - rho_D)
    
    ### Averaged error 
    mean_error = np.zeros(len(rP), dtype=dtype)
    
    if plasticity_approx==0:
        
        # Averaged error for all postsynaptic E's a PV projects to (backprop-like rule)
        UEP = np.outer(diff_IE, np.ones_like(rP, dtype=dtype))
        UEP[...] = np.where(TEP, UEP, np.nan)
        for i in range(len(rP)):
            mean_error[i] = np.nanmean(UEP[:,i]) # numba does not support the second argument
            
    elif plasticity_approx==1:
        # Averaged error for all presynaptic E's a PV receives input from (backprop-like rule)  
        UPE = np.outer(np.ones_like(rP, dtype=dtype), diff_IE)
        UPE[...] = np.where(TPE, UPE, np.nan)
        UPE *= wPE 
        for i in range(len(rP)):
            mean_error[i] = np.nanmean(UPE[i,:])
    
    ### Weight updates
    
    # PV --> soma of E
    dw = etaEP * np.outer(diff_IE,rP)
    dw[...] = np.where(dw>=dw_min, dw, dw_min)
    dw[...] = np.where(dw<=dw_max, dw, dw_max)
    
    wEP -= dw
    wEP *= TEP
    wEP[...] = np.where(wEP<=0,wEP,dtype(0))        
    mean_weights[0] = np.nanmean(np.where(TEP, wEP, np.nan))
    
    # SOM --> dendrites of E
    dw = etaDS * np.outer(diff_ID,rS)
    dw[...] = np.where(dw>=dw_min, dw, dw_min)
    dw[...] = np.where(dw<=dw_max, dw, dw_max)
    
    wDS -= dw
    wDS *= TDS
    wDS[...] = np.where(wDS<=0,wDS,dtype(0)) 
    mean_weights[1] = np.nanmean(np.where(TDS, wDS, np.nan))
        
    # SOM --> PV
    dw = etaPS * np.outer(mean_error,rS)
    dw[...] = np.where(dw>=dw_min, dw, dw_min)
    dw[...] = np.where(dw<=dw_max, dw, dw_max)
    
    wPS += dw
    wPS *= TPS
    wPS[...] = np.where(wPS<=0,wPS,dtype(0)) 
    mean_weights[2] = np.nanmean(np.where(TPS, wPS, np.nan))
        
    # VIP --> PV
    dw = etaPV * np.outer(mean_error,rV)
    dw[...] = np.where(dw>=dw_min, dw, dw_min)
    dw[...] = np.where(dw<=dw_max, dw, dw_max)
    
    wPV += dw 
    wPV *= TPV
    wPV[...] = np.where(wPV<=0,wPV,dtype(0)) 
    mean_weights[3] = np.nanmean(np.where(TPV, wPV, np.nan))
    
    return mean_weights


def BackgroundInput(NeuPar, NetPar, r0: list = None, Xternal: list = None):
    
    NCells = NeuPar.NCells
    Nb = np.cumsum(NCells, dtype=np.int32)
    
    inp_ext_soma = np.zeros(np.sum(NCells), dtype=dtype)
    inp_ext_dend = np.zeros(NCells[0], dtype=dtype)
    
    wSE = NetPar.wSE
    wSV = NetPar.wSV
    wVE = NetPar.wVE
    
    if (Xternal is not None and r0 is None):
        
        inp_ext_soma[:Nb[0]] = Xternal[0]
        inp_ext_dend[:] = Xternal[-1]
        inp_ext_soma[Nb[0]:Nb[1]] = Xternal[1]
        inp_ext_soma[Nb[1]:Nb[2]] = Xternal[2]
        inp_ext_soma[Nb[2]:] = Xternal[3]
    
    elif(Xternal is None and r0 is not None):
        
        wED = NetPar.wED
        wEP = NetPar.wEP
        wPE = NetPar.wPE
        wPP = NetPar.wPP
        wPS = NetPar.wPS
        wPV = NetPar.wPV
        wSP = NetPar.wSP
        wSS = NetPar.wSS
        wVP = NetPar.wVP
        wVS = NetPar.wVS
        wVV = NetPar.wVV
        wDE = NetPar.wDE
        wDS = NetPar.wDS
        
        if len(r0)<sum(NCells):
            rE_base = np.repeat(r0[0],NCells[0])
            rD_base = np.zeros(NCells[0])
            rP_base = np.repeat(r0[1],NCells[1])
            rS_base = np.repeat(r0[2],NCells[2])
            rV_base = np.repeat(r0[3],NCells[3])
        else:
            rE_base, rP_base, rS_base, rV_base, rD_base  = np.split(r0,Nb)
        
        inp_ext_soma[:Nb[0]] = rE_base - wED @ rD_base - wEP @ rP_base
        inp_ext_soma[Nb[0]:Nb[1]] = rP_base - wPE @ rE_base - wPP @ rP_base - wPS @ rS_base - wPV @ rV_base
        inp_ext_soma[Nb[1]:Nb[2]] = rS_base - wSE @ rE_base - wSP @ rP_base - wSS @ rS_base - wSV @ rV_base
        inp_ext_soma[Nb[2]:] = rV_base - wVE @ rE_base - wVP @ rP_base - wVS @ rS_base - wVV @ rV_base
        inp_ext_dend = rD_base - wDE @ rE_base - wDS @ rS_base
        
    return inp_ext_soma, inp_ext_dend

 