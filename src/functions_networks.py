#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:31:58 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import os
from numba import njit

dtype = np.float32

# %% functions

## functions for mean-field network/s

def rate_dynamics_mfn(tau_E, tau_I, tc_var, w_var, U, V, W, rates, mean, 
                      var, feedforward_input, dt, n):
    
    # Initialise
    rates_new = rates.copy() 
    mean_new = mean.copy()
    dr_1 = np.zeros(len(rates_new), dtype=dtype)
    dr_2 = np.zeros(len(rates_new), dtype=dtype)
    #dr_mem_1 = np.zeros((1,1), dtype=dtype)
    #dr_mem_2 = np.zeros((1,1), dtype=dtype)
    
    var_new = var.copy() 
    #dr_var_1 = np.zeros((1,1), dtype=dtype) 
    #dr_var_2 = np.zeros((1,1), dtype=dtype) 
    
    # RK 2nd order
    dr_mem_1 = (U @ rates_new) / tau_E
    dr_var_1 = (-var_new + sum(w_var * rates_new[:2])**n) / tc_var 
    dr_1 = -rates_new + W @ rates_new + V @ np.array([mean_new]) + feedforward_input
    dr_1[:4] /= tau_E 
    dr_1[4:] /= tau_I 

    rates_new[:] += dt * dr_1
    mean_new += dt * dr_mem_1
    var_new += dt * dr_var_1
    
    dr_mem_2 = (U @ rates_new) / tau_E
    dr_var_2 = (-var_new + sum(w_var * rates_new[:2])**n) / tc_var
    dr_2 = -rates_new + W @ rates_new + V @ mean_new + feedforward_input
    dr_2[:4] /= tau_E 
    dr_2[4:] /= tau_I
    
    rates[:] += dt/2 * (dr_1 + dr_2)
    mean += dt/2 * (dr_mem_1 + dr_mem_2)
    var += dt/2 * (dr_var_1 + dr_var_2)
    
    # Rectify
    rates[rates<0] = 0

    return [rates, mean[0], var] # [rates, mean, var]


def run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, fixed_input, stimuli, VS = 1, VV = 0,
                    dt = dtype(1), w_PE_to_V = dtype([1,1]), n=2, record_interneuron_activity = False):
    
    ### neuron and network parameters
    tau_E, tau_I  = tau_pe
    neurons_feedforward = np.array([1, 1, 0, 0, 1, 0, VS, VV], dtype=dtype)
    
    ### initialise
    num_points = len(stimuli)
    m_neuron = np.zeros_like(stimuli, dtype=dtype)
    v_neuron = np.zeros_like(stimuli, dtype=dtype)
    rates_lower = np.zeros((num_points, 8), dtype=dtype)
    
    if fixed_input.ndim==1:
        fixed_input = np.tile(fixed_input, (num_points,1))
    
    ### run mean-field network
    for id_stim, stim in enumerate(stimuli):
        
        feedforward_input = fixed_input[id_stim,:] + stim * neurons_feedforward
        
        ## rates of PE circuit and M neuron
        [rates_lower[id_stim,:], 
         m_neuron[id_stim], v_neuron[id_stim]] = rate_dynamics_mfn(tau_E, tau_I, tc_var_per_stim, w_PE_to_V, 
                                                                   w_PE_to_P, w_P_to_PE, w_PE_to_PE,
                                                                   rates_lower[id_stim-1,:], m_neuron[id_stim-1], 
                                                                   v_neuron[id_stim-1], feedforward_input, dt, n)

     
    ret = (m_neuron, v_neuron, rates_lower[:,:2],)
    
    if record_interneuron_activity:
        ret += (rates_lower[:,4:], )
    
    return ret                                                               


def run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                            tc_var_pred, tau_pe, fixed_input, stimuli, VS = 1, VV = 0, dt = dtype(1), n = 2, 
                            w_PE_to_V = dtype([1,1]), v_PE_to_V  = dtype([1,1]), record_pe_activity = False,
                            fixed_input_lower = None, fixed_input_higher = None, record_interneuron_activity = False):
    
    ### neuron and network parameters
    tau_E, tau_I  = tau_pe
    neurons_feedforward = np.array([1, 1, 0, 0, 1, 0, VS, VV], dtype=dtype)
    
    ### initialise
    num_points = len(stimuli)
    m_neuron_lower = np.zeros_like(stimuli, dtype=dtype)
    v_neuron_lower = np.zeros_like(stimuli, dtype=dtype)
    m_neuron_higher = np.zeros_like(stimuli, dtype=dtype)
    v_neuron_higher = np.zeros_like(stimuli, dtype=dtype)
    rates_lower = np.zeros((num_points, 8), dtype=dtype)
    rates_higher = np.zeros((num_points, 8), dtype=dtype)
    
    if fixed_input is not None:
        if fixed_input.ndim==1:
            fixed_input_lower = np.tile(fixed_input, (num_points,1))
            fixed_input_higher = np.tile(fixed_input, (num_points,1))
        else:
            fixed_input_lower = fixed_input
            fixed_input_higher = fixed_input
    else:
        if fixed_input_lower.ndim==1:
            fixed_input_lower = np.tile(fixed_input_lower, (num_points,1))
        if fixed_input_higher.ndim==1:
            fixed_input_higher = np.tile(fixed_input_higher, (num_points,1))
            
    
    ### run mean-field network
    for id_stim, stim in enumerate(stimuli):
        
        ## run lower PE circuit
        feedforward_input_lower = fixed_input_lower[id_stim,:] + stim * neurons_feedforward
        
        [rates_lower[id_stim,:], m_neuron_lower[id_stim], 
         v_neuron_lower[id_stim]] = rate_dynamics_mfn(tau_E, tau_I, tc_var_per_stim, w_PE_to_V, w_PE_to_P,
                                                      w_P_to_PE, w_PE_to_PE, rates_lower[id_stim-1,:],
                                                      m_neuron_lower[id_stim-1], v_neuron_lower[id_stim-1], 
                                                     feedforward_input_lower, dt, n)
        
        
        ## run higher PE circuit
        feedforward_input_higher = fixed_input_higher[id_stim,:] + m_neuron_lower[id_stim-1] * neurons_feedforward
        
        [rates_higher[id_stim,:], m_neuron_higher[id_stim], 
         v_neuron_higher[id_stim]] = rate_dynamics_mfn(tau_E, tau_I, tc_var_pred, v_PE_to_V, v_PE_to_P,
                                                      v_P_to_PE, v_PE_to_PE, rates_higher[id_stim-1,:],
                                                      m_neuron_higher[id_stim-1], v_neuron_higher[id_stim-1], 
                                                      feedforward_input_higher, dt, n)
                                                       
    ### compute weighted output
    v_neuron_lower[np.isinf(1/v_neuron_lower)] = 1e-30
    v_neuron_higher[np.isinf(1/v_neuron_higher)] = 1e-30
    
    alpha = (1/v_neuron_lower) / ((1/v_neuron_lower) + (1/v_neuron_higher))
    beta = (1/v_neuron_higher) / ((1/v_neuron_lower) + (1/v_neuron_higher))
   
    weighted_output = alpha * stimuli + beta * m_neuron_lower
    
    ret = (m_neuron_lower, v_neuron_lower, m_neuron_higher, v_neuron_higher, alpha, beta, weighted_output,)
    
    if record_pe_activity:
        ret += (rates_lower[:,:2], rates_higher[:,:2], )
        
    if record_interneuron_activity:
        ret += (rates_lower[:,4:], rates_higher[:,4:], )
    
    return ret

    
## functions for population network/s

#@njit(cache=True)
def drdt(tau_inv_E, tau_inv_I, tau_inv_var, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV, wVE, wVP, wVS, wVV,
         wEM, wDM, wPM, wSM, wVM, wME, wVarE, rE, rD, rP, rS, rV, r_mem, r_var, StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend): 
    
    drE = tau_inv_E * (-rE + wED @ rD + wEP @ rP + wEM @ r_mem + StimSoma_E)
    drD = tau_inv_E * (-rD + wDE @ rE + wDS @ rS + wDM @ r_mem + StimDend)
    drP = tau_inv_I * (-rP + wPE @ rE + wPP @ rP + wPS @ rS + wPV @ rV + wPM @ r_mem + StimSoma_P)
    drS = tau_inv_I * (-rS + wSE @ rE + wSP @ rP + wSS @ rS + wSV @ rV + wSM @ r_mem + StimSoma_S)
    drV = tau_inv_I * (-rV + wVE @ rE + wVP @ rP + wVS @ rS + wVV @ rV + wVM @ r_mem + StimSoma_V)
    
    dr_mem = tau_inv_E * (wME @ rE)
    dr_var = tau_inv_var * (-r_var + (wVarE @ rE)**2)
    
    return drE, drD, drP, drS, drV, dr_mem, dr_var



def rate_dynamics(tau_inv_E, tau_inv_I, tau_inv_var, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV, wVE, wVP, wVS, wVV,
                  wEM, wDM, wPM, wSM, wVM, wME, wVarE, rE, rD, rP, rS, rV, r_mem, r_var, StimSoma_E, StimSoma_P, StimSoma_S, 
                  StimSoma_V, StimDend, dt):
    
    rE0 = rE.copy()
    rD0 = rD.copy()
    rP0 = rP.copy()
    rS0 = rS.copy()
    rV0 = rV.copy()
    r_mem0 = r_mem.copy()
    r_var0 = r_var.copy()
    
    drE1, drD1, drP1, drS1, drV1, dr_mem1, dr_var1 = drdt(tau_inv_E, tau_inv_I, tau_inv_var, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, 
                                                          wSE, wSP, wSS, wSV, wVE, wVP, wVS, wVV, wEM, wDM, wPM, wSM, wVM, wME, wVarE, 
                                                          rE0, rD0, rP0, rS0, rV0, r_mem0, r_var0, StimSoma_E, StimSoma_P, StimSoma_S, 
                                                          StimSoma_V, StimDend)
    
    rE0[:] += dt * drE1
    rD0[:] += dt * drD1
    rP0[:] += dt * drP1
    rS0[:] += dt * drS1
    rV0[:] += dt * drV1
    r_mem0[:] += dt * dr_mem1
    r_var0[:] += dt * dr_var1
    
    drE2, drD2, drP2, drS2, drV2, dr_mem2, dr_var2 = drdt(tau_inv_E, tau_inv_I, tau_inv_var, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, 
                                                          wSE, wSP, wSS, wSV, wVE, wVP, wVS, wVV, wEM, wDM, wPM, wSM, wVM, wME, wVarE, 
                                                          rE0, rD0, rP0, rS0, rV0, r_mem0, r_var0, StimSoma_E, StimSoma_P, StimSoma_S, 
                                                          StimSoma_V, StimDend)
    
    rE[:] += dt/2 * (drE1 + drE2)
    rD[:] += dt/2 * (drD1 + drD2)
    rP[:] += dt/2 * (drP1 + drP2) 
    rS[:] += dt/2 * (drS1 + drS2) 
    rV[:] += dt/2 * (drV1 + drV2) 
    r_mem[:] += dt/2 * (dr_mem1 + dr_mem2)
    r_var[:] += dt/2 * (dr_var1 + dr_var2)
    
    rE[rE<0] = 0
    rP[rP<0] = 0
    rS[rS<0] = 0
    rV[rV<0] = 0
    rD[rD<0] = 0

    return


def run_population_net(NeuPar, NetPar, StimPar, RatePar, dt, folder: str, fln: str = ''):
    
    ### Neuron parameters
    NCells = NeuPar.NCells
    N_total = np.int32(sum(NCells))
    nE = NCells[0]
    
    ind_break = np.cumsum(NCells[1:],dtype=np.int32)[:-1]
    
    tau_inv_E = NeuPar.tau_inv_E
    tau_inv_I = NeuPar.tau_inv_I
    tau_inv_var = NeuPar.tau_inv_var
    
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
    
    wEM = NetPar.wEM
    wDM = NetPar.wDM 
    wPM = NetPar.wPM
    wSM = NetPar.wSM
    wVM = NetPar.wVM
    wME = NetPar.wME
    wVarE = NetPar.wVarE
    
    ## Stimulation protocol & Inputs
    stimuli = iter(StimPar.stimuli)
    neurons_visual = StimPar.neurons_visual
    inp_ext_soma = StimPar.inp_ext_soma
    inp_ext_dend = StimPar.inp_ext_dend 
    
    ### Initial activity levels
    rE = RatePar.rE0
    rD = RatePar.rD0
    rP = RatePar.rP0
    rS = RatePar.rS0
    rV = RatePar.rV0
    r_mem = RatePar.r_mem0
    r_var = RatePar.r_var0
    
    ### Initialisation
    StimSoma_E = np.zeros(nE, dtype=dtype)
    StimSoma_P = np.zeros(NCells[1], dtype=dtype)
    StimSoma_S = np.zeros(NCells[2], dtype=dtype)
    StimSoma_V = np.zeros(NCells[3], dtype=dtype)
    StimDend = np.zeros(NCells[0], dtype=dtype)
    stim_IN = np.zeros(N_total-nE, dtype=dtype)
    
    ### Path & file for data to be stored
    path = '../results/data/' + folder
    if not os.path.exists(path):
        os.mkdir(path)

    fp = open(path +'/Data_PopulationNetwork_' + fln + '.dat','w')
    
    ### Main loop
    for id_stim, stim in enumerate(stimuli):
        
        stim_IN[:] = stim * neurons_visual[nE:] + inp_ext_soma[nE:]
        StimSoma_P[:], StimSoma_S[:], StimSoma_V[:] = np.split(stim_IN, ind_break)
        
        StimSoma_E[:] = stim * neurons_visual[:nE] + inp_ext_soma[:nE]
        StimDend[:] = inp_ext_dend
    
        rate_dynamics(tau_inv_E, tau_inv_I, tau_inv_var, wEP, wED, wDS, wDE, wPE, wPP, wPS, wPV, wSE, wSP, wSS, wSV, wVE, wVP, wVS,
                      wVV, wEM, wDM, wPM, wSM, wVM, wME, wVarE, rE, rD, rP, rS, rV, r_mem, r_var, 
                      StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend, dt)
        
        fp.write("%f" % ((id_stim+1) * dt))
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
        fp.write(" %f" % r_mem[0])
        fp.write(" %f" % r_var[0])
        fp.write("\n") 
        
    fp.closed
    return
