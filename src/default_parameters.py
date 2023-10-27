#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:31:58 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from typing import NamedTuple

dtype = np.float32


# %% functions

### default parameters for the mean-field networks

def default_para_mfn(mfn_flag, baseline_activity = dtype([0, 0, 0, 0, 4, 4, 4, 4]), one_column=True):
    
    ### time constants
    tc_var_per_stim = dtype(5000)
    tc_var_pred = dtype(5000)
    tau_pe = [dtype(60), dtype(2)]
    
    ### weights
    
    ## load data  
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + mfn_flag + '.pickle'
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    with open(filename,'rb') as f:
        xopt, W, optimize_flag, _, _ = pickle.load(f)
   
    ## connectivity within PE circuits
    w_PE_to_PE = np.copy(W)
    v_PE_to_PE = np.copy(W)
    
    w_PE_to_PE[optimize_flag!=0] = xopt
    v_PE_to_PE[optimize_flag!=0] = xopt
    
    ## external background input (here I assume it is the same for both parts)
    r_target = baseline_activity
    fixed_input = (np.eye(8, dtype=dtype) - w_PE_to_PE) @ r_target
    
    ## Define connectivity between PE circuit and P
    w_PE_to_P = np.zeros((1,8), dtype=dtype)     
    w_PE_to_P[0,0] = -0.003          # nPE onto P
    w_PE_to_P[0,1] =  0.003          # pPE onto P
    
    w_P_to_PE = np.zeros((8,1), dtype=dtype)     
    w_P_to_PE[2:4,0] = dtype(1)     # onto dendrites
    w_P_to_PE[5,0] = dtype(1)       # onto PV neuron receiving prediction
    w_P_to_PE[6,0] = dtype(1-VS)    # onto SOM neuron
    w_P_to_PE[7,0] = dtype(1-VV)    # onto VIP neuron
    
    v_PE_to_P = np.zeros((1,8), dtype=dtype)     
    v_PE_to_P[0,0] = -0.7*1e-3          # nPE onto P
    v_PE_to_P[0,1] =  0.7*1e-3          # pPE onto P
    
    v_P_to_PE = np.zeros((8,1), dtype=dtype)     
    v_P_to_PE[2:4,0] = dtype(1)     # onto dendrites
    v_P_to_PE[5,0] = dtype(1)       # onto PV neuron receiving prediction
    v_P_to_PE[6,0] = dtype(1-VS)    # onto SOM neuron
    v_P_to_PE[7,0] = dtype(1-VV)    # onto VIP neuron
    
    ### correct weights to make sure that gain of nPE = gain of pPE
    if mfn_flag=='10':
        nPE_scale = 1.015
        pPE_scale = 1.023
    elif mfn_flag=='01':
        nPE_scale = 1.7
        pPE_scale = 1.7
    elif mfn_flag=='11':
        nPE_scale = 2.49
        pPE_scale = 2.53
    
    if one_column:
        w_PE_to_P[0,0] *= nPE_scale
        w_PE_to_P[0,1] *= pPE_scale
        w_PE_to_V = [nPE_scale, pPE_scale]
    else:
        w_PE_to_P[0,0] *= nPE_scale * 15 
        w_PE_to_P[0,1] *= pPE_scale * 15
        w_PE_to_V = [nPE_scale, pPE_scale]
    
    v_PE_to_P[0,0] *= nPE_scale
    v_PE_to_P[0,1] *= pPE_scale
    v_PE_to_V = [nPE_scale, pPE_scale]
        
    return [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
            v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
            tc_var_per_stim, tc_var_pred, tau_pe, fixed_input]


### parameters for the population network

def random_uniform_from_moments(mean, sd, num):
    
    b = np.sqrt(12) * sd / 2 + mean
    a = 2 * mean - b
    rnd = dtype(np.random.uniform(a, b, size = num))
        
    return rnd


class Neurons(NamedTuple):
        
    NCells: list = np.array([140,20,20,20], dtype=np.int32)
    tau_inv_E: dtype = dtype(1.0/60.0)
    tau_inv_I: dtype = dtype(1.0/2.0)
    tau_inv_var: dtype = dtype(1.0/2000)


class Network:
    def __init__(self, Neurons, Dict_w, Dict_t, weight_name, neurons_visual, gain_factors_nPE = None, 
                 gain_factors_pPE = None, nPE_true = None, pPE_true = None, mean = 1, std = 0, p_conn = None):
        
        ### get number of cells
        NE, NP, NS, NV = Neurons.NCells
        
        ### PE circuit connectivity
        self.weight_name = weight_name
        
        for i in range(25):
            
            m,n = np.unravel_index(i,(5,5))
        
            exec('self.' + weight_name[m][n] + ' = Dict_w["' + weight_name[m][n] + '"]')
            exec('self.T' + weight_name[m][n][1:] + ' = Dict_t["T' + weight_name[m][n][1:]+ '"]')         
        
        ### create connectivity between PE circuit and M or V neuron
        neurons_visual_P = neurons_visual[NE:NE+NP]
        neurons_visual_S = neurons_visual[NE+NP:NE+NP+NS]
        neurons_visual_V = neurons_visual[NE+NP+NS:]
        
        self.wEM = np.zeros((NE, 1)) 
        self.wDM = np.ones((NE, 1))
        
        self.wPM = np.zeros((NP, 1))
        self.wPM[neurons_visual_P==0,:] = 1
        
        self.wSM = np.zeros((NS, 1))
        self.wSM[neurons_visual_S==0,:] = 1
        
        self.wVM = np.zeros((NV, 1))
        self.wVM[neurons_visual_V==0,:] = 1
        
        if gain_factors_nPE is None:
            gain_factors_nPE = np.ones((1, NE), dtype=dtype) 
        if gain_factors_pPE is None:
            gain_factors_pPE = np.ones((1, NE), dtype=dtype)
        
        if nPE_true is None:
            nPE_true_bool = np.ones((1, NE))==1
        else:
            nPE_true_bool = np.copy(nPE_true)
        if pPE_true is None:
            pPE_true_bool = np.ones((1, NE))==1
        else:
            pPE_true_bool = np.copy(pPE_true)
        
        if p_conn is not None:
            sparisty_bool = np.random.choice([0,1], p=[1-p_conn,p_conn], size=NE) == 1
            nPE_true_bool[~sparisty_bool] = False
            pPE_true_bool[~sparisty_bool] = False
        
        gain_scaling = dtype(np.random.normal(mean, std, size=NE))
        gain = gain_factors_nPE * gain_factors_pPE * gain_scaling
        gain[nPE_true_bool] *= -1/sum(nPE_true_bool)
        gain[pPE_true_bool] *= 1/sum(pPE_true_bool)
        gain[~(nPE_true_bool+pPE_true_bool)] = 0
        
        speed = dtype(0.05)
        self.wME = speed * gain
        self.wVarE = abs(gain)
                                            
 
                
class Stimulation:
    def __init__(self, mean_stimuli, std_stimuli, inp_ext_soma, inp_ext_dend, neurons_visual, seed = 186,
                 trial_duration = np.int32(100000), num_values_per_trial = np.int32(200), dist_type = 'uniform'):
    
        # Stimulation protocol
        np.random.seed(seed)
        
        if dist_type=='uniform':
            stimuli = random_uniform_from_moments(mean_stimuli, std_stimuli, num_values_per_trial)
        elif dist_type=='normal':
            stimuli = np.random.normal(mean_stimuli, std_stimuli, size=num_values_per_trial)
            
        stimuli = np.repeat(stimuli, trial_duration//num_values_per_trial)
            
        self.stimuli: dtype = stimuli
        self.neurons_visual: list = neurons_visual
        
        self.inp_ext_soma: dtype = inp_ext_soma
        self.inp_ext_dend: dtype = inp_ext_dend



class Activity_Zero:
    def __init__(self, Neurons, r0 = dtype([0,0,0,0,0]), r_mem_init = 0, r_var_init = 0):

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
            
        self.r_mem0 = r_mem_init * np.ones(1)
        self.r_var0 = r_var_init * np.ones(1)
        
        

