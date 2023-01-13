#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:31:58 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

dtype = np.float32


# %% functions

def default_para_mfn(mfn_flag, baseline_activity = dtype([0, 0, 0, 0, 4, 4, 4, 4]), one_column=True):
    
    ### time constants
    #if one_column:
    tc_var_per_stim = dtype(5000)
    tc_var_pred = dtype(5000)
    # else:
    #     tc_var_per_stim = dtype(1000)
    #     tc_var_pred = dtype(1000)
    
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