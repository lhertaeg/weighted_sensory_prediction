#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:50:33 2023

@author: loreen.hertaeg
"""

# %% Import

import pickle
import numpy as np

from src.functions_simulate import simulate_example_pe_circuit, simulate_effect_baseline

dtype = np.float32

# %% Illustrate effect of BL for one column

run_cell = False

if run_cell:
    
    file_for_data = '../results/data/moments/data_influence_of_baseline.pickle'
    add_ins = np.linspace(0,0.5,5)
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_temp = '../results/data/moments/data_temp.pickle'
    
    ### define the steady state
    trial_duration = np.int32(130000)
    n_ss = 3 * trial_duration//4
    
    ### define inputs and initialise
    M_nPE, M_pPE, M_both = np.zeros_like(add_ins), np.zeros_like(add_ins), np.zeros_like(add_ins)
    V_nPE, V_pPE, V_both = np.zeros_like(add_ins), np.zeros_like(add_ins), np.zeros_like(add_ins)
    BL_nPE, BL_pPE, BL_both = np.zeros_like(add_ins), np.zeros_like(add_ins), np.zeros_like(add_ins)
    
    ### run network with BL increased in nPE neurons
    for i, add_in in enumerate(add_ins):
        
        add_input = np.zeros(8)
        add_input[0] = add_in
        
        ### run network
        [_, _, _, _, _, m_neuron, v_neuron, PE] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_temp, add_input = add_input,
                                                                              trial_duration=trial_duration, save_PE=True)
        ### extract final M estimate
        M_nPE[i] = np.mean(m_neuron[n_ss:])
        V_nPE[i] = np.mean(v_neuron[n_ss:])
        BL_nPE[i] = np.mean(PE[np.where(abs(np.diff(PE[:,0]))<1e-6)[0],0])
        
    
    ### run network with BL increased in pPE neurons
    for i, add_in in enumerate(add_ins):
        
        add_input = np.zeros(8)
        add_input[1] = add_in
        
        ### run network
        [_, _, _, _, _, m_neuron, v_neuron, PE] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_temp, add_input = add_input,
                                                                              trial_duration=trial_duration, save_PE=True)
        ### extract final M estimate
        M_pPE[i] = np.mean(m_neuron[n_ss:])
        V_pPE[i] = np.mean(v_neuron[n_ss:])
        BL_pPE[i] = np.mean(PE[np.where(abs(np.diff(PE[:,1]))<1e-6)[0],1])
        

    ### run network with BL increased in both PE neurons
    for i, add_in in enumerate(add_ins):
        
        add_input = np.zeros(8)
        add_input[:2] = add_in
        
        ### run network
        [_, _, _, _, _, m_neuron, v_neuron, PE] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_temp, add_input = add_input,
                                                                              trial_duration=trial_duration, save_PE=True)
        ### extract final M estimate
        M_both[i] = np.mean(m_neuron[n_ss:])
        V_both[i] = np.mean(v_neuron[n_ss:])
        BL_both[i] = np.mean(PE[np.where(abs(np.diff(PE[:,1]))<1e-6)[0],1])


    
    with open(file_for_data,'wb') as f:
        pickle.dump([M_nPE, M_pPE, M_both, V_nPE, V_pPE, V_both, BL_nPE, BL_pPE, BL_both], f)
        
    
# %% Illustrate effect of BL for full network (lower/higher subnetwork)

run_cell = False

if run_cell:

    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11

    ### define target area and stimulus statistics
    column = 0      # 0: both columns, 1: lower level PE circuit, 2: higher level PE circuit
    std_mean = 1    # [0,1]  
    n_std = 0       # [0,1]
    
    file_for_data = '../results/data/moments/data_influence_of_baseline_full_' + str(std_mean) + '_' + str(n_std) + '.pickle'
 
    ### define the baselines to be tested
    baselines = np.linspace(0,0.5,5)   
 
    ### get data

    ### run for nPE neuron
    cell_id = 0
    sensory_weights_nPE = simulate_effect_baseline(mfn_flag, std_mean, n_std, column, baselines, cell_id)
    
    ### run for pPE neuron
    cell_id = 1
    sensory_weights_pPE = simulate_effect_baseline(mfn_flag, std_mean, n_std, column, baselines, cell_id)
    
    ### run for both PE neurons
    cell_id = np.array([0,1])
    sensory_weights_both = simulate_effect_baseline(mfn_flag, std_mean, n_std, column, baselines, cell_id)

    ### save data
    with open(file_for_data,'wb') as f:
        pickle.dump([baselines, sensory_weights_nPE, sensory_weights_pPE, sensory_weights_both], f)
