#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.functions_simulate import simulate_PE_circuit_P_fixed_S_constant, simulate_example_pe_circuit
from src.plot_data import plot_nPE_pPE_activity_compare, illustrate_PE_establish_M

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Illustrate activity of nPE and pPE neurons

run_cell = True
plot_only = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_mfn_' + mfn_flag + '_PE_neurons_constant_stimuli_P_fixed.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
    
        ## define parameters ranges tested:
        prediction_initial = 5
        stimulus_tested = np.linspace(0,10,21)
        
        ## run simulations
        nPE, pPE = simulate_PE_circuit_P_fixed_S_constant(mfn_flag, prediction_initial, stimulus_tested, file_for_data)
        
        ### save data
        with open(file_for_data,'wb') as f:
             pickle.dump([prediction_initial, stimulus_tested, nPE, pPE], f)
        
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [prediction_initial, stimulus_tested, nPE, pPE] = pickle.load(f)
        
    ### plot 
    plot_nPE_pPE_activity_compare(prediction_initial, stimulus_tested, nPE, pPE)
    
    
# %% run example to show how nPE and pPE establishes M robustly

run_cell = True
plot_only = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_PE_establish_M_' + mfn_flag + '.pickle'
    
    ### flag
    save_PE = True
    
    ### get data
    if not plot_only: # simulate respective network
        
        if save_PE:
            [_, _, trial_duration, _, stimuli, m_neuron, v_neuron, PE] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_for_data, save_PE=save_PE,
                                                                                                     trial_duration = np.int32(200000), 
                                                                                                     num_values_per_trial = np.int32(400))
        else:
            [_, _, trial_duration, _, stimuli, m_neuron, v_neuron] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_for_data, save_PE=save_PE)
    
    else: # load results from previous simulation
        
        if save_PE:
            with open(file_for_data,'rb') as f:
                [_, _, trial_duration, num_values_per_trial, stimuli, m_neuron, v_neuron, PE] = pickle.load(f)
        else:
            with open(file_for_data,'rb') as f:
                [_, _, trial_duration, num_values_per_trial, stimuli, m_neuron, v_neuron] = pickle.load(f)
            
    ### plot single panels
    illustrate_PE_establish_M(m_neuron, PE, stimuli, trial_duration, num_values_per_trial, [0, 20000], [180000, 200000])