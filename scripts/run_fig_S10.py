#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 18:45:22 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.functions_simulate import simulate_impact_para
from src.plot_data import plot_impact_para, plot_schema_inputs_tested

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Simulate control case for number of input statistics 

run_cell = True
load_only = True
do_plot = False

# define parameters used throughout the file
mean_trials, m_sd = dtype(5), dtype(0)
variability_within = np.array([0, 0.75, 1.5, 2.25, 3])
variability_across = np.array([3, 2.25, 1.5, 0.75, 0])
last_n = np.int32(30)

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_control.pickle'
    
    ### get data
    if not load_only: # simulate respective network
        
        [stimuli, n, gain_w_PE_to_P,
         gain_v_PE_to_P, add_input, 
         id_cells_modulated, weight] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                        m_sd, last_n = last_n, file_for_data = file_for_data)
        
    else:
        
        with open(file_for_data,'rb') as f:
            [stimuli, n_ctrl, gain_w_PE_to_P_ctrl, gain_v_PE_to_P_ctrl, 
             add_input_ctrl, id_cells_modulated_ctrl, weight_ctrl] = pickle.load(f)
            
    if do_plot:
        
        plot_schema_inputs_tested()
        

# %% Impact of activation function

run_cell = False
plot_only = True

if run_cell:

    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_activation_function.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
        
        print('Test different activation function for V neuron:\n')
        [_, _, _, _, _, _, weight_act] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                                          m_sd, last_n = last_n, n = 1, file_for_data = file_for_data)
        
        
    else:
        
        with open(file_for_data,'rb') as f:
            [_, _, _, _, _, _, weight_act] = pickle.load(f)
            
    
    ### plot data 
    plot_impact_para(variability_across, weight_ctrl, weight_act, fs=7, ms=7)
    
    
# %% Impact of connectivity parameters (weight from PE neurons to M neuron in lower PE circuit)
    
run_cell = True
plot_only = False

if run_cell:

    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_connectivity_lower.pickle'
    
    ### weights (gain) to be tested
    gains_lower = np.array([0.3, 7], dtype=dtype) 
    num_gains = len(gains_lower)
    
    ### initialise
    weights_mod_con_lower = np.zeros((len(variability_across), num_gains))
    
    ### get data
    if not plot_only: # simulate respective network
    
        print('Test different gain factors for PE to M neuron in lower PE circuit:\n')
    
        for k, gain in enumerate(gains_lower):
        
            print('Gain:', gain)
            [_, _, _, _, _, _, weight] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                                              m_sd, last_n = last_n, gain_w_PE_to_P=gain)
            
            weights_mod_con_lower[:,k] = weight
            
        with open(file_for_data,'wb') as f:
            pickle.dump([gains_lower, weights_mod_con_lower],f) 
        
    else:
        
        with open(file_for_data,'rb') as f:
            [gains_lower, weights_mod_con_lower] = pickle.load(f)
            
