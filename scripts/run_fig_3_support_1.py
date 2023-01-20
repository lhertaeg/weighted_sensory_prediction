#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.functions_simulate import simulate_sensory_weight_time_course, simulate_impact_para
from src.plot_data import plot_weight_over_trial

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Simulate control case for number of input statistics 

# Todo:
    # check implementation
    # choose parameters
    # re-run

run_cell = False
plot_only = False

# define parameters used throughout the file
mean_trials, min_std = dtype(3), dtype(0)
variability_within = np.array([0, 0.75, 1.5, 2.25, 3])
variability_across = np.array([3, 2.25, 1.5, 0.75, 0])
last_n = np.int32(50)
n_trials = np.int32(200)

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_control.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
        
        [stimuli, n, gain_w_PE_to_P,
         gain_v_PE_to_P, add_input, 
         id_cells_modulated, weight] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                        min_std, last_n = last_n, n_trials = n_trials, file_for_data = file_for_data)
        
    else:
        
        with open(file_for_data,'rb') as f:
            [stimuli, n_ctrl, gain_w_PE_to_P_ctrl, gain_v_PE_to_P_ctrl, 
             add_input_ctrl, id_cells_modulated_ctrl, weight_ctrl] = pickle.load(f)
    

# %% Impact of activation function

# Todo:
    # check implementation
    # write plotting function
    # re-run and beautify
    
run_cell = False
plot_only = False

if run_cell:

    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_activation_function.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
        
        [_, _, _, _, _, _, weight_act] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                                          min_std, last_n = last_n, n_trials = n_trials, 
                                                          n = 1, file_for_data = file_for_data)
        
        
    else:
        
        with open(file_for_data,'rb') as f:
            [_, _, _, _, _, _, weight_act] = pickle.load(f)
            
    
    ### plot data
    XXXX 
    
    
# %% Impact of baseline activity (all neuron types)

# Todo:
    # check implementation
    # write plotting function
    # re-run and beautify
    
run_cell = False
plot_only = False

if run_cell:

    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_baseline_all.pickle'
    
    ### additional input to be tested
    add_inputs = np.array([0.5, 1, 1.5, 2])
    num_inputs = len(add_inputs)
    
    ### initialise
    weights_mod_base = np.zeros((len(variability_across), num_inputs))
    
    ### get data
    if not plot_only: # simulate respective network
    
        for k, add_input in enumerate(add_inputs):
        
            [_, _, _, _, _, _, weight] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                                              min_std, last_n = last_n, n_trials = n_trials, 
                                                              add_input=add_input)
            
            weights_mod_base[:,k] = weight
            
        with open(file_for_data,'wb') as f:
            pickle.dump([add_inputs, weights_mod_base],f) 
        
    else:
        
        with open(file_for_data,'rb') as f:
            [add_inputs, weights_mod_base] = pickle.load(f)
            
    
    ### plot data
    XXXX 

# %% Impact of connectivity parameters (weight from PE neurons to M neuron in lower PE circuit)

# Todo:
    # check implementation
    # write plotting function
    # re-run and beautify
    
run_cell = False
plot_only = False

if run_cell:

    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_baseline_all.pickle'
    
    ### weights (gain) to be tested
    gains_lower = np.array([0.5, 1, 1.5, 2]) # !!!!!!!!!! decide on
    num_gains = len(gains_lower)
    
    ### initialise
    weights_mod_con_lower = np.zeros((len(variability_across), num_inputs))
    
    ### get data
    if not plot_only: # simulate respective network
    
        for k, gain in enumerate(gains_lower):
        
            [_, _, _, _, _, _, weight] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                                              min_std, last_n = last_n, n_trials = n_trials, 
                                                              gain_w_PE_to_P=gain)
            
            weights_mod_con_lower[:,k] = weight
            
        with open(file_for_data,'wb') as f:
            pickle.dump([gains_lower, weights_mod_con_lower],f) 
        
    else:
        
        with open(file_for_data,'rb') as f:
            [gains_lower, weights_mod_con_lower] = pickle.load(f)
            
    
    ### plot data
    XXXX 

# %% Impact of connectivity parameters (weight from PE neurons to M neuron in higher PE circuit)

# Todo:
    # check implementation
    # write plotting function
    # re-run and beautify
    
run_cell = False
plot_only = False

if run_cell:

    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_baseline_all.pickle'
    
    ### weights (gain) to be tested
    gains_higher = np.array([0.5, 1, 1.5, 2]) # !!!!!!!!!! decide on
    num_gains = len(gains_higher)
    
    ### initialise
    weights_mod_con_higher = np.zeros((len(variability_across), num_inputs))
    
    ### get data
    if not plot_only: # simulate respective network
    
        for k, gain in enumerate(gains_higher):
        
            [_, _, _, _, _, _, weight] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                                              min_std, last_n = last_n, n_trials = n_trials, 
                                                              gain_v_PE_to_P=gain)
            
            weights_mod_con_higher[:,k] = weight
            
        with open(file_for_data,'wb') as f:
            pickle.dump([gains_higher, weights_mod_con_higher],f) 
        
    else:
        
        with open(file_for_data,'rb') as f:
            [gains_higher, weights_mod_con_higher] = pickle.load(f)
            
    
    ### plot data
    XXXX 


# %% Impact of baseline activity (only PE neurons)

# Todo:
    # check implementation
    # write plotting function
    # re-run and beautify
    
run_cell = False
plot_only = False

if run_cell:

    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_baseline_PE.pickle'
    
    ### additional input to be tested
    id_cells_modulated = np.array([True,True,False,False,False,False,False,False])
    add_inputs = np.array([0.5, 1, 1.5, 2])
    num_inputs = len(add_inputs)
    
    ### initialise
    weights_mod_base = np.zeros((len(variability_across), num_inputs))
    
    ### get data
    if not plot_only: # simulate respective network
    
        for k, add_input in enumerate(add_inputs):
        
            [_, _, _, _, _, _, weight] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                                              min_std, last_n = last_n, n_trials = n_trials, 
                                                              add_input=add_input, id_cells_modulated = id_cells_modulated)
            
            weights_mod_base[:,k] = weight
            
        with open(file_for_data,'wb') as f:
            pickle.dump([add_inputs, weights_mod_base],f) 
        
    else:
        
        with open(file_for_data,'rb') as f:
            [add_inputs, weights_mod_base] = pickle.load(f)
            
    
    ### plot data
    XXXX 


# %% Impact of trial duration

# Todo:
    # check implementation
    # choose parameters and re-run
    # beautify plotting function --> actually make comparison of both !!!

run_cell = False
plot_only = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data_control = '../results/data/weighting/data_weighting_trail_duration_control.pickle'
    file_for_data_shorter = '../results/data/weighting/data_weighting_trail_duration_shorter.pickle'

    ### get data
    if not plot_only: # simulate respective network
        
        trial_duration_control = np.int32(5000)
        trial_duration_shorter = np.int32(1000)
        
        [n_trials, _,  _, weight_control] = simulate_sensory_weight_time_course(mfn_flag, variability_within, variability_across, mean_trials, 
                                                               min_std, trial_duration = trial_duration_control, n_trials=n_trials,
                                                               file_for_data = file_for_data_control)
                                                       
        [n_trials, _, _, weight_shorter] = simulate_sensory_weight_time_course(mfn_flag, variability_within, variability_across, mean_trials, 
                                                               min_std, trial_duration = trial_duration_control, n_trials=n_trials,
                                                               file_for_data = file_for_data_shorter)
        
    else: # load results from previous simulation

        with open(file_for_data_control,'rb') as f:
            [n_trials, _, _, weight_control] = pickle.load(f)
            
        with open(file_for_data_shorter,'rb') as f:
            [n_trials_, _, weight_shorter] = pickle.load(f)
            
    ### plot data
    plot_weight_over_trial(weight_control, n_trials)
    
