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
from src.plot_data import plot_weight_over_trial, plot_impact_para, plot_schema_inputs_tested

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Simulate control case for number of input statistics 

run_cell = False
load_only = True
do_plot = False

# define parameters used throughout the file
mean_trials, m_sd = dtype(5), dtype(0)
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
    if not load_only: # simulate respective network
        
        [stimuli, n, gain_w_PE_to_P,
         gain_v_PE_to_P, add_input, 
         id_cells_modulated, weight] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                        m_sd, last_n = last_n, n_trials = n_trials, file_for_data = file_for_data)
        
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
                                                          m_sd, last_n = last_n, n_trials = n_trials, 
                                                          n = 1, file_for_data = file_for_data)
        
        
    else:
        
        with open(file_for_data,'rb') as f:
            [_, _, _, _, _, _, weight_act] = pickle.load(f)
            
    
    ### plot data 
    plot_impact_para(weight_ctrl, weight_act, fs=7, ms=7)
    
    
# %% Impact of baseline activity (all neuron types)
    
run_cell = False
plot_only = True

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
    
        print('Increase baseline activity for all neurons:\n')
        
        for k, add_input in enumerate(add_inputs):
            
            print('Add input:', add_input)
            
            [_, _, _, _, _, _, weight] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                                              m_sd, last_n = last_n, n_trials = n_trials, 
                                                              add_input=add_input)
            
            weights_mod_base[:,k] = weight
            
        with open(file_for_data,'wb') as f:
            pickle.dump([add_inputs, weights_mod_base],f) 
        
    else:
        
        with open(file_for_data,'rb') as f:
            [add_inputs, weights_mod_base] = pickle.load(f)
            
    
    ### plot data
    plot_impact_para(weight_ctrl, weights_mod_base, para_range_tested=add_inputs,
                     colorbar_title = 'baseline', colorbar_tick_labels = ['low','high'],fs=7) 
    
    
# %% Impact of baseline activity (only PE neurons)
    
run_cell = False
plot_only = True

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
    
        print('Increase baseline activity for PE neurons only:\n')
    
        for k, add_input in enumerate(add_inputs):
        
            print('Add input:', add_input)
            [_, _, _, _, _, _, weight] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                                              m_sd, last_n = last_n, n_trials = n_trials, 
                                                              add_input=add_input, id_cells_modulated = id_cells_modulated)
            
            weights_mod_base[:,k] = weight
            
        with open(file_for_data,'wb') as f:
            pickle.dump([add_inputs, weights_mod_base],f) 
        
    else:
        
        with open(file_for_data,'rb') as f:
            [add_inputs, weights_mod_base] = pickle.load(f)
            
    
    ### plot data
    plot_impact_para(weight_ctrl, weights_mod_base, para_range_tested=add_inputs,
                     colorbar_title = 'baseline', colorbar_tick_labels = ['low','high'],
                     fs=7, loc_position=5) 
    

# %% Impact of connectivity parameters (weight from PE neurons to M neuron in lower PE circuit)
    
run_cell = False
plot_only = True

if run_cell:

    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_connectivity_lower.pickle'
    
    ### weights (gain) to be tested
    gains_lower = np.array([0.3, 1.9, 3.5, 5.1, 6.7], dtype=dtype) 
    num_gains = len(gains_lower)
    
    ### initialise
    weights_mod_con_lower = np.zeros((len(variability_across), num_gains))
    
    ### get data
    if not plot_only: # simulate respective network
    
        print('Test different gain factors for PE to M neuron in lower PE circuit:\n')
    
        for k, gain in enumerate(gains_lower):
        
            print('Gain:', gain)
            [_, _, _, _, _, _, weight] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                                              m_sd, last_n = last_n, n_trials = n_trials, 
                                                              gain_w_PE_to_P=gain)
            
            weights_mod_con_lower[:,k] = weight
            
        with open(file_for_data,'wb') as f:
            pickle.dump([gains_lower, weights_mod_con_lower],f) 
        
    else:
        
        with open(file_for_data,'rb') as f:
            [gains_lower, weights_mod_con_lower] = pickle.load(f)
            
    
    ### plot data
    gain_ratio = np.round(gains_lower / 3.5,1) # in default network, higher/lower = 3.5
    plot_impact_para(weight_ctrl, weights_mod_con_lower, para_range_tested=gains_lower,
                     colorbar_title = r'gain$_\mathrm{PE\rightarrow M}$ ratio', fs=7, 
                     colorbar_tick_labels = [gain_ratio[0] , gain_ratio[-1]])  


# %% Impact of connectivity parameters (weight from PE neurons to M neuron in higher PE circuit)
    
# probbaly not important to show ... one is enough (see above)

run_cell = False
plot_only = False

if run_cell:

    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_connectivity_higher.pickle'
    
    ### weights (gain) to be tested
    gains_higher = np.array([0.1 , 0.19, 0.28, 0.37, 0.46], dtype=dtype) 
    num_gains = len(gains_higher)
    
    ### initialise
    weights_mod_con_higher = np.zeros((len(variability_across), num_gains))
    
    ### get data
    if not plot_only: # simulate respective network
    
        print('Test different gain factors for PE to M neuron in higher PE circuit:\n')
    
        for k, gain in enumerate(gains_higher):
        
            print('Gain:', gain)
            [_, _, _, _, _, _, weight] = simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                                                              m_sd, last_n = last_n, n_trials = n_trials, 
                                                              gain_v_PE_to_P=gain)
            
            weights_mod_con_higher[:,k] = weight
            
        with open(file_for_data,'wb') as f:
            pickle.dump([gains_higher, weights_mod_con_higher],f) 
        
    else:
        
        with open(file_for_data,'rb') as f:
            [gains_higher, weights_mod_con_higher] = pickle.load(f)
            
    
    ### plot data
    plot_impact_para(weight_ctrl, weights_mod_con_higher, para_range_tested=gains_higher,
                     colorbar_title = 'gain', colorbar_tick_labels = [gains_higher[0],gains_higher[-1]],fs=7)  


# %% Impact of trial duration

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data_control = '../results/data/weighting/data_weighting_trail_duration_control.pickle'
    file_for_data_shorter = '../results/data/weighting/data_weighting_trail_duration_shorter.pickle'

    ### get data
    if not plot_only: # simulate respective network
        
        print('Test different trial durations:\n')
    
        trial_duration_control = np.int32(5000)
        trial_duration_shorter = np.int32(1000)
        
        [n_trials, _,  _, weight_control] = simulate_sensory_weight_time_course(mfn_flag, variability_within, variability_across, mean_trials, 
                                                               m_sd, trial_duration = trial_duration_control, n_trials=n_trials,
                                                               file_for_data = file_for_data_control)
                                                       
        [n_trials, _, _, weight_shorter] = simulate_sensory_weight_time_course(mfn_flag, variability_within, variability_across, mean_trials, 
                                                               m_sd, trial_duration = trial_duration_shorter, n_trials=n_trials,
                                                               file_for_data = file_for_data_shorter)
        
    else: # load results from previous simulation

        with open(file_for_data_control,'rb') as f:
            [n_trials, _, _, weight_control] = pickle.load(f)
            
        with open(file_for_data_shorter,'rb') as f:
            [n_trials, _, _, weight_shorter] = pickle.load(f)
            
    ### plot data
    plot_weight_over_trial(weight_control, weight_shorter, n_trials, 
                           leg_text=['trial duration = T', 'trial duration = T/5'])
    
