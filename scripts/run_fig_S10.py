#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:54:02 2024

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.functions_simulate import simulate_dynamic_weighting_eg
from src.plot_data import plot_output_different_weightings

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Run example in which the network faces a sudden transition to a state with high sensory noise 
# compare different approaches

run_cell = True
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### define stimulus statistics before and after transition
    min_mean_before = 0
    max_mean_before = 10
    m_sd_before = 0
    n_sd_before = 0
    
    min_mean_after = 5
    max_mean_after = 5
    m_sd_after = 0
    n_sd_after = 3

    ### filename for data
    before = str(min_mean_before) + str(max_mean_before) + str(m_sd_before) + str(n_sd_before)
    after = str(min_mean_after) + str(max_mean_after) + str(m_sd_after) + str(n_sd_after)
    file_for_data = '../results/data/weighting/data_transition_' + mfn_flag + '_' + before + '_' + after + '_compare.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
        
        [n_trials, trial_duration, _, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
         v_neuron_higher, alpha, beta, weighted_output] = simulate_dynamic_weighting_eg(mfn_flag, min_mean_before, max_mean_before, m_sd_before, 
                                                          n_sd_before, min_mean_after, max_mean_after, m_sd_after, 
                                                          n_sd_after, file_for_data = file_for_data)     
        
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [n_trials, trial_duration, _, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
             v_neuron_higher, alpha, beta, weighted_output] = pickle.load(f)
            
    ### plot data    
    plot_output_different_weightings(n_trials, trial_duration, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
                                     v_neuron_higher, weighted_output)
    
