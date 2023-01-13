#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.functions_simulate import simulate_weighting_example, simulate_weighting_exploration
from src.plot_data import plot_weighting_limit_case_example

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Two limit cases for weighting

# Todo:
    # choose parameters consistent with the subsequent plots
    # beautify plotting functions
    # rerun

run_cell = False
plot_only = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data_prediction_driven = '../results/data/weighting/data_example_limit_case_prediction_driven.pickle'
    file_for_data_sensory_driven = '../results/data/weighting/data_example_limit_case_sensory_driven.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
        
        min_mean, max_mean, min_std, max_std = 3, 3, 1, 5
        [n_trials, trial_duration, _, stimuli_1, m_neuron_lower_1, v_neuron_lower_1, m_neuron_higher_1, v_neuron_higher_1, alpha_1, beta_1, 
         weighted_output_1] = simulate_weighting_example(mfn_flag, min_mean, max_mean, min_std, max_std, 
                                                         file_for_data = file_for_data_prediction_driven)

        min_mean, max_mean, min_std, max_std = 1, 5, 0, 0
        [_, _, _, stimuli_2, m_neuron_lower_2, v_neuron_lower_2, m_neuron_higher_2, v_neuron_higher_2, alpha_2, beta_2, 
         weighted_output_2] = simulate_weighting_example(mfn_flag, min_mean, max_mean, min_std, max_std, 
                                                         file_for_data = file_for_data_sensory_driven)                                                
                                                          
    else: # load results from previous simulation

        with open(file_for_data_prediction_driven,'rb') as f:
            [n_trials, trial_duration, _, stimuli_1, m_neuron_lower_1, v_neuron_lower_1, m_neuron_higher_1, 
             v_neuron_higher_1, alpha_1, beta_1, weighted_output_1] = pickle.load(f)
            
        with open(file_for_data_sensory_driven,'rb') as f:
            [_, _, _, stimuli_2, m_neuron_lower_2, v_neuron_lower_2, m_neuron_higher_2, v_neuron_higher_2, 
             alpha_2, beta_2, weighted_output_2] = pickle.load(f)
            
    ### plot single panels
    plot_weighting_limit_case_example(n_trials, trial_duration, stimuli_1, m_neuron_lower_1, m_neuron_higher_1, 
                                      v_neuron_lower_1, v_neuron_higher_1, alpha_1, beta_1, weighted_output_1)
    
    plot_weighting_limit_case_example(n_trials, trial_duration, stimuli_2, m_neuron_lower_2, m_neuron_higher_2, 
                                      v_neuron_lower_2, v_neuron_higher_2, alpha_2, beta_2, weighted_output_2)
    
    
# %% Systematic exploration

# Todo:
    # add plotting function
    # choose parameters wisely
    # rerun ...

run_cell = False
plot_only = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_exploration.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
        
        ### within and across trial variabilities tested
        mean_trials, min_std = dtype(3), dtype(0)
        variability_within = np.linspace(0,3,5, dtype=dtype)
        variability_across = np.linspace(0,3,5, dtype=dtype)
        
        [variability_within, 
         variability_across, alpha] = simulate_weighting_exploration(mfn_flag, variability_within, variability_across, 
                                                                     mean_trials, min_std, file_for_data = file_for_data)
        
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [variability_within, variability_across, alpha] = pickle.load(f)
            
    ### plot data
    XXX
        
