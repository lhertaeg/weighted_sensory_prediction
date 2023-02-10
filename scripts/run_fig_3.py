#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.functions_simulate import simulate_weighting_example, simulate_weighting_exploration, simulate_dynamic_weighting_eg
from src.plot_data import plot_weighting_limit_case_example, plot_fraction_sensory_heatmap, plot_transitions_examples

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Two limit cases for weighting

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data_prediction_driven = '../results/data/weighting/data_example_limit_case_prediction_driven.pickle'
    file_for_data_sensory_driven = '../results/data/weighting/data_example_limit_case_sensory_driven.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
        
        min_mean, max_mean, m_sd, n_sd = 5, 5, 0, 2.3
        [n_trials, trial_duration, _, stimuli_1, m_neuron_lower_1, v_neuron_lower_1, m_neuron_higher_1, v_neuron_higher_1, alpha_1, beta_1, 
         weighted_output_1] = simulate_weighting_example(mfn_flag, min_mean, max_mean, m_sd, n_sd, 
                                                         file_for_data = file_for_data_prediction_driven)

        min_mean, max_mean, m_sd, n_sd = 1, 9, 0, 0
        [_, _, _, stimuli_2, m_neuron_lower_2, v_neuron_lower_2, m_neuron_higher_2, v_neuron_higher_2, alpha_2, beta_2, 
         weighted_output_2] = simulate_weighting_example(mfn_flag, min_mean, max_mean, m_sd, n_sd, 
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
                                      v_neuron_lower_2, v_neuron_higher_2, alpha_2, beta_2, weighted_output_2, 
                                      plot_legend = False)
    
    
# %% Systematic exploration

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_exploration.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
        
        ### within and across trial variabilities tested
        mean_trials, m_sd = dtype(5), dtype(0)
        variability_within = np.linspace(0,3,5, dtype=dtype)
        variability_across = np.linspace(0,3,5, dtype=dtype)
        
        [variability_within, 
         variability_across, alpha] = simulate_weighting_exploration(mfn_flag, variability_within, variability_across, 
                                                                     mean_trials, m_sd, file_for_data = file_for_data)
        
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [variability_within, variability_across, alpha] = pickle.load(f)
            
    ### plot data
    plot_fraction_sensory_heatmap(alpha, variability_within, variability_across, 2, 
                                xlabel='variability across trial', 
                                ylabel='variability within trial')
    
    
# %% Examples: system dynamically adjusts to input statistics and environement

# 5,5,0,0 --> 5,5,0,3 --> 0,10,0,3 --> 0,10,0,0 --> 5,5,0,0

# Todo:
    # check implementation
    # beautify plotting function
    # re-run for correct examples and parameters

run_cell = False
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
    n_sd_after = 0

    ### filename for data
    before = str(min_mean_before) + str(max_mean_before) + str(m_sd_before) + str(n_sd_before)
    after = str(min_mean_after) + str(max_mean_after) + str(m_sd_after) + str(n_sd_after)
    file_for_data = '../results/data/weighting/data_weighting_transition_' + before + '_' + after + '.pickle'
    
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
    plot_transitions_examples(n_trials, trial_duration, stimuli, alpha, beta, weighted_output, 
                              time_plot=0, ylim=[-15,20], figsize=(4,3), xlim=[50,70], plot_only_weights=True)

