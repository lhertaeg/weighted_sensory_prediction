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
from src.functions_simulate import simulate_activity_neurons
from src.plot_data import plot_weighting_limit_case_example, plot_fraction_sensory_heatmap, plot_transitions_examples, plot_neuron_activity_lower_higher

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
    file_for_data_prediction_driven = '../results/data/weighting/data_example_limit_case_prediction_driven_' + mfn_flag + '.pickle'
    file_for_data_sensory_driven = '../results/data/weighting/data_example_limit_case_sensory_driven_' + mfn_flag + '.pickle'
    
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
    
    
# %% Systematic exploration - record sensory weight

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_exploration_' + mfn_flag + '.pickle'
    
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
    

# %% Probably all below can be erased ....

# %% Systematic exploration - record neuron activity

run_cell = True
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data_within = '../results/data/weighting/data_activity_neurons_exploration_within_' + mfn_flag + '.pickle'
    file_for_data_across = '../results/data/weighting/data_activity_neurons_exploration_across_' + mfn_flag + '.pickle'
    seeds = np.array([10,20,30,40,50])
    
    ### get data
    if not plot_only: # simulate respective network
        
        ### within and across trial variabilities tested
        last_n = np.int32(100)
        n_trials = np.int32(200)
        
        mean_trials, m_sd = dtype(5), dtype(0)
        variability_within = np.linspace(0,3,5, dtype=dtype)
        variability_across = np.linspace(0,3,5, dtype=dtype)
            
        [_, _, activity_pe_neurons_lower_within, 
         activity_pe_neurons_higher_within, 
         activity_interneurons_lower_within, 
         activity_interneurons_higher_within] = simulate_activity_neurons(mfn_flag, variability_within, np.array([0], dtype=dtype), 
                                                                          mean_trials, m_sd, seeds  = seeds,
                                                                          file_for_data = file_for_data_within,
                                                                          last_n = last_n, n_trials = n_trials)
                                                                               
        [_, _, activity_pe_neurons_lower_across, 
         activity_pe_neurons_higher_across,
         activity_interneurons_lower_across,
         activity_interneurons_higher_across] = simulate_activity_neurons(mfn_flag, np.array([0], dtype=dtype), variability_across,
                                                                               mean_trials, m_sd, seeds  = seeds,
                                                                               file_for_data = file_for_data_across,
                                                                               last_n = last_n, n_trials = n_trials)                                                                  
        
    else: # load results from previous simulation

        with open(file_for_data_within,'rb') as f:
            [variability_within, _, activity_pe_neurons_lower_within, activity_pe_neurons_higher_within, 
             activity_interneurons_lower_within, activity_interneurons_higher_within] = pickle.load(f)
            
        with open(file_for_data_across,'rb') as f:
            [_, variability_across, activity_pe_neurons_lower_across, activity_pe_neurons_higher_across, 
             activity_interneurons_lower_across, activity_interneurons_higher_across] = pickle.load(f)
            
    ### plot data
    plot_neuron_activity_lower_higher(variability_within, variability_across, activity_interneurons_lower_within,
                                      activity_interneurons_higher_within, activity_interneurons_lower_across, 
                                      activity_interneurons_higher_across, activity_pe_neurons_lower_within,
                                      activity_pe_neurons_higher_within, activity_pe_neurons_lower_across,
                                      activity_pe_neurons_higher_across)
    

# %% Test see above 

# S, P, PP => S + |S-P|, P + |S-P|, P + |P-PP|, PP + |P-PP|
# either for two different stimulus var or two different trail var!!!

run_cell = False

if run_cell:
    
    mfn_flag = '10'
    
    min_mean, max_mean, m_sd, n_sd = 10, 10, 0, 0
    [n_trials, trial_duration, _, stimuli_1, m_neuron_lower_1, v_neuron_lower_1, m_neuron_higher_1, v_neuron_higher_1, alpha_1, beta_1, 
     weighted_output_1, _] = simulate_weighting_example(mfn_flag, min_mean, max_mean, m_sd, n_sd)

    mean = 10
    sd = 5
    b = np.sqrt(12) * sd / 2 + mean
    a = 2 * mean - b

    min_mean, max_mean, m_sd, n_sd = a, b, 0, 0
    [_, _, _, stimuli_2, m_neuron_lower_2, v_neuron_lower_2, m_neuron_higher_2, v_neuron_higher_2, alpha_2, beta_2, 
     weighted_output_2, _] = simulate_weighting_example(mfn_flag, min_mean, max_mean, m_sd, n_sd)
    
                                                     
    import matplotlib.pyplot as plt
    import seaborn as sns   

    # plt.figure()
    # plt.plot(stimuli_1 + abs(m_neuron_lower_1 - stimuli_1))
    # plt.plot(stimuli_2 + abs(m_neuron_lower_2 - stimuli_2))
    
    # plt.figure()
    # plt.plot(m_neuron_lower_1 + abs(m_neuron_lower_1 - stimuli_1))
    # plt.plot(m_neuron_lower_2 + abs(m_neuron_lower_2 - stimuli_2))
    
    # plt.figure()
    # plt.plot(m_neuron_lower_1 + abs(m_neuron_lower_1 - m_neuron_higher_1))
    # plt.plot(m_neuron_lower_2 + abs(m_neuron_lower_2 - m_neuron_higher_2))
    
    # plt.figure()
    # plt.plot(m_neuron_higher_1 + abs(m_neuron_lower_1 - m_neuron_higher_1))
    # plt.plot(m_neuron_higher_2 + abs(m_neuron_lower_2 - m_neuron_higher_2))
    
    plt.figure()
    plt.plot(stimuli_1)
    plt.plot(m_neuron_lower_1)
    plt.plot(m_neuron_higher_1)
    
    plt.figure()
    plt.plot(stimuli_2)
    plt.plot(m_neuron_lower_2)
    plt.plot(m_neuron_higher_2)
   
   
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
    n_sd_before = 3
    
    min_mean_after = 0
    max_mean_after = 10
    m_sd_after = 0
    n_sd_after = 0

    ### filename for data
    before = str(min_mean_before) + str(max_mean_before) + str(m_sd_before) + str(n_sd_before)
    after = str(min_mean_after) + str(max_mean_after) + str(m_sd_after) + str(n_sd_after)
    file_for_data = '../results/data/weighting/data_weighting_transition_' + mfn_flag + '_' + before + '_' + after + '.pickle'
    
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

