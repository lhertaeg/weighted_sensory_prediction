#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.functions_simulate import simulate_weighting_example, simulate_weighting_exploration, simulate_sensory_weight_time_course
from src.plot_data import plot_weighting_limit_case_example, plot_fraction_sensory_heatmap, plot_weight_over_trial

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Two limit cases for weighting

run_cell = False
plot_only = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data_prediction_driven = '../results/data/weighting/data_example_limit_case_prediction_driven_' + mfn_flag + '.pickle'
    file_for_data_sensory_driven = '../results/data/weighting/data_example_limit_case_sensory_driven_' + mfn_flag + '.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
        
        min_mean, max_mean, m_sd, n_sd = 5, 5, 0, np.sqrt(5)
        [n_trials, trial_duration, _, stimuli_1, m_neuron_lower_1, v_neuron_lower_1, m_neuron_higher_1, v_neuron_higher_1, alpha_1, beta_1, 
         weighted_output_1, _] = simulate_weighting_example(mfn_flag, min_mean, max_mean, m_sd, n_sd, 
                                                            file_for_data = file_for_data_prediction_driven)

        min_mean, max_mean, m_sd, n_sd = 1, 9, 0, 0
        [_, _, _, stimuli_2, m_neuron_lower_2, v_neuron_lower_2, m_neuron_higher_2, v_neuron_higher_2, alpha_2, beta_2, 
         weighted_output_2, _] = simulate_weighting_example(mfn_flag, min_mean, max_mean, m_sd, n_sd, 
                                                            file_for_data = file_for_data_sensory_driven)                                                
                                                            
    else: # load results from previous simulation

        with open(file_for_data_prediction_driven,'rb') as f:
            [n_trials, trial_duration, _, stimuli_1, m_neuron_lower_1, v_neuron_lower_1, m_neuron_higher_1, 
             v_neuron_higher_1, alpha_1, beta_1, weighted_output_1,_] = pickle.load(f)
            
        with open(file_for_data_sensory_driven,'rb') as f:
            [_, _, _, stimuli_2, m_neuron_lower_2, v_neuron_lower_2, m_neuron_higher_2, v_neuron_higher_2, 
             alpha_2, beta_2, weighted_output_2,_] = pickle.load(f)
            
    ### plot single panels
    plot_weighting_limit_case_example(n_trials, trial_duration, stimuli_1, m_neuron_lower_1, m_neuron_higher_1, 
                                      v_neuron_lower_1, v_neuron_higher_1, alpha_1, beta_1, weighted_output_1)
    
    plot_weighting_limit_case_example(n_trials, trial_duration, stimuli_2, m_neuron_lower_2, m_neuron_higher_2, 
                                      v_neuron_lower_2, v_neuron_higher_2, alpha_2, beta_2, weighted_output_2, 
                                      plot_legend = False)
    
    
# %% Systematic exploration - record sensory weight

run_cell = False
plot_only = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/weighting/data_weighting_exploration_' + mfn_flag + '.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
        
        ### within and across trial variabilities tested
        mean_trials, m_sd = dtype(5), dtype(0)
        variability_within = np.linspace(0,5,6, dtype=dtype)
        variability_across = np.linspace(0,5,6, dtype=dtype)
        
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
    

# %% Impact of trial duration

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
        
        print('Test different trial durations:\n')
        
        mean_trials, m_sd = dtype(5), dtype(0)
        variability_within = np.array([0, 0.75, 1.5, 2.25, 3])
        variability_across = np.array([3, 2.25, 1.5, 0.75, 0])
    
        trial_duration_control = np.int32(5000)
        trial_duration_shorter = np.int32(1000)
        
        [n_trials, _,  _, weight_control] = simulate_sensory_weight_time_course(mfn_flag, variability_within, variability_across, mean_trials, 
                                                               m_sd, trial_duration = trial_duration_control,
                                                               file_for_data = file_for_data_control)
                                                       
        [n_trials, _, _, weight_shorter] = simulate_sensory_weight_time_course(mfn_flag, variability_within, variability_across, mean_trials, 
                                                               m_sd, trial_duration = trial_duration_shorter,
                                                               file_for_data = file_for_data_shorter)
        
    else: # load results from previous simulation

        with open(file_for_data_control,'rb') as f:
            [n_trials, _, _, weight_control] = pickle.load(f)
            
        with open(file_for_data_shorter,'rb') as f:
            [n_trials, _, _, weight_shorter] = pickle.load(f)
            
    ### plot data
    plot_weight_over_trial(weight_control, weight_shorter, n_trials, 
                           leg_text=['trial duration = T', 'trial duration = T/5'])
    
    