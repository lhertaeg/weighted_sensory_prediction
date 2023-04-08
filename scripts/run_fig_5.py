#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
from src.functions_simulate import simulate_weighting_example, stimuli_moments_from_uniform
from src.plot_data import plot_trial_mean_vs_sd, plot_example_contraction_bias
import pickle

import matplotlib.pyplot as plt

dtype = np.float32

# %% Notes 

### CHECK IPAD AGAIN ... slope is not sensory weight (only in limit case)
# how can we express P as function of current S?

# If I get interesting results for contraction bias, I should show that figure before neuromod fig 
# and actually discuss what a neuromod does to the bias!

# %% Increasing the range of trail means will lead to decreased slope (SD of input != 0) - examples

run_cell = True
plot_only = False

if run_cell:
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### define statistics
    m_std, n_std = dtype(0), dtype(5)
    min_mean_1, max_mean_1 = dtype(15), dtype(25)
    min_mean_2, max_mean_2 = dtype(10), dtype(30)

    ### filenames for data
    file_for_data_1 = '../results/data/behavior/data_contraction_bias_trial_mean_range_' + str(max_mean_1 - min_mean_1) + '_SD_' + str(n_std) + '.pickle'
    file_for_data_2 = '../results/data/behavior/data_contraction_bias_trial_mean_range_' + str(max_mean_2 - min_mean_2) + '_SD_' + str(n_std) + '.pickle'
    
    ### get data
    if not plot_only:
        
        ## run for smaller range
        [n_trials, _, _, stimuli_1, _, _, _, _, a_1, _, weighted_output_1, trial_means_1] = simulate_weighting_example(mfn_flag, min_mean_1, max_mean_1, 
                                                                                                                     m_std, n_std, n_trials=n_trials,
                                                                                                                     file_for_data = file_for_data_1)
        
        ## run for larger range 
        [n_trials, _, _, stimuli_2, _, _, _, _, a_2, _, weighted_output_2, trial_means_2] = simulate_weighting_example(mfn_flag, min_mean_2, max_mean_2, 
                                                                                                                     m_std, n_std, n_trials=n_trials,
                                                                                                                     file_for_data = file_for_data_2)
                                                                          
    else:
        
        with open(file_for_data_1,'rb') as f:
            [n_trials, _, _, stimuli_1, _, _, _, _, _, _, weighted_output_1, trial_means_1] = pickle.load(f)
            
        with open(file_for_data_2,'rb') as f:
            [n_trials, _, _, stimuli_2, _, _, _, _, _, _, weighted_output_2, trial_means_2] = pickle.load(f)
            
    
    ### plot data 
    weighted_output = np.vstack((weighted_output_1, weighted_output_2))
    stimuli = np.vstack((stimuli_1, stimuli_2))
    
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30))     


# %% Even without input SD contraction bias occurs - examples

run_cell = False
plot_only = True

if run_cell:
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### define statistics
    m_std, n_std = dtype(0), dtype(0)
    min_mean_1, max_mean_1 = dtype(15), dtype(25)
    min_mean_2, max_mean_2 = dtype(10), dtype(30)

    ### filenames for data
    file_for_data_1 = '../results/data/behavior/data_contraction_bias_trial_mean_range_' + str(max_mean_1 - min_mean_1) + '.pickle'
    file_for_data_2 = '../results/data/behavior/data_contraction_bias_trial_mean_range_' + str(max_mean_2 - min_mean_2) + '.pickle'
    
    ### get data
    if not plot_only:
        
        ## run for smaller range
        [n_trials, _, _, stimuli_1, _, _, _, _, _, _, weighted_output_1, trial_means_1] = simulate_weighting_example(mfn_flag, min_mean_1, max_mean_1, 
                                                                                                                     m_std, n_std, n_trials=n_trials,
                                                                                                                     file_for_data = file_for_data_1)
        
        ## run for larger range 
        [n_trials, _, _, stimuli_2, _, _, _, _, _, _, weighted_output_2, trial_means_2] = simulate_weighting_example(mfn_flag, min_mean_2, max_mean_2, 
                                                                                                                     m_std, n_std, n_trials=n_trials,
                                                                                                                     file_for_data = file_for_data_2)
                                                                          
    else:
        
        with open(file_for_data_1,'rb') as f:
            [n_trials, _, _, stimuli_1, _, _, _, _, _, _, weighted_output_1, trial_means_1] = pickle.load(f)
            
        with open(file_for_data_2,'rb') as f:
            [n_trials, _, _, stimuli_2, _, _, _, _, _, _, weighted_output_2, trial_means_2] = pickle.load(f)
            
    
    ### plot data 
    weighted_output = np.vstack((weighted_output_1, weighted_output_2))
    stimuli = np.vstack((stimuli_1, stimuli_2))
    
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30))                                                                          


# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %% Illustrate mean and std of trials (stimuli) for two cases: 1) without scalar variability, 2) with scalar variability

run_cell = False

if run_cell:
    
    ### define numbe rof trials and values drawn per trail
    n_trials = 1000
    num_values_per_trial = 10
    
    ### compute 
    stimuli_without = stimuli_moments_from_uniform(n_trials, num_values_per_trial, min_mean=10, max_mean=20, m_sd=0, n_sd=7.5)
    stimuli_with = stimuli_moments_from_uniform(n_trials, num_values_per_trial, min_mean=10, max_mean=20, m_sd=0.5, n_sd=0)
    
    ### plot
    plot_trial_mean_vs_sd(stimuli_without, n_trials)
    plot_trial_mean_vs_sd(stimuli_with, n_trials)
    

# %% Example contraction bias, no scalar variabilty

run_cell = False
plot_only = True

if run_cell:
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/behavior/data_contraction_bias_example_without_scalar_variability.pickle'
    
    ### get data
    if not plot_only:
        
        ## define statistics
        min_mean, max_mean, m_std, n_std = dtype(10), dtype(20), dtype(0), dtype(2)
    
        ## run
        [n_trials, _, _, stimuli, _, _, _, _, _, _, weighted_output] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                                  m_std, n_std, n_trials=n_trials,
                                                                                                  file_for_data = file_for_data)
                                                                          
    else:
        
        with open(file_for_data,'rb') as f:
            [n_trials, _, _, stimuli, _, _, _, _, _, _, weighted_output] = pickle.load(f)
            
    
    ### plot data 
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30))                                                                          
    
    
# %% Example contraction bias, with scalar variabilty

run_cell = False
plot_only = True

if run_cell:
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/behavior/data_contraction_bias_example_with_scalar_variability.pickle'
    
    ### get data
    if not plot_only:
        
        ## define statistics
        min_mean, max_mean, m_std, n_std = dtype(10), dtype(20), dtype(0.15), dtype(0)
    
        ## run
        [n_trials, _, _, stimuli, _, _, _, _, alpha, _, weighted_output] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                                  m_std, n_std, n_trials=n_trials,
                                                                                                  file_for_data = file_for_data)
                                                                          
    else:
        
        with open(file_for_data,'rb') as f:
            [n_trials, _, _, stimuli, _, _, _, _, alpha, _, weighted_output] = pickle.load(f)
            
    
    ### plot data 
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30)) 
 
    
# %% Tests

run_cell = False
plot_only = True

if run_cell:
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/behavior/data_contraction_bias_test.pickle'
    
    ### get data
    if not plot_only:
        
        ## define statistics
        min_mean, max_mean, m_std, n_std = dtype(10), dtype(50), dtype(0), dtype(9)
    
        ## run
        [n_trials, _, _, stimuli, _, _, _, _, alpha, _, weighted_output, trial_means] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                                                   m_std, n_std, n_trials=n_trials,
                                                                                                                   natural_numbers=False,
                                                                                                                   file_for_data = file_for_data)
                                                                          
    else:
        
        with open(file_for_data,'rb') as f:
            [n_trials, _, _, stimuli, _, _, _, _, alpha, _, weighted_output, trial_means] = pickle.load(f)
            
    
    ### plot data 
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30))#, trial_means=trial_means)
    

# be careful: if rectification kicks in, lower end is usually bigger than upper end!!!!! Think it through

# two dimensions: 1) is bias at lower end < bias upper end? 2) diff between cases with and without scalar variability
# 4 possibilities:
    # i) yes, yes --> kinda expected
    # ii) yes, no --> could also be interesting, question would be why
    # iii) no, yes --> why does it work without scalar variability but not with, as predicted?
    # iv) no, no --> why doesn't scalar variability make a difference?