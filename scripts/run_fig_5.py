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

# Contraction bias occurs also without std in visual stimuli (mean goes from a to b)
# then it is a consequence of the previous stimulus (it takes a while to catch up with the real new stimulus)

# if mean stimulus always the same but std for stimulus high, you also see contration bias, but
# here it is related to the fact that prediction is weighted stronger

# the slope is mainly determined by mean_min and mean_max, std of visual stimuli has effect but 
# comparably littel in comparison to across trial variabilty

# this probably explains why scalar varibailty doesn't seem to have the most prominent effect

# Check again: Does really trail variability define mostly bias?
# Is it correct that stimulus variability does only little to it?
# Should I plot/simulate differently? stimuli only 1, 2, 3, 4, ... ?
# Look at equations: Under which circumstances would I get bias imbalance?
# how does rectification come into play?

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

run_cell = True
plot_only = False

if run_cell:
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/behavior/data_contraction_bias_test.pickle'
    
    ### get data
    if not plot_only:
        
        ## define statistics
        min_mean, max_mean, m_std, n_std = dtype(15), dtype(15), dtype(0), dtype(5)
    
        ## run
        [n_trials, _, _, stimuli, _, _, _, _, alpha, _, weighted_output] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                                  m_std, n_std, n_trials=n_trials,
                                                                                                  file_for_data = file_for_data)
                                                                          
    else:
        
        with open(file_for_data,'rb') as f:
            [n_trials, _, _, stimuli, _, _, _, _, alpha, _, weighted_output] = pickle.load(f)
            
    
    ### plot data 
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30))  
    
    # import numpy as np
    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd
    
    # trials_sensory = np.mean(np.split(stimuli, n_trials),1)
    # trials_alpha = np.mean(np.split(alpha, n_trials),1)
    
    # plt.figure()
    # plt.plot(trials_sensory, trials_alpha, '.')
    
    # trials_estimated = np.mean(np.split(weighted_output, n_trials),1)
    
    # plt.figure()
    # plt.plot(trials_sensory, (trials_estimated - trials_sensory), '.')
    # ax = plt.gca()
    # ax.axhline(0,color='k', ls=':')
    

# maybe look at alpha for each trial mean ... doesn't that already give you estimate of sensory input ... I mean 
# it is exactly the slope isn't it? And then you can immediately get the bias
# because weighted_output = alpha * sensory + beta & prediction
# and in principle you have all quantities, then you simply calculate it for min and max!?

# be careful: if rectification kicks in, lower end is usually bigger than upper end!!!!! Think it through

# two dimensions: 1) is bias at lower end < bias upper end? 2) diff between cases with and without scalar variability
# 4 possibilities:
    # i) yes, yes --> kinda expected
    # ii) yes, no --> could also be interesting, question would be why
    # iii) no, yes --> why does it work without scalar variability but not with, as predicted?
    # iv) no, no --> why doesn't scalar variability make a difference?