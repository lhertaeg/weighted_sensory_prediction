#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:30:07 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
from src.functions_simulate import simulate_weighting_example
from src.plot_data import plot_example_contraction_bias
import pickle

dtype = np.float32

# %% Scalar variability ....

run_cell = True
plot_only = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### define statistics
    # m_std, n_std = dtype(0), dtype(5)
    min_mean_1, max_mean_1 = dtype(15), dtype(25)
    min_mean_2, max_mean_2 = dtype(25), dtype(35)
    m_std, n_std = dtype(1), dtype(-14)

    ### filenames for data
    file_for_data_1 = '../results/data/behavior/data_contraction_bias_trial_mean_range_' + str(max_mean_1 - min_mean_1) + '_max_mean_' + str(max_mean_1) + '.pickle'
    file_for_data_2 = '../results/data/behavior/data_contraction_bias_trial_mean_range_' + str(max_mean_2 - min_mean_2) + '_max_mean_' + str(max_mean_2) + '.pickle'
    
    ### get data
    if not plot_only:
        
        ## run for smaller range
        [n_trials, _, _, stimuli_1, _, _, _, _,
         _, _, weighted_output_1, trial_means_1] = simulate_weighting_example(mfn_flag, min_mean_1, max_mean_1, m_std, n_std,
                                                                              file_for_data = file_for_data_1)
        
        ## run for larger range 
        [n_trials, _, _, stimuli_2, _, _, _, _,
         _, _, weighted_output_2, trial_means_2] = simulate_weighting_example(mfn_flag, min_mean_2, max_mean_2, m_std, n_std,
                                                                              file_for_data = file_for_data_2)
                                                                          
    else:
        
        with open(file_for_data_1,'rb') as f:
            [n_trials, _, _, stimuli_1, _, _, _, _, a1, _, weighted_output_1, trial_means_1] = pickle.load(f)
            
        with open(file_for_data_2,'rb') as f:
            [n_trials, _, _, stimuli_2, _, _, _, _, a2, _, weighted_output_2, trial_means_2] = pickle.load(f)
            
    
    ### plot data 
    weighted_output = np.vstack((weighted_output_2, weighted_output_1))
    stimuli = np.vstack((stimuli_2, stimuli_1))
    min_means = np.array([min_mean_2, min_mean_1])
    max_means = np.array([max_mean_2, max_mean_1])
    m_std = np.array([m_std, m_std])
    n_std = np.array([n_std, n_std])
    
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30), ms=2,
                                  min_means=min_means, max_means=max_means, m_std=m_std, n_std=n_std, figsize=(5,2.8))
    