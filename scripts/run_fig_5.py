#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
from src.functions_simulate import simulate_weighting_example, stimuli_moments_from_uniform, simulate_slope_vs_trial_duration, simulate_slope_vs_variances
from src.plot_data import plot_trial_mean_vs_sd, plot_example_contraction_bias, plot_slope_trail_duration, plot_slope_variability
import pickle

import matplotlib.pyplot as plt

dtype = np.float32

# %% Notes 

### CHECK IPAD AGAIN ... slope is not sensory weight (only in limit case)
# how can we express P as function of current S?

# If I get interesting results for contraction bias, I should show that figure before neuromod fig 
# and actually discuss what a neuromod does to the bias!


# %% If T is very large, dependence on sensory weight is small

# I guess this is also the reason why I get onlt small differences ...

run_cell = False

if run_cell:

    ########
    sigma_s = np.linspace(0,10,21)
    sigma_p = np.linspace(0,10,21)
    
    ss, sp = np.meshgrid(sigma_s, sigma_p)
    
    e = np.exp(-3)
    
    m = e/(1+ss**2/sp**2) + (1-e)
    
    #plt.imshow(m, vmin=0, vmax=1)
    plt.figure()
    plt.contourf(sigma_s, sigma_p, m)
    plt.colorbar()
    
    plt.xlabel('sigma_s')
    plt.ylabel('sigma_p')
    
    #########
    x = np.linspace(0,1,100)
    
    y = e * x + (1-e)
    
    plt.figure()
    plt.plot(x,y)


# %% Scalar variability ....

run_cell = False
plot_only = True

if run_cell:
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### define statistics
    # m_std, n_std = dtype(0), dtype(5)
    min_mean_1, max_mean_1 = dtype(15), dtype(25)
    min_mean_2, max_mean_2 = dtype(25), dtype(35)

    ### filenames for data
    file_for_data_1 = '../results/data/behavior/data_contraction_bias_trial_mean_range_' + str(max_mean_1 - min_mean_1) + '_max_mean_' + str(max_mean_1) + '.pickle'
    file_for_data_2 = '../results/data/behavior/data_contraction_bias_trial_imean_range_' + str(max_mean_2 - min_mean_2) + '_max_mean_' + str(max_mean_2) + '.pickle'
    
    ### get data
    if not plot_only:
        
        ## run for smaller range
        m_std, n_std = dtype(0.25), dtype(0)
        [n_trials, _, _, stimuli_1, _, _, _, _, _, _, weighted_output_1, trial_means_1] = simulate_weighting_example(mfn_flag, min_mean_1, max_mean_1, 
                                                                                                                     m_std, n_std, n_trials=n_trials,
                                                                                                                     file_for_data = file_for_data_1)
        
        ## run for larger range 
        m_std, n_std = dtype(0.25), dtype(0)
        [n_trials, _, _, stimuli_2, _, _, _, _, _, _, weighted_output_2, trial_means_2] = simulate_weighting_example(mfn_flag, min_mean_2, max_mean_2, 
                                                                                                                     m_std, n_std, n_trials=n_trials,
                                                                                                                     file_for_data = file_for_data_2)
                                                                          
    else:
        
        with open(file_for_data_1,'rb') as f:
            [n_trials, _, _, stimuli_1, _, _, _, _, a1, _, weighted_output_1, trial_means_1] = pickle.load(f)
            
        with open(file_for_data_2,'rb') as f:
            [n_trials, _, _, stimuli_2, _, _, _, _, a2, _, weighted_output_2, trial_means_2] = pickle.load(f)
            
    
    ### plot data 
    weighted_output = np.vstack((weighted_output_2, weighted_output_1))
    stimuli = np.vstack((stimuli_2, stimuli_1))
    
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30)) 


# %% Slope as a function of variances

run_cell = False
plot_only = True

if run_cell:
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filenames for data
    file_for_data_1 = '../results/data/behavior/data_contraction_bias_increasing_input_sd.pickle'
    file_for_data_2 = '../results/data/behavior/data_contraction_bias_increasing_pred_sd.pickle'
    
    m_std, min_mean = dtype(0), dtype(15)
    
    ### get data
    if not plot_only:
        
        ### define statistics
        max_mean  = dtype(25)
        input_stds = np.linspace(0,8,5, dtype=dtype) 
    
        ### run for different input sd
        [_, fitted_slopes_1] = simulate_slope_vs_variances(mfn_flag, min_mean, max_mean, m_std, input_stds, n_trials=n_trials, file_for_data = file_for_data_1)
        
        ### define statistics
        n_std = dtype(5)
        max_mean_arr = np.linspace(20,48,5)
        
        ## run for different trial mean variability
        [_, fitted_slopes_2] = simulate_slope_vs_variances(mfn_flag, min_mean, max_mean_arr, m_std, n_std, n_trials=n_trials, file_for_data = file_for_data_2)
                                                                          
    else:
        
        with open(file_for_data_1,'rb') as f:
            [input_stds, fitted_slopes_1] = pickle.load(f)
            
        with open(file_for_data_2,'rb') as f:
            [max_mean_arr, fitted_slopes_2] = pickle.load(f)
            
    
    ### plot data 
    label_text = ['Stimulus', 'Trial']
    plot_slope_variability(input_stds, (max_mean_arr - min_mean)/np.sqrt(12), fitted_slopes_1, fitted_slopes_2, label_text)
    

# %% Increasing input SD decreases slope - examples

run_cell = False
plot_only = True

if run_cell:
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### define statistics
    m_std, n_std_1, n_std_2 = dtype(0), dtype(1), dtype(7)
    min_mean, max_mean = dtype(15), dtype(25)

    ### filenames for data
    file_for_data_1 = '../results/data/behavior/data_contraction_bias_trial_input_sd_' + str(n_std_1) + '_mean_range_' + str(max_mean - min_mean) + '.pickle'
    file_for_data_2 = '../results/data/behavior/data_contraction_bias_trial_input_sd_' + str(n_std_2) + '_mean_range_' + str(max_mean - min_mean) + '.pickle'
    
    ### get data
    if not plot_only:
        
        ## run for smaller range
        [n_trials, _, _, stimuli_1, _, _, _, _, _, _, weighted_output_1, trial_means_1] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                                                     m_std, n_std_1, n_trials=n_trials,
                                                                                                                     file_for_data = file_for_data_1)
        
        ## run for larger range 
        [n_trials, _, _, stimuli_2, _, _, _, _, _, _, weighted_output_2, trial_means_2] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                                                     m_std, n_std_2, n_trials=n_trials,
                                                                                                                     file_for_data = file_for_data_2)
                                                                          
    else:
        
        with open(file_for_data_1,'rb') as f:
            [n_trials, _, _, stimuli_1, _, _, _, _, a1, _, weighted_output_1, trial_means_1] = pickle.load(f)
            
        with open(file_for_data_2,'rb') as f:
            [n_trials, _, _, stimuli_2, _, _, _, _, a2, _, weighted_output_2, trial_means_2] = pickle.load(f)
            
    
    ### plot data 
    weighted_output = np.vstack((weighted_output_2, weighted_output_1))
    stimuli = np.vstack((stimuli_2, stimuli_1))
    
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30), ms=2)  
    
    
# %% Slope as a function of trial duration

run_cell = False
plot_only = True

if run_cell:
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11

    ### trial durations to be tested
    trial_durations = np.linspace(5000,10000,6, dtype=np.int32())

    ### filenames for data
    file_for_data_1 = '../results/data/behavior/data_contraction_bias_trial_duration_input_sd_zero.pickle'
    file_for_data_2 = '../results/data/behavior/data_contraction_bias_trial_duration_pred_sd_zero.pickle'
    
    ### get data
    if not plot_only:
        
        ### define statistics
        m_std, n_std = dtype(0), dtype(0)
        min_mean, max_mean = dtype(15), dtype(25)
        
        ### run networks with different trial durations
        [_, fitted_slopes_1] = simulate_slope_vs_trial_duration(mfn_flag, min_mean, max_mean, m_std, n_std, trial_durations, n_trials=n_trials, file_for_data = file_for_data_1)  
        
        ### define statistics
        m_std, n_std = dtype(0), dtype(5)
        min_mean, max_mean = dtype(15), dtype(15)
        
        ### run networks with different trial durations
        [_, fitted_slopes_2] = simulate_slope_vs_trial_duration(mfn_flag, min_mean, max_mean, m_std, n_std, trial_durations, n_trials=n_trials, file_for_data = file_for_data_2)  
        
    else:
        
        with open(file_for_data_1,'rb') as f:
            [trial_durations, fitted_slopes_1] = pickle.load(f)
            
        with open(file_for_data_2,'rb') as f:
            [trial_durations, fitted_slopes_2] = pickle.load(f)
        
        
    ### plot data
    label_text = ['Stimulus', 'Trial']
    plot_slope_trail_duration(trial_durations, fitted_slopes_1, fitted_slopes_2, label_text)
    

# %% Even without trial variability, contraction bias occurs - examples

# slope rather indepndent of input variance
# mechanism: weighting of S and P, slope a function of trial duration

run_cell = False
plot_only = True

if run_cell:
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### define statistics
    m_std, n_std_1, n_std_2 = dtype(0), dtype(2), dtype(5)
    min_mean, max_mean = dtype(15), dtype(15)

    ### filenames for data
    file_for_data_1 = '../results/data/behavior/data_contraction_bias_trial_input_sd_' + str(n_std_1) + '.pickle'
    file_for_data_2 = '../results/data/behavior/data_contraction_bias_trial_input_sd_' + str(n_std_2) + '.pickle'
    
    ### get data
    if not plot_only:
        
        ## run for smaller range
        [n_trials, _, _, stimuli_1, _, _, _, _, _, _, weighted_output_1, trial_means_1] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                                                     m_std, n_std_1, n_trials=n_trials,
                                                                                                                     file_for_data = file_for_data_1)
        
        ## run for larger range 
        [n_trials, _, _, stimuli_2, _, _, _, _, _, _, weighted_output_2, trial_means_2] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                                                     m_std, n_std_2, n_trials=n_trials,
                                                                                                                     file_for_data = file_for_data_2)
                                                                          
    else:
        
        with open(file_for_data_1,'rb') as f:
            [n_trials, _, _, stimuli_1, _, _, _, _, a1, _, weighted_output_1, trial_means_1] = pickle.load(f)
            
        with open(file_for_data_2,'rb') as f:
            [n_trials, _, _, stimuli_2, _, _, _, _, a2, _, weighted_output_2, trial_means_2] = pickle.load(f)
            
    
    ### plot data 
    weighted_output = np.vstack((weighted_output_2, weighted_output_1))
    stimuli = np.vstack((stimuli_2, stimuli_1))
    
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30), ms=2)                                                                          


# %% Even without input SD, contraction bias occurs - examples

# slope independent of trial variance
# mechanism: transient effects (steady state not reached ..., so basically history effect, it depends on previous stimulus)

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
            [n_trials, _, _, stimuli_1, _, _, _, _, a1, _, weighted_output_1, trial_means_1] = pickle.load(f)
            
        with open(file_for_data_2,'rb') as f:
            [n_trials, _, _, stimuli_2, _, _, _, _, a2, _, weighted_output_2, trial_means_2] = pickle.load(f)
            
    
    ### plot data 
    weighted_output = np.vstack((weighted_output_2, weighted_output_1))
    stimuli = np.vstack((stimuli_2, stimuli_1))
    
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30), ms=2)                                                                          
