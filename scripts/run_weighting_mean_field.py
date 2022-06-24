#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 08:01:46 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
#import pickle

from src.mean_field_model import default_para, stimuli_moments_from_uniform, run_mean_field_model, alpha_parameter_exploration
from src.plot_toy_model import plot_limit_case, plot_alpha_para_exploration_ratios, plot_fraction_sensory_comparsion, plot_alpha_para_exploration
from src.plot_toy_model import plot_manipulation_results
# from src.mean_field_model import stimuli_moments_from_uniform, run_toy_model, default_para, alpha_parameter_exploration
# from src.mean_field_model import random_uniform_from_moments, random_lognormal_from_moments, random_gamma_from_moments
# from src.mean_field_model import stimuli_from_mean_and_std_arrays
# from src.plot_mean_field_model import plot_limit_case, plot_alpha_para_exploration, plot_alpha_para_exploration_ratios


import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% erase after testing

import matplotlib.pyplot as plt

# %% Toy model - limit cases

# Note: To make sure that PE activity reaches "quasi" steady state, I showed each stimulus  "num_repeats_per_value" times

flag = 1
flg = 0

if flag==1:
    
    ### parameters
    filename = '../results/data/Prediction/Data_Optimal_Parameters_MFN_10.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)

    
    ### stimuli
    dt = 1
    n_stimuli = 60
    stimulus_duration = 5000
    num_values_per_stim = 50
    num_repeats_per_value = stimulus_duration/num_values_per_stim
    
    if flg==0:
        stimuli = stimuli_moments_from_uniform(n_stimuli, np.int32(num_values_per_stim/dt), 5, 5, 1, 5) # mean 5, SD between 1 and 5
    else:
        stimuli = stimuli_moments_from_uniform(n_stimuli, np.int32(num_values_per_stim/dt), 1, 5, 0, 0) # mean between 1 and 5, SD 0
  
    stimuli = np.repeat(stimuli, num_repeats_per_value)
    
    ### compute variances and predictions
    
    ## run model
    [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
      alpha, beta, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                          tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli)
    
    
    ### plot results
    plot_limit_case(n_stimuli, stimulus_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, 
                    variance_prediction, alpha, beta, weighted_output)


# %% Parameter exploration - time constants or weights

# Important: for proper figure, take the mean over several seeds!
# Important: To reduce the effect of prediction and mean of prediction not having reached their steady states yet, I set the initial values to the mean of the stimuli

flag = 0
flg = 1

para_exploration_name = 'tc' # tc or w

if flag==1:
    
    ### default parameters
    filename = '../results/data/Prediction/Data_Optimal_Parameters_MFN_10.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)
    
    
    ### stimuli
    dt = dtype(1)
    n_stimuli = np.int32(60)
    last_n = np.int32(30)
    stimulus_duration = np.int32(5000)# dtype(5000)
    num_values_per_stim = np.int32(50)
    num_repeats_per_value = np.int32(stimulus_duration/num_values_per_stim)
    
    if flg==0:
        stimuli = stimuli_moments_from_uniform(n_stimuli, np.int32(num_values_per_stim/dt), 5, 5, 1, 5) # mean 5, SD between 1 and 5
    else:
        stimuli = stimuli_moments_from_uniform(n_stimuli, np.int32(num_values_per_stim/dt), 1, 5, 0, 0) # mean between 1 and 5, SD 0
  
    stimuli = dtype(np.repeat(stimuli, num_repeats_per_value))
    
    ### parameters to test
    if para_exploration_name=='tc':
        para_tested_first = np.array([20,80,320,1280,5120], dtype=dtype) # np.arange(50,1500,200, dtype=dtype)
        para_tested_second = np.array([20,80,320,1280,5120], dtype=dtype) # np.arange(50,1500,200, dtype=dtype)
        ylabel = 'tau variance(P) / duration of stimulus'
        xlabel = 'tau variance(S) / duration of stimulus'
        para_first_denominator = stimulus_duration
        para_second_denominator = stimulus_duration
    elif para_exploration_name=='w':
        para_tested_first = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1], dtype=dtype) # np.arange(0.01,0.16,0.02, dtype=dtype)
        para_tested_second = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1], dtype=dtype) # np.arange(2e-4,4e-3,5e-4, dtype=dtype)
        ylabel = 'weight from PE to mean of P'
        xlabel = 'weight from PE to P'
        para_first_denominator = dtype(1)
        para_second_denominator = dtype(1)

        
    ### run parameter exploration
    fraction_sensory_mean, fraction_sensory_median, fraction_sensory_std = alpha_parameter_exploration(para_tested_first, para_tested_second, w_PE_to_P, w_P_to_PE, 
                                                                                                       w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                                                                       tc_var_pred, tau_pe, fixed_input, stimuli, stimulus_duration, 
                                                                                                       n_stimuli, last_n, para_exploration_name)
        
    ### plot
    plot_alpha_para_exploration_ratios(fraction_sensory_median, para_tested_first, para_tested_second, para_first_denominator, 
                                        para_second_denominator, 2, xlabel=xlabel, ylabel=ylabel, decimal=1e5, title='median')
    
    plot_alpha_para_exploration_ratios(fraction_sensory_mean, para_tested_first, para_tested_second, para_first_denominator, 
                                        para_second_denominator, 2, xlabel=xlabel, ylabel=ylabel, decimal=1e5, title='mean')
    
    plot_alpha_para_exploration_ratios(fraction_sensory_std, para_tested_first, para_tested_second, para_first_denominator, 
                                        para_second_denominator, 2, xlabel=xlabel, ylabel=ylabel, decimal=1e5, title='std', cmap='gray_r', vmax=0.5) # as fraction_sensory is between 0 and 1, std can only be between 0 and 0.5
    

# %% Test different std for mean and std of stimuli (exploration between limit cases)

# show examples from different corners
# maybe show this plot for several distribution types?

flag = 0

if flag==1:
    
    ### default parameters
    filename = '../results/data/Prediction/Data_Optimal_Parameters_MFN_10.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)
    
    ### stimulation & simulation parameters
    dt = dtype(1)
    n_stimuli = np.int32(60)
    last_n = np.int32(30)
    stimulus_duration = np.int32(5000)# dtype(5000)
    num_values_per_stim = np.int32(50)
    num_repeats_per_value = np.int32(stimulus_duration/num_values_per_stim)
    n_repeats = np.int32(3)
    
    ### means and std's to be tested
    mean_mean, min_std = dtype(10), dtype(0)
    std_mean_arr = np.linspace(0,5,5, dtype=dtype)
    std_std_arr = np.linspace(0,5,5, dtype=dtype)
    
    ### initialise
    fraction_sensory_median = np.zeros((len(std_std_arr),len(std_mean_arr), n_repeats), dtype=dtype)
    fraction_sensory_mean = np.zeros((len(std_std_arr),len(std_mean_arr), n_repeats), dtype=dtype)
    fraction_sensory_std = np.zeros((len(std_std_arr),len(std_mean_arr), n_repeats), dtype=dtype)
    
    for seed in range(n_repeats):

        for col, std_mean in enumerate(std_mean_arr):
            
            for row, std_std in enumerate(std_std_arr):
        
                ### define stimuli
                stimuli = stimuli_moments_from_uniform(n_stimuli, np.int32(num_values_per_stim/dt), dtype(mean_mean - np.sqrt(3)*std_mean), 
                                                       dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))
                
                stimuli = dtype(np.repeat(stimuli, num_repeats_per_value))
                
                ### run model
                [_, _, _, _, alpha, _, _] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                              tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli)
                
                ### median of fraction of sensory inputs over the last n stimuli
                fraction_sensory_median[row, col, seed] = np.median(alpha[(n_stimuli - last_n) * stimulus_duration:])
                fraction_sensory_mean[row, col, seed] = np.mean(alpha[(n_stimuli - last_n) * stimulus_duration:])
                fraction_sensory_std[row, col, seed] = np.std(alpha[(n_stimuli - last_n) * stimulus_duration:])
  
    fraction_sensory_median_averaged_over_seeds = np.mean(fraction_sensory_median,2)
    fraction_sensory_mean_averaged_over_seeds = np.mean(fraction_sensory_mean,2)
    fraction_sensory_std_averaged_over_seeds = np.mean(fraction_sensory_std,2)
    
    ### plot results
    plot_alpha_para_exploration(fraction_sensory_median_averaged_over_seeds, std_std_arr, std_mean_arr, 2, 
                                xlabel='variability across stimuli', ylabel='variability per stimulus', title='Median')
    
    plot_alpha_para_exploration(fraction_sensory_mean_averaged_over_seeds, std_std_arr, std_mean_arr, 2, 
                                xlabel='variability across stimuli', ylabel='variability per stimulus', title='Mean')
    
    plot_alpha_para_exploration(fraction_sensory_std_averaged_over_seeds, std_std_arr, std_mean_arr, 2, 
                                xlabel='variability across stimuli', ylabel='variability per stimulus', title='Std', cmap='gray_r', vmax=0.5)
    
    
# %% Impact of stimulus duration

# ##### Not done yet: you have to adapt the part about the stimuli to tailor it to te MFN
# ##### Also, think about the stim_durations and how many stimuli you should show to make it comparable between different stim_durations

# flag = 0
# flg = 1

# if flag==1:
    
#     ### default parameters
#     filename = '../results/data/Prediction/Data_Optimal_Parameters_MFN_10.pickle'
    
#     [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
#      tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)
    
#     ### stimuli durations to be tested and number of stimuli & repetitions
#     stim_durations = np.int32(np.array([25,100,500,5000]))
    
#     ### stimulation & simulation parameters
#     n_repeats = 5
    
#     dt = dtype(1)
#     n_stimuli = np.int32(60)
#     last_n = np.int32(30)
#     num_values_per_stim = np.int32(50)
#     num_repeats_per_value = np.int32(stimulus_duration/num_values_per_stim)
    
#     ### initialise array
#     fraction_sensory = np.zeros((len(stim_durations), n_stimuli, n_repeats))
    
#     ### test different stimuli durations
#     for seed in range(n_repeats):
#         for id_stim, stimulus_duration in enumerate(stim_durations):
            
#             if flg==0:
#                 stimuli = stimuli_moments_from_uniform(n_stimuli, np.int32(num_values_per_stim/dt), 5, 5, 1, 5) # mean 5, SD between 1 and 5
#             else:
#                 stimuli = stimuli_moments_from_uniform(n_stimuli, np.int32(num_values_per_stim/dt), 1, 5, 0, 0) # mean between 1 and 5, SD 0
          
#             stimuli = dtype(np.repeat(stimuli, num_repeats_per_value))
            
#             ## run model
#             [_, _, _, _, alpha, _, _] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
#                                                              tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli)
    
#             ## fraction of sensory input in weighted output stored in array
#             fraction_sensory[id_stim, :, seed] = np.mean(np.split(alpha,n_stimuli),1)
            
#     fraction_sensory_averaged_over_seeds = np.mean(fraction_sensory,2)
#     fraction_sensory_std_over_seeds = np.std(fraction_sensory,2)
    
#     ### plot results
#     plot_fraction_sensory_comparsion(fraction_sensory_averaged_over_seeds, fraction_sensory_std_over_seeds, n_repeats,
#                                      label_text=stim_durations)

# %% Activation/inactivation experiments 

# this might be also dependent on the specific distribution of inputs (S/P) onto the interneurons

# in the end, it would also be interesting to show MSE between the weighted output and the ground truth 
# (which would be the stimulus but without noise) ... do this later

flag = 0
flg = 0

if flag==1:
    
    ### parameters
    filename = '../results/data/Prediction/Data_Optimal_Parameters_MFN_10.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)

    ### stimuli
    dt = 1
    n_stimuli = 60
    last_n = np.int32(30)
    stimulus_duration = 5000
    num_values_per_stim = 50
    num_repeats_per_value = stimulus_duration/num_values_per_stim
    
    if flg==0:
        stimuli = stimuli_moments_from_uniform(n_stimuli, np.int32(num_values_per_stim/dt), 5, 5, 1, 5) # mean 5, SD between 1 and 5
    else:
        stimuli = stimuli_moments_from_uniform(n_stimuli, np.int32(num_values_per_stim/dt), 1, 5, 0, 0) # mean between 1 and 5, SD 0
  
    stimuli = np.repeat(stimuli, num_repeats_per_value)
    
    
    ### manipulation range
    manipulations = np.linspace(-5,5,7)
    
    ### initialise
    fraction_of_sensory_input_in_output = np.zeros((len(manipulations),8))
     
    ### compute variances and predictions
    for id_mod, manipulation_strength in enumerate(manipulations): 
    
        for id_cell in range(8): # nPE, pPE, nPE dend, pPE dend, PVv, PVm, SOM, VIP
        
            print(str(id_mod) + '/' + str(len(manipulations)) + ' and ' + str(id_cell) + '/' + str(8))
        
            modulation = np.zeros(8)                          
            modulation[id_cell] = manipulation_strength             
            fixed_input_plus_modulation = fixed_input + modulation
    
            ## run model
            [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
              alpha, beta, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                  tc_var_per_stim, tc_var_pred, tau_pe, fixed_input_plus_modulation, stimuli)
    
            
            ## "steady-state"                                                        
            fraction_of_sensory_input_in_output[id_mod, id_cell] = np.median(alpha[(n_stimuli - last_n) * stimulus_duration:])                                                    
                   

    ### plot results
    plot_manipulation_results(manipulations, fraction_of_sensory_input_in_output, flg)