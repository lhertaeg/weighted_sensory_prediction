#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:42:21 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
#import pickle

from src.mean_field_model import default_para, stimuli_moments_from_uniform, run_mean_field_model
from src.mean_field_model import alpha_parameter_exploration, run_mean_field_model_pred
from src.plot_toy_model import plot_limit_case, plot_alpha_para_exploration_ratios, plot_fraction_sensory_comparsion, plot_alpha_para_exploration
from src.plot_toy_model import plot_manipulation_results
from src.plot_results_mfn import plot_limit_case_new, plot_limit_case_pred

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Weighting of sensory input and predictions: transitions
# transition from reliable stimulus that does not change (mean=5, SD=0)

# Note: To make sure that PE activity reaches "quasi" steady state, I showed each stimulus  "num_repeats_per_value" times

# Stimuli:
    # there are n trials (n_trials), each consists of n stimuli (n_stimuli_per_trial), 
    # each stimulus is repeated (n_repeats_per_stim)
    # I assume a time step of 1 here
    # ... this mimicks a noisy stimulus per trail
    # ... this setup is only necessray for the weighting later (not here actually)


flag = 0
flg_trans = 4

if flag==1:
    
    ### parameters
    filename = '../results/data/Prediction/Data_Optimal_Parameters_MFN_10.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)

    
    ### stimuli
    dt = 1
    n_stimuli = 120
    stimulus_duration = 5000
    num_values_per_stim = 50
    num_repeats_per_value = stimulus_duration/num_values_per_stim
    
    stimuli = np.zeros(n_stimuli * num_values_per_stim)
    stimuli[:(n_stimuli//2 * num_values_per_stim)] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      5, 5, 0, 0) # mean 5, SD between 1 and 5
    if flg_trans==0:
        stimuli[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      5, 5, 1, 5) # mean between 1 and 5, SD 0
    elif flg_trans==1:
        stimuli[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      3, 7, 0, 0) # mean between 1 and 5, SD 0
    elif flg_trans==2:
        stimuli[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      3, 7, 1, 5) # mean between 1 and 5, SD 0
    elif flg_trans==3:
        stimuli[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                     6, 10, 0, 0) # mean between 1 and 5, SD 0
    elif flg_trans==4:
        stimuli[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      6, 10, 1, 5) # mean between 1 and 5, SD 0
    
    stimuli = np.repeat(stimuli, num_repeats_per_value)
    
    ### compute variances and predictions
    
    ## run model
    [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
      alpha, beta, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                          tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli)
    
    
    ### plot results
    plot_limit_case_new(n_stimuli, stimulus_duration, stimuli, prediction, mean_of_prediction, 
                        variance_per_stimulus, variance_prediction, alpha, beta, weighted_output)
    
    
# %% Weighting of sensory input and predictions: transitions
# transition from a noisy stimulus that does not change (mean=5, SD between 1 and 5)

# Note: To make sure that PE activity reaches "quasi" steady state, I showed each stimulus  "num_repeats_per_value" times

flag = 1
flg_trans = 4

if flag==1:
    
    ### parameters
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_10.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)

    
    ### stimuli
    dt = 1
    n_stimuli = 120
    stimulus_duration = 5000
    num_values_per_stim = 50
    num_repeats_per_value = stimulus_duration/num_values_per_stim
    
    stimuli = np.zeros(n_stimuli * num_values_per_stim)
    stimuli[:(n_stimuli//2 * num_values_per_stim)] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      5, 5, 1, 5) # mean 5, SD between 1 and 5
    if flg_trans==0:
        stimuli[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      5, 5, 0, 0) # mean between 1 and 5, SD 0
    elif flg_trans==1:
        stimuli[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      3, 7, 0, 0) # mean between 1 and 5, SD 0
    elif flg_trans==2:
        stimuli[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      3, 7, 1, 5) # mean between 1 and 5, SD 0
    elif flg_trans==3:
        stimuli[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                     6, 10, 0, 0) # mean between 1 and 5, SD 0
    elif flg_trans==4:
        stimuli[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      6, 10, 1, 5) # mean between 1 and 5, SD 0
    
    stimuli = np.repeat(stimuli, num_repeats_per_value)
    
    ### compute variances and predictions
    
    ## run model
    [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
      alpha, beta, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                          tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli)
    
    
    ### plot results
    plot_limit_case_new(n_stimuli, stimulus_duration, stimuli, prediction, mean_of_prediction, 
                        variance_per_stimulus, variance_prediction, alpha, beta, weighted_output)
     
    
# %% Weighting of fast and slow predictions
# transition from reliable stimulus that does not change (mean=5, SD=0)

# Note: To make sure that PE activity reaches "quasi" steady state, I showed each stimulus  "num_repeats_per_value" times

flag = 0
flg_trans = 4

if flag==1:
    
    ### parameters
    filename = '../results/data/Prediction/Data_Optimal_Parameters_MFN_10.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)

    
    ### stimuli
    dt = 1
    n_stimuli = 120
    stimulus_duration = 5000
    num_values_per_stim = 50
    num_repeats_per_value = stimulus_duration/num_values_per_stim
    
    trial_mean = np.zeros(n_stimuli * num_values_per_stim)
    trial_dev = np.zeros(n_stimuli * num_values_per_stim)

    trial_mean[:(n_stimuli//2 * num_values_per_stim)] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      5, 5, 0, 0) # mean 5, SD between 1 and 5
    trial_dev[:(n_stimuli//2 * num_values_per_stim)] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      0, 0, 0, 0) # mean 5, SD between 1 and 5
    if flg_trans==0:
        trial_mean[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      5, 5, 0, 0) # mean between 1 and 5, SD 0
        trial_dev[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      0, 0, 1, 5) # mean between 1 and 5, SD 0
    elif flg_trans==1:
        trial_mean[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      3, 7, 0, 0) # mean between 1 and 5, SD 0
        trial_dev[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      0, 0, 0, 0) # mean between 1 and 5, SD 0
    elif flg_trans==2:
        trial_mean[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      3, 7, 0, 0) # mean between 1 and 5, SD 0
        trial_dev[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      0, 0, 1, 5) # mean between 1 and 5, SD 0
    elif flg_trans==3:
        trial_mean[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                     6, 10, 0, 0) # mean between 1 and 5, SD 0
        trial_dev[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                     0, 0, 0, 0) # mean between 1 and 5, SD 0
    elif flg_trans==4:
        trial_mean[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      6, 10, 0, 0) # mean between 1 and 5, SD 0
        trial_dev[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      0, 0, 1, 5) # mean between 1 and 5, SD 0
    
    stimuli = np.repeat(trial_mean + trial_dev, num_repeats_per_value)
    trial_mean = np.repeat(trial_mean, num_repeats_per_value)
    
    ### compute variances and predictions
    
    ## run model
    [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
     alpha, beta, weighted_prediction] = run_mean_field_model_pred(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                                   v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                                   tc_var_pred, tau_pe, fixed_input, stimuli)
    
    
    ### plot results
    #from src.plot_results_mfn import plot_limit_case_new, plot_limit_case_pred
    plot_limit_case_pred(n_stimuli, stimulus_duration, stimuli, trial_mean, prediction, mean_of_prediction, 
                         variance_per_stimulus, variance_prediction, alpha, beta, weighted_prediction)
    

# %% Weighting of fast and slow predictions
# transition from a noisy stimulus that does not change (mean=5, SD between 1 and 5)

# Note: To make sure that PE activity reaches "quasi" steady state, I showed each stimulus  "num_repeats_per_value" times

flag = 1
flg_trans = 4

if flag==1:
    
    ### parameters
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_10.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)

    
    ### stimuli
    dt = 1
    n_stimuli = 120
    stimulus_duration = 5000
    num_values_per_stim = 50
    num_repeats_per_value = stimulus_duration/num_values_per_stim
    
    trial_mean = np.zeros(n_stimuli * num_values_per_stim)
    trial_dev = np.zeros(n_stimuli * num_values_per_stim)

    trial_mean[:(n_stimuli//2 * num_values_per_stim)] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      5, 5, 0, 0) # mean 5, SD between 1 and 5
    trial_dev[:(n_stimuli//2 * num_values_per_stim)] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      0, 0, 1, 5) # mean 5, SD between 1 and 5
    if flg_trans==0:
        trial_mean[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      5, 5, 0, 0) # mean between 1 and 5, SD 0
        trial_dev[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      0, 0, 0, 0) # mean between 1 and 5, SD 0
    elif flg_trans==1:
        trial_mean[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      3, 7, 0, 0) # mean between 1 and 5, SD 0
        trial_dev[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      0, 0, 0, 0) # mean between 1 and 5, SD 0
    elif flg_trans==2:
        trial_mean[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      3, 7, 0, 0) # mean between 1 and 5, SD 0
        trial_dev[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      0, 0, 1, 5) # mean between 1 and 5, SD 0
    elif flg_trans==3:
        trial_mean[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                     6, 10, 0, 0) # mean between 1 and 5, SD 0
        trial_dev[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                     0, 0, 0, 0) # mean between 1 and 5, SD 0
    elif flg_trans==4:
        trial_mean[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      6, 10, 0, 0) # mean between 1 and 5, SD 0
        trial_dev[(n_stimuli//2 * num_values_per_stim):] = stimuli_moments_from_uniform(n_stimuli//2, np.int32(num_values_per_stim/dt), 
                                                                                      0, 0, 1, 5) # mean between 1 and 5, SD 0
    
    stimuli = np.repeat(trial_mean + trial_dev, num_repeats_per_value)
    trial_mean = np.repeat(trial_mean, num_repeats_per_value)
    
    ### compute variances and predictions
    
    ## run model
    [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
     alpha, beta, weighted_prediction] = run_mean_field_model_pred(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                                   v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                                   tc_var_pred, tau_pe, fixed_input, stimuli)
    
    
    ### plot results
    #from src.plot_results_mfn import plot_limit_case_new, plot_limit_case_pred
    plot_limit_case_pred(n_stimuli, stimulus_duration, stimuli, trial_mean, prediction, mean_of_prediction, 
                         variance_per_stimulus, variance_prediction, alpha, beta, weighted_prediction)