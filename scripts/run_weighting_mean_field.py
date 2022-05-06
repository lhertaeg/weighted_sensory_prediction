#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 08:01:46 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
#import pickle

from src.mean_field_model import default_para, stimuli_moments_from_uniform, run_mean_field_model
from src.plot_toy_model import plot_limit_case

# from src.mean_field_model import stimuli_moments_from_uniform, run_toy_model, default_para, alpha_parameter_exploration
# from src.mean_field_model import random_uniform_from_moments, random_lognormal_from_moments, random_gamma_from_moments
# from src.mean_field_model import stimuli_from_mean_and_std_arrays
# from src.plot_mean_field_model import plot_limit_case, plot_alpha_para_exploration, plot_alpha_para_exploration_ratios


import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Toy model - limit cases

# Note: To make sure that PE activity reaches "quasi" steady state, I showed each stimulus  "num_repeats_per_value" times

flag = 1
flg = 1

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