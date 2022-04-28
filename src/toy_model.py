#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:51:18 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np

# %%

def alpha_parameter_exploration(stimuli, stimulus_duration, last_n, para_tested_first, para_tested_second, 
                                tc_var_per_stim, eta_prediction, eta_mean_prediction, tc_var_pred, flg_para):
    
    ### number of parameters and stimuli tested
    num_para_first = len(para_tested_first)
    num_para_second = len(para_tested_second)
    n_stimuli = np.int32(len(stimuli)/stimulus_duration)
    
    ### initialise
    fraction_sensory_median = np.zeros((num_para_first, num_para_second))
    
    ### run networks over all parameter to be tested
    if ((flg_para==12) or (flg_para==124)):
        
        for row, para_first in enumerate(para_tested_first):
            for col, para_second in enumerate(para_tested_second):
                
                _, _, _, _, alpha, _, _ = run_toy_model(para_first, para_second, eta_mean_prediction, tc_var_pred, stimuli)
                fraction_sensory_median[row, col] = np.median(alpha[(n_stimuli - last_n) * stimulus_duration:])
    
    elif flg_para==13:
        
        for row, para_first in enumerate(para_tested_first):
            for col, para_second in enumerate(para_tested_second):
                
                _, _, _, _, alpha, _, _ = run_toy_model(para_first, eta_prediction, para_second, tc_var_pred, stimuli)
                fraction_sensory_median[row, col] = np.median(alpha[(n_stimuli - last_n) * stimulus_duration:])
                
    elif flg_para==14:
        
        for row, para_first in enumerate(para_tested_first):
            for col, para_second in enumerate(para_tested_second):
                
                _, _, _, _, alpha, _, _ = run_toy_model(para_first, eta_prediction, eta_mean_prediction, para_second, stimuli)
                fraction_sensory_median[row, col] = np.median(alpha[(n_stimuli - last_n) * stimulus_duration:])
                
    elif flg_para==23:
        
        for row, para_first in enumerate(para_tested_first):
            for col, para_second in enumerate(para_tested_second):
                
                _, _, _, _, alpha, _, _ = run_toy_model(tc_var_per_stim, para_first, para_second, tc_var_pred, stimuli)
                fraction_sensory_median[row, col] = np.median(alpha[(n_stimuli - last_n) * stimulus_duration:])
                
    elif flg_para==24:
        
        for row, para_first in enumerate(para_tested_first):
            for col, para_second in enumerate(para_tested_second):
                
                _, _, _, _, alpha, _, _ = run_toy_model(tc_var_per_stim, para_first, eta_mean_prediction, para_second, stimuli)
                fraction_sensory_median[row, col] = np.median(alpha[(n_stimuli - last_n) * stimulus_duration:])
                
    elif flg_para==42:
        
        for row, para_first in enumerate(para_tested_first):
            for col, para_second in enumerate(para_tested_second):
                
                _, _, _, _, alpha, _, _ = run_toy_model(tc_var_per_stim, para_second, eta_mean_prediction, para_first, stimuli)
                fraction_sensory_median[row, col] = np.median(alpha[(n_stimuli - last_n) * stimulus_duration:])
                
    elif flg_para==34:
        
        for row, para_first in enumerate(para_tested_first):
            for col, para_second in enumerate(para_tested_second):
                
                _, _, _, _, alpha, _, _ = run_toy_model(tc_var_per_stim, eta_prediction, para_first, para_second, stimuli)
                fraction_sensory_median[row, col] = np.median(alpha[(n_stimuli - last_n) * stimulus_duration:])
          
    return fraction_sensory_median
    

def default_para():
    
    tc_var_per_stim = 20
    eta_prediction = 5e-3
    eta_mean_prediction = 2e-4
    tc_var_pred = 1
    
    return tc_var_per_stim, eta_prediction, eta_mean_prediction, tc_var_pred


def random_gamma_from_moments(mean, sd, n_stimuli):
    
    if ((sd!=0) & (mean!=0)):
        shape = mean**2 / sd**2
        scale = sd**2  / mean 
        rnd = np.random.gamma(shape, scale, size=n_stimuli)
    else:
        rnd = np.zeros(n_stimuli)
        
    return rnd


def random_lognormal_from_moments(mean, sd, n_stimuli):
    
    if ((sd!=0) & (mean!=0)):
        a = np.log(mean**2/np.sqrt(mean**2 + sd**2))
        b = np.sqrt(np.log(sd**2 / mean**2 + 1))
        rnd = np.random.lognormal(a, b, size=n_stimuli)
    else:
        rnd = np.zeros(n_stimuli)
        
    return rnd


def random_uniform_from_moments(mean, sd, n_stimuli):
    
    b = np.sqrt(12) * sd / 2 + mean
    a = 2 * mean -b
    rnd = np.random.uniform(a, b, size=n_stimuli)
        
    return rnd


def stimuli_from_mean_and_std_arrays(mean_stimuli, sd_stimuli, stimulus_duration, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    n_stimuli = len(mean_stimuli)
    stimuli = np.array([])
    
    for id_stim in range(n_stimuli):
        
        inputs_per_stimulus = np.random.normal(mean_stimuli[id_stim], sd_stimuli[id_stim], size=stimulus_duration)
        stimuli = np.concatenate((stimuli, inputs_per_stimulus))
        
    return stimuli


def stimuli_moments_from_uniform(n_stimuli, stimulus_duration, min_mean, max_mean, min_std, max_std, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    mean_stimuli = np.random.uniform(min_mean, max_mean, size=n_stimuli)
    sd_stimuli = np.random.uniform(min_std, max_std, size=n_stimuli)
    stimuli = np.array([])
    
    for id_stim in range(n_stimuli):
        
        inputs_per_stimulus = np.random.normal(mean_stimuli[id_stim], sd_stimuli[id_stim], size=stimulus_duration)
        stimuli = np.concatenate((stimuli, inputs_per_stimulus))
        
    return stimuli


def run_toy_model(tc_var_per_stim, eta_prediction, eta_mean_prediction, tc_var_pred, stimuli):
    
    # initialise
    pred = 0
    var_pred = 0
    var_per_stim = 0
    mean_prediction = 0
    
    prediction = np.zeros_like(stimuli)
    variance_per_stimulus = np.zeros_like(stimuli)
    mean_of_prediction = np.zeros_like(stimuli)
    variance_prediction = np.zeros_like(stimuli)
    
    for id_stim, stim in enumerate(stimuli):
        
        # compute prediction ("running mean over stimuli")
        pPE_sensory = (np.maximum(stim - pred,0))**2   
        nPE_sensory = (np.maximum(pred - stim,0))**2
        pred += eta_prediction * (pPE_sensory - nPE_sensory)
        prediction[id_stim] = pred
        
        # compute variance of sensory input
        var_per_stim = (1-1/tc_var_per_stim) * var_per_stim + (nPE_sensory + pPE_sensory)/tc_var_per_stim
        variance_per_stimulus[id_stim] = var_per_stim
        
        # compute variance of prediction
        pPE_prediction = (np.maximum(pred - mean_prediction,0))**2
        nPE_prediction = (np.maximum(mean_prediction - pred,0))**2
        mean_prediction += eta_mean_prediction * (pPE_prediction - nPE_prediction)
        mean_of_prediction[id_stim] = mean_prediction
        
        var_pred = (1-1/tc_var_pred) * var_pred + (nPE_prediction + pPE_prediction)/tc_var_pred
        variance_prediction[id_stim] = var_pred
        
    # compute weighted output
    alpha = (1/variance_per_stimulus) / ((1/variance_per_stimulus) + (1/variance_prediction))
    beta = (1/variance_prediction) / ((1/variance_per_stimulus) + (1/variance_prediction))
    weighted_output = alpha * stimuli + beta * prediction
        
    return prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, alpha, beta, weighted_output