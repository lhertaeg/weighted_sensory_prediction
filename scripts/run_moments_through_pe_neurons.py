#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:38:29 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
#import pickle

from src.mean_field_model import default_para, stimuli_moments_from_uniform, run_mean_field_model, alpha_parameter_exploration
from src.mean_field_model import run_mean_field_model_one_column

from src.toy_model import random_uniform_from_moments, random_lognormal_from_moments, random_gamma_from_moments
from src.plot_toy_model import plot_limit_case, plot_alpha_para_exploration_ratios, plot_fraction_sensory_comparsion, plot_alpha_para_exploration
from src.plot_toy_model import plot_manipulation_results

from src.plot_results_mfn import plot_prediction, plot_variance, plot_mse, plot_manipulations

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Example: PE neurons can estimate/establish the mean of a random variable drawn from a unimodal symmetrical distribution

flag = 0

if flag==1:
    
    ### parameters
    filename = '../results/data/Prediction/Data_Optimal_Parameters_MFN_10.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)

    w_PE_to_P /= 50
    tc_var_per_stim *= 100
    
    ### stimuli
    dt = 1
    n_stimuli = 100
    stimulus_duration = 5000
    num_values_per_stim = 50
    num_repeats_per_value = stimulus_duration/num_values_per_stim
    
    # stimuli between 1 and 5, SD between 0.5 and 1
    stimuli = stimuli_moments_from_uniform(n_stimuli, np.int32(num_values_per_stim/dt), 1, 5, 0.2, 0.5)
    stimuli = np.repeat(stimuli, num_repeats_per_value)
    
    ### compute variances and predictions
    
    ## run model
    prediction, variance_per_stimulus = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                        tc_var_per_stim, tau_pe, fixed_input, 
                                                                        stimuli)
    
    ### plot prediction and variance (in comparison to the data) 
    plot_prediction(n_stimuli, stimuli, stimulus_duration, prediction)  
    plot_variance(n_stimuli, stimuli, stimulus_duration, variance_per_stimulus) 
    

# %% MSE of estimated mean (prediction) and variance for different parameterisation of uniform distributions

flag = 0

if flag==1:
    
    ### parameters
    filename = '../results/data/Prediction/Data_Optimal_Parameters_MFN_10.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)

    w_PE_to_P /= 50
    tc_var_per_stim *= 100
    
    ### stimuli
    dt = 1
    n_stimuli = 60
    stimulus_duration = 5000
    num_values_per_stim = 50
    num_repeats_per_value = stimulus_duration/num_values_per_stim
    
    mean_tested = np.array([3,5,7])
    variance_tested = np.array([1,2,3])
    
    mse_prediction = np.zeros((len(mean_tested), len(variance_tested), n_stimuli * stimulus_duration))
    mse_variance = np.zeros((len(mean_tested), len(variance_tested), n_stimuli * stimulus_duration))
    trials = np.arange(n_stimuli * stimulus_duration)/stimulus_duration
    
    for i, mean_dist in enumerate(mean_tested):
        for j, var_dist in enumerate(variance_tested):
            
            print(str(i+1) + '/' + str(len(mean_tested)) + ' and ' + str(j+1) + '/' + str(len(variance_tested)))
            
            min_uniform = mean_dist - np.sqrt(3 * var_dist)
            max_uniform = mean_dist + np.sqrt(3 * var_dist)
            
            # stimuli between 1 and 5, SD between 0.5 and 1
            stimuli = stimuli_moments_from_uniform(n_stimuli, np.int32(num_values_per_stim/dt), 
                                                   min_uniform, max_uniform, 0.2, 0.5)
            stimuli = np.repeat(stimuli, num_repeats_per_value)
            
            
            ### compute variances and predictions
            
            ## run model
            prediction, variance_per_stimulus = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                                tc_var_per_stim, tau_pe, fixed_input, 
                                                                                stimuli)
            
            ## compute mean squared error
            running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
            mse_prediction[i, j, :] = (running_average - prediction)**2
            
            mean_running = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
            momentary_variance = (stimuli - mean_running)**2
            variance_running = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1)
            mse_variance[i, j, :] = (variance_running - variance_per_stimulus)**2

    ### plot mean squared errors (in comparison to the data)       
    plot_mse(trials, mse_prediction, 'MSE prediction')
    plot_mse(trials, mse_variance, 'MSE variance')                                          
     

# %% MSE of estimated mean (prediction) and variance for different distributions

flag = 0

if flag==1:
    
    ### parameters
    filename = '../results/data/Prediction/Data_Optimal_Parameters_MFN_10.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)

    w_PE_to_P /= 50
    tc_var_per_stim *= 100
    
    ### stimuli
    dt = 1
    n_stimuli = 60
    stimulus_duration = 5000
    num_values_per_stim = 50
    num_repeats_per_value = stimulus_duration/num_values_per_stim
    
    mse_prediction = np.zeros((4, n_stimuli * stimulus_duration))
    mse_variance = np.zeros((4, n_stimuli * stimulus_duration))
    trials = np.arange(n_stimuli * stimulus_duration)/stimulus_duration
    
    mean_stimuli = 5
    std_stimuli = 2
    
    for i in range(4):
        
        print(str(i+1) + '/4')    
        
        ### create stimuli
        if i==0:
            stimuli = np.random.normal(mean_stimuli, std_stimuli, size=(n_stimuli * num_values_per_stim))
        elif i==1:
            stimuli = random_uniform_from_moments(mean_stimuli, std_stimuli, (n_stimuli * num_values_per_stim))
        elif i==2:
            stimuli = random_lognormal_from_moments(mean_stimuli, std_stimuli, (n_stimuli * num_values_per_stim))
        elif i==3:
            stimuli = random_gamma_from_moments(mean_stimuli, std_stimuli, (n_stimuli * num_values_per_stim))

        stimuli = np.repeat(stimuli, num_repeats_per_value)
        
        ### compute variances and predictions
        
        ## run model
        prediction, variance_per_stimulus = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                            tc_var_per_stim, tau_pe, fixed_input, 
                                                                            stimuli)
        
        ## compute mean squared error
        running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
        mse_prediction[i, :] = (running_average - prediction)**2
        
        mean_running = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
        momentary_variance = (stimuli - mean_running)**2
        variance_running = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1)
        mse_variance[i, :] = (variance_running - variance_per_stimulus)**2

    ### plot mean squared errors (in comparison to the data)   
    legend_labels = ['normal','uniform','lognormal','gamma']
    plot_mse(trials, mse_prediction, 'MSE prediction', legend_labels=legend_labels)
    plot_mse(trials, mse_variance, 'MSE variance', legend_labels=legend_labels)                                         
   
    
# %% Network manipulations

flag = 1

if flag==1:
    
    ### parameters
    filename = '../results/data/Prediction/Data_Optimal_Parameters_MFN_10.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)

    w_PE_to_P /= 50
    tc_var_per_stim *= 100
    
    ### stimuli
    dt = 1
    n_stimuli = 100
    num_trials_above = 80   # to compute the steady state mean
    stimulus_duration = 5000
    num_values_per_stim = 50
    num_repeats_per_value = stimulus_duration/num_values_per_stim
    
    ## stimuli between 1 and 5, SD between 0.5 and 1
    stimuli = stimuli_moments_from_uniform(n_stimuli, np.int32(num_values_per_stim/dt), 1, 5, 0.2, 0.5)
    stimuli = np.repeat(stimuli, num_repeats_per_value)
    
    trials = np.arange(n_stimuli * stimulus_duration)/stimulus_duration
    manipulations = [-1, 1]
    
    ### initialise
    dev_prediction_steady = np.zeros((len(manipulations), 8))
    dev_variance_steady = np.zeros((len(manipulations), 8))
    
    ### compute variances and predictions
    ## run model
    for id_mod, manipulation_strength in enumerate(manipulations): 
    
        for id_cell in range(8): # nPE, pPE, nPE dend, pPE dend, PVv, PVm, SOM, VIP
            
            print(id_cell)
        
            modulation = np.zeros(8)                          
            modulation[id_cell] = manipulation_strength          
            fixed_input_plus_modulation = fixed_input + modulation
            
            prediction, variance_per_stimulus = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                                tc_var_per_stim, tau_pe, 
                                                                                fixed_input_plus_modulation, 
                                                                                stimuli)
            
            ## compute mean squared error
            running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
            dev_prediction = (prediction - running_average) / running_average
            
            mean_running = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
            momentary_variance = (stimuli - mean_running)**2
            variance_running = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1)
            dev_variance = (variance_per_stimulus - variance_running) / variance_running
            
            dev_prediction_steady[id_mod, id_cell] = np.mean(dev_prediction[trials>=num_trials_above]) * 100
            dev_variance_steady[id_mod, id_cell] = np.mean(dev_variance[trials>=num_trials_above]) * 100
    
    ### plot results
    xticklabels = ['nPE', 'pPE', 'nPE dend', 'pPE dend', 'PVv', 'PVm', 'SOM', 'VIP']
    
    plot_manipulations(dev_prediction_steady, xticklabels, 'Prediction')
    plot_manipulations(dev_variance_steady, xticklabels, 'Variance')
    
