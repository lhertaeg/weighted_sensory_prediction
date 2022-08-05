#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:38:29 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from src.mean_field_model import default_para, stimuli_moments_from_uniform, run_mean_field_model, alpha_parameter_exploration
from src.mean_field_model import run_mean_field_model_one_column

from src.toy_model import random_uniform_from_moments, random_lognormal_from_moments, random_gamma_from_moments, random_binary_from_moments
from src.plot_toy_model import plot_limit_case, plot_alpha_para_exploration_ratios, plot_fraction_sensory_comparsion, plot_alpha_para_exploration
from src.plot_toy_model import plot_manipulation_results

from src.plot_results_mfn import plot_prediction, plot_variance, plot_mse, plot_manipulations, plot_mse_heatmap

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Some tests to check the PE neuron activity of my mean-field networks from the last paper

flag = 0

if flag==1:
    
    ### define/load parameters
    input_flg = '01' # 10, 01, 11
    
    VS, VV = int(input_flg[0]), int(input_flg[1])
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename, VS=VS, VV=VV)
    
    w_PE_to_P = np.zeros((1,8)) # prediction constant as stimulus
    
    ### define stimuli
    test_stimuli = np.linspace(1,10,20)
    trial_duration = 500
    
    nPE = np.zeros(len(test_stimuli))
    pPE = np.zeros(len(test_stimuli))              
    
    for id_stim, stimulus in enumerate(test_stimuli):
    
        stimuli = np.repeat(stimulus, trial_duration)
        
        ## run model
        _, _, PE_activity = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                            tc_var_per_stim, tau_pe, fixed_input, 
                                                            stimuli, VS=VS, VV=VV, 
                                                            set_initial_prediction_to_value = np.mean(test_stimuli))
        nPE[id_stim] = PE_activity[-1,0]
        pPE[id_stim] = PE_activity[-1,1]
    
    ### plot
    plt.figure()
    plt.plot(test_stimuli-np.mean(test_stimuli), nPE, 'b')
    plt.plot(test_stimuli-np.mean(test_stimuli), np.maximum(np.mean(test_stimuli) - test_stimuli,0), ':b')
    plt.plot(test_stimuli-np.mean(test_stimuli), pPE, 'r')
    plt.plot(test_stimuli-np.mean(test_stimuli), np.maximum(test_stimuli - np.mean(test_stimuli),0), ':r')


# 10:
    # BL activity = 0 & a~b=1 (almost) in steady state
    # this is however not true when steady state is not met

# 01:
    # BL activity = 0 & a~b<1 in steady state
    # this is however not true or deviations more severe when steady state is not met
    
# 11:
    # BL activity = 0 & a~b<1 in steady state
    # this is however not true or deviations more severe when steady state is not met


# %% Example: PE neurons can estimate/establish the mean and variance of a normal random variable 
# mean and varaince are drawn from a unimodal distribution

flag = 0
flg_plot_only = 0

if flag==1:
    
    ### define/load parameters
    input_flg = '11' # 10, 01, 11
    
    VS, VV = int(input_flg[0]), int(input_flg[1])
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/moments/Data_example_MFN_' + input_flg + '.pickle'
    
    if flg_plot_only==0:
        
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename, VS=VS, VV=VV)
        
        if input_flg=='10':
            nPE_scale = 1.015
            pPE_scale = 1.023
        elif input_flg=='01':
            nPE_scale = 1.7 # 1.72
            pPE_scale = 1.7 # 1.68
        elif input_flg=='11':
            nPE_scale = 2.49
            pPE_scale = 2.53
            
        w_PE_to_P[0,0] *= nPE_scale
        w_PE_to_P[0,1] *= pPE_scale
        w_PE_to_V = [nPE_scale, pPE_scale]
        
        ### define stimuli
        n_trials = 500
        trial_duration = 500
        n_stimuli_per_trial = 1
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        
        mean_stimuli, std_stimuli = 5, 2
        stimuli = random_uniform_from_moments(mean_stimuli, std_stimuli, (n_trials * n_stimuli_per_trial))
        stimuli = np.repeat(stimuli, n_repeats_per_stim)
        
        ### compute variances and predictions
        
        ## run model
        prediction, variance, PE_activity = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                            tc_var_per_stim, tau_pe, fixed_input, 
                                                                            stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
    
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([n_trials, trial_duration, stimuli, prediction, variance],f)                                      
     
    else:
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [n_trials, trial_duration, stimuli, prediction, variance] = pickle.load(f)  
        
        
    ### plot prediction and variance (in comparison to the data) 
    plot_prediction(n_trials, stimuli, trial_duration, prediction)  
    plot_variance(n_trials, stimuli, trial_duration, variance) 
    

# %% MSE of estimated mean (prediction) and variance for different parameterisation of uniform distributions

### as you average for computing the steady state, there is no need to run it over several seeds
# but make sure that the range you are averaging over is fair

flag = 0
flg_plot_only = 1

if flag==1:
    
    ### load and define parameters
    input_flg = '11' # 10, 01, 11
    
    VS, VV = int(input_flg[0]), int(input_flg[1])
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/moments/data_test_input_parameterisations_' + input_flg + '.pickle'
    #file_data4plot = '../results/data/moments/data_test_input_parameterisations'
    
    if flg_plot_only==0:
    
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename, VS=VS, VV=VV)
        
        if input_flg=='10':
            nPE_scale = 1.015
            pPE_scale = 1.023
        elif input_flg=='01':
            nPE_scale = 1.7 # 1.72
            pPE_scale = 1.7 # 1.68
        elif input_flg=='11':
            nPE_scale = 2.49
            pPE_scale = 2.53
            
        w_PE_to_P[0,0] *= nPE_scale
        w_PE_to_P[0,1] *= pPE_scale
        w_PE_to_V = [nPE_scale, pPE_scale]
        
        ### define stimulus parameters
        n_trials = 1000
        trial_duration = 500
        n_stimuli_per_trial = 1
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        
        ### define mu and variance of uniform distribution
        mean_tested = np.linspace(3,7,9) # np.array([3,5])
        variance_tested = np.linspace(1,3,6) # np.array([1,2])
        
        ### initialisation
        mse_prediction = np.zeros((len(mean_tested), len(variance_tested), n_trials * trial_duration))
        mse_variance = np.zeros((len(mean_tested), len(variance_tested), n_trials * trial_duration))
        
        ### test different parameterizations
        for i, mean_dist in enumerate(mean_tested):
            for j, var_dist in enumerate(variance_tested):
                
                print(str(i+1) + '/' + str(len(mean_tested)) + ' and ' + str(j+1) + '/' + str(len(variance_tested)))
                
                ### define stimuli
                stimuli = random_uniform_from_moments(mean_dist, np.sqrt(var_dist), (n_trials * n_stimuli_per_trial))
                stimuli = np.repeat(stimuli, n_repeats_per_stim)
                
                ### compute variances and predictions
                
                ## run model
                prediction, variance, PE_activity = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                            tc_var_per_stim, tau_pe, fixed_input, 
                                                                            stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
                
                ## compute mean squared error
                running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
                mse_prediction[i, j, :] = (running_average - prediction)**2
                
                mean_running = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
                momentary_variance = (stimuli - mean_running)**2
                variance_running = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1)
                mse_variance[i, j, :] = (variance_running - variance)**2
                
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([n_trials, trial_duration, mean_tested, variance_tested, 
                         mse_prediction, mse_variance],f)  
                
    else:
        
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [n_trials, trial_duration, mean_tested, variance_tested, 
             mse_prediction, mse_variance] = pickle.load(f)

    ### plot results        
    plot_mse_heatmap(n_trials, trial_duration, mean_tested, variance_tested, mse_prediction, 
                     title='Estimating the mean')
    plot_mse_heatmap(n_trials, trial_duration, mean_tested, variance_tested, mse_variance, 
                     title='Estimating the variance')   


# %% MSE of estimated mean (prediction) and variance for different distributions

flag = 0
flg_plot_only = 1

if flag==1:
    
    ### load and define parameters
    input_flg = '11' # 10, 01, 11
    
    VS, VV = int(input_flg[0]), int(input_flg[1])
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/moments/data_test_distributions_' + input_flg + '.pickle'
    
    if flg_plot_only==0:
        
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename, VS=VS, VV=VV)
        
        if input_flg=='10':
            nPE_scale = 1.015
            pPE_scale = 1.023
        elif input_flg=='01':
            nPE_scale = 1.7 # 1.72
            pPE_scale = 1.7 # 1.68
        elif input_flg=='11':
            nPE_scale = 2.49
            pPE_scale = 2.53
            
        w_PE_to_P[0,0] *= nPE_scale
        w_PE_to_P[0,1] *= pPE_scale
        w_PE_to_V = [nPE_scale, pPE_scale]
        
        ### define stimulus parameters
        n_trials = 250
        trial_duration = 500
        n_stimuli_per_trial = 1
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        
        ### initialization
        num_repeats = 30
        mse_prediction = np.zeros((5, n_trials * trial_duration, num_repeats))
        mse_variance = np.zeros((5, n_trials * trial_duration, num_repeats))
        trials = np.arange(n_trials * trial_duration)/trial_duration
        
        ### define mean and variance of distribution
        mean_stimuli = 5
        std_stimuli = 2
        
        ### test different distributions 
        for i in range(5):
            
            for seed in range(num_repeats):
                
                np.random.seed(seed)  
                print(str(i+1) + '/5 and ' + str(seed+1) + '/' + str(num_repeats))
            
                ## create stimuli
                if i==0:
                    stimuli = np.random.normal(mean_stimuli, std_stimuli, size=(n_trials * n_stimuli_per_trial))
                elif i==1:
                    stimuli = random_uniform_from_moments(mean_stimuli, std_stimuli, (n_trials * n_stimuli_per_trial))
                elif i==2:
                    stimuli = random_lognormal_from_moments(mean_stimuli, std_stimuli, (n_trials * n_stimuli_per_trial))
                elif i==3:
                    stimuli = random_gamma_from_moments(mean_stimuli, std_stimuli, (n_trials * n_stimuli_per_trial))
                elif i==4:
                    stimuli = random_binary_from_moments(mean_stimuli, std_stimuli, (n_trials * n_stimuli_per_trial))
                    
                stimuli = np.repeat(stimuli, n_repeats_per_stim)
                
                ## compute variances and predictions
                
                ## run model
                prediction, variance, PE_activity = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                                tc_var_per_stim, tau_pe, fixed_input, 
                                                                                stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
                    
                ## compute mean squared error
                running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
                mse_prediction[i, :, seed] = (running_average - prediction)**2
                
                mean_running = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
                momentary_variance = (stimuli - mean_running)**2
                variance_running = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1)
                mse_variance[i, :, seed] = (variance_running - variance)**2
                
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([trials, mean_stimuli, std_stimuli, mse_prediction, mse_variance],f)  
                
    else:
        
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [trials, mean_stimuli, std_stimuli, mse_prediction, mse_variance] = pickle.load(f)
            
        num_repeats = np.size(mse_prediction,2)
        
        
    ### plot mean squared errors (in comparison to the data)      
    inset_labels = ['normal','uniform','lognormal','gamma', 'binary (p=0.5)']
    SEM_prediction = np.std(mse_prediction,2)/np.sqrt(num_repeats)
    SEM_variance = np.std(mse_variance,2)/np.sqrt(num_repeats)
    plot_mse(trials, mean_stimuli, std_stimuli, np.mean(mse_prediction,2), 
             SEM = SEM_prediction, title = 'Mean of stimuli', inset_labels=inset_labels)
    plot_mse(trials, mean_stimuli, std_stimuli, np.mean(mse_variance,2), 
             SEM = SEM_variance, title = 'Variance of stimuli')
    
    
# %%

# %% Test to check lognormal

# flag = 0
# seed = 28

# if flag==1:
    
#     ### define/load parameters
#     input_flg = '10' # 10, 01, 11
    
#     VS, VV = int(input_flg[0]), int(input_flg[1])
#     filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
#     file_data4plot = '../results/data/moments/Data_example_MFN_' + input_flg + '.pickle'
    
        
#     [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
#      tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename, VS=VS, VV=VV)
    
#     if input_flg=='10':
#         nPE_scale = 1.015
#         pPE_scale = 1.023
#     elif input_flg=='01':
#         nPE_scale = 1.7 # 1.72
#         pPE_scale = 1.7 # 1.68
#     elif input_flg=='11':
#         nPE_scale = 2.49
#         pPE_scale = 2.53
        
#     w_PE_to_P[0,0] *= nPE_scale
#     w_PE_to_P[0,1] *= pPE_scale
#     w_PE_to_V = [nPE_scale, pPE_scale]
    
#     ### define stimuli
#     n_trials = 250
#     trial_duration = 500
#     n_stimuli_per_trial = 1
#     n_repeats_per_stim = trial_duration/n_stimuli_per_trial
    
#     mean_stimuli, std_stimuli = 5, 2
    
#     np.random.seed(seed)  
    
#     stimuli = random_lognormal_from_moments(mean_stimuli, std_stimuli, (n_trials * n_stimuli_per_trial))
#     stimuli = np.repeat(stimuli, n_repeats_per_stim)
    
#     ### compute variances and predictions
    
#     ## run model
#     prediction, variance, PE_activity = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
#                                                                         tc_var_per_stim, tau_pe, fixed_input, 
#                                                                         stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)


        
#     ### plot prediction and variance (in comparison to the data) 
#     plot_prediction(n_trials, stimuli, trial_duration, prediction)  
#     plot_variance(n_trials, stimuli, trial_duration, variance) 