#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 08:01:46 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.mean_field_model import default_para, stimuli_moments_from_uniform, run_mean_field_model, alpha_parameter_exploration
from src.plot_toy_model import plot_limit_case, plot_alpha_para_exploration_ratios, plot_fraction_sensory_comparsion, plot_alpha_para_exploration
from src.plot_toy_model import plot_manipulation_results
# from src.mean_field_model import stimuli_moments_from_uniform, run_toy_model, default_para, alpha_parameter_exploration
# from src.mean_field_model import random_uniform_from_moments, random_lognormal_from_moments, random_gamma_from_moments
# from src.mean_field_model import stimuli_from_mean_and_std_arrays
from src.plot_results_mfn import plot_limit_case_example, plot_transitions_examples, heatmap_summary_transitions, plot_transition_course


import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% erase after testing

import matplotlib.pyplot as plt
import seaborn as sns

# %% Note!!!

# Here the stimuli are drawn from a normal distribution with mean and std 
# for each stimulus presentation, mean and std are drawn from a uniform distribution


# %% limit cases

flag = 0
flg_limit_case = 0 # 0 = mean the same, std large; 1 = mean varies, std = 0

flg_plot_only = 1

if flag==1:
    
    ### file to save data
    file_data4plot = '../results/data/weighting/data_example_limit_case_' + str(flg_limit_case) + '.pickle'
    
    if flg_plot_only==0:
        
        ### load and define parameters
        input_flg = '10'
        filename = '../results/data/moments/Data_Optimal_Parameters_MFN_10.pickle'
        
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)
        
        if input_flg=='10':
            nPE_scale = 1.015
            pPE_scale = 1.023
        elif input_flg=='01':
            nPE_scale = 1.7 # 1.72
            pPE_scale = 1.7 # 1.68
        elif input_flg=='11':
            nPE_scale = 2.49
            pPE_scale = 2.53
            
        w_PE_to_P[0,0] *= nPE_scale * 15 # !!!!!!!!!!!  to make it faster
        w_PE_to_P[0,1] *= pPE_scale * 15 # !!!!!!!!!!!
        w_PE_to_V = [nPE_scale, pPE_scale]
        
        v_PE_to_P[0,0] *= nPE_scale * 0.7 # !!!!!!!!!!! to make it slower
        v_PE_to_P[0,1] *= pPE_scale * 0.7 # !!!!!!!!!!!
        v_PE_to_V = [nPE_scale, pPE_scale]
        
        tc_var_per_stim = dtype(1000)
        tc_var_pred = dtype(1000)
        
        ### define stimuli
        n_trials = 100
        trial_duration = 5000
        n_stimuli_per_trial = 10
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        # In each trial a stimulus is shown. This stimulus may vary (depending on limit case)
        # Between trials the stimulus is either the same or varies (depending on the limit case)
        
        if flg_limit_case==0:
            stimuli = stimuli_moments_from_uniform(n_trials, np.int32(n_stimuli_per_trial), 3, 3, 1, 5) # mean 3, SD between 1 and 5
        else:
            stimuli = stimuli_moments_from_uniform(n_trials, np.int32(n_stimuli_per_trial), 1, 5, 0, 0) # mean between 1 and 5, SD 0
      
        stimuli = np.repeat(stimuli, n_repeats_per_stim)
        
        ### compute variances and predictions
        
        ## run model
        [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
          alpha, beta, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                              tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli,
                                                              w_PE_to_V = w_PE_to_V, v_PE_to_V = v_PE_to_V)
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([n_trials, trial_duration, stimuli, prediction, mean_of_prediction, 
                         variance_per_stimulus, variance_prediction, alpha, beta, weighted_output],f) 
                                                       
    else:
        
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [n_trials, trial_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, 
             variance_prediction, alpha, beta, weighted_output] = pickle.load(f)  
        
    
    ### plot results
    if flg_limit_case==0:
        plot_limit_case_example(n_trials, trial_duration, stimuli, prediction, mean_of_prediction, 
                                variance_per_stimulus, variance_prediction, alpha, beta, weighted_output)
    else:
        plot_limit_case_example(n_trials, trial_duration, stimuli, prediction, mean_of_prediction, 
                                variance_per_stimulus, variance_prediction, alpha, beta, weighted_output, plot_legend=False)


# %% Summary ... extrapolate between cases 

flag = 0
flg_plot_only = 1

if flag==1:
    
    ### file to save data
    file_data4plot = '../results/data/weighting/data_weighting_heatmap.pickle'
    
    if flg_plot_only==0:
        
        ### load and define parameters
        input_flg = '10'
        filename = '../results/data/moments/Data_Optimal_Parameters_MFN_10.pickle'
        
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)
        
        if input_flg=='10':
            nPE_scale = 1.015
            pPE_scale = 1.023
        elif input_flg=='01':
            nPE_scale = 1.7 # 1.72
            pPE_scale = 1.7 # 1.68
        elif input_flg=='11':
            nPE_scale = 2.49
            pPE_scale = 2.53
            
        w_PE_to_P[0,0] *= nPE_scale * 15 # !!!!!!!!!!! to make it faster
        w_PE_to_P[0,1] *= pPE_scale * 15 # !!!!!!!!!!!
        w_PE_to_V = [nPE_scale, pPE_scale]
        
        v_PE_to_P[0,0] *= nPE_scale * 0.7 # !!!!!!!!!!! to make it slower
        v_PE_to_P[0,1] *= pPE_scale * 0.7 # !!!!!!!!!!!
        v_PE_to_V = [nPE_scale, pPE_scale]
        
        tc_var_per_stim = dtype(1000)
        tc_var_pred = dtype(1000)
        
        ### stimulation & simulation parameters
        n_trials = np.int32(100)
        last_n = np.int32(30)
        trial_duration = np.int32(5000)# dtype(5000)
        n_stimuli_per_trial = np.int32(10)
        n_repeats_per_stim = np.int32(trial_duration/n_stimuli_per_trial)
        n_repeats = np.int32(1) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        ### means and std's to be tested
        mean_mean, min_std = dtype(3), dtype(0)
        std_mean_arr = np.linspace(0,3,5, dtype=dtype)
        std_std_arr = np.linspace(0,3,5, dtype=dtype)
        
        ### initialise
        fraction_sensory_mean = np.zeros((len(std_std_arr),len(std_mean_arr), n_repeats), dtype=dtype)
        weighted_out = np.zeros((len(std_std_arr),len(std_mean_arr), n_repeats, np.int32(n_trials * trial_duration)), dtype=dtype)
        
        for seed in range(n_repeats):
    
            for col, std_mean in enumerate(std_mean_arr):
                
                for row, std_std in enumerate(std_std_arr):
                    
                    ### display progress
                    print(str(seed+1) + '/' + str(n_repeats) + ' and ' + str(col+1) + '/' + str(len(std_mean_arr)) + ' and ' + str(row+1) + '/' + str(len(std_std_arr)))
            
                    ### define stimuli
                    stimuli = stimuli_moments_from_uniform(n_trials, n_stimuli_per_trial, dtype(mean_mean - np.sqrt(3)*std_mean), 
                                                           dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))
                    
                    stimuli = dtype(np.repeat(stimuli, n_repeats_per_stim))
                    
                    ### run model
                    [_, _, _, _, alpha, _, 
                     weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                             tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli)
                    
                    ### fraction of sensory input in weighted output
                    fraction_sensory_mean[row, col, seed] = np.mean(alpha[(n_trials - last_n) * trial_duration:])
     
                    ### weihgted output
                    weighted_out[row, col, seed, :] = weighted_output
        
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([fraction_sensory_mean, std_std_arr, std_mean_arr, weighted_out],f) 
     
    else:
        
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [fraction_sensory_mean, std_std_arr, std_mean_arr, weighted_out] = pickle.load(f)
        
    ### average over seeds
    fraction_sensory_mean_averaged_over_seeds = np.mean(fraction_sensory_mean,2)
    
    ### plot results
    plot_alpha_para_exploration(fraction_sensory_mean_averaged_over_seeds, std_std_arr, std_mean_arr, 2, 
                                xlabel='unexpected uncertainty \n(variability across trial)', ylabel='expected uncertainty \n(variability within trial)')
    
    
# %% Transition examples

flag = 0
flg_plot_only = 1

# 3300 --> 3315 --> 1515 --> 1500 --> 3300

if flag==1:
    
    ### file to save data
    state_before = '3300' # 3300, 3315, 1500, 1515
    state_after = '3315' # 3300, 3315, 1500, 1515
    file_data4plot = '../results/data/weighting/data_transition_example_' + state_before + '_' + state_after + '.pickle'
    
    if flg_plot_only==0:
        
        ### load and define parameters
        input_flg = '10'
        filename = '../results/data/moments/Data_Optimal_Parameters_MFN_10.pickle'
        
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)
        
        if input_flg=='10':
            nPE_scale = 1.015
            pPE_scale = 1.023
        elif input_flg=='01':
            nPE_scale = 1.7 # 1.72
            pPE_scale = 1.7 # 1.68
        elif input_flg=='11':
            nPE_scale = 2.49
            pPE_scale = 2.53
            
        w_PE_to_P[0,0] *= nPE_scale * 15 # !!!!!!!!!!! to make it faster
        w_PE_to_P[0,1] *= pPE_scale * 15 # !!!!!!!!!!!
        w_PE_to_V = [nPE_scale, pPE_scale]
        
        v_PE_to_P[0,0] *= nPE_scale * 0.7 # !!!!!!!!!!! to make it slower
        v_PE_to_P[0,1] *= pPE_scale * 0.7 # !!!!!!!!!!!
        v_PE_to_V = [nPE_scale, pPE_scale]
        
        tc_var_per_stim = dtype(1000)
        tc_var_pred = dtype(1000)
        
        ### define stimuli
        n_trials = 120
        trial_duration = 5000
        n_stimuli_per_trial = 10
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        # In each trial a stimulus is shown. This stimulus may vary (depending on limit case)
        # Between trials the stimulus is either the same or varies (depending on the limit case)
        
        stimuli = np.zeros(n_trials * n_stimuli_per_trial)
        mid = (n_trials*n_stimuli_per_trial)//2
        
        mu_min, mu_max = int(state_before[0]), int(state_before[1])
        sd_min, sd_max = int(state_before[2]), int(state_before[3])
        stimuli[:mid] = stimuli_moments_from_uniform(n_trials//2, np.int32(n_stimuli_per_trial), 
                                                     mu_min, mu_max, sd_min, sd_max)
        
        mu_min, mu_max = int(state_after[0]), int(state_after[1])
        sd_min, sd_max = int(state_after[2]), int(state_after[3])
        stimuli[mid:] = stimuli_moments_from_uniform(n_trials//2, np.int32(n_stimuli_per_trial), 
                                                     mu_min, mu_max, sd_min, sd_max)
        
        stimuli = np.repeat(stimuli, n_repeats_per_stim)
        
        ### compute variances and predictions
        
        ## run model
        [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
          alpha, beta, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                              tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli,
                                                              w_PE_to_V = w_PE_to_V, v_PE_to_V = v_PE_to_V)
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([n_trials, trial_duration, stimuli, prediction, mean_of_prediction, 
                         variance_per_stimulus, variance_prediction, alpha, beta, weighted_output],f) 
                                                       
    else:
        
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [n_trials, trial_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, 
             variance_prediction, alpha, beta, weighted_output] = pickle.load(f)  
        
    
    ### plot results
    # plot_limit_case_example(n_trials, trial_duration, stimuli, prediction, mean_of_prediction, 
    #                         variance_per_stimulus, variance_prediction, alpha, beta, weighted_output, time_plot=0)

    if state_before=='3300':
        plot_transitions_examples(n_trials, trial_duration, stimuli, alpha, beta, weighted_output, 
                                  time_plot=0, ylim=[-15,20])
    else:
        plot_transitions_examples(n_trials, trial_duration, stimuli, alpha, beta, weighted_output, 
                                  time_plot=0, ylim=[-15,20], plot_ylable=False)    
    
    
# %% Summary ... transitions

flag = 0
flg_plot_only = 0

if flag==1:
    
    ### file to save data
    file_data4plot = '../results/data/weighting/data_transitions.pickle'
    
    if flg_plot_only==0:
        
        ### define all transitions
        states = ['3300', '3315', '1515', '1500']
        #state_before = ['3300', '3315', '1500', '1515']
        #state_after = '3300' # 3300, 3315, 1500, 1515
        
        ### load and define parameters
        input_flg = '10'
        filename = '../results/data/moments/Data_Optimal_Parameters_MFN_10.pickle'
        
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)
        
        if input_flg=='10':
            nPE_scale = 1.015
            pPE_scale = 1.023
        elif input_flg=='01':
            nPE_scale = 1.7 # 1.72
            pPE_scale = 1.7 # 1.68
        elif input_flg=='11':
            nPE_scale = 2.49
            pPE_scale = 2.53
            
        w_PE_to_P[0,0] *= nPE_scale * 15 # !!!!!!!!!!! to make it faster
        w_PE_to_P[0,1] *= pPE_scale * 15 # !!!!!!!!!!!
        w_PE_to_V = [nPE_scale, pPE_scale]
        
        v_PE_to_P[0,0] *= nPE_scale * 0.7 # !!!!!!!!!!! to make it slower
        v_PE_to_P[0,1] *= pPE_scale * 0.7 # !!!!!!!!!!!
        v_PE_to_V = [nPE_scale, pPE_scale]
        
        tc_var_per_stim = dtype(1000)
        tc_var_pred = dtype(1000)
        
        ### define stimuli
        n_trials = 120
        trial_duration = 5000
        n_stimuli_per_trial = 10
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        # In each trial a stimulus is shown. This stimulus may vary (depending on limit case)
        # Between trials the stimulus is either the same or varies (depending on the limit case)
        
        ### initialise
        fraction_initial = np.zeros((4,4))
        fraction_steady_state = np.zeros((4,4))
        
        for j, state_before in enumerate(states):
            for i, state_after in enumerate(states):
                
                print(state_before + ' --> ' + state_after)
                
                if state_before!=state_after:
                
                    stimuli = np.zeros(n_trials * n_stimuli_per_trial)
                    mid = (n_trials*n_stimuli_per_trial)//2
                    
                    mu_min, mu_max = int(state_before[0]), int(state_before[1])
                    sd_min, sd_max = int(state_before[2]), int(state_before[3])
                    stimuli[:mid] = stimuli_moments_from_uniform(n_trials//2, np.int32(n_stimuli_per_trial), 
                                                                 mu_min, mu_max, sd_min, sd_max)
                    
                    mu_min, mu_max = int(state_after[0]), int(state_after[1])
                    sd_min, sd_max = int(state_after[2]), int(state_after[3])
                    stimuli[mid:] = stimuli_moments_from_uniform(n_trials//2, np.int32(n_stimuli_per_trial), 
                                                                 mu_min, mu_max, sd_min, sd_max)
                    
                    stimuli = np.repeat(stimuli, n_repeats_per_stim)
                    
                    ### compute variances and predictions
                    
                    ## run model
                    [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
                      alpha, beta, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                          tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli,
                                                                          w_PE_to_V = w_PE_to_V, v_PE_to_V = v_PE_to_V)
                                                                           
                    fraction_initial[i,j] = np.mean(alpha[(n_trials//2 * trial_duration):((n_trials//2 + 1) * trial_duration)])
                    fraction_steady_state[i,j] = np.mean(alpha[(n_trials - 10) * trial_duration:])
                    
                else:
                    
                    fraction_initial[i,j] = np.nan
                    fraction_steady_state[i,j] = np.nan
                                                                           
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([n_trials, trial_duration, stimuli, states, fraction_initial, fraction_steady_state],f) 
                                                       
    else:
        
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [n_trials, trial_duration, stimuli, states, fraction_initial, fraction_steady_state] = pickle.load(f)  
        
    
    ### plot results
    # plot_limit_case_example(n_trials, trial_duration, stimuli, prediction, mean_of_prediction, 
    #                         variance_per_stimulus, variance_prediction, alpha, beta, weighted_output)

    heatmap_summary_transitions(fraction_initial)#, 'Immediately after transition')
    heatmap_summary_transitions(fraction_steady_state)#, 'Steady state reached')


# %% Transitions time course

flag = 0
flg_plot_only = 1

if flag==1:
    
    ### file to save data
    file_data4plot = '../results/data/weighting/data_transitions.pickle'
    
    if flg_plot_only==0:
        
        ### define all transitions
        states = ['3300', '3315', '1515', '1500']
        
        ### load and define parameters
        input_flg = '10'
        filename = '../results/data/moments/Data_Optimal_Parameters_MFN_10.pickle'
        
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)
        
        if input_flg=='10':
            nPE_scale = 1.015
            pPE_scale = 1.023
        elif input_flg=='01':
            nPE_scale = 1.7 # 1.72
            pPE_scale = 1.7 # 1.68
        elif input_flg=='11':
            nPE_scale = 2.49
            pPE_scale = 2.53
            
        w_PE_to_P[0,0] *= nPE_scale * 15 # !!!!!!!!!!! to make it faster
        w_PE_to_P[0,1] *= pPE_scale * 15 # !!!!!!!!!!!
        w_PE_to_V = [nPE_scale, pPE_scale]
        
        v_PE_to_P[0,0] *= nPE_scale * 0.7 # !!!!!!!!!!! to make it slower
        v_PE_to_P[0,1] *= pPE_scale * 0.7 # !!!!!!!!!!!
        v_PE_to_V = [nPE_scale, pPE_scale]
        
        tc_var_per_stim = dtype(1000)
        tc_var_pred = dtype(1000)
        
        ### define stimuli
        n_trials = 120
        trial_duration = 5000
        n_stimuli_per_trial = 10
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        # In each trial a stimulus is shown. This stimulus may vary (depending on limit case)
        # Between trials the stimulus is either the same or varies (depending on the limit case)
        
        ### initialise
        window_size = np.linspace(0,5,21)
        fraction = np.zeros((4,4, len(window_size)))
        
        for j, state_before in enumerate(states):
            for i, state_after in enumerate(states):
                
                print(state_before + ' --> ' + state_after)
                
                if state_before!=state_after:
                
                    stimuli = np.zeros(n_trials * n_stimuli_per_trial)
                    mid = (n_trials*n_stimuli_per_trial)//2
                    
                    mu_min, mu_max = int(state_before[0]), int(state_before[1])
                    sd_min, sd_max = int(state_before[2]), int(state_before[3])
                    stimuli[:mid] = stimuli_moments_from_uniform(n_trials//2, np.int32(n_stimuli_per_trial), 
                                                                 mu_min, mu_max, sd_min, sd_max)
                    
                    mu_min, mu_max = int(state_after[0]), int(state_after[1])
                    sd_min, sd_max = int(state_after[2]), int(state_after[3])
                    stimuli[mid:] = stimuli_moments_from_uniform(n_trials//2, np.int32(n_stimuli_per_trial), 
                                                                 mu_min, mu_max, sd_min, sd_max)
                    
                    stimuli = np.repeat(stimuli, n_repeats_per_stim)
                    
                    ### compute variances and predictions
                    
                    ## run model
                    [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
                      alpha, beta, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                          tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli,
                                                                          w_PE_to_V = w_PE_to_V, v_PE_to_V = v_PE_to_V)
                    ### compute fraction
                    for k, span in enumerate(window_size):  
                        if span==0:
                            fraction[i,j,k] = np.mean(alpha[((n_trials//2 - 5) * trial_duration):int((n_trials//2) * trial_duration)])   
                        else:                                   
                            fraction[i,j,k] = np.mean(alpha[(n_trials//2 * trial_duration):int((n_trials//2+span) * trial_duration)])                                                 
                    
                else:
                    
                    fraction[i,j,:] = np.nan
                                                                           
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([n_trials, trial_duration, stimuli, states, window_size, fraction],f) 
                                                       
    else:
        
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [n_trials, trial_duration, stimuli, states, window_size, fraction] = pickle.load(f)  
            
            
    ### plot results
    plot_transition_course(file_data4plot)


# %% ####################################################################
####################### Below old ... ignore ############################
#########################################################################

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