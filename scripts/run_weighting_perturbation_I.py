#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 08:54:46 2022

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
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    
color_sensory = '#D76A03'
color_prediction = '#19535F'

cmap_sensory_prediction = LinearSegmentedColormap.from_list(name='cmap_sensory_prediction', 
                                                        colors=['#19535F','#fefee3','#D76A03'])

# %% Test some settings (not for publication later)

flag = 0
flg_plot_only = 0

if flag==1:
    
    ### file to save data
    #file_data4plot = '../results/data/weighting/data_example_limit_case_' + str(flg_limit_case) + '.pickle'
    
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
        n_trials = 200
        trial_duration = 5000
        n_stimuli_per_trial = 10
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        # In each trial a stimulus is shown. This stimulus may vary (depending on limit case)
        # Between trials the stimulus is either the same or varies (depending on the limit case)
        
        stimuli = stimuli_moments_from_uniform(n_trials, np.int32(n_stimuli_per_trial), 3, 3, 1, 5)
        stimuli = np.repeat(stimuli, n_repeats_per_stim)
        
        ### define perturbation
        id_cell_perturbed = 5 # !!!!!!!!!!!!!!!!!
        perturbation_strength = 1  # !!!!!!!!!!!!!!!!!
        perturbation = np.zeros((n_trials * trial_duration,8))                          
        perturbation[(n_trials * trial_duration)//2:, id_cell_perturbed] = perturbation_strength          
        fixed_input_plus_perturbation = fixed_input + perturbation
        
        ### compute variances and predictions
        
        ## run model
        [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
         alpha, beta, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                              tc_var_per_stim, tc_var_pred, tau_pe, fixed_input_plus_perturbation, stimuli,
                                                              w_PE_to_V = w_PE_to_V, v_PE_to_V = v_PE_to_V)
        ### save data for later
        # with open(file_data4plot,'wb') as f:
        #     pickle.dump([n_trials, trial_duration, stimuli, prediction, mean_of_prediction, 
        #                  variance_per_stimulus, variance_prediction, alpha, beta, weighted_output],f) 
                                                       
    else:
        
        ### load data for plotting
        # with open(file_data4plot,'rb') as f:
        #     [n_trials, trial_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, 
        #      variance_prediction, alpha, beta, weighted_output] = pickle.load(f) 
        print('')
        
    
    ### plot results
    plot_limit_case_example(n_trials, trial_duration, stimuli, prediction, mean_of_prediction, 
                            variance_per_stimulus, variance_prediction, alpha, beta, weighted_output, 
                            plot_legend=False, time_plot=0)


# %% summary statistics with one example perturbation (VIP activated) affecting both 'columns'

flag = 0
flg_plot_only = 0

if flag==1:
    
    ### file to save data
    file_data4plot = '../results/data/weighting_perturbation/data_weighting_heatmap_perturbation_example.pickle'
    
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
        n_trials = np.int32(300)
        last_n = np.int32(100)
        trial_duration = np.int32(5000)# dtype(5000)
        n_stimuli_per_trial = np.int32(10)
        n_repeats_per_stim = np.int32(trial_duration/n_stimuli_per_trial)
        n_repeats = np.int32(1) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        ### means and std's to be tested
        mean_mean, min_std = dtype(3), dtype(0)
        std_mean_arr = np.linspace(0,3,5, dtype=dtype)
        std_std_arr = np.linspace(0,3,5, dtype=dtype)
        
        ### define perturbation
        id_cell_perturbed = 7
        perturbation_strength = 1
        perturbation = np.zeros((n_trials * trial_duration,8))                          
        perturbation[(n_trials * trial_duration)//3:, id_cell_perturbed] = perturbation_strength    # first third without perturbation    
        fixed_input_plus_perturbation = fixed_input + perturbation
        
        ### initialise
        fraction_sensory_mean = np.zeros((len(std_std_arr),len(std_mean_arr), n_repeats), dtype=dtype)
        
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
                    [_, _, _, _, alpha, _, _] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                  tc_var_per_stim, tc_var_pred, tau_pe, fixed_input_plus_perturbation, stimuli)
                    
                    ### fraction of sensory input in weighted output
                    fraction_sensory_mean[row, col, seed] = np.mean(alpha[(n_trials - last_n) * trial_duration:])
     
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([fraction_sensory_mean, std_std_arr, std_mean_arr],f) 
     
    else:
        
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [fraction_sensory_mean, std_std_arr, std_mean_arr] = pickle.load(f)
        
    ### average over seeds
    fraction_sensory_mean_averaged_over_seeds = np.mean(fraction_sensory_mean,2)
    
    ### plot results
    plot_alpha_para_exploration(fraction_sensory_mean_averaged_over_seeds, std_std_arr, std_mean_arr, 2, 
                                xlabel='unexpected uncertainty \n(variability across trial)', ylabel='expected uncertainty \n(variability within trial)')
    

# %% Exc/Inh perturbation, all IN neurons, all MFN

# run for each MFN network

flag = 0

if flag==1:
    
    ### load and define parameters
    input_flg = '11'
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_10.pickle'
    file_data4plot = file_data4plot = '../results/data/weighting_perturbation/data_weighting_perturbations_' + input_flg + '.pickle'
    
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
    n_trials = np.int32(400)
    last_n = np.int32(100)
    trial_duration = np.int32(5000)# dtype(5000)
    n_stimuli_per_trial = np.int32(10)
    n_repeats_per_stim = np.int32(trial_duration/n_stimuli_per_trial)
    n_repeats = np.int32(1) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    ### perturbation specs
    perturbations = [-1, 1]
    
    ### means and std's to be tested
    mean_mean, min_std = dtype(3), dtype(0)
    std_mean_arr = np.array([3, 2.25, 1.5, 0.75, 0])
    std_std_arr = np.array([0, 0.75, 1.5, 2.25, 3])
    
    ### initalise
    frac_sens_before_pert = np.zeros((len(perturbations), 4, len(std_mean_arr)))
    frac_sens_after_pert = np.zeros((len(perturbations), 4, len(std_mean_arr)))
    
    ### compute variances and predictions
    for id_mod, perturbation_strength in enumerate(perturbations): 
            
            for id_cell_perturbed in range(4,8): # (nPE, pPE, nPE dend, pPE dend,) PVv, PVm, SOM, VIP
            
                ### display progress
                print(str(id_mod+1) + '/' + str(len(perturbations)) + ' and ' + str(id_cell_perturbed-3) + '/' + str(4))
        
                ## add perturbation
                perturbation = np.zeros((n_trials * trial_duration,8))                          
                perturbation[(n_trials * trial_duration)//2:, id_cell_perturbed] = perturbation_strength          
                fixed_input_plus_perturbation = fixed_input + perturbation
                
                for id_stim in range(len(std_mean_arr)):
                    
                    print(str(id_stim+1) + '/' + str(len(std_mean_arr)))
                    
                    ### define stimuli
                    std_mean = std_mean_arr[id_stim]
                    std_std = std_std_arr[id_stim]
                    
                    np.random.seed(186)
                    
                    stimuli = stimuli_moments_from_uniform(n_trials, n_stimuli_per_trial, dtype(mean_mean - np.sqrt(3)*std_mean), 
                                                           dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))
                    
                    stimuli = dtype(np.repeat(stimuli, n_repeats_per_stim))
                    
                    
                    ### run model
                    [_, _, _, _, alpha, _, _] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                  tc_var_per_stim, tc_var_pred, tau_pe, fixed_input_plus_perturbation, stimuli)
                    
                    ###  check for inf and nan
                    if ((sum(np.isinf(alpha))>0) or (sum(np.isnan(alpha))>0)):
                        print('Warning: computation yields nan or inf in alpha.')
                    
                    ### fraction of sensory input in weighted output
                    frac_sens_before_pert[id_mod, id_cell_perturbed-4, id_stim] = np.mean(alpha[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
                    frac_sens_after_pert[id_mod, id_cell_perturbed-4, id_stim] = np.mean(alpha[(n_trials - last_n) * trial_duration:])
     
                    
    ### save data
    with open(file_data4plot,'wb') as f:
        pickle.dump([n_trials, last_n, trial_duration, frac_sens_before_pert, frac_sens_after_pert],f)  
                                

# %% plot results from perturbation experiments (see above), only one example

### ATTENTION: AS SOON AS YO FIT TO ALL INPUTS TESTED (NOT JUST DIAGONAL), TAKE OUT (0,0) !!!!

flag = 0

if flag==1:
     
    ### define MFN
    input_flg = '10' #['10', '01', '11']
    
    ### load data
    file_data4plot = '../results/data/weighting_perturbation/data_weighting_perturbations_' + input_flg + '.pickle'
    
    with open(file_data4plot,'rb') as f:
        n_trials, last_n, trial_duration, frac_sens_before_pert, frac_sens_after_pert = pickle.load(f)  

    ### figure setup
    fig, ax = plt.subplots(1,1, tight_layout=True, figsize=(4,3))
   
    ### which IN?
    i = 3 # IN that is considered
    
    ### plotting settings etc.
    colors = ['#44729D', '#BA403C']
    marker = ['h', 'X', 'P', 'd']
    labels = ['inhibitory', 'excitatory']
    
    for j in range(2):

        ax.scatter(frac_sens_before_pert[j,i,:], frac_sens_after_pert[j,i,:], color=colors[j], lw=1, 
                   marker=marker[i], label='')
        
        m, n = np.polyfit(frac_sens_before_pert[j,i,:], frac_sens_after_pert[j,i,:],1)
        x = frac_sens_before_pert[j,i,:]
        y = m * x + n
            
        ax.plot(x,y, color=colors[j], lw=1, alpha=1, zorder=0, label=labels[j])
    
    ax.legend(loc=0, frameon=False, title='modulation')
    ax.axline((0.5, 0.5), slope=1, color='k', ls=':', alpha=1, zorder=0)

    ax.set_ylabel(r'fraction$_\mathrm{sens}$ (after)')
    ax.set_xlabel(r'fraction$_\mathrm{sens}$ (before)')
    
    sns.despine(ax=ax)
            

# %% Summarise change in weighting for all INs and all perturbations per MFN

flag = 0

if flag==1:
    
    ### define MFN
    input_flg = '10'
    
    ### load data
    file_data4plot = '../results/data/weighting_perturbation/data_weighting_perturbations_' + input_flg + '.pickle'
    
    with open(file_data4plot,'rb') as f:
        n_trials, last_n, trial_duration, frac_sens_before_pert, frac_sens_after_pert = pickle.load(f)  
    
    ### figure setup
    fig, ax = plt.subplots(1,1, tight_layout=True, figsize=(4,3))
    
    colors = ['#44729D', '#BA403C']
    marker = ['h', 'X', 'P', 'd']
    labels = ['PVv', 'PVm', 'SOM', 'VIP']
    
    ### initialise
    slopes = np.zeros((4,2))
    offsets = np.zeros((4,2))
    
    ### all IN types
    for id_cell in range(4):
        
        ### inhibitory/excitatory perturbation
        for id_pert in range(2):
            
            m, n = np.polyfit(frac_sens_before_pert[id_pert,id_cell,:], frac_sens_after_pert[id_pert,id_cell,:],1)
        
            slopes[id_cell, id_pert] = m
            offsets[id_cell, id_pert] = n
         
        ### plotting
        ax.scatter(offsets[id_cell,:], slopes[id_cell,:], c=colors, marker=marker[id_cell])
        ax.scatter(np.nan, np.nan, c='k', marker=marker[id_cell], label=labels[id_cell])
    
    ax.axhline(1, color='k', ls=':', zorder=0)
    ax.axvline(0, color='k', ls=':', zorder=0)
    ax.set_xlim([-0.1,0.25])
    ax.set_ylim([0.5,1.25])
    
    ax.legend(loc=0, frameon=False, ncol=2)
    ax.set_xlabel('offset')
    ax.set_ylabel('slope')
    
    sns.despine(ax=ax)
        
    
# %% Test: Exc/Inh perturbation, VIP in MFN1, +/- 2 perturbation, whole range

# !!!!!!!!! continue here ...

flag = 0

if flag==1:
    
    ### load and define parameters
    input_flg = '10'
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_10.pickle'
    file_data4plot = file_data4plot = '../results/data/weighting_perturbation/test_weighting_perturbations_' + input_flg + '.pickle'
    
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
    n_trials = np.int32(400)
    last_n = np.int32(100)
    trial_duration = np.int32(5000)# dtype(5000)
    n_stimuli_per_trial = np.int32(10)
    n_repeats_per_stim = np.int32(trial_duration/n_stimuli_per_trial)
    n_repeats = np.int32(1) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    ### perturbation specs
    perturbations = [-2, 2]
    
    ### means and std's to be tested
    mean_mean, min_std = dtype(3), dtype(0)
    std_mean_arr = np.linspace(0,3,5, dtype=dtype)
    std_std_arr = np.linspace(0,3,5, dtype=dtype)
    # std_mean_arr = np.array([3, 2.25, 1.5, 0.75, 0])
    # std_std_arr = np.array([0, 0.75, 1.5, 2.25, 3])
    
    ### initalise
    frac_sens_before_pert = np.zeros((len(perturbations), 4, len(std_std_arr),len(std_mean_arr)))
    frac_sens_after_pert = np.zeros((len(perturbations), 4, len(std_std_arr),len(std_mean_arr)))
    weighted_out = np.zeros((len(perturbations), 4, len(std_std_arr),len(std_mean_arr), np.int32(n_trials * trial_duration)))
    
    ### compute variances and predictions
    for id_mod, perturbation_strength in enumerate(perturbations): 
            
            for id_cell_perturbed in range(7,8): #range(4,8): # (nPE, pPE, nPE dend, pPE dend,) PVv, PVm, SOM, VIP
            
                ### display progress
                print(str(id_mod+1) + '/' + str(len(perturbations)) + ' and ' + str(id_cell_perturbed-3) + '/' + str(4))
        
                ## add perturbation
                perturbation = np.zeros((n_trials * trial_duration,8))                          
                perturbation[(n_trials * trial_duration)//2:, id_cell_perturbed] = perturbation_strength          
                fixed_input_plus_perturbation = fixed_input + perturbation
                
                for col, std_mean in enumerate(std_mean_arr):
                    for row, std_std in enumerate(std_std_arr):
                    
                        print(str(seed+1) + '/' + str(n_repeats) + ' and ' + str(col+1) + '/' + str(len(std_mean_arr)) + ' and ' + str(row+1) + '/' + str(len(std_std_arr)))
                        
                        ### define stimuli 
                        np.random.seed(186)
                        
                        stimuli = stimuli_moments_from_uniform(n_trials, n_stimuli_per_trial, dtype(mean_mean - np.sqrt(3)*std_mean), 
                                                               dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))
                        
                        stimuli = dtype(np.repeat(stimuli, n_repeats_per_stim))
                        
                        
                        ### run model
                        [_, _, _, _, alpha, _, 
                         weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                 tc_var_per_stim, tc_var_pred, tau_pe, fixed_input_plus_perturbation, stimuli)
                        
                        ###  check for inf and nan
                        if ((sum(np.isinf(alpha))>0) or (sum(np.isnan(alpha))>0)):
                            print('Warning: computation yields nan or inf in alpha.')
                        
                        ### fraction of sensory input in weighted output
                        frac_sens_before_pert[id_mod, id_cell_perturbed-4, row, col] = np.mean(alpha[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
                        frac_sens_after_pert[id_mod, id_cell_perturbed-4, row, col] = np.mean(alpha[(n_trials - last_n) * trial_duration:])
         
                        ### weihgted output
                        weighted_out[id_mod, id_cell_perturbed-4, row, col, :] = weighted_output
                    
    ### save data
    with open(file_data4plot,'wb') as f:
        pickle.dump([n_trials, last_n, trial_duration, 
                     frac_sens_before_pert, frac_sens_after_pert, weighted_out],f)  
     
       
# %% #########################################################################  
##############################################################################
##############################################################################   
            
# %% Example transition with and without perturbation (only simulations)

flag = 0
flg_plot_only = False

if flag==1:
    
    ### define all transitions
    states = ['3300', '3315', '1515', '1500']
    input_flgs = ['10', '01', '11']
    
    for input_flg in input_flgs:
        ### file to save data or load data
        filename = '../results/data/moments/Data_Optimal_Parameters_MFN_10.pickle'
        file_data4plot = '../results/data/weighting_perturbation/data_transitions_perturbations_' + input_flg + '.pickle'
          
        ### load and define parameters
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
        
        ### define stimuli and perturbation strengths
        n_trials = 200
        trial_duration = 5000
        n_stimuli_per_trial = 10
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        # In each trial a stimulus is shown. This stimulus may vary (depending on limit case)
        # Between trials the stimulus is either the same or varies (depending on the limit case)
        perturbations = [-1, 1]
        num_split = np.int32(n_trials * trial_duration / 1000)
        
        ### initialise
        fraction_without_pert = np.zeros(num_split)
        fraction_with_pert = np.zeros(num_split)
        
        fraction_with = np.nan * np.ones((len(states),len(states),4,len(perturbations),num_split))
        fraction_without = np.nan * np.ones((len(states),len(states),4,len(perturbations),num_split))
        time_half_with = np.nan * np.ones((len(states),len(states),4,num_split))
        time_half_without = np.nan * np.ones((len(states),len(states),4,num_split))
        
        for j, state_before in enumerate(states):
            for i, state_after in enumerate(states):
                
                print(state_before + ' --> ' + state_after)
                
                if state_before!=state_after:
                    np.random.seed(186)
            
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
                    
                    ### run model without perturbation
                    [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
                      alpha, beta, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                          tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli,
                                                                          w_PE_to_V = w_PE_to_V, v_PE_to_V = v_PE_to_V)
                                                                           
                    fraction_without_pert[:] = np.mean(np.array_split(alpha, num_split),1)                                                     
                    
                    ### go through perturbations
                    for id_mod, perturbation_strength in enumerate(perturbations): 
                
                        for id_cell_perturbed in range(4): # the INs
                        
                            ## show progress
                            print('Perturbaton:', perturbation_strength)
                            print('Target:', id_cell_perturbed+4)
            
                            ## potential perturbation
                            perturbation = np.zeros((n_trials * trial_duration,8))                          
                            perturbation[(n_trials * trial_duration)//2:, id_cell_perturbed + 4] = perturbation_strength          
                            fixed_input_plus_perturbation = fixed_input + perturbation  
    
                            ## run model without perturbation
                            [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
                              alpha, beta, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                                  tc_var_per_stim, tc_var_pred, tau_pe, fixed_input_plus_perturbation, stimuli,
                                                                                  w_PE_to_V = w_PE_to_V, v_PE_to_V = v_PE_to_V)
                                                                    
                            fraction_with_pert[:] = np.mean(np.array_split(alpha, num_split),1)                                                        
    
    
                            ### re-name
                            fraction_with[i, j, id_cell_perturbed, id_mod, :] = fraction_with_pert
                            fraction_without[i, j, id_cell_perturbed, id_mod, :] = fraction_without_pert
                             
                                                               
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([n_trials, trial_duration, num_split, fraction_without, fraction_with],f) 
                                                       

# %% Example transition with and without perturbation (only plotting)

# maybe add vertical lines where starting and end points are (if not too confusing)
# problems: 1) Transitions usually way too fast 2) fraction very noisy (because everything is chnaging so fast)
# 3) transitions with and without perturbations look pretty indistinguishable ...

flag = 0

if flag==1:
    
    ### define all transitions
    states = ['3300', '3315', '1515', '1500']
    marker = ['^', 's', 'o', 'D']
    marker_every = 100
    
    ### file to load data
    input_flg = '11'
    file_data4plot = '../results/data/weighting_perturbation/data_transitions_perturbations_' + input_flg + '.pickle'
    
    with open(file_data4plot,'rb') as f:
        [n_trials, trial_duration, num_split, fraction_without, fraction_with] = pickle.load(f)  
    
    dt = n_trials/num_split # dt could also be set to one (in principle arbitrary units )
    
    ### plotting
    fig, ax = plt.subplots(4,2, sharex=True, sharey=True, figsize=(10,7), tight_layout=True)
    
    for t in range(4): # targets (INs only)
        
        for p in range(2): # perturbations (inhibitory, excitatory)
        
            if p==0:
                color='#508FCE'
            else:
                color='#9E3039'

            for j in [1,3]: #, state_before in enumerate(states):
                    for i in [1,3]: #, state_after in enumerate(states):
                        
                        if j==1 and i==3:
                            marker = '.'
                        elif j==3 and i==1:
                            marker = '.'
                        
                        if j!=i:
                        
                            x = (fraction_without[i,j,t,p][:-1]+fraction_without[i,j,t,p][1:])/2
                            ax[t,p].plot(x, np.diff(fraction_without[i,j,t,p]/dt), 'k', marker=marker, markevery=marker_every)
                            
                            x = (fraction_with[i,j,t,p][:-1]+fraction_with[i,j,t,p][1:])/2
                            ax[t,p].plot(x, np.diff(fraction_with[i,j,t,p]/dt), color=color, marker=marker, markevery=marker_every)
                            
                            ax[t,p].axhline(0, color='k', alpha=0.3, ls=':', zorder=0)
                            
                            if (t==0 and p==0):
                                ax[t,p].set_title('Inhibitory perturbation')
                                
                            if (t==0 and p==1):
                                ax[t,p].set_title('Excitatory perturbation')
                                
                            if t==3:
                                ax[t,p].set_xlabel('fraction f')
                                
                            if p==0:
                                ax[t,p].set_ylabel('df/dt')
                                
                            sns.despine(ax=ax[t,p])
                            

# %% Schema for df/dt vs. f to illustrate how changes may manifest

flag = 0

if flag==1:
    
    # original
    t = np.linspace(0,100,1000)
    f1 = 0.8 * np.exp(-0.05*t) + 0.1
    f2 = 0.1 * np.exp(0.022*t)
    dt = t[1]-t[0]
    
    # perturbed
    # f1_new = 0.8 * np.exp(-0.02*t) + 0.1
    f1_new = 0.7 * np.exp(-0.05*t) + 0.15
    f2_new = 0.1 * np.exp(0.022*t) + 0.1
    
    # plot
    plt.figure()
    x = (f1[:-1] + f1[1:])/2
    plt.plot(x,np.diff(f1)/dt)
    
    x = (f1_new[:-1] + f1_new[1:])/2
    plt.plot(x,np.diff(f1_new)/dt)
    
    plt.figure()
    x = (f2[:-1] + f2[1:])/2
    plt.plot(x,np.diff(f2)/dt)
    
    x = (f2_new[:-1] + f2_new[1:])/2
    plt.plot(x,np.diff(f2_new)/dt)
    