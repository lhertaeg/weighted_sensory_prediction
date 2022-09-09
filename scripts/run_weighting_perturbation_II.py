#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 4 14:43:46 2022

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

# %% summary statistics with one example perturbation (VIP activated) affecting one 'column'

flag = 0
flg_plot_only = 1
column = 2 # 1 or 2

if flag==1:
    
    ### file to save data
    file_data4plot = '../results/data/weighting_perturbation/data_weighting_heatmap_perturbation_example_column_' + str(column) + '.pickle'
    
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
        
        if column==1:
            fixed_input_1 = fixed_input_plus_perturbation
            fixed_input_2 = fixed_input
        else:
            fixed_input_1 = fixed_input
            fixed_input_2 = fixed_input_plus_perturbation
        
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
                                                                  tc_var_per_stim, tc_var_pred, tau_pe, None, stimuli,
                                                                  fixed_input_1 = fixed_input_1,
                                                                  fixed_input_2 = fixed_input_2)
                    
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
column = 1 # 1 or 2

if flag==1:
    
    ### load and define parameters
    input_flg = '11'
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_10.pickle'
    file_data4plot = '../results/data/weighting_perturbation/data_weighting_perturbations_' + input_flg + '_column_' + str(column) + '.pickle'
    
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
                
                if column==1:
                    fixed_input_1 = fixed_input_plus_perturbation
                    fixed_input_2 = fixed_input
                else:
                    fixed_input_1 = fixed_input
                    fixed_input_2 = fixed_input_plus_perturbation
                    
                
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
                                                                  tc_var_per_stim, tc_var_pred, tau_pe, None, stimuli,
                                                                  fixed_input_1 = fixed_input_1, 
                                                                  fixed_input_2 = fixed_input_2)
                    
                    ###  check for inf and nan
                    if ((sum(np.isinf(alpha))>0) or (sum(np.isnan(alpha))>0)):
                        print('Warning: computation yields nan or inf in alpha.')
                    
                    ### fraction of sensory input in weighted output
                    frac_sens_before_pert[id_mod, id_cell_perturbed-4, id_stim] = np.mean(alpha[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
                    frac_sens_after_pert[id_mod, id_cell_perturbed-4, id_stim] = np.mean(alpha[(n_trials - last_n) * trial_duration:])
     
                    
    ### save data
    with open(file_data4plot,'wb') as f:
        pickle.dump([n_trials, last_n, trial_duration, frac_sens_before_pert, frac_sens_after_pert],f)  
                                

# %% plot results from perturbation experiments (see above)

flag = 0
column = 2 # 1 or 2

if flag==1:
     
    input_flgs = ['10', '01', '11']
    
    for j in range(2):
            
        # fig, (ax, ax_cbar) = plt.subplots(2,1, gridspec_kw={'height_ratios': [20, 1]})
        fig, ax = plt.subplots(1,1)
    
        for k, input_flg in enumerate(input_flgs):
        
            ### load data
            file_data4plot = '../results/data/weighting_perturbation/data_weighting_perturbations_' + input_flg + '_column_' + str(column) + '.pickle'
            
            with open(file_data4plot,'rb') as f:
                n_trials, last_n, trial_duration, frac_sens_before_pert, frac_sens_after_pert = pickle.load(f)  
            
            
            ### colors
            colors = ['#508FCE', '#2B6299', '#79AFB9', '#39656D']
            marker = ['o', 's', 'D']
            labels = ['MFN 1', 'MFN 2', 'MFN 3']
    
            for i in range(4):

                ax.scatter(frac_sens_before_pert[j,i,:], frac_sens_after_pert[j,i,:], color=colors[i], lw=1, marker=marker[k])
                
            ax.axline((0.5, 0.5), slope=1, color='k', ls=':', alpha=1, zorder=0)
            
            if j==0:
                ax.set_title('Inhibitory perturbation')
            elif j==1:
                ax.set_title('Excitatory perturbation')
                
            # ax.set_ylabel(r'$\Delta$ fraction$_\mathrm{sens}$ (after - before)')
            ax.set_ylabel(r'fraction$_\mathrm{sens}$ (after)')
            ax.set_xlabel(r'fraction$_\mathrm{sens}$ (before)')
            
            sns.despine(ax=ax)
       
            
# %% plot results from perturbation experiments (see above), only one example

flag = 0
column = 2 # 1 or 2

if flag==1:
     
    ### define MFN
    input_flg = '10' #['10', '01', '11']
    
    ### load data
    file_data4plot = '../results/data/weighting_perturbation/data_weighting_perturbations_' + input_flg + '_column_' + str(column) + '.pickle'
    
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
column = 2 # 1 or 2

if flag==1:
    
    ### define MFN
    input_flg = '11'
    
    ### load data
    file_data4plot = '../results/data/weighting_perturbation/data_weighting_perturbations_' + input_flg + '_column_' + str(column) + '.pickle'
    
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
    ax.set_xlim([-0.12,0.25])
    ax.set_ylim([0.5,1.25])
    
    ax.legend(loc=0, frameon=False, ncol=2)
    ax.set_xlabel('offset')
    ax.set_ylabel('slope')
    
    sns.despine(ax=ax)
                   

# %% Test: Exc/Inh perturbation, whole range

flag = 0
column = 1 # 1 or 2

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
    perturbations = [-1, 1]
    
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
            
            for id_cell_perturbed in range(4,8): #range(4,8): # (nPE, pPE, nPE dend, pPE dend,) PVv, PVm, SOM, VIP
            
                ### display progress
                print(str(id_mod+1) + '/' + str(len(perturbations)) + ' and ' + str(id_cell_perturbed-3) + '/' + str(4))
        
                ## add perturbation
                perturbation = np.zeros((n_trials * trial_duration,8))                          
                perturbation[(n_trials * trial_duration)//2:, id_cell_perturbed] = perturbation_strength          
                fixed_input_plus_perturbation = fixed_input + perturbation
                
                if column==1:
                    fixed_input_1 = fixed_input_plus_perturbation
                    fixed_input_2 = fixed_input
                else:
                    fixed_input_1 = fixed_input
                    fixed_input_2 = fixed_input_plus_perturbation
              
                
                for col, std_mean in enumerate(std_mean_arr):
                    for row, std_std in enumerate(std_std_arr):
                    
                        print(str(seed+1) + '/' + str(n_repeats) + ' and ' + str(col+1) + '/' + str(len(std_mean_arr)) + ' and ' + str(row+1) + '/' + str(len(std_std_arr)))
                        
                        ### define stimuli 
                        np.random.seed(186)
                        
                        stimuli = stimuli_moments_from_uniform(n_trials, n_stimuli_per_trial, dtype(mean_mean - np.sqrt(3)*std_mean), 
                                                               dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))
                        
                        stimuli = dtype(np.repeat(stimuli, n_repeats_per_stim))
                        
                        
                        ### run model
                        [_, _, _, _, alpha, _, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                  tc_var_per_stim, tc_var_pred, tau_pe, None, stimuli,
                                                                  fixed_input_1 = fixed_input_1, 
                                                                  fixed_input_2 = fixed_input_2)
                        
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
           
       
# %% plot results from perturbation experiments (see above), only one example

### ADAPT ... !!!

flag = 0

if flag==1:
     
    ### define MFN
    input_flg = '10' #['10', '01', '11']
    
    ### load data
    file_data4plot = '../results/data/weighting_perturbation/test_weighting_perturbations_' + input_flg + '.pickle'
    
    with open(file_data4plot,'rb') as f:
        [_, _, _, frac_sens_before_pert, frac_sens_after_pert, _] = pickle.load(f)  

    ## figure setup
    fig, ax = plt.subplots(1,1, tight_layout=True, figsize=(4,3))
   
    ### which IN?
    i = 3 # IN that is considered
    
    ### plotting settings etc.
    colors = ['#44729D', '#BA403C']
    marker = ['h', 'X', 'P', 'd']
    labels = ['inhibitory', 'excitatory']
    
    for j in range(2):

        before = frac_sens_before_pert[j,i,:,:].flatten()[1:] # take out (0,0)
        after = frac_sens_after_pert[j,i,:,:].flatten()[1:] # take out (0,0)
        ax.scatter(before, after, color=colors[j], lw=1, marker=marker[i], label='')
        
        m, n = np.polyfit(before, after,1)
        after_lin = m * before + n
            
        ax.plot(before, after_lin, color=colors[j], lw=1, alpha=1, zorder=0, label=labels[j])
    
    ax.legend(loc=0, frameon=False, title='modulation')
    ax.axline((0.5, 0.5), slope=1, color='k', ls=':', alpha=1, zorder=0)

    ax.set_ylabel(r'fraction$_\mathrm{sens}$ (after)')
    ax.set_xlabel(r'fraction$_\mathrm{sens}$ (before)')
    
    sns.despine(ax=ax)
        