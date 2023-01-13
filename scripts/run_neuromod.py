#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:05:25 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle
import pandas as pd

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

# %% Effect of neuromodulators on weighted output

flag = 0

if flag==1:
    
    ### initialise
    z_exp_low_unexp_high = np.zeros(3)
    z_exp_high_unexp_low = np.zeros(3)
    columns = np.array([1,3,2])
    colors = ['#19535F', '#D76A03']
    
    ### define neuromodulator (by target IN)
    id_cell = 3
    
    if id_cell==0:
        title = 'DA activates PVs (sens)'
    elif id_cell==1:
        title = 'DA activates PVs (pred)'
    elif id_cell==2:
        title = 'NA/NE activates SOMs'
    elif id_cell==3:
        title = 'ACh or 5-HT activates VIPs'
    
    for idx, column in enumerate(columns):
    
        ### define MFN
        input_flg = '10' #['10', '01', '11']
        
        ### load data
        if column!=3: # only one of the two columns
            file_data4plot = '../results/data/weighting_perturbation/test_weighting_perturbations_' + input_flg + '_column_' + str(column) + '.pickle'
        
            with open(file_data4plot,'rb') as f:
                [_, _, _, frac_sens_before_pert, frac_sens_after_pert, _] = pickle.load(f)
        
        else: # both columns 
            file_data4plot = '../results/data/weighting_perturbation/test_weighting_perturbations_' + input_flg + '.pickle'
    
            with open(file_data4plot,'rb') as f:
                [_, _, _, frac_sens_before_pert, frac_sens_after_pert, _] = pickle.load(f)
         
        # reminder   
        # std_mean_arr = np.linspace(0,3,5, dtype=dtype)    # column
        # std_std_arr = np.linspace(0,3,5, dtype=dtype)     # row
            
        frac_exp_low_unexp_high_before = frac_sens_before_pert[1, id_cell, 0, 4]
        frac_exp_low_unexp_high_after = frac_sens_after_pert[1, id_cell, 0, 4]
        
        frac_exp_high_unexp_low_before = frac_sens_before_pert[1, id_cell, 4, 0]
        frac_exp_high_unexp_low_after = frac_sens_after_pert[1, id_cell, 4, 0]
        
        z_exp_low_unexp_high[idx] = (frac_exp_low_unexp_high_after - frac_exp_low_unexp_high_before) / frac_exp_low_unexp_high_before
        z_exp_high_unexp_low[idx] = (frac_exp_high_unexp_low_after - frac_exp_high_unexp_low_before) / frac_exp_high_unexp_low_before

    
    ### plot
    
    # ax.plot(np.arange(3), z_exp_low_unexp_high * 100, 's', color=colors[1])
    # ax.plot(np.arange(3), z_exp_high_unexp_low * 100, 's', color=colors[0])
    
    fig, ax = plt.subplots(1,1, tight_layout=True, figsize=(4,3))
    marker = ['<', 's', '>']
    text = ['low-level', 'global', 'higher-level']
    
    for i in range(3):
        ax.plot(1, z_exp_low_unexp_high[i] * 100, marker=marker[i], color=colors[1])
        ax.plot(2, z_exp_high_unexp_low[i] * 100, marker=marker[i], color=colors[0])
        ax.plot(np.nan, np.nan, marker=marker[i], color='k', label=text[i], ls='None')
    
    ylim = ax.get_ylim()
    ax.axhspan(0, ylim[1], color=colors[1], alpha=0.1)
    ax.axhspan(ylim[0], 0, color=colors[0], alpha=0.1)
    ax.set_ylim(ylim)
    
    ax.set_xticks([1,2])
    ax.set_xticklabels(['Unpredictable, \nreliable stimuli', 'Predictable, \nnoisy stimuli'])
    ax.set_xlim([0.5,2.5])
    
    ax.set_ylabel(r'change in $\alpha$ (normalised, %)')
    ax.legend(loc=0, title='PE circuit', framealpha=0.5)
    ax.set_title(title)
    
    sns.despine(ax=ax)


# %% How does the predicton change when neurmodulators activate specfic INs in either the first, the second or both PE circuits

flag = 0

if flag==1:

    ### load and define parameters
    input_flg = '10'
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/neuromod/test_prediction_neuromod_' + input_flg + '.pickle'
    
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
    
    ### means and std's to be tested
    mean_mean, min_std = dtype(3), dtype(0)
    std_mean_arr = np.array([0, 3])
    std_std_arr = np.array([3, 0])
    
    ### initalise
    prediction_before_pert = np.zeros((2, 3, 4)) # 2 limit cases, target column/s, INs affected by neuromod
    prediction_after_pert = np.zeros((2, 3, 4))
    
    ### main loop
    for row in range(2): # two limit cases
        
        print('Limit case: ', row)
        
        std_mean = std_mean_arr[row]
        std_std = std_std_arr[row]

        ## define stimuli  (
        # Please note: to make it comparable, I repeated the set of stimuli, so that before and after neuromods is driven by the same sequence of stimuli)
        np.random.seed(186)
        
        stimuli = stimuli_moments_from_uniform(n_trials//2, n_stimuli_per_trial, dtype(mean_mean - np.sqrt(3)*std_mean), 
                                           dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))
    
        stimuli = np.tile(stimuli,2)
        stimuli = dtype(np.repeat(stimuli, n_repeats_per_stim))
        
        ## define target IN affected
        for id_cell_perturbed in range(4): # 0-3: PV, PV, SOM, VIP
        
            print('- Target IN id:', id_cell_perturbed)
        
            ## add perturbation
            perturbation = np.zeros((n_trials * trial_duration,8))                          
            perturbation[(n_trials * trial_duration)//2:, id_cell_perturbed + 4] = 1          
            fixed_input_plus_perturbation = fixed_input + perturbation

            ## define target PE circuit
            for column in range(3): # 1st or 2nd PE circuit, 0: both PE circuits
            
                print('-- Target column id:', column)  
                
                if column==1:
                    fixed_input_1 = fixed_input_plus_perturbation
                    fixed_input_2 = fixed_input
                elif column==2:
                    fixed_input_1 = fixed_input
                    fixed_input_2 = fixed_input_plus_perturbation
                elif column==0:
                    fixed_input_1 = fixed_input_plus_perturbation
                    fixed_input_2 = fixed_input_plus_perturbation
                    
                ## run model
                [prediction, _, _, _, _, _, _] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                      tc_var_per_stim, tc_var_pred, tau_pe, None, stimuli,
                                                                      fixed_input_1 = fixed_input_1, 
                                                                      fixed_input_2 = fixed_input_2)
        
                ##  check for inf and nan
                if ((sum(np.isinf(prediction))>0) or (sum(np.isnan(prediction))>0)):
                    print('Warning: computation yields nan or inf in prediction.')
            
                ## save prediction before and after neuromods
                prediction_before_pert[row, column, id_cell_perturbed] = np.mean(prediction[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
                prediction_after_pert[row, column, id_cell_perturbed] = np.mean(prediction[(n_trials - last_n) * trial_duration:])

        
    ### save data
    with open(file_data4plot,'wb') as f:
        pickle.dump([n_trials, last_n, trial_duration, std_mean_arr, std_std_arr, prediction_before_pert, prediction_after_pert],f)  


# %% plot results (see above), each neuromod one figure

flag = 0

if flag==1:
    
    ### load data
    id_cell = 3
    input_flg = '10'
    file_data4plot = '../results/data/neuromod/test_prediction_neuromod_' + input_flg + '.pickle'
    
    with open(file_data4plot,'rb') as f:
        [_, _, _, _, _, prediction_before_pert, prediction_after_pert] = pickle.load(f) 
        
    ### initialise
    z_exp_low_unexp_high = np.zeros(3)
    z_exp_high_unexp_low = np.zeros(3)
    columns = np.array([1,0,2])
    colors = ['#19535F', '#D76A03']
    
    ### define neuromodulator (by target IN)
    if id_cell==0:
        title = 'DA activates PVs (sens)'
    elif id_cell==1:
        title = 'DA activates PVs (pred)'
    elif id_cell==2:
        title = 'NA/NE activates SOMs'
    elif id_cell==3:
        title = 'ACh or 5-HT activates VIPs'
    
    for idx, column in enumerate(columns):
            
        pred_exp_low_unexp_high_before = prediction_before_pert[1, column, id_cell]
        pred_exp_low_unexp_high_after = prediction_after_pert[1, column, id_cell]
        
        pred_exp_high_unexp_low_before = prediction_before_pert[0, column, id_cell]
        pred_exp_high_unexp_low_after = prediction_after_pert[0, column, id_cell]
        
        z_exp_low_unexp_high[idx] = (pred_exp_low_unexp_high_after - pred_exp_low_unexp_high_before) / pred_exp_low_unexp_high_before
        z_exp_high_unexp_low[idx] = (pred_exp_high_unexp_low_after - pred_exp_high_unexp_low_before) / pred_exp_high_unexp_low_before
    
    ### plot
    fig, ax = plt.subplots(1,1, tight_layout=True, figsize=(4,3))
    
    ax.plot(np.arange(3), z_exp_low_unexp_high * 100, '.-', color=colors[1])
    ax.plot(np.arange(3), z_exp_high_unexp_low * 100, '.-', color=colors[0])
    ax.axhline(0, color='k', ls=':')
    ax.set_ylim([-12, 24])
    
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['first', 'both', 'second'])
    
    ax.set_ylabel(r'change in prediction (normalised, %)')
    ax.set_xlabel('target PE circuit')
    ax.set_title(title)
    
    sns.despine(ax=ax)
    
    
# %% plot results (see above), each column one figure

flag = 0

if flag==1:
    
    ### load data
    column = 2
    input_flg = '10'
    file_data4plot = '../results/data/neuromod/test_prediction_neuromod_' + input_flg + '.pickle'
    
    with open(file_data4plot,'rb') as f:
        [_, _, _, _, _, prediction_before_pert, prediction_after_pert] = pickle.load(f) 
        
    ### initialise
    z_exp_low_unexp_high = np.zeros(4)
    z_exp_high_unexp_low = np.zeros(4)
    colors = ['#19535F', '#D76A03']
    
    for id_cell in range(4):
            
        pred_exp_low_unexp_high_before = prediction_before_pert[1, column, id_cell]
        pred_exp_low_unexp_high_after = prediction_after_pert[1, column, id_cell]
        
        pred_exp_high_unexp_low_before = prediction_before_pert[0, column, id_cell]
        pred_exp_high_unexp_low_after = prediction_after_pert[0, column, id_cell]
        
        z_exp_low_unexp_high[id_cell] = (pred_exp_low_unexp_high_after - pred_exp_low_unexp_high_before) / pred_exp_low_unexp_high_before
        z_exp_high_unexp_low[id_cell] = (pred_exp_high_unexp_low_after - pred_exp_high_unexp_low_before) / pred_exp_high_unexp_low_before
    
    ### plot
    fig, ax = plt.subplots(1,1, tight_layout=True, figsize=(5,3))
    
    X = np.arange(4)
    ax.bar(X - 0.25/2, z_exp_low_unexp_high*100, color = colors[1], width = 0.25)
    ax.bar(X + 0.25/2, z_exp_high_unexp_low*100, color = colors[0], width = 0.25)
    ax.axhline(0, color='k', ls=':')
    ax.set_ylim([-12, 24])
    
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(['DA\n[PV1]', 'DA\n[PV2]', 'NA\n[SOM]', 'ACh/\n5-HT\n[VIP]'])
    
    ax.set_ylabel('change in prediction\n(normalised, %)')
    ax.set_xlabel('neuromodulator')
    
    sns.despine(ax=ax)
    
    
# %% Systematically exploring the regimes for which the system relies more on sensory inputs or predictions after interneurons are activated

flag = 0

if flag==1:
    
    ### define target area and stimulus statistics
    column = 2      # 0: both, 1: lower level PE circuit, 2: higher level PE circuit
    std_mean = 1    # uncertainty of environement [0, 0.5, 1]
    std_std = 0     # uncertainty of stimulus [1, 0.5, 0]
    
    ### load and define parameters
    input_flg = '10'
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/neuromod/data_neuromod_column_' + str(column) + '_SDm_' + str(std_mean) + '_SDs_' + str(std_std) + '_netflg_' + input_flg + '.pickle'
    
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
    n_trials = np.int32(200)
    last_n = np.int32(50)
    trial_duration = np.int32(5000)# dtype(5000)
    n_stimuli_per_trial = np.int32(10)
    n_repeats_per_stim = np.int32(trial_duration/n_stimuli_per_trial)
    
    ### fixed values for stimulus/environement statistics
    mean_mean, min_std = dtype(3), dtype(0)
    
    ### define IN activation combinations
    nums = 10
    xp, xs = np.meshgrid(np.linspace(0, 1, nums), np.linspace(0, 1, nums)) # xp varies along the columns, xs varies along the rows
    xv = np.sqrt(1 - xp**2 - xs**2) # xv[i,j], i corresponds to xs, j corresponds to xp
    
    ### define stimuli
    np.random.seed(186)
    
    stimuli = stimuli_moments_from_uniform(n_trials, n_stimuli_per_trial, dtype(mean_mean - np.sqrt(3)*std_mean), 
                                           dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))

    stimuli = dtype(np.repeat(stimuli, n_repeats_per_stim))
    
    
    ### run model without perturbation
    print('Without neuromodulation')
    [_, _, _, _, alpha, _, _] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli)
    
    alpha_before_pert = np.mean(alpha[(n_trials - last_n) * trial_duration:])
    
    ### initialise
    alpha_after_pert = np.zeros((nums, nums))
    
    ### define target IN affected
    print('With neuromodulation')
    for i in range(nums):
        for j in range(nums):
    
            print('-- xp:', xp[i,j], 'xs:', xs[i,j], 'xv:', xv[i,j])
            
            if ~np.isnan(xv[i,j]):
        
                ## add perturbation
                perturbation = np.zeros((n_trials * trial_duration,8))                          
                perturbation[(n_trials * trial_duration)//2:, 4:6] = xp[i,j]
                perturbation[(n_trials * trial_duration)//2:, 6] = xs[i,j]
                perturbation[(n_trials * trial_duration)//2:, 7] = xv[i,j]
                fixed_input_plus_perturbation = fixed_input + perturbation
                    
                if column==1:
                    fixed_input_1 = fixed_input_plus_perturbation
                    fixed_input_2 = fixed_input
                elif column==2:
                    fixed_input_1 = fixed_input
                    fixed_input_2 = fixed_input_plus_perturbation
                elif column==0:
                    fixed_input_1 = fixed_input_plus_perturbation
                    fixed_input_2 = fixed_input_plus_perturbation
                    
                ## run model
                [_, _, _, _, alpha, _, _] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                 tc_var_per_stim, tc_var_pred, tau_pe, None, stimuli,
                                                                 fixed_input_1 = fixed_input_1, 
                                                                 fixed_input_2 = fixed_input_2)
                
                ## save alpha after neuromods
                alpha_after_pert[i, j] = np.mean(alpha[(n_trials - last_n) * trial_duration:])
            
            else:
                
                alpha_after_pert[i, j] = np.nan
                
        
    ### save data
    with open(file_data4plot,'wb') as f:
        pickle.dump([n_trials, last_n, trial_duration, xp, xs, xv, alpha_before_pert, alpha_after_pert],f)

    
# %%  Plot results from above

flag = 0

# XXXX WHY ARE XTICKS and YTICKS WRONG?

if flag==1:
    
    ### define target area and stimulus statistics
    column = 2    # 0: both, 1: lower level PE circuit, 2: higher level PE circuit
    std_mean = 0    # uncertainty of environement
    std_std = 1    # uncertainty of stimulus
    
    ### load and define parameters
    input_flg = '10'
    file_data4plot = '../results/data/neuromod/data_neuromod_column_' + str(column) + '_SDm_' + str(std_mean) + '_SDs_' + str(std_std) + '_netflg_' + input_flg + '.pickle'
    
    ### load data
    with open(file_data4plot,'rb') as f:
        [_, _, _, xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f)  
        
    ### plot data
    deviation = 100 * (alpha_after_pert - alpha_before_pert) / alpha_before_pert
    
    plt.figure()
    index = np.round(np.unique(xp),2)
    columns = np.round(np.unique(xs),2)
    data = pd.DataFrame(deviation, index=index, columns=columns)
    ax = sns.heatmap(deviation, cmap=cmap_sensory_prediction, vmin=-100, vmax=100, 
                     xticklabels=3, yticklabels=3, cbar_kws={'label': 'Change in sensory weight (%)'})
    
    ax.set_facecolor('#DBDBDB')
    ax.invert_yaxis()
    
    ax.set_xlabel(r'Fraction of PV neurons activated, x$_\mathrm{PV}$')
    ax.set_ylabel(r'Fraction of SOM neurons activated, x$_\mathrm{SOM}$')
    ax.text(5.5, 9, r'x$_\mathrm{VIP}$ = $\sqrt{1-x_\mathrm{PV}^2 - x_\mathrm{SOM}^2}$')
    ax.text(0.15, 0.1, 'VIP\nonly')
    ax.text(9.15, 0.1, 'PV\nonly')
    ax.text(0.1, 9.1, 'SOM\nonly')


# %% Plot all heatmaps together (see above)

flag = 0

if flag==1:

    columns = [1,0,2]
    
    fig, axs = plt.subplots(3,3, figsize=(8.5,6))# tight_layout=True)
    cbar_ax = fig.add_axes([.91, .3, .02, .4])
    
    for row in range(3):
        
        if row==0:
            std_mean = 1
            std_std = 0
        elif row==1:
            std_mean = 0.5
            std_std = 0.5
        elif row==2:
            std_mean = 0
            std_std = 1
        
        for c in range(3):
            
            column = columns[c]
            
            ## load data
            input_flg = '10'
            file_data4plot = '../results/data/neuromod/data_neuromod_column_' + str(column) + '_SDm_' + str(std_mean) + '_SDs_' + str(std_std) + '_netflg_' + input_flg + '.pickle'
    
            with open(file_data4plot,'rb') as f:
                [_, _, _, xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f) 
                
            ## compute deviation
            deviation = 100 * (alpha_after_pert - alpha_before_pert) / alpha_before_pert
            
            ind = np.round(np.unique(xp),2)
            col = np.round(np.unique(xs),2)
            data = pd.DataFrame(deviation, index=ind, columns=col)
            
            s = sns.heatmap(deviation, cmap=cmap_sensory_prediction, vmin=-100, vmax=100, ax=axs[row,c], 
                             xticklabels=9, yticklabels=9, cbar=row == 0, cbar_ax=None if row else cbar_ax)
            
            if ((c==0) & (row==0)):
                
                axs[row,c].text(1, 5, r'x$_\mathrm{VIP}$ = $\sqrt{1-x_\mathrm{PV}^2 - x_\mathrm{SOM}^2}$')
                axs[row,c].text(0.15, 0.1, r'$\bigstar$')
                axs[row,c].text(9.15, 0.1, r'$\clubsuit$')
                axs[row,c].text(0.1, 9.1, r'$\spadesuit$')
            
            if ((c==0) & (row==2)):
                
                axs[row,c].tick_params(left=True, bottom=True)
                axs[row,c].set_xticks([0.5,9.5])
                axs[row,c].set_yticks([0.5,9.5])
                axs[row,c].set_xticklabels([0,1])
                axs[row,c].set_yticklabels([0,1])
            else:
                axs[row,c].tick_params(left=False, bottom=False)
                axs[row,c].set_xticks([])
                axs[row,c].set_yticks([])
            
                
            if ((c==0) & (row==1)):
                
                axs[row,c].set_ylabel(r'Fraction of SOM neurons activated, x$_\mathrm{SOM}$', labelpad=20)
                
            if ((c==1) & (row==2)):
                
                axs[row,c].set_xlabel(r'Fraction of PV neurons activated, x$_\mathrm{PV}$', labelpad=20)
                
            axs[row,c].tick_params(size=2.0,pad=2.0)
            axs[row,c].set_facecolor('#DBDBDB')
            axs[row,c].invert_yaxis()
    
    cbar_ax.set_ylabel('Change in sensory weight (%)', fontsize=10, labelpad = 0)
    cbar_ax.tick_params(size=2.0,pad=2.0)
    cbar_ax.locator_params(nbins=3)
    
    fig.tight_layout(rect=[0, 0, .9, 1])
    
  
# %% Test other presentation of heatmap results

flag = 1

if flag==1:
    
    plt.figure()
    ax = plt.gca()
    
    # ax.axline((0, 0), (0.5, 1))
    # ax.axline((0.5, 1), (1, 0))
    # ax.axline((0, 0), (1, 0))
    
    column = 2
    std_mean = 0
    std_std = 1
    
    ## load data
    input_flg = '10'
    file_data4plot = '../results/data/neuromod/data_neuromod_column_' + str(column) + '_SDm_' + str(std_mean) + '_SDs_' + str(std_std) + '_netflg_' + input_flg + '.pickle'

    with open(file_data4plot,'rb') as f:
        [_, _, _, xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f) 
        
    ## compute deviation
    deviation = 100 * (alpha_after_pert - alpha_before_pert) / alpha_before_pert
    
    num_nans_per_row = np.count_nonzero(np.isnan(deviation),axis=0)
    
    for i in range(10):
        deviation[i,:] = np.roll(deviation[i,:], num_nans_per_row[i]//2)
    
    ind = np.round(np.unique(xp),2)
    col = np.round(np.unique(xs),2)
    data = pd.DataFrame(deviation, index=ind, columns=col)
    
    s = sns.heatmap(deviation, cmap=cmap_sensory_prediction, vmin=-100, vmax=100, ax=ax, 
                     xticklabels=9, yticklabels=9, cbar=row == 0, cbar_ax=None if row else cbar_ax)
    
    ax.invert_yaxis()
    
    # columns = [1,0,2]
    
    # fig, axs = plt.subplots(3,3, figsize=(8.5,6))# tight_layout=True)
    # cbar_ax = fig.add_axes([.91, .3, .02, .4])
    
    # for row in range(3):
        
    #     if row==0:
    #         std_mean = 1
    #         std_std = 0
    #     elif row==1:
    #         std_mean = 0.5
    #         std_std = 0.5
    #     elif row==2:
    #         std_mean = 0
    #         std_std = 1
        
    #     for c in range(3):
            
    #         column = columns[c]
            
    #         ## load data
    #         input_flg = '10'
    #         file_data4plot = '../results/data/neuromod/data_neuromod_column_' + str(column) + '_SDm_' + str(std_mean) + '_SDs_' + str(std_std) + '_netflg_' + input_flg + '.pickle'
    
    #         with open(file_data4plot,'rb') as f:
    #             [_, _, _, xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f) 
                
    #         ## compute deviation
    #         deviation = 100 * (alpha_after_pert - alpha_before_pert) / alpha_before_pert
            
    #         ind = np.round(np.unique(xp),2)
    #         col = np.round(np.unique(xs),2)
    #         data = pd.DataFrame(deviation, index=ind, columns=col)
            
    #         s = sns.heatmap(deviation, cmap=cmap_sensory_prediction, vmin=-100, vmax=100, ax=axs[row,c], 
    #                          xticklabels=9, yticklabels=9, cbar=row == 0, cbar_ax=None if row else cbar_ax)
            
    #         if ((c==0) & (row==0)):
                
    #             axs[row,c].text(1, 5, r'x$_\mathrm{VIP}$ = $\sqrt{1-x_\mathrm{PV}^2 - x_\mathrm{SOM}^2}$')
    #             axs[row,c].text(0.15, 0.1, r'$\bigstar$')
    #             axs[row,c].text(9.15, 0.1, r'$\clubsuit$')
    #             axs[row,c].text(0.1, 9.1, r'$\spadesuit$')
            
    #         if ((c==0) & (row==2)):
                
    #             axs[row,c].tick_params(left=True, bottom=True)
    #             axs[row,c].set_xticks([0.5,9.5])
    #             axs[row,c].set_yticks([0.5,9.5])
    #             axs[row,c].set_xticklabels([0,1])
    #             axs[row,c].set_yticklabels([0,1])
    #         else:
    #             axs[row,c].tick_params(left=False, bottom=False)
    #             axs[row,c].set_xticks([])
    #             axs[row,c].set_yticks([])
            
                
    #         if ((c==0) & (row==1)):
                
    #             axs[row,c].set_ylabel(r'Fraction of SOM neurons activated, x$_\mathrm{SOM}$', labelpad=20)
                
    #         if ((c==1) & (row==2)):
                
    #             axs[row,c].set_xlabel(r'Fraction of PV neurons activated, x$_\mathrm{PV}$', labelpad=20)
                
    #         axs[row,c].tick_params(size=2.0,pad=2.0)
    #         axs[row,c].set_facecolor('#DBDBDB')
    #         axs[row,c].invert_yaxis()
    
    # cbar_ax.set_ylabel('Change in sensory weight (%)', fontsize=10, labelpad = 0)
    # cbar_ax.tick_params(size=2.0,pad=2.0)
    # cbar_ax.locator_params(nbins=3)
    
    # fig.tight_layout(rect=[0, 0, .9, 1])    
    