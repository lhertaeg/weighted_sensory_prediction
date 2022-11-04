#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:05:25 2022

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
    fig, ax = plt.subplots(1,1, tight_layout=True, figsize=(4,3))
    
    ax.plot(np.arange(3), z_exp_low_unexp_high * 100, '.-', color=colors[1])
    ax.plot(np.arange(3), z_exp_high_unexp_low * 100, '.-', color=colors[0])
    
    ylim = ax.get_ylim()
    ax.axhspan(0, ylim[1], color=colors[1], alpha=0.1)
    ax.axhspan(ylim[0], 0, color=colors[0], alpha=0.1)
    ax.set_ylim(ylim)
    
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['first', 'both', 'second'])
    
    ax.set_ylabel(r'change in $\alpha$ (normalised, %)')
    ax.set_xlabel('target PE circuit')
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


# %% plot results (see above)

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
    

# %% How fast does the prediction change when statistics/environment changes (compare with and without neuromodulator)
# here: how fast does the prediction establish

# CONTINUE HERE XXXXX
# I changed run_mean_field_model such that number of outputs can vary (see implementation ... pretty cool and flexible)
# with that I can decide to output PE neuron activity in case I need it and use this as measure for "update speed" when mean is increased or decreased!
# (look at notes in presentation)
# (I changed the number of trials to check the implementation, don't forget to change back!)

flag = 1

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
    n_trials = np.int32(2) #np.int32(200)
    last_n = np.int32(100)
    trial_duration = np.int32(5000)# dtype(5000)
    n_stimuli_per_trial = np.int32(10)
    n_repeats_per_stim = np.int32(trial_duration/n_stimuli_per_trial)
    
    ### means and std's to be tested
    mean_mean, min_std = dtype(30), dtype(0) ###################################### !!!!!!!!!!!
    std_mean_arr = np.array([0, 3])
    std_std_arr = np.array([3, 0])
    
    ### initalise
    prediction_before_pert = np.zeros((2, 3, 4)) # 2 limit cases, target column/s, INs affected by neuromod
    prediction_after_pert = np.zeros((2, 3, 4))
    
    ### main loop
    for row in range(1):#2): # two limit cases
        
        print('Limit case: ', row)
        
        std_mean = std_mean_arr[row]
        std_std = std_std_arr[row]

        ## define stimuli
        np.random.seed(186)
        
        stimuli = stimuli_moments_from_uniform(n_trials, n_stimuli_per_trial, dtype(mean_mean - np.sqrt(3)*std_mean), 
                                           dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))
    
        stimuli = dtype(np.repeat(stimuli, n_repeats_per_stim))
        
        ## define target IN affected
        for id_cell_perturbed in range(1):#4): # 0-3: PV, PV, SOM, VIP
            id_cell_perturbed=2
            print('- Target IN id:', id_cell_perturbed)
        
            ## add perturbation
            plt.figure()
            for id_pert in range(2): ############################################ !!!!!!!!!!!!!!!
            
                if id_pert==0:
                    fixed_input_plus_perturbation = fixed_input
                else:
                    perturbation = np.zeros((n_trials * trial_duration,8))                          
                    perturbation[:, id_cell_perturbed + 4] = 1   # perturbation[(n_trials * trial_duration)//2:, id_cell_perturbed + 4] = 1         
                    fixed_input_plus_perturbation = fixed_input + perturbation
                        
    
                ## define target PE circuit
                for column in range(1):#3): # 1st or 2nd PE circuit, 0: both PE circuits
                
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
                    [prediction, _, _, _, _, _, _, a] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                          tc_var_per_stim, tc_var_pred, tau_pe, None, stimuli,
                                                                          fixed_input_1 = fixed_input_1, 
                                                                          fixed_input_2 = fixed_input_2, test=True)
                    
                    
                    running_mean = np.cumsum(prediction)/np.arange(1,len(prediction)+1)
                    #plt.plot(prediction)
                    #plt.plot(np.diff(running_mean))
                    plt.plot(running_mean)
        
                # ##  check for inf and nan
                # if ((sum(np.isinf(prediction))>0) or (sum(np.isnan(prediction))>0)):
                #     print('Warning: computation yields nan or inf in prediction.')
            
                # ## save prediction before and after neuromods
                # prediction_before_pert[row, column, id_cell_perturbed] = np.mean(prediction[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
                # prediction_after_pert[row, column, id_cell_perturbed] = np.mean(prediction[(n_trials - last_n) * trial_duration:])

        
    # ### save data
    # with open(file_data4plot,'wb') as f:
    #     pickle.dump([n_trials, last_n, trial_duration, std_mean_arr, std_std_arr, prediction_before_pert, prediction_after_pert],f) 
    

# %% plot results (see above)
