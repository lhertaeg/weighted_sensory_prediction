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
                                

# %% plot results from perturbation experiments (see above)

flag = 1

if flag==1:
     
    input_flgs = ['10', '01', '11']
    
    for j in range(2):
            
        fig, (ax, ax_cbar) = plt.subplots(2,1, gridspec_kw={'height_ratios': [20, 1]})
    
        for k, input_flg in enumerate(input_flgs):
        
            ### load data
            file_data4plot = '../results/data/weighting_perturbation/data_weighting_perturbations_' + input_flg + '.pickle'
            
            with open(file_data4plot,'rb') as f:
                n_trials, last_n, trial_duration, frac_sens_before_pert, frac_sens_after_pert = pickle.load(f)  
            
            
            ### colors
            colors = ['#508FCE', '#2B6299', '#79AFB9', '#39656D']
            marker = ['o', 's', 'D']
            labels = ['MFN 1', 'MFN 2', 'MFN 3']
    
            
            for i in range(4):
                
                ax.plot(frac_sens_after_pert[j,i,:] - frac_sens_before_pert[j,i,:], '-', 
                         color=colors[i], lw=1, marker=marker[k])
            
            cmap = ListedColormap([cmap_sensory_prediction(0), 
                                   cmap_sensory_prediction(0.25),
                                   cmap_sensory_prediction(0.5),
                                   cmap_sensory_prediction(0.75),
                                   cmap_sensory_prediction(0.99)])
            cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, orientation='horizontal', 
                                      ticks=[0.15,0.85], label='Input regime')
            cbar.ax.set_xticklabels(['var(S)>>var(P)', 'var(S)<<var(P)'])  # vertically oriented colorbar
            
            ax.axhline(0,color='k', ls='-', alpha=0.1, zorder=0)
            ax.axhspan(0,1, color=color_sensory, alpha=0.05)
            ax.axhspan(0,-1, color=color_prediction, alpha=0.05)
            ax.set_ylim([-0.15,0.15]) # can be between -1 and 1 (in the end we should maybe show range between -1 and 1)
            #ax.set_ylim([-1,1])
            ax.set_xticks([0,1,2,3,4])
            ax.set_xticklabels([])#'P-driven', 'S-driven'])
            
            
            if j==0:
                ax.set_title('Inhibitory perturbation')
            elif j==1:
                ax.set_title('Excitatory perturbation')
                
            ax.set_ylabel(r'$\Delta$ fraction$_\mathrm{sens}$ (after - before)')
            
            sns.despine(ax=ax)
            
            
# %% Example transition with and without perturbation

#### Very crude just the first tests

flag = 0
flg_plot_only = False

# 3300 --> 3315 --> 1515 --> 1500 --> 3300

if flag==1:
    
    ### file to save data
    state_before = '3315' # 3300, 3315, 1500, 1515
    state_after = '3300' # 3300, 3315, 1500, 1515
    # file_data4plot = '../results/data/weighting/data_transition_example_' + state_before + '_' + state_after + '.pickle'
    
    if not flg_plot_only:
        
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
        
        ### potential perturbation
        id_cell_perturbed = 7
        perturbation_strength = 1 # 0
        perturbation = np.zeros((n_trials * trial_duration,8))                          
        perturbation[(n_trials * trial_duration)//2:, id_cell_perturbed] = perturbation_strength  
        #perturbation[:, id_cell_perturbed] = perturbation_strength          
        fixed_input_plus_perturbation = fixed_input + perturbation  
        
        
        ### initialise
        window_size = np.linspace(0,20,81)
        fraction_without_pert = np.zeros(len(window_size))
        fraction_with_pert = np.zeros(len(window_size))
        
        
        ### run model without perturbation
        [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
          alpha, beta, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                              tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli,
                                                              w_PE_to_V = w_PE_to_V, v_PE_to_V = v_PE_to_V)
                                                               
        for k, span in enumerate(window_size):  
            if span==0:
                fraction_without_pert[k] = np.mean(alpha[((n_trials//2 - 5) * trial_duration):int((n_trials//2) * trial_duration)])   
            else:                                   
                fraction_without_pert[k] = np.mean(alpha[(n_trials//2 * trial_duration):int((n_trials//2+span) * trial_duration)])                                                 
                                      

         ### run model with perturbation
        [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
          alpha, beta, weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                              tc_var_per_stim, tc_var_pred, tau_pe, fixed_input_plus_perturbation, stimuli,
                                                              w_PE_to_V = w_PE_to_V, v_PE_to_V = v_PE_to_V)
                                                               
        for k, span in enumerate(window_size):  
            if span==0:
                fraction_with_pert[k] = np.mean(alpha[((n_trials//2 - 5) * trial_duration):int((n_trials//2) * trial_duration)])   
            else:                                   
                fraction_with_pert[k] = np.mean(alpha[(n_trials//2 * trial_duration):int((n_trials//2+span) * trial_duration)])                                                 
                                      
                                                               
        # ### save data for later
        # with open(file_data4plot,'wb') as f:
        #     pickle.dump([n_trials, trial_duration, stimuli, prediction, mean_of_prediction, 
        #                  variance_per_stimulus, variance_prediction, alpha, beta, weighted_output],f) 
                                                       
    else:
        
        # ### load data for plotting
        # with open(file_data4plot,'rb') as f:
        #     [n_trials, trial_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, 
        #      variance_prediction, alpha, beta, weighted_output] = pickle.load(f)  
        print('')
        

    ### plot results  
    plt.figure()
    ax = plt.gca()
    ax.plot(window_size, fraction_without_pert, linestyle='--', color='k')
    ax.plot(window_size, fraction_with_pert, linestyle='--', color='r')
    
    mid_without = (fraction_without_pert[-1]+fraction_without_pert[0])/2
    mid_with = (fraction_with_pert[-1]+fraction_with_pert[0])/2
    
    time_half_without = window_size[np.where(fraction_without_pert>mid_without)[0][0]]
    time_half_with = window_size[np.where(fraction_with_pert>mid_with)[0][0]]
    
    dt = (window_size[1]-window_size[0])/2
    ax.plot(time_half_without - dt, mid_without, marker='*', color='k', ms=10)
    ax.plot(time_half_with - dt, mid_with, color='r', marker='*', ms=10)
    