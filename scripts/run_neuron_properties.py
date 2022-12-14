#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 08:56:15 2022

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

# %% No squared activation function

# Results: 
    # var(S)<var(P): weighting with abs will lead to decaresed sensory weight
    # var(S)>var(P): weighting with abs will lead to increased sensory weight
    # In principle, the weighting in general is fine, that is, when var(S)<var(P), system is sensory driven, while when var(S)>var(P), the system is more prediction driven
    # it just seems that there is a tendency to taking the "other" more into account
    # however, it is interesting to see that at the corners, both approaches are equal, is it?

flag = 0

if flag==1:
    
    # define fixed neuron parameter
    tau = 500
    
    
    # define stimuli
    mu = 10
    var = 9
    stimuli = np.random.normal(mu, np.sqrt(var), 10000)
    
    
    # initialise 
    v_neuron_2 = np.zeros_like(stimuli)
    v_neuron = np.zeros_like(stimuli)
    
    for id_stim, s in enumerate(stimuli): 
        
        v_neuron_2[id_stim] = (1-1/tau) * v_neuron_2[id_stim-1] + (s - mu)**2/tau
        v_neuron[id_stim] = (1-1/tau) * v_neuron[id_stim-1] + abs(s - mu)/tau


    # # plot
    # plt.figure()
    # plt.plot(v_neuron_2)
    # plt.plot(v_neuron)
    
    
    ### implications for the weighting 
    # only approximately because taking abs, see above, is not equal to the std but close enough
    x = np.linspace(0.1,5,101) # s
    y = np.linspace(0.1,5,100) # p
    
    X, Y = np.meshgrid(x, y)
    
    weighting_abs = 1/(1+X/Y)
    weighting_squared = 1/(1+X**2/Y**2)
    
    f, axs = plt.subplots(1,3, figsize=(15,5))
    
    axs[0].plot(x, weighting_squared[8,:])
    axs[0].plot(x, weighting_abs[8,:])
    axs[0].set_title(str(x[8]))
    axs[0].legend(['squared', 'approx(abs)'])
    
    axs[1].plot(x, weighting_squared[50,:])
    axs[1].plot(x, weighting_abs[50,:])
    axs[1].set_title(str(x[50]))
    
    axs[2].plot(x, weighting_squared[99,:])
    axs[2].plot(x, weighting_abs[99,:])
    axs[2].set_title(str(x[99]))
    
    plt.figure()
    plt.plot(weighting_squared.flatten(), weighting_abs.flatten(), '.')
    ax = plt.gca()
    ax.axline((0.5,0.5), slope=1, ls=':', color='k')
    ax.set_xlabel('alpha (squared weighting)')
    ax.set_ylabel('alpha ("abs" weighting)')
    
    
# %% Higher task demands (expressed through increased BL activity in PE neurons) 
# out: prediction, variance and sensory weight for one BL for 3 targets

# look at O'Reilly 2012
# for cognitive load:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7116493/
    # https://www.nature.com/articles/s41598-017-07897-z
    # https://onlinelibrary.wiley.com/doi/pdf/10.1002/brb3.128
    # cognitive load --> arousal/stress --> increased BL
    
flag = 0

if flag==1:
    
    ### load and define parameters
    input_flg = '10'
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/neuro_props/test_baseline_' + input_flg + '.pickle'
    
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
    num_inpu_conds = 5
    mean_mean, min_std = dtype(3), dtype(0)
    std_mean_arr = np.linspace(0, 1, num_inpu_conds) # !!!!!!!!!!!!!!! to ensure that inputs don't get too negative and hence neurons are knocked out ...
    std_std_arr = np.linspace(0, 1, num_inpu_conds)[::-1]
    add_input = 1
    
    ### initalise
    pred_1_before, pred_1_after = np.zeros((num_inpu_conds, 3)), np.zeros((num_inpu_conds, 3))
    pred_2_before, pred_2_after = np.zeros((num_inpu_conds, 3)), np.zeros((num_inpu_conds, 3))
    var_1_before, var_1_after = np.zeros((num_inpu_conds, 3)), np.zeros((num_inpu_conds, 3))
    var_2_before, var_2_after = np.zeros((num_inpu_conds, 3)), np.zeros((num_inpu_conds, 3))
    alpha_before, alpha_after = np.zeros((num_inpu_conds, 3)), np.zeros((num_inpu_conds, 3))
    
    ### main loop
    for row in range(num_inpu_conds): # X limit cases
        
        print('Input condition: ', row)
        
        std_mean = std_mean_arr[row]
        std_std = std_std_arr[row]

        ## define stimuli
        # Please note: to make it comparable, I repeated the set of stimuli, so that before and after neuromods is driven by the same sequence of stimuli)
        np.random.seed(186)
        
        stimuli = stimuli_moments_from_uniform(n_trials//2, n_stimuli_per_trial, dtype(mean_mean - np.sqrt(3)*std_mean), 
                                               dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))
        
        stimuli = np.tile(stimuli,2)
        stimuli = dtype(np.repeat(stimuli, n_repeats_per_stim))

        ## add perturbation
        perturbation = np.zeros((n_trials * trial_duration,8))                          
        perturbation[(n_trials * trial_duration)//2:, :2] = add_input          
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
            [pred_1, var_1, pred_2, var_2, alpha, _, _] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                               tc_var_per_stim, tc_var_pred, tau_pe, None, stimuli,
                                                                               fixed_input_1 = fixed_input_1, 
                                                                               fixed_input_2 = fixed_input_2)
            
        
            ## save prediction before and after neuromods
            pred_1_before[row, column] = np.mean(pred_1[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
            pred_2_before[row, column] = np.mean(pred_2[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
            var_1_before[row, column] = np.mean(var_1[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
            var_2_before[row, column] = np.mean(var_2[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
            alpha_before[row, column] = np.mean(alpha[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
            
            pred_1_after[row, column] = np.mean(pred_1[(n_trials - last_n) * trial_duration:])
            pred_2_after[row, column] = np.mean(pred_2[(n_trials - last_n) * trial_duration:])
            var_1_after[row, column] = np.mean(var_1[(n_trials - last_n) * trial_duration:])
            var_2_after[row, column] = np.mean(var_2[(n_trials - last_n) * trial_duration:])
            alpha_after[row, column] = np.mean(alpha[(n_trials - last_n) * trial_duration:])

        
    ### save data
    with open(file_data4plot,'wb') as f:
        pickle.dump([n_trials, last_n, trial_duration, std_mean_arr, std_std_arr, pred_1_before, 
                     pred_2_before, var_1_before, var_2_before, alpha_before, pred_1_after, 
                     pred_2_after, var_1_after, var_2_after, alpha_after],f) 
        
        
# %% Plot data (see above)

flag = 0

if flag==1:
    
    ### load data
    input_flg = '10'
    file_data4plot = '../results/data/neuro_props/test_baseline_' + input_flg + '.pickle'
    
    with open(file_data4plot,'rb') as f:
        [n_trials, last_n, trial_duration, std_mean_arr, std_std_arr, pred_1_before, 
         pred_2_before, var_1_before, var_2_before, alpha_before, pred_1_after, 
         pred_2_after, var_1_after, var_2_after, alpha_after] = pickle.load(f) 
        
        
    ### plot
    titles = ['Changes in lower-level', 
             'Changes in higher-level']
    labels = ['global', 'lower', 'higher']
    colors = ['#46351D', '#CF7963', '#4AC3D3']
    ma = ['s', '<' , '>']
    ms = 8
    fs = 10
    
    for i in range(2):
    
        if i==0:
            pred_after, pred_before = pred_1_after, pred_1_before
            var_after, var_before = var_1_after, var_1_before
        elif i==1:
            pred_after, pred_before = pred_2_after, pred_2_before
            var_after, var_before = var_2_after, var_2_before
        
        f, ax = plt.subplots(1,1, figsize=(3,3), tight_layout=True)
        ax.axvline(0, ls=':', color='k', alpha=0.5)
        ax.plot((pred_after[:,0] - pred_before[:,0])/pred_before[:,0] * 100, 
                (var_after[:,0] - var_before[:,0])/var_before[:,0] * 100, label=labels[0], marker=ma[0], 
                 markeredgecolor='k', markeredgewidth=0.4, ls = "None", markersize=ms, color=colors[0])
        ax.plot((pred_after[:,1] - pred_before[:,1])/pred_before[:,1] * 100, 
                (var_after[:,1] - var_before[:,1])/var_before[:,1] * 100, label=labels[1], marker=ma[1], 
                markeredgecolor='k', markeredgewidth=0.4, ls = "None", markersize=ms, color=colors[1])
        ax.plot((pred_after[:,2] - pred_before[:,2])/pred_before[:,2] * 100, 
                (var_after[:,2] - var_before[:,2])/var_before[:,2] * 100, label=labels[2], marker=ma[2], 
                markeredgecolor='k', markeredgewidth=0.4, ls = "None", markersize=ms, color=colors[2])
        ax.set_xlim([-10,10])
        ax.set_ylabel('Variance (%)')
        ax.set_xlabel('Mean (%)')
        ax.set_ylim([-10, 450])
        ax.set_title(titles[i], fontsize=fs)
        sns.despine(ax=ax)
    
        if i==0:
            ax.legend(loc=2, title='PE circuit with \nincreased BL', framealpha=0.2, 
                      fontsize=fs-2, title_fontsize=fs-2)
    
    
    f, ax = plt.subplots(1,1, figsize=(4,3), tight_layout=True)
    ax.plot(alpha_before[:,0], alpha_after[:,0], label=labels[0], marker=ma[0],
            markeredgecolor='k', markeredgewidth=0.4, markersize=ms, color=colors[0])
    ax.plot(alpha_before[:,1], alpha_after[:,1], label=labels[1], marker=ma[1],
            markeredgecolor='k', markeredgewidth=0.4, markersize=ms, color=colors[1])
    ax.plot(alpha_before[:,2], alpha_after[:,2], label=labels[2], marker=ma[2],
            markeredgecolor='k', markeredgewidth=0.4, markersize=ms, color=colors[2])
    ax.axline((0.5,0.5), slope=1, color='k', ls=':')
    ax.set_ylim([0,1])
    
    ax.set_title('Changes in weighting', fontsize=fs)
    ax.set_ylabel('Sensory weight (BL increased)')
    ax.set_xlabel('Sensory weight (BL zero)')
    sns.despine(ax=ax)
    
    
# %% Higher task demands (expressed through increased BL activity in PE neurons) 
# out: sensory weight for several different task demands (task complexity => BL activity), PE neurons in both columns affected

flag = 0

if flag==1:
    
    ### load and define parameters
    input_flg = '10'
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/neuro_props/test_task_complexity_' + input_flg + '.pickle'
    
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
    num_inpu_conds = 5
    mean_mean, min_std = dtype(3), dtype(0)
    std_mean_arr = np.linspace(0, 1, num_inpu_conds) # !!!!!!!!!!!!!!! to ensure that inputs don't get too negative and hence neurons are knocked out ...
    std_std_arr = np.linspace(0, 1, num_inpu_conds)[::-1]
    
    ### different BLs
    add_inputs = np.array([0.5, 1, 1.5, 2])
    num_inputs = len(add_inputs)
    
    ###initalise
    alpha_before, alpha_after = np.zeros((num_inpu_conds, num_inputs)), np.zeros((num_inpu_conds, num_inputs))
    
    ### main loop
    for row in range(num_inpu_conds): # input conditions
        
        print('Input condition: ', row)
        
        std_mean = std_mean_arr[row]
        std_std = std_std_arr[row]

        ## define stimuli
        # Please note: to make it comparable, I repeated the set of stimuli, so that before and after neuromods is driven by the same sequence of stimuli)
        np.random.seed(186)
        
        stimuli = stimuli_moments_from_uniform(n_trials//2, n_stimuli_per_trial, dtype(mean_mean - np.sqrt(3)*std_mean), 
                                               dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))
    
        stimuli = np.tile(stimuli,2)
        stimuli = dtype(np.repeat(stimuli, n_repeats_per_stim))

        ## add perturbation
        for column, add_input in enumerate(add_inputs):
            
            print('-- Additional inp to increase BL: ', add_input)
            
            perturbation = np.zeros((n_trials * trial_duration,8))                          
            perturbation[(n_trials * trial_duration)//2:, :2] = add_input          
            fixed_input_plus_perturbation = fixed_input + perturbation

            ## define target PE circuit
            fixed_input_1 = fixed_input_plus_perturbation
            fixed_input_2 = fixed_input_plus_perturbation
                
            ## run model
            [pred_1, var_1, pred_2, var_2, alpha, _, _] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                               tc_var_per_stim, tc_var_pred, tau_pe, None, stimuli,
                                                                               fixed_input_1 = fixed_input_1, 
                                                                               fixed_input_2 = fixed_input_2)
            
        
            ## save prediction before and after neuromods
            alpha_before[row, column] = np.mean(alpha[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
            alpha_after[row, column] = np.mean(alpha[(n_trials - last_n) * trial_duration:])

        
    ### save data
    with open(file_data4plot,'wb') as f:
        pickle.dump([n_trials, last_n, trial_duration, std_mean_arr, std_std_arr, 
                     add_inputs, alpha_before, alpha_after],f) 
     
        
# %% plot results for different BL activities (different cognitive loads)
# either from cell above, or from below

flag = 0

if flag==1:
    
    ### load and define parameters
    input_flg = '10'
    #file_data4plot = '../results/data/neuro_props/test_task_complexity_' + input_flg + '.pickle'
    file_data4plot = '../results/data/neuro_props/test_task_complexity_all_neurons_' + input_flg + '.pickle'
    
    with open(file_data4plot,'rb') as f:
        [_, _, _, _, _, add_inputs, alpha_before, alpha_after] = pickle.load(f)
        
    ### plotting
    ms = 8
    fs = 10
    colors = sns.color_palette("viridis_r", n_colors=len(add_inputs))
    f, ax = plt.subplots(1,1, figsize=(4.5,3), tight_layout=True)
    
    for i in range(len(add_inputs)):
        
        ax.plot(alpha_before[:,i], alpha_after[:,i], marker='s', markeredgecolor='k', 
                markeredgewidth=0.4, markersize=ms, color=colors[i], label=str(add_inputs[i]))
     
    ax.axline((0.5,0.5), slope=1, color='k', ls=':')
    #ax.set_ylim([0,1])
    
    axins1 = inset_axes(ax, width="30%", height="5%", loc=2)
    
    cmap = ListedColormap(colors)
    cb = mpl.colorbar.ColorbarBase(axins1, cmap=cmap, orientation='horizontal', ticks=[0.1,0.9])
    cb.outline.set_visible(False)
    cb.ax.set_title('Baseline', fontsize=fs, pad = 0)
    cb.ax.set_xticklabels(['low', 'high'], fontsize=fs)
    axins1.xaxis.set_ticks_position("bottom")
    axins1.tick_params(size=2.0,pad=2.0)
    
    ax.set_ylabel('Sensory weight \n(baseline increased)')
    ax.set_xlabel('Sensory weight (control)')
    sns.despine(ax=ax)
    
    
# %% Higher task demands (expressed through increased BL activity of ALL neurons) 
# out: sensory weight for several different task demands (task complexity => BL activity), PE neurons in both columns affected

flag = 0

if flag==1:
    
    ### load and define parameters
    input_flg = '10'
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/neuro_props/test_task_complexity_all_neurons_' + input_flg + '.pickle'
    
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
    trial_duration = np.int32(5000) # dtype(5000)
    n_stimuli_per_trial = np.int32(10)
    n_repeats_per_stim = np.int32(trial_duration/n_stimuli_per_trial)
    
    ### means and std's to be tested
    num_inpu_conds = 5
    mean_mean, min_std = dtype(3), dtype(0)
    std_mean_arr = np.linspace(0, 1, num_inpu_conds) # !!!!!!!!!!!!!!! to ensure that inputs don't get too negative and hence neurons are knocked out ...
    std_std_arr = np.linspace(0, 1, num_inpu_conds)[::-1]
    
    ### different BLs
    add_inputs = np.array([0.5, 1, 1.5, 2])
    num_inputs = len(add_inputs)
    
    ### initalise
    alpha_before, alpha_after = np.zeros((num_inpu_conds, num_inputs)), np.zeros((num_inpu_conds, num_inputs))
    
    ### main loop
    for row in range(num_inpu_conds): # input conditions
        
        print('Input condition: ', row)
        
        std_mean = std_mean_arr[row]
        std_std = std_std_arr[row]

        ## define stimuli
        # Please note: to make it comparable, I repeated the set of stimuli, so that before and after neuromods is driven by the same sequence of stimuli)
        np.random.seed(186)
        
        stimuli = stimuli_moments_from_uniform(n_trials//2, n_stimuli_per_trial, dtype(mean_mean - np.sqrt(3)*std_mean), 
                                               dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))
    
        stimuli = np.tile(stimuli,2)
        stimuli = dtype(np.repeat(stimuli, n_repeats_per_stim))

        ## add perturbation
        for column, add_input in enumerate(add_inputs):
            
            print('-- Additional inp to increase BL: ', add_input)
            
            # increase BL in all neurons (not in dendrites, for simplicity)
            perturbation = np.zeros((n_trials * trial_duration,8))                          
            perturbation[(n_trials * trial_duration)//2:, :2] = add_input  
            perturbation[(n_trials * trial_duration)//2:, 4:] = add_input 
            fixed_input_plus_perturbation = fixed_input + perturbation

            ## define target PE circuit
            fixed_input_1 = fixed_input_plus_perturbation
            fixed_input_2 = fixed_input_plus_perturbation
                
            ## run model
            [pred_1, var_1, pred_2, var_2, alpha, _, _] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                               tc_var_per_stim, tc_var_pred, tau_pe, None, stimuli,
                                                                               fixed_input_1 = fixed_input_1, 
                                                                               fixed_input_2 = fixed_input_2)
            
        
            ## save prediction before and after neuromods
            alpha_before[row, column] = np.mean(alpha[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
            alpha_after[row, column] = np.mean(alpha[(n_trials - last_n) * trial_duration:])

        
    ### save data
    with open(file_data4plot,'wb') as f:
        pickle.dump([n_trials, last_n, trial_duration, std_mean_arr, std_std_arr, 
                     add_inputs, alpha_before, alpha_after],f) 
     

# %% Test different "time constants" for updating prediction/memory neuron in both PE circuits
# basically 

flag = 0
flg_plot_only = 1

if flag==1:
    
    ### file to save data
    file_data4plot = '../results/data/neuro_props/data_when_weighting_goes_wrong.pickle'
    
    if flg_plot_only==0:
        
        ### load and define parameters
        input_flg = '10'
        filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
        
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
        
        ### means and std's to be tested
        num_conditions = 5
        mean_mean, min_std = dtype(3), dtype(0)
        std_mean_arr = np.linspace(0,3, num_conditions, dtype=dtype)
        std_std_arr = np.linspace(0,3, num_conditions, dtype=dtype)[::-1]
        
        ### parameters to be tested
        sv = np.linspace(1,6,6)
        
        ### initialise
        fraction_sensory_mean = np.zeros((num_conditions, len(sv)), dtype=dtype)
        
        for l in range(num_conditions):
            
            std_mean = std_mean_arr[l]
            std_std = std_std_arr[l]
                
            ### display progress
            print(l)
    
            ### define stimuli
            stimuli = stimuli_moments_from_uniform(n_trials, n_stimuli_per_trial, dtype(mean_mean - np.sqrt(3)*std_mean), 
                                                   dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))
            
            stimuli = dtype(np.repeat(stimuli, n_repeats_per_stim))
            
            for k, s in enumerate(sv):
                
                w_PE_to_P_scaled = np.copy(w_PE_to_P)
                v_PE_to_P_scaled = np.copy(v_PE_to_P)
                
                w_PE_to_P_scaled[0,:] /= s 
                v_PE_to_P_scaled[0,:] *= s
                
                ### run model
                [_, _, _, _, alpha, _, 
                 weighted_output] = run_mean_field_model(w_PE_to_P_scaled, w_P_to_PE, w_PE_to_PE, v_PE_to_P_scaled, v_P_to_PE, v_PE_to_PE, 
                                                         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli)
                
                ### fraction of sensory input in weighted output
                fraction_sensory_mean[l, k] = np.mean(alpha[(n_trials - last_n) * trial_duration:])
 
    
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([fraction_sensory_mean, std_std_arr, std_mean_arr, sv],f) 
     
    else:
        
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [fraction_sensory_mean, std_std_arr, std_mean_arr, sv] = pickle.load(f)
            
    ### plot
    fs = 10
    
    f, ax = plt.subplots(1,1, figsize=(5,4), tight_layout=True)
    colors = sns.color_palette("viridis_r", n_colors=np.size(fraction_sensory_mean,1)-1)
    
    for i in range(np.size(fraction_sensory_mean,1)-1):
        ax.plot(fraction_sensory_mean[:,0], fraction_sensory_mean[:,i+1],  marker='s', color=colors[i],
                markeredgecolor='k', markeredgewidth=0.4)
        
    ax.axline((0.5,0.5), slope=1, color='k', ls=':')
    
    ax.set_xlabel('Sensory weight (ctrl)')
    ax.set_ylabel('Sensory weight \n(parameters modulated)')
    
    axins1 = inset_axes(ax, width="30%", height="5%", loc=2)
    
    cmap = ListedColormap(colors)
    cb = mpl.colorbar.ColorbarBase(axins1, cmap=cmap, orientation='horizontal', ticks=[0.1,0.9])
    cb.outline.set_visible(False)
    cb.ax.set_title('Scaling', fontsize=fs, pad = 0)
    cb.ax.set_xticklabels([str(sv[1]**2), str(sv[-1]**2)], fontsize=fs)
    axins1.xaxis.set_ticks_position("bottom")
    axins1.tick_params(size=2.0,pad=2.0)
    
    sns.despine(ax=ax)    

        