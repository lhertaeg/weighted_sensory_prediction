#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:27:24 2022

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

from src.plot_results_mfn import plot_prediction, plot_variance, plot_mse, plot_manipulations, plot_mse_heatmap, plot_limit_case_example
from src.plot_results_mfn import plot_diff_heatmap, plot_transitions_examples, plot_deviation_vs_effect_size, plot_deviation_vs_PE_II

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

inch = 2.54

# %% Example: PE neurons can estimate/establish the mean and variance of a normal random variable 
# mean and varaince are drawn from a unimodal distribution

flag = 0

if flag==1:
    
    ### define/load parameters
    input_flg = '10' # 10, 01, 11
    
    VS, VV = int(input_flg[0]), int(input_flg[1])
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/moments/Data_example_MFN_' + input_flg + '.pickle'
    
    ### load data for plotting
    with open(file_data4plot,'rb') as f:
        [n_trials, trial_duration, stimuli, prediction, variance] = pickle.load(f)  
           
    ### plot prediction and variance (in comparison to the data) 
    fig_size = (6 / inch, 3 / inch)
    plot_prediction(n_trials, stimuli, trial_duration, prediction, figsize=fig_size, lw=1, fs=7, 
                    legend_flg=False, mse_flg=False)  
    plot_variance(n_trials, stimuli, trial_duration, variance, figsize=fig_size, lw=1, fs=7, 
                  legend_flg=False, mse_flg=False)
    
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
        title = 'NA activates SOMs'
    elif id_cell==3:
        title = 'ACh activates VIPs'
    
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
    fig_size = (6 / inch, 4 / inch)
    fs = 7
    lw = 1
    ms = 4
    
    fig, ax = plt.subplots(1,1, tight_layout=True, figsize=fig_size)
    marker = ['<', 's', '>']
    text = ['low-level PE', 'both PE', 'high-level PE']
    
    for i in range(3):
        ax.plot(1, z_exp_low_unexp_high[i] * 100, marker=marker[i], color=colors[1], markersize=ms, markeredgecolor='k', markeredgewidth=0.4)
        ax.plot(2, z_exp_high_unexp_low[i] * 100, marker=marker[i], color=colors[0], markersize=ms, markeredgecolor='k', markeredgewidth=0.4)
        ax.plot(np.nan, np.nan, marker=marker[i], color='k', label=text[i], ls='None', markersize=ms-2)
    
    ylim = ax.get_ylim()
    ax.axhspan(0, ylim[1], color=colors[1], alpha=0.1)
    ax.axhspan(ylim[0], 0, color=colors[0], alpha=0.1)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', labelsize=fs)
    
    ax.set_xticks([1,2])
    ax.set_xticklabels(['Unpredictable, \nreliable stimuli', 'Predictable, \nnoisy stimuli'])
    ax.set_xlim([0.5,2.5])
    
    ax.set_ylabel('Change in weight \n(normalized, %)', fontsize=fs)
    ax.legend(loc=0, framealpha=0.5, fontsize=fs, labelspacing = 0.1)
    ax.set_title(title, fontsize=fs)
    
    sns.despine(ax=ax)
    
# %% Estimate deviation direction by how strongly/much a neuron is driven by sensory input or prediction

flag = 1

if flag==1:
    
    input_flgs = ['10', '01', '11']
    marker = ['o', 's', 'D']
    labels = ['MFN 1', 'MFN 2', 'MFN 3']
    moment_flg = 1 # 0 = mean, 1 = variance
    
    fig_size = (6 / inch, 4 / inch)
    fs = 7
    lw = 1
    ms = 6
    
    plot_deviation_vs_PE_II(moment_flg, input_flgs, marker, labels, perturbation_direction=1, 
                            plot_deviation_gradual = False, figsize=fig_size, fontsize=fs, markersize=ms,
                            linewidth=lw, legend_II_flg=False, xlim=[-1,1], ylim=[-1.4,1.4], plot_inds=[4,5,6,7])
    
# %% Summary ... extrapolate between cases 

flag = 0

if flag==1:
    
    ### file to save data
    file_data4plot = '../results/data/weighting/data_weighting_heatmap.pickle'
        
    ### load data for plotting
    with open(file_data4plot,'rb') as f:
        [fraction_sensory_mean, std_std_arr, std_mean_arr, weighted_out] = pickle.load(f)
        
    ### average over seeds
    fraction_sensory_mean_averaged_over_seeds = np.mean(fraction_sensory_mean,2)
    
    ### plot results
    fig_size = (5.5 / inch, 3.5 / inch)
    fs = 7
    
    plot_alpha_para_exploration(fraction_sensory_mean_averaged_over_seeds, std_std_arr, std_mean_arr, 2, 
                                xlabel='Var. across trials', ylabel='Variance \nwithin trial',
                                figsize=fig_size, fs=fs)
    
    
# %% limit cases

flag = 0
flg_limit_case = 1 # 0 = mean the same, std large; 1 = mean varies, std = 0

if flag==1:
    
    ### file to save data
    file_data4plot = '../results/data/weighting/data_example_limit_case_' + str(flg_limit_case) + '.pickle'
    
    ### load data for plotting
    with open(file_data4plot,'rb') as f:
        [n_trials, trial_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, 
         variance_prediction, alpha, beta, weighted_output] = pickle.load(f)  
        
    ### XXX
    fig_size = (4.5 / inch, 3.5 / inch)
    fs = 7
    
    ### plot results
    if flg_limit_case==0:
        plot_limit_case_example(n_trials, trial_duration, stimuli, prediction, mean_of_prediction, 
                                variance_per_stimulus, variance_prediction, alpha, beta, weighted_output,
                                flg_fraction_only=True, figsize=fig_size, fs=fs)
    else:
        plot_limit_case_example(n_trials, trial_duration, stimuli, prediction, mean_of_prediction, 
                                variance_per_stimulus, variance_prediction, alpha, beta, weighted_output, 
                                plot_legend=False, flg_fraction_only=True, figsize=fig_size, fs=fs)


# %% Transitions

flag = 0

# 3300 --> 3315 --> 1515 --> 1500 --> 3300

if flag==1:
    
    ### file to save data
    state_before = '1515' # 3300, 3315, 1500, 1515
    state_after = '1500' # 3300, 3315, 1500, 1515
    file_data4plot = '../results/data/weighting/data_transition_example_' + state_before + '_' + state_after + '.pickle'
     
    ### load data for plotting
    with open(file_data4plot,'rb') as f:
        [n_trials, trial_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, 
         variance_prediction, alpha, beta, weighted_output] = pickle.load(f)  
        
    ### XXX
    fig_size = (3 / inch, 3 / inch)
    fs = 7

    plot_transitions_examples(n_trials, trial_duration, stimuli, alpha, beta, weighted_output, 
                              time_plot=0, ylim=[-15,20], xlim=[55,65], figsize=fig_size, 
                              plot_only_weights=True, fs=fs, plot_ylable=False)