#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:18:31 2022

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
from src.plot_results_mfn import plot_diff_heatmap, plot_transitions_examples, plot_deviation_vs_effect_size, plot_deviation_vs_PE_II

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


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
    plot_prediction(n_trials, stimuli, trial_duration, prediction, figsize=(8,5), lw=2, fs=12)  
    plot_variance(n_trials, stimuli, trial_duration, variance, figsize=(8,5), lw=2, fs=12) 
    
    
# %% MSE of estimated mean (prediction) and variance for different parameterisation of uniform distributions

### as you average for computing the steady state, there is no need to run it over several seeds
# but make sure that the range you are averaging over is fair

flag = 0

if flag==1:
    
    ### load and define parameters
    input_flg = '10' # 10, 01, 11
    
    VS, VV = int(input_flg[0]), int(input_flg[1])
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/moments/data_test_input_parameterisations_' + input_flg + '.pickle'  
      
    ### load data for plotting
    with open(file_data4plot,'rb') as f:
        [n_trials, trial_duration, mean_tested, variance_tested, 
         mse_prediction, mse_variance, diff_pred, diff_var] = pickle.load(f)

    ### plot results        
    # plot_mse_heatmap(n_trials, trial_duration, mean_tested, variance_tested, mse_prediction, 
    #                  title='MSE between running & predicted mean', vmax=15, fs=12)
    # plot_mse_heatmap(n_trials, trial_duration, mean_tested, variance_tested, mse_variance, 
    #                  title='MSE between running & predicted variance', flg=1, vmax=15, fs=12) 
    
    plot_diff_heatmap(n_trials, trial_duration, mean_tested, variance_tested, diff_pred, 
                      title='Estimating the mean', vmax=50, fs=12)
    
    plot_diff_heatmap(n_trials, trial_duration, mean_tested, variance_tested, diff_var, 
                      title='Estimating the variance', flg=1, vmax=50, fs=12)
    
 
# %% Transition examples

flag = 0
flg_plot_only = 1

# 3300 --> 3315 --> 1515 --> 1500 --> 3300

if flag==1:
    
    ### file to save data
    state_before = '1500' # 3300, 3315, 1500, 1515
    state_after = '3300' # 3300, 3315, 1500, 1515
    file_data4plot = '../results/data/weighting/data_transition_example_' + state_before + '_' + state_after + '.pickle'
     
    ### load data for plotting
    with open(file_data4plot,'rb') as f:
        [n_trials, trial_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, 
         variance_prediction, alpha, beta, weighted_output] = pickle.load(f)  
        

    plot_transitions_examples(n_trials, trial_duration, stimuli, alpha, beta, weighted_output, 
                              time_plot=0, ylim=[-15,20], xlim=[40,80])
    
    
# %% changes in variance or mean as a function of gain of nPE and pPE neurons

flag = 0

if flag==1:
    
    file_data4plot = '../results/data/perturbations/data_deviations_vs_effect_size.pickle'
        
    ### load data for plotting
    with open(file_data4plot,'rb') as f:
        [scaling_factors, dev_prediction_steady, dev_variance_steady] = pickle.load(f)
        
            
    # ### plot
    plot_deviation_vs_effect_size(scaling_factors, dev_prediction_steady, 'Bias in mean', plot_legend=False)
    plot_deviation_vs_effect_size(scaling_factors, dev_variance_steady, 'Bias in variance')
    
    
# %% Estimate deviation direction by how strongly/much a neuron is driven by sensory input or prediction

flag = 0

if flag==1:
    
    input_flgs = ['10', '01', '11']
    marker = ['o', 's', 'D']
    labels = ['MFN 1', 'MFN 2', 'MFN 3']
    moment_flg = 1 # 0 = mean, 1 = variance
    
    
    plot_deviation_vs_PE_II(moment_flg, input_flgs, marker, labels, perturbation_direction=1, 
                            plot_deviation_gradual = False, xlim=[-1,1], ylim=[-1.4,1.4], plot_inds=[4,5,6,7])
    


