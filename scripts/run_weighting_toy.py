#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:32:45 2022

@author: loreen.hertaeg
"""


# %% Import

import numpy as np
#import pickle

from src.toy_model import stimuli_moments_from_uniform, run_toy_model, default_para, alpha_parameter_exploration
from src.toy_model import random_uniform_from_moments, random_lognormal_from_moments, random_gamma_from_moments
from src.toy_model import stimuli_from_mean_and_std_arrays
from src.plot_toy_model import plot_limit_case, plot_alpha_para_exploration, plot_alpha_para_exploration_ratios
from src.plot_toy_model import plot_fraction_sensory_comparsion

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Notes on parameter explorations (and different predictions and links that can be made)

# link to psychiatric disorders: time constants and eta's
# link to experimental design: stimulus duration, stimulus distribution
# link to environment: parameterisation of distribution

# %% Toy model - limit cases

flag = 1
flg = 0

if flag==1:
    
    ### stimuli
    n_stimuli = 20
    stimulus_duration = 1000
    
    
    if flg==0:
        stimuli = stimuli_moments_from_uniform(n_stimuli, stimulus_duration, 5, 5, 1, 5) # mean 5, SD between 1 and 5
    else:
        stimuli = stimuli_moments_from_uniform(n_stimuli, stimulus_duration, 1, 5, 0, 0) # mean between 1 and 5, SD 0
  
    
    ### compute variances and predictions
    
    ## parameters
    tc_var_per_stim, eta_prediction, eta_mean_prediction, tc_var_pred = default_para()
    
    ## run model
    [prediction, variance_per_stimulus, mean_of_prediction, variance_prediction, 
     alpha, beta, weighted_output] = run_toy_model(tc_var_per_stim, eta_prediction, eta_mean_prediction, tc_var_pred, stimuli)

    
    ### plot results
    plot_limit_case(n_stimuli, stimulus_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, 
                    variance_prediction, alpha, beta, weighted_output)
    
    
# %% Parameter exploration - time constants (or its ratio) vs. eta_P/eta_PP

# I used the median, not the mean because I don't want the results to be biased by the initial period of a stimulus 
# (I am rather intersted in the steady state) --> this can of course be improved or changed in later stages

# Important: for proper figure, take the mean over several seeds!

flag = 0

flg_para = 12
flg = 0

if flag==1:
    
    ### stimuli
    last_n = 10
    n_stimuli = 30
    stimulus_duration = 1000
    
    if flg==0:
        stimuli = stimuli_moments_from_uniform(n_stimuli, stimulus_duration, 5, 5, 1, 5) # mean 5, SD between 1 and 5
    else:
        stimuli = stimuli_moments_from_uniform(n_stimuli, stimulus_duration, 1, 5, 0, 0) # mean between 1 and 5, SD 0
       
    ### default parameters and paramter ranges to be tested
    tc_var_stim, eta_prediction, eta_mean_prediction, tc_var_pred = default_para()
    
    if flg_para==12:
        #para_tested_first = np.round(np.linspace(1,1400,5),0)
        #para_tested_second = np.round(np.linspace(0.01,5,12) * eta_mean_prediction,6)
        para_tested_first = 2**np.arange(3,12)
        para_tested_second = np.round(2**np.array([-1,0,1,2,3,4,5,6,7,8], dtype=dtype) * eta_mean_prediction,6)
        ylabel = 'tau of input variance'
        xlabel = 'eta_P/eta_PP'  
        para_first_denominator = 1
        para_second_denominator = eta_mean_prediction
    elif flg_para==42:
        # para_tested_first = np.round(np.linspace(1,1400,5),0)
        # para_tested_second = np.round(np.linspace(0.01,5,12) * eta_mean_prediction,6)
        para_tested_first = 2**np.arange(3,12)
        para_tested_second = np.round(2**np.array([-1,0,1,2,3,4,5,6,7,8], dtype=dtype) * eta_mean_prediction,6)
        ylabel = 'tau of prediction variance'
        xlabel = 'eta_P/eta_PP'
        para_first_denominator = 1
        para_second_denominator = eta_mean_prediction

        
    ### run parameter exploration
    fraction_sensory_median = alpha_parameter_exploration(stimuli, stimulus_duration, last_n, para_tested_first, para_tested_second, 
                                                          tc_var_stim, eta_prediction, eta_mean_prediction, tc_var_pred, flg_para)
            
      
    ### plot
    plot_alpha_para_exploration_ratios(fraction_sensory_median, para_tested_first, para_tested_second, para_first_denominator, 
                                       para_second_denominator, 2, xlabel=xlabel, ylabel=ylabel, decimal=100)#, vmax=None)


# %% Impact of stimulus duration

flag = 0
flg = 1

if flag==1:
    
    ### stimuli durations to be tested and number of stimuli & repetitions
    # n_stim_durations = 5
    stim_durations = np.int32(np.array([5,20,100,1000]))
    n_stimuli = 200
    n_repeats = 50
    
    ### initialise array
    fraction_sensory = np.zeros((len(stim_durations), n_stimuli, n_repeats))
    
    ### test different stimuli durations
    for seed in range(n_repeats):
        for id_stim, stimulus_duration in enumerate(stim_durations):
            
            if flg==0:
                stimuli = stimuli_moments_from_uniform(n_stimuli, stimulus_duration, 5, 5, 1, 5, seed=seed) # mean 5, SD between 1 and 5
            else:
                stimuli = stimuli_moments_from_uniform(n_stimuli, stimulus_duration, 1, 5, 0, 0, seed=seed) # mean between 1 and 5, SD 0
          
            ### compute variances and predictions
            
            ## parameters
            tc_var_per_stim, eta_prediction, eta_mean_prediction, tc_var_pred = default_para()
            
            ## run model
            [_, _, _, _, alpha, _, _] = run_toy_model(tc_var_per_stim, eta_prediction, eta_mean_prediction, tc_var_pred, stimuli)
    
            ## fraction of sensory input in weighted output stored in array
            fraction_sensory[id_stim, :, seed] = np.mean(np.split(alpha,n_stimuli),1)
            
    fraction_sensory_averaged_over_seeds = np.mean(fraction_sensory,2)
    fraction_sensory_std_over_seeds = np.std(fraction_sensory,2)
    
    ### plot results
    plot_fraction_sensory_comparsion(fraction_sensory_averaged_over_seeds, fraction_sensory_std_over_seeds, n_repeats,
                                     label_text=stim_durations)

# %% Impact of different distributions for the stimuli mean
# stimuli std taken from uniform distribution

# flg = 0 ... it should not make a difference because mean is always the same!
# flg = 1 ... I would hope to see a difference

flag = 0
flg = 0

if flag==1:
    
    ### parameters
    stimulus_duration = 1000
    n_stimuli = 50
    n_repeats = 20
    
    ### define mean and std of distributions
    if flg==0: # prediction-driven
        mean_mean = 5
        sd_mean = 1e-10
        mean_sd = 3
        sd_sd = 1
    elif flg==1: # sensory driven
        mean_mean = 3
        sd_mean = 1
        mean_sd = 0
        sd_sd = 0
        
    ### initialise array
    fraction_sensory = np.zeros((4, np.int32(n_stimuli*stimulus_duration), n_repeats))
    alpha_median_per_stimulus = np.zeros((4, n_stimuli))
    distribution_names = ['normal', 'uniform', 'log-normal', 'gamma']
    
    ### run for different seeds and distribution types
    for flg_dist in range(4):
        for seed in range(n_repeats):
    
            np.random.seed(seed)
            
            ### create stimuli
            if flg_dist==0:
                mean_stimuli = np.random.normal(mean_mean, sd_mean, size=n_stimuli)
            elif flg_dist==1:
                mean_stimuli = random_uniform_from_moments(mean_mean, sd_mean, n_stimuli)
            elif flg_dist==2:
                mean_stimuli = random_lognormal_from_moments(mean_mean, sd_mean, n_stimuli)
            elif flg_dist==3:
                mean_stimuli = random_gamma_from_moments(mean_mean, sd_mean, n_stimuli)
            
            sd_stimuli = random_uniform_from_moments(mean_sd, sd_sd, n_stimuli)
            stimuli = stimuli_from_mean_and_std_arrays(mean_stimuli, sd_stimuli, stimulus_duration)
            
            ### default parameters
            tc_var_per_stim, eta_prediction, eta_mean_prediction, tc_var_pred = default_para()
            
            ### run model
            [_, _, _, _, alpha, _, _] = run_toy_model(tc_var_per_stim, eta_prediction, eta_mean_prediction, tc_var_pred, stimuli)
            
            ### fraction of sensory input in weighted output stored in array
            fraction_sensory[flg_dist, :, seed] = alpha
            alpha_median_per_stimulus[flg_dist,:] = np.median(np.split(alpha,n_stimuli),1)
            
        fraction_sensory_averaged_over_seeds = np.mean(fraction_sensory,2)
        fraction_sensory_std_over_seeds = np.std(fraction_sensory,2)
    
    ### plot results
    plot_fraction_sensory_comparsion(fraction_sensory_averaged_over_seeds, fraction_sensory_std_over_seeds, 
                                      n_repeats, cmap='hls', label_text=distribution_names)

# %% Impact of different distributions for the stimuli std
# stimuli mean taken from uniform distribution

# flg = 0 ... I would hope to see a difference
# flg = 1 ... it should not make a difference

flag = 0
flg = 0

if flag==1:
    
    ### parameters
    stimulus_duration = 1000
    n_stimuli = 50
    n_repeats = 20
    
    ### define mean and std of distributions
    if flg==0:
        mean_mean = 5
        sd_mean = 1e-10
        mean_sd = 3
        sd_sd = 1
    elif flg==1:
        mean_mean = 3
        sd_mean = 1
        mean_sd = 0
        sd_sd = 0
        
    ### initialise array
    fraction_sensory = np.zeros((4, np.int32(n_stimuli*stimulus_duration), n_repeats))
    distribution_names = ['normal', 'uniform', 'log-normal', 'gamma']
    
    ### run for different seeds and distribution types
    for flg_dist in range(4):
        for seed in range(n_repeats):
    
            np.random.seed(seed)
            
            ### create stimuli
            if flg_dist==0:
                sd_stimuli = np.random.normal(mean_sd, sd_sd, size=n_stimuli)
            elif flg_dist==1:
                sd_stimuli = random_uniform_from_moments(mean_sd, sd_sd, n_stimuli)
            elif flg_dist==2:
                sd_stimuli = random_lognormal_from_moments(mean_sd, sd_sd, n_stimuli)
            elif flg_dist==3:
                sd_stimuli = random_gamma_from_moments(mean_sd, sd_sd, n_stimuli)
            sd_stimuli[sd_stimuli<0] = 0
            
            mean_stimuli = random_uniform_from_moments(mean_mean, sd_mean, n_stimuli)
            stimuli = stimuli_from_mean_and_std_arrays(mean_stimuli, sd_stimuli, stimulus_duration)
            
            ### default parameters
            tc_var_per_stim, eta_prediction, eta_mean_prediction, tc_var_pred = default_para()
            
            ### run model
            [_, _, _, _, alpha, _, _] = run_toy_model(tc_var_per_stim, eta_prediction, eta_mean_prediction, tc_var_pred, stimuli)
            
            ### fraction of sensory input in weighted output stored in array
            fraction_sensory[flg_dist, :, seed] = alpha
            
        fraction_sensory_averaged_over_seeds = np.mean(fraction_sensory,2)
        fraction_sensory_std_over_seeds = np.std(fraction_sensory,2)
    
    ### plot results
    plot_fraction_sensory_comparsion(fraction_sensory_averaged_over_seeds, fraction_sensory_std_over_seeds, 
                                     n_repeats, cmap='hls', label_text=distribution_names)
 

# %% Test different std for mean and std of stimuli (exploration between limit cases)

# show examples from different corners
# maybe show this plot for several distribution types?

flag = 0

if flag==1:
    
    ### default parameters
    tc_var_per_stim, eta_prediction, eta_mean_prediction, tc_var_pred = default_para()
    
    mean_mean = 10
    min_std = 0
    
    last_n = 10
    n_stimuli = 20
    stimulus_duration = 1000
    n_repeats = 5
    
    ### means and std's to be tested
    std_mean_arr = np.linspace(0,5,10)
    std_std_arr = np.linspace(0,5,10)
    
    ### initialise
    fraction_sensory_median = np.zeros((len(std_std_arr),len(std_mean_arr), n_repeats))
    
    for seed in range(n_repeats):

        for col, std_mean in enumerate(std_mean_arr):
            
            for row, std_std in enumerate(std_std_arr):
        
                ### define stimuli
                stimuli = stimuli_moments_from_uniform(n_stimuli, stimulus_duration, mean_mean - np.sqrt(3)*std_mean, 
                                                       mean_mean + np.sqrt(3)*std_mean, min_std, min_std + 2*np.sqrt(3)*std_std)
                
                ### run model
                [_, _, _, _, alpha, _, _] = run_toy_model(tc_var_per_stim, eta_prediction, eta_mean_prediction, tc_var_pred, stimuli)
                
                ### median of fraction of sensory inputs over the last n stimuli
                fraction_sensory_median[row, col, seed] = np.median(alpha[(n_stimuli - last_n) * stimulus_duration:])
  
    fraction_sensory_median_averaged_over_seeds = np.mean(fraction_sensory_median,2)
    
    ### plot results
    plot_alpha_para_exploration(fraction_sensory_median_averaged_over_seeds, std_std_arr, std_mean_arr, 2, 
                                xlabel='variability across stimuli', ylabel='variability per stimulus')
    