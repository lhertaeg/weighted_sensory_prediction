#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
from src.functions_simulate import simulate_weighting_example


import matplotlib.pyplot as plt



# %% Different mean across trials, same SD, smaller vs. larger mean range

run_cell = False
#plot_only = False

if run_cell: 
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### shorter mean range
    min_mean, max_mean, min_std, max_std = 20, 25, 5, 5
    
    [n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
     v_neuron_higher, alpha, beta, weighted_output] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                 min_std, max_std, n_trials=n_trials,
                                                                                 file_for_data = None)
                                                                                    
    trials_sensory = np.mean(np.split(stimuli, n_trials),1)
    trials_estimated = np.mean(np.split(weighted_output, n_trials),1)                                                                              
    

    plt.figure()
    plt.plot(trials_sensory, trials_estimated, 'o', alpha = 0.2, color='b')
    ax = plt.gca()
    ax.axline((np.mean(stimuli), np.mean(stimuli)), slope=1, color='k', ls=':')
    
    m_1, n =  np.polyfit(trials_sensory, trials_estimated, 1)
    ax.plot(trials_sensory, m_1 * trials_sensory + n, '--', color='b')
    
    ### larger mean range
    min_mean, max_mean, min_std, max_std = 15, 30, 5, 5
    
    [n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
     v_neuron_higher, alpha, beta, weighted_output] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                 min_std, max_std, n_trials=n_trials,
                                                                                 file_for_data = None)
                                                                                 
    trials_sensory = np.mean(np.split(stimuli, n_trials),1)
    trials_estimated = np.mean(np.split(weighted_output, n_trials),1)                                                                              
    
    ax.plot(trials_sensory, trials_estimated, 'o', alpha = 0.2, color='r')
    
    m_2, n =  np.polyfit(trials_sensory, trials_estimated, 1)
    ax.plot(trials_sensory, m_2 * trials_sensory + n, '--', color='r')


# %% Different mean across trials, same SD, mean smaller vs. larger

run_cell = False
#plot_only = False

if run_cell: 
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### larger mean
    min_mean, max_mean, min_std, max_std = 20, 25, 5, 5
    
    [n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
     v_neuron_higher, alpha, beta, weighted_output] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                 min_std, max_std, n_trials=n_trials,
                                                                                 file_for_data = None)
                                                                                    
    trials_sensory = np.mean(np.split(stimuli, n_trials),1)
    trials_estimated = np.mean(np.split(weighted_output, n_trials),1)                                                                              
    

    plt.figure()
    plt.plot(trials_sensory, trials_estimated, 'o', alpha = 0.2, color='b')
    ax = plt.gca()
    ax.axline((np.mean(stimuli), np.mean(stimuli)), slope=1, color='k', ls=':')
    
    m_1, n =  np.polyfit(trials_sensory, trials_estimated, 1)
    ax.plot(trials_sensory, m_1 * trials_sensory + n, '--', color='b')
    
    ### smaller mean
    min_mean, max_mean, min_std, max_std = 10, 15, 5, 5
    
    [n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
     v_neuron_higher, alpha, beta, weighted_output] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                 min_std, max_std, n_trials=n_trials,
                                                                                 file_for_data = None)
                                                                                 
    trials_sensory = np.mean(np.split(stimuli, n_trials),1)
    trials_estimated = np.mean(np.split(weighted_output, n_trials),1)                                                                              
    
    ax.plot(trials_sensory, trials_estimated, 'o', alpha = 0.2, color='r')
    
    m_2, n =  np.polyfit(trials_sensory, trials_estimated, 1)
    ax.plot(trials_sensory, m_2 * trials_sensory + n, '--', color='r')



# %% Same mean across trials, small and large SD

run_cell = False
#plot_only = False

if run_cell: 
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### large SD
    min_mean, max_mean, min_std, max_std = 15, 15, 5, 5
    
    [n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
     v_neuron_higher, alpha, beta, weighted_output] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                 min_std, max_std, n_trials=n_trials,
                                                                                 file_for_data = None)
                                                                                    
    trials_sensory = np.mean(np.split(stimuli, n_trials),1)
    trials_estimated = np.mean(np.split(weighted_output, n_trials),1)                                                                              
    

    plt.figure()
    plt.plot(trials_sensory, trials_estimated, 'o', alpha = 0.2, color='b')
    ax = plt.gca()
    ax.axline((np.mean(stimuli), np.mean(stimuli)), slope=1, color='k', ls=':')
    
    m_1, n =  np.polyfit(trials_sensory, trials_estimated, 1)
    ax.plot(trials_sensory, m_1 * trials_sensory + n, '--', color='b')
    
    ### small SD
    min_mean, max_mean, min_std, max_std = 15, 15, 0, 0
    
    [n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
     v_neuron_higher, alpha, beta, weighted_output] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                 min_std, max_std, n_trials=n_trials,
                                                                                 file_for_data = None)
                                                                                 
    trials_sensory = np.mean(np.split(stimuli, n_trials),1)
    trials_estimated = np.mean(np.split(weighted_output, n_trials),1)                                                                              
    
    ax.plot(trials_sensory, trials_estimated, 'o', alpha = 0.2, color='r')
    
    m_2, n =  np.polyfit(trials_sensory, trials_estimated, 1)
    ax.plot(trials_sensory, m_2 * trials_sensory + n, '--', color='r')


# %% Different mean across trials, small and large SD

run_cell = True
#plot_only = False

if run_cell: 
    
    n_trials = 200
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### large SD
    min_mean, max_mean, min_std, max_std = 10, 15, 5, 5
    
    [n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
     v_neuron_higher, alpha, beta, weighted_output] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                 min_std, max_std, n_trials=n_trials,
                                                                                 file_for_data = None)
                                                                                    
    trials_sensory = np.mean(np.split(stimuli, n_trials),1)
    trials_estimated = np.mean(np.split(weighted_output, n_trials),1)                                                                              
    

    plt.figure()
    plt.plot(trials_sensory, trials_estimated, 'o', alpha = 0.2, color='b')
    ax = plt.gca()
    ax.axline((np.mean(stimuli), np.mean(stimuli)), slope=1, color='k', ls=':')
    
    m_1, n =  np.polyfit(trials_sensory, trials_estimated, 1)
    ax.plot(trials_sensory, m_1 * trials_sensory + n, '--', color='b')
    
    ### small SD
    min_mean, max_mean, min_std, max_std = 10, 15, 0, 0
    
    [n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
     v_neuron_higher, alpha, beta, weighted_output] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                 min_std, max_std, n_trials=n_trials,
                                                                                 file_for_data = None)
                                                                                 
    trials_sensory = np.mean(np.split(stimuli, n_trials),1)
    trials_estimated = np.mean(np.split(weighted_output, n_trials),1)                                                                              
    
    ax.plot(trials_sensory, trials_estimated, 'o', alpha = 0.2, color='r')
    
    m_2, n =  np.polyfit(trials_sensory, trials_estimated, 1)
    ax.plot(trials_sensory, m_2 * trials_sensory + n, '--', color='r')
    

# %% test first ideas

run_cell = False
#plot_only = False

if run_cell: 
    
    n_trials = 100
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    #file_for_data = '../results/data/weighting/data_example_limit_case_prediction_driven.pickle'
    
    min_mean, max_mean, min_std, max_std = 10, 15, 5, 5
    
    [n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
     v_neuron_higher, alpha, beta, weighted_output] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                 min_std, max_std, n_trials=n_trials,
                                                                                 file_for_data = None)
    # compute trail mean (real and estimated)                                                                                    
    trials_sensory = np.mean(np.split(stimuli, n_trials),1)
    trials_estimated = np.mean(np.split(weighted_output, n_trials),1)                                                                              
    
    # plot
    plt.figure()
    plt.plot(trials_sensory, trials_estimated, 'o', alpha = 0.2, color='b')
    ax = plt.gca()
    ax.axline((np.mean(stimuli), np.mean(stimuli)), slope=1, color='k', ls=':')
    
    m, n =  np.polyfit(trials_sensory, trials_estimated, 1)
    ax.plot(trials_sensory, m * trials_sensory + n, '--', color='b')
    
    # contraction bias reproduceable in the current setting
    # fitting a line shows that nicely
    # however, not sure if I can see bias(a) < bias (b) (rather not) and SD(a) < SD(b)
    # however, we can for sure increase std and then see how this affects bias etc. => testable prediction?
    
    
    min_mean, max_mean, min_std, max_std = 10, 15, 1, 1
    
    [n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
     v_neuron_higher, alpha, beta, weighted_output] = simulate_weighting_example(mfn_flag, min_mean, max_mean, 
                                                                                 min_std, max_std, n_trials=n_trials,
                                                                                 file_for_data = None)
                                                                                 
    trials_sensory = np.mean(np.split(stimuli, n_trials),1)
    trials_estimated = np.mean(np.split(weighted_output, n_trials),1)                                                                              
    
    # plot
    ax.plot(trials_sensory, trials_estimated, 'o', alpha = 0.2, color='r')
    
    m, n =  np.polyfit(trials_sensory, trials_estimated, 1)
    ax.plot(trials_sensory, m * trials_sensory + n, '--', color='r')
    
    
# %% test

# v = 20

# x = np.linspace(1,100,1000)
# y = (1/x) / ((1/x) + (1/v))

# plt.figure()
# plt.plot(x,y)

                                                                     