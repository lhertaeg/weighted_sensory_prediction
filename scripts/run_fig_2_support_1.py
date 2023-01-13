#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.functions_simulate import simulate_example_pe_circuit
from src.plot_data import plot_mse_test_distributions

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Testing different distributions

# Todo:
    # choose parameters consistent with the subsequent plots
    # beautify plotting functions:
        # find a nice way to illustrate the distributions
        # take nice color scheme
        # inset steady state
    # rerun (you have to because something in the functions changed!)
    
run_cell = True
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_test_distributions_mfn_' + mfn_flag + '.pickle'
    
    ### stimulus statistics
    mean = 5
    std = 2
        
    ### get data
    if not plot_only: # simulate respective network
    
        ## define all distributions to be tested
        dist_types = ['uniform', 'normal', 'gamma', 'binary_equal_prop', 'binary_unequal_prop']
        
        ## number of repetitions
        num_reps = np.int32(20)
        
        ## initialise
        trial_duration = np.int32(100000)
        mse_mean = np.zeros((len(dist_types), trial_duration, num_reps))
        mse_variance = np.zeros((len(dist_types), trial_duration, num_reps))
        
        ## test different distributions
        for i, dist_type in enumerate(dist_types):
            
            print('Testing', dist_type, 'distribution')
            
            for seed in range(num_reps):
                
                # run network
                temp_file = '../results/data/moments/temp.pickle'
                [_, _, trial_duration, _, stimuli, 
                 m_neuron, v_neuron] = simulate_example_pe_circuit(mfn_flag, mean, std, temp_file, trial_duration=trial_duration, 
                                                                   dist_type = dist_type, seed = seed*100)
                
                # compute mean squared error            
                trials = np.arange(len(stimuli))/trial_duration
                running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1) 
                mse_mean[i, :, seed] = (running_average - m_neuron)**2
                
                momentary_variance = (stimuli - running_average)**2
                running_variance = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1)
                mse_variance[i, :, seed] = (running_variance - v_neuron)**2
                
        ## save data
        with open(file_for_data,'wb') as f:
            pickle.dump([trial_duration, dist_types, num_reps, mse_mean, mse_variance],f)
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [trial_duration, dist_types, num_reps, mse_mean, mse_variance] = pickle.load(f)
            
    ### plot single panels
    sem_mean = np.std(mse_mean,2)/np.sqrt(num_reps)
    sem_variance = np.std(mse_variance,2)/np.sqrt(num_reps)
    
    plot_mse_test_distributions(np.mean(mse_mean,2), SEM = sem_mean, title = 'Mean of stimuli', 
                                x_lim=[0,50000], mean=mean, std=std, dist_types=dist_types)
    plot_mse_test_distributions(np.mean(mse_variance,2), SEM = sem_variance, 
                                title = 'Variance of stimuli', x_lim=[0,50000])
        