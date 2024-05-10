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
    
run_cell = True
plot_only = False

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
        timesteps = 100000
        dev_mean = np.zeros((len(dist_types), timesteps, num_reps))
        dev_variance = np.zeros((len(dist_types), timesteps, num_reps))
        
        ## test different distributions
        for i, dist_type in enumerate(dist_types):
            
            print('Testing', dist_type, 'distribution')
            
            for seed in range(num_reps):
                
                # run network
                temp_file = '../results/data/moments/temp.pickle'
                [_, _, trial_duration, _, stimuli, 
                 m_neuron, v_neuron] = simulate_example_pe_circuit(mfn_flag, mean, std, temp_file, seed = seed*100,
                                                                   dist_type = dist_type)
                
                # compute mean squared error            
                trials = np.arange(len(stimuli))/trial_duration
                running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1) 
                dev_mean[i, :, seed] = abs(running_average - m_neuron) / running_average
                
                momentary_variance = (stimuli - running_average)**2
                running_variance = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1)
                dev_variance[i, :, seed] = abs(running_variance - v_neuron) / running_variance
                
        ## save data
        with open(file_for_data,'wb') as f:
            pickle.dump([trial_duration, dist_types, num_reps, dev_mean, dev_variance],f)
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [trial_duration, dist_types, num_reps, dev_mean, dev_variance] = pickle.load(f)
            
    ### plot single panels
    sem_mean = np.std(dev_mean * 100,2)/np.sqrt(num_reps)
    sem_variance = np.std(dev_variance * 100,2)/np.sqrt(num_reps)
    
    plot_mse_test_distributions(np.mean(dev_mean * 100,2), SEM = sem_mean, title = 'Mean of stimuli', fs=8, 
                                mean=mean, std=std, dist_types=dist_types)
    plot_mse_test_distributions(np.mean(dev_variance * 100,2), SEM = sem_variance, fs = 8,
                                title = 'Variance of stimuli')
    