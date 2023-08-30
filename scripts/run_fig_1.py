#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.functions_simulate import simulate_PE_circuit_P_fixed_S_constant
from src.plot_data import plot_example_mean, plot_example_variance, plot_mse_heatmap, plot_interneuron_activity_heatmap
from src.plot_data import plot_neuron_activity

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% How do the interneurons & PE neurons behave with different input statistics? (later you could combine it with the one above!)

run_cell = True
plot_only = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_mfn_' + mfn_flag + '_PE_neurons_constant_stimuli_P_fixed.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
    
        ## define parameters ranges tested:
        prediction_initial = 5
        stimulus_tested = np.linspace(0,10,21)
        
        ## run simulations
        nPE, pPE = simulate_PE_circuit_P_fixed_S_constant(mfn_flag, prediction_initial, stimulus_tested, file_for_data)
        
        ### save data
        with open(file_for_data,'wb') as f:
             pickle.dump([prediction_initial, stimulus_tested, nPE, pPE], f)
        
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [prediction_initial, stimulus_tested, nPE, pPE] = pickle.load(f)
        
    ### plot 
    plt.figure()
    plt.plot(stimulus_tested - prediction_initial, nPE)
    plt.plot(stimulus_tested - prediction_initial, pPE)
    
    plt.figure()
    plt.plot(stimulus_tested - prediction_initial, nPE**2)
    plt.plot(stimulus_tested - prediction_initial, pPE**2)
    plt.plot(stimulus_tested - prediction_initial, (stimulus_tested - prediction_initial)**2)

