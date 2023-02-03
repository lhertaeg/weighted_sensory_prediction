#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.functions_simulate import simulate_neuromod
from src.plot_data import plot_heatmap_neuromod, plot_combination_activation_INs

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Activate fractions of IN neurons in lower/higher PE circuit or both for a specified input statistics

run_cell = True
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### define target area and stimulus statistics
    column = 0      # 0: both, 1: lower level PE circuit, 2: higher level PE circuit
    std_mean = 0   # uncertainty of environement [0, 0.5, 1]
    n_std = 1       # uncertainty of stimulus [1, 0.5, 0]
    
    ### filename for data
    identifier = '_column_' + str(column) + '_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
    file_for_data = '../results/data/neuromod/data_weighting_neuromod' + identifier + '.pickle'
    
    ### get data
    if not plot_only: # simulate respective network

        nums = 11
        xp, xs = np.meshgrid(np.linspace(0, 1, nums), np.linspace(0, 1, nums))
        xv = np.sqrt(1 - xp**2 - xs**2)
        
        [_, _, _, alpha_before_pert, alpha_after_pert] = simulate_neuromod(mfn_flag, std_mean, n_std, column, 
                                                                            xp, xs, xv, file_for_data = file_for_data)
        
    else:
        
        with open(file_for_data,'rb') as f:
            [xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f)
            
     
    ### plot data
    plot_heatmap_neuromod(xp, xs, alpha_before_pert, alpha_after_pert)
    plot_combination_activation_INs(xp, xs, xv, alpha_before_pert, alpha_after_pert)