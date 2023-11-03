#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:13:56 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import os.path

from src.functions_simulate import simulate_neuromod_combos
from src.plot_data import plot_neuromod_impact_inter

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Activate fractions of IN neurons in lower/higher PE circuit or both for a specified input statistics

run_cell = False

if run_cell:
    
    # define combo activation resolution
    nums = 11
    
    # run through all combinations
    for mfn_flag in ['10', '01', '11']: # choose mean-field network to simulate
        
        print('MFN code: ', mfn_flag)
        print(' ')
    
        for column in range(3): # 0: both, 1: lower level PE circuit, 2: higher level PE circuit

            for std_mean in [0, 1]: # uncertainty of environement [0, 1]
                
                n_std = 1 - std_mean # uncertainty of stimulus [1, 0]
                    
                for IN_combo_flg in range(3): # 0: PV - SOM, 1: SOM - VIP, 2: VIP - PV
        
                    # filename for data & define activation combos
                    identifier = '_column_' + str(column) + '_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
                    print(identifier)
        
                    if IN_combo_flg==0:
                        file_for_data = '../results/data/neuromod/data_weighting_neuromod_PV-SOM_' + mfn_flag + identifier + '.pickle'
                        xp = np.linspace(0, 1, nums)
                        xs = 1 - xp
                        xv = np.zeros_like(xp)
                        
                    elif IN_combo_flg==1:
                        file_for_data = '../results/data/neuromod/data_weighting_neuromod_SOM-VIP_' + mfn_flag + identifier + '.pickle'
                        xp = np.zeros(nums)
                        xs = np.linspace(0, 1, nums)
                        xv = 1 - xs
                        
                    elif IN_combo_flg==2:
                        file_for_data = '../results/data/neuromod/data_weighting_neuromod_VIP-PV_' + mfn_flag + identifier + '.pickle'
                        xv = np.linspace(0, 1, nums)
                        xp = 1 - xv
                        xs = np.zeros(nums)
            
        
                    if not os.path.exists(file_for_data):
                        [_, _, _, alpha_before_pert, 
                         alpha_after_pert] = simulate_neuromod_combos(mfn_flag, std_mean, n_std, column, 
                                                                      xp, xs, xv, file_for_data = file_for_data)
        

# %% Plot results above

run_cell = False

if run_cell:
    
    column = 1
    std_mean = 1
    n_std = 0
    
    plot_neuromod_impact_inter(column, std_mean, n_std)
