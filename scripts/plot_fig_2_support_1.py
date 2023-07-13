#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path

from src.plot_data import plot_mse_test_distributions, plot_mse_test_distributions


# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_2_S1.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)
    
    
# %% Define figure structure

figsize=(9/inch,7/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(2, 1, figure=fig, hspace=0.4)#, hspace=1.2)

ax_G = fig.add_subplot(G[:,0])
ax_G.axis('off')
ax_G.set_title('Mean & variance are estimated correctly for various stimulus distributions', fontsize=fs, pad=10)

ax_A = fig.add_subplot(G[0,0])
plt.setp(ax_A.get_xticklabels(), visible=False)
ax_B = fig.add_subplot(G[1,0])


# %% MSE for different distributions

# data files
file_for_data = '../results/data/moments/data_test_distributions_mfn_10.pickle'

if not os.path.exists(file_for_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data,'rb') as f:
        [trial_duration, dist_types, num_reps, mse_mean, mse_variance] = pickle.load(f)
    
    # plot data
    mean, std = 5, 2
    var_tested = std**2
    
    mse_mean_normalised = mse_mean/mean**2 * 100
    mse_variance_normalised = mse_variance/var_tested**2 * 100
    
    sem_mean = np.std(mse_mean_normalised,2)/np.sqrt(num_reps)
    sem_variance = np.std(mse_variance_normalised,2)/np.sqrt(num_reps)
    
    plot_mse_test_distributions(np.mean(mse_mean_normalised,2), SEM = sem_mean, fs=fs, mean=mean, std=std, dist_types=dist_types, 
                                inset_steady_state=True, plot_xlabel = False, ax=ax_A)
    plot_mse_test_distributions(np.mean(mse_variance_normalised,2), SEM = sem_variance, fs = fs, inset_steady_state = True, ax=ax_B)


# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)