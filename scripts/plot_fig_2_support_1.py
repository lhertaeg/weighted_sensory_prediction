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

from src.plot_data import plot_mse_test_distributions


# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_2_S1.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)
    
    
# %% Define figure structure

figsize=(8/inch,6/inch)
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
        [trial_duration, dist_types, num_reps, dev_mean, dev_variance] = pickle.load(f)
    
    # plot data
    sem_mean = np.std(dev_mean * 100,2)/np.sqrt(num_reps)
    sem_variance = np.std(dev_variance * 100,2)/np.sqrt(num_reps)
    
    plot_mse_test_distributions(np.mean(dev_mean * 100,2), SEM = sem_mean, fs=fs, dist_types=dist_types, 
                                plot_xlabel = False, ax=ax_A)
    plot_mse_test_distributions(np.mean(dev_variance * 100,2), SEM = sem_variance, fs = fs, ax=ax_B)


# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)