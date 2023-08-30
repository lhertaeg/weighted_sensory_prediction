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


# %% import

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path
import seaborn as sns

from matplotlib.collections import LineCollection

from src.plot_data import plot_example_contraction_bias

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_5_S1.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(3/inch,3/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(1, 1, figure=fig)#, hspace=0.6)
ax_A = fig.add_subplot(G[0,0])

# %% Scalar variability introduces stronger contraction bias for higher values

# data_files
min_mean_1, max_mean_1 = dtype(15), dtype(25)
min_mean_2, max_mean_2 = dtype(25), dtype(35)
m_std, n_std = dtype(1), dtype(-14)
    
file_for_data_1 = '../results/data/behavior/data_contraction_bias_trial_mean_range_' + str(max_mean_1 - min_mean_1) + '_max_mean_' + str(max_mean_1) + '.pickle'
file_for_data_2 = '../results/data/behavior/data_contraction_bias_trial_imean_range_' + str(max_mean_2 - min_mean_2) + '_max_mean_' + str(max_mean_2) + '.pickle'

if (not os.path.exists(file_for_data_1) or not os.path.exists(file_for_data_2)):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data_1,'rb') as f:
        [n_trials, _, _, stimuli_1, _, _, _, _, a1, _, weighted_output_1, trial_means_1] = pickle.load(f)
        
    with open(file_for_data_2,'rb') as f:
        [n_trials, _, _, stimuli_2, _, _, _, _, a2, _, weighted_output_2, trial_means_2] = pickle.load(f)
        
    # plot data
    weighted_output = np.vstack((weighted_output_2, weighted_output_1))
    stimuli = np.vstack((stimuli_2, stimuli_1))
    min_means = np.array([min_mean_2, min_mean_1])
    max_means = np.array([max_mean_2, max_mean_1])
    m_std = np.array([m_std, m_std])
    n_std = np.array([n_std, n_std])
    
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30), ms=2, fs=6,
                                  min_means=min_means, max_means=max_means, m_std=m_std, n_std=n_std, 
                                  figsize=(5,2.8), ax = ax_A) 
    

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
