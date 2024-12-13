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

from src.plot_data import plot_examples_spatial_M, plot_examples_spatial_V, plot_deviation_spatial

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_S8.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)
    
    
# %% Define figure structure

figsize=(18/inch,6/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(6, 3, figure=fig, width_ratios=[6,4,3], hspace=2.5, wspace=0.5)

ax_A = fig.add_subplot(G[:,0])
ax_A.axis('off')
ax_A.text(-0.4, 1, 'A', transform=ax_A.transAxes, fontsize=fs+1)

ax_BC = fig.add_subplot(G[:,1:])
ax_BC.axis('off')
ax_BC.set_title('Mean and variance estimated correctly for stimuli that vary in space', fontsize=fs, pad=20)

ax_B1 = fig.add_subplot(G[:3,1])
ax_B1.text(-0.3, 1., 'B', transform=ax_B1.transAxes, fontsize=fs+1)
plt.setp(ax_B1.get_xticklabels(), visible=False)
ax_B2 = fig.add_subplot(G[3:,1])

ax_C1 = fig.add_subplot(G[:3,2])
ax_C1.text(-0.5, 1.1, 'C', transform=ax_C1.transAxes, fontsize=fs+1)
plt.setp(ax_C1.get_xticklabels(), visible=False)
ax_C2 = fig.add_subplot(G[3:,2])


# %% Show examples with transitions

# data files
file_before = '../results/data/moments/data_example_spatial_mfn_10.pickle'
file_after_1 = '../results/data/moments/data_example_spatial_mfn_10_diff_noise_levels.pickle'
file_after_2 = '../results/data/moments/data_example_spatial_mfn_10_diff_means.pickle'

if not os.path.exists(not os.path.exists(file_before) or not os.path.exists(file_after_1) or not os.path.exists(file_after_2)):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_before,'rb') as f:
        [mean_stimulus, spatial_std, spatial_noise, num_time_steps, m_neuron_before, v_neuron_before, rates_final_before] = pickle.load(f)
    
    with open(file_after_1,'rb') as f:
            [mean_stimulus, spatial_std, spatial_noise, num_time_steps, m_neuron_1, v_neuron_1, rates_final] = pickle.load(f)
            
    with open(file_after_2,'rb') as f:
            [mean_stimulus, spatial_std, spatial_noise, num_time_steps, m_neuron_2, v_neuron_2, rates_final] = pickle.load(f)

    # plot data
    m_neuron_after = np.vstack((m_neuron_1, m_neuron_2))
    v_neuron_after = np.vstack((v_neuron_1, v_neuron_2))
    plot_examples_spatial_M(num_time_steps, m_neuron_before, m_neuron_after, 4, np.array([4,6]), 
                            labels=['Stimulus 1',r'Stimulus 2 ($\sigma^2$ changed)','Stimulus 2 ($\mu$ changed)'], show_xlabel=False, ax=ax_B1)
    plot_examples_spatial_V(num_time_steps, v_neuron_before, v_neuron_after, 4, np.array([6,4]), 
                            labels=['Stimulus 1',r'Stimulus 2 ($\sigma^2$ changed)','Stimulus 2 ($\mu$ changed)'], ax=ax_B2)


# %% Heatmaps

# data files
file_data = '../results/data/moments/data_spatial_mfn_diff_input_statistics.pickle'

if not os.path.exists(file_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_data,'rb') as f:
        [means_tested, stds_tested, deviation_mean, deviation_std] = pickle.load(f)


    x_examples = [4, 4, 6]
    y_examples = [4, 6, 4]
    markers_examples = ['o', '^', '^']
    
    plot_deviation_spatial(deviation_mean, means_tested, stds_tested, vmin=0.35, vmax=1.2, fs=6, show_xlabel=False, ax=ax_C1,
                           x_examples = x_examples, y_examples = y_examples, markers_examples = markers_examples)
    plot_deviation_spatial(deviation_std, means_tested, stds_tested, vmin=4.2, vmax=4.4, fs=6, show_mean=False, ax=ax_C2,
                           x_examples = x_examples, y_examples = y_examples, markers_examples = markers_examples)
    

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
    