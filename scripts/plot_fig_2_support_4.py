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

from src.plot_data import plot_example_stimuli_smoothed, plot_dev_cntns, plot_example_mean, plot_example_variance

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_2_S4.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)
    
# %% Define figure structure

figsize=(13/inch,8/inch)
fig = plt.figure(figsize=figsize)
G = gridspec.GridSpec(2, 2, figure=fig, hspace=0.8, wspace=0.4)
CD = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[1,:], wspace=0.25)

ax_A = fig.add_subplot(G[0,0])
ax_A.text(-0.25, 1.1, 'A', transform=ax_A.transAxes, fontsize=fs+1)
ax_B = fig.add_subplot(G[0,1])
ax_B.text(-0.25, 1.1, 'B', transform=ax_B.transAxes, fontsize=fs+1)
ax_C = fig.add_subplot(CD[0,0])
ax_C.text(-0.2, 1.1, 'C', transform=ax_C.transAxes, fontsize=fs+1)
ax_D = fig.add_subplot(CD[0,1])
ax_D.text(-0.15, 1.1, 'D', transform=ax_D.transAxes, fontsize=fs+1)


# %% Continuous signals

# data files
file_for_example = '../results/data/moments/data_net_example_cntns_input_one_column.pickle'
file_for_data = '../results/data/moments/data_cntns_input_one_column.pickle'

with open(file_for_example,'rb') as f:
        [stimuli, trial_duration, m_neuron, v_neuron] = pickle.load(f)
        
with open(file_for_data,'rb') as f:
    [hann_windows, dev_mean, dev_variance] = pickle.load(f)
    
    
# plot
plot_example_stimuli_smoothed([hann_windows[0], hann_windows[-1]], ax=ax_A)

plot_dev_cntns(hann_windows, dev_mean, dev_variance, ax = ax_B)

plot_example_mean(stimuli, trial_duration, m_neuron, mse_flg=False, ax1=ax_C)
plot_example_variance(stimuli, trial_duration, v_neuron, mse_flg=False, ax1=ax_D)

ax_C.set_title('M neuron encodes mean', fontsize=fs, pad=10)
ax_D.set_title('V neuron encodes varaince', fontsize=fs, pad=10)
ax_D.set_ylabel(' ')
    

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
    