#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% import

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path
import seaborn as sns

from matplotlib.collections import LineCollection

from src.plot_data import plot_fraction_sensory_heatmap, plot_transitions_examples

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_3_S1.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(18/inch,3/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4, width_ratios=[1,4])
G1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=G[0,0])
G2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=G[0,1], wspace=0.3)

ax_A = fig.add_subplot(G1[0,0])
ax_A.text(-0.4, 1.37, 'A', transform=ax_A.transAxes, fontsize=fs+1)

ax_B = fig.add_subplot(G2[0,:])
ax_B.axis('off')
ax_B.text(-0.1, 1.2, 'B', transform=ax_B.transAxes, fontsize=fs+1)

ax_B1 = fig.add_subplot(G2[0,0])
ax_B1.set_title(r'1 $\longrightarrow$ 2', fontsize=fs, pad=10)
ax_B2 = fig.add_subplot(G2[0,1])
plt.setp(ax_B2.get_yticklabels(), visible=False)
ax_B2.set_title(r'2 $\longrightarrow$ 3', fontsize=fs, pad=10)
ax_B3 = fig.add_subplot(G2[0,2])
plt.setp(ax_B3.get_yticklabels(), visible=False)
ax_B3.set_title(r'3 $\longrightarrow$ 4', fontsize=fs, pad=10)
ax_B4 = fig.add_subplot(G2[0,3])
plt.setp(ax_B4.get_yticklabels(), visible=False)
ax_B4.set_title(r'4 $\longrightarrow$ 1', fontsize=fs, pad=10)


# %% Systematic exploration - plot sensory weight

# data files
file_for_data = '../results/data/weighting/data_weighting_exploration_10.pickle'

if not os.path.exists(file_for_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data,'rb') as f:
        [variability_within, variability_across, alpha] = pickle.load(f)
        
    
    # plot data
    plot_fraction_sensory_heatmap(alpha, variability_within, variability_across, 2, xlabel='Trial variability', 
                                  ylabel='Stimulus variability', square=True, ax = ax_A)
    
    
    ax_A.annotate("", xy=(0.5, 4.4), xycoords='data', xytext=(0.5, 0.7), textcoords='data', fontsize=fs,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=1 , ec='k'))
    
    ax_A.annotate("", xy=(0.7, 4.5), xycoords='data', xytext=(4.4, 4.5), textcoords='data', fontsize=fs,
                 arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", lw=1 , ec='k'))
    
    ax_A.annotate("", xy=(4.5, 4.4), xycoords='data', xytext=(4.5, 0.7), textcoords='data', fontsize=fs,
                 arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", lw=1 , ec='k'))
    
    ax_A.annotate("", xy=(0.7, 0.5), xycoords='data', xytext=(4.4, 0.5), textcoords='data', fontsize=fs,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", lw=1 , ec='k'))

    ax_A.text(0.5, 0.5, '1', fontsize=fs, verticalalignment='center', horizontalalignment='center')
    ax_A.text(0.5, 4.5, '2', fontsize=fs, verticalalignment='center', horizontalalignment='center')
    ax_A.text(4.5, 4.5, '3', fontsize=fs, verticalalignment='center', horizontalalignment='center')
    ax_A.text(4.5, 0.5, '4', fontsize=fs, verticalalignment='center', horizontalalignment='center')
    
# %% Plot transitions

x_before = np.array([[5,5,0,0], [5,5,0,3], [0,10,0,3], [0,10,0,0]])
x_after = np.array([[5,5,0,3], [0,10,0,3], [0,10,0,0], [5,5,0,0]])
axs = [ax_B1, ax_B2, ax_B3, ax_B4]

for i in range(4):
    
    if i==0:
        boolean = True
    else:
        boolean = False
        
    min_mean_before,max_mean_before, m_sd_before, n_sd_before  = x_before[i]
    min_mean_after, max_mean_after, m_sd_after, n_sd_after = x_after[i]
    
    before = str(min_mean_before) + str(max_mean_before) + str(m_sd_before) + str(n_sd_before)
    after = str(min_mean_after) + str(max_mean_after) + str(m_sd_after) + str(n_sd_after)
    file_for_data = '../results/data/weighting/data_weighting_transition_10_' + before + '_' + after + '.pickle'
    
    with open(file_for_data,'rb') as f:
        [n_trials, trial_duration, _, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
         v_neuron_higher, alpha, beta, weighted_output] = pickle.load(f)
    
    plot_transitions_examples(n_trials, trial_duration, stimuli, alpha, beta, weighted_output, time_plot=0, plot_ylable=boolean, 
                              ylim=[-15,20], figsize=(4,3), xlim=[40,60], plot_only_weights=True, ax2=axs[i])    


# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
