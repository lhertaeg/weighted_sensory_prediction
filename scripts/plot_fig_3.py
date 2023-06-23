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

from src.plot_data import plot_weighting_limit_case_example, plot_fraction_sensory_heatmap, plot_weight_over_trial, plot_neuron_activity_lower_higher

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_3.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(16/inch,12/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(3, 1, figure=fig, hspace=1.0)
G1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[0,0], wspace=0.3, width_ratios=[1,3])
G11 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=G1[0,0])
G12 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G1[0,1])
G2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[1,0], width_ratios=[2.7,1], wspace = 0.5)
G21 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G2[0,0], width_ratios=[0.7,1,1], wspace = 0.5)
G3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[2,0], width_ratios=[2.7,1], wspace = 0.5)
G31 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G3[0,0], width_ratios=[0.7,1,1], wspace = 0.5)

#ax_A = fig.add_subplot(G11[0,0])
#ax_A.set_title('XXX', fontsize=fs, pad=1)

ax_B1 = fig.add_subplot(G12[0,0])
ax_B2 = fig.add_subplot(G12[0,1])
ax_B3 = fig.add_subplot(G12[0,2])
plt.setp(ax_B2.get_yticklabels(), visible=False)

#ax_D1 = fig.add_subplot(G21[0,0])
ax_D2 = fig.add_subplot(G21[0,1])
ax_D3 = fig.add_subplot(G21[0,2])
ax_D4 = fig.add_subplot(G2[0,1])

#ax_E1 = fig.add_subplot(G31[0,0])
ax_E2 = fig.add_subplot(G31[0,1])
ax_E3 = fig.add_subplot(G31[0,2])
ax_E4 = fig.add_subplot(G3[0,1])


# %% Neuron activities

# data files
file_for_data_within = '../results/data/weighting/data_activity_neurons_exploration_within_10.pickle'
file_for_data_across = '../results/data/weighting/data_activity_neurons_exploration_across_10.pickle'

if (not os.path.exists(file_for_data_within) or not os.path.exists(file_for_data_across)):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data_within,'rb') as f:
        [variability_within, _, activity_pe_neurons_lower_within, activity_pe_neurons_higher_within, 
         activity_interneurons_lower_within, activity_interneurons_higher_within] = pickle.load(f)
            
    with open(file_for_data_across,'rb') as f:
        [_, variability_across, activity_pe_neurons_lower_across, activity_pe_neurons_higher_across, 
         activity_interneurons_lower_across, activity_interneurons_higher_across] = pickle.load(f)
        
    # plot data
    plot_neuron_activity_lower_higher(variability_within, variability_across, activity_interneurons_lower_within,
                                      activity_interneurons_higher_within, activity_interneurons_lower_across, 
                                      activity_interneurons_higher_across, activity_pe_neurons_lower_within,
                                      activity_pe_neurons_higher_within, activity_pe_neurons_lower_across,
                                      activity_pe_neurons_higher_across, ax1=ax_B1, ax2=ax_B2, ax3=ax_B3)


# %% Example limit cases

# data files
file_for_data_prediction_driven = '../results/data/weighting/data_example_limit_case_prediction_driven_10.pickle'
file_for_data_sensory_driven = '../results/data/weighting/data_example_limit_case_sensory_driven_10.pickle'

if (not os.path.exists(file_for_data_prediction_driven) or not os.path.exists(file_for_data_sensory_driven)):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data_prediction_driven,'rb') as f:
            [n_trials, trial_duration, _, stimuli_1, m_neuron_lower_1, v_neuron_lower_1, m_neuron_higher_1, 
              v_neuron_higher_1, alpha_1, beta_1, weighted_output_1] = pickle.load(f)
            
    with open(file_for_data_sensory_driven,'rb') as f:
        [_, _, _, stimuli_2, m_neuron_lower_2, v_neuron_lower_2, m_neuron_higher_2, v_neuron_higher_2, 
          alpha_2, beta_2, weighted_output_2] = pickle.load(f)
        
    
    # plot data
    plot_weighting_limit_case_example(n_trials, trial_duration, stimuli_1, m_neuron_lower_1, m_neuron_higher_1, 
                                      v_neuron_lower_1, v_neuron_higher_1, alpha_1, beta_1, weighted_output_1,
                                      ax1 = ax_D2, ax3 = ax_D3)
    
    plot_weighting_limit_case_example(n_trials, trial_duration, stimuli_2, m_neuron_lower_2, m_neuron_higher_2, 
                                      v_neuron_lower_2, v_neuron_higher_2, alpha_2, beta_2, weighted_output_2, 
                                      plot_legend = False, ax1 = ax_E2, ax3 = ax_E3)

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
    plot_fraction_sensory_heatmap(alpha, variability_within, variability_across, 2, xlabel='variability across trial', 
                                  ylabel='variability within trial', ax = ax_D4)


# %% Impact of trial duration

# data files
file_for_data_control = '../results/data/weighting/data_weighting_trail_duration_control.pickle'
file_for_data_shorter = '../results/data/weighting/data_weighting_trail_duration_shorter.pickle'

if (not os.path.exists(file_for_data_control) or not os.path.exists(file_for_data_shorter)):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data_control,'rb') as f:
        [n_trials, _, _, weight_control] = pickle.load(f)
            
    with open(file_for_data_shorter,'rb') as f:
        [n_trials, _, _, weight_shorter] = pickle.load(f)
        
    # plot data
    plot_weight_over_trial(weight_control, weight_shorter, n_trials, ax = ax_E4,
                            leg_text=['trial duration = T', 'trial duration = T/5'])
    
    
# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)

# %% Barplots 

# min_mean = 3 # 1, 5
# max_mean = 5 # 10, 5

# epsilon = 0.1
# sd_stimuli = epsilon + 1 # 0, 2

# num_bars = 20
# nums = 1000

# y = np.zeros((1000,num_bars))
# x = np.ones_like(y)

# for i in range(num_bars):
    
#     x[:,i] *= i
    
#     mean_trial = np.random.uniform(min_mean, max_mean)
#     min_value = mean_trial - np.sqrt(3)*sd_stimuli
#     max_value = mean_trial + np.sqrt(3)*sd_stimuli
    
#     y[:,i] = np.linspace(min_value, max_value, nums)

# norm = plt.Normalize(y.min(), y.max())
# fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

# for i in range(num_bars):
#     points = np.array([x[:,i], y[:,i]]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)

#     lc = LineCollection(segments, cmap='viridis', norm=norm)
#     lc.set_array(y[:,i])
#     lc.set_linewidth(10)
#     line = axs.add_collection(lc)

# axs.axhline(np.mean(y), color='k', ls=':')

# axs.set_xlim(x.min()-1, x.max()+1)
# axs.set_xticks([])
# axs.set_yticks([])
# axs.set_ylim(0, 11)
# plt.show()

# sns.despine(ax=axs, bottom=True, top=True, left=True, right=True)