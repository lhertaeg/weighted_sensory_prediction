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

from src.plot_data import plot_weighting_limit_case_example, plot_fraction_sensory_heatmap, plot_weight_over_trial
from src.plot_data import plot_neuron_activity_lower_higher, plot_standalone_colorbar

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_3_partial.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(18/inch,15/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(3, 1, figure=fig, hspace=1.0)
G1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[0,0], wspace=0.3, width_ratios=[1,3])
G11 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=G1[0,0])
G12 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G1[0,1])
G2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[1,0], width_ratios=[3,1], wspace = 0.3)
G21 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G2[0,0], width_ratios=[0.7,1,1], wspace = 0.7)
G3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[2,0], width_ratios=[3,1], wspace = 0.3)
G31 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G3[0,0], width_ratios=[0.7,1,1], wspace = 0.7)

ax_A = fig.add_subplot(G1[0,0])
ax_A.axis('off')
ax_A.text(-0.45, 1.25, 'A', transform=ax_A.transAxes, fontsize=fs+1)

ax_B = fig.add_subplot(G12[0,:])
ax_B.axis('off')
ax_B.text(-0.13, 1.25, 'B', transform=ax_B.transAxes, fontsize=fs+1)

ax_B1 = fig.add_subplot(G12[0,0])
ax_B1.axis('off')

ax_C = fig.add_subplot(G21[0,:])
ax_C.axis('off')
ax_C.text(-0.15, 1.25, 'C', transform=ax_C.transAxes, fontsize=fs+1)

ax_D2 = fig.add_subplot(G21[0,1])
ax_D2.set_title('Example in which the network relies more strongly on sensory inputs', fontsize=fs, pad=10)
ax_D3 = fig.add_subplot(G21[0,2])
ax_D4 = fig.add_subplot(G2[0,1])
ax_D4.set_title('Sensory weight for different \ntrial and stimulus variabilities', fontsize=fs, pad=10)
ax_D4.text(-0.4, 1.25, 'E', transform=ax_D4.transAxes, fontsize=fs+1)

ax_D = fig.add_subplot(G31[0,:])
ax_D.axis('off')
ax_D.text(-0.15, 1.25, 'D', transform=ax_D.transAxes, fontsize=fs+1)

ax_E2 = fig.add_subplot(G31[0,1])
ax_E2.set_title('Example in which the network relies more strongly on prediction', fontsize=fs, pad=10)
ax_E3 = fig.add_subplot(G31[0,2])
ax_E4 = fig.add_subplot(G3[0,1])
ax_E4.set_title('Sensory weight decreases \nwith shorter trials', fontsize=fs, pad=10)
ax_E4.text(-0.35, 1.25, 'F', transform=ax_E4.transAxes, fontsize=fs+1)


# %% Weighted output - equation

text_equation_part_1 = r'Weighted output = $\alpha \cdot$ Stimulus + (1-$\alpha$) $\cdot$ Prediction'
text_equation_part_2 = r'= $\left(1 + \frac{V_\mathrm{lower}}{V_\mathrm{higher}}\right)^{-1} \cdot$ Stimulus + $\left(1 + \frac{V_\mathrm{higher}}{V_\mathrm{lower}}\right)^{-1} \cdot$ M'
ax_B1.text(-0.0,1.05, text_equation_part_1, fontsize=fs+1)
ax_B1.text(0.79,0.82, text_equation_part_2, fontsize=fs+1)

ax_B1.text(0.0,0.1, r'$\alpha$: Sensory weight $\in$ [0,1]', fontsize=fs+1)
ax_B1.text(1.8,0.25, r'$\alpha$ = 0: Prediction-driven', fontsize=fs+1, color='white')
ax_B1.text(1.8,-0.05, r'$\alpha$ = 1: Sensory-driven', fontsize=fs+1, color='white')


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
                                      plot_legend = False, ax1 = ax_E2, ax3 = ax_E3)
    
    plot_weighting_limit_case_example(n_trials, trial_duration, stimuli_2, m_neuron_lower_2, m_neuron_higher_2, 
                                      v_neuron_lower_2, v_neuron_higher_2, alpha_2, beta_2, weighted_output_2, 
                                      plot_legend = True, ax1 = ax_D2, ax3 = ax_D3)

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
                                  ylabel='Stimulus variability', ax = ax_D4)


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
    plot_weight_over_trial(weight_control, weight_shorter, n_trials, ax = ax_E4, lw=0.5,
                            leg_text=['trial duration', 'trial duration / 5'])
    
    
# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
