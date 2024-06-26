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

from src.plot_data import plot_neuron_activity, plot_example_mean, plot_example_variance, plot_mse_heatmap

# %% Universal parameters

fs = 6
inch = 2.54

# %% Define files and paths

figure_name = 'Fig_2_partial.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)
    
# %% Define figure structure

figsize=(18/inch,13/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(3, 1, figure=fig, hspace=1.2)
G1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G[0,0], wspace=0.4, width_ratios=[1,1,1])
G11 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=G1[0,0])
G12 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G1[0,1], wspace=0.3)
G13 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G1[0,2], wspace=0.3)
G2 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=G[1,0], width_ratios=[1,5,2], height_ratios=[5, 1], wspace = 0.5)
G3 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=G[2,0], width_ratios=[1,5,2], height_ratios=[5, 1], wspace = 0.5)

ax_A = fig.add_subplot(G1[0,0])
ax_A.axis('off')
ax_A.text(-0.2, 1.25, 'A', transform=ax_A.transAxes, fontsize=fs+1)

ax_B = fig.add_subplot(G12[0,:])
ax_B.axis('off')
ax_B.set_title('PE neuron activity for \ndifferent input statistics', fontsize=fs, pad=10)
ax_B.text(-0.3, 1.25, 'B', transform=ax_B.transAxes, fontsize=fs+1)

ax_B1 = fig.add_subplot(G12[0,0])
ax_B2 = fig.add_subplot(G12[0,1])
plt.setp(ax_B2.get_yticklabels(), visible=False)

ax_C = fig.add_subplot(G13[0,:])
ax_C.axis('off')
ax_C.set_title('Interneuron activity for \ndifferent input statistics', fontsize=fs, pad=10)
ax_C.text(-0.2, 1.25, 'C', transform=ax_C.transAxes, fontsize=fs+1)

ax_C1 = fig.add_subplot(G13[0,0])
ax_C2 = fig.add_subplot(G13[0,1])
plt.setp(ax_C2.get_yticklabels(), visible=False)

ax_D = fig.add_subplot(G2[:,:])
ax_D.axis('off')
ax_D.text(-0.05, 1.25, 'D', transform=ax_D.transAxes, fontsize=fs+1)

ax_D21 = fig.add_subplot(G2[:,1])
ax_D3 = fig.add_subplot(G2[:,2])

ax_E = fig.add_subplot(G3[:,:])
ax_E.axis('off')
ax_E.text(-0.05, 1.25, 'E', transform=ax_E.transAxes, fontsize=fs+1)

ax_E21 = fig.add_subplot(G3[:,1])
ax_E3 = fig.add_subplot(G3[:,2])


# %% Neuron activity with increasing stimulus mean and variance

# data files
file_for_data = '../results/data/moments/data_heatmap_mfn_10.pickle' 

if not os.path.exists(file_for_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data,'rb') as f:
        [trial_duration, num_values_per_trial, means_tested, variances_tested, 
         dev_mean, dev_variance, activity_pe_neurons, activity_interneurons] = pickle.load(f)
    
    # plot data
    end_of_initial_phase = np.int32(trial_duration * 0.75)
    plot_neuron_activity(end_of_initial_phase, means_tested, variances_tested, activity_interneurons, None, id_fixed=3,
                         ax1=ax_C1, ax2=ax_C2)
    plot_neuron_activity(end_of_initial_phase, means_tested, variances_tested, None, activity_pe_neurons, id_fixed=3,
                         ax1=ax_B1, ax2=ax_B2)


# %% Example: Estimating mean and variance

# data files
file_for_data = '../results/data/moments/data_example_mfn_10.pickle'

if not os.path.exists(file_for_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data,'rb') as f:
        [_, _, trial_duration, _, stimuli, m_neuron, v_neuron] = pickle.load(f)
    
    # plot data
    plot_example_mean(stimuli, trial_duration, m_neuron, mse_flg = False, ax1=ax_D21)
    plot_example_variance(stimuli, trial_duration, v_neuron, mse_flg = False, ax1=ax_E21)
    

# %% Systematic exploration

# data files
file_for_data = '../results/data/moments/data_heatmap_mfn_10.pickle' 

if not os.path.exists(file_for_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data,'rb') as f:
        [trial_duration, num_values_per_trial, means_tested, variances_tested, 
         dev_mean, dev_variance, _, _] = pickle.load(f)
        
    # plot data
    plot_mse_heatmap(means_tested, variances_tested, dev_mean, 
                     title='M neuron encodes mean \nfor a wide range of input statistics', 
                     x_example=5, y_example=2**2, ax1=ax_D3)
    plot_mse_heatmap(means_tested, variances_tested, dev_variance, show_mean=False, 
                     title='V neuron encodes variance \nfor a wide range of input statistics', 
                     x_example=5, y_example=2**2, digits_round=1, ax1=ax_E3)


# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
