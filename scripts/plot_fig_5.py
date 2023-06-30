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

from src.plot_data import plot_example_contraction_bias, plot_slope_trail_duration, plot_slope_variability, plot_contraction_bias_illustration


# %% Universal parameters

fs = 6
inch = 2.54
dtype = np.float32

# %% Define files and paths

figure_name = 'Fig_5.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)
    
    
# %% Define figure structure

figsize=(18/inch,10/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(2, 4, figure=fig, hspace = 1, wspace=2, width_ratios=[1,1,1,1])

G1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[0,:2],  wspace=0.8)
G2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[0,2:],  wspace=0.8)
G_lower = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=G[1,:], width_ratios=[1,1,1,1.2],  wspace=1)
G3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G_lower[0,:3],  wspace=0.8)
G31 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G3[0,:2],  wspace=0.5)
G4 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=G_lower[0,3])

ax_A = fig.add_subplot(G1[0,:])
ax_A.axis('off')
ax_A.set_title('Illustration of contraction bias', fontsize=fs, pad=10)

ax_A1 = fig.add_subplot(G1[0,0])
ax_A2 = fig.add_subplot(G1[0,1])

ax_B = fig.add_subplot(G2[0,:])
ax_B.axis('off')
ax_B.set_title('Contraction bias in the model', fontsize=fs, pad=10)

ax_B1 = fig.add_subplot(G2[0,0])
ax_B2 = fig.add_subplot(G2[0,1])

ax_C = fig.add_subplot(G31[0,:])
ax_C.axis('off')
ax_C.set_title('Bias independent of stimulus variability \nwhen trial variability is zero, or vice versa', fontsize=fs, pad=10)

ax_C1 = fig.add_subplot(G31[0,0])
ax_C2 = fig.add_subplot(G31[0,1])


ax_D = fig.add_subplot(G3[0,2])
ax_D.set_title('Bias depends \non trial duration', fontsize=fs, pad=10)

ax_E = fig.add_subplot(G4[0,0])
ax_E.set_title('Bias increases when \nvariability scales with mean', fontsize=fs, pad=10)

# %% Illustration contractin bias

plot_contraction_bias_illustration(ax1=ax_A1, ax2=ax_A2)


# %% Example: Slope decreases (contrcation bias increases) with stimulus variability

# data files
m_std, n_std_1, n_std_2 = dtype(0), dtype(1), dtype(7)
min_mean, max_mean = dtype(15), dtype(25)

### filenames for data
file_for_data_1 = '../results/data/behavior/data_contraction_bias_trial_input_sd_' + str(n_std_1) + '_mean_range_' + str(max_mean - min_mean) + '.pickle'
file_for_data_2 = '../results/data/behavior/data_contraction_bias_trial_input_sd_' + str(n_std_2) + '_mean_range_' + str(max_mean - min_mean) + '.pickle'

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
    min_means = np.array([min_mean, min_mean])
    max_means = np.array([max_mean, max_mean])
    m_std = np.array([m_std, m_std])
    n_std = np.array([n_std_2, n_std_1])
    
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30), ms=2, fs=6,
                                  min_means=min_means, max_means=max_means, m_std=m_std, n_std=n_std, ax=ax_B1)


# %% Slope decreases (contrcation bias increases) with stimulus variability
# slope increases (contraction bias decreases) with trial variability

# data_files
file_for_data_1 = '../results/data/behavior/data_contraction_bias_increasing_input_sd.pickle'
file_for_data_2 = '../results/data/behavior/data_contraction_bias_increasing_pred_sd.pickle'

if (not os.path.exists(file_for_data_1) or not os.path.exists(file_for_data_2)):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data_1,'rb') as f:
        [input_stds, fitted_slopes_1] = pickle.load(f)
        
    with open(file_for_data_2,'rb') as f:
        [max_mean_arr, fitted_slopes_2] = pickle.load(f)
        
    ### plot data 
    label_text = ['Stim', 'Trial']
    plot_slope_variability(input_stds, (max_mean_arr - min_mean)/np.sqrt(12), fitted_slopes_1, fitted_slopes_2, 
                           label_text, fs=6, ax = ax_B2)


# %% Example: Contraction bias occurs even without stimulus variance

# data files
m_std, n_std = dtype(0), dtype(0)
min_mean_1, max_mean_1 = dtype(15), dtype(25)
min_mean_2, max_mean_2 = dtype(10), dtype(30)

file_for_data_1 = '../results/data/behavior/data_contraction_bias_trial_mean_range_' + str(max_mean_1 - min_mean_1) + '.pickle'
file_for_data_2 = '../results/data/behavior/data_contraction_bias_trial_mean_range_' + str(max_mean_2 - min_mean_2) + '.pickle'

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
                                  min_means=min_means, max_means=max_means, m_std=m_std, n_std=n_std, ax=ax_C1)  


# %% Example: Contraction bias occurs even without trial variability

# data files
m_std, n_std_1, n_std_2 = dtype(0), dtype(2), dtype(5)
min_mean, max_mean = dtype(15), dtype(15)

file_for_data_1 = '../results/data/behavior/data_contraction_bias_trial_input_sd_' + str(n_std_1) + '.pickle'
file_for_data_2 = '../results/data/behavior/data_contraction_bias_trial_input_sd_' + str(n_std_2) + '.pickle'

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
    min_means = np.array([min_mean, min_mean])
    max_means = np.array([max_mean, max_mean])
    m_std = np.array([m_std, m_std])
    n_std = np.array([n_std_2, n_std_1])
    
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30), ms=2, plot_ylabel=False,
                                  min_means=min_means, max_means=max_means, m_std=m_std, n_std=n_std, fs=6, 
                                  show_marker_inset=True, ax = ax_C2) 
    
    
# %% Slope depends on trial duration

# data files
file_for_data_1 = '../results/data/behavior/data_contraction_bias_trial_duration_input_sd_zero.pickle'
file_for_data_2 = '../results/data/behavior/data_contraction_bias_trial_duration_pred_sd_zero.pickle'

if (not os.path.exists(file_for_data_1) or not os.path.exists(file_for_data_2)):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data_1,'rb') as f:
        [trial_durations, fitted_slopes_1] = pickle.load(f)
            
    with open(file_for_data_2,'rb') as f:
        [trial_durations, fitted_slopes_2] = pickle.load(f)
        
    # plot data
    label_text = ['Stim', 'Trial']
    plot_slope_trail_duration(trial_durations, fitted_slopes_1, fitted_slopes_2, label_text, fs=6, ax=ax_D)
    


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
                                  figsize=(5,2.8), ax = ax_E) 
    

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)


# %% Define figure structure

# figsize=(18/inch,12/inch)
# fig = plt.figure(figsize=figsize)

# G = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4, width_ratios=[2,1.3])

# G1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=G[0,0],  hspace=0.8)
# G11 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G1[0,0], wspace=0.5)
# G12 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G1[1,0], wspace=0.5)

# ax_A1 = fig.add_subplot(G11[0,0])
# ax_A2 = fig.add_subplot(G11[0,1])

# ax_C1 = fig.add_subplot(G12[0,0])
# ax_C2 = fig.add_subplot(G12[0,1])

# G2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=G[0,1],  hspace=0.8)
# G21 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=G2[0,0])
# G22 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=G2[1,0])

# ax_B = fig.add_subplot(G21[0,0])
# ax_D = fig.add_subplot(G22[0,0])

