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

from src.plot_data import plot_example_contraction_bias, plot_slope_trail_duration, plot_slope_variability
from src.plot_data import plot_illustration_input_cond, plot_illustration_bias_results, plot_illustration_trial_duration

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

figsize=(15/inch,10/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(5, 2, figure=fig, wspace=0.5, height_ratios=[1,1,1,1.3,1.3]) 

ax_A = fig.add_subplot(G[:2,0])
ax_A.text(-0.25, 1.15,'A', transform=ax_A.transAxes, fontsize=fs+1)

B = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=G[:2,1], width_ratios=[1.5,1])
CD = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=G[3:,0], hspace=1.2)
C = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=CD[0,0])
D = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=CD[1,0])
E = gridspec.GridSpecFromSubplotSpec(5, 2, subplot_spec=G[3:,1], width_ratios=[1.5,1])

ax_B = fig.add_subplot(B[0,:])
ax_B.axis('off')
ax_B.set_title('Bias increases/decreases with stimulus/trial variability', fontsize=fs, pad=10, loc='center')
ax_B.text(-0.25, 1.7,'B', transform=ax_B.transAxes, fontsize=fs+1)
ax_B1 = fig.add_subplot(B[:,0])
ax_B2 = fig.add_subplot(B[1,1])
ax_B3 = fig.add_subplot(B[2,1])

ax_C = fig.add_subplot(C[0,:])
ax_C.axis('off')
ax_C.set_title('Bias independent of trial var. if stimulus is not noisy', fontsize=fs, pad=10, loc='center')
ax_C.text(-0.25, 1.75,'C', transform=ax_C.transAxes, fontsize=fs+1)
ax_C1 = fig.add_subplot(C[:,0])
ax_C2 = fig.add_subplot(C[0,1])
ax_C3 = fig.add_subplot(C[1,1])

ax_D = fig.add_subplot(D[0,:])
ax_D.axis('off')
ax_D.set_title('Bias independent of stimulus noise if mean is fixed', fontsize=fs, pad=10, loc='center')
ax_D.text(-0.25, 1.75,'D', transform=ax_D.transAxes, fontsize=fs+1)
ax_D1 = fig.add_subplot(D[:,0])
ax_D2 = fig.add_subplot(D[0,1])
ax_D3 = fig.add_subplot(D[1,1])

ax_E = fig.add_subplot(E[0,:])
ax_E.axis('off')
ax_E.text(-0.25, 0.8,'E', transform=ax_E.transAxes, fontsize=fs+1)
ax_E1 = fig.add_subplot(E[1:,0])
ax_E1.set_title('Bias depends on trial duration', fontsize=fs, pad=10, loc='center')
ax_E2 = fig.add_subplot(E[2,1])


# %% Illustrate input condition for A

std_stims = np.array([2, 1]) 
mean_trails = np.array([20,20])
std_trails = np.array([3, 3])

ax_A1 = ax_A.inset_axes((0.7,0.8,.25,.25))
ax_A2 = ax_A.inset_axes((0.7,0.5,.25,.25))

plot_illustration_input_cond(std_stims, mean_trails, std_trails, ax_1 = ax_A1, ax_2 = ax_A2, labels=False)

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
    
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30), ms=2, fs=6, marker='o',
                                  min_means=min_means, max_means=max_means, m_std=m_std, n_std=n_std, ax=ax_A)


# %% Slope decreases (contraction bias increases) with stimulus variability
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
                            label_text, fs=6, ax = ax_B1)


# %% Illustrate the predictions

plot_illustration_bias_results(ax1=ax_B2, ax2=ax_B3)

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
    
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30), ms=2, fs=6, plot_xlabel=False,
                                  min_means=min_means, max_means=max_means, m_std=m_std, n_std=n_std, ax=ax_C1) 
    
    # Illustrate input condition for D
    std_stims = np.array([1e-1, 1e-1])
    mean_trails = np.array([20, 20])
    std_trails = np.array([5.5, 3])
    slopes = np.array([weighted_output_1[0], weighted_output_2[0]])
    
    plot_illustration_input_cond(std_stims, mean_trails, std_trails, slopes=slopes, ax_1 = ax_C2, ax_2 = ax_C3)


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
    
    plot_example_contraction_bias(weighted_output, stimuli, n_trials, num_trial_ss=np.int32(30), ms=2, plot_ylabel=True,
                                  min_means=min_means, max_means=max_means, m_std=m_std, n_std=n_std, fs=6, 
                                  show_marker_inset=True, ax = ax_D1) 
    
    # Illustrate input condition for D

    std_stims = np.array([0.5, 1])
    mean_trails = np.array([15, 15])
    std_trails = np.array([0.01, 0.01])
    slopes = np.array([weighted_output_1[0], weighted_output_2[0]])
    
    plot_illustration_input_cond(std_stims, mean_trails, std_trails, slopes=slopes, ax_1 = ax_D2, ax_2 = ax_D3)
    
    
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
    label_text = ['Stim var = 0', 'Trial var = 0']
    plot_slope_trail_duration(trial_durations, fitted_slopes_1, fitted_slopes_2, label_text, fs=6, ax=ax_E1)
    
    # plot illustration trail duration
    plot_illustration_trial_duration(ax_1=ax_E2)
    

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
