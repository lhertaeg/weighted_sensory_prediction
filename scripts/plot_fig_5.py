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

figsize=(8/inch,18/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1,6], hspace=0.3) #, hspace = 1, wspace=2)
G_lower = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=G[1,0], hspace=1.5)

G1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=G[0,0], width_ratios=[1,2], wspace=0.7)#,  wspace=0.8) , width_ratios=[1,1,1,1.2]
G2 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=G_lower[0,0], width_ratios=[2,1.5], wspace=0.3)#,  wspace=0.8)
G3 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=G_lower[1,0], width_ratios=[2,1.5], wspace=0.3)
G4 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=G_lower[2,0], width_ratios=[2,1.5], wspace=0.3)
G5 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=G_lower[3,0], width_ratios=[2,1.5], wspace=0.3)

ax_A1 = fig.add_subplot(G1[0,0])
ax_A1.text(-0.4, 1.3,'A', transform=ax_A1.transAxes, fontsize=fs+1)
ax_A2 = fig.add_subplot(G1[1,0])
ax_A = fig.add_subplot(G1[:,1])

ax_B = fig.add_subplot(G2[:,0])
ax_B.set_title('Bias increases/decreases with stimulus/trial variabilty', fontsize=fs, pad=10, loc='left')
ax_B.text(-0.3, 1.3,'B', transform=ax_B.transAxes, fontsize=fs+1)
ax_B1 = fig.add_subplot(G2[0,1])
ax_B2 = fig.add_subplot(G2[1,1])

ax_C = fig.add_subplot(G3[:,0])
ax_C.set_title('Bias independent of trial variabilty if stimulus is not noisy', fontsize=fs, pad=10, loc='left')
ax_C.text(-0.3, 1.4,'C', transform=ax_C.transAxes, fontsize=fs+1)
ax_C1 = fig.add_subplot(G3[0,1])
ax_C2 = fig.add_subplot(G3[1,1])

ax_D = fig.add_subplot(G4[:,0])
ax_D.set_title('Bias independent of stimulus noise if stimulus mean is fixed', fontsize=fs, pad=10, loc='left')
ax_D.text(-0.3, 1.4,'D', transform=ax_D.transAxes, fontsize=fs+1)
ax_D1 = fig.add_subplot(G4[0,1])
ax_D2 = fig.add_subplot(G4[1,1])

ax_E = fig.add_subplot(G5[:,0])
ax_E.set_title('Bias depeneds on trial duration', fontsize=fs, pad=10, loc='left')
ax_E.text(-0.3, 1.2,'E', transform=ax_E.transAxes, fontsize=fs+1)
ax_E1 = fig.add_subplot(G5[0,1])


# %% Illustrate input condition for A

std_stims = np.array([2, 1]) # 1, 7
mean_trails = np.array([20,20])
std_trails = np.array([3, 3])

plot_illustration_input_cond(std_stims, mean_trails, std_trails, ax_1 = ax_A1, ax_2 = ax_A2, labels=True)

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
                                  min_means=min_means, max_means=max_means, m_std=m_std, n_std=n_std, ax=ax_A)


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
                            label_text, fs=6, ax = ax_B)


# %% Illustrate the predictions

plot_illustration_bias_results(ax1=ax_B1, ax2=ax_B2)

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
                                  min_means=min_means, max_means=max_means, m_std=m_std, n_std=n_std, ax=ax_C) 
    
    # Illustrate input condition for D
    std_stims = np.array([1e-1, 1e-1])
    mean_trails = np.array([20, 20])
    std_trails = np.array([5.5, 3])
    slopes = np.array([weighted_output_1[0], weighted_output_2[0]])
    
    plot_illustration_input_cond(std_stims, mean_trails, std_trails, slopes=slopes, ax_1 = ax_C1, ax_2 = ax_C2)


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
                                  show_marker_inset=True, ax = ax_D) 
    
    # Illustrate input condition for D

    std_stims = np.array([2, 3])
    mean_trails = np.array([15, 15])
    std_trails = np.array([0.01, 0.01])
    slopes = np.array([weighted_output_1[0], weighted_output_2[0]])
    
    plot_illustration_input_cond(std_stims, mean_trails, std_trails, slopes=slopes, ax_1 = ax_D1, ax_2 = ax_D2)#, ax3=ax_E3)
    
    
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
    plot_slope_trail_duration(trial_durations, fitted_slopes_1, fitted_slopes_2, label_text, fs=6, ax=ax_E)
    
    # plot illustration trail duration
    plot_illustration_trial_duration(ax_1=ax_E1)
    

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
