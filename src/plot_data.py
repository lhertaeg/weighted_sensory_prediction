#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:18:17 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from src.functions_simulate import random_uniform_from_moments, random_gamma_from_moments
from src.functions_simulate import random_binary_from_moments, random_binary_from_moments, random_lognormal_from_moments


# %% set colors

Col_Rate_E = '#9E3039'
Col_Rate_nE = '#955F89'
Col_Rate_nD = '#BF9BB8' 
Col_Rate_pE = '#CB9173' 
Col_Rate_pD = '#DEB8A6' 
Col_Rate_PVv = '#508FCE' 
Col_Rate_PVm = '#2B6299' 
Col_Rate_SOM = '#79AFB9' 
Col_Rate_VIP = '#39656D' 

color_sensory = '#D76A03'
color_prediction = '#19535F'

cmap_sensory_prediction = LinearSegmentedColormap.from_list(name='cmap_sensory_prediction', 
                                                            colors=['#19535F','#fefee3','#D76A03'])

# %% plot functions

def plot_weighting_limit_case_example(n_trials, trial_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, 
                            variance_prediction, alpha, beta, weighted_output, time_plot = 0.8, plot_legend=True,
                            flg_fraction_only=False, figsize=(12,3), fs=12):
    
    if flg_fraction_only:
        f, ax3 = plt.subplots(1, 1, figsize=figsize, sharex=True, tight_layout=True)
    else:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, sharex=True, tight_layout=True)
    
    time = np.arange(len(stimuli))/trial_duration
    
    if not flg_fraction_only:
        ax1.plot(time[time > time_plot * time[-1]], stimuli[time > time_plot * time[-1]], color='#D76A03', label='stimulus')
        ax1.plot(time[time > time_plot * time[-1]], weighted_output[time > time_plot * time[-1]], color='#5E0035', label='weighted output')
        #ax1.axvspan(time_inset*time[-1],time[-1], color='#F5F5F5')
        ax1.set_ylabel('Activity', fontsize=fs)
        ax1.set_xlabel('Time (#trials)', fontsize=fs)
        #ax1.set_title('Sensory inputs and predictions')
        if plot_legend:
            ax1.legend(loc=0)#, ncol=2)
        ax1.set_xlim([time_plot * time[-1],time[-1]])
        ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax1.tick_params(axis='both', labelsize=fs)
        sns.despine(ax=ax1)
        
        # inset_ax1 = inset_axes(ax1, width="30%", height="30%", loc=4)
        # inset_ax1.plot(time[time>time_inset*time[-1]], stimuli[time>time_inset*time[-1]], color='#D76A03', label='stimulus')
        # inset_ax1.plot(time[time>time_inset*time[-1]],weighted_output[time>time_inset*time[-1]], color='#5E0035', label='weighted output')
        # inset_ax1.set_xticks([])
        # inset_ax1.set_yticks([])
        # inset_ax1.set_facecolor('#F5F5F5')
        
        #ax2.plot(time,stimuli, color='#D76A03', label='stimulus')
        ax2.plot(time[time > time_plot * time[-1]], prediction[time > time_plot * time[-1]], color='#19535F', label='prediction')
        ax2.plot(time[time > time_plot * time[-1]], mean_of_prediction[time > time_plot * time[-1]], color='#70A9A1', label='mean of prediction')
        ax2.set_xlabel('Time (#trials)')
        if plot_legend:
            ax2.legend(loc=0)#, ncol=2)
        ax2.set_xlim([time_plot * time[-1],time[-1]])
        ax2.set_ylim([ax1.get_ylim()[0], ax1.get_ylim()[1]])
        ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax2.tick_params(axis='both', labelsize=fs)
        sns.despine(ax=ax2)
    
    ax3.plot(time[time > time_plot * time[-1]], alpha[time > time_plot * time[-1]], color='#D76A03', label='stimulus')
    ax3.plot(time[time > time_plot * time[-1]], beta[time > time_plot * time[-1]], color='#19535F', label='prediction')
    ax3.set_ylabel('Weights', fontsize=fs)
    ax3.set_xlabel('Time (#trials)', fontsize=fs)
    #if plot_legend:
    #    ax3.legend(loc=0)#, ncol=2)
    ax3.set_xlim([time_plot * time[-1],time[-1]])
    ax3.set_ylim([0,1])
    ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.tick_params(axis='both', labelsize=fs)
    sns.despine(ax=ax3)
    
    

def plot_mse_test_distributions(mse, mean=None, std=None, SEM=None, title=None, 
                                dist_types=None, x_lim = None, pa=0.8):
    
    ### show mean squared error
    fig = plt.figure(tight_layout=True)
    plt.locator_params(nbins=3)
    ax = plt.gca()
    
    time = np.arange(0, np.size(mse,1), 1)
        
    num_rows = np.size(mse,0)
    colors = ['#562c2c', '#f2542d', '#0e9594', '#127475', '#05668d'] #sns.color_palette("tab10", n_colors=num_rows)
    
    for i in range(num_rows):
        ax.plot(time/time[-1], mse[i,:], color=colors[i])
        if SEM is not None:
            ax.fill_between(time/time[-1], mse[i,:] - SEM[i,:], mse[i,:] + SEM[i,:], 
                            color=colors[i], alpha=0.5)
    if x_lim is not None:   
        ax.set_xlim(x_lim/time[-1])
            
        # if inset_labels is not None:
            
        #     for i in range(5):
                
        #         if i==0:
        #             ax1 = fig.add_axes([0.3,0.65,0.15,0.15])
        #             stimuli = np.random.normal(mean, sd, size=1000)
        #         elif i==1:
        #             ax1 = fig.add_axes([0.5,0.65,0.15,0.15])
        #             stimuli = random_uniform_from_moments(mean, sd, 1000)
        #         elif i==2:
        #             ax1 = fig.add_axes([0.7,0.65,0.15,0.15])
        #             stimuli = random_lognormal_from_moments(mean, sd, 1000)
        #         elif i==3:
        #             ax1 = fig.add_axes([0.3,0.35,0.15,0.15])
        #             stimuli = random_gamma_from_moments(mean, sd, 1000)
        #         elif i==4:
        #             ax1 = fig.add_axes([0.5,0.35,0.15,0.15])
        #             stimuli = random_binary_from_moments(mean, sd, 1000)
                    
        #         ax1.hist(stimuli, color=colors[i], density=True)
        #         ax1.set_xlim([0,15])
        #         ax1.set_yticks([])
        #         ax1.set_xticks([])
        #         ax1.set_title(inset_labels[i], fontsize=10)
        #         sns.despine(ax=ax1)

    
    ax.set_xlabel('Time / trial duration')
    ax.set_ylabel('MSE')
    if title is not None:
        ax.set_title(title)
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax)
    
    ### show different distributions
    if dist_types is not None:
        num_dist = len(dist_types)
        f, axs = plt.subplots(1, num_dist, figsize=(15,3), tight_layout=True, sharex=True)
        
        for i, dist_type in enumerate(dist_types):
            
            if dist_type=='uniform':
                rnds = random_uniform_from_moments(mean, std, 10000)
            elif dist_type=='normal':
                rnds = np.random.normal(mean, std, size=10000)
            elif dist_type=='lognormal':
                rnds = random_lognormal_from_moments(mean, std, 10000)
            elif dist_type=='gamma':
                rnds = random_gamma_from_moments(mean, std, 10000)
            elif dist_type=='binary_equal_prop':
                rnds = random_binary_from_moments(mean, std, 10000)
            elif dist_type=='binary_unequal_prop':
                rnds = random_binary_from_moments(mean, std, 10000, pa=pa)
            
            axs[i].hist(rnds, density=True, color=colors[i])
            sns.despine(ax=axs[i])



def plot_mse_heatmap(end_of_initial_phase, means_tested, variances_tested, mse, vmax = None, 
                     title=None, show_mean=True, fs=5):
    
    plt.figure()
    
    if show_mean:
        color_cbar = LinearSegmentedColormap.from_list(name='mse_mean', colors=['#FEFAE0', '#19535F'])
    else:
        color_cbar = LinearSegmentedColormap.from_list(name='mse_variance', colors=['#FEFAE0', '#452144'])
    
    MSE_steady_state = np.mean(mse[:, :, end_of_initial_phase:],2)
    data = pd.DataFrame(MSE_steady_state.T, index=variances_tested, columns=means_tested)
    if vmax is None:
        vmax = np.ceil(np.max(MSE_steady_state))
    ax1 = sns.heatmap(data, vmin=0, vmax=vmax, cmap=color_cbar, xticklabels=3, yticklabels=2,
                      cbar_kws={'label': r'MSE$_\mathrm{\infty}$', 'ticks': [0, vmax]}) # , 'pad':0.01
    ax1.set_xlabel('mean of stimuli', fontsize=fs)
    ax1.set_ylabel('variance of stimuli', fontsize=fs)
    if title is not None:
        ax1.set_title(title, fontsize=fs, pad=20)
    
    ax1.invert_yaxis()
    
    sns.despine(ax=ax1, bottom=False, top=False, right=False, left=False)
    
    

def plot_example_mean(stimuli, trial_duration, m_neuron, perturbation_time = None, 
                      figsize=(6,4.5), ylim_mse=None, lw=1, fs=5, tight_layout=True, 
                      legend_flg=True, mse_flg=True):
    
    ### set figure architecture
    if mse_flg:
        f, ((ax1), (ax2)) = plt.subplots(2, 1, sharex=True, figsize=figsize,
                                         gridspec_kw={'height_ratios': [5, 1]}, tight_layout=tight_layout)
    else:   
        f, ax1 = plt.subplots(1, 1, figsize=figsize, tight_layout=tight_layout)
    
    ### set number of ticks
    ax1.locator_params(nbins=3)
    if mse_flg:
        ax2.locator_params(nbins=3)
    
    ### plot 
    trials = np.arange(len(stimuli))/trial_duration
    running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
    
    ax1.axvspan(perturbation_time, trials[-1], alpha=0.1, facecolor='#1E000E', edgecolor=None)
    ax1.plot(trials, stimuli, color='#D76A03', label='stimulus', alpha=0.3, lw=lw)
    ax1.plot(trials, running_average, color='#D76A03', label='running average', lw=lw)
    ax1.plot(trials, m_neuron, color='#19535F', label='predicted mean', lw=lw)
    ax1.set_xlim([0,max(trials)])
    ax1.set_ylabel('Activity (1/s)', fontsize=fs)
    ax1.set_title('Mean of sensory inputs', fontsize=fs+1)
    ax1.tick_params(axis='both', labelsize=fs)
    if legend_flg:
        ax1.legend(loc=0, ncol=2, frameon=False, handlelength=1)
    sns.despine(ax=ax1)
    
    if mse_flg:
        ax2.axvspan(perturbation_time, trials[-1], alpha=0.1, facecolor='#1E000E', edgecolor=None)
        ax2.plot(trials, (running_average - m_neuron)**2, color='#AF1B3F')
        if ylim_mse is not None:
            ax2.set_ylim(ylim_mse)
        ax2.set_ylabel('MSE', fontsize=fs)
        ax2.set_xlabel('Time (#stimuli)', fontsize=fs)
        sns.despine(ax=ax2)
    
    
def plot_example_variance(stimuli, trial_duration, v_neuron, perturbation_time = None, 
                          figsize=(6,4.5), lw=1, fs=5, tight_layout=True, 
                          legend_flg=True, mse_flg=True):
    
    if mse_flg:
        f, ((ax1), (ax2)) = plt.subplots(2, 1, sharex=True, figsize=figsize,
                                     gridspec_kw={'height_ratios': [5, 1]}, tight_layout=tight_layout)
    else:
        f, ax1 = plt.subplots(1, 1, figsize=figsize, tight_layout=tight_layout)
    
    ax1.locator_params(nbins=3)
    if mse_flg:
        ax2.locator_params(nbins=3)
    
    mean_running = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
    momentary_variance = (stimuli - mean_running)**2
    variance_running = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1)
    trials = np.arange(len(stimuli))/trial_duration
    
    ax1.axvspan(perturbation_time, trials[-1], alpha=0.1, facecolor='#1E000E', edgecolor=None)
    ax1.plot(trials, momentary_variance, color='#D76A03', label='variance of stimuli', alpha=0.3, lw=lw)
    ax1.plot(trials, variance_running, color='#D76A03', label='running average', lw=lw)
    ax1.plot(trials, v_neuron, color='#452144', label='predicted variance', lw=lw)
    ax1.set_xlim([0,max(trials)])
    ax1.set_ylabel('Activity (1/s)', fontsize=fs)
    ax1.set_title('Variance of sensory inputs', fontsize=fs+1)
    ax1.tick_params(axis='both', labelsize=fs)
    if legend_flg:
        ax1.legend(loc=0, ncol=2, frameon=False, handlelength=1)
    sns.despine(ax=ax1)
    
    if mse_flg:
        ax2.axvspan(perturbation_time, trials[-1], alpha=0.1, facecolor='#1E000E', edgecolor=None)
        ax2.plot(trials, (variance_running - v_neuron)**2, color='#AF1B3F')
        ax2.set_ylabel('MSE', fontsize=fs)
        ax2.set_xlabel('Time (#stimuli)', fontsize=fs)
        sns.despine(ax=ax2)
  