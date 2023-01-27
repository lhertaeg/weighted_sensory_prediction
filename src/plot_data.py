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

####
color_stimuli_background = '#FED5AF'
color_running_average_stimuli = '#D76A03'
color_m_neuron = '#19535F'
color_v_neuron = '#452144'
color_mse = '#AF1B3F' 
color_weighted_output = '#5E0035'
color_mean_prediction = '#70A9A1' 

cmap_sensory_prediction = LinearSegmentedColormap.from_list(name='cmap_sensory_prediction', 
                                                            colors=[color_m_neuron,'#fefee3',
                                                                    color_running_average_stimuli])

# %% plot functions


def plot_weight_over_trial(fraction_course, n_trials):
    
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(4,4))
    colors = sns.color_palette('magma', n_colors=5)
    
    for id_stim in range(5):
        
        alpha_split = np.array_split(fraction_course[:,id_stim], n_trials)
        alpha_avg_trial = np.mean(alpha_split,0)
        alpha_std_trial = np.std(alpha_split,0)
        sem = alpha_std_trial/np.sqrt(n_trials)
        trial_fraction = np.linspace(0,1,len(alpha_avg_trial))
        
        ax.plot(trial_fraction, alpha_avg_trial, color=colors[id_stim])
        ax.fill_between(trial_fraction, alpha_avg_trial-sem, alpha_avg_trial+sem, alpha=0.3, color=colors[id_stim])
    
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    
    ax.set_ylabel('Sensory weight')
    ax.set_xlabel('time/trial duration')
    sns.despine(ax=ax)


def plot_transitions_examples(n_trials, trial_duration, stimuli, alpha, beta, weighted_output, 
                              time_plot = 0, ylim=None, xlim=None, plot_ylable=True, lw=1, 
                              figsize=(3.5,5), plot_only_weights=False, fs=12, transition_point=60):
    
    f1, ax1 = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    f1, ax2 = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    
    time = np.arange(len(stimuli))/trial_duration
    
    if not plot_only_weights:
        
        for i in range(n_trials):
            ax1.axvspan(2*i, (2*i+1), color='#F5F4F5')

        ax1.plot(time[time > time_plot * time[-1]], stimuli[time > time_plot * time[-1]], 
             color=color_stimuli_background, lw=lw, marker='|', ls="None")  
        ax1.plot(time[time > time_plot * time[-1]], weighted_output[time > time_plot * time[-1]], color=color_m_neuron, label='weighted output')
        if plot_ylable:
            ax1.set_ylabel('Activity', fontsize=fs)
        else:
            ax1.set_ylabel('Activity', color='white', fontsize=fs)
            ax1.tick_params(axis='y', colors='white')
        #ax1.set_xlabel('Time (#trials)')
        ax1.set_xlim([time_plot * time[-1],time[-1]])
        if ylim is not None:
            ax1.set_ylim(ylim)
        if xlim is not None:
            ax1.set_xlim(xlim)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax1.tick_params(axis='both', labelsize=fs)
        sns.despine(ax=ax1)
    
    for i in range(n_trials):
        ax2.axvspan(2*i, (2*i+1), color='#F5F4F5')
    
    ax2.plot(time[time > time_plot * time[-1]], alpha[time > time_plot * time[-1]], color=color_running_average_stimuli, label='stimulus')
    ax2.plot(time[time > time_plot * time[-1]], beta[time > time_plot * time[-1]], color=color_m_neuron, label='prediction')
    if plot_ylable:
        ax2.set_ylabel('Weights', fontsize=fs)
    else:
        ax2.set_ylabel('Weights', color='white', fontsize=fs)
        ax2.tick_params(axis='y', colors='white')
    ax2.axvline(transition_point, color='k', ls='--')
    ax2.set_xlabel('Time (#trials)', fontsize=fs)
    ax2.set_xlim([time_plot * time[-1],time[-1]])
    ax2.set_ylim([0,1.05])
    if xlim is not None:
        ax2.set_xlim(xlim)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.tick_params(axis='both', labelsize=fs)
    sns.despine(ax=ax2)
    


def plot_fraction_sensory_heatmap(fraction_sensory_median, para_tested_first, para_tested_second, every_n_ticks, xlabel='', 
                                ylabel='', vmin=0, vmax=1, decimal = 1e2, title='', cmap = cmap_sensory_prediction,
                                figsize=(5,5), fs=12):
    
    plt.figure(tight_layout=True, figsize=figsize)
    index = np.round(decimal*para_tested_first)/decimal
    columns = np.round(decimal*para_tested_second)/decimal
    
    data = pd.DataFrame(fraction_sensory_median, columns=columns, index=index)
    ax = sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=cmap, xticklabels=every_n_ticks, 
                     yticklabels=every_n_ticks, cbar_kws={'label': 'Sensory \nweight'})
    
    ax.locator_params(nbins=2)
    ax.tick_params(axis='both', labelsize=fs)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.yaxis.label.set_size(fs)
    
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_title(title)


def plot_weighting_limit_case_example(n_trials, trial_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, 
                            variance_prediction, alpha, beta, weighted_output, time_plot = 0.8, plot_legend=True,
                            figsize=(4,3), fs=12, lw=1):
    
    f1, ax1 = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    f1, ax2 = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    f1, ax3 = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    
    time = np.arange(len(stimuli))/trial_duration
    
    for i in range(n_trials):
        ax1.axvspan(2*i, (2*i+1), color='#F5F4F5')
    ax1.plot(time[time > time_plot * time[-1]], stimuli[time > time_plot * time[-1]], 
             color=color_stimuli_background, lw=lw, marker='|', ls="None") # , label='stimuli'
    ax1.plot(time[time > time_plot * time[-1]], weighted_output[time > time_plot * time[-1]], 
             color=color_weighted_output, label='weighted output')  
    ax1.set_ylabel('Activity', fontsize=fs)
    ax1.set_xlabel('Time (#trials)', fontsize=fs)
    if plot_legend:
        ax1.legend(loc=0, frameon=False)
    ax1.set_xlim([time_plot * time[-1],time[-1]])
    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax1.tick_params(axis='both', labelsize=fs)
    sns.despine(ax=ax1)
    
    for i in range(n_trials):
        ax2.axvspan(2*i, (2*i+1), color='#F5F4F5')
    ax2.plot(time[time > time_plot * time[-1]], prediction[time > time_plot * time[-1]], 
             color=color_m_neuron, label='prediction')
    ax2.plot(time[time > time_plot * time[-1]], mean_of_prediction[time > time_plot * time[-1]], 
             color=color_mean_prediction, label='mean of prediction')
    ax2.set_ylabel('Activity', fontsize=fs)
    ax2.set_xlabel('Time (#trials)')
    if plot_legend:
        ax2.legend(loc=0, frameon=False)
    ax2.set_xlim([time_plot * time[-1],time[-1]])
    ax2.set_ylim([ax1.get_ylim()[0], ax1.get_ylim()[1]])
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.tick_params(axis='both', labelsize=fs)
    sns.despine(ax=ax2)
    
    for i in range(n_trials):
        ax3.axvspan(2*i, (2*i+1), color='#F5F4F5')
    ax3.plot(time[time > time_plot * time[-1]], alpha[time > time_plot * time[-1]], 
             color=color_running_average_stimuli, label='feedforward')
    ax3.plot(time[time > time_plot * time[-1]], beta[time > time_plot * time[-1]], 
             color=color_m_neuron, label='feedback')
    ax3.set_ylabel('Weights', fontsize=fs)
    ax3.set_xlabel('Time (#trials)', fontsize=fs)
    if plot_legend:
        ax3.legend(loc=0, frameon=False)
    ax3.set_xlim([time_plot * time[-1],time[-1]])
    ax3.set_ylim([0,1])
    ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.tick_params(axis='both', labelsize=fs)
    sns.despine(ax=ax3)
    
    

def plot_mse_test_distributions(mse, dist_types=None, mean=None, std=None, SEM=None, title=None,
                                plot_dists=False, x_lim = None, pa=0.8, inset_steady_state=False,
                                fig_size=(5,3), fs = 5):
    
    ### show mean squared error
    fig = plt.figure(figsize = fig_size, tight_layout=True)
    ax = plt.gca()
    ax.locator_params(nbins=3)
    
    time = np.arange(0, np.size(mse,1), 1)
        
    num_rows = np.size(mse,0)
    colors = ['#093A3E', '#E0A890', '#99C24D', '#6D213C', '#0E9594'] #sns.color_palette("tab10", n_colors=num_rows)
    
    for i in range(num_rows):
        if dist_types is not None:
            ax.plot(time/time[-1], mse[i,:], color=colors[i], label=dist_types[i])
        else:
            ax.plot(time/time[-1], mse[i,:], color=colors[i])
            
        if SEM is not None:
            ax.fill_between(time/time[-1], mse[i,:] - SEM[i,:], mse[i,:] + SEM[i,:], 
                            color=colors[i], alpha=0.3)
    if x_lim is not None:   
        ax.set_xlim(x_lim/time[-1])
            
    if inset_steady_state:
        
        ax1 = fig.add_axes([0.5,0.5,0.4,0.3])
        f = 0.5
        
        for i in range(num_rows):
            ax1.plot(time[time>time[-1]*f]/time[-1], mse[i,time>time[-1]*f], color=colors[i])
            if SEM is not None:
                ax1.fill_between(time[time>time[-1]*f]/time[-1], mse[i,time>time[-1]*f] - SEM[i,time>time[-1]*f], 
                                mse[i,time>time[-1]*f] + SEM[i,time>time[-1]*f], color=colors[i], alpha=0.3)
        
        ax1.locator_params(nbins=3)
        ax1.tick_params(axis='both', labelsize=fs)
        sns.despine(ax=ax1)

    if dist_types is not None:
        ax.legend(loc=4, ncol=2, handlelength=1, frameon=False, fontsize=fs)
    
    ax.tick_params(axis='both', labelsize=fs)
    ax.set_xlabel('Time / trial duration', fontsize=fs)
    ax.set_ylabel('MSE', fontsize=fs)
    if title is not None:
        ax.set_title(title, fontsize=fs)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    sns.despine(ax=ax)
    
    ### show different distributions
    if plot_dists:
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



def plot_mse_heatmap(end_of_initial_phase, means_tested, variances_tested, mse, vmax = None, lw = 1, 
                     title=None, show_mean=True, fs=5, figsize=(5,5), x_example = None, y_example = None):
    
    f = plt.subplots(1,1, figsize=figsize, tight_layout=True)
    
    if show_mean:
        color_cbar = LinearSegmentedColormap.from_list(name='mse_mean', colors=['#FEFAE0', color_m_neuron])
    else:
        color_cbar = LinearSegmentedColormap.from_list(name='mse_variance', colors=['#FEFAE0', color_v_neuron])
    
    MSE_steady_state = np.mean(mse[:, :, end_of_initial_phase:],2)
    data = pd.DataFrame(MSE_steady_state.T, index=variances_tested, columns=means_tested)
    if vmax is None:
        vmax = np.ceil(10 * np.max(MSE_steady_state))/10 # np.ceil(np.max(MSE_steady_state))
    ax1 = sns.heatmap(data, vmin=0, vmax=vmax, cmap=color_cbar, xticklabels=3, yticklabels=2,
                      cbar_kws={'label': r'MSE$_\mathrm{\infty}$', 'ticks': [0, vmax]})
    
    ax1.invert_yaxis()
    
    mx, nx = np.polyfit([means_tested.min()-0.5, means_tested.max()+0.5], ax1.get_xlim(), 1)
    my, ny = np.polyfit([variances_tested.min()-0.5, variances_tested.max()+0.5], ax1.get_ylim(), 1)
    
    ax1.plot(mx * np.array([means_tested.min()-0.5, means_tested.max()+0.5]) + nx, 
             my * np.array([means_tested.min()-0.5, means_tested.max()+0.5]) + ny, color='k', ls=':')
    
    
    if x_example is not None:
        ax1.plot(mx * x_example + nx, my * y_example + ny, marker='*', color='k')
        
        
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.set_xlabel('mean of stimuli', fontsize=fs)
    ax1.set_ylabel('variance of stimuli', fontsize=fs)
    ax1.figure.axes[-1].yaxis.label.set_size(fs)
    ax1.figure.axes[-1].tick_params(labelsize=fs)
    
    if title is not None:
        ax1.set_title(title, fontsize=fs, pad=10)
    
    
    
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
    
    ax1.plot(trials, stimuli, color=color_stimuli_background, label='stimuli', lw=lw, marker='|', ls="None")
    ax1.plot(trials, running_average, color=color_running_average_stimuli, label='running average', lw=lw)
    ax1.plot(trials, m_neuron, color=color_m_neuron, label='M neuron', lw=lw)
    ax1.set_xlim([0,max(trials)])
    ax1.set_ylabel('Activity (1/s)', fontsize=fs)
    ax1.set_title('Activity of M neuron encodes mean of stimuli', fontsize=fs+1)
    ax1.tick_params(axis='both', labelsize=fs)
    if legend_flg:
        ax1.legend(loc=4, ncol=3, handlelength=1, frameon=False, fontsize=fs)
    sns.despine(ax=ax1)
    
    if mse_flg:
        ax2.plot(trials, (running_average - m_neuron)**2, color=color_mse)
        if ylim_mse is not None:
            ax2.set_ylim(ylim_mse)
        ax2.set_ylabel('MSE', fontsize=fs)
        ax2.set_xlabel('Time / trial duration', fontsize=fs)
        ax2.tick_params(axis='both', labelsize=fs)
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
    
    ax1.plot(trials, momentary_variance, color=color_stimuli_background, label='instantaneous variance',
             lw=lw, ls="None", marker='|')
    ax1.plot(trials, variance_running, color=color_running_average_stimuli, label='running average', lw=lw)
    ax1.plot(trials, v_neuron, color=color_v_neuron, label='V neuron', lw=lw)
    ax1.set_xlim([0,max(trials)])
    ax1.set_ylabel('Activity (1/s)', fontsize=fs)
    ax1.set_title('Activity of V neuron encodes variance of stimuli', fontsize=fs+1)
    ax1.tick_params(axis='both', labelsize=fs)
    if legend_flg:
        ax1.legend(loc=1, ncol=2, frameon=False, handlelength=1, fontsize=fs)
    sns.despine(ax=ax1)
    
    if mse_flg:
        ax2.plot(trials, (variance_running - v_neuron)**2, color=color_mse)
        ax2.set_ylabel('MSE', fontsize=fs)
        ax2.set_xlabel('Time / trial duration', fontsize=fs)
        ax2.tick_params(axis='both', labelsize=fs)
        sns.despine(ax=ax2)
  