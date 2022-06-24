#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:20:53 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

dtype = np.float32

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

# %% functions


def plot_limit_case_pred(n_stimuli, stimulus_duration, stimuli, trial_mean, prediction, mean_of_prediction, 
                         variance_per_stimulus, variance_prediction, alpha, beta, weighted_prediction):
    
    f, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(15,8))
    
    
    time = np.arange(len(stimuli))/1000
    
    ax1.plot(time, stimuli, color='#D76A03', label='stimulus')
    #ax1.plot(mean_per_stimulus, color='g')
    ax1.plot(time, prediction, color='#19535F', label='prediction (fast)')
    ax1.plot(time, mean_of_prediction, color='#BFCC94', label='prediction (slow)')
    #ax1.axhline(np.mean(stimuli), ls=':')
    ax1.set_xlim([0,len(stimuli)/1000])
    ax1.set_ylabel('Activity')
    ax1.set_title('Sensory inputs and predictions')
    ax1.legend(loc=0, ncol=3)
    sns.despine(ax=ax1)
    
    var_per_stimulus = np.var(np.array_split(stimuli, n_stimuli),1)
    # for i in range(n_stimuli):
    #     ax2.axhline(var_per_stimulus[i], i/n_stimuli, (i+1)/n_stimuli, color='g')
    ax2.plot(time, variance_per_stimulus, color='#19535F', label=r'E[(S-P$_{fast})^2]$')
    ax2.plot(time, variance_prediction, color='#BFCC94', label=r'E[(P$_{fast}$-P$_{slow})^2]$')
    #ax2.axhline(np.var(stimuli), ls=':')
    ax2.set_xlim([0,len(stimuli)/1000])
    ax2.set_ylabel('Activity')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Variances of fast and slow predictions')
    ax2.legend(loc=0, ncol=2)
    sns.despine(ax=ax2)
    
    ax3.plot(time, np.cumsum((trial_mean - prediction)**2), color='#19535F', label='fast prediction')
    ax3.plot(time, np.cumsum((trial_mean - mean_of_prediction)**2), color='#BFCC94', label='slow prediction')
    ax3.plot(time, np.cumsum((trial_mean - weighted_prediction)**2), color='#582630', label='weighted prediction')
    ax3.set_xlim([0,len(stimuli)/1000])
    ax3.set_ylabel('Sum of MSE over time')
    ax3.set_title('Weighted prediction compared to sensory inputs')
    ax3.legend(loc=0, ncol=3)
    sns.despine(ax=ax3)
    
    ax4.plot(time, alpha, color='#19535F', label='fast prediction')
    ax4.plot(time, beta, color='#BFCC94', label='slow prediction')
    ax4.set_ylabel('Fraction')
    ax4.set_xlabel('Time (s)')
    ax4.set_title('Fraction of fast & slow component in weighted prediction')
    sns.despine(ax=ax4)


def plot_limit_case_new(n_stimuli, stimulus_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, variance_prediction,
                        alpha, beta, weighted_output):
    
    f, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(15,8))
    
    
    time = np.arange(len(stimuli))/1000
    
    ax1.plot(time, stimuli, color='#D76A03', label='stimulus')
    #ax1.plot(mean_per_stimulus, color='g')
    ax1.plot(time, prediction, color='#19535F', label='prediction')
    ax1.plot(time, mean_of_prediction, color='#BFCC94', label='mean of prediction')
    #ax1.axhline(np.mean(stimuli), ls=':')
    ax1.set_xlim([0,len(stimuli)/1000])
    ax1.set_ylabel('Activity')
    ax1.set_title('Sensory inputs and predictions')
    ax1.legend(loc=0, ncol=3)
    sns.despine(ax=ax1)
    
    
    var_per_stimulus = np.var(np.array_split(stimuli, n_stimuli),1)
    # for i in range(n_stimuli):
    #     ax2.axhline(var_per_stimulus[i], i/n_stimuli, (i+1)/n_stimuli, color='g')
    ax2.plot(time, variance_per_stimulus, color='#D76A03', label='var(stimulus)')
    ax2.plot(time, variance_prediction, color='#19535F', label='var(prediction)')
    #ax2.axhline(np.var(stimuli), ls=':')
    ax2.set_xlim([0,len(stimuli)/1000])
    ax2.set_ylabel('Activity')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Variances of sensory inputs and predictions')
    ax2.legend(loc=0, ncol=2)
    sns.despine(ax=ax2)
    
    ax3.plot(time, stimuli, color='#D76A03', label='stimulus')
    ax3.plot(time, prediction, color='#19535F', label='prediction')
    ax3.plot(time, weighted_output, color='#582630', label='weighted output')
    ax3.set_xlim([0,len(stimuli)/1000])
    ax3.set_ylabel('Activity')
    ax3.set_title('Weighted output compared to sensory inputs & predictions')
    ax3.legend(loc=0, ncol=3)
    sns.despine(ax=ax3)
    
    ax4.plot(time, alpha, color='#D76A03', label='stimulus')
    ax4.plot(time, beta, color='#19535F', label='prediction')
    ax4.set_ylabel('Fraction')
    ax4.set_xlabel('Time (s)')
    ax4.set_title('Fraction of sensory input & prediction in weighted output')
    sns.despine(ax=ax4)
    

def plot_manipulations(data, xticklabels, title, ylim=[-100,100]):
    
    plt.figure(tight_layout=True)
    
    x = np.repeat(np.arange(np.size(data,1)), np.size(data,0))
    y = np.ndarray.flatten(data, 'F')
    hue = np.zeros(len(y))
    hue[1::2] = 1
    
    sns.barplot(x, y, hue, palette=['#60B2E5','#FF858D'])#, hue_order=['A','B'])
    
    ax = plt.gca()
    ax.set_xticks(np.arange(np.size(data,1)))
    ax.set_xticklabels(xticklabels)
    ax.axhline(0.0, color='k')
    ax.set_ylabel('Deviation (%)')
    ax.set_title(title)
    ax.set_ylim(ylim)
    
    h, l = ax.get_legend_handles_labels() 
    ax.legend(h, ['deactivate', 'activate'])

    sns.despine(ax=ax, bottom=True)


def plot_mse(trials, mse, ylabel, legend_labels=None):
    
    plt.figure(tight_layout=True)
    ax = plt.gca()
    
    if mse.ndim==1:
        ax.plot(trials, mse, color = '#D76A03', alpha = 1)#, label='running average')
    
    elif mse.ndim==2:
        
        num_rows = np.size(mse,0)
        colors = sns.color_palette("tab10", n_colors=num_rows)
        
        for i in range(num_rows):
            ax.plot(trials, mse[i,:], color=colors[i], label=legend_labels[i])
            
        ax.legend(loc=0)
        
    elif mse.ndim==3:
        
        num_rows = np.size(mse,0)
        num_cols = np.size(mse,1)

        alpha = np.arange(1,num_cols+1)/num_cols
        colors = sns.color_palette("viridis_r", n_colors=num_rows)
        
        for i in range(num_rows):
            for j in range(num_cols):
                ax.plot(trials, mse[i,j,:], color=colors[i], alpha=alpha[j])
    
    ax.set_xlabel('Time (#trials)')
    ax.set_xlim([0,max(trials)])
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax)
    

def plot_prediction(n_stimuli, stimuli, stimulus_duration, prediction):
    
    plt.figure(tight_layout=True)
    ax = plt.gca()
    
    # for i in range(n_stimuli):
    #     ax.axvline((i+1), ls='--', color='k', alpha=0.2)
    
    trials = np.arange(len(stimuli))/stimulus_duration
    ax.plot(trials, stimuli, color='#D76A03', label='stimulus', alpha=0.3)
    ax.plot(trials, np.cumsum(stimuli)/np.arange(1,len(stimuli)+1), color='#D76A03', label='running average')
    ax.plot(trials, prediction, color='#19535F', label='prediction')
    ax.set_xlim([0,max(trials)])
    ax.set_ylabel('Activity')
    ax.set_xlabel('Time (#trials)')
    ax.set_title('Sensory inputs and predictions')
    ax.legend(loc=0, ncol=3)
    sns.despine(ax=ax)
    
    
def plot_variance(n_stimuli, stimuli, stimulus_duration, variance_per_stimulus):
    
    plt.figure(tight_layout=True)
    ax = plt.gca()
    
    # for i in range(n_stimuli):
    #     ax.axvline((i+1), ls='--', color='k', alpha=0.2)
    
    mean_running = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
    momentary_variance = (stimuli - mean_running)**2
    variance_running = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1)
    
    trials = np.arange(len(stimuli))/stimulus_duration
    ax.plot(trials, momentary_variance, color='#D76A03', label='trial variance', alpha=0.3)
    ax.plot(trials, variance_running, color='#D76A03', label='running average')
    ax.plot(trials, variance_per_stimulus, color='#19535F', label='prediction')
    ax.set_xlim([0,max(trials)])
    ax.set_ylabel('Activity')
    ax.set_xlabel('Time (#trials)')
    ax.set_title('Variance of sensory inputs')
    ax.legend(loc=0, ncol=3)
    sns.despine(ax=ax)



# ### plot results

    
    
#     #var_per_stimulus = np.var(np.array_split(stimuli, n_stimuli),1)
    
#     plt.figure()
#     plt.plot(, color='#D76A03', label='variance')
#     plt.plot(var_per_stimulus, 'r')
#     plt.plot(variance_per_stimulus, color='#19535F', label='estimated variance')
#     ax = plt.gca()
#     ax.axhline(np.var(stimuli))   