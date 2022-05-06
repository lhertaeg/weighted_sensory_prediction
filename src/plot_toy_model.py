#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:14:44 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap

# %% set colors

color_sensory = '#D76A03'
color_prediction = '#19535F'

cmap_sensory_prediction = LinearSegmentedColormap.from_list(name='cmap_sensory_prediction', 
                                                            colors=['#19535F','#fefee3','#D76A03'])

# %%


def plot_fraction_sensory_comparsion(fraction_sensory, fraction_sensory_std, n_repeats, label_text, cmap='rocket_r'):
    
    plt.figure()
    ax = plt.subplot(1,1,1)
    cmap_stim_duraations = sns.color_palette(cmap, n_colors=np.size(fraction_sensory,0))
    #sns.light_palette(cmap, n_colors=np.size(fraction_sensory,0),reverse=True)
    
    for i in range(np.size(fraction_sensory,0)):
        ax.plot(fraction_sensory[i,:], color=cmap_stim_duraations[i], label=str(label_text[i]), lw=2)
        ax.fill_between(np.arange(len(fraction_sensory[i,:])), fraction_sensory[i,:]-fraction_sensory_std[i,:]/np.sqrt(n_repeats), 
                        fraction_sensory[i,:]+fraction_sensory_std[i,:]/np.sqrt(n_repeats), 
                        color=cmap_stim_duraations[i], alpha=0.5)
        
    ax.legend(loc=0)
    ax.set_ylabel('Fraction')
    ax.set_xlabel('stimuli')
    sns.despine(ax=ax)


def plot_alpha_para_exploration_ratios(fraction_sensory_median, para_tested_first, para_tested_second, para_first_denominator,
                                       para_second_denominator, every_n_ticks, xlabel='', ylabel='', vmin=0, vmax=1, decimal=1e5):
    
    plt.figure(tight_layout=True)
    index = np.round(decimal*para_tested_first/para_first_denominator)/decimal
    columns = np.round(decimal*para_tested_second/para_second_denominator)/decimal
    data = pd.DataFrame(fraction_sensory_median, columns=columns, index=index)
    ax = sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=cmap_sensory_prediction, 
                     xticklabels=every_n_ticks, yticklabels=every_n_ticks)
    ax.invert_yaxis()
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    

def plot_alpha_para_exploration(fraction_sensory_median, para_tested_first, para_tested_second, every_n_ticks, xlabel='', 
                                ylabel='', vmin=0, vmax=1, decimal = 1e2):
    
    plt.figure(tight_layout=True)
    index = np.round(decimal*para_tested_first)/decimal
    columns = np.round(decimal*para_tested_second)/decimal
    data = pd.DataFrame(fraction_sensory_median, columns=columns, index=index)
    ax = sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=cmap_sensory_prediction,
                     xticklabels=every_n_ticks, yticklabels=every_n_ticks)
    ax.invert_yaxis()
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    

def plot_limit_case(n_stimuli, stimulus_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, variance_prediction,
                    alpha, beta, weighted_output):
    
    f, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(15,8))
    
    for i in range(n_stimuli):
        ax1.axvline((i+1)*stimulus_duration, ls='--', color='k', alpha=0.2)
    ax1.plot(stimuli, color='#D76A03', label='stimulus')
    #ax1.plot(mean_per_stimulus, color='g')
    ax1.plot(prediction, color='#19535F', label='prediction')
    ax1.plot(mean_of_prediction, color='#BFCC94', label='mean of prediction')
    ax1.axhline(np.mean(stimuli), ls=':')
    ax1.set_xlim([0,len(stimuli)])
    ax1.set_ylabel('Activity')
    ax1.set_title('Sensory inputs and predictions')
    ax1.legend(loc=0, ncol=3)
    sns.despine(ax=ax1)
    
    
    var_per_stimulus = np.var(np.array_split(stimuli, n_stimuli),1)
    for i in range(n_stimuli):
        ax2.axvline((i+1)*stimulus_duration, ls='--', color='k', alpha=0.2)
    for i in range(n_stimuli):
        ax2.axhline(var_per_stimulus[i], i/n_stimuli, (i+1)/n_stimuli, color='g')
    ax2.plot(variance_per_stimulus, color='#D76A03', label='var(stimulus)')
    ax2.plot(variance_prediction, color='#19535F', label='var(prediction)')
    #ax2.axhline(np.var(stimuli), ls=':')
    ax2.set_xlim([0,len(stimuli)])
    ax2.set_ylabel('Activity')
    ax2.set_title('Variances of sensory inputs and predictions')
    ax2.legend(loc=0, ncol=2)
    sns.despine(ax=ax2)
    
    for i in range(n_stimuli):
        ax3.axvline((i+1)*stimulus_duration, ls='--', color='k', alpha=0.2)
    ax3.plot(stimuli, color='#D76A03', label='stimulus')
    ax3.plot(prediction, color='#19535F', label='prediction')
    ax3.plot(weighted_output, color='#582630', label='weighted output')
    ax3.set_xlim([0,len(stimuli)])
    ax3.set_ylabel('Activity')
    ax3.set_xlabel('Time (ms)')
    ax3.set_title('Weighted output compared to sensory inputs & predictions')
    ax3.legend(loc=0, ncol=3)
    sns.despine(ax=ax3)
    
    for i in range(n_stimuli):
        ax4.axvline((i+1)*stimulus_duration, ls='--', color='k', alpha=0.2)
    ax4.plot(alpha, color='#D76A03', label='stimulus')
    ax4.plot(beta, color='#19535F', label='prediction')
    ax4.set_ylabel('Fraction')
    ax4.set_xlabel('Time (ms)')
    ax4.set_title('Fraction of sensory input & prediction in weighted output')
    sns.despine(ax=ax4)