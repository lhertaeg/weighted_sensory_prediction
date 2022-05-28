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

# %%


def plot_manipulation_results(manipulations, fraction_of_sensory_input_in_output, flg_limit_case):

    plt.figure()
    ax = plt.subplot(1,1,1)    

    label_text = ['nE', 'pE', 'nD', 'pD', 'PVv', 'PVm', 'SOM', 'VIP']
    colors_cells = [Col_Rate_nE, Col_Rate_pE, Col_Rate_nD, Col_Rate_pD, Col_Rate_PVv, Col_Rate_PVm, Col_Rate_SOM, Col_Rate_VIP]
    linestyles = ['-', '-', '--', '--', '-', '-', '-', '-']

    for  id_cell in range(8):
        ax.plot(manipulations, fraction_of_sensory_input_in_output[:,id_cell], label=label_text[id_cell], 
                color=colors_cells[id_cell], ls = linestyles[id_cell])
    
    ax.legend(loc=0)
    ax.set_ylabel('Fraction sensory')
    ax.set_xlabel('manipulation strength')
    if flg_limit_case==0:
        ax.set_title('Sensory input uncertain (noiy), prediction reliable')
    elif flg_limit_case==1:
        ax.set_title('Sensory input highly reliable, prediction uncertain (noiy)')
    sns.despine(ax=ax)


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


def plot_alpha_para_exploration_ratios(fraction_sensory, para_tested_first, para_tested_second, para_first_denominator, para_second_denominator, 
                                       every_n_ticks, xlabel='', ylabel='', vmin=0, vmax=1, decimal=dtype(1e5), title='', cmap=cmap_sensory_prediction):
    
    plt.figure(tight_layout=True)
    
    index = np.round(decimal*para_tested_first/para_first_denominator)/decimal
    columns = np.round(decimal*para_tested_second/para_second_denominator)/decimal
    
    ax = sns.heatmap(fraction_sensory, vmin=vmin, vmax=vmax, cmap=cmap, 
                      xticklabels=columns, yticklabels=index)
    
    ax.set_xticks(np.arange(0.5,len(columns),every_n_ticks))
    ax.set_xticklabels(columns[0::every_n_ticks])
    
    ax.set_yticks(np.arange(0.5,len(index),every_n_ticks))
    ax.set_yticklabels(index[0::every_n_ticks])
    
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    

def plot_alpha_para_exploration(fraction_sensory_median, para_tested_first, para_tested_second, every_n_ticks, xlabel='', 
                                ylabel='', vmin=0, vmax=1, decimal = 1e2, title='', cmap = cmap_sensory_prediction):
    
    plt.figure(tight_layout=True)
    index = np.round(decimal*para_tested_first)/decimal
    columns = np.round(decimal*para_tested_second)/decimal
    
    data = pd.DataFrame(fraction_sensory_median, columns=columns, index=index)
    ax = sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=cmap, xticklabels=every_n_ticks, yticklabels=every_n_ticks)
    
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    

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