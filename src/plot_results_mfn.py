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
import matplotlib

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.ticker as ticker

from src.toy_model import random_uniform_from_moments, random_lognormal_from_moments, random_gamma_from_moments, random_binary_from_moments

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
    

def plot_heatmap_perturbation_all(data_original, xticklabels=True, 
                                  yticklabels=True, figsize=(3.5,2.5)):
    
    plt.figure(tight_layout=True, figsize=figsize)
    
    # define labels for annotation
    labels = np.chararray((8,3), unicode=True)
    labels[data_original==1] = '+'
    labels[data_original==-1] = '-'
    labels[data_original==0] = '0'
    
    # edit data to color only the rows that are consistent across configurations
    data = data_original
    sum_columns = np.sum(data,1)
    id_mix = np.where(abs(sum_columns)<3)
    data[id_mix,:] = 0
    
    # plot heatmap
    custom_cmap = LinearSegmentedColormap.from_list(name='custom_cmap', 
                                                    colors=['#91C1DE','#EBE9EC','#DFA6A4']) 
    ax = sns.heatmap(data, annot=labels, linewidths=.5, fmt = '', cmap=custom_cmap, cbar=False)
    
    if xticklabels:
        ax.set_xticklabels(['10','01','11'])
    else:
        ax.set_xticklabels(['10','01','11'])
        ax.tick_params(axis='x', colors='white')
    
    if yticklabels:
        ax.set_yticklabels(['nPE', 'pPE', 'nPE dend', 'pPE dend', 'PVv', 'PVm', 'SOM', 'VIP'], rotation=45)
    else:
        ax.set_yticklabels(['nPE', 'pPE', 'nPE dend', 'pPE dend', 'PVv', 'PVm', 'SOM', 'VIP'], rotation=45)
        ax.tick_params(axis='y', colors='white')

def plot_deviations_upon_perturbations(dev_prediction_steady_10, dev_variance_steady_10,
                                       figsize=(4,5)):

    for i in range(2):
        
        y = np.zeros(16)
        y[0::2] = dev_prediction_steady_10[i::2]
        y[1::2] = dev_variance_steady_10[i::2]
       
        x = np.arange(8)
        x = np.repeat(x,2)
        
        z = ['mean','variance']
        z = np.tile(z,8)
        
        if i==0:
            lvTmp = np.linspace(0,1.0,12)
        else:
            lvTmp = np.linspace(0.4,1.0,12)
    
        plt.figure(tight_layout=True, figsize=figsize)
        ax = sns.barplot(y,x,hue=z, orient='h', palette=matplotlib.cm.Paired(lvTmp))
        ax.axvline(0, color='k')
        
        if i==0:
            ax.set_yticklabels(['nPE', 'pPE', 'nPE dend', 'pPE dend', 'PVv', 'PVm', 'SOM', 'VIP'])
            ax.set_title('Inhibitory perturbation')
        else:
            ax.set_yticklabels([])
            ax.set_title('Excitatory perturbation')
        
        ax.yaxis.tick_right()
        ax.tick_params(right = False)
        ax.set_xlabel('deviation (%)')
        ax.legend(frameon=False, loc=0)
        sns.despine(ax=ax, left=True)    


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


def plot_mse(trials, mean, sd, mse, SEM=None, title=None, inset_labels=None):
    
    fig = plt.figure(tight_layout=True)
    plt.locator_params(nbins=3)
    ax = plt.gca()
    
    if mse.ndim==1:
        ax.plot(trials, mse, color = '#D76A03', alpha = 1)#, label='running average')
    
    elif mse.ndim==2:
        
        num_rows = np.size(mse,0)
        colors = sns.color_palette("tab10", n_colors=num_rows)
        
        for i in range(num_rows):
            ax.plot(trials, mse[i,:], color=colors[i])
            if SEM is not None:
                ax.fill_between(trials, mse[i,:] - SEM[i,:], mse[i,:] + SEM[i,:], 
                                color=colors[i], alpha=0.5)
            
        if inset_labels is not None:
            
            for i in range(5):
                
                if i==0:
                    ax1 = fig.add_axes([0.3,0.65,0.15,0.15])
                    stimuli = np.random.normal(mean, sd, size=1000)
                elif i==1:
                    ax1 = fig.add_axes([0.5,0.65,0.15,0.15])
                    stimuli = random_uniform_from_moments(mean, sd, 1000)
                elif i==2:
                    ax1 = fig.add_axes([0.7,0.65,0.15,0.15])
                    stimuli = random_lognormal_from_moments(mean, sd, 1000)
                elif i==3:
                    ax1 = fig.add_axes([0.3,0.35,0.15,0.15])
                    stimuli = random_gamma_from_moments(mean, sd, 1000)
                elif i==4:
                    ax1 = fig.add_axes([0.5,0.35,0.15,0.15])
                    stimuli = random_binary_from_moments(mean, sd, 1000)
                    
                ax1.hist(stimuli, color=colors[i], density=True)
                ax1.set_xlim([0,15])
                ax1.set_yticks([])
                ax1.set_xticks([])
                ax1.set_title(inset_labels[i], fontsize=10)
                sns.despine(ax=ax1)

    elif mse.ndim==3:
        
        num_rows = np.size(mse,0)
        num_cols = np.size(mse,1)

        alpha = np.arange(1,num_cols+1)/num_cols
        colors = sns.color_palette("viridis_r", n_colors=num_rows)
        
        for i in range(num_rows):
            for j in range(num_cols):
                ax.plot(trials, mse[i,j,:], color=colors[i], alpha=alpha[j])
    
    ax.set_xlabel('Time (#stimuli)')
    ax.set_xlim([0,max(trials)])
    ax.set_ylabel('MSE')
    if title is not None:
        ax.set_title(title)
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax)

    
def plot_mse_heatmap(n_trials, trial_duration, mean_tested, variance_tested, mse, vmax = 5, 
                     title=None, flg=0):
    
    plt.figure()
    MSE_steady_state = np.mean(mse[:,:,(n_trials-500) * trial_duration:],2)
    #MSE_max = np.max(mse,2)
    if flg==0:
        MSE = MSE_steady_state / mean_tested[:,None]**2
    else:
        MSE = MSE_steady_state/variance_tested[None,:]**2
    MSE *= 100
    
    data = pd.DataFrame(MSE.T, index=variance_tested, columns=mean_tested)
    ax1 = sns.heatmap(data, vmin=0, vmax=vmax, cmap='rocket_r', xticklabels=3, yticklabels=2,
                      cbar_kws={'label': r'MSE$_\mathrm{ss}$ (normalised) [%]', 'ticks': [0, vmax]}) # , 'pad':0.01
    ax1.set_xlabel('mean of stimuli')
    ax1.set_ylabel('variance of stimuli')
    if title is not None:
        ax1.set_title(title)
    
    ax1.invert_yaxis()
    
    # axes = fig.add_axes([0.15,0.15,0.15,0.1]) 
    # axes.patch.set_alpha(0.01)
    # axes.plot(mse_variance[0,0,:], 'k')
    # axes.set_ylim([0,50])
    # axes.set_xticks([])
    # axes.set_yticks([])
    # sns.despine(ax=axes)
    

def plot_prediction(n_stimuli, stimuli, stimulus_duration, prediction, 
                    perturbation_time = None, figsize=(6,4.5), ylim_mse=None):
    
    f, ((ax1), (ax2)) = plt.subplots(2, 1, sharex=True, figsize=figsize,
                                     gridspec_kw={'height_ratios': [5, 1]})
    plt.locator_params(nbins=3)
    
    trials = np.arange(len(stimuli))/stimulus_duration
    running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
    
    ax1.axvspan(perturbation_time, trials[-1], alpha=0.1, facecolor='#1E000E', edgecolor=None)
    ax1.plot(trials, stimuli, color='#D76A03', label='stimulus', alpha=0.3)
    ax1.plot(trials, running_average, color='#D76A03', label='running average')
    ax1.plot(trials, prediction, color='#19535F', label='predicted mean')
    ax1.set_xlim([0,max(trials)])
    ax1.set_ylabel('Activity (1/s)')
    ax1.set_title('Mean of sensory inputs')
    ax1.legend(loc=0, ncol=3, frameon=False, handlelength=1)
    sns.despine(ax=ax1)
    
    ax2.axvspan(perturbation_time, trials[-1], alpha=0.1, facecolor='#1E000E', edgecolor=None)
    ax2.plot(trials, (running_average - prediction)**2, color='#AF1B3F')
    if ylim_mse is not None:
        ax2.set_ylim(ylim_mse)
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('Time (#stimuli)')
    sns.despine(ax=ax2)
    
    
def plot_variance(n_stimuli, stimuli, stimulus_duration, variance_per_stimulus, 
                  perturbation_time = None, figsize=(6,4.5)):
    
    f, ((ax1), (ax2)) = plt.subplots(2, 1, sharex=True, figsize=figsize,
                                     gridspec_kw={'height_ratios': [5, 1]})
    plt.locator_params(nbins=3)
    
    mean_running = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
    momentary_variance = (stimuli - mean_running)**2
    variance_running = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1)
    trials = np.arange(len(stimuli))/stimulus_duration
    
    ax1.axvspan(perturbation_time, trials[-1], alpha=0.1, facecolor='#1E000E', edgecolor=None)
    ax1.plot(trials, momentary_variance, color='#D76A03', label='stimuli variance', alpha=0.3)
    ax1.plot(trials, variance_running, color='#D76A03', label='running average')
    ax1.plot(trials, variance_per_stimulus, color='#19535F', label='predicted variance')
    ax1.set_xlim([0,max(trials)])
    ax1.set_ylabel('Activity (1/s)')
    ax1.set_title('Variance of sensory inputs')
    ax1.legend(loc=0, ncol=2, frameon=False, handlelength=1)
    sns.despine(ax=ax1)
    
    ax2.axvspan(perturbation_time, trials[-1], alpha=0.1, facecolor='#1E000E', edgecolor=None)
    ax2.plot(trials, (variance_running - variance_per_stimulus)**2, color='#AF1B3F')
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('Time (#stimuli)')
    sns.despine(ax=ax2)
  