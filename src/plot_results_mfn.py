#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:20:53 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.legend import Legend

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


def plot_alpha_over_trial(fraction_course, n_trials):
    
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
    
    ax.set_ylabel('fraction')
    ax.set_xlabel('time/trial duration')
    sns.despine(ax=ax)
    
    

def plot_transition_course(file_data4plot):
    
    ### load data
    with open(file_data4plot,'rb') as f:
        [n_trials, trial_duration, stimuli, states, window_size, fraction] = pickle.load(f) 
    
    ### universal setting
    marker = ['^', 's', 'o', 'D']
    
    for j in range(4):
    
        ### figure
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(5,3))
        
        for i in range(4):
            if sum(np.isnan(fraction[i,j,:]))==0:
                ax.plot(window_size, fraction[i,j,:], linestyle='--', color=[0.5,0.5,0.5])
                ax.scatter(window_size, fraction[i,j,:], c=fraction[i,j,:], cmap=cmap_sensory_prediction, 
                            marker=marker[i], vmin=0, vmax=1, edgecolors='k', linewidth=0.5, zorder=20)
            
            ax.scatter(window_size[0], fraction[i,j,0], c=fraction[i,j,0], marker=marker[j], s=100, 
                       cmap=cmap_sensory_prediction, vmin=0, vmax=1, edgecolors='k', linewidth=0.5, zorder=30)
        
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        
        ax.set_ylabel('fraction sensory input')
        ax.set_xlabel('window size (#trials after transition)')
        ax.set_ylim([0,1.05])
        
        sns.despine(ax=ax)


def plot_deviation_vs_PE_II(moment_flg, input_flgs, marker, labels, perturbation_direction, 
                            plot_deviation_gradual = False, plot_inds = [0,1,4,5,6,7], xlim=None, ylim=None,
                            figsize=(5.5,4), fontsize=12, markersize=5, linewidth=1, legend_II_flg=True):
    
    ### define figure size
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=figsize)
    
    ### inityialise 
    leg1, leg2 = [], []
    
    for i, input_flg in enumerate(input_flgs):
        
        file_load = '../results/data/perturbations/data_pe_vs_neuron_stim_' + input_flg + '.pickle'
        file_load2 = '../results/data/perturbations/data_perturbations_' + input_flg + '.pickle'
        
        with open(file_load,'rb') as f:
            [_, slopes_nPE, slopes_pPE] = pickle.load(f)
            
        with open(file_load2,'rb') as f:
            [_, _, _, dev_prediction_steady, dev_variance_steady] = pickle.load(f)
            
        
        ### pick slopes that are relevant according to plot_ids
        for _ in range(2):
            slopes_nPE.insert(2,0)
            slopes_pPE.insert(2,0)
            
        slopes_nPE = np.asarray(slopes_nPE)[plot_inds]
        slopes_pPE = np.asarray(slopes_pPE)[plot_inds]
        
        ### define whether you look at mean or variance
        if moment_flg==0:
            data = dev_prediction_steady
        else:
            data = dev_variance_steady
            
        ### decide on perturbation data to be used in plot
        if perturbation_direction == -1: # inhibitory perturbation
            data_pert = data[0, plot_inds]
            title = 'Inhibitory perturbation'
        else: # excitatory perturbation
            data_pert = data[1, plot_inds]
            title = 'Excitatory perturbation'

        
        ### plot for all networks
        if plot_deviation_gradual:
            sc = ax.scatter(slopes_pPE, slopes_nPE, marker=marker[i], lw=0, 
                             c=data_pert, cmap='vlag', zorder=10, s=markersize**2) 
            
            if i==0:
                plt.colorbar(sc, shrink=0.5, label='deviation (%)')
                
            ax.set_title(title, pad=20, loc='right', fontsize=fontsize)
            
        else:
            color_inds = np.sign(data_pert) / perturbation_direction
            ax.scatter(slopes_pPE, slopes_nPE, marker=marker[i], lw=0, 
                              c=color_inds, cmap='vlag', zorder=10, s=markersize**2)
            
        ### add text to markers
        m = -1
        x = slopes_pPE
        y = slopes_nPE
        names = [r'E$_\mathrm{n}$',r'E$_\mathrm{p}$','0','0','P','P','S','V']
        names = np.asarray(names)[plot_inds]
        
        for k,l in zip(x,y):
            m += 1
            ax.annotate(names[m],  xy=(k, l), color='white',
                        fontsize="x-small", weight='normal',
                        horizontalalignment='center',
                        verticalalignment='center', zorder=20)

        
        ### legend - part I
        leg1 += ax.plot(np.nan, np.nan, 'k', marker=marker[i], label=labels[i], linestyle='None',
                        markersize=markersize-3)
        
        
    if moment_flg==0:
        ax.axline((0, 0), slope=1, color=[0.5,0.5,0.5], ls=':', alpha = 0.5)
    else:
        ax.axline((0, 0), slope=-1, color=[0.5,0.5,0.5], ls=':', alpha = 0.5)
        
        
    ### show legend 
    ax.legend(leg1, labels, loc=0, frameon=False, fontsize=fontsize) # loc=3
     
    ### legend - part  II
    if legend_II_flg:
        if plot_deviation_gradual==False:
            cmap =  plt.get_cmap('vlag') 
            leg2 += ax.plot(np.nan, np.nan, '-', c=cmap(0), label='deviation opposite to perturbation direction')
            leg2 += ax.plot(np.nan, np.nan, '-', c=cmap(0.99), label='deviation along perturbation direction')
            
            leg = Legend(ax, leg2, ['opposite to pert. dir.', 'along perturbation dir.'],
                     loc=1, title='Deviation', handlelength=1, frameon=True, fontsize=fontsize)
            ax.add_artist(leg)
    
    ### improve plot appearance   
    ax.set_ylabel('gain nPE', loc='top', rotation=0, labelpad=-10, fontsize=fontsize)
    ax.set_xlabel('gain pPE', loc='right', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))

    sns.despine(ax=ax)


def plot_deviation_vs_PE(moment_flg, input_flgs, marker, labels, perturbation_direction, 
                         plot_deviation_gradual = False):
    
    ### define figure size
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(5,4))
    
    ### inityialise 
    leg1, leg2 = [], []
    
    for i, input_flg in enumerate(input_flgs):
        
        file_load = '../results/data/perturbations/data_neuron_drive_' + input_flg + '.pickle'
        file_load2 = '../results/data/perturbations/data_perturbations_' + input_flg + '.pickle'
        
        with open(file_load,'rb') as f:
            [_, _, slopes_feedforward, slopes_feedback] = pickle.load(f)
            
        with open(file_load2,'rb') as f:
            [_, _, _, dev_prediction_steady, dev_variance_steady] = pickle.load(f)
            
        
        ### define whether you look at mean or variance
        if moment_flg==0:
            data = dev_prediction_steady
        else:
            data = dev_variance_steady
            
        ### decide on perturbation data to be used in plot
        if perturbation_direction == -1: # inhibitory perturbation
            data_pert = data[0,[0,1,4,5,6,7]]
            title = 'Inhibitory perturbation'
        else: # excitatory perturbation
            data_pert = data[1,[0,1,4,5,6,7]]
            title = 'Excitatory perturbation'

        ### plot for all networks
        if plot_deviation_gradual:
            sc = ax.scatter(slopes_feedforward, slopes_feedback, marker=marker[i], lw=0, 
                             c=data_pert, cmap='vlag', zorder=10, s=100) 
            
            if i==0:
                plt.colorbar(sc, shrink=0.5, label='deviation (%)')
                
            ax.set_title(title, pad=20, loc='right')
            
        else:
            color_inds = np.sign(data_pert) / perturbation_direction
            ax.scatter(slopes_feedforward, slopes_feedback, marker=marker[i], lw=0, 
                             c=color_inds, cmap='vlag', zorder=10, s=100)
            
        ### add text to markers
        m = -1
        x = slopes_feedforward
        y = slopes_feedback
        names = [r'E$_\mathrm{n}$',r'E$_\mathrm{p}$','P','P','S','V']
        
        for k,l in zip(x,y):
            m += 1
            ax.annotate(names[m],  xy=(k, l), color='white',
                        fontsize="x-small", weight='normal',
                        horizontalalignment='center',
                        verticalalignment='center', zorder=20)

        
        ### legend - part I
        leg1 += ax.plot(np.nan, np.nan, 'k', marker=marker[i], label=labels[i], linestyle='None')
        
    ### show legend 
    ax.legend(leg1, labels, loc=3, frameon=False)
     
    ### legend - part  II
    if plot_deviation_gradual==False:
        cmap =  plt.get_cmap('vlag') 
        leg2 += ax.plot(np.nan, np.nan, '-', c=cmap(0), label='deviation opposite to perturbation direction')
        leg2 += ax.plot(np.nan, np.nan, '-', c=cmap(0.99), label='deviation along perturbation direction')
        
        leg = Legend(ax, leg2, ['opposite to pert. dir.', 'along perturbation dir.'],
                 loc=1, frameon=False, title='Deviation', handlelength=1)
        ax.add_artist(leg)
    
    ### improve plot appearance   
    ax.set_ylabel('activity / nPE', loc='top', rotation=0, labelpad=-10)
    ax.set_xlabel('activity / pPE', loc='right')
    
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))

    sns.despine(ax=ax)


def plot_deviation_vs_effect_size(x, y, title, plot_legend=True):
    
    fig, ax = plt.subplots(1, 1, figsize=(4,3), tight_layout=True)
    
    ax.plot(x, y[0, :], '.-', label='nEP', color=Col_Rate_nE)
    ax.plot(x, y[1, :], '.-', label='pPE', color=Col_Rate_pE)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    
    ax.spines['bottom'].set_position('zero')
    ax.set_xlabel('gain', loc='right')
    ax.set_ylabel('deviation (%)')
    ax.set_title(title)
    
    if plot_legend:
        ax.legend(loc=0)
    
    sns.despine(ax=ax)
    
    # xlabel ylabel, title


def heatmap_summary_transitions(data, title=None):
    
    plt.figure(tight_layout=True)
    g = sns.heatmap(data, vmin=0, vmax=1, cmap=cmap_sensory_prediction, 
                    cbar_kws={'label': 'fraction sensory input', 'ticks': [0, 1]})
    g.set_facecolor('#DBDBDB')
    
    g.set_xticks([])
    g.set_yticks([])
    g.xaxis.set_label_position('top') 
    
    g.set_ylabel('transition to ...', labelpad=30)
    g.set_xlabel('transition from ...', labelpad=30)
    
    if title is not None:
        g.set_title(title)
    

def plot_transitions_examples(n_trials, trial_duration, stimuli, alpha, beta, weighted_output, 
                              time_plot = 0, ylim=None, xlim=None, plot_ylable=True, 
                              figsize=(3.5,5), plot_only_weights=False, fs=12, transition_point=60):
    
    if plot_only_weights:
        fig, ax2 = plt.subplots(1, 1, figsize=figsize, sharex=True, tight_layout=True)
    else:  
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, tight_layout=True)
    
    time = np.arange(len(stimuli))/trial_duration
    
    if not plot_only_weights:
        ax1.plot(time[time > time_plot * time[-1]], stimuli[time > time_plot * time[-1]], color='#D76A03', label='stimulus')
        ax1.plot(time[time > time_plot * time[-1]], weighted_output[time > time_plot * time[-1]], color='#5E0035', label='weighted output')
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
    
    ax2.plot(time[time > time_plot * time[-1]], alpha[time > time_plot * time[-1]], color='#D76A03', label='stimulus')
    ax2.plot(time[time > time_plot * time[-1]], beta[time > time_plot * time[-1]], color='#19535F', label='prediction')
    if plot_ylable:
        ax2.set_ylabel('Weights', fontsize=fs)
    else:
        ax2.set_ylabel('Weights', color='white', fontsize=fs)
        ax2.tick_params(axis='y', colors='white')
    ax2.axvline(transition_point, color='k', alpha=0.5, ls='--')
    ax2.set_xlabel('Time (#trials)', fontsize=fs)
    ax2.set_xlim([time_plot * time[-1],time[-1]])
    ax2.set_ylim([0,1.05])
    if xlim is not None:
        ax2.set_xlim(xlim)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.tick_params(axis='both', labelsize=fs)
    sns.despine(ax=ax2)
    

def plot_limit_case_example(n_trials, trial_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, 
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
                     title=None, flg=0, fs=5):
    
    plt.figure()
    MSE_steady_state = np.mean(mse[:,:,(n_trials-500) * trial_duration:],2)
    #MSE_max = np.max(mse,2)
    
    if flg==0:
        MSE = MSE_steady_state / mean_tested[:,None]**2
        cbar_label = r'MSE$_\mathrm{\infty}$ (normalised by mean$^2$) [%]'
        color_cbar = LinearSegmentedColormap.from_list(name='mse_prediction', colors=['#FEFAE0', '#19535F'])
    else:
        MSE = MSE_steady_state/variance_tested[None,:]**2
        cbar_label = r'MSE$_\mathrm{\infty}$ (normalised by variance) [%]'
        color_cbar = LinearSegmentedColormap.from_list(name='mse_variance', colors=['#FEFAE0', '#452144'])
        
    MSE *= 100
    
    data = pd.DataFrame(MSE.T, index=variance_tested, columns=mean_tested)
    ax1 = sns.heatmap(data, vmin=0, vmax=vmax, cmap=color_cbar, xticklabels=3, yticklabels=2,
                      cbar_kws={'label': cbar_label, 'ticks': [0, vmax]}) # , 'pad':0.01
    ax1.set_xlabel('mean of stimuli', fontsize=fs)
    ax1.set_ylabel('variance of stimuli', fontsize=fs)
    if title is not None:
        ax1.set_title(title, fontsize=fs, pad=20)
    
    ax1.invert_yaxis()
    
    sns.despine(ax=ax1,bottom=False, top=False, right=False, left=False)
    


def plot_diff_heatmap(n_trials, trial_duration, mean_tested, variance_tested, diff, vmax = 5, 
                     title=None, flg=0, fs=5):
    
    plt.figure()
    diff_steady_state = np.abs(np.mean(diff[:,:,(n_trials-500) * trial_duration:],2))
    
    if flg==0:
        cbar_label = 'dP/P (%)'
        color_cbar = LinearSegmentedColormap.from_list(name='mse_prediction', colors=['#FEFAE0', '#19535F'])
    else:
        cbar_label = 'dV/V (%)'
        color_cbar = LinearSegmentedColormap.from_list(name='mse_variance', colors=['#FEFAE0', '#452144'])
        
    diff_steady_state *= 100
    
    data = pd.DataFrame(diff_steady_state.T, index=variance_tested, columns=mean_tested)
    ax1 = sns.heatmap(data, vmin=0, vmax=vmax, cmap=color_cbar, xticklabels=3, yticklabels=2,
                      cbar_kws={'label': cbar_label, 'ticks': [0, vmax]}) # , 'pad':0.01
    
    ax1.set_xlabel('mean of stimuli', fontsize=fs)
    ax1.set_ylabel('variance of stimuli', fontsize=fs)
    if title is not None:
        ax1.set_title(title, fontsize=fs, pad=20)
    
    ax1.invert_yaxis()
    
    sns.despine(ax=ax1,bottom=False, top=False, right=False, left=False)
    
    

def plot_prediction(n_stimuli, stimuli, stimulus_duration, prediction, 
                    perturbation_time = None, figsize=(6,4.5), ylim_mse=None, lw=1, fs=5,
                    tight_layout=True, legend_flg=True, mse_flg=True):
    
    if mse_flg:
        f, ((ax1), (ax2)) = plt.subplots(2, 1, sharex=True, figsize=figsize,
                                     gridspec_kw={'height_ratios': [5, 1]}, tight_layout=tight_layout)
    else:
        f, ax1 = plt.subplots(1, 1, figsize=figsize, tight_layout=tight_layout)
    
    ax1.locator_params(nbins=3)
    if mse_flg:
        ax2.locator_params(nbins=3)
    
    trials = np.arange(len(stimuli))/stimulus_duration
    running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
    
    ax1.axvspan(perturbation_time, trials[-1], alpha=0.1, facecolor='#1E000E', edgecolor=None)
    ax1.plot(trials, stimuli, color='#D76A03', label='stimulus', alpha=0.3, lw=lw)
    ax1.plot(trials, running_average, color='#D76A03', label='running average', lw=lw)
    ax1.plot(trials, prediction, color='#19535F', label='predicted mean', lw=lw)
    ax1.set_xlim([0,max(trials)])
    ax1.set_ylabel('Activity (1/s)', fontsize=fs)
    ax1.set_title('Mean of sensory inputs', fontsize=fs+1)
    ax1.tick_params(axis='both', labelsize=fs)
    if legend_flg:
        ax1.legend(loc=0, ncol=2, frameon=False, handlelength=1)
    sns.despine(ax=ax1)
    
    if mse_flg:
        ax2.axvspan(perturbation_time, trials[-1], alpha=0.1, facecolor='#1E000E', edgecolor=None)
        ax2.plot(trials, (running_average - prediction)**2, color='#AF1B3F')
        if ylim_mse is not None:
            ax2.set_ylim(ylim_mse)
        ax2.set_ylabel('MSE', fontsize=fs)
        ax2.set_xlabel('Time (#stimuli)', fontsize=fs)
        sns.despine(ax=ax2)
    
    
def plot_variance(n_stimuli, stimuli, stimulus_duration, variance_per_stimulus, 
                  perturbation_time = None, figsize=(6,4.5), lw=1, fs=5,
                  tight_layout=True, legend_flg=True, mse_flg=True):
    
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
    trials = np.arange(len(stimuli))/stimulus_duration
    
    ax1.axvspan(perturbation_time, trials[-1], alpha=0.1, facecolor='#1E000E', edgecolor=None)
    ax1.plot(trials, momentary_variance, color='#D76A03', label='variance of stimuli', alpha=0.3, lw=lw)
    ax1.plot(trials, variance_running, color='#D76A03', label='running average', lw=lw)
    ax1.plot(trials, variance_per_stimulus, color='#452144', label='predicted variance', lw=lw)
    ax1.set_xlim([0,max(trials)])
    ax1.set_ylabel('Activity (1/s)', fontsize=fs)
    ax1.set_title('Variance of sensory inputs', fontsize=fs+1)
    ax1.tick_params(axis='both', labelsize=fs)
    if legend_flg:
        ax1.legend(loc=0, ncol=2, frameon=False, handlelength=1)
    sns.despine(ax=ax1)
    
    if mse_flg:
        ax2.axvspan(perturbation_time, trials[-1], alpha=0.1, facecolor='#1E000E', edgecolor=None)
        ax2.plot(trials, (variance_running - variance_per_stimulus)**2, color='#AF1B3F')
        ax2.set_ylabel('MSE', fontsize=fs)
        ax2.set_xlabel('Time (#stimuli)', fontsize=fs)
        sns.despine(ax=ax2)
  