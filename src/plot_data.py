#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:18:17 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


def plot_example_contraction_bias(weighted_output, stimuli, n_trials, figsize=(4,3), 
                                  fs=7, lw=1, num_trial_ss=np.int32(50)):
    
    trials_sensory_all = np.mean(np.split(stimuli, n_trials),1)
    trials_estimated_all = np.mean(np.split(weighted_output, n_trials),1)
    
    # to take out transient ...
    trials_sensory = trials_sensory_all[num_trial_ss:]
    trials_estimated = trials_estimated_all[num_trial_ss:]

    f, ax = plt.subplots(1,1, tight_layout=True, figsize=figsize)                                                                              
    ax.plot(trials_sensory, trials_estimated, 'o', alpha = 1, color=color_stimuli_background)
    ax.axline((np.mean(stimuli), np.mean(stimuli)), slope=1, color='k', ls=':')
    
    # points = np.arange(np.ceil(np.min(trials_sensory)), np.floor(np.max(trials_sensory)))
    # for point in points:
    #     bools = (trials_sensory>point-0.5) & (trials_sensory<=point+0.5)
    #     ax.plot(point, np.mean(trials_estimated[bools]), 's', color=color_weighted_output)
    
    ###########
    mean_sensory = np.mean(stimuli)
    
    ### below mean
    bools = (trials_sensory<=mean_sensory) # (trials_sensory>=min_mean) & (trials_sensory<=mean_sensory)
    x = trials_sensory[bools]
        
    p1 =  np.polyfit(x, trials_estimated[bools], 1)
    ax.plot(x, np.polyval(p1, x), '--')
    
    print(p1)
    
    ### above mean
    bools = (trials_sensory>=mean_sensory) # (trials_sensory>=mean_sensory) & (trials_sensory<max_mean)
    x = trials_sensory[bools]
        
    p2 =  np.polyfit(x, trials_estimated[bools], 1)
    ax.plot(x, np.polyval(p2, x), '--')
    print(p2)
    ###########
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel('mean sensory input per trail', fontsize=fs)
    ax.set_ylabel('estimated sensory input per trail', fontsize=fs)
    sns.despine(ax=ax)
    
    

def plot_trial_mean_vs_sd(stimuli, n_trials, figsize=(3,3), fs=7, lw=1):
    
    trials_sensory_mu = np.mean(np.split(stimuli, n_trials),1)
    trials_sensory_sd = np.std(np.split(stimuli, n_trials),1)
    
    f, ax = plt.subplots(1,1, tight_layout=True, figsize=figsize)
    ax.plot(trials_sensory_mu, trials_sensory_sd, 'o', color=color_sensory, alpha=0.2)
    
    p = np.polyfit(trials_sensory_mu, trials_sensory_sd, 1)
    x = np.unique(np.round(trials_sensory_mu,1))
    ax.plot(x, np.polyval(p, x), color='k', ls=':')
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel('trial mean', fontsize=fs)
    ax.set_ylabel('trial std', fontsize=fs)
    sns.despine(ax=ax)


def plot_combination_activation_INs(xp, xs, xv, alpha_before_pert, alpha_after_pert, 
                                    figsize=(4,4), fs=7, lw=1, f_max = 0.99, f_min=1.05, s=40):
    
    f = plt.figure(figsize=figsize)
    
    # compute deviation and define combinations of interest
    deviation = 100 * (alpha_after_pert - alpha_before_pert) / alpha_before_pert
    
    xp_no_change = xp[np.abs(deviation)<=5]
    xs_no_change = xs[np.abs(deviation)<=5]
    xv_no_change = xv[np.abs(deviation)<=5]
    
    xp_max = xp[alpha_after_pert>=f_max*np.nanmax(alpha_after_pert)]
    xs_max = xs[alpha_after_pert>=f_max*np.nanmax(alpha_after_pert)]
    xv_max = xv[alpha_after_pert>=f_max*np.nanmax(alpha_after_pert)]
    
    xp_min = xp[alpha_after_pert<=f_min*np.nanmin(alpha_after_pert)]
    xs_min = xs[alpha_after_pert<=f_min*np.nanmin(alpha_after_pert)]
    xv_min = xv[alpha_after_pert<=f_min*np.nanmin(alpha_after_pert)]
    
    ax = f.add_subplot(1, 1, 1, projection='3d', elev=20, azim=45)
    #ax.scatter(xp.flatten(), xs.flatten(), xv.flatten(), c='#A89FA1', marker='.', zorder=0)
    
    #ax.contour(xp, xs, xv, zdir='z', offset=-0., cmap='gray')
    ax.scatter(xp_no_change, xs_no_change, xv_no_change, marker='o', s=s, color='#447604', alpha=1) #, edgecolor='k'
    ax.scatter(xp_max, xs_max, xv_max, marker='^', s=s, color='#8F250C', alpha=1) #, edgecolor='k'
    ax.scatter(xp_min, xs_min, xv_min, marker='v', s=s, color='#177E89', alpha=1) #, edgecolor='k'
    ax.plot_wireframe(xp, xs, xv, color='#A89FA1', alpha=0.5)
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs, pad=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.zaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel('fraction of PV', fontsize=fs, labelpad=0)
    ax.set_ylabel('fraction of SOM', fontsize=fs, labelpad=0)
    ax.set_zlabel('fraction of VIP', fontsize=fs, rotation=-90, labelpad=0)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])
    sns.despine(ax=ax)
    
    
def plot_points_of_interest_neuromod(columns, std_means, n_std_all, figsize=(8,3), fs=7,
                                     f_max = 0.99, f_min=1.05, s=40, show_inter=False):
    
    ### layout
    n_col = len(columns)
    n_row = len(std_means)
    
    fig = plt.figure(figsize=figsize, tight_layout=True)
    G = gridspec.GridSpec(1, n_col, figure=fig, wspace=0.3)
    
    ### run over 3 example statisitcs
    for j, column in enumerate(columns):
        
        ax = fig.add_subplot(G[0,j], projection='3d', elev=30, azim=45)
        
        xp_no_change, xs_no_change, xv_no_change = np.array([]), np.array([]), np.array([])
        xp_max, xs_max, xv_max = np.array([]), np.array([]), np.array([])
        xp_min, xs_min, xv_min = np.array([]), np.array([]), np.array([])
        
        for i, std_mean in enumerate(std_means):
    
            n_std = n_std_all[i]

            ### filename for data
            identifier = '_column_' + str(column) + '_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
            file_for_data = '../results/data/neuromod/data_weighting_neuromod' + identifier + '.pickle'
            
            ### load data
            with open(file_for_data,'rb') as f:
                [xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f)
            
            ### compute deviation and define combinations of interest
            deviation = 100 * (alpha_after_pert - alpha_before_pert) / alpha_before_pert
            
            xp_no_change = np.concatenate((xp_no_change, xp[np.abs(deviation)<=5]))
            xs_no_change = np.concatenate((xs_no_change,xs[np.abs(deviation)<=5]))
            xv_no_change = np.concatenate((xv_no_change, xv[np.abs(deviation)<=5]))
            
            xp_max = np.concatenate((xp_max, xp[alpha_after_pert>=f_max*np.nanmax(alpha_after_pert)]))
            xs_max = np.concatenate((xs_max, xs[alpha_after_pert>=f_max*np.nanmax(alpha_after_pert)]))
            xv_max = np.concatenate((xv_max, xv[alpha_after_pert>=f_max*np.nanmax(alpha_after_pert)]))
            
            xp_min = np.concatenate((xp_min, xp[alpha_after_pert<=f_min*np.nanmin(alpha_after_pert)]))
            xs_min = np.concatenate((xs_min, xs[alpha_after_pert<=f_min*np.nanmin(alpha_after_pert)]))
            xv_min = np.concatenate((xv_min, xv[alpha_after_pert<=f_min*np.nanmin(alpha_after_pert)]))
            
        if show_inter:
            values, counts = np.unique(np.stack((xp_no_change,xs_no_change,xv_no_change),axis=0), axis=1, return_counts=True)
            xp_no_change, xs_no_change, xv_no_change = values[:, np.where(counts==n_row)[0]]
            
            values, counts = np.unique(np.stack((xp_max,xs_max,xv_max),axis=0), axis=1, return_counts=True)
            xp_max, xs_max, xv_max = values[:, np.where(counts==n_row)[0]]
            
            values, counts = np.unique(np.stack((xp_min,xs_min,xv_min),axis=0), axis=1, return_counts=True)
            xp_min, xs_min, xv_min = values[:, np.where(counts==n_row)[0]]
             
        ax.scatter(xp_no_change, xs_no_change, xv_no_change, marker='o', s=s, color='#447604', alpha=1) #, edgecolor='k'
        ax.scatter(xp_max, xs_max, xv_max, marker='^', s=s, color='#8F250C', alpha=1) #, edgecolor='k'
        ax.scatter(xp_min, xs_min, xv_min, marker='v', s=s, color='#177E89', alpha=1) #, edgecolor='k'
            
        ax.plot_wireframe(xp, xs, xv, color='#A89FA1', alpha=0.5)
        
        ax.tick_params(size=2.0) 
        ax.tick_params(axis='both', labelsize=fs, pad=0)
        ax.set_xlim([0,1]), ax.set_ylim([0,1]), ax.set_zlim([0,1])
        
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.zaxis.set_major_locator(plt.MaxNLocator(3))
        
        if j==0:
            ax.set_xlabel('fraction of PV', fontsize=fs, labelpad=0)
            ax.set_ylabel('fraction of SOM', fontsize=fs, labelpad=0)
            ax.set_zlabel('fraction of VIP', fontsize=fs, rotation=-90, labelpad=0)
        else:
            ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])

        sns.despine(ax=ax)
            

        
def plot_heatmap_neuromod(xp, xs, alpha_before_pert, alpha_after_pert, figsize=(3,2.5), fs=7, 
                          axs=None, cbar_ax=None, ticks=False, plot_labels=True):
    
    if axs is None:
        f, ax = plt.subplots(1,1, tight_layout=True, figsize=figsize)
    else:
        ax = axs
    
    # compute deviation and define index & columns
    deviation = 100 * (alpha_after_pert - alpha_before_pert) / alpha_before_pert
    index = np.round(np.unique(xp),2)
    columns = np.round(np.unique(xs),2)
    n_row = len(index)
    n_col = len(columns)
    
    # create annotation
    Text = np.chararray((n_row,n_col),unicode=True,itemsize=15)
    Text[:] = ' '
    #Text[np.abs(deviation)<=5] = r'$\bullet$' 
    Text[deviation>5] = '+'
    Text[deviation<-5] = '-'
    Anno = pd.DataFrame(Text, columns=columns, index=index)
    
    # create heatmap
    data = pd.DataFrame(alpha_after_pert, index=index, columns=columns)
    ax = sns.heatmap(data, cmap=cmap_sensory_prediction, vmin=0, vmax=1, 
                     xticklabels=5, yticklabels=5, cbar=cbar_ax,
                     annot=Anno, fmt = '', annot_kws={"fontsize": fs})  
    
    ax.invert_yaxis()
    ax.set_facecolor('#DBDBDB')
    
    if cbar_ax is not None:
        cbar_ax.set_ylabel('Change in sensory weight (%)', fontsize=10, labelpad = 0)
        cbar_ax.tick_params(size=2.0,pad=2.0)
        cbar_ax.locator_params(nbins=3)
    
    if ticks==False:
        ax.tick_params(left=False, bottom=False)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.tick_params(left=True, bottom=True)
        ax.set_xticks([0.5, n_col-0.5])
        ax.set_yticks([0.5, n_col-0.5])
        ax.set_xticklabels([0,1])
        ax.set_yticklabels([0,1])
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    
    if plot_labels:
        ax.set_xlabel('Fraction of PV \nneurons activated', fontsize=fs, labelpad=10)
        ax.set_ylabel('Fraction of SOM \nneurons activated', fontsize=fs, labelpad=10)


def plot_neuromod_per_net(columns, std_means, n_std_all, figsize=(8,6), fs=7):
    
    ### layout
    n_col = len(columns)
    n_row = len(std_means)
    
    fig = plt.figure(figsize=figsize, tight_layout=True)
    cbar_ax = fig.add_axes([.92, .4, .01, .2])
    G = gridspec.GridSpec(1, n_col, figure=fig, wspace=0.3)

    for j, column in enumerate(columns):
        
        G_col = gridspec.GridSpecFromSubplotSpec(n_row, 1, subplot_spec=G[:,j], hspace=0.15)
        
        for i, std_mean in enumerate(std_means):
    
            n_std = n_std_all[i]
            axes = fig.add_subplot(G_col[i,0])
            
            ### filename for data
            identifier = '_column_' + str(column) + '_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
            file_for_data = '../results/data/neuromod/data_weighting_neuromod' + identifier + '.pickle'
            
            ### load data
            with open(file_for_data,'rb') as f:
                [xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f)
            
            if (j==0) and (i==n_row-1):
                plot_heatmap_neuromod(xp, xs, alpha_before_pert, alpha_after_pert, axs=axes, 
                                      ticks=True) # 
            else:
                plot_heatmap_neuromod(xp, xs, alpha_before_pert, alpha_after_pert, axs=axes, 
                                      plot_labels=False)

    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_sensory_prediction)
    
    cbar_ax.set_ylabel('Sensory weight (after)', fontsize=fs, labelpad = 10)
    cbar_ax.tick_params(size=2.0,pad=2.0)
    cbar_ax.tick_params(axis='both', labelsize=fs)
    cbar_ax.locator_params(nbins=3)     


def plot_schema_inputs_tested(figsize=(2,2), fs=7, lw=0.5):
    
    inputs = np.nan*np.ones((5,5))
    di = np.diag_indices(5)
    inputs[di] = np.linspace(0,1,5)
    
    colors = sns.color_palette('Blues_d', n_colors=11)
    f, ax = plt.subplots(1,1, tight_layout=True, figsize=figsize)
    
    data = pd.DataFrame(inputs)
    ax = sns.heatmap(data, vmin=0, vmax=1, cmap=colors, cbar=False, linewidths=lw, linecolor='k')
    
    ax.set_xlabel('variability across trial', fontsize=fs)
    ax.set_ylabel('variability within trial', fontsize=fs)
    ax.set_xticks([]), ax.set_yticks([])
    
    

def plot_impact_para(weight_ctrl, weight_act, para_range_tested = [], fs=12, 
                         ms = 5, lw=1, figsize=(3,3), color='r', cmap='rocket_r',
                         colorbar_title = None, colorbar_tick_labels = None, loc_position=2):
    
        f, ax = plt.subplots(1,1, tight_layout=True, figsize=figsize)
        colors_inputs =  sns.color_palette('Blues_d', n_colors=len(weight_act))
        
        if len(para_range_tested)==0:
            ax.plot(weight_ctrl, weight_act, '-', ms=ms, lw=lw, color=colors_inputs[0],zorder=0)
            ax.scatter(weight_ctrl, weight_act, s=ms**2, lw=lw, color=colors_inputs)
        else:
            colors_para = sns.color_palette(cmap, n_colors=len(para_range_tested))
            
            for i in range(len(para_range_tested)):
                ax.plot(weight_ctrl, weight_act[:,i], '-s', ms=ms, lw=lw, color=colors_para[i], 
                         markeredgewidth=0.4, markeredgecolor='k')
            
            for j in range(np.size(weight_act,0)):
               #ax.axvline(weight_ctrl[j], color=colors_inputs[j], ymin=0, ymax=0.1, zorder=0)
               ax.plot(weight_ctrl[j], 0, 'v', color=colors_inputs[j], )
                
            axins1 = inset_axes(ax, width="30%", height="5%", loc=loc_position)
    
            cmap = ListedColormap(colors_para)
            cb = mpl.colorbar.ColorbarBase(axins1, cmap=cmap, orientation='horizontal', ticks=[0.1,0.9])
            cb.outline.set_visible(False)
            cb.ax.set_title(colorbar_title, fontsize=fs, pad = 0)
            cb.ax.set_xticklabels([colorbar_tick_labels[0], colorbar_tick_labels[1]], fontsize=fs)
            axins1.xaxis.set_ticks_position("bottom")
            axins1.tick_params(axis='both', labelsize=fs)
            axins1.tick_params(size=2.0,pad=2.0)
          
        ax.axline((0.5,0.5), slope=1, ls=':', color='k')
        ymin, ymax = ax.get_ylim()
        ax.axhspan(ymin,0.5, color=color_m_neuron, alpha=0.07, zorder=0)
        ax.axhspan(0.5, ymax, color=color_sensory, alpha=0.07, zorder=0)
        ax.set_ylim([ymin, ymax])
        
        ax.tick_params(size=2.0) 
        ax.tick_params(axis='both', labelsize=fs)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_xlabel('sensory weight (default network)', fontsize=fs)
        ax.set_ylabel('sensory weight (network altered)', fontsize=fs)
    
        sns.despine(ax=ax)
        
        

def plot_weight_over_trial(weight_ctrl, weight_mod, n_trials, id_stims = [0,4], fs=7, leg_text=None):
    
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(4,4))
    colors = sns.color_palette('Blues_d', n_colors=np.size(weight_ctrl, 1))
    
    for id_stim in id_stims:
        
        ### control
        alpha_split = np.array_split(weight_ctrl[:,id_stim], n_trials)
        alpha_avg_trial = np.mean(alpha_split,0)
        alpha_std_trial = np.std(alpha_split,0)
        sem = alpha_std_trial/np.sqrt(n_trials)
        trial_fraction = np.linspace(0,1,len(alpha_avg_trial))
        
        ax.plot(trial_fraction, alpha_avg_trial, color=colors[id_stim])
        ax.fill_between(trial_fraction, alpha_avg_trial-sem, alpha_avg_trial+sem, alpha=0.3, color=colors[id_stim])
    
        ### modulated
        alpha_split = np.array_split(weight_mod[:,id_stim], n_trials)
        alpha_avg_trial = np.mean(alpha_split,0)
        alpha_std_trial = np.std(alpha_split,0)
        sem = alpha_std_trial/np.sqrt(n_trials)
        trial_fraction = np.linspace(0,1,len(alpha_avg_trial))
        
        ax.plot(trial_fraction, alpha_avg_trial, color=colors[id_stim], ls='--')
        ax.fill_between(trial_fraction, alpha_avg_trial-sem, alpha_avg_trial+sem, alpha=0.3, color=colors[id_stim])
    
    ymin, ymax = ax.get_ylim()
    ax.axhspan(ymin,0.5, color=color_m_neuron, alpha=0.07, zorder=0)
    ax.axhspan(0.5, ymax, color=color_sensory, alpha=0.07, zorder=0)
    ax.set_ylim([ymin, ymax])
    
    #ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    
    if leg_text is not None:
        ax.plot(np.nan, np.nan, ls='-', color='k', label=leg_text[0])
        ax.plot(np.nan, np.nan, ls='--', color='k', label=leg_text[1])
        ax.legend(loc=0, frameon=False, handlelength=1, fontsize=fs)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', labelsize=fs)
    ax.set_ylabel('sensory weight', fontsize=fs)
    ax.set_xlabel('time/trial duration', fontsize=fs)
    sns.despine(ax=ax)


def plot_transitions_examples(n_trials, trial_duration, stimuli, alpha, beta, weighted_output, 
                              time_plot = 0, ylim=None, xlim=None, plot_ylable=True, lw=1, 
                              figsize=(3.5,5), plot_only_weights=False, fs=12, transition_point=60):
    
    f1, ax2 = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    time = np.arange(len(stimuli))/trial_duration
    
    if not plot_only_weights:
        
        f1, ax1 = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        
        for i in range(n_trials):
            ax1.axvspan(2*i, (2*i+1), color='#F5F4F5')

        ax1.plot(time[time > time_plot * time[-1]], stimuli[time > time_plot * time[-1]], 
             color=color_stimuli_background, lw=lw, marker='|', ls="None")  
        ax1.plot(time[time > time_plot * time[-1]], weighted_output[time > time_plot * time[-1]], color=color_weighted_output, label='weighted output')
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
                                figsize=(6,4.5), fs=12):
    
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
  