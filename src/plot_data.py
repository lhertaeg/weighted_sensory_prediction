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

from matplotlib.legend_handler import HandlerTuple
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from src.functions_simulate import random_uniform_from_moments, random_gamma_from_moments
from src.functions_simulate import random_binary_from_moments, random_lognormal_from_moments


# %% set colors

Col_Rate_E = '#9E3039'
Col_Rate_nE = '#D98289' #'#CF636C' #'#955F89' #A06A94
Col_Rate_nD = '#BF9BB8' 
Col_Rate_pE = '#9E3039' # '#CB9173' 
Col_Rate_pD = '#DEB8A6' 
Col_Rate_PVv = '#508FCE'  #508FCE
Col_Rate_PVm = '#2B6299' 
Col_Rate_SOM = '#1F4770' #285C8F, '#79AFB9' 
Col_Rate_VIP = '#7CAE7A' #163350, '#39656D'  #85B79D

color_sensory = '#D76A03'
color_prediction = '#39656D' #'#19535F'

color_illustration_background = '#EBEBEB'

####
color_stimuli_background = '#FEDFC2'
color_running_average_stimuli = '#D76A03'
color_m_neuron = '#39656D' #'#19535F'
color_m_neuron_light = '#83c5be' #'#79AFB9'
color_m_neuron_dark = '#2A4A50'
color_v_neuron = '#754E06' #'#452144'
color_v_neuron_light = '#b08968'#'#E99B0C'
color_v_neuron_dark = '#3A2703'
color_mse = '#AF1B3F' 
color_weighted_output = '#5E0035'
color_mean_prediction = '#70A9A1' 

cmap_sensory_prediction = LinearSegmentedColormap.from_list(name='cmap_sensory_prediction', 
                                                            colors=[color_m_neuron,'#D6D6D6',
                                                                    color_running_average_stimuli]) # fefee3

# %% plot functions

def normal_dist(x , mean , sd):
    prob_density = np.exp(-0.5*((x-mean)/sd)**2) / (np.sqrt(2 * np.pi)*sd)
    return prob_density


# from https://gist.github.com/ihincks
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_gain_factors(gains_nPE, gains_pPE, figsize=(10,1), fs = 6):
    
    fig = plt.figure(figsize=figsize)
    
    G = gridspec.GridSpec(1, 2, figure=fig, hspace=1, width_ratios=[19,1])
    G1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=G[0,0], hspace=0.5)

    ax_B = fig.add_subplot(G1[1,0])
    ax_A = fig.add_subplot(G1[0,0], sharex=ax_B)
    plt.setp(ax_A.get_xticklabels(), visible=False)
    ax_C = fig.add_subplot(G[0,1])
    
    ax_B.tick_params(size=2.0) 
    ax_B.tick_params(axis='both', labelsize=fs)
    ax_A.tick_params(size=2.0) 
    ax_A.tick_params(axis='both', labelsize=fs)
    ax_C.tick_params(size=2.0) 
    ax_C.tick_params(axis='both', labelsize=fs)
    
    max_gain = max(np.nanmax(gains_nPE), np.nanmax(gains_pPE))
    min_gain = min(np.nanmin(gains_nPE), np.nanmin(gains_pPE))
    
    data = pd.DataFrame(gains_nPE)
    g = sns.heatmap(data.T, xticklabels=20, yticklabels=0, vmin=min_gain, vmax=max_gain, ax = ax_A, 
                    cbar_ax = ax_C, cbar_kws={'label': r'log(gain)'})
    g.set_facecolor('#EBEBEB')
    
    data = pd.DataFrame(gains_pPE)
    g = sns.heatmap(data.T, xticklabels=20, yticklabels=0, vmin=min_gain, vmax=max_gain, 
                    ax = ax_B, cbar = False)
    g.set_facecolor('#EBEBEB')
    
    ax_B.set_xlabel('Neurons', fontsize=fs)
    
    ax_C.tick_params(labelsize=fs)
    ax_C.tick_params(size=2.0)
    ax_C.yaxis.label.set_size(fs)
    
    

def plot_nPE_pPE_activity_compare(P, S, nPE, pPE, fs = 6, lw = 1, ms = 1, ax1 = None):
    
    if ax1 is None:
        _, ax1 = plt.subplots(1,1)
        
    # activity
    ax1.plot(S-P, nPE, color=Col_Rate_nE, lw=lw)
    ax1.plot(S-P, pPE, color=Col_Rate_pE, lw=lw)
    
    ax1.tick_params(size=2.0) 
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.locator_params(nbins=3, axis='x')
    ax1.set_yticks([5])
    
    ax1.set_xlabel('Sensory - Prediction', fontsize=fs)
    ax1.set_ylabel('Activity', fontsize=fs, rotation=0)
    ax1.spines['left'].set_position('center')
    ax1.set_ylim(bottom=0)
    ax1.yaxis.set_label_coords(0.5, 1.1)
    sns.despine(ax=ax1)


def plot_standalone_colorbar(parent_axes, fs=6):
    
    ax_cbar = parent_axes.inset_axes((-0.4,0.5,1,0.1))
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, orientation='horizontal', cmap=cmap_sensory_prediction)
    
    cbar.outline.set_linewidth(0.1)
    
    ax_cbar.set_title('Sensory weight', fontsize=fs, pad = 5)
    ax_cbar.tick_params(size=2.0,pad=2.0)
    ax_cbar.tick_params(axis='both', labelsize=fs)
    ax_cbar.locator_params(nbins=3)     
    

def plot_illustration_neuromod_results(ax = None, fs=6):
    
    if ax is None:
        _, ax1 = plt.subplots(1,1)
    else:
        ax1 = ax
        
    ax1.set_xticks([]), ax1.set_yticks([])
    ax1.patch.set_facecolor('#EFF6FB')
    sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)    
    
    ax_3 = ax1.inset_axes((0.5,0,0.5,0.3)) 
    ax_2 = ax1.inset_axes((0.5,0.35,0.5,0.3))
    ax_1 = ax1.inset_axes((0.5,0.7,0.5,0.3))
    
    color_before = '#3D4451'#
    color_after = '#808A9F' 
    
    X = np.array([1,2])
    dx = 0.3
    
    before = np.array([0.8, 0.2])
    after = np.array([0.55, 0.45])
    ax_1.bar(X + 0.00, before, color = color_before, width = dx, label='Before')
    ax_1.bar(X + dx, after, color = color_after, width = dx, label='After')
    ax_1.axhline(0.5, color='k', ls=':')
    ax_1.set_ylim([0,1])
    ax_1.set_xlim([X[0] - 1.5*dx, X[1] + 3*dx])
    ax_1.set_yticks([0.5])
    ax_1.set_xticks([])
    ax_1.tick_params(size=2.0) 
    ax_1.tick_params(axis='both', labelsize=fs)
    sns.despine(ax=ax_1)
    
    ax_1.legend(loc=1, fontsize=fs, frameon=False, handlelength=1, ncol=1, bbox_to_anchor=(1.,1.2))
    ax_1.patch.set_facecolor('#EFF6FB')
    
    before = np.array([0.8, 0.2])
    after = np.array([0.78, 0.22])
    ax_2.bar(X + 0.00, before, color = color_before, width = dx)
    ax_2.bar(X + dx, after, color = color_after, width = dx)
    ax_2.axhline(0.5, color='k', ls=':')
    ax_2.set_ylim([0,1])
    ax_2.set_xlim([X[0] - 1.5*dx, X[1] + 3*dx])
    ax_2.set_yticks([0.5])
    ax_2.set_xticks([])
    ax_2.tick_params(size=2.0) 
    ax_2.tick_params(axis='both', labelsize=fs)
    ax_2.set_ylabel('Sensory weight', fontsize=fs)
    ax_2.patch.set_facecolor('#EFF6FB')
    sns.despine(ax=ax_2)
    
    before = np.array([0.8, 0.2])
    after = np.array([0.4, 0.15])
    ax_3.bar(X + 0.00, before, color = color_before, width = dx)
    ax_3.bar(X + dx, after, color = color_after, width = dx)
    ax_3.axhline(0.5, color='k', ls=':')
    ax_3.set_ylim([0,1])
    ax_3.set_xlim([X[0] - 1.5*dx, X[1] + 3*dx])
    ax_3.set_yticks([0.5])
    ax_3.set_xticks([1 + dx/2, 2 + dx/2])
    ax_3.set_xticklabels(['Sens. \ndriven \n(before)', 'Pred. \ndriven \n(before)'], fontsize=fs)
    ax_3.tick_params(size=2.0) 
    ax_3.tick_params(axis='both', labelsize=fs)
    ax_3.patch.set_facecolor('#EFF6FB')
    sns.despine(ax=ax_3)


def plot_illustration_trial_duration(fs=6, lw=1, ax_1 = None):
    
    if ax_1 is None:
        _, ax1 = plt.subplots(1,1)
    else:
        ax1=ax_1
        
    # colors
    colors = [color_running_average_stimuli, '#6A2E35']
        
    # set up
    ax_1L = ax1.inset_axes((0,0,0.4,1)) 
    ax_1R = ax1.inset_axes((0.6,0,0.4,1))
    ax_1L.patch.set_facecolor(color_illustration_background)
    ax_1R.patch.set_facecolor(color_illustration_background)
    
    ax1.set_yticks([])
    ax1.set_xticks([])
    sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
    
    # plot
    ax_1L.plot([0,1], [1,1], color=colors[0], lw=lw+1)
    ax_1L.plot([0,2], [2,2], color=colors[1], lw=lw+1)
    
    ax_1L.text(0.9, 2.5, 'Duration', color='k', fontsize=fs, ha='center') #Duration
    
    ax_1L.set_ylim([0.5,3.5])
    ax_1L.set_yticks([])
    ax_1L.set_xticks([])
    sns.despine(ax=ax_1L, top=True, right=True, left=True, bottom=True)
    
    slope = np.array([2,1])
    for i in range(2): 
        ax_1R.bar(i, slope[i], color=colors[i], width=0.4)
    
    ax_1R.set_xlabel('m',fontsize=fs)
    ax_1R.set_xlim([-1,2])
    ax_1R.set_yticks([])
    ax_1R.set_xticks([])
    sns.despine(ax=ax_1R, top=True, right=True, left=True)
    

def plot_illustration_bias_results(fs=6, lw=1, ax1 = None, ax2 = None):
    
    if ax1 is None:
        _, ax1 = plt.subplots(1,1)
        
    if ax2 is None:
        _, ax1 = plt.subplots(1,1)

    # colors
    colors = [color_running_average_stimuli, '#6A2E35']
    
    # dependenc on trial variability
    ax_1L = ax1.inset_axes((0,0,0.4,1)) 
    ax_1R = ax1.inset_axes((0.6,0,0.4,1)) 
    ax_1L.patch.set_facecolor(color_illustration_background)
    ax_1R.patch.set_facecolor(color_illustration_background)
    
    a = np.array([1,2])
    b = np.array([5,4])
    slopes = np.array([1,2])
    
    for i in range(2):
        
        x = np.linspace(a[i]-1, b[i]+1, 1000)
        y = np.zeros_like(x)
        y[(x>=a[i]) & (x<=b[i])] = 1/(b[i]-a[i])
        ax_1L.plot(x, y, c=colors[i], lw=lw)
        ax_1R.bar(i, slopes[i], color=colors[i], width=0.4)
        
    ax_1R.set_xlim([-1,2])
    ax1.set_yticks([]), ax_1L.set_yticks([]), ax_1R.set_yticks([])
    ax1.set_xticks([]), ax_1L.set_xticks([]), ax_1R.set_xticks([])
    sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
    sns.despine(ax=ax_1L, top=True, right=True, left=True)
    sns.despine(ax=ax_1R, top=True, right=True, left=True)
    
    # dependence on stimulus variability
    ax_2L = ax2.inset_axes((0,0,0.4,1)) 
    ax_2R = ax2.inset_axes((0.6,0,0.4,1)) 
    ax_2L.patch.set_facecolor(color_illustration_background)
    ax_2R.patch.set_facecolor(color_illustration_background)
    
    stds = np.array([2,1])
    slopes = np.array([2,1])
    
    for i in range(2):
        
        x = np.linspace(-4, 4, 1000)
        z = normal_dist(x, 0, stds[i])
        ax_2L.plot(x, z, c=colors[i], lw=lw)
        ax_2R.bar(i, slopes[i], color=colors[i], width=0.4)
       
    ax_2R.set_xlabel('m', fontsize=fs)
    ax_2R.set_xlim([-1,2])
    ax2.set_yticks([]), ax_2L.set_yticks([]), ax_2R.set_yticks([])
    ax2.set_xticks([]), ax_2L.set_xticks([]), ax_2R.set_xticks([])
    sns.despine(ax=ax2, top=True, right=True, left=True, bottom=True)
    sns.despine(ax=ax_2L, top=True, right=True, left=True)
    sns.despine(ax=ax_2R, top=True, right=True, left=True)
    

def plot_illustration_input_cond(std_stims, mean_trails, std_trails, slopes = None, ms = 1,
                                 fs=6, lw=1, ax_1 = None, ax_2 = None, color = None, labels=False):
    
    if ax_1 is None:
        _, ax1 = plt.subplots(1,1)
        
    if ax_2 is None:
        _, ax1 = plt.subplots(1,1)
        
    if slopes is not None:
        ax1= ax_1.inset_axes((0,0,0.4,1)) 
        ax2= ax_2.inset_axes((0,0,0.4,1)) 
        ax3 = inset_axes(ax_1, width="40%", height="100%", loc=1, bbox_to_anchor=(0.1,0,1,1), bbox_transform=ax_1.transAxes)
        
        ax_1.set_yticks([]), ax_1.set_xticks([])
        ax_2.set_yticks([]), ax_2.set_xticks([])
        sns.despine(ax=ax_1, top=True, right=True, left=True, bottom=True)
        sns.despine(ax=ax_2, top=True, right=True, left=True, bottom=True)
        
        ax1.patch.set_facecolor(color_illustration_background)
        ax2.patch.set_facecolor(color_illustration_background)
        ax3.patch.set_facecolor(color_illustration_background)
        
    else:
        ax1 = ax_1
        ax2 = ax_2

    # colors
    if color is None:
        colors = [color_running_average_stimuli, '#6A2E35']
    else:
        colors = [color, color]

    # trail distribution
    for i in range(len(mean_trails)):
        
        b = np.sqrt(12) * std_trails[i] / 2 + mean_trails[i]
        a = 2 * mean_trails[i] - b
        x = np.linspace(a-2, b+2, 1000)
        y = np.zeros_like(x)
        y[(x>=a) & (x<=b)] = 1/(b-a)
        
        ax1.plot(x, y, c=colors[i], lw=lw)

        # means tested
        mean_stims_tested = np.array([a, (a+b)/2, b])
        for j, mean_tested in enumerate(mean_stims_tested):
            ax1.plot(mean_tested, -0.05, marker='v', color=colors[i], ms=ms)
         
    ylims = ax1.get_ylim()
    if ylims[1]>10:
        ax1.set_ylim([-0.1,0.5])
        
    x_lims = ax1.get_xlim()
    if labels:
        ax1.set_title('Distributions', fontsize=fs)
        ax1.text(-0.1,0.6, 'Trial', fontsize=fs, rotation=90, va='bottom', transform = ax1.transAxes)

    # stimulus distribution
    x = np.linspace(x_lims[0], x_lims[1], 1000)
    
    # colors
    if color is None:
        colors = [color_running_average_stimuli, '#6A2E35']
    else:
        colors = [lighten_color(color, 0.5), lighten_color(color, 0.5)]

    for i in range(len(std_stims)):
        
        b = np.sqrt(12) * std_trails[i] / 2 + mean_trails[i]
        a = 2 * mean_trails[i] - b
        mean_stims_tested = np.array([a, (a+b)/2, b])
        
        for j in range(3):
            
            z = normal_dist(x, mean_stims_tested[j], std_stims[i])
            ax2.plot(x, z, c=colors[i], lw=lw)
    
    if labels:
        ax2.text(-0.1,0, 'Stimulus', fontsize=fs, rotation=90, va='bottom', transform = ax2.transAxes)
    
    # same x range
    xlim_1 = ax1.get_xlim()
    xlim_2 = ax2.get_xlim()
    
    x_min = np.min(np.array([xlim_1[0], xlim_2[0]]))
    x_max = np.min(np.array([xlim_1[1], xlim_2[1]]))
    
    ax1.set_xlim([x_min, x_max])
    ax2.set_xlim([x_min, x_max])
    
    # get rid of ticks etc
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
    sns.despine(ax=ax2, top=True, right=True, left=True, bottom=True)
    
    if slopes is not None:
        ax3.bar(np.array([1]), slopes[0], color=colors[0], width=0.4)
        ax3.bar(np.array([2]), slopes[1], color=colors[1], width=0.4)
        ax3.set_xlabel('m', fontsize=fs)
        
        ax3.set_xlim([0,3])
        ax3.set_ylim([0,ax3.get_ylim()[1]+1])
        
        ax3.set_xticks([])
        ax3.set_yticks([])
        sns.despine(ax=ax3, top=True, right=True, left=True)
    
    

def plot_deviation_spatial(deviation, means_tested, stds_tested, vmin=0, vmax=1, fs=6, show_mean = True, show_xlabel=True, 
                           x_examples = None, y_examples = None, markers_examples = None, ms = 1, ax=None):
    
    if ax is None:
        _, ax = plt.subplots(1,1)
        
    if show_mean:
        color_cbar = LinearSegmentedColormap.from_list(name='mse_mean', colors=['#FEFAE0', color_m_neuron])
    else:
        color_cbar = LinearSegmentedColormap.from_list(name='mse_variance', colors=['#FEFAE0', color_v_neuron])
    

    data = pd.DataFrame(abs(deviation.T)*100, index=np.round(means_tested,1), columns=np.round(stds_tested**2,1))
    sns.heatmap(data, cmap = color_cbar, vmin=vmin, vmax=vmax, xticklabels=2, yticklabels=2,
                cbar_kws={'label': 'Normalised \nerror (%)', 'ticks':[vmin, vmax]}, ax=ax)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.tick_params(size=2.0)
    cbar.ax.yaxis.label.set_size(fs)
    
    ax.invert_yaxis()
    
    if markers_examples is not None:
        
        variances_tested = stds_tested**2
        dx = np.diff(means_tested)[0]/2
        dy = np.diff(variances_tested)[0]/2
        mx, nx = np.polyfit([means_tested.min()-dx, means_tested.max()+dx], ax.get_xlim(), 1)
        my, ny = np.polyfit([variances_tested.min()-dy, variances_tested.max()+dy], ax.get_ylim(), 1)
        
        for i, x in enumerate(x_examples):
            y = y_examples[i]
            ax.plot(mx * x + nx, my * y + ny, marker=markers_examples[i], color='k', ms=ms)
    
    if show_xlabel:
        ax.set_xlabel('Mean', fontsize=fs) 
    ax.set_ylabel('Variance', fontsize=fs) 
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.locator_params(nbins=3, axis='both')
    
    

def plot_examples_spatial_V(num_time_steps, v_neuron_before, v_neuron_after, std_before, std_after, labels, ax=None, fs=6, lw=1):
    
    if ax is None:
        _, ax = plt.subplots(1,1)
    
    colors = ['#FFBB33', '#E9724C']

    ndim = v_neuron_after.ndim
    
    ax.plot(np.arange(num_time_steps), v_neuron_before, c=color_v_neuron, label=labels[0], lw=lw)
    ax.axhline(std_before, 0, 0.5, ls=':', c=color_v_neuron, lw=lw)
    
    if ndim==1:
        ax.plot(np.arange(num_time_steps, 2*num_time_steps), v_neuron_after, c=colors[0], label=labels[1], lw=lw)
        ax.axhline(std_after, 0.5, 1, ls=':', color=colors[0], lw=lw)
    else:
        
        for idim in range(ndim):
            ax.plot(np.arange(num_time_steps, 2*num_time_steps), v_neuron_after[idim,:], c=colors[idim], label=labels[idim+1], lw=lw)
            ax.axhline(std_after[idim], 0.5, 1, ls=':', color=colors[idim], lw=lw)
    
    ax.axvspan(num_time_steps,2*num_time_steps, color='k', alpha=0.03, zorder=0)
    ax.legend(loc=0, frameon=False, handlelength=1, fontsize=fs, borderaxespad=0, borderpad=0.2)
    ax.set_xlim([0,2*num_time_steps])
    ax.set_xticks([0,num_time_steps,2*num_time_steps])
    ax.set_xticklabels([0,1,2])
    
    ax.set_xlabel('Time (fraction of stimuli)', fontsize=fs)
    ax.set_ylabel('Activity (1/s)', fontsize=fs)
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.locator_params(nbins=3, axis='y')
    sns.despine(ax=ax)


def plot_examples_spatial_M(num_time_steps, m_neuron_before, m_neuron_after, mean_before, mean_after, labels, 
                            show_xlabel = True, ax=None, fs=6, lw=1):
    
    if ax is None:
        _, ax = plt.subplots(1,1)
    
    colors = ['#FFBB33', '#E9724C']

    ndim = m_neuron_after.ndim
    
    ax.plot(np.arange(num_time_steps), m_neuron_before, c=color_m_neuron, label=labels[0], lw=lw)
    ax.axhline(mean_before, 0, 0.5, ls=':', c=color_m_neuron, lw=lw)
    
    if ndim==1:
        ax.plot(np.arange(num_time_steps, 2*num_time_steps), m_neuron_after, c=colors[0], label=labels[1], lw=lw)
        ax.axhline(mean_after, 0.5, 1, ls=':', color=colors[0], lw=lw)
    else:
        
        for idim in range(ndim):
            ax.plot(np.arange(num_time_steps, 2*num_time_steps), m_neuron_after[idim,:], c=colors[idim], label=labels[idim+1], lw=lw)
            ax.axhline(mean_after[idim], 0.5, 1, ls=':', color=colors[idim], lw=lw)
    
    ax.axvspan(num_time_steps,2*num_time_steps, color='k', alpha=0.03, zorder=0)
    ax.legend(loc=0, frameon=False, handlelength=1, fontsize=fs, borderaxespad=0, borderpad=0.2)
    ax.set_xlim([0,2*num_time_steps])
    ax.set_xticks([0,num_time_steps,2*num_time_steps])
    ax.set_xticklabels([0,1,2])
    
    if show_xlabel:
        ax.set_xlabel('Time (fraction of stimuli)', fontsize=fs)
    ax.set_ylabel('Activity (1/s)', fontsize=fs)
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.locator_params(nbins=3, axis='y')
    sns.despine(ax=ax)


def plot_deviation_in_population_net(x, num_seeds, M_steady_state, V_steady_state, xlabel, mean=5, var=4, 
                                     ax=None, fs=6, lw=1, ylim=None, plt_ylabel=True):
    
    if ax is None:
        _, ax = plt.subplots(1,1)
    
    M_avg = np.mean((M_steady_state - mean)/mean * 100, 1)
    V_avg = np.mean((V_steady_state - var)/var * 100, 1)
    
    M_sem = np.std((M_steady_state - mean)/mean * 100, 1)/np.sqrt(num_seeds)
    V_sem = np.std((V_steady_state - var)/var * 100, 1)/np.sqrt(num_seeds)
    
    ax.plot(x, M_avg, color=color_m_neuron, lw=lw)
    ax.fill_between(x, M_avg - M_sem/2, M_avg + M_sem/2, color=color_m_neuron, alpha=0.5)
    
    ax.plot(x, V_avg, color=color_v_neuron, lw=lw)
    ax.fill_between(x, V_avg - V_sem/2, V_avg + V_sem/2, color=color_v_neuron, alpha=0.5)
    
    ax.set_xlim([x[0], x[-1]])
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(xlabel, fontsize=fs)
    if plt_ylabel:
        ax.set_ylabel('Normalised \nerror (%)', fontsize=fs)
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.locator_params(nbins=3, axis='both')
    sns.despine(ax=ax)
    


def plot_M_and_V_for_population_example(t, r_mem, r_var, mean=5, var=4, ax=None, fs=6, lw=1):
    
    if ax is None:
        _, ax = plt.subplots(1,1)

    ax.plot(t/t[-1], r_mem, color=color_m_neuron, lw=lw, label='M neuron')
    ax.axhline(mean, color=color_m_neuron, ls='--', lw=lw)
    ax.plot(t/t[-1], r_var, color=color_v_neuron, lw=lw, label='V neuron')
    ax.axhline(var, color=color_v_neuron, ls='--', lw=lw)
    
    ax.legend(loc=0, frameon=False, fontsize=fs, handlelength=1)
    ax.set_ylabel('Activity (1/s)', fontsize=fs)
    ax.set_xlabel('Time (fraction of stimulus duration)', fontsize=fs)

    ax.set_xlim([t[0]/t[-1],t[-1]/t[-1]])
    ax.set_ylim(bottom=0)
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.locator_params(nbins=3, axis='both')
    sns.despine(ax=ax)


def plot_legend_illustrations(ax, fs=6, mew=2):
    
    p1, = ax.plot(np.nan, np.nan, marker=10, color='k', ms=4, ls='None')
    p2, = ax.plot(np.nan, np.nan, marker=11, color='k', ms=4, ls='None')
    p3, = ax.plot(np.nan, np.nan, marker='_', color=color_m_neuron, ms=6, ls='None', markeredgewidth=mew)
    p4, = ax.plot(np.nan, np.nan, marker='_', color=color_v_neuron, ms=6, ls='None', markeredgewidth=mew)
    p5, = ax.plot(np.nan, np.nan, marker='_', color=color_m_neuron_light, ms=6, ls='None', markeredgewidth=mew)
    p6, = ax.plot(np.nan, np.nan, marker='_', color=color_v_neuron_light, ms=6, ls='None', markeredgewidth=mew)
    
    ax.legend([p1, p2, (p3, p4), (p5, p6)], ['pPE neuron targeted', 'nPE neuron targeted', 'in lower PE circuit', 'in higher PE circuit'],
              handler_map={tuple: HandlerTuple(ndivide=None)}, loc=6, fontsize=fs, frameon=False, 
              bbox_to_anchor=(-0.1, 0.5), ncol=1)


def plot_influence_interneurons_baseline_or_gain(plot_baseline=True, plot_annotation=True, ax=None, fs=6, s=10, lw=1):
    
    ### load data
    file_for_data = '../results/data/neuromod/data_PE_props_vs_neuromod_10.pickle'
    with open(file_for_data,'rb') as f:
            [results_base_nPE_10, results_base_pPE_10, results_gain_nPE_10, results_gain_pPE_10] = pickle.load(f) 
            
    file_for_data = '../results/data/neuromod/data_PE_props_vs_neuromod_11.pickle'
    with open(file_for_data,'rb') as f:
            [results_base_nPE_11, results_base_pPE_11, results_gain_nPE_11, results_gain_pPE_11] = pickle.load(f) 
            
    file_for_data = '../results/data/neuromod/data_PE_props_vs_neuromod_01.pickle'
    with open(file_for_data,'rb') as f:
            [results_base_nPE_01, results_base_pPE_01, results_gain_nPE_01, results_gain_pPE_01] = pickle.load(f)
            
    ### plot data
    def create_legend_for_INS(c):
        
        if c==0:
            res = 'PV'
        elif c==1:
            res = 'SOM'
        elif c==2:
            res = 'VIP'
            
        return res
    
    colors = [Col_Rate_PVv, Col_Rate_SOM, Col_Rate_VIP]
    
    if ax is None:
        _, ax = plt.subplots(1,1)
    
    if plot_baseline:
        ax.scatter((results_base_nPE_10[1:] - results_base_nPE_10[0]), (results_base_pPE_10[1:] - results_base_pPE_10[0]),
                        c=colors, marker='o', s=s)
        
        ax.scatter((results_base_nPE_11[1:] - results_base_nPE_11[0]), (results_base_pPE_11[1:] - results_base_pPE_11[0]), 
                   c=colors, marker='s', s=s)
        
        ax.scatter((results_base_nPE_01[1:] - results_base_nPE_01[0]), (results_base_pPE_01[1:] - results_base_pPE_01[0]), 
                    c=colors, marker='d', s=s)
        
    else:
        ax.scatter((results_gain_nPE_10[1:] - results_gain_nPE_10[0]), (results_gain_pPE_10[1:] - results_gain_pPE_10[0]),
                        c=colors, marker='o', s=s)
        
        ax.scatter((results_gain_nPE_11[1:] - results_gain_nPE_11[0]), (results_gain_pPE_11[1:] - results_gain_pPE_11[0]), 
                   c=colors, marker='s', s=s)
        
        ax.scatter((results_gain_nPE_01[1:] - results_gain_nPE_01[0]), (results_gain_pPE_01[1:] - results_gain_pPE_01[0]), 
                    c=colors, marker='d', s=s)

    ax.axhline(0, ls=':', color='k', alpha=0.5, lw=lw, zorder=0)
    ax.axvline(0, ls=':', color='k', alpha=0.5, lw=lw, zorder=0)
    
    xbound = max(np.abs(ax.get_xlim()))
    ybound = max(np.abs(ax.get_ylim()))
    ax.set_xlim([-xbound,xbound])
    ax.set_ylim([-ybound,ybound])
    
    if plot_baseline:
        ax.set_xlabel('Baseline of nPE', fontsize=fs)#, loc='left', labelpad=120)
        ax.set_ylabel('Baseline of pPE', fontsize=fs)#, loc='bottom', labelpad=150)
    else:
        ax.set_xlabel(r'$\Delta$ gain of nPE', fontsize=fs)#, loc='left', labelpad=120)
        ax.set_ylabel(r'$\Delta$ gain of pPE', fontsize=fs)#, loc='bottom', labelpad=150)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    
    if plot_annotation:
        ax.text(0.6,0.7,'increase', color='#525252', fontsize=fs, rotation=0, transform=ax.transAxes)
        ax.text(0.3,0.1,'decrease', color='#525252', fontsize=fs, rotation=0, transform=ax.transAxes)
    
    ax.axline((0,0),slope=-1, color='k', ls='-', alpha=0.3, lw=lw, zorder=0) 
    
    x = np.linspace(-xbound, xbound, 10)
    y = -x
    ax.fill_between(x, y, ax.get_ylim()[1], zorder=0, color='k', alpha=0.05) # (x, V_avg - V_sem/2, V_avg + V_sem/2, color=color_v_neuron, alpha=0.5)
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.locator_params(nbins=3, axis='both')
    sns.despine(ax=ax)


def plot_influence_interneurons_gain_baseline(ax=None, fs=6, s=10, lw=1):
    
    ### load data
    file_for_data = '../results/data/neuromod/data_PE_props_vs_neuromod_10.pickle'
    with open(file_for_data,'rb') as f:
            [results_base_nPE_10, results_base_pPE_10, results_gain_nPE_10, results_gain_pPE_10] = pickle.load(f) 
            
    file_for_data = '../results/data/neuromod/data_PE_props_vs_neuromod_11.pickle'
    with open(file_for_data,'rb') as f:
            [results_base_nPE_11, results_base_pPE_11, results_gain_nPE_11, results_gain_pPE_11] = pickle.load(f) 
            
    file_for_data = '../results/data/neuromod/data_PE_props_vs_neuromod_01.pickle'
    with open(file_for_data,'rb') as f:
            [results_base_nPE_01, results_base_pPE_01, results_gain_nPE_01, results_gain_pPE_01] = pickle.load(f)
            
    ### plot data
    def create_legend_for_INS(c):
        
        if c==0:
            res = 'PV'
        elif c==1:
            res = 'SOM'
        elif c==2:
            res = 'VIP'
            
        return res
    
    colors = [Col_Rate_PVv, Col_Rate_SOM, Col_Rate_VIP]
    
    if ax is None:
        _, ax = plt.subplots(1,1)
    
    ax.scatter((results_base_nPE_10[1:] - results_base_nPE_10[0]) + (results_base_pPE_10[1:] - results_base_pPE_10[0]),
                (results_gain_nPE_10[1:] - results_gain_nPE_10[0]) + (results_gain_pPE_10[1:] - results_gain_pPE_10[0]), 
                c=colors, marker='o', s=s)
    
    ax.scatter((results_base_nPE_11[1:] - results_base_nPE_11[0]) + (results_base_pPE_11[1:] - results_base_pPE_11[0]),
                (results_gain_nPE_11[1:] - results_gain_nPE_11[0]) + (results_gain_pPE_11[1:] - results_gain_pPE_11[0]), 
                c=colors, marker='s', s=s)
    
    ax.scatter((results_base_nPE_01[1:] - results_base_nPE_01[0]) + (results_base_pPE_01[1:] - results_base_pPE_01[0]),
                (results_gain_nPE_01[1:] - results_gain_nPE_01[0]) + (results_gain_pPE_01[1:] - results_gain_pPE_01[0]), 
                c=colors, marker='d', s=s)
    
    xbound = max(np.abs(ax.get_xlim()))
    ybound = max(np.abs(ax.get_ylim()))
    ax.set_xlim([-xbound,xbound])
    ax.set_ylim([-ybound,ybound])
    
    ax.set_xlabel('Baseline of nPE & pPE', fontsize=fs)
    ax.set_ylabel(r'$\Delta$ gain of nPE & pPE', fontsize=fs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    
    ax.text(0.2,0.2,'Variance \nincreases', color='#525252', fontsize=fs)
    ax.text(-0.6,0,'Variance \ndecreases', color='#525252', fontsize=fs)
    
    ax.axline((0,0),slope=-0.5, color='k', ls='-', alpha=0.3, lw=lw, zorder=0) 
    ax.axline((0,0),slope=-1, color='k', ls='-', alpha=0.3, lw=lw, zorder=0) 
    ax.axline((0,0),slope=-2, color='k', ls='-', alpha=0.3, lw=lw, zorder=0) 
    
    ax.annotate("Input increases", xy=(-0.9, 0.25), xycoords='data', xytext=(-0.15, 0.6), textcoords='data', fontsize=fs,
                arrowprops=dict(arrowstyle="<-", connectionstyle="arc3, rad=0.2", lw=lw , ec='k'))
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.locator_params(nbins=3, axis='both')
    sns.despine(ax=ax)
    


def illustrate_sensory_weight_variance(ax = None, fs=6):
    
    if ax is None:
        _, ax = plt.subplots(1,1)
    
    sigma2_sens_fix, sigma2_pred_fix = 1, 1
    
    values_tested = np.linspace(-0.9,0.9,101)
    sigma2_delta_sens, sigma2_delta_pred = np.meshgrid(values_tested, values_tested)
    
    sensory_weight = (1/(sigma2_sens_fix + sigma2_delta_sens)) / (1/(sigma2_pred_fix + sigma2_delta_pred) + 1/(sigma2_sens_fix + sigma2_delta_sens))
    
    data = pd.DataFrame(sensory_weight, index=np.round(values_tested,1), columns=np.round(values_tested,1))
    sns.heatmap(data, cmap=cmap_sensory_prediction, vmin=0, vmax=1, xticklabels=20, yticklabels=20, square=True, 
               cbar_kws={'label': r'sensory weight', 'ticks':[0, 0.5, 1]}, ax=ax)
    ax.invert_yaxis()
    
    ax.set_xlabel(r'$\Delta$ sensory var', fontsize=fs) # variability
    ax.set_ylabel(r'$\Delta$ trial var', fontsize=fs) # variability
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.tick_params(axis='x', rotation=0)
    ax.locator_params(nbins=2, axis='both')
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.tick_params(size=2.0)
    cbar.ax.yaxis.label.set_size(fs)
               
    
def plot_neuromod_impact(pert_strength, alpha, xp, xs, xv, figsize=(2,2.5), fs=6, lw=1, s=15, show_ylabel=True, show_xlabel = True,
                            flg_plot_xlabel=True, flg_plot_bars = True, ax1 = None, ax2=None, highlight=True, file_for_data = None):
    
        if ax1 is None:
            f, (ax1, ax2) = plt.subplots(2,1, tight_layout=True, figsize=figsize, sharey=True, sharex=True)
           
        markers = ['o','s','d']
        
        ### load data
        if file_for_data is None:
            file_for_data = '../results/data/neuromod/data_neuromod_' + str(xp) + '_' + str(xs) + '_' + str(xv) + '.pickle'
            
        with open(file_for_data,'rb') as f:
            [xp, xs, xv, pert_strength, alpha] = pickle.load(f) 
        
        # plot
        for i, mfn_flag in enumerate(['10','01','11']): 
            
            ax2.axhline(alpha[0,i,0], color=cmap_sensory_prediction(alpha[0,i,0]), ls=':', zorder=0)
            ax1.axhline(alpha[0,i,1], color=cmap_sensory_prediction(alpha[0,i,1]), ls=':', zorder=0)
            
            ax2.scatter(pert_strength, alpha[:,i,0], marker=markers[i], c=alpha[:,i,0], cmap=cmap_sensory_prediction, 
                        vmin=0, vmax=1, s=s, clip_on=False, zorder=10)
            
            ax1.scatter(pert_strength, alpha[:,i,1], marker=markers[i], c=alpha[:,i,1], cmap=cmap_sensory_prediction, 
                        vmin=0, vmax=1, s=s, clip_on=False, zorder=10)
            

            
            if show_ylabel:
                ax1.set_ylabel('Sensory weight', fontsize=fs)
                ax2.set_ylabel('Sensory weight', fontsize=fs)
                
            ax1.set_xticklabels([])
            if show_xlabel:
                ax2.set_xlabel('Perturbation (1/s)', fontsize=fs)
            ax1.set_ylim([0,1])
            ax2.set_ylim([0,1])
            
            ax1.tick_params(size=2.0) 
            ax1.tick_params(axis='both', labelsize=fs)
            ax1.locator_params(nbins=2, axis='both')
            
            ax2.tick_params(size=2.0) 
            ax2.tick_params(axis='both', labelsize=fs)
            ax2.locator_params(nbins=2, axis='both')
             
            sns.despine(ax=ax1)
            sns.despine(ax=ax2)
            
            
def plot_neuromod_impact_inter(column, std_mean, n_std, figsize=(12,3), fs=6, lw=1, s=50, flg_plot_xlabel=True, 
                               flg_plot_bars = True, ax1 = None, ax2=None, ax3=None, highlight=True):
    
    if ax1 is None:
        f, (ax1, ax2, ax3) = plt.subplots(1,3, tight_layout=True, figsize=figsize, sharey=True)
       
    markers = ['o','s','d']
    identifier = '_column_' + str(column) + '_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
    
    # plot
    for i, mfn_flag in enumerate(['10','01','11']): 
        
        # SOM - VIP
        file_for_data = '../results/data/neuromod/data_weighting_neuromod_SOM-VIP_' + mfn_flag + identifier + '.pickle'
        with open(file_for_data,'rb') as f:
            [xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f) 
    
        ax1.axhline(alpha_before_pert, ls='-', color=cmap_sensory_prediction(alpha_before_pert), zorder=0, lw=lw)
        
        if highlight:
            ax1.axvspan(-0.04,0.04, color='#EFF6FB', alpha=0.2, zorder=0)
            ax1.axvspan(0.96,1.04, color='#EFF6FB', alpha=0.2, zorder=0)
            ax1.axvspan(0.45,0.55, color='#EFF6FB', alpha=0.2, zorder=0)
        
        ax1.scatter(xv, alpha_after_pert, marker=markers[i], c=alpha_after_pert, cmap=cmap_sensory_prediction, 
                    vmin=0, vmax=1, s=s) #ec='k'
        
        if flg_plot_bars:
            ax1.annotate('', xy=(0, -0.3), xycoords='axes fraction', xytext=(1, -0.3), 
                     arrowprops=dict(arrowstyle="wedge", color=Col_Rate_VIP)) #'#525252'
            
            ax1.annotate('', xy=(1, -0.4), xycoords='axes fraction', xytext=(0, -0.4), 
                     arrowprops=dict(arrowstyle="wedge", color=Col_Rate_SOM)) 
        
            ax1.text(0, -0.6, 'SOM', fontsize=fs, transform=ax1.transAxes, horizontalalignment='left', color=Col_Rate_SOM)
            ax1.text(1, -0.2, 'VIP', fontsize=fs, transform=ax1.transAxes, horizontalalignment='right', color=Col_Rate_VIP)
        
        ax1.set_xticks([])
        ax1.set_ylabel('Sensory weight', fontsize=fs)
        ax1.set_ylim([0,1])
        
        # VIP - PV
        file_for_data = '../results/data/neuromod/data_weighting_neuromod_VIP-PV_' + mfn_flag + identifier + '.pickle'
        with open(file_for_data,'rb') as f:
            [xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f) 
    
        ax2.axhline(alpha_before_pert, ls='-', color=cmap_sensory_prediction(alpha_before_pert), zorder=0, lw=lw)
        
        if highlight:
            ax2.axvspan(-0.04,0.04, color='#EFF6FB', alpha=0.2, zorder=0)
            ax2.axvspan(0.55,1.04, color='#EFF6FB', alpha=0.2, zorder=0)
        
        ax2.scatter(xp, alpha_after_pert, marker=markers[i], c=alpha_after_pert, cmap=cmap_sensory_prediction, 
                    vmin=0, vmax=1, s=s) #ec='k',
        
        if flg_plot_bars:
            ax2.annotate('', xy=(0, -0.3), xycoords='axes fraction', xytext=(1, -0.3), 
                     arrowprops=dict(arrowstyle="wedge", color=Col_Rate_PVv))
            
            ax2.annotate('', xy=(1, -0.4), xycoords='axes fraction', xytext=(0, -0.4), 
                     arrowprops=dict(arrowstyle="wedge", color=Col_Rate_VIP)) 
        
            ax2.text(0, -0.6, 'VIP', fontsize=fs, transform=ax2.transAxes, horizontalalignment='left', color=Col_Rate_VIP)
            ax2.text(1, -0.2, 'PV', fontsize=fs, transform=ax2.transAxes, horizontalalignment='right', color=Col_Rate_PVv)
        
        ax2.set_xticks([])
        if flg_plot_xlabel:
            ax2.set_xlabel('Fraction of interneurons activated', fontsize=fs)
        ax2.set_ylim([0,1])
        
        # PV - SOM
        file_for_data = '../results/data/neuromod/data_weighting_neuromod_PV-SOM_' + mfn_flag + identifier + '.pickle'
        with open(file_for_data,'rb') as f:
            [xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f) 
    
        ax3.axhline(alpha_before_pert, ls='-', color=cmap_sensory_prediction(alpha_before_pert), zorder=0, lw=lw)
        
        if highlight:
            ax3.axvspan(-0.04,0.45, color='#EFF6FB', alpha=0.2, zorder=0)
            ax3.axvspan(0.96,1.04, color='#EFF6FB', alpha=0.2, zorder=0)
        
        ax3.scatter(xs, alpha_after_pert, marker=markers[i], c=alpha_after_pert, cmap=cmap_sensory_prediction, 
                    vmin=0, vmax=1, s=s) # ec='k', 
        
        if flg_plot_bars:
            ax3.annotate('', xy=(0, -0.3), xycoords='axes fraction', xytext=(1, -0.3), 
                     arrowprops=dict(arrowstyle="wedge", color=Col_Rate_SOM))
            
            ax3.annotate('', xy=(1, -0.4), xycoords='axes fraction', xytext=(0, -0.4), 
                     arrowprops=dict(arrowstyle="wedge", color=Col_Rate_PVv)) 
        
            ax3.text(0, -0.6, 'PV', fontsize=fs, transform=ax3.transAxes, horizontalalignment='left', color=Col_Rate_PVv)
            ax3.text(1, -0.2, 'SOM', fontsize=fs, transform=ax3.transAxes, horizontalalignment='right', color=Col_Rate_SOM)
        
        ax3.set_xticks([])
        ax3.set_ylim([0,1])
        
        ax1.tick_params(size=2.0) 
        ax1.tick_params(axis='both', labelsize=fs)
        ax1.locator_params(nbins=2, axis='y')
        
        ax2.tick_params(size=2.0) 
        ax2.tick_params(axis='both', labelsize=fs)
        ax2.locator_params(nbins=2, axis='y')
        
        ax3.tick_params(size=2.0) 
        ax3.tick_params(axis='both', labelsize=fs)
        ax3.locator_params(nbins=2, axis='y')
        
        sns.despine(ax=ax1)
        sns.despine(ax=ax2)
        sns.despine(ax=ax3)


def plot_changes_upon_input2PE_neurons_new(std_mean = 1, n_std = 1, mfn_flag = '10', fs=6, lw=1, ms=3, mew=2,
                                           alpha = 1, ax1 = None, ax2 = None):
    
    if ax1 is None:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(2,2), sharex=True, sharey=True)
    
    ### rload data
    identifier = '_column_1_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
    file_for_data = '../results/data/neuromod/data_moments_vs_PE_neurons_' + mfn_flag + identifier + '.pickle'
    
    with open(file_for_data,'rb') as f:
        [pert_strength, m_act_lower1, v_act_lower1, v_act_higher1] = pickle.load(f)
    
    identifier = '_column_2_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
    file_for_data = '../results/data/neuromod/data_moments_vs_PE_neurons_' + mfn_flag + identifier + '.pickle'
    
    with open(file_for_data,'rb') as f:
        [pert_strength, m_act_lower2, v_act_lower2, v_act_higher2] = pickle.load(f)
    
    
    # PE and V in same subnetwork
    ax1.plot(pert_strength, v_act_lower1[:,0], color=color_v_neuron, marker=11, ls='None', lw=lw, ms=ms)
    ax1.plot(pert_strength, v_act_lower1[:,0], color=color_v_neuron, lw=lw, ms=ms, zorder=0, alpha=alpha)
    ax1.plot(pert_strength, v_act_lower1[:,1], color=color_v_neuron, lw=lw, ms=ms, zorder=0, alpha=alpha)
    ax1.plot(pert_strength, v_act_lower1[:,1], color=color_v_neuron, marker=10, ls='None', lw=lw, ms=ms)
    
    ax1.plot(pert_strength, v_act_higher2[:,0], color=color_v_neuron_light, marker=11, ls='None', lw=lw, ms=ms)
    ax1.plot(pert_strength, v_act_higher2[:,0], color=color_v_neuron_light, lw=lw, ms=ms, zorder=0, alpha=alpha)
    ax1.plot(pert_strength, v_act_higher2[:,1], color=color_v_neuron_light, marker=10, ls='None', lw=lw, ms=ms)
    ax1.plot(pert_strength, v_act_higher2[:,1], color=color_v_neuron_light, lw=lw, ms=ms, zorder=0, alpha=alpha)
    
    
    ### PE and V in different subnetworks  
    ax2.plot(pert_strength, v_act_lower2[:,0], color=color_v_neuron_light, marker=11, ls='None', lw=lw, ms=ms)
    ax2.plot(pert_strength, v_act_lower2[:,0], color=color_v_neuron, lw=lw, ms=ms, zorder=0, alpha=alpha)
    ax2.plot(pert_strength, v_act_lower2[:,1], color=color_v_neuron_light, marker=10, ls='None', lw=lw, ms=ms)
    ax2.plot(pert_strength, v_act_lower2[:,1], color=color_v_neuron, lw=lw, ms=ms, zorder=0, alpha=alpha)
    
    ax2.plot(pert_strength, v_act_higher1[:,0], color=color_v_neuron, marker=11, ls='None', lw=lw, ms=ms)
    ax2.plot(pert_strength, v_act_higher1[:,0], color=color_v_neuron_light, lw=lw, ms=ms, zorder=0, alpha=alpha)
    ax2.plot(pert_strength, v_act_higher1[:,1], color=color_v_neuron, marker=10, ls='None', lw=lw, ms=ms)
    ax2.plot(pert_strength, v_act_higher1[:,1], color=color_v_neuron_light, lw=lw, ms=ms, zorder=0, alpha=alpha)

    
    # make legends
    ax1.plot(np.nan, np.nan, color='k', ls="None", marker=10, label='pPE targeted', ms=ms)
    ax1.plot(np.nan, np.nan, color='k', ls="None", marker=11, label='nPE targeted', ms=ms)
    ax1.legend(loc=0, frameon=False, fontsize=fs, handletextpad=0.2)
    
    ax2.plot(np.nan, np.nan, ls="None", marker='_', label='Lower circuit', ms=6, 
             color=color_v_neuron, markeredgewidth=mew)
    ax2.plot(np.nan, np.nan, ls="None", marker='_', label='Higher circuit', ms=6, 
             color=color_v_neuron_light, markeredgewidth=mew)
    ax2.legend(loc=0, frameon=False, fontsize=fs, handletextpad=0.2)

    # polish plot
    ax1.tick_params(size=2.0) 
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    
    ax2.tick_params(size=2.0) 
    ax2.tick_params(axis='both', labelsize=fs)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    
    ax1.set_ylabel('V neuron', fontsize=fs, labelpad=10)
    ax2.set_ylabel('V neuron', fontsize=fs)
    ax1.set_xlabel('Perturbation of PE (1/s)', fontsize=fs)
    ax2.set_xlabel('Perturbation of PE (1/s)', fontsize=fs)
    
    ax2.set_ylim([0.5,1.2])
    
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
            

def plot_illustration_changes_upon_baseline_PE(BL = np.linspace(0,5,11), mean = 10, sd = 3, factor_for_visibility = 20,
                                               alpha=0.2, lw=1, ms=4, fs=6, ax1 = None, ax2 = None, ax3 = None):
    
    if ax1 is None:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, sharey=True)
    
    ### parameters
    b = np.sqrt(12) * sd / 2 + mean
    a = 2 * mean - b
    
    ### BL of nPE or pPE neurons in lower circuit   
    
    # prediction
    P_lower_nPE = (b + a)/2 - BL/(b - a)
    P_lower_pPE = (b + a)/2 + BL/(b - a)
    
    # variance in lower circuit
    V_lower_nPE = (b-a)**2/12 + BL**2/(b-a)**2 * (1 + 2*BL/(b-a)) + BL * (BL + (b-a)/2)
    V_lower_pPE = (b-a)**2/12 + BL**2/(b-a)**2 * (1 + 2*BL/(b-a)) + BL * (BL + (b-a)/2)
    
    # variance in higher circuit
    V_higher_nPE = (b-a)**2 / 12 + factor_for_visibility * (P_lower_nPE - P_lower_nPE[0])**2
    V_higher_pPE = (b-a)**2 / 12 + factor_for_visibility * (P_lower_pPE - P_lower_pPE[0])**2
    
    # plot
    ax1.plot(BL, P_lower_nPE, color=color_m_neuron, marker=11, ls='None', ms=ms)
    ax2.plot(BL, V_lower_nPE, color=color_v_neuron, marker=11, ls='None', ms=ms)
    ax3.plot(BL, V_higher_nPE, color=color_v_neuron, marker=11, ls='None', ms=ms)
    
    ax1.plot(BL, P_lower_nPE, lw=lw, color='k', alpha=alpha, zorder=0)
    ax2.plot(BL, V_lower_nPE, lw=lw, color='k', alpha=alpha, zorder=0)
    ax3.plot(BL, V_higher_nPE, lw=lw, color='k', alpha=alpha, zorder=0)
    
    ax1.plot(BL, P_lower_pPE, color=color_m_neuron, marker=10, ls='None', ms=ms)
    ax2.plot(BL, V_lower_pPE, color=color_v_neuron, marker=10, ls='None', ms=ms)
    ax3.plot(BL, V_higher_pPE, color=color_v_neuron, marker=10, ls='None', ms=ms)
    
    ax1.plot(BL, P_lower_pPE, color='k', lw=lw, alpha=alpha, zorder=0)
    ax2.plot(BL, V_lower_pPE, color='k', lw=lw, alpha=alpha, zorder=0)
    ax3.plot(BL, V_higher_pPE, color='k', lw=lw, alpha=alpha, zorder=0)
    
    ### BL of nPE or pPE neurons in higher circuit 
    
    # prediction
    P_higher_nPE = (b + a)/2 * np.ones_like(BL)
    P_higher_pPE = (b + a)/2 * np.ones_like(BL)
    
    # variance in lower circuit
    V_lower_nPE = (b-a)**2/12 * np.ones_like(BL)
    V_lower_pPE = (b-a)**2/12 * np.ones_like(BL)
    
    # variance in higher circuit
    V_higher_nPE = (b-a)**2/12 + BL**2/(b-a)**2 * (1 + 2*BL/(b-a)) + BL * (BL + (b-a)/2)
    V_higher_pPE = (b-a)**2/12 + BL**2/(b-a)**2 * (1 + 2*BL/(b-a)) + BL * (BL + (b-a)/2)
    
    # plot
    ax1.plot(BL, P_higher_nPE, color=color_m_neuron_light, marker=11, ls='None', ms=ms)
    ax2.plot(BL, V_lower_nPE, color=color_v_neuron_light, marker=11, ls='None', ms=ms)
    ax3.plot(BL, V_higher_nPE, color=color_v_neuron_light, marker=11, ls='None', ms=ms)
    
    ax1.plot(BL, P_higher_nPE, color='k', lw=lw, alpha=alpha, zorder=0)
    ax2.plot(BL, V_lower_nPE, color='k', lw=lw, alpha=alpha, zorder=0)
    ax3.plot(BL, V_higher_nPE, color='k', lw=lw, alpha=alpha, zorder=0)
    
    ax1.plot(BL, P_higher_pPE, color=color_m_neuron_light, marker=10, ls='None', ms=ms)
    ax2.plot(BL, V_lower_pPE, color=color_v_neuron_light, marker=10, ls='None', ms=ms)
    ax3.plot(BL, V_higher_pPE, color=color_v_neuron_light, marker=10, ls='None', ms=ms)
    
    ax1.plot(BL, P_higher_pPE, color='k', lw=lw, alpha=alpha, zorder=0)
    ax2.plot(BL, V_lower_pPE, color='k', lw=lw, alpha=alpha, zorder=0)
    ax3.plot(BL, V_higher_pPE, color='k', lw=lw, alpha=alpha, zorder=0)
    
    ax3.annotate('', xy=(0, -0.2), xycoords='axes fraction', xytext=(1, -0.2), 
                 arrowprops=dict(arrowstyle="wedge, tail_width=0.3, shrink_factor=0.5", color='#525252')) # simple, wedge, fancy, -|>
    
    ax3.text(0, -0.35, 'increase', fontsize=fs, transform=ax3.transAxes, horizontalalignment='left')
    
    ax1.set_ylabel('Prediction', fontsize=fs, labelpad=7)
    ax2.set_ylabel('Variance (lower)', fontsize=fs, labelpad=7)
    ax3.set_ylabel('Variance (higher)', fontsize=fs, labelpad=7)
    
    ax1.set_xticks([]), ax1.set_yticks([])
    ax2.set_xticks([]), ax2.set_yticks([])
    ax3.set_xticks([]), ax3.set_yticks([])
    
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    sns.despine(ax=ax3)
    
    
def plot_illustration_changes_upon_gain_PE(gains = np.linspace(0.5, 1.5, 11), mean = 10, sd = 3, factor_for_visibility = 1,
                                           alpha=0.2, ms=4, fs=6, lw=1, ax1 = None, ax2 = None, ax3 = None):
    
    if ax1 is None:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, sharey=True)#, tight_layout=True)
    
    ### parameters
    b = np.sqrt(12) * sd / 2 + mean
    a = 2 * mean - b
    
    ### gain of nPE or pPE neurons in lower circuit   
    P_lower_nPE = (b - gains*a + np.sqrt(gains) * (a-b)) / (1-gains)
    P_lower_nPE[np.isnan(P_lower_nPE)] = (b+a)/2
    
    P_lower_pPE = (gains*b - a + np.sqrt(gains) * (a-b)) / (gains-1)
    P_lower_pPE[np.isnan(P_lower_pPE)] = (b+a)/2
    
    V_lower_nPE = (b-a)**2/(3*(1-gains)**3) * (gains**2 * (1 - np.sqrt(gains))**3 - (gains - np.sqrt(gains))**3)
    V_lower_nPE[np.isnan(V_lower_nPE)] = (b-a)**2/12
    V_lower_pPE = (b-a)**2/(3*(gains-1)**3) * ((gains - np.sqrt(gains))**3 - gains**2 * (1 - np.sqrt(gains))**3)
    V_lower_pPE[np.isnan(V_lower_pPE)] = (b-a)**2/12
    
    V_higher_nPE = (b-a)**2 / 12 + factor_for_visibility * (P_lower_nPE - P_lower_nPE[gains==1])**2
    V_higher_pPE = (b-a)**2 / 12 + factor_for_visibility * (P_lower_pPE - P_lower_pPE[gains==1])**2
    
    # plot
    ax1.plot(gains, P_lower_nPE, color=color_m_neuron, marker=11, ms=ms, ls='None')
    ax2.plot(gains, V_lower_nPE, color=color_v_neuron, marker=11, ms=ms, ls='None')
    ax3.plot(gains, V_higher_nPE, color=color_v_neuron, marker=11, ms=ms, ls='None')
    
    ax1.plot(gains, P_lower_nPE, color='k', lw=lw, alpha=0.2, zorder=0)
    ax2.plot(gains, V_lower_nPE, color='k', lw=lw, alpha=0.2, zorder=0)
    ax3.plot(gains, V_higher_nPE, color='k', lw=lw, alpha=0.2, zorder=0)
    
    ax1.plot(gains, P_lower_pPE, color=color_m_neuron, marker=10, ms=ms, ls='None')
    ax2.plot(gains, V_lower_pPE, color=color_v_neuron, marker=10, ms=ms, ls='None')
    ax3.plot(gains, V_higher_pPE, color=color_v_neuron, marker=10, ms=ms, ls='None')
    
    ax1.plot(gains, P_lower_pPE, color='k', lw=lw, alpha=0.2, zorder=0)
    ax2.plot(gains, V_lower_pPE, color='k', lw=lw, alpha=0.2, zorder=0)
    ax3.plot(gains, V_higher_pPE, color='k', lw=lw, alpha=0.2, zorder=0)
    
    ### gain of nPE or pPE neurons in higher circuit 
    P_higher_nPE = (b+a)/2 * np.ones_like(gains)
    P_higher_pPE = (b+a)/2 * np.ones_like(gains)
    
    V_lower_nPE = (b-a)**2/12 * np.ones_like(gains)
    V_lower_pPE = (b-a)**2/12 * np.ones_like(gains)
    
    V_higher_nPE = (b-a)**2/(3*(1-gains)**3) * (gains*(1 - np.sqrt(gains))**3 - (gains - np.sqrt(gains))**3)
    V_higher_nPE[np.isnan(V_higher_nPE)] = (b-a)**2/12
    V_higher_pPE = (b-a)**2/(3*(gains-1)**3) * ((gains - np.sqrt(gains))**3 - gains * (1 - np.sqrt(gains))**3)
    V_higher_pPE[np.isnan(V_higher_pPE)] = (b-a)**2/12
    
    # plot
    ax1.plot(gains, P_higher_nPE, color=color_m_neuron_light, marker=11, ms=ms, ls='None')
    ax2.plot(gains, V_lower_nPE, color=color_v_neuron_light, marker=11, ms=ms, ls='None')
    ax3.plot(gains, V_higher_nPE, color=color_v_neuron_light, marker=11, ms=ms, ls='None')
    
    ax1.plot(gains, P_higher_nPE, color='k', lw=lw, alpha=0.2, zorder=0)
    ax2.plot(gains, V_lower_nPE, color='k', lw=lw, alpha=0.2, zorder=0)
    ax3.plot(gains, V_higher_nPE, color='k', lw=lw, alpha=0.2, zorder=0)
    
    ax1.plot(gains, P_higher_pPE, color=color_m_neuron_light, marker=10, ms=ms, ls='None')
    ax2.plot(gains, V_lower_pPE, color=color_v_neuron_light, marker=10, ms=ms, ls='None')
    ax3.plot(gains, V_higher_pPE, color=color_v_neuron_light, marker=10, ms=ms, ls='None')
    
    ax1.plot(gains, P_higher_pPE, color='k', lw=lw, alpha=0.2, zorder=0)
    ax2.plot(gains, V_lower_pPE, color='k', lw=lw, alpha=0.2, zorder=0)
    ax3.plot(gains, V_higher_pPE, color='k', lw=lw, alpha=0.2, zorder=0)
    
    ax3.annotate('', xy=(0.5, -0.2), xycoords='axes fraction', xytext=(0, -0.2), 
                 arrowprops=dict(arrowstyle="wedge", color='#525252')) # simple, wedge, fancy, -|>
    
    ax3.annotate('', xy=(0.5, -0.2), xycoords='axes fraction', xytext=(1, -0.2), 
                 arrowprops=dict(arrowstyle="wedge", color='#525252')) # simple, wedge, fancy, -|>
    
    ax3.text(0, -0.35, 'decrease', fontsize=fs, transform=ax3.transAxes, horizontalalignment='left')
    ax3.text(1, -0.35, 'increase', fontsize=fs, transform=ax3.transAxes, horizontalalignment='right')
    
    ax1.set_xticks([]), ax1.set_yticks([])
    ax2.set_xticks([]), ax2.set_yticks([])
    ax3.set_xticks([]), ax3.set_yticks([])
    
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    sns.despine(ax=ax3)


def plot_slope_variability(sd_1, sd_2, fitted_slopes_1, fitted_slopes_2, label_text, figsize=(3,2.5), fs=7, lw=1, ms=3, ax = None):
    
    if ax is None:
        f, ax = plt.subplots(1,1, tight_layout=True, figsize=(3,2.5))    
    ax.plot(sd_1, abs(fitted_slopes_1 - 1), '.-', color='k', label=label_text[0], lw=lw, ms=ms) 
    ax.plot(sd_2, abs(fitted_slopes_2 - 1), '.-', color=[0.5, 0.5, 0.5], label=label_text[1], lw=lw, ms=ms) 
    
    ax.legend(loc=0, handlelength=1, fontsize=fs, frameon=False)
    
    ax.set_xlabel('Variabilty, SD', fontsize=fs)
    ax.set_ylabel('|Slope|, m', fontsize=fs) # actually slope + 1
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    
    sns.despine(ax=ax)
    

def plot_slope_trail_duration(trial_durations,fitted_slopes_1, fitted_slopes_2, label_text, figsize=(3,2.5), fs=7, lw=1, ms=3, ax = None):
    
    if ax is None:
        f, ax = plt.subplots(1,1, tight_layout=True, figsize=(3,2.5))    
    ax.plot(trial_durations / 5000, abs(fitted_slopes_1-1), '.-', lw=lw, ms=ms, color='k', label=label_text[0])
    ax.plot(trial_durations / 5000, abs(fitted_slopes_2-1), '.-', lw=lw, ms=ms, color=[0.5, 0.5, 0.5], label=label_text[1])
    
    ax.set_xlabel(r'Trial duration / T$_0$', fontsize=fs)
    ax.set_ylabel('|Slope|, m', fontsize=fs)
    ax.legend(loc=4, handlelength=1, fontsize=fs, frameon=False, bbox_to_anchor=(1.9, 0))
    
    ax.set_ylim(bottom=0)
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    sns.despine(ax=ax)


def plot_std_vs_mean(min_means, max_means, m_std, n_std, lw=1, fs=6, figsize=(3,3), ax=None):
    
    colors = [color_running_average_stimuli, '#6A2E35']
    
    if ax is None:
        f, ax = plt.subplots(1,1, tight_layout=True, figsize=figsize)
    
    for i in range(2):
        x = np.linspace(min_means[i],max_means[i],10)
        ax.plot(x, m_std[i]*x+n_std[i], color=colors[i], lw=lw)
        
    ax.set_xlabel('mean (1/s)', fontsize=fs)
    ax.set_ylabel('SD (1/s)', fontsize=fs)
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    
    sns.despine(ax=ax)
        

def plot_example_contraction_bias(weighted_output, stimuli, n_trials, figsize=(2.5,2.8), ms=2,
                                  fs=7, lw=1.2, num_trial_ss=np.int32(50), trial_means=None, marker='.',
                                  min_means = None, max_means = None, m_std = None, n_std = None, 
                                  show_marker_inset=False, ax = None, plot_ylabel=True, plot_xlabel=True):
    
    if weighted_output.ndim==1:
        
        if trial_means is not None:
            trials_sensory_all = trial_means
        else:
            trials_sensory_all = np.mean(np.split(stimuli, n_trials),1)
        trials_estimated_all = np.mean(np.split(weighted_output, n_trials),1)
        
        # to take out transient ...
        trials_sensory = trials_sensory_all[num_trial_ss:]
        trials_estimated = trials_estimated_all[num_trial_ss:]
    
        if ax is None:
            f, ax = plt.subplots(1,1, tight_layout=True, figsize=figsize)                                                                              
        ax.plot(trials_sensory, (trials_estimated - trials_sensory), 'o', alpha = 1, color=color_stimuli_background, ms=ms)
        ax.axline((np.mean(stimuli), 0), slope=0, color='k', ls=':')
         
    else:
    
        if ax is None:
            f, ax = plt.subplots(1,1, tight_layout=True, figsize=figsize) 
            
        colors = [color_running_average_stimuli, '#6A2E35']
        colors_lighter = ['#FDCA9B', '#B7A4B6']
        
        for i in range(np.size(weighted_output,0)):
            
            if trial_means is not None:
                trials_sensory_all = trial_means[i,:]
            else:
                trials_sensory_all = np.mean(np.split(stimuli[i,:], n_trials),1)
            trials_estimated_all = np.mean(np.split(weighted_output[i,:], n_trials),1)
            
            # to take out transient ...
            trials_sensory = trials_sensory_all[num_trial_ss:]
            trials_estimated = trials_estimated_all[num_trial_ss:]
        
            ax.plot(trials_sensory, trials_estimated - trials_sensory, marker=marker, ls='None', color = colors_lighter[i], markersize=ms, zorder=0)#, color=color_stimuli_background)
            
            # plot line through data
            p =  np.polyfit(trials_sensory, (trials_estimated - trials_sensory), 1)
            x = np.linspace(min_means[i], max_means[i], 10)
            ax.plot(x, np.polyval(p, x), '-', color = colors[i], label=str(round(p[0]+1,2)), lw=lw)
            
            xmin, xmax = ax.get_xlim()[0], ax.get_xlim()[1]
            x = np.linspace(xmin, xmax, 10)
            ax.plot(x, np.polyval(p, x), '--', color = colors[i], lw=lw)
            
        ax.axline((np.mean(stimuli), 0), slope=0, color='k', ls=':', zorder=0, lw=lw)
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    if plot_xlabel:
        ax.set_xlabel('Stimulus (1/s)', fontsize=fs)
    if plot_ylabel:
        ax.set_ylabel('Bias (1/s)', fontsize=fs)
    sns.despine(ax=ax)
    

def plot_schema_inputs_tested(figsize=(2,2), fs=7, lw=0.5):
    
    inputs = np.nan*np.ones((5,5))
    di = np.diag_indices(5)
    inputs[di] = np.linspace(0,1,5)
    f, ax = plt.subplots(1,1, tight_layout=True, figsize=figsize)
    
    Text = np.chararray((5,5),unicode=True,itemsize=15)
    Text[:] = ' '
    Text[0,0] = 'o'
    Text[1,1] = 'v'
    Text[2,2] = '*'
    Text[3,3] = 'd'
    Text[4,4] = 's'
    Anno = pd.DataFrame(Text)

    data = pd.DataFrame(inputs)
    ax = sns.heatmap(data, vmin=0, vmax=1, cmap=ListedColormap(['white']), cbar=False, 
                     linewidths=lw, linecolor='k', annot=Anno, fmt = '', annot_kws={"fontsize": fs}) # cmap=colors
    
    ax.set_xlabel('variability across trial', fontsize=fs)
    ax.set_ylabel('variability within trial', fontsize=fs)
    ax.set_xticks([]), ax.set_yticks([])
    

def plot_impact_para(variability_across, weight_ctrl, weight_act, para_range_tested = [], fs=6, legend_title= '',
                     ms = 5, lw=1, figsize=(3,3), alpha=0.1, plot_ylabel=True, label_text = None, ax=None):
    
        if ax is None:
            _, ax = plt.subplots(1,1, tight_layout=True, figsize=figsize)
            
        
        if len(para_range_tested)==0:
            ax.plot(variability_across, weight_ctrl, 'k-', lw=lw, alpha=alpha, zorder=0) 
            ax.plot(variability_across, weight_act, 'k-', lw=lw, alpha=alpha, zorder=0) 
            ax.scatter(variability_across, weight_ctrl, marker='o', s=ms**2, c=weight_ctrl, 
                       cmap=cmap_sensory_prediction, vmin=0, vmax=1) 
            ax.scatter(variability_across, weight_act, marker='s', s=ms**2, c=weight_act, 
                       cmap=cmap_sensory_prediction, vmin=0, vmax=1)  
            
            if label_text is not None:
                ax.scatter(np.nan, np.nan, marker='o', s=(ms-1)**2, c='k', label=label_text[0])
                ax.scatter(np.nan, np.nan, marker='s', s=(ms-1)**2, c='k', label=label_text[1])
                
                if label_text is not None:
                    legend = ax.legend(loc=2, frameon=False, fontsize=fs, ncol=2, handletextpad=0.1, columnspacing=1,
                                       title=legend_title, bbox_to_anchor=(-0.05, 1.45))
                    plt.setp(legend.get_title(), fontsize=fs)
            
        else:
            ax.plot(variability_across, weight_ctrl, 'k-', lw=lw, alpha=alpha, zorder=0) 
            ax.scatter(variability_across, weight_ctrl, marker='o', s=ms**2, c=weight_ctrl, cmap=cmap_sensory_prediction, vmin=0, vmax=1)
            
            if label_text is not None:
                ax.scatter(np.nan, np.nan, marker='o', s=(ms-1)**2, c='k', label=label_text[0])
            
            markers = ['<','>', 1, 2, 3, 4]
            
            for i in range(len(para_range_tested)):
                ax.plot(variability_across, weight_act[i,:], 'k-', lw=lw, alpha=alpha, zorder=0) 
                ax.scatter(variability_across, weight_act[i,:], s=ms**2, marker=markers[i], c=weight_act[i,:], cmap=cmap_sensory_prediction, vmin=0, vmax=1)
                
                if label_text is not None:
                    ax.scatter(np.nan, np.nan, marker=markers[i], s=(ms-1)**2, c='k', label=label_text[i+1])
                    
            if label_text is not None:
                legend = ax.legend(loc=2, frameon=False, fontsize=fs, ncol=3, handletextpad=0.1, columnspacing=1,
                                   title=legend_title, bbox_to_anchor=(-0.05, 1.45))
                plt.setp(legend.get_title(), fontsize=fs)
                
                
        ax.set_ylim([0, 1])
        
        ax.tick_params(size=2.0) 
        ax.tick_params(axis='both', labelsize=fs)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        
        if plot_ylabel:
            ax.set_ylabel('Sensory weight', fontsize=fs)
        
        ax.annotate('', xy=(0, -0.3), xycoords='axes fraction', xytext=(1, -0.3), 
                 arrowprops=dict(arrowstyle="wedge", color='#525252'))
        
        ax.annotate('', xy=(1, -0.4), xycoords='axes fraction', xytext=(0, -0.4), 
                 arrowprops=dict(arrowstyle="wedge", color='#525252')) 
    
        ax.text(0, -0.6, 'Stimulus variability', fontsize=fs, transform=ax.transAxes, horizontalalignment='left')
        ax.text(1, -0.2, 'Trial variability', fontsize=fs, transform=ax.transAxes, horizontalalignment='right')
    
        sns.despine(ax=ax)
        
        
        
def plot_weight_over_trial(weight_ctrl, weight_mod, n_trials, fs=6, lw=1, leg_text=None, marker_every_n = 500, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(4,4))
    
    #for id_stim in id_stims:
    id_stim = 0
    s = 30
        
    ### control
    alpha_split = np.array_split(weight_ctrl[:,id_stim], n_trials)
    alpha_avg_trial = np.mean(alpha_split,0)
    alpha_std_trial = np.std(alpha_split,0)
    sem = alpha_std_trial/np.sqrt(n_trials)
    trial_fraction = np.linspace(0,1,len(alpha_avg_trial))
    
    ax.scatter(trial_fraction[::marker_every_n], alpha_avg_trial[::marker_every_n], marker='o', s = s, linewidth=lw, 
               c=alpha_avg_trial[::marker_every_n], cmap=cmap_sensory_prediction, vmin=0, vmax=1, edgecolors='k')
    ax.fill_between(trial_fraction, alpha_avg_trial-sem, alpha_avg_trial+sem, alpha=0.1, color='k', zorder=0)

    ### modulated
    alpha_split = np.array_split(weight_mod[:,id_stim], n_trials)
    alpha_avg_trial = np.mean(alpha_split,0)
    alpha_std_trial = np.std(alpha_split,0)
    sem = alpha_std_trial/np.sqrt(n_trials)
    trial_fraction = np.linspace(0,1,len(alpha_avg_trial))
    
    marker_every_n //= 5
    ax.scatter(trial_fraction[::marker_every_n], alpha_avg_trial[::marker_every_n], marker='s', s = s, linewidth=lw, 
               c=alpha_avg_trial[::marker_every_n], cmap=cmap_sensory_prediction, vmin=0, vmax=1, edgecolors='k')
    ax.fill_between(trial_fraction, alpha_avg_trial-sem, alpha_avg_trial+sem, alpha=0.1, color='k', zorder=0)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim([ymin, ymax])
    
    ax.set_xlim([0,1])
    
    if leg_text is not None:
        ax.plot(np.nan, np.nan, 'o', color='k', label=leg_text[0], ms=4)
        ax.plot(np.nan, np.nan, 's', color='k', label=leg_text[1], ms=4)
        ax.legend(loc=10, frameon=False, handlelength=2, fontsize=fs, borderpad=2, bbox_to_anchor=[0.8,0.4])
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', labelsize=fs)
    ax.tick_params(size=2.0) 
    ax.set_ylabel('Sensory weight', fontsize=fs)
    ax.set_xlabel('Fraction of time elapsed in trial', fontsize=fs)
    sns.despine(ax=ax)


def plot_transitions_examples(n_trials, trial_duration, stimuli, alpha, beta, weighted_output, 
                              time_plot = 0, ylim=None, xlim=None, plot_ylable=True, lw=1, plot_xlabel = True, 
                              figsize=(3.5,5), plot_only_weights=False, fs=6, transition_point=50, ax2 = None):
    
    if ax2 is None:
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
    
    ax2.plot(time[time > time_plot * time[-1]], alpha[time > time_plot * time[-1]], color='k', lw=lw, label='stimulus')

    n_every = 20
    x = time[time > time_plot * time[-1]]
    y = alpha[time > time_plot * time[-1]]
    ax2.scatter(x[::n_every], y[::n_every], c=y[::n_every], cmap=cmap_sensory_prediction, vmin=0, vmax=1, marker='.', zorder=5, s=1)
    
    if plot_ylable:
        ax2.set_ylabel('Sensory weight', fontsize=fs)
        
    ax2.axvline(transition_point, color='k', ls='--', lw=lw, zorder=10)
    if plot_xlabel:
        ax2.set_xlabel('Time (number of trials)', fontsize=fs)
    ax2.set_xlim([time_plot * time[-1],time[-1]])
    ax2.set_ylim([0,1.05])
    if xlim is not None:
        ax2.set_xlim(xlim)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.tick_params(axis='both', labelsize=fs)
    ax2.tick_params(size=2.0) 
    sns.despine(ax=ax2)
    


def plot_fraction_sensory_heatmap(fraction_sensory_median, para_tested_first, para_tested_second, every_n_ticks, xlabel='', 
                                  ylabel='', vmin=0, vmax=1, decimal = 1e2, title='', cmap = cmap_sensory_prediction,
                                  figsize=(6,4.5), fs=6, data_example = None, data_text = None, square = False, 
                                  xticklabels = None, yticklabels = None, ax = None):
    
    if ax is None:
        plt.figure(tight_layout=True, figsize=figsize)
        ax = plt.gca()
        
    index = np.round(decimal*para_tested_first)/decimal
    columns = np.round(decimal*para_tested_second)/decimal
    
    data = pd.DataFrame(fraction_sensory_median, columns=columns, index=index)
    sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=cmap, xticklabels=every_n_ticks, square=square,
                yticklabels=every_n_ticks, cbar_kws={'label': 'Sensory weight', 'ticks':[0, 0.5, 1]}, ax=ax)
    
    ax.invert_yaxis()
    
    if data_example is not None:
        
        x = data_example[0,:]
        y = data_example[1,:]
        
        mx, nx = np.polyfit([columns.min()-0.5, columns.max()+0.5], ax.get_xlim(), 1)
        my, ny = np.polyfit([index.min()-0.5, index.max()+0.5], ax.get_ylim(), 1)
        
        for i in range(len(x)):
            ax.text(mx * x[i] + nx, my * y[i] + ny, data_text[i], fontsize=fs, ha='center', va='center')
    
    ax.locator_params(nbins=2)
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.tick_params(size=2.0)
    cbar.ax.yaxis.label.set_size(fs)
    
    ax.set_xticks([columns.min()+0.5, columns.max()+0.5])
    ax.set_yticks([index.min()+0.5, index.max()+0.5])
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)    
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)


def plot_weighting_limit_case_example(n_trials, trial_duration, stimuli, prediction, mean_of_prediction, variance_per_stimulus, 
                            variance_prediction, alpha, beta, weighted_output, time_plot = 0.8, plot_legend=True,
                            figsize=(4,3), fs=6, lw=1, plot_prediction=False, ax1=None, ax2=None, ax3=None):
    if ax1 is None:
        f1, ax1 = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    if ((ax2 is None) and plot_prediction):
        f1, ax2 = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    if ax3 is None:
        f1, ax3 = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    
    time = np.arange(len(stimuli))/trial_duration
    
    for i in range(n_trials):
        ax1.axvspan(2*i, (2*i+1), color='#F5F4F5')
    ax1.plot(time[time > time_plot * time[-1]], stimuli[time > time_plot * time[-1]], 
             color=color_stimuli_background, lw=lw, marker='|', ls="None")
    ax1.plot(time[time > time_plot * time[-1]], weighted_output[time > time_plot * time[-1]], 
             color=color_weighted_output, lw=lw) 
    ax1.set_ylabel('Weighted output (1/s)', fontsize=fs)
    ax1.set_xlabel('Time (number of trials)', fontsize=fs)
    
    if plot_legend:
        ax1.plot(np.nan, np.nan, color=color_stimuli_background, ms=6, marker='_', 
                 markeredgewidth = 2,ls="None", label='Input')
        ax1.legend(loc=2, fontsize=fs, frameon=True, handletextpad=0.1, bbox_to_anchor=(0,1.05))
        
    ax1.set_xlim([time_plot * time[-1],time[-1]])
    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.tick_params(size=2.0) 
    sns.despine(ax=ax1)
    
    if plot_prediction:
        for i in range(n_trials):
            ax2.axvspan(2*i, (2*i+1), color='#F5F4F5')
        ax2.plot(time[time > time_plot * time[-1]], prediction[time > time_plot * time[-1]], 
                 color=color_m_neuron, lw=lw, label='prediction')
        ax2.plot(time[time > time_plot * time[-1]], mean_of_prediction[time > time_plot * time[-1]], 
                 color=color_mean_prediction, lw=lw, label='mean of prediction')
        ax2.set_ylabel('Activity (1/s)', fontsize=fs)
        ax2.set_xlabel('Time (number of trials)')
        if plot_legend:
            ax2.legend(loc=0, frameon=False, fontsize=fs)
        ax2.set_xlim([time_plot * time[-1],time[-1]])
        ax2.set_ylim([ax1.get_ylim()[0], ax1.get_ylim()[1]])
        ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax2.tick_params(axis='both', labelsize=fs)
        ax2.tick_params(size=2.0) 
        sns.despine(ax=ax2)
    
    for i in range(n_trials):
        ax3.axvspan(2*i, (2*i+1), color='#F5F4F5', zorder=0)
   
    ax3.plot(time[time > time_plot * time[-1]], alpha[time > time_plot * time[-1]], 
             color='k', lw=lw)
    
    n_every = 10
    x = time[time > time_plot * time[-1]]
    y = alpha[time > time_plot * time[-1]]
    ax3.scatter(x[::n_every], y[::n_every], c=y[::n_every], cmap=cmap_sensory_prediction, vmin=0, vmax=1, marker='.', zorder=5, s=1)
    
    ax3.set_ylabel('Sensory weight', fontsize=fs)
    ax3.set_xlabel('Time (number of trials)', fontsize=fs)

    ax3.set_xlim([time_plot * time[-1],time[-1]])
    ax3.set_ylim([0,1])
    ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.tick_params(axis='both', labelsize=fs)
    ax3.tick_params(size=2.0) 
    sns.despine(ax=ax3)
    

def plot_mse_test_distributions(dev, dist_types=None, mean=None, std=None, SEM=None, title=None,
                                plot_dists=False, x_lim = None, pa=0.8, fig_size=(5,3), fs = 5, 
                                plot_ylabel = True, plot_xlabel = True, figsize=(5,5), ax = None):
    
    ### show mean squared error
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    ax.locator_params(nbins=3)
    
    time = np.arange(0, np.size(dev,1), 1)
    f = 0.25
        
    num_rows = np.size(dev,0)
    colors = ['#093A3E', '#E0A890', '#99C24D', '#6D213C', '#0E9594']
    
    for i in range(num_rows):
        if dist_types is not None:
            
            if dist_types[i]=='uniform':
                dist_types_label = dist_types[i]
            elif dist_types[i]=='normal':
                dist_types_label = dist_types[i]
            elif dist_types[i]=='lognormal':
                dist_types_label = dist_types[i]
            elif dist_types[i]=='gamma':
                dist_types_label = dist_types[i]
            elif dist_types[i]=='binary_equal_prop':
                dist_types_label = 'binary ' + r'(p$_a$=0.5)'
            elif dist_types[i]=='binary_unequal_prop':
                dist_types_label = 'binary ' + r'(p$_a$=0.8)'
            
            ax.plot(time[time>time[-1]*f]/time[-1], dev[i,time>time[-1]*f], color=colors[i], label=dist_types_label)
        else:
            ax.plot(time[time>time[-1]*f]/time[-1], dev[i,time>time[-1]*f], color=colors[i])
            
        if SEM is not None:
            ax.fill_between(time[time>time[-1]*f]/time[-1], dev[i,time>time[-1]*f] - SEM[i,time>time[-1]*f], 
                            dev[i,time>time[-1]*f] + SEM[i,time>time[-1]*f], color=colors[i], alpha=0.3)
    if x_lim is not None:   
        ax.set_xlim(x_lim/time[-1])
            
    if dist_types is not None:
        ax.legend(loc=0, ncol=2, handlelength=1, frameon=False, fontsize=fs)#, bbox_to_anchor=(0.05, 1))
    
    ax.tick_params(size=2)
    ax.tick_params(axis='both', labelsize=fs)
    if plot_xlabel:
        ax.set_xlabel('Time (fraction of stimulus duration)', fontsize=fs)
    if plot_ylabel:
        ax.set_ylabel('Normalised \nerror (%)', fontsize=fs)
    if title is not None:
        ax.set_title(title, fontsize=fs)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=f)
    ax.set_xticks([f, 0.5, f+0.5, 1])
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
        
        
def plot_neuron_activity(end_of_initial_phase, means_tested, variances_tested, activity_interneurons,
                         activity_pe_neurons, id_fixed = 0, vmax = None, lw = 1, fs=6, figsize=(3,3),
                         ax1 = None, ax2 = None):
    
    if activity_interneurons is not None:
        temp_int_activity = np.mean(activity_interneurons[:, :, end_of_initial_phase:, :], 2)
    
        int_activity = np.zeros((temp_int_activity.shape[0], temp_int_activity.shape[1], 3))
        int_activity[:,:,0] = 0.5 *(temp_int_activity[:,:,0] + temp_int_activity[:,:,1])
        int_activity[:,:,1:] = temp_int_activity[:,:,2:]
        
        colors_in = [Col_Rate_PVv, Col_Rate_SOM, Col_Rate_VIP]
    
    if activity_pe_neurons is not None:
        pe_activity = np.mean(activity_pe_neurons[:, :, end_of_initial_phase:, :], 2)

        colors_pe = [Col_Rate_nE, Col_Rate_pE]
    
    if ax1 is None:
        _, ax1 = plt.subplots(1,1, figsize=figsize, tight_layout=True)

    if ax2 is None:
        _, ax2 = plt.subplots(1,1, figsize=figsize, tight_layout=True)
    
    if activity_interneurons is not None: 
        for i in range(3):
            ax1.plot(means_tested, int_activity[:,id_fixed,i], color=colors_in[i], linewidth=lw)
            
        ax1.text(means_tested[0], 1 * np.max(int_activity[:,id_fixed,:]),'PV', fontsize=fs, color=Col_Rate_PVv)
        ax1.text(means_tested[0], 0.95 * np.max(int_activity[:,id_fixed,:]),'SOM', fontsize=fs, color=Col_Rate_SOM)
        ax1.text(means_tested[0], 0.9 * np.max(int_activity[:,id_fixed,:]),'VIP', fontsize=fs, color=Col_Rate_VIP)
    
    if activity_pe_neurons is not None:
        for i in range(2):
            ax1.plot(means_tested, pe_activity[:,id_fixed,i], color=colors_pe[i], linewidth=lw)
            
        ax1.text(means_tested[0], 0.9 * np.min(pe_activity[:,id_fixed,:]),'nPE', fontsize=fs, color=Col_Rate_nE)
        ax1.text(means_tested[0], 1.05 * np.max(pe_activity[:,id_fixed,:]),'pPE', fontsize=fs, color=Col_Rate_pE)
            
    
    ax1.tick_params(size=2)
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.set_ylabel('Activity (1/s)', fontsize=fs)
    ax1.set_xlabel('Input mean', fontsize=fs)
    ax1.locator_params(nbins=3)
    sns.despine(ax=ax1)
    
    if activity_interneurons is not None:
        for i in range(3):
            ax2.plot(variances_tested, int_activity[id_fixed,:,i], color=colors_in[i], linewidth=lw)
    
    if activity_pe_neurons is not None:
        for i in range(2):
            ax2.plot(variances_tested, pe_activity[id_fixed,:,i], color=colors_pe[i], linewidth=lw)
    
    max_y1 = ax1.get_ylim()[1]
    max_y2 = ax2.get_ylim()[1]
    min_y1 = ax1.get_ylim()[0]
    min_y2 = ax2.get_ylim()[0]
    
    if max_y1>max_y2:
        ax2.set_ylim(top=max_y1)
    else:
        ax1.set_ylim(top=max_y2)
        
    if min_y1<min_y2:
        ax2.set_ylim(bottom=min_y1)
    else:
        ax1.set_ylim(bottom=min_y2)
    
    ax2.tick_params(size=2)
    ax2.tick_params(axis='both', labelsize=fs)
    ax2.set_xlabel('Input variance', fontsize=fs)
    ax2.locator_params(nbins=3)
    sns.despine(ax=ax2)
    

def plot_mse_heatmap(means_tested, variances_tested, dev, vmax = None, lw = 1, ax1=None, title=None, 
                     show_mean=True, fs=6, figsize=(5,5), x_example = None, y_example = None, digits_round = 10):
    
    if ax1 is None:
        f, ax1 = plt.subplots(1,1, figsize=figsize, tight_layout=True)
    
    if show_mean:
        color_cbar = LinearSegmentedColormap.from_list(name='mse_mean', colors=['#FEFAE0', color_m_neuron])
    else:
        color_cbar = LinearSegmentedColormap.from_list(name='mse_variance', colors=['#FEFAE0', color_v_neuron])
    
    dev_abs = 100 * np.abs(dev.T)
    data = pd.DataFrame(dev_abs, index=variances_tested, columns=means_tested)
    if vmax is None:
        vmax = np.ceil(digits_round * np.max(dev_abs))/digits_round 
    
    vmin = np.floor(digits_round * np.min(dev_abs))/digits_round 
    
    sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=color_cbar, xticklabels=3, yticklabels=2, ax=ax1,
                      cbar_kws={'label': 'Normalised \nerror (%)', 'ticks': [vmin, vmax]})
    
    ax1.invert_yaxis()
    
    mx, nx = np.polyfit([means_tested.min()-0.5, means_tested.max()+0.5], ax1.get_xlim(), 1)
    my, ny = np.polyfit([variances_tested.min()-0.5, variances_tested.max()+0.5], ax1.get_ylim(), 1)

    
    if x_example is not None:
        ax1.plot(mx * x_example + nx, my * y_example + ny, marker='*', color='k')
        
    
    ax1.tick_params(size=2.0) 
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.set_xlabel('Input mean', fontsize=fs)
    ax1.set_ylabel('Input variance', fontsize=fs)
    ax1.figure.axes[-1].yaxis.label.set_size(fs)
    ax1.figure.axes[-1].tick_params(labelsize=fs)
    ax1.locator_params(nbins=3)
    
    if title is not None:
        ax1.set_title(title, fontsize=fs, pad=10)
    
    sns.despine(ax=ax1, bottom=False, top=False, right=False, left=False)
   

def plot_example_mean(stimuli, trial_duration, m_neuron, perturbation_time = None, 
                      figsize=(6,4.5), ylim_mse=None, lw=1, fs=6, tight_layout=True, 
                      legend_flg=True, mse_flg=True, ax1=None, ax2=None):
    
    ### set figure architecture
    if ax1 is None:
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
    
    ax1.plot(trials, stimuli, color=color_stimuli_background, lw=lw, marker='|', ls="None")
    ax1.plot(np.nan, np.nan, color=color_stimuli_background, label=r'sensory input, $S$', marker='s', ls="None")
    ax1.plot(trials, running_average, color=color_running_average_stimuli, label=r'averaged mean, $\overline{S}$', lw=lw)
    ax1.plot(trials, m_neuron, color=color_m_neuron, lw=lw) 
    ax1.set_xlim([0,max(trials)])
    ax1.set_ylabel('Activity (1/s)', fontsize=fs)
    
    if ~mse_flg:
        ax1.set_xlabel('Time (fraction of stimulus duration)', fontsize=fs)
        
    ax1.set_title('Activity of M neuron encodes input mean', fontsize=fs, pad=10)
    ax1.tick_params(axis='both', labelsize=fs)
    
    if legend_flg:
        ax1.legend(loc=4, ncol=1, handlelength=1, frameon=True, fontsize=fs, markerscale=1)
        
    ax1.tick_params(size=2.0) 
    sns.despine(ax=ax1)
    
    if mse_flg:
        ax2.plot(trials, (running_average - m_neuron)**2, color=color_m_neuron)
        if ylim_mse is not None:
            ax2.set_ylim(ylim_mse)
        ax2.set_xlim([0,max(trials)])
        ax2.set_ylabel('MSE', fontsize=fs)
        ax2.set_xlabel('Time (fraction of stimulus duration)', fontsize=fs)
        ax2.tick_params(size=2.0) 
        ax2.tick_params(axis='both', labelsize=fs)
        sns.despine(ax=ax2)
    
    
def plot_example_variance(stimuli, trial_duration, v_neuron, perturbation_time = None, 
                          figsize=(6,4.5), lw=1, fs=6, tight_layout=True, 
                          legend_flg=True, mse_flg=True, ax1=None, ax2=None):
    if ax1 is None:
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
    
    ax1.plot(trials, momentary_variance, color=color_stimuli_background, ls="None", marker='|')
    ax1.plot(np.nan, np.nan, color=color_stimuli_background, label=r'$(S-\overline{S})^2$', marker='s', ls="None")
    ax1.plot(trials, variance_running, color=color_running_average_stimuli, label='averaged \nvariance', lw=lw)
    ax1.plot(trials, v_neuron, color=color_v_neuron, lw=lw) 
    ax1.set_xlim([0,max(trials)])
    ax1.set_ylabel('Activity (1/s)', fontsize=fs)
    if ~mse_flg:
        ax1.set_xlabel('Time (fraction of stimulus duration)', fontsize=fs)
    ax1.set_title('Activity of V neuron encodes input variance', fontsize=fs, pad=10)
    ax1.tick_params(axis='both', labelsize=fs)
    if legend_flg:
        ax1.legend(loc=1, ncol=1, frameon=True, handlelength=1, fontsize=fs)
    ax1.tick_params(size=2.0) 
    sns.despine(ax=ax1)
    
    if mse_flg:
        ax2.plot(trials, (variance_running - v_neuron)**2, color=color_v_neuron)
        ax2.set_xlim([0,max(trials)])
        ax2.set_ylabel('MSE', fontsize=fs)
        ax2.set_xlabel('Time (fraction of stimulus duration)', fontsize=fs)
        ax2.tick_params(size=2.0)
        ax2.tick_params(axis='both', labelsize=fs)
        sns.despine(ax=ax2)
  