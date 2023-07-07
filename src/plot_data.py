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
from src.functions_simulate import random_binary_from_moments, random_binary_from_moments, random_lognormal_from_moments


# %% set colors

Col_Rate_E = '#9E3039'
Col_Rate_nE = '#A06A94' #'#955F89'
Col_Rate_nD = '#BF9BB8' 
Col_Rate_pE = '#9E3039' # '#CB9173' 
Col_Rate_pD = '#DEB8A6' 
Col_Rate_PVv = '#508FCE' 
Col_Rate_PVm = '#2B6299' 
Col_Rate_SOM = '#285C8F' #'#79AFB9' 
Col_Rate_VIP = '#163350' #'#39656D' 

color_sensory = '#D76A03'
color_prediction = '#39656D' #'#19535F'

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


def plot_legend_illustrations(ax, fs=6, mew=2):
    
    p1, = ax.plot(np.nan, np.nan, marker=10, color='k', ms=4, ls='None')
    p2, = ax.plot(np.nan, np.nan, marker=11, color='k', ms=4, ls='None')
    p3, = ax.plot(np.nan, np.nan, marker='_', color=color_m_neuron, ms=6, ls='None', markeredgewidth=mew)
    p4, = ax.plot(np.nan, np.nan, marker='_', color=color_v_neuron, ms=6, ls='None', markeredgewidth=mew)
    p5, = ax.plot(np.nan, np.nan, marker='_', color=color_m_neuron_light, ms=6, ls='None', markeredgewidth=mew)
    p6, = ax.plot(np.nan, np.nan, marker='_', color=color_v_neuron_light, ms=6, ls='None', markeredgewidth=mew)
    
    ax.legend([p1, p2, (p3, p4), (p5, p6)], ['pPE neuron targeted', 'nPE neuron targeted', 'in lower PE circuit', 'in higher PE circuit'],
              handler_map={tuple: HandlerTuple(ndivide=None)}, loc=6, fontsize=fs, frameon=False, 
              bbox_to_anchor=(-0.6, 0.5), ncol=2)




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
        sc = ax.scatter((results_base_nPE_10[1:] - results_base_nPE_10[0]), (results_base_pPE_10[1:] - results_base_pPE_10[0]),
                        c=colors, marker='o', s=s)
        
        ax.scatter((results_base_nPE_11[1:] - results_base_nPE_11[0]), (results_base_pPE_11[1:] - results_base_pPE_11[0]), 
                   c=colors, marker='s', s=s)
        
        ax.scatter((results_base_nPE_01[1:] - results_base_nPE_01[0]), (results_base_pPE_01[1:] - results_base_pPE_01[0]), 
                    c=colors, marker='d', s=s)
        
    else:
        sc = ax.scatter((results_gain_nPE_10[1:] - results_gain_nPE_10[0]), (results_gain_pPE_10[1:] - results_gain_pPE_10[0]),
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
    
    sc = ax.scatter((results_base_nPE_10[1:] - results_base_nPE_10[0]) + (results_base_pPE_10[1:] - results_base_pPE_10[0]),
                (results_gain_nPE_10[1:] - results_gain_nPE_10[0]) + (results_gain_pPE_10[1:] - results_gain_pPE_10[0]), 
                c=colors, marker='o', s=s)
    
    ax.scatter((results_base_nPE_11[1:] - results_base_nPE_11[0]) + (results_base_pPE_11[1:] - results_base_pPE_11[0]),
                (results_gain_nPE_11[1:] - results_gain_nPE_11[0]) + (results_gain_pPE_11[1:] - results_gain_pPE_11[0]), 
                c=colors, marker='s', s=s)
    
    ax.scatter((results_base_nPE_01[1:] - results_base_nPE_01[0]) + (results_base_pPE_01[1:] - results_base_pPE_01[0]),
                (results_gain_nPE_01[1:] - results_gain_nPE_01[0]) + (results_gain_pPE_01[1:] - results_gain_pPE_01[0]), 
                c=colors, marker='d', s=s)
    
    # handles = sc.legend_elements()[0]
    # legend1 = ax.legend(title='Targets', handles=handles, labels=['PV','SOM','VIP'], frameon=False, fontsize=fs)
    # plt.setp(legend1.get_title(),fontsize=fs)
    # ax.add_artist(legend1)
    
    xbound = max(np.abs(ax.get_xlim()))
    ybound = max(np.abs(ax.get_ylim()))
    ax.set_xlim([-xbound,xbound])
    ax.set_ylim([-ybound,ybound])
   # ax.axhspan(-ybound, 0, xmin=0, xmax=0.5, alpha=0.1)
   # ax.axhspan(0, ybound, xmin=0.5, xmax=1, alpha=0.1, color='r')
    
    ax.set_xlabel('Baseline of nPE & pPE', fontsize=fs)#, loc='left', labelpad=120)
    ax.set_ylabel(r'$\Delta$ gain of nPE & pPE', fontsize=fs)#, loc='bottom', labelpad=150)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    # plt.scatter(np.nan, np.nan, c='k', marker='o', label='10', s=s)
    # plt.scatter(np.nan, np.nan, c='k', marker='s', label='11', s=s)
    # plt.scatter(np.nan, np.nan, c='k', marker='d', label='01', s=s)
    # legend2 = ax.legend(loc="upper right", title="MFN", frameon=False, fontsize=fs)
    # plt.setp(legend2.get_title(),fontsize=fs)
    # ax.add_artist(legend2)
    
    ax.text(0.3,0.2,'Variance \nincreases', color='#525252', fontsize=fs)
    ax.text(-1.2,-0.2,'Variance \ndecreases', color='#525252', fontsize=fs)
    
    ax.axline((0,0),slope=-0.5, color='k', ls='-', alpha=0.3, lw=lw, zorder=0) 
    ax.axline((0,0),slope=-1, color='k', ls='-', alpha=0.3, lw=lw, zorder=0) 
    ax.axline((0,0),slope=-2, color='k', ls='-', alpha=0.3, lw=lw, zorder=0) 
    
    ax.annotate("Input increases", xy=(-0.9, 0.25), xycoords='data', xytext=(-0.15, 0.6), textcoords='data', fontsize=fs,
                 arrowprops=dict(arrowstyle="<-", connectionstyle="arc3, rad=0.2", lw=lw , ec='k'))
    
    #ax.axhline(0, color='k', ls='--', alpha=0.5)
    #ax.axvline(0, color='k', ls='--', alpha=0.5)
    
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
                

def plot_neuromod_impact_inter(column, std_mean, n_std, figsize=(12,3), fs=6, lw=1, s=50, flg_plot_xlabel=True, 
                               flg_plot_bars = True, ax1 = None, ax2=None, ax3=None):
    
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
        #ax1.plot(xv, alpha_after_pert, ':', color='k', zorder=0)
        ax1.scatter(xv, alpha_after_pert, marker=markers[i], c=alpha_after_pert, cmap=cmap_sensory_prediction, 
                    vmin=0, vmax=1, s=s) #ec='k'
        
        if flg_plot_bars:
            ax1.annotate('', xy=(0, -0.3), xycoords='axes fraction', xytext=(1, -0.3), 
                     arrowprops=dict(arrowstyle="wedge", color='#525252'))
            
            ax1.annotate('', xy=(1, -0.4), xycoords='axes fraction', xytext=(0, -0.4), 
                     arrowprops=dict(arrowstyle="wedge", color='#525252')) 
        
            ax1.text(0, -0.6, 'SOM', fontsize=fs, transform=ax1.transAxes, horizontalalignment='left')
            ax1.text(1, -0.2, 'VIP', fontsize=fs, transform=ax1.transAxes, horizontalalignment='right')
        
        ax1.set_xticks([])
        ax1.set_ylabel('Sensory weight', fontsize=fs)
        ax1.set_ylim([0,1])
        
        # VIP - PV
        file_for_data = '../results/data/neuromod/data_weighting_neuromod_VIP-PV_' + mfn_flag + identifier + '.pickle'
        with open(file_for_data,'rb') as f:
            [xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f) 
    
        ax2.axhline(alpha_before_pert, ls='-', color=cmap_sensory_prediction(alpha_before_pert), zorder=0, lw=lw)
        #ax2.plot(xp, alpha_after_pert, ':', color='k', zorder=0)
        ax2.scatter(xp, alpha_after_pert, marker=markers[i], c=alpha_after_pert, cmap=cmap_sensory_prediction, 
                    vmin=0, vmax=1, s=s) #ec='k',
        
        if flg_plot_bars:
            ax2.annotate('', xy=(0, -0.3), xycoords='axes fraction', xytext=(1, -0.3), 
                     arrowprops=dict(arrowstyle="wedge", color='#525252'))
            
            ax2.annotate('', xy=(1, -0.4), xycoords='axes fraction', xytext=(0, -0.4), 
                     arrowprops=dict(arrowstyle="wedge", color='#525252')) 
        
            ax2.text(0, -0.6, 'VIP', fontsize=fs, transform=ax2.transAxes, horizontalalignment='left')
            ax2.text(1, -0.2, 'PV', fontsize=fs, transform=ax2.transAxes, horizontalalignment='right')
        
        ax2.set_xticks([]) #ax2.set_xticks([0,0.5,1])
        if flg_plot_xlabel:
            ax2.set_xlabel('Fraction of interneurons activated', fontsize=fs)
        ax2.set_ylim([0,1])
        
        # PV - SOM
        file_for_data = '../results/data/neuromod/data_weighting_neuromod_PV-SOM_' + mfn_flag + identifier + '.pickle'
        with open(file_for_data,'rb') as f:
            [xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f) 
    
        ax3.axhline(alpha_before_pert, ls='-', color=cmap_sensory_prediction(alpha_before_pert), zorder=0, lw=lw)
       # ax3.plot(xs, alpha_after_pert, ':', color='k', zorder=0)
        ax3.scatter(xs, alpha_after_pert, marker=markers[i], c=alpha_after_pert, cmap=cmap_sensory_prediction, 
                    vmin=0, vmax=1, s=s) # ec='k', 
        
        if flg_plot_bars:
            ax3.annotate('', xy=(0, -0.3), xycoords='axes fraction', xytext=(1, -0.3), 
                     arrowprops=dict(arrowstyle="wedge", color='#525252'))
            
            ax3.annotate('', xy=(1, -0.4), xycoords='axes fraction', xytext=(0, -0.4), 
                     arrowprops=dict(arrowstyle="wedge", color='#525252')) 
        
            ax3.text(0, -0.6, 'PV', fontsize=fs, transform=ax3.transAxes, horizontalalignment='left')
            ax3.text(1, -0.2, 'SOM', fontsize=fs, transform=ax3.transAxes, horizontalalignment='right')
        
        ax3.set_xticks([]) #ax3.set_xticks([0,0.5,1])
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


def plot_contraction_bias_illustration(ax1 = None, ax2 = None, fs = 6, lw=1, ms=1):
    
    if ax1 is None:
        _, ax1 = plt.subplots(1,1, tight_layout=True)
        
    if ax2 is None:
        _, ax2 = plt.subplots(1,1, tight_layout=True)
        
    x = np.linspace(10,30,11)
    m = 0.8
    n = (1 - m) * np.mean(x)
    y = m * x + n
    
    num_samples = 20
    z = np.zeros((len(x), num_samples))
    
    for i, xi in enumerate(x):
        z[i,:] = np.random.normal(y[i], 1, size=num_samples)
    
    ax1.plot(x,z, 'o', color = color_stimuli_background, ms=ms)
    ax1.plot(x, y, color = color_running_average_stimuli, lw=lw, label=m)
    ax1.axline((20,20), slope=1, color='k', ls=':', lw=lw)
    
    ax1.tick_params(size=2.0) 
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    
    leg = ax1.legend(loc=0, title='Slope', fontsize=fs, frameon=False, handlelength=1)
    plt.setp(leg.get_title(),fontsize=fs)
    ax1.set_xlabel('True input', fontsize=fs)
    ax1.set_ylabel('Estimated input', fontsize=fs)
    
    X = np.tile(x, (num_samples, 1)).T
    ax2.plot(X, z-X, 'o', color = color_stimuli_background, ms=ms)
    ax2.plot(x, (y-x), color = color_running_average_stimuli, lw=lw)
    ax2.axhline(0, ls=':', color='k', lw=lw)
    
    ax2.annotate("", xy=(10, 0), xycoords='data', xytext=(10, 2), textcoords='data',
                 arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", lw=lw, ec=color_m_neuron)) # "arc3, rad=0.3"
    ax2.annotate("", xy=(30, 0), xycoords='data', xytext=(30, -2), textcoords='data',
                 arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", lw=lw, ec=color_m_neuron)) # "arc3, rad=0.3"
    ax2.text(10,-3,'contraction \nbias', fontsize=fs, color=color_m_neuron)
    
    ax2.tick_params(size=2.0) 
    ax2.tick_params(axis='both', labelsize=fs)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    
    ax2.set_xlabel('True input', fontsize = fs)
    ax2.set_ylabel('Estimated - True', fontsize=fs)
    
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    


def plot_changes_upon_input2PE_neurons(std_mean = 1, n_std = 1, mfn_flag = '10', fs=6, lw=1, ms=4,
                                       alpha = 0.2, ax1 = None, ax2 = None, ax3 = None):
    
    color_columns_mean = [color_m_neuron, lighten_color(color_m_neuron, amount=0.7)]
    color_columns_variance = [color_v_neuron, lighten_color(color_v_neuron, amount=0.7)]
    
    if ax1 is None:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, sharey=True)
    # axs[1,0].plot(np.nan, np.nan, color='k')
    # axs[1,0].plot(np.nan, np.nan, color='k', ls='--')
    
    ### results lower column
    column = 1
    identifier = '_column_' + str(column) + '_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
    file_for_data = '../results/data/neuromod/data_moments_vs_PE_neurons_' + mfn_flag + identifier + '.pickle'
    
    with open(file_for_data,'rb') as f:
        [pert_strength, m_act_lower, v_act_lower, v_act_higher] = pickle.load(f)
        
    ax1.plot(pert_strength, m_act_lower[:,0], color=color_m_neuron, marker=11, ls='None', lw=lw, ms=ms)
    ax2.plot(pert_strength, v_act_lower[:,0], color=color_v_neuron, marker=11, ls='None', lw=lw, ms=ms)
    ax3.plot(pert_strength, v_act_higher[:,0], color=color_v_neuron, marker=11, ls='None', lw=lw, ms=ms)
    
    ax1.plot(pert_strength, m_act_lower[:,0], color='k', lw=lw, ms=ms, zorder=0, alpha=alpha)
    ax2.plot(pert_strength, v_act_lower[:,0], color='k', lw=lw, ms=ms, zorder=0, alpha=alpha)
    ax3.plot(pert_strength, v_act_higher[:,0], color='k', lw=lw, ms=ms, zorder=0, alpha=alpha)
    
    ax1.plot(pert_strength, m_act_lower[:,1], color=color_m_neuron, marker=10, ls='None', lw=lw, ms=ms)
    ax2.plot(pert_strength, v_act_lower[:,1], color=color_v_neuron, marker=10, ls='None', lw=lw, ms=ms)
    ax3.plot(pert_strength, v_act_higher[:,1], color=color_v_neuron, marker=10, ls='None', lw=lw, ms=ms)
    
    ax1.plot(pert_strength, m_act_lower[:,1], color='k', lw=lw, ms=ms, zorder=0, alpha=alpha)
    ax2.plot(pert_strength, v_act_lower[:,1], color='k', lw=lw, ms=ms, zorder=0, alpha=alpha)
    ax3.plot(pert_strength, v_act_higher[:,1], color='k', lw=lw, ms=ms, zorder=0, alpha=alpha)
    
    ### results higher column
    column = 2
    identifier = '_column_' + str(column) + '_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
    file_for_data = '../results/data/neuromod/data_moments_vs_PE_neurons_' + mfn_flag + identifier + '.pickle'
    
    with open(file_for_data,'rb') as f:
        [pert_strength, m_act_lower, v_act_lower, v_act_higher] = pickle.load(f)
        
    ax1.plot(pert_strength, m_act_lower[:,0], color=color_m_neuron_light, marker=11, ls='None', lw=lw, ms=ms)
    ax2.plot(pert_strength, v_act_lower[:,0], color=color_v_neuron_light, marker=11, ls='None', lw=lw, ms=ms)
    ax3.plot(pert_strength, v_act_higher[:,0], color=color_v_neuron_light, marker=11, ls='None', lw=lw, ms=ms)
    
    ax1.plot(pert_strength, m_act_lower[:,0], color='k', lw=lw, ms=ms, zorder=0, alpha=alpha)
    ax2.plot(pert_strength, v_act_lower[:,0], color='k', lw=lw, ms=ms, zorder=0, alpha=alpha)
    ax3.plot(pert_strength, v_act_higher[:,0], color='k', lw=lw, ms=ms, zorder=0, alpha=alpha)
    
    ax1.plot(pert_strength, m_act_lower[:,1], color=color_m_neuron_light, marker=10, ls='None', lw=lw, ms=ms)
    ax2.plot(pert_strength, v_act_lower[:,1], color=color_v_neuron_light, marker=10, ls='None', lw=lw, ms=ms)
    ax3.plot(pert_strength, v_act_higher[:,1], color=color_v_neuron_light, marker=10, ls='None', lw=lw, ms=ms)

    ax1.plot(pert_strength, m_act_lower[:,1], color='k', lw=lw, ms=ms, zorder=0, alpha=alpha)
    ax2.plot(pert_strength, v_act_lower[:,1], color='k', lw=lw, ms=ms, zorder=0, alpha=alpha)
    ax3.plot(pert_strength, v_act_higher[:,1], color='k', lw=lw, ms=ms, zorder=0, alpha=alpha)

    ax1.tick_params(size=2.0) 
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    
    ax2.tick_params(size=2.0) 
    ax2.tick_params(axis='both', labelsize=fs)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    
    ax3.tick_params(size=2.0) 
    ax3.tick_params(axis='both', labelsize=fs)
    ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
    
    ax1.set_ylabel('M neuron', fontsize=fs)
    ax2.set_ylabel('V neuron (lower)', fontsize=fs, labelpad=10)
    ax3.set_ylabel('V neuron (higher)', fontsize=fs)
    ax3.set_xlabel('Perturbation strength', fontsize=fs)
    
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    sns.despine(ax=ax3)
            

def plot_illustration_changes_upon_baseline_PE(BL = np.linspace(0,5,11), mean = 10, sd = 3, factor_for_visibility = 20,
                                               alpha=0.2, lw=1, ms=4, fs=6, ax1 = None, ax2 = None, ax3 = None):
    
    if ax1 is None:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, sharey=True)#, tight_layout=True)
    
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
    
    V_lower_nPE = (b-a)**2/(3*(1-gains)**3) * (gains*(1 - np.sqrt(gains))**3 - (gains - np.sqrt(gains))**3)
    V_lower_nPE[np.isnan(V_lower_nPE)] = (b-a)**2/12
    V_lower_pPE = (b-a)**2/(3*(gains-1)**3) * ((gains - np.sqrt(gains))**3 - gains * (1 - np.sqrt(gains))**3)
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



def plot_illustration_changes_upon_baseline_PE_old(BL = np.linspace(0,5,11), mean = 10,
                                               sd_tested = np.linspace(1,3,2), factor_for_visibility = 1):
    
    f, axs = plt.subplots(2, 2, sharex=True, sharey='row')#, tight_layout=True)
    colors_sd_mean = [lighten_color(color_m_neuron, amount=0.7), color_m_neuron]
    colors_sd_variance = [lighten_color(color_v_neuron, amount=0.7), color_v_neuron]
        
    labels_mean = ['low SD', 'high SD']
    
    # BL of pPE neurons, BL of nPE fixed at zero
    p_BL = BL
    axs[1,0].plot(np.nan, np.nan, color='k')
    axs[1,0].plot(np.nan, np.nan, color='k', ls='--')
    
    for i, sd in enumerate(sd_tested):
        
        ### prediction
        b = np.sqrt(12) * sd / 2 + mean
        a = 2 * mean - b
        
        P = (b + a)/2 + p_BL/(b - a)
        axs[0,0].plot(p_BL, P, color=colors_sd_mean[i], label = labels_mean[i])
        axs[0,0].plot(0, mean, 'o', color=colors_sd_mean[i])
        
        ### variance (in same column)
        V_same = (b-a)**2/12 + p_BL**2/(b-a)**2 * (1 + 2*p_BL/(b-a)) + p_BL * (p_BL + (b-a)/2)
        axs[1,0].plot(p_BL, V_same, color=colors_sd_variance[i])
        axs[1,0].plot(0, (b-a)**2/12, 'o', color=colors_sd_variance[i])
        
        ### variance of higher PE circuit when PE neurons altered in lower PE circuit
        V_mix = (b-a)**2 / 12 + factor_for_visibility * (P - P[0])**2
        axs[1,0].plot(p_BL, V_mix, ls='--', color=colors_sd_variance[i])
        
        
    # BL of nPE neurons, BL of pPE fixed at zero
    n_BL = BL
    
    for i, sd in enumerate(sd_tested):
        
        ### prediction
        b = np.sqrt(12) * sd / 2 + mean
        a = 2 * mean - b
        
        P = (b + a)/2 - n_BL/(b - a)
        axs[0,1].plot(n_BL, P, color=colors_sd_mean[i])
        axs[0,1].plot(0, mean, 'o', color=colors_sd_mean[i])
        
        ### variance (in same column)
        V_same = (b-a)**2/12 + n_BL**2/(b-a)**2 * (1 + 2*n_BL/(b-a)) + n_BL * (n_BL + (b-a)/2)
        axs[1,1].plot(n_BL, V_same, color=colors_sd_variance[i])
        axs[1,1].plot(0, (b-a)**2/12, 'o', color=colors_sd_variance[i])
        
        ### variance of higher PE circuit when PE neurons altered in lower PE circuit
        V_mix = (b-a)**2 / 12 + factor_for_visibility * (P - P[0])**2
        axs[1,1].plot(p_BL, V_mix, ls='--', color=colors_sd_variance[i])
        
    axs[0,0].set_ylabel('Activity M neuron', labelpad=10)
    axs[1,0].set_ylabel('Activity V neuron', labelpad=10)
    
    axs[1,0].set_xlabel('BL of pPE neuron', labelpad=10)
    axs[1,1].set_xlabel('BL of nPE neuron', labelpad=10)
    
    # ymin = axs[0,1].get_ylim()[0]
    # ymax = axs[0,0].get_ylim()[1]
    # axs[0,0].set_ylim([ymin, ymax])
    # axs[0,1].set_ylim([ymin, ymax])
    
    axs[0,0].legend(loc=0, frameon=False, handlelength=1)
    axs[1,0].legend(['same column', 'lower/upper col.'], frameon=False, handlelength=1.5)
    
    for i in range(2):
        for j in range(2):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            sns.despine(ax=axs[i,j])
            
            
def plot_illustration_changes_upon_gain_PE_old(gains = np.linspace(0.5, 1.5, 11), mean = 10, 
                                           sd_tested = np.linspace(1,3,2), factor_for_visibility = 5):
    
    f, axs = plt.subplots(2, 2, sharex=True, sharey='row')#, tight_layout=True)
    colors_sd_mean = [lighten_color(color_m_neuron, amount=0.7), color_m_neuron]
    colors_sd_variance = [lighten_color(color_v_neuron, amount=0.7), color_v_neuron]
    labels_mean = ['low SD', 'high SD']
    
    # BL of pPE neurons, BL of nPE fixed at zero
    m = gains
    axs[1,0].plot(np.nan, np.nan, color='k')
    axs[1,0].plot(np.nan, np.nan, color='k', ls='--')
    
    for i, sd in enumerate(sd_tested):
        
        ### prediction
        b = np.sqrt(12) * sd / 2 + mean
        a = 2 * mean - b
        
        P = (m*b - a + np.sqrt(m) * (a-b)) / (m-1)
        P[np.isnan(P)] = (b+a)/2
        axs[0,0].plot(m, P, color=colors_sd_mean[i], label = labels_mean[i])
        axs[0,0].plot(1, mean, 'o', color=colors_sd_mean[i])
        
        ### variance (in same column)
        V_same = (b-a)**2/(3*(m-1)**3) * ((m - np.sqrt(m))**3 - m * (1 - np.sqrt(m))**3)
        V_same[np.isnan(V_same)] = (b-a)**2/12
        axs[1,0].plot(m, V_same, color=colors_sd_variance[i])
        axs[1,0].plot(1, (b-a)**2/12, 'o', color=colors_sd_variance[i])
        
        ### variance of higher PE circuit when PE neurons altered in lower PE circuit
        V_mix = (b-a)**2 / 12 + factor_for_visibility * (P - P[m==1])**2
        axs[1,0].plot(m, V_mix, ls='--', color=colors_sd_variance[i])
        
        
    # BL of nPE neurons, BL of pPE fixed at zero
    n = gains
    
    for i, sd in enumerate(sd_tested):
        
        ### prediction
        b = np.sqrt(12) * sd / 2 + mean
        a = 2 * mean - b
        
        P = (b - n*a + np.sqrt(n) * (a-b)) / (1-n)
        P[np.isnan(P)] = (b+a)/2
        axs[0,1].plot(n, P, color=colors_sd_mean[i])
        axs[0,1].plot(1, mean, 'o', color=colors_sd_mean[i])
        
        ### variance (in same column)
        V_same = (b-a)**2/(3*(1-n)**3) * (n*(1 - np.sqrt(n))**3 - (n - np.sqrt(n))**3)
        V_same[np.isnan(V_same)] = (b-a)**2/12
        axs[1,1].plot(n, V_same, color=colors_sd_variance[i])
        axs[1,1].plot(1, (b-a)**2/12, 'o', color=colors_sd_variance[i])
        
        ### variance of higher PE circuit when PE neurons altered in lower PE circuit
        V_mix = (b-a)**2 / 12 + factor_for_visibility * (P - P[n==1])**2
        axs[1,1].plot(n, V_mix, ls='--', color=colors_sd_variance[i])
        
    axs[0,0].set_ylabel('Activity M neuron', labelpad=10)
    axs[1,0].set_ylabel('Activity V neuron', labelpad=10)
    
    axs[1,0].set_xlabel('gain of pPE neuron', labelpad=10)
    axs[1,1].set_xlabel('gain of nPE neuron', labelpad=10)
    
    # ymin = axs[0,1].get_ylim()[0]
    # ymax = axs[0,0].get_ylim()[1]
    # axs[0,0].set_ylim([ymin, ymax])
    # axs[0,1].set_ylim([ymin, ymax])
    
    axs[0,0].legend(loc=0, frameon=False, handlelength=1)
    axs[1,0].legend(['same column', 'lower/upper'], frameon=False, handlelength=1.5)
    
    for i in range(2):
        for j in range(2):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            sns.despine(ax=axs[i,j])
    

def plot_bar_neuromod_stacked(column, mfn_flag, figsize=(4,3), fs=7, lw=1, std_means=[1, 0], n_std_all = [0, 1], dgrad = 0.001, axs=None):
    
    if axs is None:
        f, ax = plt.subplots(1,1, tight_layout=True, figsize=figsize)
    else:
        ax = axs
    
    for condition in range(2):
        
        std_mean = std_means[condition]
        n_std = n_std_all[condition]

        ### filename for data
        identifier = '_column_' + str(column) + '_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
        file_for_data = '../results/data/neuromod/data_weighting_neuromod_' + mfn_flag + identifier + '.pickle'
        
        ### load data
        with open(file_for_data,'rb') as f:
            [xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f)
         
        x = np.concatenate((xv[xp==0][::-1][:-1], xv[xs==0])) 
        sens_weight = np.concatenate((alpha_after_pert[xp==0][::-1][:-1], alpha_after_pert[xs==0]))
        
        for nom in range(len(x)):
            
            if condition==0:
                z = [nom+1-0.3,nom+0.95]
            else:
                z = [nom+1.05,nom + 1 + 0.3]
            
            num_lines = np.int32(abs(alpha_before_pert - sens_weight[nom]) / dgrad)
            gradient_lines = np.linspace(alpha_before_pert, sens_weight[nom], num_lines)
            
            for gline in gradient_lines:
                ax.plot(z, [gline, gline], color=cmap_sensory_prediction(gline))
        
        ax.axhline(alpha_before_pert, ls=':', color=cmap_sensory_prediction(alpha_before_pert))
        ax.set_ylim([0,1])
        ax.set_xlim([0,len(x)+1])
        
        ax.set_xticks([1,11,21])
        ax.set_xticklabels(['SOM only','VIP only', 'PV only'], fontsize=fs)
        
        ax.tick_params(size=2.0) 
        ax.tick_params(axis='both', labelsize=fs)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        
        if column==0:
            ax.set_title('Global neuromodulation', fontsize=fs)
        elif column==1:
            ax.set_title('Neuromodulation in lower PE circuit', fontsize=fs)
        elif column==2:
            ax.set_title('Neuromodulation in higher PE circuit', fontsize=fs)
        
        sns.despine(ax=ax)
        
        
def plot_neurmod_results(input_condition, figsize=(10,3), fs=7, lw=1, std_means=[1, 0], n_std_all = [0, 1], 
                         dgrad = 0.001, column = 0, ax = None):
    
    if ax is None:
        f, (ax1, ax2, ax3) = plt.subplots(1,3, tight_layout=True, figsize=figsize)
    
    # define all MFN to be plotted
    mfn_flags = ['10','01','11']

    # define input case
    std_mean = std_means[input_condition]
    n_std = n_std_all[input_condition]
    
    colors = ['r','b','g']
    markers = ['o','s','d']

    for i, mfn_flag in enumerate(mfn_flags):
    
        ### filename for data
        identifier = '_column_' + str(column) + '_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
        file_for_data = '../results/data/neuromod/data_weighting_neuromod_' + mfn_flag + identifier + '.pickle'
        
        ### load data
        with open(file_for_data,'rb') as f:
            [xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f)
         
        x = np.concatenate((xv[xp==0][::-1][:-1], xv[xs==0])) 
        sens_weight = np.concatenate((alpha_after_pert[xp==0][::-1][:-1], alpha_after_pert[xs==0]))
        
        ### 
        ax1.plot(xv[xp==0], alpha_after_pert[xp==0], color=colors[i], marker=markers[i])
        ax1.axhline(alpha_before_pert, ls=':', color=colors[i]) #color=cmap_sensory_prediction(alpha_before_pert))
        
        ax2.plot(xv[xs==0], alpha_after_pert[xs==0], color=colors[i], marker=markers[i])
        ax2.axhline(alpha_before_pert, ls=':', color=colors[i])
        
        # ids_columns_per_row = np.nanargmin(xv,0)
        # x = np.nanmin(xv,0)
        # sens_weight = 
        # ax3.plot(xv[xs==0], alpha_after_pert[xs==0], color=colors[i], marker=markers[i])
        # ax3.axhline(alpha_before_pert, ls=':', color=colors[i])
        
        
        # ax.set_ylim([0,1])
        # ax.set_xlim([0,len(x)+1])
        
        # ax.set_xticks([1,11,21])
        # ax.set_xticklabels(['SOM only','VIP only', 'PV only'], fontsize=fs)
        
        # ax.tick_params(size=2.0) 
        # ax.tick_params(axis='both', labelsize=fs)
        # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        
        # if column==0:
        #     ax.set_title('Global neuromodulation', fontsize=fs)
        # elif column==1:
        #     ax.set_title('Neuromodulation in lower PE circuit', fontsize=fs)
        # elif column==2:
        #     ax.set_title('Neuromodulation in higher PE circuit', fontsize=fs)
        
        sns.despine(ax=ax)
    

def plot_bar_neuromod(column, mfn_flag, flag_what, figsize=(4,3), fs=7, lw=1, std_means=[1, 0], n_std_all = [0, 1], dgrad = 0.001, axs=None):
    
    if axs is None:
        f, ax = plt.subplots(1,1, tight_layout=True, figsize=figsize)
    else:
        ax = axs
    
    for condition in range(2):
        
        std_mean = std_means[condition]
        n_std = n_std_all[condition]

        ### filename for data
        identifier = '_column_' + str(column) + '_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
        file_for_data = '../results/data/neuromod/data_weighting_neuromod_' + mfn_flag + identifier + '.pickle'
        
        ### load data
        with open(file_for_data,'rb') as f:
            [xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f)
         
        if flag_what==0:
            x = xv[xs==0]
            sens_weight = alpha_after_pert[xs==0]
        elif flag_what==1:
            x = xv[xp==0]
            sens_weight = alpha_after_pert[xp==0]
        
        #ax = plt.gca()
        
        for nom in range(len(x)):
            
            if condition==0:
                z = [nom+1-0.3,nom+0.95]
            else:
                z = [nom+1.05,nom + 1 + 0.3]
            
            num_lines = np.int32(abs(alpha_before_pert - sens_weight[nom]) / dgrad)
            gradient_lines = np.linspace(alpha_before_pert, sens_weight[nom], num_lines)
            
            for gline in gradient_lines:
                ax.plot(z, [gline, gline], color=cmap_sensory_prediction(gline))
        
        ax.axhline(alpha_before_pert, ls=':', color=cmap_sensory_prediction(alpha_before_pert))
        ax.set_ylim([0,1])
        ax.set_xlim([0,len(x)+1])
        
        ax.set_xticks([1,11])
        if flag_what==0:
            ax.set_xticklabels(['VIP only', 'PV only'], fontsize=fs)
        else:
            ax.set_xticklabels(['VIP only', 'SOM only'], fontsize=fs)
        
        ax.tick_params(size=2.0) 
        ax.tick_params(axis='both', labelsize=fs)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        
        if column==0:
            ax.set_title('Global neuromodulation', fontsize=fs)
        elif column==1:
            ax.set_title('Neuromodulation in lower PE circuit', fontsize=fs)
        elif column==2:
            ax.set_title('Neuromodulation in higher PE circuit', fontsize=fs)
        
        sns.despine(ax=ax)


def plot_slope_variability(sd_1, sd_2, fitted_slopes_1, fitted_slopes_2, label_text, figsize=(3,2.5), fs=7, lw=1, ms=3, ax = None):
    
    if ax is None:
        f, ax = plt.subplots(1,1, tight_layout=True, figsize=(3,2.5))    
    ax.plot(sd_1, fitted_slopes_1, '.-', color='k', label=label_text[0], lw=lw, ms=ms) 
    ax.plot(sd_2, fitted_slopes_2, '.-', color='k', alpha=0.3, label=label_text[1], lw=lw, ms=ms) 
    
    leg = ax.legend(loc=1, handlelength=1, fontsize=fs, frameon=False)
    #leg.set_title('Variability',prop={'size':fs})
    
    ax.set_xlabel('Standard deviation', fontsize=fs)
    ax.set_ylabel('Slope', fontsize=fs)
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    
    sns.despine(ax=ax)
    

def plot_slope_trail_duration(trial_durations,fitted_slopes_1, fitted_slopes_2, label_text, figsize=(3,2.5), fs=7, lw=1, ax = None):
    
    if ax is None:
        f, ax = plt.subplots(1,1, tight_layout=True, figsize=(3,2.5))    
    ax.plot(trial_durations / 5000, fitted_slopes_1, lw=lw, ls='-', color='k', label=label_text[0]) # color='k'
    ax.plot(trial_durations / 5000, fitted_slopes_2, lw=lw, ls='-', color='k', alpha=0.3, label=label_text[1]) # color='k', alpha = 0.5
    ax.axhline(1, color='k', ls=':', lw=lw)
    
    ax.set_xlabel(r'trial duration / T$_0$', fontsize=fs)
    ax.set_ylabel('Slope', fontsize=fs)
    leg = ax.legend(loc=0, handlelength=1, fontsize=fs, frameon=False)
    #leg.set_title('Zero variance',prop={'size':fs})
    
    ax.set_ylim([0.7,1.02])
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    sns.despine(ax=ax)


def plot_example_contraction_bias(weighted_output, stimuli, n_trials, figsize=(2.5,2.8), ms=1,
                                  fs=7, lw=1, num_trial_ss=np.int32(50), trial_means=None,
                                  min_means = None, max_means = None, m_std = None, n_std = None, 
                                  show_marker_inset=False, ax = None, plot_ylabel=True):
    
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
        ax.plot(trials_sensory, (trials_estimated - trials_sensory), 'o', alpha = 1, color=color_stimuli_background)
        ax.axline((np.mean(stimuli), 0), slope=0, color='k', ls=':')
        
        if min_means is not None:
            axin = ax.inset_axes((0.8,0.15,.2,.2)) #axin = inset_axes(ax, width="20%", height="20%", loc=4, borderpad=2)
            x = np.linspace(min_means, max_means, 10)
            axin.plot(x, np.maximum(m_std * x + n_std,0))
        
    else:
    
        if ax is None:
            f, ax = plt.subplots(1,1, tight_layout=True, figsize=figsize) 
        #colors = sns.color_palette("Set2", n_colors = weighted_output.ndim)
        colors = [color_running_average_stimuli, '#6A2E35']
        colors_lighter = [color_stimuli_background, '#D9ABB0']
        
        if min_means is not None:
            axin = ax.inset_axes((0.8,0.8,.2,.2))
        
        for i in range(np.size(weighted_output,0)):
            
            if trial_means is not None:
                trials_sensory_all = trial_means[i,:]
            else:
                trials_sensory_all = np.mean(np.split(stimuli[i,:], n_trials),1)
            trials_estimated_all = np.mean(np.split(weighted_output[i,:], n_trials),1)
            
            # to take out transient ...
            trials_sensory = trials_sensory_all[num_trial_ss:]
            trials_estimated = trials_estimated_all[num_trial_ss:]
        
            ax.plot(trials_sensory, trials_estimated - trials_sensory, '.', color = colors_lighter[i], markersize=ms, zorder=0)#, color=color_stimuli_background)
            
            # plot line through data
            p =  np.polyfit(trials_sensory, (trials_estimated - trials_sensory), 1)
            x = np.linspace(min_means[i], max_means[i], 10)
            ax.plot(x, np.polyval(p, x), '-', color = colors[i], label=str(round(p[0]+1,2)), lw=lw)
            
            xmin, xmax = ax.get_xlim()[0], ax.get_xlim()[1]
            x = np.linspace(xmin, xmax, 10)
            #ax.plot(trials_sensory, np.polyval(p, trials_sensory), color = lighten_color(colors[i],amount=2), marker='x')
            ax.plot(x, np.polyval(p, x), '--', color = colors[i], lw=lw)
            
            if min_means is not None:
                x = np.linspace(min_means[i], max_means[i], 10)
                if show_marker_inset:
                    axin.plot(x, np.maximum(m_std[i] * x + n_std[i],0), color=colors[i], ls='-', marker='o', ms=ms-1)
                else:
                    axin.plot(x, np.maximum(m_std[i] * x + n_std[i],0), color=colors[i], ls='-', lw=lw)
                    
        ax.axline((np.mean(stimuli), 0), slope=0, color='k', ls=':', zorder=0, lw=lw)
        # leg = ax.legend(loc=0, fontsize=fs, frameon=False, handlelength=1)
        # leg.set_title('Slope',prop={'size':fs})
    
    if min_means is not None:
        # axin.tick_params(size=1.0)
        axin.set_xlabel('mean', fontsize=fs, labelpad=3)
        axin.set_ylabel('std', fontsize=fs, labelpad=3)
        # axin.tick_params(axis='both', labelsize=fs)
        # axin.xaxis.set_major_locator(plt.MaxNLocator(2))
        # axin.yaxis.set_major_locator(plt.MaxNLocator(2))
        axin.set_xticks([])
        axin.set_yticks([])
        
        x_lims = axin.get_xlim()
        if (x_lims[1]-x_lims[0])<2:
            axin.set_xlim([np.mean(x_lims)-1, np.mean(x_lims)+1])
            
        axin.set_ylim([axin.get_ylim()[0]-0.5, axin.get_ylim()[1]+0.5])
        sns.despine(ax=axin)
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel('Input / trial', fontsize=fs)
    if plot_ylabel:
        ax.set_ylabel('Output - Input / trial', fontsize=fs)
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
    
    
def plot_points_of_interest_neuromod(mfn_flag, columns, std_means, n_std_all, figsize=(8,3), fs=7,
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
            file_for_data = '../results/data/neuromod/data_weighting_neuromod_' + mfn_flag + identifier + '.pickle'
            
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
    ax.set_facecolor('w') # #DBDBDB
    
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


def plot_neuromod_per_net(mfn_flag, columns, std_means, n_std_all, figsize=(8,6), fs=7):
    
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
            file_for_data = '../results/data/neuromod/data_weighting_neuromod_' + mfn_flag + identifier + '.pickle'
            
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
    #colors = sns.color_palette('Blues_d', n_colors=11)
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
    
    

def plot_impact_para(variability_across, weight_ctrl, weight_act, para_range_tested = [], fs=6, 
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
                
                ax.legend(loc=0, frameon=False, fontsize=fs)
            
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
                ax.legend(loc=0, frameon=False, fontsize=fs, ncol=1)
                
                
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
        
        
        
def plot_weight_over_trial(weight_ctrl, weight_mod, n_trials, fs=6, lw=1, leg_text=None, marker_every_n = 500, ax=None): #id_stims = [0,4]
    
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
    
    ax.plot(trial_fraction, alpha_avg_trial, color='k', zorder=0, lw=lw)#colors[id_stim])
    ax.scatter(trial_fraction[::marker_every_n], alpha_avg_trial[::marker_every_n], marker='o', s = s, linewidth=lw, 
               c=alpha_avg_trial[::marker_every_n], cmap=cmap_sensory_prediction, vmin=0, vmax=1, edgecolors='k')
    #ax.plot(trial_fraction[::marker_every_n], alpha_avg_trial[::marker_every_n], color='k')#colors[id_stim])
    ax.fill_between(trial_fraction, alpha_avg_trial-sem, alpha_avg_trial+sem, alpha=0.2, color='k', zorder=0) #colors[id_stim])

    ### modulated
    alpha_split = np.array_split(weight_mod[:,id_stim], n_trials)
    alpha_avg_trial = np.mean(alpha_split,0)
    alpha_std_trial = np.std(alpha_split,0)
    sem = alpha_std_trial/np.sqrt(n_trials)
    trial_fraction = np.linspace(0,1,len(alpha_avg_trial))
    
    marker_every_n //= 5
    ax.plot(trial_fraction, alpha_avg_trial, ls='-', color='k', zorder=0, lw=lw) #colors[id_stim])
    ax.scatter(trial_fraction[::marker_every_n], alpha_avg_trial[::marker_every_n], marker='s', s = s, linewidth=lw, 
               c=alpha_avg_trial[::marker_every_n], cmap=cmap_sensory_prediction, vmin=0, vmax=1, edgecolors='k')
    #ax.plot(trial_fraction[::(marker_every_n//5)], alpha_avg_trial[::(marker_every_n//5)], color='k') #colors[id_stim])
    ax.fill_between(trial_fraction, alpha_avg_trial-sem, alpha_avg_trial+sem, alpha=0.2, color='k', zorder=0) #colors[id_stim])

    ymin, ymax = ax.get_ylim()
   # ax.axhspan(ymin,0.5, color=color_m_neuron, alpha=0.07, zorder=0)
   # ax.axhspan(0.5, ymax, color=color_sensory, alpha=0.07, zorder=0)
    ax.set_ylim([ymin, ymax])
    
    #ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    
    if leg_text is not None:
        ax.plot(np.nan, np.nan, 'o', color='k', label=leg_text[0], ms=4)
        ax.plot(np.nan, np.nan, 's', color='k', label=leg_text[1], ms=4)
        ax.legend(loc=4, frameon=False, handlelength=2, fontsize=fs, borderpad=2)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', labelsize=fs)
    ax.tick_params(size=2.0) 
    ax.set_ylabel('Sensory weight', fontsize=fs)
    ax.set_xlabel('Time / trial duration', fontsize=fs)
    sns.despine(ax=ax)


def plot_transitions_examples(n_trials, trial_duration, stimuli, alpha, beta, weighted_output, 
                              time_plot = 0, ylim=None, xlim=None, plot_ylable=True, lw=1, 
                              figsize=(3.5,5), plot_only_weights=False, fs=6, transition_point=60, ax2=None):
    
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
    
    ax2.plot(time[time > time_plot * time[-1]], alpha[time > time_plot * time[-1]], color='k', lw=lw, label='stimulus')

    n_every = 20
    x = time[time > time_plot * time[-1]]
    y = alpha[time > time_plot * time[-1]]
    ax2.scatter(x[::n_every], y[::n_every], c=y[::n_every], cmap=cmap_sensory_prediction, vmin=0, vmax=1, marker='.', zorder=5, s=1)
    
    if plot_ylable:
        ax2.set_ylabel('Sensory weight', fontsize=fs)
    # else:
    #     ax2.set_ylabel('Weights', color='white', fontsize=fs)
    #     ax2.tick_params(axis='y', colors='white')
    ax2.axvline(transition_point, color='k', ls='--', lw=lw, zorder=10)
    ax2.set_xlabel('Time (#trials)', fontsize=fs)
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
                                figsize=(6,4.5), fs=6, square=False, ax=None):
    
    if ax is None:
        plt.figure(tight_layout=True, figsize=figsize)
        ax = plt.gca()
        
    index = np.round(decimal*para_tested_first)/decimal
    columns = np.round(decimal*para_tested_second)/decimal
    
    data = pd.DataFrame(fraction_sensory_median, columns=columns, index=index)
    sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=cmap, xticklabels=every_n_ticks, square=square,
                yticklabels=every_n_ticks, cbar_kws={'label': 'Sensory weight', 'ticks':[0, 0.5, 1]}, ax=ax)
    
    ax.locator_params(nbins=2)
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.tick_params(size=2.0)
    cbar.ax.yaxis.label.set_size(fs)
    
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    #ax.set_title(title, fontsize=fs)


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
             color=color_stimuli_background, lw=lw, marker='|', ls="None") # , label='stimuli'
    ax1.plot(time[time > time_plot * time[-1]], weighted_output[time > time_plot * time[-1]], 
             color=color_weighted_output, lw=lw, label='weighted \noutput')  
    ax1.set_ylabel('Activity (1/s)', fontsize=fs)
    ax1.set_xlabel('Time (#trials)', fontsize=fs)
    if plot_legend:
        #ax1.legend(loc=0, frameon=False, fontsize=fs, handlelength=1)
        ax1.text(time_plot * time[-1] + 1, 0.85 * np.max(weighted_output[time > time_plot * time[-1]]), 
                 'weighted \noutput', color=color_weighted_output, fontsize=fs)
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
        ax2.set_xlabel('Time (#trials)')
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
             color='k', lw=lw)#, label='feedforward')
    
    n_every = 10
    x = time[time > time_plot * time[-1]]
    y = alpha[time > time_plot * time[-1]]
    ax3.scatter(x[::n_every], y[::n_every], c=y[::n_every], cmap=cmap_sensory_prediction, vmin=0, vmax=1, marker='.', zorder=5, s=1)
    
    # ax3.plot(time[time > time_plot * time[-1]], beta[time > time_plot * time[-1]], 
    #          color=color_m_neuron, lw=lw, label='feedback')
    ax3.set_ylabel('Sensory weights', fontsize=fs)
    ax3.set_xlabel('Time (#trials)', fontsize=fs)
    # if plot_legend:
    #     ax3.legend(loc=0, frameon=False, fontsize=fs)
    ax3.set_xlim([time_plot * time[-1],time[-1]])
    ax3.set_ylim([0,1])
    ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.tick_params(axis='both', labelsize=fs)
    ax3.tick_params(size=2.0) 
    sns.despine(ax=ax3)
    
    

def plot_mse_test_distributions(mse, dist_types=None, mean=None, std=None, SEM=None, title=None,
                                plot_dists=False, x_lim = None, pa=0.8, inset_steady_state=False,
                                fig_size=(5,3), fs = 5, plot_ylabel = True, plot_xlabel = True, 
                                figsize=(5,5), ax = None):
    
    ### show mean squared error
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
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
        
        ax1 = ax.inset_axes([0.6,0.7,0.4,0.3])
        f = 0.5
        
        for i in range(num_rows):
            ax1.plot(time[time>time[-1]*f]/time[-1], mse[i,time>time[-1]*f], color=colors[i])
            if SEM is not None:
                ax1.fill_between(time[time>time[-1]*f]/time[-1], mse[i,time>time[-1]*f] - SEM[i,time>time[-1]*f], 
                                mse[i,time>time[-1]*f] + SEM[i,time>time[-1]*f], color=colors[i], alpha=0.3)
        
        ax1.tick_params(size=2)
        ax1.locator_params(nbins=3)
        ax1.tick_params(axis='both', labelsize=fs)
        sns.despine(ax=ax1)

    if dist_types is not None:
        ax.legend(loc=4, ncol=2, handlelength=1, frameon=False, fontsize=fs)
    
    ax.tick_params(size=2)
    ax.tick_params(axis='both', labelsize=fs)
    if plot_xlabel:
        ax.set_xlabel('Time / trial duration', fontsize=fs)
    if plot_ylabel:
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


def plot_interneuron_activity_heatmap(end_of_initial_phase, means_tested, variances_tested, activity_interneurons,
                                      vmax = None, lw = 1, fs=5, figsize=(3,3)):
    
    for i in range(4):
        int_activity = activity_interneurons[:,:,:,i]
        int_activity = np.mean(int_activity[:, :, end_of_initial_phase:],2)
        
        f = plt.subplots(1,1, figsize=figsize, tight_layout=True)
        
        data = pd.DataFrame(int_activity.T, index=variances_tested, columns=means_tested)
        ax = sns.heatmap(data, vmin=0, xticklabels=3, yticklabels=2)#, cmap=color_cbar
                      #cbar_kws={'label': r'normalised MSE$_\mathrm{\infty}$ (%)', 'ticks': [vmin, vmax]})
                      
        ax.invert_yaxis()
        
        ax.tick_params(axis='both', labelsize=fs)
        ax.set_xlabel('mean of stimuli', fontsize=fs)
        ax.set_ylabel('variance of stimuli', fontsize=fs)
        ax.figure.axes[-1].yaxis.label.set_size(fs)
        ax.figure.axes[-1].tick_params(labelsize=fs)
    
        sns.despine(ax=ax, bottom=False, top=False, right=False, left=False)
        
        
def plot_neuron_activity(end_of_initial_phase, means_tested, variances_tested, activity_interneurons,
                         activity_pe_neurons, id_fixed = 0, vmax = None, lw = 1, fs=6, figsize=(3,3),
                         ax1 = None, ax2 = None):
    
    if activity_interneurons is not None:
        temp_int_activity = np.mean(activity_interneurons[:, :, end_of_initial_phase:, :], 2)
    
        int_activity = np.zeros((temp_int_activity.shape[0], temp_int_activity.shape[1], 3))
        int_activity[:,:,0] = 0.5 *(temp_int_activity[:,:,0] + temp_int_activity[:,:,1])
        int_activity[:,:,1:] = temp_int_activity[:,:,2:]
        
        #colors_in = [Col_Rate_PVv, Col_Rate_PVm, Col_Rate_SOM, Col_Rate_VIP]
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
            
        ax1.text(means_tested[0], 0.8 * np.min(pe_activity[:,id_fixed,:]),'nPE', fontsize=fs, color=Col_Rate_nE)
        ax1.text(means_tested[0], 1.1 * np.max(pe_activity[:,id_fixed,:]),'pPE', fontsize=fs, color=Col_Rate_pE)
            
    
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
    
    
def plot_neuron_activity_lower_higher(variability_within, variability_across, activity_interneurons_lower_within,
                                      activity_interneurons_higher_within, activity_interneurons_lower_across, 
                                      activity_interneurons_higher_across, activity_pe_neurons_lower_within,
                                      activity_pe_neurons_higher_within, activity_pe_neurons_lower_across,
                                      activity_pe_neurons_higher_across, lw = 1, fs=6, figsize=(3,3), ax1 = None, ax2 = None, ax3 = None):
    
    # average PV's
    activity_int_lower_across = activity_interneurons_lower_across[:,:,:,1:]
    activity_int_higher_across = activity_interneurons_higher_across[:,:,:,1:]
    activity_int_lower_within = activity_interneurons_lower_within[:,:,:,1:]
    activity_int_higher_within = activity_interneurons_higher_within[:,:,:,1:]
    
    activity_int_lower_across[:,:,:,0] = 0.5 * (activity_interneurons_lower_across[:,:,:,0] + activity_interneurons_lower_across[:,:,:,1])
    activity_int_higher_across[:,:,:,0] = 0.5 * (activity_interneurons_higher_across[:,:,:,0] + activity_interneurons_higher_across[:,:,:,1])
    activity_int_lower_within[:,:,:,0] = 0.5 * (activity_interneurons_lower_within[:,:,:,0] + activity_interneurons_lower_within[:,:,:,1])
    activity_int_higher_within[:,:,:,0] = 0.5 * (activity_interneurons_higher_within[:,:,:,0] + activity_interneurons_higher_within[:,:,:,1])
    
    #color_INs = [Col_Rate_PVv, Col_Rate_PVm, Col_Rate_SOM, Col_Rate_VIP]
    color_INs = [Col_Rate_PVv, Col_Rate_SOM, Col_Rate_VIP]
    color_PEs = [Col_Rate_nE, Col_Rate_pE]
    
    m, n = 5, 10
    
    if ax1 is None:
        _, ax1 = plt.subplots(1,1, figsize=figsize, tight_layout=True)
        
    for i in range(3):
        
        mean_lower = np.mean(activity_int_lower_across[0,:,:,i],1)
        mean_higher = np.mean(activity_int_higher_across[0,:,:,i],1)
        
        ax1.scatter(mean_lower- mean_lower[0], mean_higher-mean_higher[0], marker='o', s=m*variability_across + n, color=color_INs[i])
    
    for i in range(2):
        
        mean_lower = np.mean(activity_pe_neurons_lower_across[0,:,:,i],1)
        mean_higher = np.mean(activity_pe_neurons_higher_across[0,:,:,i],1)
        
        ax1.scatter(mean_lower - mean_lower[0], mean_higher - mean_higher[0], marker='o', s=m*variability_across + n, color=color_PEs[i])
        
    ax1.axline((np.mean(mean_lower),np.mean(mean_lower)), slope=1, color='k', ls=':')
    ax1.tick_params(size=2)
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.set_xlabel(r'$\Delta$ FR lower (1/s)', fontsize=fs)
    ax1.set_ylabel(r'$\Delta$ FR higher (1/s)', fontsize=fs)
    ax1.locator_params(nbins=3)
    sns.despine(ax=ax1)
    
    if ax2 is None:
        _, ax2 = plt.subplots(1,1, figsize=figsize, tight_layout=True)
        
    for i in range(3):
        
        mean_lower = np.mean(activity_int_lower_within[:,0,:,i],1)
        mean_higher = np.mean(activity_int_higher_within[:,0,:,i],1)
        
        ax2.scatter(mean_lower- mean_lower[0], mean_higher-mean_higher[0], marker='o', s=m*variability_within + n, color=color_INs[i])
    
    for i in range(2):
        
        mean_lower = np.mean(activity_pe_neurons_lower_within[:,0,:,i],1)
        mean_higher = np.mean(activity_pe_neurons_higher_within[:,0,:,i],1)
        
        ax2.scatter(mean_lower - mean_lower[0], mean_higher - mean_higher[0], marker='o', s=m*variability_within + n, color=color_PEs[i])
        
    ax2.axline((np.mean(mean_lower),np.mean(mean_lower)), slope=1, color='k', ls=':')
    ax2.tick_params(size=2)
    ax2.tick_params(axis='both', labelsize=fs)
    ax2.set_xlabel(r'$\Delta$ FR lower (1/s)', fontsize=fs)
    ax2.locator_params(nbins=3)
    #ax2.set_ylabel(r'$\Delta$ FR higher', fontsize=fs)
    sns.despine(ax=ax2)
    
    if ax3 is None:
        _, ax3 = plt.subplots(1,1, figsize=figsize, tight_layout=True) 
        
    #labels_IN = ['PV1','PV2','SOM','VIP']
    labels_IN = ['PV','SOM','VIP']
    for i in range(3):
        
        x = np.empty(len(variability_across))
        y = np.empty(len(variability_across))
        x[:] = np.nan
        y[:] = np.nan
        
        ax3.scatter(x, y, marker='o', s=m*variability_across + n, color=color_INs[i], label=labels_IN[i])
    
    labels_E = ['nPE','pPE']
    for i in range(2):
        
        x = np.empty(len(variability_across))
        y = np.empty(len(variability_across))
        x[:] = np.nan
        y[:] = np.nan
        
        sc = ax3.scatter(x, y, marker='o', s=m*variability_across + n, color=color_PEs[i], label=labels_E[i])
    
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    legend1 = ax3.legend(loc=2, title="Cell type", fontsize=fs,frameon=False) #*sc.legend_elements()
    plt.setp(legend1.get_title(),fontsize=fs)
    ax3.add_artist(legend1)

    kw = dict(prop="sizes", color='k', func=lambda s: (s-n)/m) #fmt="$ {x:.2f}"
    legend2 = ax3.legend(*sc.legend_elements(**kw),loc=1, title="Variability", fontsize=fs, frameon=False)
    plt.setp(legend2.get_title(),fontsize=fs)
    
    sns.despine(ax=ax3, top=True, left=True, right=True, bottom=True)



def plot_mse_heatmap(end_of_initial_phase, means_tested, variances_tested, mse, vmax = None, lw = 1, flg_var=False, ax1=None,
                     title=None, show_mean=True, fs=6, figsize=(5,5), x_example = None, y_example = None, digits_round = 10):
    
    if ax1 is None:
        f, ax1 = plt.subplots(1,1, figsize=figsize, tight_layout=True)
    
    if show_mean:
        color_cbar = LinearSegmentedColormap.from_list(name='mse_mean', colors=['#FEFAE0', color_m_neuron])
    else:
        color_cbar = LinearSegmentedColormap.from_list(name='mse_variance', colors=['#FEFAE0', color_v_neuron])
    
    MSE_steady_state = np.mean(mse[:, :, end_of_initial_phase:],2)
    if flg_var:
        MSE_norm = 100 * MSE_steady_state.T/variances_tested[:,None]**2
    else:
        MSE_norm = 100 * MSE_steady_state.T/means_tested**2
        
    data = pd.DataFrame(MSE_norm, index=variances_tested, columns=means_tested)
    if vmax is None:
        vmax = np.ceil(digits_round * np.max(MSE_norm))/digits_round 
    
    vmin = np.floor(digits_round * np.min(MSE_norm))/digits_round 
    
    sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=color_cbar, xticklabels=3, yticklabels=2, ax=ax1,
                      cbar_kws={'label': r'normalised MSE$_\mathrm{\infty}$ (%)', 'ticks': [vmin, vmax]})
    
    ax1.invert_yaxis()
    
    mx, nx = np.polyfit([means_tested.min()-0.5, means_tested.max()+0.5], ax1.get_xlim(), 1)
    my, ny = np.polyfit([variances_tested.min()-0.5, variances_tested.max()+0.5], ax1.get_ylim(), 1)
    
    ax1.plot(mx * np.array([means_tested.min()-0.5, means_tested.max()+0.5]) + nx, 
             my * np.array([means_tested.min()-0.5, means_tested.max()+0.5]) + ny, color='k', ls=':')
    
    
    if x_example is not None:
        ax1.plot(mx * x_example + nx, my * y_example + ny, marker='*', color='k')
        
    
    ax1.tick_params(size=2.0) 
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.set_xlabel('Input mean', fontsize=fs)
    ax1.set_ylabel('Input variance', fontsize=fs)
    ax1.figure.axes[-1].yaxis.label.set_size(fs)
    ax1.figure.axes[-1].tick_params(labelsize=fs)
    
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
    
    ax1.plot(trials, stimuli, color=color_stimuli_background, label='input', lw=lw, marker='|', ls="None")
    ax1.plot(trials, running_average, color=color_running_average_stimuli, label='running average', lw=lw)
    ax1.plot(trials, m_neuron, color=color_m_neuron, lw=lw) #, label='M neuron'
    ax1.set_xlim([0,max(trials)])
    ax1.set_ylabel('Activity (1/s)', fontsize=fs)
    if ~mse_flg:
        ax1.set_xlabel('Time / stimulus duration', fontsize=fs)
    ax1.set_title('Activity of M neuron encodes input mean', fontsize=fs, pad=10)
    ax1.tick_params(axis='both', labelsize=fs)
    if legend_flg:
        ax1.legend(loc=4, ncol=3, handlelength=1, frameon=False, fontsize=fs)
    ax1.tick_params(size=2.0) 
    sns.despine(ax=ax1)
    
    if mse_flg:
        ax2.plot(trials, (running_average - m_neuron)**2, color=color_m_neuron) # color_mse
        if ylim_mse is not None:
            ax2.set_ylim(ylim_mse)
        ax2.set_xlim([0,max(trials)])
        ax2.set_ylabel('MSE', fontsize=fs)
        ax2.set_xlabel('Time / stimulus duration', fontsize=fs)
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
    
    ax1.plot(trials, momentary_variance, color=color_stimuli_background, label='instant. variance',
             lw=lw, ls="None", marker='|')
    ax1.plot(trials, variance_running, color=color_running_average_stimuli, label='averaged variance', lw=lw)
    ax1.plot(trials, v_neuron, color=color_v_neuron, lw=lw) #, label='V neuron'
    ax1.set_xlim([0,max(trials)])
    ax1.set_ylabel('Activity (1/s)', fontsize=fs)
    if ~mse_flg:
        ax1.set_xlabel('Time / stimulus duration', fontsize=fs)
    ax1.set_title('Activity of V neuron encodes input variance', fontsize=fs, pad=10)
    ax1.tick_params(axis='both', labelsize=fs)
    if legend_flg:
        ax1.legend(loc=1, ncol=2, frameon=False, handlelength=1, fontsize=fs)
    ax1.tick_params(size=2.0) 
    sns.despine(ax=ax1)
    
    if mse_flg:
        ax2.plot(trials, (variance_running - v_neuron)**2, color=color_v_neuron) # color_mse
        ax2.set_xlim([0,max(trials)])
        ax2.set_ylabel('MSE', fontsize=fs)
        ax2.set_xlabel('Time / trial duration', fontsize=fs)
        ax2.tick_params(size=2.0)
        ax2.tick_params(axis='both', labelsize=fs)
        sns.despine(ax=ax2)
  