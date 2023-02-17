#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:29:48 2023

@author: loreen.hertaeg
"""

# %% imports

import numpy as np
import jax.numpy as jnp
from jax import random

from jax.lax import scan
from jax import vmap

#from src.plot_data import plot_example_mean, plot_example_variance, plot_mse_heatmap

import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

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


# %% Functions

def default_para_mfn(mfn_flag, baseline_activity = jnp.array([0, 0, 0, 0, 4, 4, 4, 4]), one_column=True):
    
    ### time constants
    tc_var_per_stim = 5000.
    tc_var_pred = 5000.
    
    tau_pe = jnp.array([60, 2])
    
    ### weights
    
    ## load data  
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + mfn_flag + '.pickle'
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    with open(filename,'rb') as f:
        xopt, W, optimize_flag, _, _ = pickle.load(f)
    
    xopt = jnp.asarray(xopt)
    optimize_flag = jnp.asarray(optimize_flag)
    
    ## connectivity within PE circuits
    w_PE_to_PE = jnp.copy(W)
    v_PE_to_PE = jnp.copy(W)
    
    w_PE_to_PE = w_PE_to_PE.at[optimize_flag!=0].set(xopt)
    v_PE_to_PE = v_PE_to_PE.at[optimize_flag!=0].set(xopt)
    
    ## external background input (here I assume it is the same for both parts)
    r_target = baseline_activity
    fixed_input = (jnp.eye(8) - w_PE_to_PE) @ r_target
    
    ## Define connectivity between PE circuit and P
    w_PE_to_P = jnp.zeros((1,8)) 
    w_PE_to_P = w_PE_to_P.at[0,0].set(-0.003)  # nPE onto P
    w_PE_to_P = w_PE_to_P.at[0,1].set(0.003)   # pPE onto P
    
    w_P_to_PE = jnp.zeros((8,1))  
    w_P_to_PE = w_P_to_PE.at[2:4,0].set(1.)          # onto dendrites
    w_P_to_PE = w_P_to_PE.at[5,0].set(1.)            # onto PV neuron receiving prediction
    w_P_to_PE = w_P_to_PE.at[6,0].set(1.-VS)         # onto SOM neuron
    w_P_to_PE = w_P_to_PE.at[7,0].set(1.-VV)         # onto VIP neuron    
    
    v_PE_to_P = jnp.zeros((1,8)) 
    v_PE_to_P = v_PE_to_P.at[0,0].set(-0.7*1e-3)    # nPE onto P
    v_PE_to_P = v_PE_to_P.at[0,1].set(0.7*1e-3)
    
    v_P_to_PE = jnp.zeros((8,1)) 
    v_P_to_PE = v_P_to_PE.at[2:4,0].set(1.)         # onto dendrites
    v_P_to_PE = v_P_to_PE.at[5,0].set(1.)           # onto PV neuron receiving prediction
    v_P_to_PE = v_P_to_PE.at[6,0].set(1.-VS)        # onto SOM neuron
    v_P_to_PE = v_P_to_PE.at[7,0].set(1.-VV)        # onto VIP neuron  
    
    ### correct weights to make sure that gain of nPE = gain of pPE
    if mfn_flag=='10':
        nPE_scale = 1.015
        pPE_scale = 1.023
    elif mfn_flag=='01':
        nPE_scale = 1.7
        pPE_scale = 1.7
    elif mfn_flag=='11':
        nPE_scale = 2.49
        pPE_scale = 2.53
    
    if one_column:
        w_PE_to_P = w_PE_to_P.at[0,0].multiply(nPE_scale)
        w_PE_to_P = w_PE_to_P.at[0,1].multiply(pPE_scale)
        w_PE_to_V = jnp.array([nPE_scale, pPE_scale])
    else:
        w_PE_to_P = w_PE_to_P.at[0,0].multiply(nPE_scale * 15)
        w_PE_to_P = w_PE_to_P.at[0,1].multiply(pPE_scale * 15)
        w_PE_to_V = jnp.array([nPE_scale, pPE_scale])
    
    v_PE_to_P = v_PE_to_P.at[0,0].multiply(nPE_scale)
    v_PE_to_P = v_PE_to_P.at[0,1].multiply(pPE_scale)
    v_PE_to_V = jnp.array([nPE_scale, pPE_scale])
        
    return [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
            v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
            tc_var_per_stim, tc_var_pred, tau_pe, fixed_input]



def random_uniform_from_moments(mean, sd, num, seed=186):
    
    b = jnp.sqrt(12) * sd / 2 + mean
    a = 2 * mean - b
    
    key = random.PRNGKey(seed)
    rnd = random.uniform(key, shape=(num,), minval=a, maxval=b) #jnp.random.uniform(a, b, size = num)
        
    return rnd


def run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, fixed_input, 
                    stimuli, VS = 1, VV = 0, dt = 1., w_PE_to_V = jnp.array([1,1]), n=2):
    
    ### neuron and network parameters
    tau_E, tau_I  = tau_pe
    neurons_feedforward = jnp.array([1, 1, 0, 0, 1, 0, VS, VV])
    
    ### initialise
    num_points = len(stimuli)
    m_neuron = jnp.zeros_like(stimuli)
    v_neuron = jnp.zeros_like(stimuli)
    rates_lower = jnp.zeros((num_points, 8))
    
    if fixed_input.ndim==1:
        fixed_input = jnp.tile(fixed_input, (num_points,1))
        
    current_pe_rates = rates_lower[0,:]
    current_m_rate = m_neuron[0]
    current_v_rate = v_neuron[0]
    
    ### run mean-field network
    for id_stim, stim in enumerate(stimuli):
        
        feedforward_input = fixed_input[id_stim,:] + stim * neurons_feedforward
        
        ## rates of PE circuit and M neuron
        [current_pe_rates, current_m_rate, 
         current_v_rate] = rate_dynamics_mfn(tau_E, tau_I, tc_var_per_stim, w_PE_to_V, 
                                             w_PE_to_P, w_P_to_PE, w_PE_to_PE,
                                             current_pe_rates, current_m_rate, 
                                             current_v_rate, feedforward_input, dt, n)
        
        rates_lower = rates_lower.at[id_stim,:].set(current_pe_rates)
        m_neuron = m_neuron.at[id_stim].set(current_m_rate)
        v_neuron = v_neuron.at[id_stim].set(current_v_rate)
        

    return m_neuron, v_neuron, rates_lower[:,:2]


def rate_dynamics_mfn(tau_E, tau_I, tc_var, w_var, U, V, W, rates, mean, 
                      var, feedforward_input, dt, n):
    
    # Initialise
    rates_new = jnp.copy(rates) 
    mean_new = jnp.copy(mean)
    dr_1 = jnp.zeros(len(rates_new))
    dr_2 = jnp.zeros(len(rates_new))
    var_new = jnp.copy(var) 
    
    # RK 2nd order
    dr_mem_1 = (U @ rates_new) / tau_E
    dr_var_1 = (-var_new + sum(w_var * rates_new[:2])**n) / tc_var 
    dr_1 = -rates_new + W @ rates_new + V @ jnp.array([mean_new]) + feedforward_input
    dr_1 = dr_1.at[:4].divide(tau_E)
    dr_1 = dr_1.at[4:].divide(tau_I)

    rates_new = rates_new.at[:].add(dt * dr_1)
    mean_new += dt * dr_mem_1
    var_new += dt * dr_var_1
    
    dr_mem_2 = (U @ rates_new) / tau_E
    dr_var_2 = (-var_new + sum(w_var * rates_new[:2])**n) / tc_var
    dr_2 = -rates_new + W @ rates_new + V @ mean_new + feedforward_input
    dr_2 = dr_2.at[:4].divide(tau_E)
    dr_2 = dr_2.at[4:].divide(tau_I)
    
    rates = rates.at[:].add(dt/2 * (dr_1 + dr_2))
    mean += dt/2 * (dr_mem_1 + dr_mem_2)
    var += dt/2 * (dr_var_1 + dr_var_2)
    
    # Rectify
    rates = rates.at[rates<0].set(0.)

    return [rates, mean[0], var]


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


# %% First few tests with jax

do_run = True

if do_run:

    mfn_flag = '10'

    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
    
    mean_stimuli = 5.
    std_stimuli = 2.
    num_values_per_trial = 200
    
    trial_duration = 100000
    repeats_per_value = trial_duration//num_values_per_trial
    
    stimuli = random_uniform_from_moments(mean_stimuli, std_stimuli, num_values_per_trial)
    stimuli = np.repeat(stimuli, repeats_per_value)
    
    m_neuron, v_neuron, pe = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                             fixed_input, stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
    
    plot_example_mean(stimuli, trial_duration, m_neuron, figsize=(4,3), fs=7, lw=1.2)
    plot_example_variance(stimuli, trial_duration, v_neuron, figsize=(4,3), fs=7, lw=1.2)

