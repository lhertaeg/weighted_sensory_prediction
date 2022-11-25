#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 08:56:15 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.mean_field_model import default_para, stimuli_moments_from_uniform, run_mean_field_model, alpha_parameter_exploration
from src.plot_toy_model import plot_limit_case, plot_alpha_para_exploration_ratios, plot_fraction_sensory_comparsion, plot_alpha_para_exploration
from src.plot_toy_model import plot_manipulation_results
# from src.mean_field_model import stimuli_moments_from_uniform, run_toy_model, default_para, alpha_parameter_exploration
# from src.mean_field_model import random_uniform_from_moments, random_lognormal_from_moments, random_gamma_from_moments
# from src.mean_field_model import stimuli_from_mean_and_std_arrays
from src.plot_results_mfn import plot_limit_case_example, plot_transitions_examples, heatmap_summary_transitions, plot_transition_course

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% erase after testing

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    
color_sensory = '#D76A03'
color_prediction = '#19535F'

cmap_sensory_prediction = LinearSegmentedColormap.from_list(name='cmap_sensory_prediction', 
                                                        colors=['#19535F','#fefee3','#D76A03'])

# %% No squared activation function

# Results: 
    # var(S)<var(P): weighting with abs wil lead to decaresed sensory weight
    # var(S)>var(P): weighting with abs will lead to increased sensory weight
    # In principle, the weighting in general is fine, that is, when var(S)<var(P), system is sensory driven, while when var(S)>var(P), the system is more prediction driven
    # it just seems that there is a tendency to taking the "other" more into account
    # however, it is inetresting to see that at the corners, both approaches are equal

flag = 0

if flag==1:
    
    # define fixed neuron parameter
    tau = 500
    
    
    # define stimuli
    mu = 10
    var = 9
    stimuli = np.random.normal(mu, np.sqrt(var), 10000)
    
    
    # initialise 
    v_neuron_2 = np.zeros_like(stimuli)
    v_neuron = np.zeros_like(stimuli)
    
    for id_stim, s in enumerate(stimuli): 
        
        v_neuron_2[id_stim] = (1-1/tau) * v_neuron_2[id_stim-1] + (s - mu)**2/tau
        v_neuron[id_stim] = (1-1/tau) * v_neuron[id_stim-1] + abs(s - mu)/tau


    # # plot
    # plt.figure()
    # plt.plot(v_neuron_2)
    # plt.plot(v_neuron)
    
    
    ### implications for the weighting 
    # only approximately because taking abs, see above, is not equal to the std but close enough
    x = np.linspace(0.1,5,101) # s
    y = np.linspace(0.1,5,100) # p
    
    X, Y = np.meshgrid(x, y)
    
    weighting_abs = 1/(1+X/Y)
    weighting_squared = 1/(1+X**2/Y**2)
    
    f, axs = plt.subplots(1,3, figsize=(15,5))
    
    axs[0].plot(x, weighting_squared[8,:])
    axs[0].plot(x, weighting_abs[8,:])
    axs[0].set_title(str(x[8]))
    axs[0].legend(['squared', 'approx(abs)'])
    
    axs[1].plot(x, weighting_squared[50,:])
    axs[1].plot(x, weighting_abs[50,:])
    axs[1].set_title(str(x[50]))
    
    axs[2].plot(x, weighting_squared[99,:])
    axs[2].plot(x, weighting_abs[99,:])
    axs[2].set_title(str(x[99]))
    
    plt.figure()
    plt.plot(weighting_squared.flatten(), weighting_abs.flatten(), '.')
    ax = plt.gca()
    ax.axline((0.5,0.5), slope=1, ls=':', color='k')
    ax.set_xlabel('alpha (squared weighting)')
    ax.set_ylabel('alpha ("abs" weighting)')
    
    
# %% Higher task demands (expressed through increased BL activity in PE neurons) 
# alters the variance stimation and the "optimal" weighting of S and P

# look at O'Reilly 2012
# for cognitive load:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7116493/
    # https://www.nature.com/articles/s41598-017-07897-z
    # https://onlinelibrary.wiley.com/doi/pdf/10.1002/brb3.128
    # cognitive load --> arousal/stress --> increased BL
    

    