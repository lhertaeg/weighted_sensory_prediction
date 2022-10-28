#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:05:25 2022

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


# %% Effect of neuromodulators on weighted output

flag = 1

if flag==1:
    
    ### initialise
    z_exp_low_unexp_high = np.zeros(3)
    z_exp_high_unexp_low = np.zeros(3)
    columns = np.array([1,3,2])
    colors = ['#19535F', '#D76A03']
    
    ### define neuromodulator (by target IN)
    id_cell = 3
    
    if id_cell==0:
        title = 'DA activates PVs (sens)'
    elif id_cell==1:
        title = 'DA activates PVs (pred)'
    elif id_cell==2:
        title = 'NA/NE activates SOMs'
    elif id_cell==3:
        title = 'ACh or 5-HT activates VIPs'
    
    for idx, column in enumerate(columns):
    
        ### define MFN
        input_flg = '10' #['10', '01', '11']
        
        ### load data
        if column!=3: # only one of the two columns
            file_data4plot = '../results/data/weighting_perturbation/test_weighting_perturbations_' + input_flg + '_column_' + str(column) + '.pickle'
        
            with open(file_data4plot,'rb') as f:
                [_, _, _, frac_sens_before_pert, frac_sens_after_pert, _] = pickle.load(f)
        
        else: # both columns 
            file_data4plot = '../results/data/weighting_perturbation/test_weighting_perturbations_' + input_flg + '.pickle'
    
            with open(file_data4plot,'rb') as f:
                [_, _, _, frac_sens_before_pert, frac_sens_after_pert, _] = pickle.load(f)
         
        # reminder   
        # std_mean_arr = np.linspace(0,3,5, dtype=dtype)    # column
        # std_std_arr = np.linspace(0,3,5, dtype=dtype)     # row
            
        frac_exp_low_unexp_high_before = frac_sens_before_pert[1, id_cell, 0, 4]
        frac_exp_low_unexp_high_after = frac_sens_after_pert[1, id_cell, 0, 4]
        
        frac_exp_high_unexp_low_before = frac_sens_before_pert[1, id_cell, 4, 0]
        frac_exp_high_unexp_low_after = frac_sens_after_pert[1, id_cell, 4, 0]
        
        z_exp_low_unexp_high[idx] = (frac_exp_low_unexp_high_after - frac_exp_low_unexp_high_before) / frac_exp_low_unexp_high_before
        z_exp_high_unexp_low[idx] = (frac_exp_high_unexp_low_after - frac_exp_high_unexp_low_before) / frac_exp_high_unexp_low_before
    
    ### plot
    fig, ax = plt.subplots(1,1, tight_layout=True, figsize=(4,3))
    
    ax.plot(np.arange(3), z_exp_low_unexp_high * 100, '.-', color=colors[1])
    ax.plot(np.arange(3), z_exp_high_unexp_low * 100, '.-', color=colors[0])
    
    ylim = ax.get_ylim()
    ax.axhspan(0, ylim[1], color=colors[1], alpha=0.1)
    ax.axhspan(ylim[0], 0, color=colors[0], alpha=0.1)
    ax.set_ylim(ylim)
    
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['first', 'both', 'second'])
    
    ax.set_ylabel(r'change in $\alpha$ (normalised, %)')
    ax.set_xlabel('target PE circuit')
    ax.set_title(title)
    
    sns.despine(ax=ax)
        
