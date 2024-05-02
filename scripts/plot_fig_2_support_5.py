#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path

from src.plot_data import plot_impact_baseline

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_2_S5.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)
    
    
# %% Define figure structure

figsize=(12/inch,9/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2,1], hspace=0.3)
AB = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=G[0,0])
C = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G[1,0])
                      
ax_A1 = fig.add_subplot(AB[0,0])
ax_A2 = fig.add_subplot(AB[0,1], sharey=ax_A1)
ax_A3 = fig.add_subplot(AB[0,2], sharey=ax_A1)
ax_A1.text(-0.45, 1., 'A', transform=ax_A1.transAxes, fontsize=fs+1)
plt.setp(ax_A2.get_yticklabels(), visible=False)
plt.setp(ax_A3.get_yticklabels(), visible=False)

ax_B1 = fig.add_subplot(AB[1,0])
ax_B2 = fig.add_subplot(AB[1,1], sharey=ax_B1)
ax_B3 = fig.add_subplot(AB[1,2], sharey=ax_B1) 
ax_B1.text(-0.45, 1., 'B', transform=ax_B1.transAxes, fontsize=fs+1)
plt.setp(ax_B2.get_yticklabels(), visible=False)
plt.setp(ax_B3.get_yticklabels(), visible=False)

ax_C1 = fig.add_subplot(C[0,0])
ax_C2 = fig.add_subplot(C[0,1], sharey=ax_C1)
ax_C3 = fig.add_subplot(C[0,2], sharey=ax_C1)
ax_C1.text(-0.45, 1., 'C', transform=ax_C1.transAxes, fontsize=fs+1)
plt.setp(ax_C2.get_yticklabels(), visible=False)
plt.setp(ax_C3.get_yticklabels(), visible=False)

plt.setp(ax_A1.get_xticklabels(), visible=False)
plt.setp(ax_B1.get_xticklabels(), visible=False)
plt.setp(ax_A2.get_xticklabels(), visible=False)
plt.setp(ax_B2.get_xticklabels(), visible=False)
plt.setp(ax_A3.get_xticklabels(), visible=False)
plt.setp(ax_B3.get_xticklabels(), visible=False)

axs = np.array([[ax_A1, ax_A2, ax_A3], [ax_B1, ax_B2, ax_B3], [ax_C1, ax_C2, ax_C3]])  


# %% Plot

input_to_increase_baseline = np.linspace(0,0.5,5)
plot_impact_baseline(input_to_increase_baseline, axs=axs) 


# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
    