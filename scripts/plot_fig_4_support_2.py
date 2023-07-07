#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.functions_simulate import simulate_neuromod
from src.plot_data import plot_heatmap_neuromod, plot_combination_activation_INs

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% import

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path
import seaborn as sns

from matplotlib.collections import LineCollection

from src.plot_data import plot_influence_interneurons_baseline_or_gain

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_4_S2.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(9/inch,5/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(1, 2, figure=fig, wspace=0.7)

ax_A = fig.add_subplot(G[0,0])
ax_A.text(-0.4, 1.1, 'A', transform=ax_A.transAxes, fontsize=fs+1)
ax_B = fig.add_subplot(G[0,1])
ax_B.text(-0.4, 1.1, 'B', transform=ax_B.transAxes, fontsize=fs+1)

# %% How do the interneurons influence the PE neuron properties?

plot_influence_interneurons_baseline_or_gain(ax=ax_A, plot_annotation=False)
plot_influence_interneurons_baseline_or_gain(plot_baseline=False, ax=ax_B)

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)