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

from src.plot_data import plot_influence_interneurons_gain_baseline

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_4_S3.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(5/inch,4/inch)
fig = plt.figure(figsize=figsize, tight_layout=True)

G = gridspec.GridSpec(1, 1, figure=fig)
ax_A = fig.add_subplot(G[0,0])


# %% How do the interneurons influence the PE neuron properties?

plot_influence_interneurons_gain_baseline(ax=ax_A)

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)