#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% import

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from matplotlib.collections import LineCollection

# %%

min_mean = 3 # 1, 5
max_mean = 5 # 10, 5

epsilon = 0.1
sd_stimuli = epsilon + 1 # 0, 2

num_bars = 20
nums = 1000

y = np.zeros((1000,num_bars))
x = np.ones_like(y)

for i in range(num_bars):
    
    x[:,i] *= i
    
    mean_trial = np.random.uniform(min_mean, max_mean)
    min_value = mean_trial - np.sqrt(3)*sd_stimuli
    max_value = mean_trial + np.sqrt(3)*sd_stimuli
    
    y[:,i] = np.linspace(min_value, max_value, nums)

norm = plt.Normalize(y.min(), y.max())
fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

for i in range(num_bars):
    points = np.array([x[:,i], y[:,i]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(y[:,i])
    lc.set_linewidth(10)
    line = axs.add_collection(lc)

axs.axhline(np.mean(y), color='k', ls=':')

axs.set_xlim(x.min()-1, x.max()+1)
axs.set_xticks([])
axs.set_yticks([])
axs.set_ylim(0, 11)
plt.show()

sns.despine(ax=axs, bottom=True, top=True, left=True, right=True)