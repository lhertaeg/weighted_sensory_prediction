#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:15:10 2022

@author: loreen.hertaeg
"""
import numpy as np
import matplotlib.pyplot as plt

from Default_values import DefaultParameters_within, DefaultParameters_between
from Functions_Network_all import Neurons, InitialRate, Stimulation, SaveData
from Functions_Network_all import Connection_within, Connection_between

dtype = np.float32

# %% 

# neuron parameters
num_cells_0 = [1, 1]
num_cells_1 = [2, 2, 1]
num_cells_2 = [1, 1]
num_cells_3 = [2, 2, 1]
num_cells_4 = [1, 1]
num_cells_5 = [2, 2, 1]
num_cells_6 = [1, 1]

NeuPar = Neurons(num_cells_0 = num_cells_0, num_cells_1 = num_cells_1, num_cells_2 = num_cells_2,
                 num_cells_3 = num_cells_3, num_cells_4 = num_cells_4, num_cells_5 = num_cells_5,
                 num_cells_6 = num_cells_6)

# initialise all neurons/compartments
IniRate = InitialRate(NeuPar)

# define sensory inputs (stats) and external background inputs 
S1 = np.random.uniform(1, 5, size=500)
S2 = np.random.uniform(3, 9, size=500)
sensory_input_mean_per_trial = np.concatenate((S1,S2))

S2_1 = np.random.uniform(1, 5, size=500)
S2_2 = np.random.uniform(1, 7, size=500)
sensory_input_sd_per_trial = sensory_input_mean_per_trial = np.concatenate((S2_1,S2_2))

background = np.zeros(17, dtype=dtype)
num_time_steps_per_trial = 1000

StimPar = Stimulation(NeuPar, background, sensory_input_mean_per_trial, sensory_input_sd_per_trial, num_time_steps_per_trial)

# set weights between neurons
[num_cells_areas, conn_prob_within, indegree_within, digagonal_within, total_weight_within, mean_weight_within,
sd_weight_within, min_weight_within, max_weight_within, weight_dist_within, weight_name_within, temp_name_within] = DefaultParameters_within()

[num_cells_region_pre, num_cells_region_post, conn_prob_between, indegree_between, digagonal_between, total_weight_between, 
 mean_weight_between, sd_weight_between, min_weight_between, max_weight_between, weight_dist_between, weight_name_between, temp_name_between] = DefaultParameters_between()

Connection_within = Connection_within(num_cells_areas, conn_prob_within, indegree_within, digagonal_within, total_weight_within, mean_weight_within,
                                      sd_weight_within, min_weight_within, max_weight_within, weight_dist_within, weight_name_within, temp_name_within)

Connection_between = Connection_between(num_cells_region_pre, num_cells_region_post, conn_prob_between, indegree_between, digagonal_between, 
                                        total_weight_between, mean_weight_between, sd_weight_between, min_weight_between, max_weight_between, 
                                        weight_dist_between, weight_name_between, temp_name_between)

# recording parameters
SavePar = SaveData()

