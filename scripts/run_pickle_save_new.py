#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 08:03:15 2023

@author: loreen.hertaeg
"""

# %% Import

import pickle
import numpy as np

from src.default_parameters import Neurons, Activity_Zero, Network, Stimulation
from src.functions_save import load_network_para

import sys
sys.path.append("../../PE_Circuits")

# %% load pickle data and save in different format

# load data
path = '../../PE_Circuits/Results/Data/Plasticity'
filename = path + '/Data_NetworkParameters_Example_Target_Input_After.pickle'

with open(filename,'rb') as f:
    NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, LearnPar = pickle.load(f)
    
# rename and save differently
weight_name = NetPar.weight_name
neurons_visual = InPar.neurons_visual
inp_ext_soma = InPar.inp_ext_soma
inp_ext_dend = InPar.inp_ext_dend

Dict_w = {}
Dict_t = {}

for i in range(25):
    
    m,n = np.unravel_index(i,(5,5))
    exec('Dict_w["' + NetPar.weight_name[m][n] + '"] = NetPar.' + NetPar.weight_name[m][n])
    exec('Dict_t["T' + NetPar.weight_name[m][n][1:] + '"] = NetPar.T' + NetPar.weight_name[m][n][1:])


file_for_data = '../results/data/population/data_population_network_parameters.pickle'

with open(file_for_data,'wb') as f:
    pickle.dump([neurons_visual, inp_ext_soma, inp_ext_dend, weight_name, Dict_w, Dict_t],f)
