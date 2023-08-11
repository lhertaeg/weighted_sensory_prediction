#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:31:58 2023

@author: loreen.hertaeg
"""

# %% Import

import os
import pickle


# %% functions


def load_network_para(folder):
    
    path = '../results/data/' + folder
    filename = path + '/data_population_network_parameters.pickle'
    
    with open(filename,'rb') as f:
        NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, LearnPar = pickle.load(f)
        
    return NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, LearnPar