#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:12:21 2022

@author: loreen.hertaeg
"""

import os
import tables
import numpy as np
from numba import njit
from typing import NamedTuple

dtype = np.float32

# %% classes

class Neurons(NamedTuple):
        
        num_cells_0: list = np.array([80,20], dtype=np.int32)       # E, I
        num_cells_1: list = np.array([80, 80, 20], dtype=np.int32)  # E, D, I
        num_cells_2: list = np.array([80,20], dtype=np.int32)
        num_cells_3: list = np.array([80, 80, 20], dtype=np.int32)
        num_cells_4: list = np.array([80,20], dtype=np.int32)
        num_cells_5: list = np.array([80, 80, 20], dtype=np.int32)
        num_cells_6: list = np.array([80,20], dtype=np.int32)
        
        tci_0: list = np.array([1/20, 1/10], dtype=dtype)
        tci_1: list = np.array([1/20, 1/20, 1/10], dtype=dtype)
        tci_2: list = np.array([1/20, 1/10], dtype=dtype)
        tci_3: list = np.array([1/20, 1/20, 1/10], dtype=dtype)
        tci_4: list = np.array([1/20, 1/10], dtype=dtype)
        tci_5: list = np.array([1/20, 1/20, 1/10], dtype=dtype)
        tci_6: list = np.array([1/20, 1/10], dtype=dtype)
        
        leak_0: list = np.array([1, 1], dtype=dtype)
        leak_1: list = np.array([1, 1, 1], dtype=dtype)
        leak_2: list = np.array([1, 1], dtype=dtype)
        leak_3: list = np.array([1, 1, 1], dtype=dtype)
        leak_4: list = np.array([1, 1], dtype=dtype)
        leak_5: list = np.array([1, 1, 1], dtype=dtype)
        leak_6: list = np.array([1, 1], dtype=dtype)
        
        rheo_0: list = np.array([0, 0], dtype=dtype)
        rheo_1: list = np.array([0, 0, 0], dtype=dtype)
        rheo_2: list = np.array([0, 0], dtype=dtype)
        rheo_3: list = np.array([0, 0, 0], dtype=dtype)
        rheo_4: list = np.array([0, 0], dtype=dtype)
        rheo_5: list = np.array([0, 0, 0], dtype=dtype)
        rheo_6: list = np.array([0, 0], dtype=dtype)
        
        p_0: list = np.array([1, 1], dtype=dtype)
        p_1: list = np.array([1, 1, 1], dtype=dtype)
        p_2: list = np.array([2, 1], dtype=dtype)
        p_3: list = np.array([1, 1, 1], dtype=dtype)
        p_4: list = np.array([1, 1], dtype=dtype)
        p_5: list = np.array([1, 1, 1], dtype=dtype)
        p_6: list = np.array([1, 1], dtype=dtype)
 
    
class Connection_within:
    def __init__(self, num_cells_areas, conn_prob_within, indegree_within, digagonal_within, total_weight_within, mean_weight_within,
                 sd_weight_within, min_weight_within, max_weight_within, weight_dist_within, weight_name_within, temp_name_within): # all lists that either contain lists or arrays ...
    
        # create connectivity matrices for each area (within)
        for l in range(7):
            
            conn_prob = conn_prob_within[l]
            indegree = indegree_within[l]
            diagonal = digagonal_within[l]
            total_weight = total_weight_within[l]
            mean_weight = mean_weight_within[l]
            sd_weight = sd_weight_within[l]
            min_weight = min_weight_within[l]
            max_weight = max_weight_within[l]
            weight_dist = weight_dist_within[l]
            num_cells = num_cells_areas[l]
            weight_name = weight_name_within[l]
            temp_name = temp_name_within[l]
                 
            num_types = len(num_cells)
            
            for j in range(num_types):
                for i in range(num_types):
                    
                    ConnTemp = create_connectivity_template(num_cells, conn_prob[i,j], indegree[i,j], diagonal[i,j])
                    ConnWeight = create_connectivity(ConnTemp, total_weight[i,j], mean_weight[i,j], sd_weight[i,j], min_weight[i,j], 
                                                     max_weight[i,j], weight_dist[i,j])
            
                    exec('self.' + weight_name[i,j] + ' = ConnWeight')
                    exec('self.' + temp_name[i,j] + ' = ConnTemp') 
                

class Connection_between: 
    def __init__(self, num_cells_region_pre, num_cells_region_post, conn_prob_between, indegree_between, digagonal_between, 
                 total_weight_between, mean_weight_between, sd_weight_between, min_weight_between, max_weight_between, 
                 weight_dist_between, weight_name_between, temp_name_between):
        
        # pre_regions = input, 0, 1, 2, 3, 4, 5, 6
        # post_regions = 0, 1, 2, 3, 4, 5, 6
        
        # num_cells_region_pre = list or array (1D)
        # num_cells_region_post = list, each entry is array of length 2 or 3
        # others same as num_cells_region_post
        
        # E in pre_region onto E, D, I in post_region
        for l in range(8): # pre_regions
            
            conn_prob = conn_prob_between[l]
            indegree = indegree_between[l]
            diagonal = digagonal_between[l]
            total_weight = total_weight_between[l]
            mean_weight = mean_weight_between[l]
            sd_weight = sd_weight_between[l]
            min_weight = min_weight_between[l]
            max_weight = max_weight_between[l]
            weight_dist = weight_dist_between[l]
            weight_name = weight_name_between[l]
            temp_name = temp_name_between[l]
            num_cells_post = num_cells_region_post[l]
            num_cells_pre = num_cells_region_pre[l]
            
            for j in range(len(num_cells_post)):
                
                num_cells = [num_cells_post[j], num_cells_pre]
                
                ConnTemp = create_connectivity_template(num_cells, conn_prob[j], indegree[j], diagonal[j])
                ConnWeight = create_connectivity(ConnTemp, total_weight[j], mean_weight[j], sd_weight[j], min_weight[j], 
                                                     max_weight[j], weight_dist[j])
            
                exec('self.' + weight_name[j] + ' = ConnWeight')
                exec('self.' + temp_name[j] + ' = ConnTemp') 
        

class Stimulation:
    def __init__(self, NeuPar, background, sensory_input_mean_per_trial, sensory_input_sd_per_trial, num_time_steps_per_trial, flg_back = 0, dt=dtype(0.1)):
        
        num_cells_0 = NeuPar.num_cells_0
        num_cells_1 = NeuPar.num_cells_1
        num_cells_2 = NeuPar.num_cells_2
        num_cells_3 = NeuPar.num_cells_3
        num_cells_4 = NeuPar.num_cells_4
        num_cells_5 = NeuPar.num_cells_5
        num_cells_6 = NeuPar.num_cells_6
        
        self.sensory_input_mean_per_trial = sensory_input_mean_per_trial
        self.sensory_input_sd_per_trial = sensory_input_sd_per_trial
        self.num_time_steps_per_trial = num_time_steps_per_trial
        self.dt = dt
        
        if flg_back==0: # background is interpreted as external input 
            self.XE_0 = np.repeat(background[0], num_cells_0[0])
            self.XI_0 = np.repeat(background[1], num_cells_0[0])
            self.XE_1 = np.repeat(background[2], num_cells_1[0])
            self.XD_1 = np.repeat(background[3], num_cells_1[1])
            self.XI_1 = np.repeat(background[4], num_cells_1[2]) 
            self.XE_2 = np.repeat(background[5], num_cells_2[0])
            self.XI_2 = np.repeat(background[6], num_cells_2[1]) 
            self.XE_3 = np.repeat(background[7], num_cells_3[0])
            self.XD_3 = np.repeat(background[8], num_cells_3[1]) 
            self.XI_3 = np.repeat(background[9], num_cells_3[2])
            self.XE_4 = np.repeat(background[10], num_cells_4[0])
            self.XI_4 = np.repeat(background[11], num_cells_4[1]) 
            self.XE_5 = np.repeat(background[12], num_cells_5[0])
            self.XD_5 = np.repeat(background[13], num_cells_5[1])
            self.XI_5 = np.repeat(background[14], num_cells_5[2])
            self.XE_6 = np.repeat(background[15], num_cells_6[0])
            self.XI_6 = np.repeat(background[16], num_cells_6[1])
        else: # background is interpreted as baseline rates (activity) in the absence of sensory input => external input needs to be computed ...
            print('XXX') # 
        
        
class InitialRate:
    def __init__(self, NeuPar, rE_0=None, rI_0=None, rE_1=None, rD_1=None, rI_1=None, rE_2=None, rI_2=None, rE_3=None, 
                 rD_3=None, rI_3=None, rE_4=None, rI_4=None, rE_5=None, rD_5=None, rI_5=None, rE_6=None, rI_6=None):
        
        num_cells_0 = NeuPar.num_cells_0
        num_cells_1 = NeuPar.num_cells_1
        num_cells_2 = NeuPar.num_cells_2
        num_cells_3 = NeuPar.num_cells_3
        num_cells_4 = NeuPar.num_cells_4
        num_cells_5 = NeuPar.num_cells_5
        num_cells_6 = NeuPar.num_cells_6
        
        if rE_0 is None: self.rE_0 = np.zeros(num_cells_0[0], dtype=dtype)
        else: self.rE_0 = rE_0
        
        if rI_0 is None: self.rI_0 = np.zeros(num_cells_0[1], dtype=dtype)
        else: self.rI_0 = rI_0
            
        if rE_1 is None: self.rE_1 = np.zeros(num_cells_1[0], dtype=dtype)
        else: self.rE_1 = rE_1
            
        if rD_1 is None: self.rD_1 = np.zeros(num_cells_1[1], dtype=dtype)
        else: self.rD_1 = rD_1
        
        if rI_1 is None: self.rI_1 = np.zeros(num_cells_1[2], dtype=dtype)
        else: self.rI_1 = rI_1
        
        if rE_2 is None: self.rE_2 = np.zeros(num_cells_2[0], dtype=dtype)
        else: self.rE_2 = rE_2
        
        if rI_2 is None: self.rI_2 = np.zeros(num_cells_2[1], dtype=dtype)
        else: self.rI_2 = rI_2
        
        if rE_3 is None: self.rE_3 = np.zeros(num_cells_3[0], dtype=dtype)
        else: self.rE_3 = rE_3
            
        if rD_3 is None: self.rD_3 = np.zeros(num_cells_3[1], dtype=dtype)
        else: self.rD_3 = rD_3
        
        if rI_3 is None: self.rI_3 = np.zeros(num_cells_3[2], dtype=dtype)
        else: self.rI_3 = rI_3
        
        if rE_4 is None: self.rE_4 = np.zeros(num_cells_4[0], dtype=dtype)
        else: self.rE_4 = rE_4
            
        if rI_4 is None: self.rI_4 = np.zeros(num_cells_4[1], dtype=dtype)
        else: self.rI_4 = rI_4
        
        if rE_5 is None: self.rE_5 = np.zeros(num_cells_5[0], dtype=dtype)
        else: self.rE_5 = rE_5
            
        if rD_5 is None: self.rD_5 = np.zeros(num_cells_5[1], dtype=dtype)
        else: self.rD_5 = rD_5
        
        if rI_5 is None: self.rI_5 = np.zeros(num_cells_5[2], dtype=dtype)
        else: self.rI_5 = rI_5
        
        if rE_6 is None: self.rE_6 = np.zeros(num_cells_6[0], dtype=dtype)
        else: self.rE_6 = rE_6
        
        if rI_6 is None: self.rI_6 = np.zeros(num_cells_6[1], dtype=dtype)
        else: self.rI_6 = rI_6
        

class SaveData(NamedTuple):
    
    save_every_n_steps: dtype = dtype(500)
    save_pop:  bool = True
    save_id:  list = np.zeros(17, dtype=np.int32) 
    

# %% functions

def create_connectivity_template(num_cells, conn_prob, indegree, diagonal=False):

    num_ones = np.int32(conn_prob * num_cells[1])
    binary = np.array([0] * (num_cells[1] - num_ones) + [1] * num_ones)
    
    if diagonal:
        Mtx = np.eye(num_cells[0], dtype=np.int32())
    else:
        Mtx = np.zeros((num_cells[0],num_cells[1]), dtype=np.int32())
        if indegree:
            for k in range(num_cells[0]):
                np.random.shuffle(binary)
                Mtx[k,:] = binary
        else:
            Mtx = np.random.choice([0,1], size=(num_cells[0],num_cells[1]), p=[1-conn_prob,conn_prob])
            
    return Mtx


def create_connectivity(template, total_weight=1, mean_weight=1, sd_weight=1, min_weight=-np.inf, max_weight=np.inf, weight_dist=0):
    
    # weight_dist: 0 = no distribution, 1 = uniform, 2 = normal
    m, n = np.size(template,0), np.size(template,1)
    
    if weight_dist==0:
        W = template * mean_weight / np.sum(template,1)[:,None]
        
    elif weight_dist==1:
        W = template * np.random.uniform(min_weight, max_weight, size=(m,n)) / np.sum(template,1)[:,None]
        
    elif weight_dist==2:
        W = template * np.random.normal(mean_weight, sd_weight, size=(m,n))
        
        N = len(W[((W<min_weight) | (W>max_weight)) & (W!=0)])
        while N>0:
            W[((W<min_weight) | (W>max_weight)) & (W!=0)] = np.random.normal(mean_weight, sd_weight, size=N)
            N = len(W[((W<min_weight) | (W>max_weight)) & (W!=0)])
            
        W /= np.sum(template,1)[:,None]
        
    if total_weight is not None:
        norm = total_weight / np.sum(W, 1)
        W *= norm[:,None]
    
    W[np.isnan(W)] = 0
    
    return W
    

@njit(cache=True)
def drdt_EI(rE, rI, gE, gI, tc_inv_E, tc_inv_I, rheo_E, rheo_I,
            pE, pI, wEE, wEI, wIE, wII, IE, II):
    
    drE = tc_inv_E * (-gE * rE + (wEE @ rE + wEI @ rI + IE - rheo_E)**pE)
    drI = tc_inv_I * (-gI * rI + (wIE @ rE + wII @ rI + II - rheo_I)**pI)
    
    return drE, drI


@njit(cache=True)
def drdt_EDI(rE, rD, rI, gE, gD, gI, tc_inv_E, tc_inv_D, tc_inv_I, rheo_E, rheo_D, rheo_I,
            pE, pD, pI, wEE, wED, wEI, wDE, wDI, wIE, wII, IE, ID, II):
    
    drE = tc_inv_E * (-gE * rE + (wEE @ rE + wED @ rD + wEI @ rI + IE - rheo_E)**pE)
    drD = tc_inv_D * (-gD * rD + (wDE @ rE + wDI @ rI + ID - rheo_D)**pD)
    drI = tc_inv_I * (-gI * rI + (wIE @ rE + wII @ rI + II - rheo_I)**pI)
    
    return drE, drD, drI


def RateDynamics(rE_0, rI_0, rE_1, rD_1, rI_1, rE_2, rI_2, rE_3, rD_3, rI_3, rE_4, rI_4, rE_5, rD_5, rI_5, rE_6, rI_6,
                 leak_0, leak_1, leak_2, leak_3, leak_4, leak_5, leak_6, tci_0, tci_1, tci_2, tci_3, tci_4, tci_5, tci_6,
                 rheo_0, rheo_1, rheo_2, rheo_3, rheo_4, rheo_5, rheo_6, p_0, p_1, p_2, p_3, p_4, p_5, p_6, wEE_0, wEI_0,
                 wIE_0, wII_0, wEE_1, wED_1, wEI_1, wDE_1, wDI_1, wIE_1, wII_1, wEE_2, wEI_2, wIE_2, wII_2, wEE_3, wED_3,
                 wEI_3, wDE_3, wDI_3, wIE_3, wII_3, wEE_4, wEI_4, wIE_4, wII_4, wEE_5, wED_5, wEI_5, wDE_5, wDI_5, wIE_5, 
                 wII_5, wEE_6, wEI_6, wIE_6, wII_6, IE_0, II_0, IE_1, ID_1, II_1, IE_2, II_2, IE_3, ID_3, II_3, IE_4, II_4,
                 IE_5, ID_5, II_5, IE_6, II_6, dt):
    
    rE_0 = rE_0.copy()
    rI_0 = rI_0.copy()
    rE_1 = rE_1.copy()
    rD_1 = rD_1.copy()
    rI_1 = rI_1.copy()
    rE_2 = rE_2.copy()
    rI_2 = rI_2.copy()
    rE_3 = rE_3.copy()
    rD_3 = rD_3.copy()
    rI_3 = rI_3.copy()
    rE_4 = rE_4.copy()
    rI_4 = rI_4.copy()
    rE_5 = rE_5.copy()
    rD_5 = rD_5.copy()
    rI_5 = rI_5.copy()
    rE_6 = rE_6.copy()
    rI_6 = rI_6.copy()

    drE0_1, drI0_1 = drdt_EI(rE_0, rI_0, leak_0[0], leak_0[1], tci_0[0], tci_0[1], 
                             rheo_0[0], rheo_0[1], p_0[0], p_0[1], wEE_0, wEI_0, wIE_0, wII_0, IE_0, II_0)
    
    drE2_1, drI2_1 = drdt_EI(rE_2, rI_2, leak_2[0], leak_2[1], tci_2[0], tci_2[1], 
                             rheo_2[0], rheo_2[1], p_2[0], p_2[1], wEE_2, wEI_2, wIE_2, wII_2, IE_2, II_2)
    
    drE4_1, drI4_1 = drdt_EI(rE_4, rE_4, leak_4[0], leak_4[1], tci_4[0], tci_4[1], 
                             rheo_4[0], rheo_4[1], p_4[0], p_4[1], wEE_4, wEI_4, wIE_4, wII_4, IE_4, II_4)
    
    drE6_1, drI6_1 = drdt_EI(rE_6, rI_6, leak_6[0], leak_6[1], tci_6[0], tci_6[1], 
                             rheo_6[0], rheo_6[1], p_6[0], p_6[1], wEE_6, wEI_6, wIE_6, wII_6, IE_6, II_6)
    
    drE1_1, drD1_1, drI1_1 = drdt_EDI(rE_1, rD_1, rI_1, leak_1[0], leak_1[1], leak_1[2], tci_1[0], tci_1[1], tci_1[2],
                                      rheo_1[0], rheo_1[1], rheo_1[2], p_1[0], p_1[1], p_1[2], wEE_1, wED_1, wEI_1, wDE_1, wDI_1,
                                      wIE_1, wII_1, IE_1, ID_1, II_1)
    
    drE3_1, drD3_1, drI3_1 = drdt_EDI(rE_3, rD_3, rI_3, leak_3[0], leak_3[1], leak_3[2], tci_3[0], tci_3[1], tci_3[2],
                                      rheo_3[0], rheo_3[1], rheo_3[2], p_3[0], p_3[1], p_3[2], wEE_3, wED_3, wEI_3, wDE_3, wDI_3,
                                      wIE_3, wII_3, IE_3, ID_3, II_3)
    
    drE5_1, drD5_1, drI5_1 = drdt_EDI(rE_5, rD_5, rI_5, leak_5[0], leak_5[1], leak_5[2], tci_5[0], tci_5[1], tci_5[2],
                                      rheo_5[0], rheo_5[1], rheo_5[2], p_5[0], p_5[1], p_5[2], wEE_5, wED_5, wEI_5, wDE_5, wDI_5,
                                      wIE_5, wII_5, IE_5, ID_5, II_5)    

    rE_0[:] += dt * drE0_1
    rI_0[:] += dt * drI0_1
    rE_1[:] += dt * drE1_1
    rD_1[:] += dt * drD1_1
    rI_1[:] += dt * drI1_1
    rE_2[:] += dt * drE2_1
    rI_2[:] += dt * drI2_1
    rE_3[:] += dt * drE3_1
    rD_3[:] += dt * drD3_1
    rI_3[:] += dt * drI3_1
    rE_4[:] += dt * drE4_1
    rI_4[:] += dt * drI4_1
    rE_5[:] += dt * drE5_1
    rD_5[:] += dt * drD5_1
    rI_5[:] += dt * drI5_1
    rE_6[:] += dt * drE6_1
    rI_6[:] += dt * drI6_1
    
    drE0_2, drI0_2 = drdt_EI(rE_0, rI_0, leak_0[0], leak_0[1], tci_0[0], tci_0[1], 
                             rheo_0[0], rheo_0[1], p_0[0], p_0[1], wEE_0, wEI_0, wIE_0, wII_0, IE_0, II_0)
    
    drE2_2, drI2_2 = drdt_EI(rE_2, rI_2, leak_2[0], leak_2[1], tci_2[0], tci_2[1], 
                             rheo_2[0], rheo_2[1], p_2[0], p_2[1], wEE_2, wEI_2, wIE_2, wII_2, IE_2, II_2)
    
    drE4_2, drI4_2 = drdt_EI(rE_4, rE_4, leak_4[0], leak_4[1], tci_4[0], tci_4[1], 
                             rheo_4[0], rheo_4[1], p_4[0], p_4[1], wEE_4, wEI_4, wIE_4, wII_4, IE_4, II_4)
    
    drE6_2, drI6_2 = drdt_EI(rE_6, rI_6, leak_6[0], leak_6[1], tci_6[0], tci_6[1], 
                             rheo_6[0], rheo_6[1], p_6[0], p_6[1], wEE_6, wEI_6, wIE_6, wII_6, IE_6, II_6)
    
    drE1_2, drD1_2, drI1_2 = drdt_EDI(rE_1, rD_1, rI_1, leak_1[0], leak_1[1], leak_1[2], tci_1[0], tci_1[1], tci_1[2],
                                      rheo_1[0], rheo_1[1], rheo_1[2], p_1[0], p_1[1], p_1[2], wEE_1, wED_1, wEI_1, wDE_1, wDI_1,
                                      wIE_1, wII_1, IE_1, ID_1, II_1)
    
    drE3_2, drD3_2, drI3_2 = drdt_EDI(rE_3, rD_3, rI_3, leak_3[0], leak_3[1], leak_3[2], tci_3[0], tci_3[1], tci_3[2],
                                      rheo_3[0], rheo_3[1], rheo_3[2], p_3[0], p_3[1], p_3[2], wEE_3, wED_3, wEI_3, wDE_3, wDI_3,
                                      wIE_3, wII_3, IE_3, ID_3, II_3)
    
    drE5_2, drD5_2, drI5_2 = drdt_EDI(rE_5, rD_5, rI_5, leak_5[0], leak_5[1], leak_5[2], tci_5[0], tci_5[1], tci_5[2],
                                      rheo_5[0], rheo_5[1], rheo_5[2], p_5[0], p_5[1], p_5[2], wEE_5, wED_5, wEI_5, wDE_5, wDI_5,
                                      wIE_5, wII_5, IE_5, ID_5, II_5)

    rE_0[:] += dt/2 * (drE0_1 + drE0_2)
    rI_0[:] += dt/2 * (drI0_1 + drI0_2)
    rE_1[:] += dt/2 * (drE1_1 + drE1_2)
    rD_1[:] += dt/2 * (drD1_1 + drD1_2)
    rI_1[:] += dt/2 * (drI1_1 + drI1_2)
    rE_2[:] += dt/2 * (drE2_1 + drE2_2)
    rI_2[:] += dt/2 * (drI2_1 + drI2_2)
    rE_3[:] += dt/2 * (drE3_1 + drE3_2)
    rD_3[:] += dt/2 * (drD3_1 + drD3_2)
    rI_3[:] += dt/2 * (drI3_1 + drI3_2)
    rE_4[:] += dt/2 * (drE4_1 + drE4_2)
    rI_4[:] += dt/2 * (drI4_1 + drI4_2)
    rE_5[:] += dt/2 * (drE5_1 + drE5_2)
    rD_5[:] += dt/2 * (drD5_1 + drD5_2)
    rI_5[:] += dt/2 * (drI5_1 + drI5_2)
    rE_6[:] += dt/2 * (drE6_1 + drE6_2)
    rI_6[:] += dt/2 * (drI6_1 + drI6_2)

    rE_0[rE_0<0] = 0
    rI_0[rI_0<0] = 0
    rE_1[rE_1<0] = 0
    rD_1[rD_1<0] = 0
    rI_1[rI_1<0] = 0
    rE_2[rE_2<0] = 0
    rI_2[rI_2<0] = 0
    rE_3[rE_3<0] = 0
    rD_3[rD_3<0] = 0
    rI_3[rI_3<0] = 0
    rE_4[rE_4<0] = 0
    rI_4[rI_4<0] = 0
    rE_5[rE_5<0] = 0
    rD_5[rD_5<0] = 0
    rI_5[rI_5<0] = 0
    rE_6[rE_6<0] = 0
    rI_6[rI_6<0] = 0

    return 


def Inputs_ff_and_fb(stimulus, rE_0, rE_1, rE_2, rE_3, rE_4, rE_5, rE_6, vES_0, vIS_0, vES_1, vDS_1, vIS_1, vEE_10, vDE_10, vIE_10,
                     vEE_21, vIE_21, vEE_32, vDE_32, vIE_32, vEE_34, vDE_34, vIE_34, vEE_43, vIE_43, vEE_50, vDE_50, vIE_50, vEE_56,
                     vDE_56, vIE_56, vEE_65, vIE_65, XE_0, XI_0, XE_1, XD_1, XI_1, XE_2, XI_2, XE_3, XD_3, XI_3, XE_4, XI_4, XE_5,
                     XD_5, XI_5, XE_6, XI_6):
    
    # region 0
    In_E0 = vES_0 * stimulus + XE_0
    In_I0 = vIS_0 * stimulus + XI_0
    
    # region 1
    In_E1 = vEE_10 @ rE_0 + vES_1 * stimulus + XE_1
    In_D1 = vDE_10 @ rE_0 + vDS_1 * stimulus + XD_1
    In_I1 = vIE_10 @ rE_0 + vIS_1 * stimulus + XI_1
    
    # region 2
    In_E2 = vEE_21 @ rE_1 + XE_2
    In_I2 = vIE_21 @ rE_1 + XI_2
    
    # region 3
    In_E3 = vEE_32 @ rE_2 + vEE_34 @ rE_4 + XE_3
    In_D3 = vDE_32 @ rE_2 + vDE_34 @ rE_4 + XD_3
    In_I3 = vIE_32 @ rE_2 + vIE_34 @ rE_4 + XI_3
    
    # region 4
    In_E4 = vEE_43 @ rE_3 + XE_4
    In_I4 = vIE_43 @ rE_3 + XI_4
    
    # region 5
    In_E5 = vEE_50 @ rE_0 + vEE_56 @ rE_6 + XE_5
    In_D5 = vDE_50 @ rE_0 + vDE_56 @ rE_6 + XD_5
    In_I5 = vIE_50 @ rE_0 + vIE_56 @ rE_6 + XI_5
    
    # region 6
    In_E6 = vEE_65 @ rE_5 + XE_6
    In_I6 = vIE_65 @ rE_5 + XI_6
    
    return [In_E0, In_I0, In_E1, In_D1, In_I1, In_E2, In_I2, In_E3, In_D3, In_I3, In_E4, In_I4, In_E5, In_D5, In_I5, In_E6, In_I6]


def save_rates(save_pop, counter, save_id, rE_0, rI_0, rE_1, rD_1, rI_1, rE_2, rI_2, rE_3, rD_3, rI_3, rE_4, rI_4, rE_5, rD_5, rI_5, rE_6, rI_6, hdf_rates):
    
    if save_pop:
        hdf_rates[counter, 0] = np.mean(rE_0)
        hdf_rates[counter, 1] = np.mean(rI_0)
        hdf_rates[counter, 2] = np.mean(rE_1)
        hdf_rates[counter, 3] = np.mean(rD_1)
        hdf_rates[counter, 4] = np.mean(rI_1)
        hdf_rates[counter, 5] = np.mean(rE_2)
        hdf_rates[counter, 6] = np.mean(rI_2)
        hdf_rates[counter, 7] = np.mean(rE_3)
        hdf_rates[counter, 8] = np.mean(rD_3)
        hdf_rates[counter, 9] = np.mean(rI_3)
        hdf_rates[counter, 10] = np.mean(rE_4)
        hdf_rates[counter, 11] = np.mean(rI_4)
        hdf_rates[counter, 12] = np.mean(rE_5)
        hdf_rates[counter, 13] = np.mean(rD_5)
        hdf_rates[counter, 14] = np.mean(rI_5)
        hdf_rates[counter, 15] = np.mean(rE_6)
        hdf_rates[counter, 16] = np.mean(rI_6)
         
    else:
        hdf_rates[counter, 0] = rE_0[save_id[0]]
        hdf_rates[counter, 1] = rI_0[save_id[1]]
        hdf_rates[counter, 2] = rE_1[save_id[2]]
        hdf_rates[counter, 3] = rD_1[save_id[3]]
        hdf_rates[counter, 4] = rI_1[save_id[4]]
        hdf_rates[counter, 5] = rE_2[save_id[5]]
        hdf_rates[counter, 6] = rI_2[save_id[6]]
        hdf_rates[counter, 7] = rE_3[save_id[7]]
        hdf_rates[counter, 8] = rD_3[save_id[8]]
        hdf_rates[counter, 9] = rI_3[save_id[9]]
        hdf_rates[counter, 10] = rE_4[save_id[10]]
        hdf_rates[counter, 11] = rI_4[save_id[11]]
        hdf_rates[counter, 12] = rE_5[save_id[12]]
        hdf_rates[counter, 13] = rD_5[save_id[13]]
        hdf_rates[counter, 14] = rI_5[save_id[14]]
        hdf_rates[counter, 15] = rE_6[save_id[15]]
        hdf_rates[counter, 16] = rI_6[save_id[16]]
    
    return 


def RunStaticNetwork(NeuPar, ConnPar_within, ConPar_between, StimPar, RatePar, SavePar, folder: str, fln: str = ''):
    
    ### Neuron parameters
    num_cells_0 = NeuPar.num_cells_0
    num_cells_1 = NeuPar.num_cells_1
    num_cells_2 = NeuPar.num_cells_2 
    num_cells_3 = NeuPar.num_cells_3
    num_cells_4 = NeuPar.num_cells_4
    num_cells_5 = NeuPar.num_cells_5
    num_cells_6 = NeuPar.num_cells_6
    
    tci_0 = NeuPar.tci_0
    tci_1 = NeuPar.tci_1
    tci_2 = NeuPar.tci_2
    tci_3 = NeuPar.tci_3
    tci_4 = NeuPar.tci_4
    tci_5 = NeuPar.tci_5
    tci_6 = NeuPar.tci_6
    
    leak_0 = NeuPar.leak_0
    leak_1 = NeuPar.leak_1
    leak_2 = NeuPar.leak_2
    leak_3 = NeuPar.leak_3
    leak_4 = NeuPar.leak_4
    leak_5 = NeuPar.leak_5
    leak_6 = NeuPar.leak_6
    
    rheo_0 = NeuPar.rheo_0
    rheo_1 = NeuPar.rheo_1
    rheo_2 = NeuPar.rheo_2
    rheo_3 = NeuPar.rheo_3
    rheo_4 = NeuPar.rheo_4
    rheo_5 = NeuPar.rheo_5
    rheo_6 = NeuPar.rheo_6
    
    p_0 = NeuPar.p_0
    p_1 = NeuPar.p_1
    p_2 = NeuPar.p_2
    p_3 = NeuPar.p_3
    p_4 = NeuPar.p_4
    p_5 = NeuPar.p_5
    p_6 = NeuPar.p_6
    
    ### Connection parameters - within regions
    wEE_0 = ConnPar_within.wEE_0
    wEI_0 = ConnPar_within.wEI_0
    wIE_0 = ConnPar_within.wIE_0
    wII_0 = ConnPar_within.wII_0
    
    wEE_1 = ConnPar_within.wEE_1
    wED_1 = ConnPar_within.wED_1
    wEI_1 = ConnPar_within.wEI_1
    wDE_1 = ConnPar_within.wDE_1
    wDI_1 = ConnPar_within.wDI_1
    wIE_1 = ConnPar_within.wIE_1
    wII_1 = ConnPar_within.wII_1
    
    wEE_2 = ConnPar_within.wEE_2
    wEI_2 = ConnPar_within.wEI_2
    wIE_2 = ConnPar_within.wIE_2
    wII_2 = ConnPar_within.wII_2
    
    wEE_3 = ConnPar_within.wEE_3
    wED_3 = ConnPar_within.wED_3
    wEI_3 = ConnPar_within.wEI_3
    wDE_3 = ConnPar_within.wDE_3
    wDI_3 = ConnPar_within.wDI_3
    wIE_3 = ConnPar_within.wIE_3
    wII_3 = ConnPar_within.wII_3
    
    wEE_4 = ConnPar_within.wEE_4
    wEI_4 = ConnPar_within.wEI_4
    wIE_4 = ConnPar_within.wIE_4
    wII_4 = ConnPar_within.wII_4
    
    wEE_5 = ConnPar_within.wEE_5
    wED_5 = ConnPar_within.wED_5
    wEI_5 = ConnPar_within.wEI_5
    wDE_5 = ConnPar_within.wDE_5
    wDI_5 = ConnPar_within.wDI_5
    wIE_5 = ConnPar_within.wIE_5
    wII_5 = ConnPar_within.wII_5
    
    wEE_6 = ConnPar_within.wEE_6
    wEI_6 = ConnPar_within.wEI_6
    wIE_6 = ConnPar_within.wIE_6
    wII_6 = ConnPar_within.wII_6
    
    
    ### Connection parameters - between regions
    vES_0 = ConPar_between.vES_0
    vIS_0 = ConPar_between.vIS_0 
    vES_1 = ConPar_between.vES_1 
    vDS_1 = ConPar_between.vDS_1 
    vIS_1 = ConPar_between.vIS_1
    vEE_10 = ConPar_between.vEE_10 
    vDE_10 = ConPar_between.vDE_10 
    vIE_10 = ConPar_between.vIE_10 
    vEE_21 = ConPar_between.vEE_21 
    vIE_21 = ConPar_between.vIE_21 
    vEE_32 = ConPar_between.vEE_32 
    vDE_32 = ConPar_between.vDE_32 
    vIE_32 = ConPar_between.vIE_32 
    vEE_34 = ConPar_between.vEE_34 
    vDE_34 = ConPar_between.vDE_34 
    vIE_34 = ConPar_between.vIE_34 
    vEE_43 = ConPar_between.vEE_43 
    vIE_43 = ConPar_between.vIE_43 
    vEE_50 = ConPar_between.vEE_50 
    vDE_50 = ConPar_between.vDE_50 
    vIE_50 = ConPar_between.vIE_50 
    vEE_56 = ConPar_between.vEE_56 
    vDE_56 = ConPar_between.vDE_56 
    vIE_56 = ConPar_between.vIE_56 
    vEE_65 = ConPar_between.vEE_65 
    vIE_65 = ConPar_between.vIE_65
    
    ### Rate (initial) parameters
    rE_0 = RatePar.rE_0
    rI_0 = RatePar.rI_0
    rE_1 = RatePar.rE_1
    rD_1 = RatePar.rD_1
    rI_1 = RatePar.rI_1
    rE_2 = RatePar.rE_2
    rI_2 = RatePar.rI_2
    rE_3 = RatePar.rE_3
    rD_3 = RatePar.rD_3
    rI_3 = RatePar.rI_3
    rE_4 = RatePar.rE_4
    rI_4 = RatePar.rI_4
    rE_5 = RatePar.rE_5
    rD_5 = RatePar.rD_5
    rI_5 = RatePar.rI_5
    rE_6 = RatePar.rE_6
    rI_6 = RatePar.rI_6
    
    ### Stimulation parameters
    sensory_input_mean_per_trial = StimPar.sensory_input_mean_per_trial
    sensory_input_sd_per_trial = StimPar.sensory_input_sd_per_trial
    num_time_steps_per_trial = StimPar.num_time_steps_per_trial
    NStim = len(sensory_input_mean_per_trial)
    dt = StimPar.dt
    
    ### Background input to each neuron
    XE_0 = StimPar.XE_0
    XI_0 = StimPar.XI_0
    XE_1 = StimPar.XE_1 
    XD_1 = StimPar.XD_1 
    XI_1 = StimPar.XI_1 
    XE_2 = StimPar.XE_2 
    XI_2 = StimPar.XI_2 
    XE_3 = StimPar.XE_3 
    XD_3 = StimPar.XD_3 
    XI_3 = StimPar.XI_3 
    XE_4 = StimPar.XE_4 
    XI_4 = StimPar.XI_4 
    XE_5 = StimPar.XE_5
    XD_5 = StimPar.XD_5 
    XI_5 = StimPar.XI_5 
    XE_6 = StimPar.XE_6 
    XI_6 = StimPar.XI_6
    
    ### save parameters
    counter = 0
    save_every_n_steps = SavePar.save_every_n_steps
    save_pop = SavePar.save_pop
    save_id = SavePar.save_id
    
    ### Path & file for data to be stored
    path = 'Results/Data/' + folder
    if not os.path.exists(path):
        os.mkdir(path)

    # fp = open(path +'/Data_StaticNetwork_' + fln + '.dat','w')
    hdf = tables.open_file(path + '/Data_StaticNetwork_' + fln + '.hdf', 'w')
    atom = tables.Float32Atom()
    hdf_rates = hdf.create_carray(hdf.root, 'rates', atom, (np.int32(np.ceil(num_time_steps_per_trial * NStim / save_every_n_steps)),17))

    ### Main loop
    for s in range(NStim):
    
        stimulus_in_trial = np.random.normal(sensory_input_mean_per_trial[s], sensory_input_sd_per_trial[s], size=num_time_steps_per_trial)  
    
        for tstep in range(num_time_steps_per_trial):
            
            # feedforward and feedback input to neurons in the different "regions"
            [In_E0, In_I0, In_E1, In_D1, In_I1, In_E2, In_I2, In_E3, In_D3, In_I3, In_E4, In_I4, 
             In_E5, In_D5, In_I5, In_E6, In_I6] = Inputs_ff_and_fb(stimulus_in_trial, rE_0, rE_1, rE_2, rE_3, rE_4, rE_5, rE_6, vES_0, 
                                                                   vIS_0, vES_1, vDS_1, vIS_1, vEE_10, vDE_10, vIE_10, vEE_21, vIE_21, 
                                                                   vEE_32, vDE_32, vIE_32, vEE_34, vDE_34, vIE_34, vEE_43, vIE_43, 
                                                                   vEE_50, vDE_50, vIE_50, vEE_56, vDE_56, vIE_56, vEE_65, vIE_65, 
                                                                   XE_0, XI_0, XE_1, XD_1, XI_1, XE_2, XI_2, XE_3, XD_3, XI_3, XE_4, 
                                                                   XI_4, XE_5, XD_5, XI_5, XE_6, XI_6)
            
            # neuron dynamics
            RateDynamics(rE_0, rI_0, rE_1, rD_1, rI_1, rE_2, rI_2, rE_3, rD_3, rI_3, rE_4, rI_4, rE_5, rD_5, rI_5, rE_6, rI_6,
                         leak_0, leak_1, leak_2, leak_3, leak_4, leak_5, leak_6, tci_0, tci_1, tci_2, tci_3, tci_4, tci_5, tci_6,
                         rheo_0, rheo_1, rheo_2, rheo_3, rheo_4, rheo_5, rheo_6, p_0, p_1, p_2, p_3, p_4, p_5, p_6, wEE_0, wEI_0,
                         wIE_0, wII_0, wEE_1, wED_1, wEI_1, wDE_1, wDI_1, wIE_1, wII_1, wEE_2, wEI_2, wIE_2, wII_2, wEE_3, wED_3,
                         wEI_3, wDE_3, wDI_3, wIE_3, wII_3, wEE_4, wEI_4, wIE_4, wII_4, wEE_5, wED_5, wEI_5, wDE_5, wDI_5, wIE_5, 
                         wII_5, wEE_6, wEI_6, wIE_6, wII_6, In_E0, In_I0, In_E1, In_D1, In_I1, In_E2, In_I2, In_E3, In_D3, In_I3, In_E4, In_I4, 
                         In_E5, In_D5, In_I5, In_E6, In_I6, dt)
            
            # save neuron dynamics in file
            if ((s * num_time_steps_per_trial + tstep) % save_every_n_steps == 0):
                save_rates(save_pop, counter, save_id, rE_0, rI_0, rE_1, rD_1, rI_1, rE_2, rI_2, rE_3, rD_3, rI_3, rE_4, rI_4, rE_5, rD_5, rI_5, rE_6, rI_6, hdf_rates)
                counter += 1
                hdf.flush()

    return
            