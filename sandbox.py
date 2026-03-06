#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import pandas as pd
from Instance_v2 import *

N = nx.DiGraph()
for i in range(5):
    j = i + 1
    N.add_edge(i, j, dH=2, dL=1, time=1)




# In[5]:


parameters = dict()
parameters['source'] = 1
parameters['sink'] = 5
parameters['T'] = 8
parameters['L'] = 5
parameters['D'] = 1
parameters['charge_rate'] = dict((i, 1) for i in N.nodes)
parameters['battery_nodes'] = [] #list(N.nodes) [] #list(N.nodes)#[3, 5, 7, 9]
parameters['tractor_nodes'] = list(N.nodes) #[]
parameters['swap_nodes'] =  [] #[1] + list(range(0, 24, 2))
parameters['charge_nodes'] = [] #[i for i in N.nodes if i not in parameters['mobile_nodes']] #list(range(11, 24, 2))
parameters['mobile_charge_rate'] = 5
parameters['bat_swap_time'] = 1
parameters['tr_swap_time'] = 1
parameters['charge_cost'] = dict(((i, t), 2) for i in N.nodes for t in range(parameters['T']))
parameters['surplus_cost'] = dict((i, 0) for i in N.nodes)
parameters['stat_cost'] = dict((i, 50) for i in N.nodes)
parameters['step_size'] = 5
parameters['MAX_ITER'] = 20
parameters['speed_curve'] ={0: {'speed': 60 * 1/(35*3.6), 'minbat': 0, 'maxbat': 11}, \
                   1: {'speed': 60 * 1/(25*3.6), 'minbat': 11, 'maxbat': 41}, \
                   2: {'speed': 60 * 1/(30*3.6), 'minbat': 41, 'maxbat': 61}, \
                   3: {'speed': 60 * 1/(45*3.6), 'minbat': 61, 'maxbat': 81}, \
                   4: {'speed': 60 * 1/(80*3.6), 'minbat': 81, 'maxbat': 91}, \
                   5: {'speed': 60 * 1/(130*3.6), 'minbat': 91, 'maxbat': 96}, \
                   6: {'speed': 60 * 1/(400 * 3.6), 'minbat': 96, 'maxbat': 100}}

# In[6]:



# initialize
new_path, edge_types, __ = sub.astar_search((parameters['source'], parameters['L'], 0),lambda x: x[0] == parameters['sink'] and x[2] == parameters['T'] - 1, lambda x: 0, lambda x: 0)
I = Instance(N, parameters, new_path, edge_types)
I.solve()