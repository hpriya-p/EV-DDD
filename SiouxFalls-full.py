#!/usr/bin/env python
# coding: utf-8

# In[3]:


import networkx as nx
import pandas as pd
from Instance_v2 import *


# # Time DDD on Sioux Falls network
# 
# We will include all charge levels, but gradually granularize time via DDD

# ## Units of original dataset
# - Time: 0.01 hrs
# - Length: set to free flow travel time; will assume 60 km/hr here 

# In[ ]:


df = pd.read_csv("data/SiouxFalls_net.tntp", sep='\t', lineterminator='\n')
N = nx.DiGraph()
for ind, row in df.iterrows():

    t = row['free_flow_time']
    print(t)
    dist = int(t * 0.01 * 60) # in km
    dist = int(dist/5 * 100) # convert to percent of tank
    N.add_edge(row['init_node'], row['term_node'], dH=dist, dL=int(dist/2), time=t)


# In[5]:


parameters = dict()
parameters['source'] = 1
parameters['sink'] = 24
parameters['T'] = 20 # 1 hour
parameters['L'] = 100
parameters['charge_rate'] = dict((i, 2) for i in N.nodes)
parameters['battery_nodes'] = []
parameters['tractor_nodes'] = [i for i in N.nodes]
parameters['charge_nodes'] = []
parameters['bat_swap_time'] = 1
parameters['tr_swap_time'] = 2
parameters['charge_cost'] = dict(((i, t), 299) for i in N.nodes for t in range(parameters['T']))
parameters['surplus_cost'] = dict((i, 25) for i in N.nodes)
parameters['stat_cost'] = dict((i, 100) for i in N.nodes)
parameters['rec_penalty'] = 1
parameters['step_size'] = 5
parameters['MAX_ITER'] = 1000
parameters['speed_curve'] ={0: {'speed':1, 'minbat': 0, 'maxbat': 50}, 1: {'speed': 1, 'minbat': 50, 'maxbat': 100}}
parameters['N_tractors'] = 2
parameters['N_chargers'] = 0
parameters['N_batteries'] = 0
# In[6]:


I = Instance(N, parameters, config='heuristic')
I.model.display()
print("constructed instance")


# In[ ]:


result = I.run_DDD()


# In[ ]:


print(result)


# In[ ]:




