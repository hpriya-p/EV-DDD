# %%
import folium
from datetime import datetime
import pandas as pd
import numpy as np
import tqdm 
from matplotlib import pyplot as plt
import config
import pickle
import networkx as nx
import requests
import time 
from ast import literal_eval
from joblib import Parallel, delayed
from routingpy.routers import get_router_by_name


# %%
router = get_router_by_name("osrm")(base_url="http://router.project-osrm.org")

# %%
GENERATE_HEATMAP_EDGES = False
GENERATE_HEATMAP_PATHS = True
COMPUTE_DISTANCE_MATRIX = True 
COMPUTE_STATION_DF = False 

# %% [markdown]
# # Query information from Open Street Maps API

# %%
def query_wrapper(orig, dest, invert=True):
    if invert:
        return router.directions([(orig[1], orig[0]), (dest[1], dest[0])], 'driving')
    else:
        return router.directions(locations=[orig, dest], profile='driving')
    

# %%
query_wrapper((42.393167, -71.064352),(40.718037, -73.932309))

# %%
def bat_consumption(dist, time, start_elev_m, end_elev_m, start_speed, end_speed):
    # To compute the reduction in SOC from this output: if X (Joules) returned,
    # X * (2.7778 x 10^-7 J/kwh)/(battery capacity in kwh)
    
    m = 24493.988 #kg
    Af = 3.825 #m^2
    Cpi = 0.011
    Cd = 0.9
    nt = .9 # transmission efficiency of battery
    nmd = .85 # electrical machine efficiency, also called motor efficiency
    nr = .05 # regenerative braking efficiency
    rho = 1.225 #kg/m
    g = 9.8 #m/s^2
    V = dist/time
    dVdt = (end_speed - start_speed)/time
    alpha = (end_elev_m - start_elev_m)/dist # all units in meters
    nwh = 1
    
    W_acc = 0 # accessory load
    
    W_tract = m * V * dVdt + .5 * Cd * Af * rho * V**3 + m * g * V * Cpi +  m * g * V * np.sin(alpha)
    
    if W_tract > 0:
        W_dis = (W_tract + W_acc)/(nmd * nt)
    else:
        W_dis = W_acc

    if W_tract < 0:
        W_chg = - 1* W_acc + abs(W_tract)/nwh * nmd * nt
    else:
        W_chg = 0
    
    return (W_dis - W_chg) * time
    

def get_elevation(locations, starting_j=0):
    """
    Returns elevation of list of locations (each location is a tuple (lat, long))
    
    starting_j allows for a subset of this list to be queried (primarily used for recovery if code crashes in-between)
    """
    elevation_dict = dict()
    N = len(locations)
    for j in tqdm.tqdm(range(starting_j, N)): 
        x = locations[j]
        req_str = 'https://api.open-elevation.com/api/v1/lookup?locations='
        req_str += str(x[0]) +"," +  str(x[1])
        
        r = requests.get(req_str, timeout=3600)

        if r.status_code == 200: # successful query
            for x in r.json()['results']:
                elevation_dict[(np.round(x['latitude'], 6), np.round(x['longitude'], 6))] = x['elevation']
        
        elif r.status_code == 504: # likely too many queries were made in a short period of time
            print("sleeping")
            time.sleep(30)
            r = requests.get(req_str, timeout=3600)
                             
        else: # some other error has occured
            print(r.content)
            return elevation_dict, j

    return elevation_dict

def get_data(orig, dest, elev):
    try:
        P = query_wrapper(orig, dest)
        time.sleep(1)
    except:
        return np.inf, np.inf, np.inf
    
    # Code to approximate starting and ending speed (used for battery consumption computation)
    j = 0
    dur = 0
    while dur == 0:
        j += 1
        first_step = query_wrapper(P.geometry[0], P.geometry[j], invert=False)
        time.sleep(1)
        dur = first_step.duration
    j = 0
    dur = 0
    while dur == 0:
        j += 1
        last_step = query_wrapper(P.geometry[0 - j - 1], P.geometry[-1], invert=False)
        dur = last_step.duration
    v0 = first_step.distance/first_step.duration
    v1 = last_step.distance/last_step.duration
    
    
    return P.distance, P.duration, bat_consumption(P.distance, P.duration, elev[orig], elev[dest], v0, v1)
    



# %%


# %% [markdown]
# # Filtering charging station dataset

# %%
stations = pd.read_csv("data/all_fuel_stations.csv")[['Latitude', 'Longitude']].values.tolist()

# %% [markdown]
# # Computing Distance Matrix 

 
elev_dict = get_elevation(stations, 0)

# %%
n = len(stations)
def process(i): 
        try:
            x = (np.round(stations[i][0], 6), np.round(stations[i][1], 6))
            nbrs = [j for j in range(n) if abs(x[0] - stations[j][0]) <= 4 and abs(x[1] - stations[j][1]) <= 5]
            res = []
            for j in nbrs:
                if i == j:
                    continue
                y = (np.round(stations[j][0], 6), np.round(stations[j][1], 6))
                val = get_data(x, y, elev_dict)
                res.append({'pt1': x, 'pt2': y, 'dist': val[0], 'time': val[1], 'bat': val[2]})
            return res 
        except:
            print("ERROR: ", i)
            return None

# Use joblib.Parallel with threading backend since tasks are IO-bound
results = Parallel(n_jobs=4, backend='threading')(delayed(process)(i) for i in tqdm.tqdm(range(n)))
records = [x for x in results if x is not None]
pd.DataFrame(records).to_csv("D_parallel_out.csv")

