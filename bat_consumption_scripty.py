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
COMPUTE_DISTANCE_MATRIX = False 
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
    # To compute the reduction in SOC from this output: if X returned,
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

# %%
with open("data/pickled/v2_elev.pickle", 'rb') as file:
    elev_dict = pickle.load(file)

#elev_dict = get_elevation(stations, 0)

# %%
if COMPUTE_DISTANCE_MATRIX:
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

    records = [process(i) for i in tqdm.tqdm(range(n))]
    records = [x for x in records if x is not None]
    pd.DataFrame(records).to_csv("D_parallel_out.csv")


# %% [markdown]
# # Plotting Heatmap of Power Demand

# %%
MAX_LAT = max([x[0] for x in stations_to_keep])
MIN_LAT = min([x[0] for x in stations_to_keep])
MAX_LNG = max([x[1] for x in stations_to_keep])
MIN_LNG = min([x[1] for x in stations_to_keep])

# %%
def c(x):
    """
    Explanation: To convert to kw, we divide bat_consumption (in J/s) by 100. 
    
    Ideally, (bat consumption) = [a fixed constant (2880)] * (time); while this doesn't hold exactly,
    the following heatmap shows that it holds within a multiplicative factor
  
    
    """ 
    kw = x/(1000)
    y = kw/2880
    if 0 <= y < 1:
        return "#33FFFF"
    elif 1 <= y < 10:
        return "#33C4FF"
    elif 10 <= y < 20:
        return "#3399FF"
    elif 20 <= y <= 30:
        return "#7300e6"
    elif y > 30:
        return "#1ced38"
    else:
        return "#FF5B33"
    
fmap = folium.Map(location = ((config.MAX_LAT + config.MIN_LAT)/2, (config.MAX_LNG + config.MIN_LNG)/2), zoom_start=5)

if GENERATE_HEATMAP_EDGES:    
    bat_consumptions = []

    n = len(stations_to_keep)


    stations_to_keep = [tuple(x) for x in stations_to_keep]
    for i in range(n):
        for j in tqdm.tqdm(range(i+1, n)):
            x = stations_to_keep[i]
            y = stations_to_keep[j]

            P = D[x][y]
            if P == np.inf:
                continue
            lenP = len(P['edges'])
            for r in range(lenP):
                e = P['edges'][r]
                start = P['nodes'][r]
                end = P['nodes'][r + 1]

                snapped_path=gmaps.snap_to_roads([(start['lat'], start['lng']), (end['lat'], end['lng'])], interpolate=True)
                to_plot = [(start['lat'], start['lng'])] + [(x['location']['latitude'], x['location']['longitude']) for x in snapped_path] + [ (end['lat'], end['lng'])]

                bat = bat_consumption(e[2]['speed'], start['elev'], end['elev'], e[2]['dist'] * 1000)
                bat_consumptions.append(bat/e[2]['time'])
                folium.PolyLine(locations=to_plot, color=c(bat)).add_to(fmap)
        fmap.save("heatmap.html")
            
fmap

# %%
fmap

# %%
csv = pd.read_csv("D_parallel_out.csv")
csv = csv[[str(i) for i in range(147)]]
csv = csv.replace(np.NaN, None)

# %%
records = []
inf = np.inf 
for __, row in csv.iterrows():
    for x in row:
        if x is not None:
            x = x.replace('inf', 'None')
            try:
                records.append(literal_eval(x))
            except:
                print(x)
                break
df = pd.DataFrame(records)

# %%
print(df)
df.to_csv("data/D_clean_Jun12.csv")

# %%
LA = (34.01665, -118.208679)
NW = (40.67626, -74.24808)

# %%
from ast import literal_eval
fmap = folium.Map(location = ((config.MAX_LAT + config.MIN_LAT)/2, (config.MAX_LNG + config.MIN_LNG)/2), zoom_start=5)

if GENERATE_HEATMAP_PATHS:
    G = nx.DiGraph()
    for ind, row in df.iterrows():
        x = row['pt1']
        y = row['pt2'] 
        G.add_edge(x, y, dist=row['dist'], time=row['time'], bat=row['bat'])
             


    for x in nx.shortest_path(G, LA, NW, weight='bat'):
        for y in G.neighbors(x):
            print(y)
            try:
                SP = nx.shortest_path(G, x, y, weight="bat") # shortest charge path
                if sum(G[SP[i]][SP[i+1]]['time'] for i in range(len(SP)-1)) <= 1.10 * nx.shortest_path_length(G, x, y, weight="time"):
                    #folium.PolyLine(locations=nx.shortest_path(G, x, y, weight="time"), color="gray").add_to(fmap)
                    continue
                else:
                    folium.PolyLine(locations=nx.shortest_path(G, x, y, weight="time"), dash_array='10', color="red").add_to(fmap)
                    folium.PolyLine(locations=SP, color="#3399FF", dash_array='17').add_to(fmap)
            except:
                continue
 

        fmap.save("heatmap_path.html")
            
fmap

# %%
fmap

# %% [markdown]
# # Computing Station_DF

# %%
electric_stations = pd.read_csv("fuel_station_locations.csv")[['Latitude', 'Longitude']].values.tolist()

def snap_to_stations(pt):
    candidate_stat =  [x for x in electric_stations if abs(x[0] - pt[0]) <= 1 and abs(x[1] - pt[1]) <= 1]
    if len(candidate_stat) == 0:
        return None, np.inf
    closest = None
    closest_val = np.inf
    for x in candidate_stat:
        dist = query_wrapper(x, pt).distance
        if dist < closest_val:
            closest = x
            closest_val = dist
    
    return closest, closest_val

# %%
from joblib import Parallel, delayed
def process(line):
    res = []
    parts = line.strip().split(',')
    lat = float(parts[0])/(10**7)
    lng = float(parts[1])/(10**7) 
    if not (config.MIN_LAT <= lat <= config.MAX_LAT and config.MIN_LNG <= lng <= config.MAX_LNG):
        return None  

    try:
        closest_stat, dist = snap_to_stations((lat, lng))
        station_info = dict([('lat',lat ), ('lng', lng), ('total_chargers_at_stat', int(parts[2])), ('snap_to', closest_stat), ('dist_snap', dist)])

        charger_types = dict()
        i = 3
        while i < len(parts) - 1:
            charger = int(parts[i])
            if charger in charger_types:
                charger_types[charger]["n_chargers"] += 1
            else:
                charger_types[charger] = {'n_chargers': 1, 'kw': float(parts[i+1])}
            i += 2
        for t in charger_types.keys():
            node_info = dict(list(station_info.items()) + [('type', t)] + list(charger_types[t].items()))
            res.append(node_info)

    except:
        print("ERROR OCCURRED")
        res = None
    
    return res

records =  Parallel(n_jobs=10)(delayed(process)(i) for i in tqdm.tqdm(open('relevant_stat_data.txt')))

# %%
stations_df = pd.DataFrame(sum([x for x in records if x is not None], [])).dropna().reset_index(drop=True)
stations_df['index'] = stations_df.index
print(stations_df)
print(stations_df['snap_to'])
stations_df['snap_to'] = stations_df['snap_to'].apply(lambda x: tuple(x))

# %%
# Keeping only the closest points to each station
snap_dist = dict()
for stat, stat_df in stations_df.groupby('snap_to'):
    snap_dist[stat] = stat_df.sort_values(by='dist_snap').iloc[0]
rows = [v['index'] for k,v in snap_dist.items()] # if v['dist_snap'] <= 20]
stations_df = stations_df.loc[rows]
print(stations_df.shape)

# %%
stations_df

# %%
stations_df.to_csv("processed.csv", index=True)

# %%
snap_to_stations((34.017194, -118.263477)) #LA

# %%
snap_to_stations((40.738932, -74.175708)) #Newark


