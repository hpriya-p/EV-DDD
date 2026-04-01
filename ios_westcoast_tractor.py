import time
import pandas as pd
import networkx as nx
import ast
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from math import radians, sin, cos, sqrt, atan2
import json
import uuid
import os
import Instance_v2

# ============================================================
# Constants
# ============================================================

# add seed
random_seed = 1

np.random.seed(random_seed)

PROCESSED_STATIONS_DF = "data/processed/stations_data.csv"
LB = 0
# bounding box
MAX_LAT = 49
MIN_LAT = 32
MAX_LNG = -67
MIN_LNG = -125

# West Coast bounding box
WEST_COAST_MIN_LAT = 34
WEST_COAST_MAX_LAT = 39
WEST_COAST_MIN_LNG = -125
WEST_COAST_MAX_LNG = -115

# Truck properties
RANGE = 190 # miles
 

# for a 100kwh station,
speed_curve ={0: {'speed': 3600 * 1/(35*3.6), 'minbat': 0, 'maxbat': 10},
                   1: {'speed': 3600 * 1/(25*3.6), 'minbat': 10, 'maxbat': 40},
                   2: {'speed': 3600 * 1/(37.5*3.6), 'minbat': 40, 'maxbat': 80},
                   3: {'speed': 3600 * 1/(172.5*3.6), 'minbat': 80, 'maxbat': 100}}


# Other parameters
BAT_FROM_TIME = 3600 * 1000 * .75 # battery consumption = (3600 * 1000 * .8) * (time in hrs)

DISTANCE_THRESHOLD = 40  # miles
EDGE_THRESHOLD = 190  # miles - max edge length for connectivity

# Major West Coast cities (name, lat, lng)
cities = dict(
    [('Seattle', (47.6062, -122.3321)),
    ('Portland', (45.5152, -122.6784)),
    ('San Francisco', (37.7749, -122.4194)),
    ('Los Angeles', (34.0522, -118.2437)),
    ('San Diego', (32.7157, -117.1611)),
    ('Sacramento', (38.5816, -121.4944)),
    ('Las Vegas', (36.1699, -115.1398)),
    ('Phoenix', (33.4484, -112.0740)),
    ('Reno', (39.5296, -119.8138)),
    ('Eugene', (44.0521, -123.0868)),
    ('Oakland', (37.8044, -122.2712)),
    ('San Jose', (37.3382, -121.8863))]
)

# ============================================================
# Functions
# ============================================================

def haversine_miles(lat1, lng1, lat2, lng2):
    """Calculate distance in miles between two lat/lng points."""
    R = 3958.8  # Earth's radius in miles

    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c

def is_west_coast(lat, lng):
    """Check if a point is within the West Coast bounding box."""
    return (WEST_COAST_MIN_LAT <= lat <= WEST_COAST_MAX_LAT and
            WEST_COAST_MIN_LNG <= lng <= WEST_COAST_MAX_LNG)

def find_closest_node(target_lat, target_lng, node_list):
    """Find the node in node_list closest to target coordinates."""
    min_dist = float('inf')
    closest_node = None
    closest_idx = None
    for idx, node in enumerate(node_list):
        dist = haversine_miles(target_lat, target_lng, node[0], node[1])
        if dist < min_dist:
            min_dist = dist
            closest_node = node
            closest_idx = idx
    return closest_idx, closest_node, min_dist

# ============================================================
# Load data
# ============================================================

other_nodes_df = pd.read_csv("data/D_clean_Jun12.csv")
stations_df = pd.read_csv(PROCESSED_STATIONS_DF)

stations_df['snap_to'] = stations_df['snap_to'].apply(ast.literal_eval)
stations_df.reset_index(inplace=True)

stations_df.index = stations_df['index']

# Extract unique nodes from other_nodes_df (pt1 and pt2 are coordinate tuples as strings)
other_nodes_df['pt1'] = other_nodes_df['pt1'].apply(lambda x: x if isinstance(x, tuple) else ast.literal_eval(x))
other_nodes_df['pt2'] = other_nodes_df['pt2'].apply(lambda x: x if isinstance(x, tuple) else ast.literal_eval(x))

# Get unique coordinates from both pt1 and pt2
all_other_nodes = set(other_nodes_df['pt1'].tolist() + other_nodes_df['pt2'].tolist())

# Filter stations_df for West Coast
stations_west = stations_df[
    (stations_df['lat'] >= WEST_COAST_MIN_LAT) &
    (stations_df['lat'] <= WEST_COAST_MAX_LAT) &
    (stations_df['lng'] >= WEST_COAST_MIN_LNG) &
    (stations_df['lng'] <= WEST_COAST_MAX_LNG)
]

# Get station coordinates as tuples
station_coords = set(zip(stations_west['lat'], stations_west['lng']))


# ============================================================
# Find the largest subset of nodes where no pair is within 10 miles
# This is the Maximum Independent Set problem on the proximity graph
# ============================================================

# Get unique nodes from other_nodes_df (West Coast only)
west_coast_nodes = [node for node in all_other_nodes if is_west_coast(node[0], node[1])]

conflict_graph = nx.Graph()
conflict_graph.add_nodes_from(range(len(west_coast_nodes)))

for i in range(len(west_coast_nodes)):
    for j in range(i + 1, len(west_coast_nodes)):
        lat1, lng1 = west_coast_nodes[i]
        lat2, lng2 = west_coast_nodes[j]
        dist = haversine_miles(lat1, lng1, lat2, lng2)
        if dist <= DISTANCE_THRESHOLD:
            conflict_graph.add_edge(i, j)

print(f"Conflict graph: {conflict_graph.number_of_nodes()} nodes, {conflict_graph.number_of_edges()} edges")

# Find maximum independent set using networkx approximation
# (exact MIS is NP-hard, but approximation works well for this size)
independent_set_indices = nx.approximation.maximum_independent_set(conflict_graph)

# Convert indices back to coordinates
max_independent_set = [west_coast_nodes[i] for i in independent_set_indices]

print(f"\nMaximum independent set size: {len(max_independent_set)}")
print(f"(Out of {len(west_coast_nodes)} West Coast nodes)")
print(f"\nThese {len(max_independent_set)} nodes are all more than {DISTANCE_THRESHOLD} miles apart from each other:")
for node in sorted(max_independent_set):
    print(f"  {node}")

# ============================================================
# Combine stations + independent set, build graph
# ============================================================
print(len(station_coords))
all_selected_nodes = list(station_coords) + list(max_independent_set)
all_selected_nodes = list(set(all_selected_nodes))  # Remove duplicates

# Build graph connecting nodes within reasonable driving distance
N = nx.Graph()
for i, node1 in enumerate(all_selected_nodes):
    N.add_node(i, pos=node1)

for i in range(len(all_selected_nodes)):
    for j in range(i + 1, len(all_selected_nodes)):
        lat1, lng1 = all_selected_nodes[i]
        lat2, lng2 = all_selected_nodes[j]
        dist = haversine_miles(lat1, lng1, lat2, lng2)
        if dist <= EDGE_THRESHOLD:
            N.add_edge(i, j, weight=dist)

# Set charger_rate for nodes that correspond to stations
station_lookup = {(row['lat'], row['lng']): row['n_chargers'] * row['kw']
                  for _, row in stations_df.iterrows()}

station_nodes = []
for i, node in enumerate(all_selected_nodes):
    if node in station_lookup:
        N.nodes[i]['charger_rate'] = station_lookup[node]
        station_nodes.append(i)
    else:
        N.nodes[i]['charger_rate'] = 250 # new stations are 250 kwh, single charger 

TRUCK_SPEED = 65  # miles per hour
DL_FACTOR = 0.5   # dL = dH * factor (lighter load uses less battery)
for u, v, data in N.edges(data=True):
    dist = data.pop('weight', 0)
    data['dH'] = int(dist/190 * 100)  # convert to percentage of battery range
    data['dL'] = int(data['dH'] * DL_FACTOR)
    data['time'] = int(np.ceil(dist / TRUCK_SPEED * 4)) # unit: 15 minutes

# print 'charger_rate' for each node
print("Num of charging stations", len(station_nodes))
print("Num of proxy stations", len(N.nodes) - len(station_nodes))
RESULTS_DIR = 'results_ios'
NODES_FILE = os.path.join(RESULTS_DIR, 'experiment_nodes.json')
la_idx, la_node, la_dist = find_closest_node(cities['Los Angeles'][0], cities['Los Angeles'][1], all_selected_nodes)
oak_idx, oak_node, oak_dist = find_closest_node(cities['Oakland'][0], cities['Oakland'][1], all_selected_nodes)
print(la_dist, oak_dist)
driving_time_hrs = nx.shortest_path_length(N, la_idx, oak_idx, 'time')
print(f"Driving time from LA to Oakland node: {driving_time_hrs * 15:.2f} min)")

T = 4 * 7 # unit: 15 minutes; want to cover 4 hrs 
parameters = {
    'T': T, 
    'L': 100,
    'D': 500, # commodity flow = 111377/365 thousand tons per day from San Jose -> Oakland. 13 ton per truck, and 23% EV truck market share by 2030 (https://blog.ucs.org/sam-wilson/we-can-electrify-one-in-three-heavy-duty-trucks-by-2030-heres-how/#:~:text=States%20are%20taking%20the%20lead%20in%20this,even%20without%20purchase%20incentives%20(see%20graph%20below)_
    'step_size': 4,
    'MAX_ITER': 1000,
    'value_for_time': 27/4, # units are 15 minutes
    'charge_nodes':  station_nodes + [i for i in N.nodes if(i not in station_nodes) and i % 2 == 0],
    'existing_charge_nodes': station_nodes,
    'battery_nodes': [],
    'tractor_nodes': [i for i in N.nodes if i not in station_nodes and i %2 == 1],
    'mobile_nodes': [],
    'charge_rate': {i : N.nodes[i]['charger_rate']
                   for i in N.nodes},
    'charge_cost': {(i, t): 0.6 for i in N.nodes for t in range(T)},
    'stat_cost': {i: 0 for i in N.nodes},
    'surplus_cost': {i: 0 for i in N.nodes},
    'N_tractors': 100,
    'N_batteries': 0,
    'N_chargers': 0,
    'speed_curve': {0: {'speed': 67/(2.5 * 4), 'minbat': 0, 'maxbat': 100}},
    'bat_swap_time': 0,
    'tr_swap_time': 1,
    'sources': [la_idx],
    'sinks': [oak_idx]
}

print(la_idx, oak_idx)
last_run_soln = {'x_load': {((34, 100, 0, 0), (28, 35, 8, 0)): 182.0, ((28, 35, 8, 0), (20, 19, 10, 0)): 182.0, ((20, 19, 10, 0), (20, 92, 11, 0)): 82.0, ((20, 92, 11, 0), (12, 1, 22, 0)): 82.0, ((12, 1, 22, 0), (12, 1, 27, 0)): 82.0, ((34, 100, 0, 0), (6, 46, 7, 0)): 304.0, ((6, 46, 7, 0), (28, 36, 9, 0)): 249.0, ((28, 36, 9, 0), (20, 20, 11, 0)): 249.0, ((20, 20, 11, 0), (20, 92, 12, 0)): 83.0, ((20, 92, 12, 0), (12, 1, 23, 0)): 83.0, ((12, 1, 23, 0), (12, 1, 27, 0)): 83.0, ((34, 100, 0, 0), (0, 62, 5, 0)): 7.0, ((0, 62, 5, 0), (0, 93, 6, 0)): 7.0, ((0, 93, 6, 0), (0, 97, 7, 0)): 7.0, ((0, 97, 7, 0), (20, 40, 14, 0)): 14.0, ((20, 40, 14, 0), (20, 92, 15, 0)): 14.0, ((20, 92, 15, 0), (12, 1, 26, 0)): 69.0, ((12, 1, 26, 0), (12, 1, 27, 0)): 69.0, ((20, 20, 11, 0), (20, 42, 12, 0)): 83.0, ((20, 42, 12, 0), (20, 92, 13, 0)): 83.0, ((20, 92, 13, 0), (12, 1, 24, 0)): 83.0, ((12, 1, 24, 0), (12, 1, 27, 0)): 83.0, ((34, 100, 0, 0), (34, 100, 1, 0)): 7.0, ((34, 100, 1, 0), (0, 62, 6, 0)): 7.0, ((0, 62, 6, 0), (0, 97, 7, 0)): 7.0, ((20, 20, 11, 0), (20, 20, 12, 0)): 83.0, ((20, 20, 12, 0), (20, 68, 13, 0)): 83.0, ((20, 68, 13, 0), (20, 92, 14, 0)): 83.0, ((20, 92, 14, 0), (12, 1, 25, 0)): 83.0, ((12, 1, 25, 0), (12, 1, 27, 0)): 83.0, ((6, 46, 7, 0), (6, 46, 9, 0)): 55.0, ((6, 46, 9, 0), (28, 36, 11, 0)): 55.0, ((28, 36, 11, 0), (20, 20, 13, 0)): 55.0, ((20, 20, 13, 0), (20, 21, 14, 0)): 55.0, ((20, 21, 14, 0), (20, 92, 15, 0)): 55.0, ((20, 19, 10, 0), (20, 95, 11, 0)): 100.0, ((20, 95, 11, 0), (12, 4, 22, 0)): 100.0, ((12, 4, 22, 0), (12, 4, 27, 0)): 100.0}, 'x_ener': {((33, 100, 0, 0), (33, 100, 1, 0)): 100.0, ((33, 100, 1, 0), (33, 100, 3, 0)): 100.0, ((33, 100, 3, 0), (3, 93, 5, 0)): 100.0, ((3, 93, 5, 0), (29, 86, 7, 0)): 100.0, ((29, 86, 7, 0), (3, 79, 9, 0)): 100.0, ((3, 79, 9, 0), (20, 74, 11, 0)): 100.0, ((20, 74, 11, 0), (20, 74, 27, 0)): 100.0, ((34, 100, 0, 0), (28, 35, 8, 0)): 182.0, ((28, 35, 8, 0), (20, 19, 10, 0)): 182.0, ((20, 19, 10, 0), (33, 6, 14, 0)): 100.0, ((33, 6, 14, 0), (29, 1, 16, 0)): 100.0, ((29, 1, 16, 0), (29, 1, 27, 0)): 100.0, ((34, 100, 0, 0), (6, 46, 7, 0)): 304.0, ((6, 46, 7, 0), (28, 36, 9, 0)): 249.0, ((28, 36, 9, 0), (20, 20, 11, 0)): 249.0, ((20, 20, 11, 0), (20, 20, 12, 0)): 166.0, ((20, 20, 12, 0), (20, 92, 13, 0)): 83.0, ((20, 92, 13, 0), (12, 1, 24, 0)): 83.0, ((12, 1, 24, 0), (12, 1, 27, 0)): 83.0, ((20, 20, 11, 0), (20, 92, 12, 0)): 83.0, ((20, 92, 12, 0), (12, 1, 23, 0)): 83.0, ((12, 1, 23, 0), (12, 1, 25, 0)): 165.0, ((12, 1, 25, 0), (12, 1, 26, 0)): 248.0, ((12, 1, 26, 0), (12, 1, 27, 0)): 303.0, ((6, 46, 7, 0), (6, 46, 9, 0)): 55.0, ((6, 46, 9, 0), (28, 36, 11, 0)): 55.0, ((28, 36, 11, 0), (20, 20, 13, 0)): 55.0, ((20, 20, 13, 0), (20, 92, 14, 0)): 83.0, ((20, 92, 14, 0), (12, 1, 25, 0)): 83.0, ((20, 19, 10, 0), (20, 92, 11, 0)): 82.0, ((20, 92, 11, 0), (12, 1, 22, 0)): 82.0, ((12, 1, 22, 0), (12, 1, 23, 0)): 82.0, ((20, 20, 12, 0), (20, 20, 13, 0)): 83.0, ((20, 20, 13, 0), (20, 20, 14, 0)): 55.0, ((20, 20, 14, 0), (20, 92, 15, 0)): 55.0, ((20, 92, 15, 0), (12, 1, 26, 0)): 55.0, ((34, 100, 0, 0), (0, 62, 5, 0)): 7.0, ((0, 62, 5, 0), (14, 61, 6, 0)): 7.0, ((14, 61, 6, 0), (14, 63, 7, 0)): 7.0, ((14, 63, 7, 0), (14, 65, 8, 0)): 7.0, ((14, 65, 8, 0), (14, 65, 10, 0)): 7.0, ((14, 65, 10, 0), (0, 64, 11, 0)): 7.0, ((0, 64, 11, 0), (14, 61, 12, 0)): 7.0, ((14, 61, 12, 0), (0, 60, 13, 0)): 7.0, ((0, 60, 13, 0), (20, 3, 20, 0)): 7.0, ((20, 3, 20, 0), (20, 55, 21, 0)): 7.0, ((20, 55, 21, 0), (20, 55, 27, 0)): 7.0, ((34, 100, 0, 0), (36, 98, 1, 0)): 7.0, ((36, 98, 1, 0), (34, 96, 2, 0)): 7.0, ((34, 96, 2, 0), (0, 58, 7, 0)): 7.0, ((0, 58, 7, 0), (14, 57, 8, 0)): 7.0, ((14, 57, 8, 0), (14, 59, 9, 0)): 7.0, ((14, 59, 9, 0), (14, 61, 10, 0)): 7.0, ((14, 61, 10, 0), (0, 60, 11, 0)): 7.0, ((0, 60, 11, 0), (14, 57, 12, 0)): 7.0, ((14, 57, 12, 0), (0, 56, 13, 0)): 7.0, ((0, 56, 13, 0), (0, 56, 27, 0)): 7.0}}

instance = Instance_v2.Instance(N, parameters, ['heuristic'], seed=None)

soln, val, props = instance.run_DDD(LB=0, LP=False)
      

# Convert tuple keys to strings for JSON serialization
def serialize_dict(d):
    return {str(k): v for k, v in d.items()}

result = {
    'config': instance.config,
    'objective': val,
    'properties': props,
    'solution': {k: serialize_dict(v) for k, v in soln.items()
                    if isinstance(v, dict)},
}
result_file = os.path.join(RESULTS_DIR, f'SJ_Oak-chargerOnly.json')
with open(result_file, 'w') as f:
    json.dump(result, f, indent=2)

