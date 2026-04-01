"""
Plot flow solution (x_load, x_ener) on a California basemap.
Nodes and edges come from ios_westcoast_tractor.py (lines 1-220).
"""

# ============================================================
# Reproduce setup from ios_westcoast_tractor.py lines 1-220
# ============================================================

import time
import pandas as pd
import networkx as nx
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from adjustText import adjust_text
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from math import radians, sin, cos, sqrt, atan2

random_seed = 1
np.random.seed(random_seed)

PROCESSED_STATIONS_DF = "data/processed/stations_data.csv"
WEST_COAST_MIN_LAT = 34
WEST_COAST_MAX_LAT = 39
WEST_COAST_MIN_LNG = -125
WEST_COAST_MAX_LNG = -115
RANGE = 190
DISTANCE_THRESHOLD = 20
EDGE_THRESHOLD = 200

speed_curve = {
    0: {'speed': 3600 * 1/(35*3.6),   'minbat': 0,  'maxbat': 10},
    1: {'speed': 3600 * 1/(25*3.6),   'minbat': 10, 'maxbat': 40},
    2: {'speed': 3600 * 1/(37.5*3.6), 'minbat': 40, 'maxbat': 80},
    3: {'speed': 3600 * 1/(172.5*3.6),'minbat': 80, 'maxbat': 100},
}
BAT_FROM_TIME = 3600 * 1000 * .75

def haversine_miles(lat1, lng1, lat2, lng2):
    R = 3958.8
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def is_west_coast(lat, lng):
    return (WEST_COAST_MIN_LAT <= lat <= WEST_COAST_MAX_LAT and
            WEST_COAST_MIN_LNG <= lng <= WEST_COAST_MAX_LNG)

def find_closest_node(target_lat, target_lng, node_list):
    min_dist = float('inf')
    closest_idx = None
    closest_node = None
    for idx, node in enumerate(node_list):
        dist = haversine_miles(target_lat, target_lng, node[0], node[1])
        if dist < min_dist:
            min_dist = dist
            closest_node = node
            closest_idx = idx
    return closest_idx, closest_node, min_dist

# Load data
other_nodes_df = pd.read_csv("data/D_clean_Jun12.csv")
stations_df = pd.read_csv(PROCESSED_STATIONS_DF)
stations_df['snap_to'] = stations_df['snap_to'].apply(ast.literal_eval)
stations_df.reset_index(inplace=True)
stations_df.index = stations_df['index']

other_nodes_df['pt1'] = other_nodes_df['pt1'].apply(
    lambda x: x if isinstance(x, tuple) else ast.literal_eval(x))
other_nodes_df['pt2'] = other_nodes_df['pt2'].apply(
    lambda x: x if isinstance(x, tuple) else ast.literal_eval(x))
all_other_nodes = set(other_nodes_df['pt1'].tolist() + other_nodes_df['pt2'].tolist())

stations_west = stations_df[
    (stations_df['lat'] >= WEST_COAST_MIN_LAT) &
    (stations_df['lat'] <= WEST_COAST_MAX_LAT) &
    (stations_df['lng'] >= WEST_COAST_MIN_LNG) &
    (stations_df['lng'] <= WEST_COAST_MAX_LNG)
]
station_coords = set(zip(stations_west['lat'], stations_west['lng']))

west_coast_nodes = [n for n in all_other_nodes if is_west_coast(n[0], n[1])]
conflict_graph = nx.Graph()
conflict_graph.add_nodes_from(range(len(west_coast_nodes)))
for i in range(len(west_coast_nodes)):
    for j in range(i + 1, len(west_coast_nodes)):
        lat1, lng1 = west_coast_nodes[i]
        lat2, lng2 = west_coast_nodes[j]
        if haversine_miles(lat1, lng1, lat2, lng2) <= DISTANCE_THRESHOLD:
            conflict_graph.add_edge(i, j)

independent_set_indices = nx.approximation.maximum_independent_set(conflict_graph)
max_independent_set = [west_coast_nodes[i] for i in independent_set_indices]

all_selected_nodes = list(set(list(station_coords) + list(max_independent_set)))

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

station_lookup = {(row['lat'], row['lng']): row['n_chargers'] * row['kw']
                  for _, row in stations_df.iterrows()}
station_nodes = []
for i, node in enumerate(all_selected_nodes):
    if node in station_lookup:
        N.nodes[i]['charger_rate'] = station_lookup[node]
        station_nodes.append(i)
    else:
        N.nodes[i]['charger_rate'] = 250

TRUCK_SPEED = 65
DL_FACTOR = 0.5
for u, v, data in N.edges(data=True):
    dist = data.pop('weight', 0)
    data['dH'] = int(dist / 190 * 100)
    data['dL'] = int(data['dH'] * DL_FACTOR)
    data['time'] = int(np.ceil(dist / TRUCK_SPEED * 4))

print(f"Nodes: {N.number_of_nodes()}, station nodes: {len(station_nodes)}")

cities = {
    'Los Angeles': (34.0522, -118.2437),
    'Oakland':     (37.8044, -122.2712),
}
la_idx,  la_node,  _ = find_closest_node(cities['Los Angeles'][0], cities['Los Angeles'][1],  all_selected_nodes)
oak_idx, oak_node, _ = find_closest_node(cities['Oakland'][0],     cities['Oakland'][1],      all_selected_nodes)
print(f"LA node idx={la_idx}  ({la_node}), Oakland node idx={oak_idx}  ({oak_node})")

# ============================================================
# Flow data
# ============================================================

x_load = {((34, 85, 0, 0), (28, 20, 8, 0)): 11.0, ((28, 20, 8, 0), (28, 100, 11, 0)): 11.0, ((28, 100, 11, 0), (20, 84, 13, 0)): 11.0, ((20, 84, 13, 0), (20, 100, 14, 0)): 11.0, ((20, 100, 14, 0), (20, 100, 15, 0)): 11.0, ((20, 100, 15, 0), (20, 100, 16, 0)): 11.0, ((20, 100, 16, 0), (3, 89, 18, 0)): 11.0, ((3, 89, 18, 0), (33, 74, 20, 0)): 11.0, ((33, 74, 20, 0), (33, 74, 23, 0)): 11.0, ((34, 96, 0, 0), (1, 60, 5, 0)): 262.0, ((1, 60, 5, 0), (39, 27, 9, 0)): 262.0, ((39, 27, 9, 0), (39, 32, 10, 0)): 262.0, ((39, 32, 10, 0), (20, 10, 13, 0)): 262.0, ((20, 10, 13, 0), (20, 92, 14, 0)): 262.0, ((20, 92, 14, 0), (20, 92, 23, 0)): 262.0, ((34, 100, 0, 0), (34, 100, 1, 0)): 222.0, ((34, 100, 1, 0), (20, 19, 11, 0)): 222.0, ((20, 19, 11, 0), (20, 100, 12, 0)): 222.0, ((20, 100, 12, 0), (20, 100, 23, 0)): 222.0, ((34, 100, 0, 0), (28, 35, 8, 0)): 455.0, ((28, 35, 8, 0), (20, 19, 10, 0)): 455.0, ((20, 19, 10, 0), (20, 100, 11, 0)): 1.0, ((20, 100, 11, 0), (12, 9, 22, 0)): 1.0, ((12, 9, 22, 0), (12, 9, 23, 0)): 1.0, ((34, 100, 0, 0), (5, 78, 3, 0)): 26.0, ((5, 78, 3, 0), (5, 100, 4, 0)): 26.0, ((5, 100, 4, 0), (2, 72, 8, 0)): 26.0, ((2, 72, 8, 0), (9, 66, 9, 0)): 26.0, ((9, 66, 9, 0), (28, 45, 12, 0)): 26.0, ((28, 45, 12, 0), (20, 29, 14, 0)): 26.0, ((20, 29, 14, 0), (20, 86, 15, 0)): 26.0, ((20, 86, 15, 0), (20, 86, 23, 0)): 26.0, ((34, 100, 0, 0), (14, 59, 5, 0)): 214.0, ((14, 59, 5, 0), (16, 38, 8, 0)): 132.0, ((16, 38, 8, 0), (16, 100, 10, 0)): 132.0, ((16, 100, 10, 0), (11, 78, 13, 0)): 132.0, ((11, 78, 13, 0), (11, 100, 14, 0)): 132.0, ((11, 100, 14, 0), (11, 100, 23, 0)): 132.0, ((14, 59, 5, 0), (0, 56, 6, 0)): 82.0, ((0, 56, 6, 0), (0, 92, 7, 0)): 82.0, ((0, 92, 7, 0), (16, 68, 10, 0)): 82.0, ((16, 68, 10, 0), (16, 92, 11, 0)): 82.0, ((16, 92, 11, 0), (11, 70, 14, 0)): 82.0, ((11, 70, 14, 0), (11, 92, 15, 0)): 82.0, ((11, 92, 15, 0), (11, 92, 23, 0)): 82.0, ((34, 100, 0, 0), (0, 62, 5, 0)): 160.0, ((0, 62, 5, 0), (0, 93, 6, 0)): 160.0, ((0, 93, 6, 0), (4, 86, 7, 0)): 160.0, ((4, 86, 7, 0), (16, 69, 10, 0)): 160.0, ((16, 69, 10, 0), (16, 93, 11, 0)): 160.0, ((16, 93, 11, 0), (11, 71, 14, 0)): 160.0, ((11, 71, 14, 0), (11, 93, 15, 0)): 160.0, ((11, 93, 15, 0), (11, 93, 23, 0)): 160.0, ((20, 19, 10, 0), (20, 89, 11, 0)): 454.0, ((20, 89, 11, 0), (8, 16, 20, 0)): 454.0, ((8, 16, 20, 0), (8, 16, 23, 0)): 454.0}
x_ener = {((34, 85, 0, 0), (28, 20, 8, 0)): 11.0, ((28, 20, 8, 0), (6, 15, 10, 0)): 11.0, ((6, 15, 10, 0), (9, 10, 12, 0)): 11.0, ((9, 10, 12, 0), (2, 7, 13, 0)): 11.0, ((2, 7, 13, 0), (9, 4, 14, 0)): 11.0, ((9, 4, 14, 0), (9, 4, 23, 0)): 1.0, ((9, 4, 14, 0), (9, 6, 15, 0)): 10.0, ((9, 6, 15, 0), (9, 9, 17, 0)): 10.0, ((9, 9, 17, 0), (9, 11, 18, 0)): 10.0, ((9, 11, 18, 0), (6, 6, 20, 0)): 10.0, ((6, 6, 20, 0), (28, 1, 22, 0)): 10.0, ((28, 1, 22, 0), (28, 1, 23, 0)): 10.0, ((34, 96, 0, 0), (1, 60, 5, 0)): 262.0, ((1, 60, 5, 0), (39, 27, 9, 0)): 262.0, ((39, 27, 9, 0), (9, 19, 12, 0)): 262.0, ((9, 19, 12, 0), (2, 16, 13, 0)): 262.0, ((2, 16, 13, 0), (9, 13, 14, 0)): 262.0, ((9, 13, 14, 0), (28, 3, 17, 0)): 23.0, ((28, 3, 17, 0), (28, 3, 23, 0)): 23.0, ((9, 13, 14, 0), (39, 5, 17, 0)): 239.0, ((39, 5, 17, 0), (39, 5, 23, 0)): 239.0, ((34, 100, 0, 0), (14, 59, 5, 0)): 214.0, ((14, 59, 5, 0), (0, 56, 6, 0)): 82.0, ((0, 56, 6, 0), (0, 57, 7, 0)): 82.0, ((0, 57, 7, 0), (0, 61, 8, 0)): 71.0, ((0, 61, 8, 0), (0, 63, 9, 0)): 71.0, ((0, 63, 9, 0), (14, 62, 10, 0)): 35.0, ((14, 62, 10, 0), (14, 64, 11, 0)): 35.0, ((14, 64, 11, 0), (0, 63, 12, 0)): 35.0, ((0, 63, 12, 0), (4, 60, 13, 0)): 35.0, ((4, 60, 13, 0), (16, 52, 16, 0)): 35.0, ((16, 52, 16, 0), (11, 41, 19, 0)): 35.0, ((11, 41, 19, 0), (22, 34, 21, 0)): 25.0, ((22, 34, 21, 0), (22, 34, 23, 0)): 25.0, ((34, 100, 0, 0), (0, 62, 5, 0)): 160.0, ((0, 62, 5, 0), (0, 62, 6, 0)): 11.0, ((0, 62, 6, 0), (14, 61, 7, 0)): 11.0, ((14, 61, 7, 0), (14, 61, 12, 0)): 11.0, ((14, 61, 12, 0), (0, 60, 13, 0)): 11.0, ((0, 60, 13, 0), (0, 61, 14, 0)): 11.0, ((0, 61, 14, 0), (14, 60, 15, 0)): 1.0, ((14, 60, 15, 0), (4, 58, 16, 0)): 1.0, ((4, 58, 16, 0), (4, 61, 20, 0)): 1.0, ((4, 61, 20, 0), (14, 59, 21, 0)): 1.0, ((14, 59, 21, 0), (14, 59, 23, 0)): 1.0, ((34, 100, 0, 0), (34, 100, 1, 0)): 222.0, ((34, 100, 1, 0), (20, 19, 11, 0)): 222.0, ((20, 19, 11, 0), (20, 63, 12, 0)): 78.0, ((20, 63, 12, 0), (20, 77, 13, 0)): 78.0, ((20, 77, 13, 0), (20, 78, 14, 0)): 78.0, ((20, 78, 14, 0), (20, 84, 15, 0)): 78.0, ((20, 84, 15, 0), (22, 73, 18, 0)): 59.0, ((22, 73, 18, 0), (11, 66, 20, 0)): 59.0, ((11, 66, 20, 0), (16, 55, 23, 0)): 59.0, ((20, 19, 11, 0), (20, 22, 12, 0)): 95.0, ((20, 22, 12, 0), (20, 32, 13, 0)): 94.0, ((20, 32, 13, 0), (20, 84, 14, 0)): 94.0, ((20, 84, 14, 0), (3, 79, 16, 0)): 94.0, ((3, 79, 16, 0), (20, 74, 18, 0)): 94.0, ((20, 74, 18, 0), (22, 63, 21, 0)): 94.0, ((22, 63, 21, 0), (11, 56, 23, 0)): 94.0, ((0, 61, 14, 0), (0, 62, 15, 0)): 10.0, ((0, 62, 15, 0), (14, 61, 16, 0)): 10.0, ((14, 61, 16, 0), (14, 63, 17, 0)): 10.0, ((14, 63, 17, 0), (16, 53, 20, 0)): 10.0, ((16, 53, 20, 0), (4, 45, 23, 0)): 10.0, ((20, 84, 15, 0), (3, 79, 17, 0)): 19.0, ((3, 79, 17, 0), (20, 74, 19, 0)): 19.0, ((20, 74, 19, 0), (22, 63, 22, 0)): 19.0, ((22, 63, 22, 0), (22, 63, 23, 0)): 19.0, ((20, 22, 12, 0), (3, 17, 14, 0)): 1.0, ((3, 17, 14, 0), (29, 10, 16, 0)): 1.0, ((29, 10, 16, 0), (33, 5, 18, 0)): 1.0, ((33, 5, 18, 0), (31, 0, 20, 0)): 1.0, ((31, 0, 20, 0), (31, 0, 23, 0)): 1.0, ((34, 100, 0, 0), (28, 35, 8, 0)): 455.0, ((28, 35, 8, 0), (20, 19, 10, 0)): 455.0, ((20, 19, 10, 0), (20, 22, 11, 0)): 455.0, ((20, 22, 11, 0), (3, 17, 13, 0)): 1.0, ((3, 17, 13, 0), (31, 5, 16, 0)): 1.0, ((31, 5, 16, 0), (31, 5, 23, 0)): 1.0, ((14, 59, 5, 0), (16, 38, 8, 0)): 132.0, ((16, 38, 8, 0), (11, 27, 11, 0)): 132.0, ((11, 27, 11, 0), (22, 20, 13, 0)): 109.0, ((22, 20, 13, 0), (20, 9, 16, 0)): 109.0, ((20, 9, 16, 0), (20, 13, 17, 0)): 109.0, ((20, 13, 17, 0), (3, 8, 19, 0)): 1.0, ((3, 8, 19, 0), (3, 8, 23, 0)): 1.0, ((20, 19, 11, 0), (20, 27, 12, 0)): 49.0, ((20, 27, 12, 0), (6, 14, 16, 0)): 49.0, ((6, 14, 16, 0), (9, 9, 18, 0)): 49.0, ((9, 9, 18, 0), (2, 6, 19, 0)): 49.0, ((2, 6, 19, 0), (9, 3, 20, 0)): 48.0, ((9, 3, 20, 0), (9, 5, 21, 0)): 10.0, ((9, 5, 21, 0), (9, 5, 23, 0)): 9.0, ((9, 3, 20, 0), (9, 3, 23, 0)): 38.0, ((2, 6, 19, 0), (2, 6, 23, 0)): 1.0, ((9, 5, 21, 0), (9, 8, 23, 0)): 1.0, ((0, 62, 5, 0), (0, 76, 6, 0)): 8.0, ((0, 76, 6, 0), (4, 69, 7, 0)): 8.0, ((4, 69, 7, 0), (0, 66, 8, 0)): 8.0, ((0, 66, 8, 0), (4, 59, 9, 0)): 8.0, ((4, 59, 9, 0), (16, 51, 12, 0)): 9.0, ((16, 51, 12, 0), (11, 40, 15, 0)): 9.0, ((11, 40, 15, 0), (11, 40, 23, 0)): 8.0, ((0, 62, 5, 0), (0, 63, 6, 0)): 138.0, ((0, 63, 6, 0), (4, 60, 7, 0)): 10.0, ((4, 60, 7, 0), (4, 63, 11, 0)): 10.0, ((4, 63, 11, 0), (0, 60, 12, 0)): 10.0, ((0, 60, 12, 0), (4, 53, 13, 0)): 10.0, ((4, 53, 13, 0), (16, 45, 16, 0)): 10.0, ((16, 45, 16, 0), (11, 34, 19, 0)): 10.0, ((11, 34, 19, 0), (11, 34, 23, 0)): 10.0, ((0, 63, 6, 0), (0, 64, 7, 0)): 128.0, ((0, 64, 7, 0), (0, 76, 8, 0)): 93.0, ((0, 76, 8, 0), (4, 69, 9, 0)): 93.0, ((4, 69, 9, 0), (0, 66, 10, 0)): 93.0, ((0, 66, 10, 0), (4, 59, 11, 0)): 93.0, ((4, 59, 11, 0), (16, 51, 14, 0)): 93.0, ((16, 51, 14, 0), (11, 40, 17, 0)): 93.0, ((11, 40, 17, 0), (11, 40, 23, 0)): 93.0, ((11, 41, 19, 0), (11, 41, 23, 0)): 10.0, ((0, 64, 7, 0), (14, 63, 8, 0)): 35.0, ((14, 63, 8, 0), (14, 65, 9, 0)): 35.0, ((14, 65, 9, 0), (16, 55, 12, 0)): 35.0, ((16, 55, 12, 0), (4, 47, 15, 0)): 35.0, ((4, 47, 15, 0), (0, 44, 16, 0)): 35.0, ((0, 44, 16, 0), (4, 37, 17, 0)): 35.0, ((4, 37, 17, 0), (16, 29, 20, 0)): 35.0, ((16, 29, 20, 0), (11, 18, 23, 0)): 35.0, ((0, 57, 7, 0), (0, 62, 8, 0)): 11.0, ((0, 62, 8, 0), (4, 59, 9, 0)): 11.0, ((4, 59, 9, 0), (4, 62, 13, 0)): 10.0, ((4, 62, 13, 0), (16, 54, 16, 0)): 10.0, ((16, 54, 16, 0), (14, 44, 19, 0)): 10.0, ((14, 44, 19, 0), (0, 43, 20, 0)): 10.0, ((0, 43, 20, 0), (4, 40, 21, 0)): 10.0, ((4, 40, 21, 0), (4, 40, 23, 0)): 10.0, ((0, 63, 9, 0), (0, 75, 10, 0)): 36.0, ((0, 75, 10, 0), (4, 68, 11, 0)): 36.0, ((4, 68, 11, 0), (16, 60, 14, 0)): 36.0, ((16, 60, 14, 0), (14, 50, 17, 0)): 36.0, ((14, 50, 17, 0), (0, 49, 18, 0)): 36.0, ((0, 49, 18, 0), (4, 46, 19, 0)): 36.0, ((4, 46, 19, 0), (16, 38, 22, 0)): 36.0, ((16, 38, 22, 0), (16, 38, 23, 0)): 36.0, ((20, 22, 11, 0), (20, 34, 12, 0)): 454.0, ((20, 34, 12, 0), (20, 66, 13, 0)): 454.0, ((20, 66, 13, 0), (20, 74, 14, 0)): 454.0, ((20, 74, 14, 0), (20, 77, 15, 0)): 454.0, ((20, 77, 15, 0), (20, 84, 16, 0)): 454.0, ((20, 84, 16, 0), (22, 73, 19, 0)): 454.0, ((22, 73, 19, 0), (11, 66, 21, 0)): 454.0, ((11, 66, 21, 0), (11, 66, 23, 0)): 454.0, ((11, 40, 15, 0), (22, 33, 17, 0)): 1.0, ((22, 33, 17, 0), (20, 22, 20, 0)): 1.0, ((20, 22, 20, 0), (20, 28, 21, 0)): 1.0, ((20, 28, 21, 0), (20, 29, 22, 0)): 1.0, ((20, 29, 22, 0), (20, 36, 23, 0)): 1.0, ((20, 13, 17, 0), (20, 20, 18, 0)): 108.0, ((20, 20, 18, 0), (20, 23, 19, 0)): 108.0, ((20, 23, 19, 0), (20, 35, 20, 0)): 108.0, ((20, 35, 20, 0), (20, 40, 21, 0)): 108.0, ((20, 40, 21, 0), (20, 41, 22, 0)): 108.0, ((20, 41, 22, 0), (20, 48, 23, 0)): 108.0, ((34, 100, 0, 0), (5, 78, 3, 0)): 26.0, ((5, 78, 3, 0), (32, 68, 6, 0)): 26.0, ((32, 68, 6, 0), (36, 61, 8, 0)): 26.0, ((36, 61, 8, 0), (34, 59, 9, 0)): 26.0, ((34, 59, 9, 0), (34, 62, 11, 0)): 26.0, ((34, 62, 11, 0), (5, 51, 14, 0)): 26.0, ((5, 51, 14, 0), (2, 23, 18, 0)): 26.0, ((2, 23, 18, 0), (9, 17, 19, 0)): 26.0, ((9, 17, 19, 0), (9, 17, 23, 0)): 26.0, ((11, 27, 11, 0), (28, 11, 15, 0)): 23.0, ((28, 11, 15, 0), (6, 6, 17, 0)): 23.0, ((6, 6, 17, 0), (9, 1, 19, 0)): 23.0, ((9, 1, 19, 0), (9, 1, 23, 0)): 23.0, ((0, 62, 5, 0), (34, 43, 10, 0)): 3.0, ((34, 43, 10, 0), (36, 41, 11, 0)): 3.0, ((36, 41, 11, 0), (34, 39, 12, 0)): 3.0, ((34, 39, 12, 0), (34, 43, 14, 0)): 3.0, ((34, 43, 14, 0), (34, 50, 17, 0)): 3.0, ((34, 50, 17, 0), (34, 56, 20, 0)): 3.0, ((34, 56, 20, 0), (34, 61, 23, 0)): 3.0}

# ============================================================
# Aggregate physical flows: sum flow over all arcs with same (i, j)
# ============================================================

def aggregate_physical_flows(flow_dict):
    """Sum flow values keyed by physical (src_node, dst_node), skip self-loops."""
    agg = {}
    for (src, dst), val in flow_dict.items():
        i, j = src[0], dst[0]
        if i != j:
            agg[(i, j)] = agg.get((i, j), 0) + val
    return agg

load_flows = aggregate_physical_flows(x_load)
ener_flows = aggregate_physical_flows(x_ener)

# ============================================================
# Identify charging events: same physical node, battery increases
# ============================================================

def find_charge_events(flow_dict):
    """
    Find arcs where the truck stays at the same node and battery level increases.
    Returns list of (node_id, t_start, g_before, g_after, flow).
    """
    events = []
    for (src, dst), val in flow_dict.items():
        i, g1, t1, _ = src
        j, g2, t2, _ = dst
        if i == j and g2 > g1:
            events.append((i, t1, g1, g2, val))
    return events

load_charge_events = find_charge_events(x_load)

# Group by node: collect all unique charge events (node -> list of (t, g_before, g_after))
charge_by_node = {}
for (node, t_start, g_before, g_after, fval) in load_charge_events:
    charge_by_node.setdefault(node, []).append((t_start, g_before, g_after, fval))

# ============================================================
# Plot
# ============================================================

# node positions: N.nodes[i]['pos'] = (lat, lng)
# cartopy wants (lng, lat) for transforms

fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Map extent: west coast bounding box with a little padding
ax.set_extent([WEST_COAST_MIN_LNG - 0.5, WEST_COAST_MAX_LNG + 0.5,
               WEST_COAST_MIN_LAT - 0.3, WEST_COAST_MAX_LAT + 0.3],
              crs=ccrs.PlateCarree())

# Background features
ax.add_feature(cfeature.OCEAN, facecolor='#d0e8f7', zorder=0)
ax.add_feature(cfeature.LAND, facecolor='#f5f0e8', zorder=0)
ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor='#888888', zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#555555', zorder=1)
ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor='#aaccee', zorder=1)
ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5,
             xlocs=range(-125, -114, 2), ylocs=range(34, 40))

# -- Draw all network edges (light gray background) --
for u, v in N.edges():
    lat1, lng1 = N.nodes[u]['pos']
    lat2, lng2 = N.nodes[v]['pos']
    ax.plot([lng1, lng2], [lat1, lat2],
            color='#cccccc', linewidth=0.4, alpha=0.5,
            transform=ccrs.PlateCarree(), zorder=2)

# -- Draw all nodes --
station_set = set(station_nodes)
all_nodes_in_flow = set()
for (src, dst) in list(x_load.keys()) + list(x_ener.keys()):
    all_nodes_in_flow.add(src[0])
    all_nodes_in_flow.add(dst[0])

for i in N.nodes():
    lat, lng = N.nodes[i]['pos']
    if i == 0:
        color, size, zorder = '#e74c3c', 80, 5
    elif i in station_set:
        color, size, zorder = '#e67e22', 60, 4
    else:
        color, size, zorder = '#7fb3d3', 25, 3
    ax.scatter(lng, lat, s=size, c=color, zorder=zorder,
               transform=ccrs.PlateCarree(), edgecolors='white', linewidths=0.4)
    if i == 0:
        ax.scatter(lng, lat, s=300, facecolors='none', edgecolors='red',
                   linewidths=2.5, zorder=6, transform=ccrs.PlateCarree())

# -- Origin (LA) and Destination (Oakland) markers --
for idx, name, color, marker in [
        (la_idx,  'Los Angeles\n(Origin)',  '#27ae60', 's'),
        (oak_idx, 'Oakland\n(Destination)', '#8e44ad', 'D'),
]:
    lat, lng = N.nodes[idx]['pos']
    ax.scatter(lng, lat, s=200, c=color, marker=marker, zorder=10,
               transform=ccrs.PlateCarree(), edgecolors='white', linewidths=1.2)

# Collect node-ID text objects for adjustText
_node_texts = []
for i in sorted(all_nodes_in_flow):
    if i in N.nodes:
        lat, lng = N.nodes[i]['pos']
        t = ax.text(lng, lat, str(i), fontsize=6.5, color='#333333',
                    transform=ccrs.PlateCarree(), zorder=5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, ec='none'))
        _node_texts.append(t)

# ============================================================
# Draw flow arrows
# ============================================================

# Normalise arrow width: map flow to linewidth 1-6
all_flow_vals = list(load_flows.values()) + list(ener_flows.values())
max_flow = max(all_flow_vals) if all_flow_vals else 1

def flow_lw(val):
    return 1.0 + 5.0 * (val / max_flow)

LOAD_COLOR = '#c0392b'   # deep red
ENER_COLOR = '#2980b9'   # blue

# Offset perpendicular to edge so load/ener arrows don't overlap
OFFSET_DEG = 0.04  # degrees

def draw_flow_arrows(ax, flow_agg, color, offset_sign):
    """Draw directed arrows for each physical edge in flow_agg."""
    for (i, j), val in flow_agg.items():
        if i not in N.nodes or j not in N.nodes:
            continue
        lat1, lng1 = N.nodes[i]['pos']
        lat2, lng2 = N.nodes[j]['pos']

        # Perpendicular offset to separate load vs ener arrows
        dlat, dlng = lat2 - lat1, lng2 - lng1
        length = (dlat**2 + dlng**2) ** 0.5
        if length > 0:
            perp_lat = -dlng / length * OFFSET_DEG * offset_sign
            perp_lng =  dlat / length * OFFSET_DEG * offset_sign
        else:
            perp_lat = perp_lng = 0

        # Shorten arrow slightly so arrowhead doesn't overlap node marker
        shrink = 0.07
        slat1 = lat1 + dlat * shrink + perp_lat
        slng1 = lng1 + dlng * shrink + perp_lng
        slat2 = lat2 - dlat * shrink + perp_lat
        slng2 = lng2 - dlng * shrink + perp_lng

        lw = flow_lw(val)
        ax.annotate('',
                    xy=(slng2, slat2), xytext=(slng1, slat1),
                    xycoords='data', textcoords='data',
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=lw, mutation_scale=12 + lw * 2),
                    transform=ccrs.PlateCarree(), zorder=6)


draw_flow_arrows(ax, load_flows, LOAD_COLOR, offset_sign=+1)
draw_flow_arrows(ax, ener_flows, ENER_COLOR, offset_sign=-1)

# ============================================================
# Annotate charging events
# ============================================================

_charge_texts = []
for node, events in charge_by_node.items():
    if node not in N.nodes:
        continue
    lat, lng = N.nodes[node]['pos']

    ax.plot(lng, lat, marker='$\u26a1$', markersize=12, color='#f39c12',
            transform=ccrs.PlateCarree(), zorder=8,
            markeredgewidth=0.5, markeredgecolor='black')

    lines = []
    for (t_start, g_before, g_after, fval) in sorted(events, key=lambda x: x[0]):
        lines.append(f't={t_start}: {g_before}%\u2192{g_after}%')
    label = f'n{node}: ' + ',  '.join(lines)

    t = ax.text(lng, lat, label,
                transform=ccrs.PlateCarree(), zorder=9,
                fontsize=6.5, color='#7d3c00', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='#fef9e7', alpha=0.9,
                          ec='#f39c12', lw=1.0))
    _charge_texts.append(t)

# ============================================================
# Legend
# ============================================================

legend_handles = [
    mpatches.Patch(facecolor='#e74c3c', edgecolor='white', label='Node 0'),
    mpatches.Patch(facecolor='#e67e22', edgecolor='white', label='Charging station node'),
    mpatches.Patch(facecolor='#7fb3d3', edgecolor='white', label='Non-station node'),
    plt.Line2D([0], [0], color=LOAD_COLOR, linewidth=2.5, label='x_load flow (trucks)'),
    plt.Line2D([0], [0], color=ENER_COLOR, linewidth=2.5, label='x_ener flow (energy)'),
    plt.Line2D([0], [0], marker='$\u26a1$', color='#f39c12', markersize=10,
               linewidth=0, label='Charging event (x_load)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#27ae60',
               markersize=10, linewidth=0, label=f'Origin – Los Angeles (node {la_idx})'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#8e44ad',
               markersize=10, linewidth=0, label=f'Destination – Oakland (node {oak_idx})'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=8,
          framealpha=0.9, edgecolor='#aaaaaa')

ax.set_title('Flow Solution on West Coast Network\n'
             f'x_load: {len(load_flows)} travel edges  |  x_ener: {len(ener_flows)} travel edges',
             fontsize=12)

# -- Origin/Destination annotation boxes (fixed positions, drawn last) --
for idx, name, color in [
        (la_idx,  'Los Angeles\n(Origin)',  '#27ae60'),
        (oak_idx, 'Oakland\n(Destination)', '#8e44ad'),
]:
    lat, lng = N.nodes[idx]['pos']
    dx = -0.8 if idx == la_idx else 0.6
    dy =  0.3
    ax.annotate(f'{name}\n(node {idx})',
                xy=(lng, lat), xytext=(lng + dx, lat + dy),
                transform=ccrs.PlateCarree(), zorder=12,
                fontsize=9, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.95,
                          ec=color, lw=1.8),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Spread all text labels to avoid overlaps
adjust_text(
    _node_texts + _charge_texts,
    ax=ax,
    expand=(1.3, 1.5),
    force_text=(0.4, 0.6),
    force_static=(0.2, 0.4),
    max_move=0.8,
    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
)

plt.tight_layout()
plt.savefig('flow_plot_solution.png', dpi=150, bbox_inches='tight')
print("Saved to flow_plot_solution.png")
plt.show()
