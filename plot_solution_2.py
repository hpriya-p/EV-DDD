"""
Plot flow solution 2 (x_load, x_ener) on a California basemap.
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

x_load = {((34, 100, 0, 0), (9, 54, 6, 0)): 3.0, ((9, 54, 6, 0), (39, 37, 9, 0)): 3.0, ((39, 37, 9, 0), (3, 12, 12, 0)): 3.0, ((3, 12, 12, 0), (3, 85, 13, 0)): 3.0, ((3, 85, 13, 0), (3, 85, 15, 0)): 3.0, ((34, 100, 0, 0), (2, 59, 5, 0)): 394.0, ((2, 59, 5, 0), (28, 31, 9, 0)): 2.0, ((28, 31, 9, 0), (28, 98, 10, 0)): 2.0, ((28, 98, 10, 0), (20, 82, 12, 0)): 2.0, ((20, 82, 12, 0), (20, 87, 13, 0)): 2.0, ((20, 87, 13, 0), (20, 87, 15, 0)): 2.0, ((2, 59, 5, 0), (2, 72, 6, 0)): 7.0, ((2, 72, 6, 0), (9, 66, 7, 0)): 7.0, ((9, 66, 7, 0), (28, 45, 10, 0)): 7.0, ((28, 45, 10, 0), (20, 29, 12, 0)): 7.0, ((20, 29, 12, 0), (20, 75, 13, 0)): 7.0, ((20, 75, 13, 0), (20, 75, 15, 0)): 7.0, ((2, 59, 5, 0), (2, 94, 6, 0)): 26.0, ((2, 94, 6, 0), (9, 88, 7, 0)): 26.0, ((9, 88, 7, 0), (28, 67, 10, 0)): 26.0, ((28, 67, 10, 0), (20, 51, 12, 0)): 26.0, ((20, 51, 12, 0), (20, 83, 13, 0)): 26.0, ((20, 83, 13, 0), (20, 83, 15, 0)): 194.0, ((2, 59, 5, 0), (2, 100, 6, 0)): 168.0, ((2, 100, 6, 0), (9, 94, 7, 0)): 168.0, ((9, 94, 7, 0), (28, 73, 10, 0)): 168.0, ((28, 73, 10, 0), (20, 57, 12, 0)): 168.0, ((20, 57, 12, 0), (20, 83, 13, 0)): 168.0, ((2, 59, 5, 0), (6, 41, 8, 0)): 191.0, ((6, 41, 8, 0), (6, 55, 9, 0)): 191.0, ((6, 55, 9, 0), (28, 45, 11, 0)): 191.0, ((28, 45, 11, 0), (20, 29, 13, 0)): 191.0, ((20, 29, 13, 0), (20, 75, 14, 0)): 191.0, ((20, 75, 14, 0), (20, 75, 15, 0)): 191.0, ((34, 100, 0, 0), (6, 46, 7, 0)): 154.0, ((6, 46, 7, 0), (28, 36, 9, 0)): 154.0, ((28, 36, 9, 0), (3, 12, 12, 0)): 154.0, ((3, 12, 12, 0), (3, 96, 13, 0)): 152.0, ((3, 96, 13, 0), (3, 96, 15, 0)): 152.0, ((3, 12, 12, 0), (3, 12, 15, 0)): 2.0, ((34, 100, 0, 0), (32, 85, 2, 0)): 349.0, ((32, 85, 2, 0), (36, 70, 4, 0)): 336.0, ((36, 70, 4, 0), (1, 39, 8, 0)): 336.0, ((1, 39, 8, 0), (39, 6, 12, 0)): 336.0, ((39, 6, 12, 0), (39, 16, 13, 0)): 336.0, ((39, 16, 13, 0), (39, 16, 15, 0)): 336.0, ((32, 85, 2, 0), (32, 100, 3, 0)): 13.0, ((32, 100, 3, 0), (5, 80, 6, 0)): 13.0, ((5, 80, 6, 0), (5, 100, 7, 0)): 13.0, ((5, 100, 7, 0), (2, 72, 11, 0)): 13.0, ((2, 72, 11, 0), (9, 66, 12, 0)): 13.0, ((9, 66, 12, 0), (28, 45, 15, 0)): 13.0}
x_ener = {((3, 100, 0, 0), (29, 93, 2, 0)): 3.0, ((29, 93, 2, 0), (3, 86, 4, 0)): 3.0, ((3, 86, 4, 0), (20, 81, 6, 0)): 3.0, ((20, 81, 6, 0), (28, 73, 8, 0)): 3.0, ((28, 73, 8, 0), (6, 68, 10, 0)): 3.0, ((6, 68, 10, 0), (39, 62, 12, 0)): 3.0, ((39, 62, 12, 0), (15, 56, 14, 0)): 3.0, ((15, 56, 14, 0), (15, 56, 15, 0)): 3.0, ((33, 100, 0, 0), (29, 95, 2, 0)): 2.0, ((29, 95, 2, 0), (3, 88, 4, 0)): 2.0, ((3, 88, 4, 0), (20, 83, 6, 0)): 2.0, ((20, 83, 6, 0), (28, 75, 8, 0)): 2.0, ((28, 75, 8, 0), (6, 70, 10, 0)): 2.0, ((6, 70, 10, 0), (39, 64, 12, 0)): 2.0, ((39, 64, 12, 0), (15, 58, 14, 0)): 2.0, ((15, 58, 14, 0), (15, 58, 15, 0)): 2.0, ((34, 100, 0, 0), (6, 46, 7, 0)): 154.0, ((6, 46, 7, 0), (28, 36, 9, 0)): 154.0, ((28, 36, 9, 0), (3, 12, 12, 0)): 154.0, ((3, 12, 12, 0), (3, 12, 15, 0)): 2.0, ((34, 100, 0, 0), (9, 54, 6, 0)): 3.0, ((9, 54, 6, 0), (39, 37, 9, 0)): 3.0, ((39, 37, 9, 0), (3, 12, 12, 0)): 3.0, ((3, 12, 12, 0), (20, 7, 14, 0)): 155.0, ((20, 7, 14, 0), (20, 7, 15, 0)): 3.0, ((34, 100, 0, 0), (2, 59, 5, 0)): 394.0, ((2, 59, 5, 0), (9, 56, 6, 0)): 198.0, ((9, 56, 6, 0), (6, 51, 8, 0)): 191.0, ((6, 51, 8, 0), (28, 41, 10, 0)): 191.0, ((28, 41, 10, 0), (20, 25, 12, 0)): 191.0, ((20, 25, 12, 0), (20, 39, 13, 0)): 191.0, ((20, 39, 13, 0), (20, 47, 14, 0)): 191.0, ((20, 47, 14, 0), (20, 48, 15, 0)): 191.0, ((2, 59, 5, 0), (6, 41, 8, 0)): 191.0, ((6, 41, 8, 0), (28, 36, 10, 0)): 191.0, ((28, 36, 10, 0), (20, 28, 12, 0)): 191.0, ((20, 28, 12, 0), (20, 39, 13, 0)): 191.0, ((20, 39, 13, 0), (20, 53, 14, 0)): 191.0, ((20, 53, 14, 0), (20, 61, 15, 0)): 191.0, ((20, 7, 14, 0), (20, 8, 15, 0)): 152.0, ((2, 59, 5, 0), (14, 40, 10, 0)): 3.0, ((14, 40, 10, 0), (14, 43, 12, 0)): 3.0, ((14, 43, 12, 0), (0, 42, 13, 0)): 3.0, ((0, 42, 13, 0), (14, 41, 14, 0)): 3.0, ((14, 41, 14, 0), (4, 39, 15, 0)): 3.0, ((9, 56, 6, 0), (9, 57, 7, 0)): 7.0, ((9, 57, 7, 0), (9, 60, 9, 0)): 7.0, ((9, 60, 9, 0), (9, 61, 10, 0)): 7.0, ((9, 61, 10, 0), (9, 65, 12, 0)): 7.0, ((9, 65, 12, 0), (2, 62, 13, 0)): 7.0, ((2, 62, 13, 0), (9, 56, 14, 0)): 7.0, ((9, 56, 14, 0), (9, 56, 15, 0)): 7.0, ((2, 59, 5, 0), (28, 31, 9, 0)): 2.0, ((28, 31, 9, 0), (39, 25, 11, 0)): 2.0, ((39, 25, 11, 0), (9, 17, 14, 0)): 2.0, ((9, 17, 14, 0), (2, 14, 15, 0)): 2.0, ((34, 100, 0, 0), (32, 85, 2, 0)): 349.0, ((32, 85, 2, 0), (36, 70, 4, 0)): 349.0, ((36, 70, 4, 0), (34, 68, 5, 0)): 13.0, ((34, 68, 5, 0), (34, 71, 7, 0)): 13.0, ((34, 71, 7, 0), (32, 64, 9, 0)): 13.0, ((32, 64, 9, 0), (25, 53, 12, 0)): 13.0, ((25, 53, 12, 0), (5, 41, 15, 0)): 13.0, ((36, 70, 4, 0), (1, 39, 8, 0)): 336.0, ((1, 39, 8, 0), (39, 6, 12, 0)): 336.0, ((39, 6, 12, 0), (6, 0, 14, 0)): 336.0, ((6, 0, 14, 0), (6, 0, 15, 0)): 336.0}

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

fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.set_extent([WEST_COAST_MIN_LNG - 0.5, WEST_COAST_MAX_LNG + 0.5,
               WEST_COAST_MIN_LAT - 0.3, WEST_COAST_MAX_LAT + 0.3],
              crs=ccrs.PlateCarree())

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

all_flow_vals = list(load_flows.values()) + list(ener_flows.values())
max_flow = max(all_flow_vals) if all_flow_vals else 1

def flow_lw(val):
    return 1.0 + 5.0 * (val / max_flow)

LOAD_COLOR = '#c0392b'   # deep red
ENER_COLOR = '#2980b9'   # blue
OFFSET_DEG = 0.04

def draw_flow_arrows(ax, flow_agg, color, offset_sign):
    """Draw directed arrows for each physical edge in flow_agg."""
    for (i, j), val in flow_agg.items():
        if i not in N.nodes or j not in N.nodes:
            continue
        lat1, lng1 = N.nodes[i]['pos']
        lat2, lng2 = N.nodes[j]['pos']

        dlat, dlng = lat2 - lat1, lng2 - lng1
        length = (dlat**2 + dlng**2) ** 0.5
        if length > 0:
            perp_lat = -dlng / length * OFFSET_DEG * offset_sign
            perp_lng =  dlat / length * OFFSET_DEG * offset_sign
        else:
            perp_lat = perp_lng = 0

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
               markersize=10, linewidth=0, label=f'Origin \u2013 Los Angeles (node {la_idx})'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#8e44ad',
               markersize=10, linewidth=0, label=f'Destination \u2013 Oakland (node {oak_idx})'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=8,
          framealpha=0.9, edgecolor='#aaaaaa')

ax.set_title('Flow Solution 2 on West Coast Network\n'
             f'x_load: {len(load_flows)} travel edges  |  x_ener: {len(ener_flows)} travel edges',
             fontsize=12)

# -- Origin/Destination annotation boxes --
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
plt.savefig('flow_plot_solution_2.png', dpi=150, bbox_inches='tight')
print("Saved to flow_plot_solution_2.png")
plt.show()
