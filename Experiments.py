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
random_seed = 123

np.random.seed(random_seed)

PROCESSED_STATIONS_DF = "data/processed/stations_data.csv"
LB = 30
# bounding box
MAX_LAT = 49
MIN_LAT = 32
MAX_LNG = -67
MIN_LNG = -125

# West Coast bounding box
WEST_COAST_MIN_LAT = 34
WEST_COAST_MAX_LAT = 45
WEST_COAST_MIN_LNG = -125
WEST_COAST_MAX_LNG = -114

# Truck properties
RANGE = 190 # miles
 

# for a 100kwh station,
speed_curve ={0: {'speed': 3600 * 1/(35*3.6), 'minbat': 0, 'maxbat': 10},
                   1: {'speed': 3600 * 1/(25*3.6), 'minbat': 10, 'maxbat': 40},
                   2: {'speed': 3600 * 1/(37.5*3.6), 'minbat': 40, 'maxbat': 80},
                   3: {'speed': 3600 * 1/(172.5*3.6), 'minbat': 80, 'maxbat': 100}}


# Other parameters
BAT_FROM_TIME = 3600 * 1000 * .75 # battery consumption = (3600 * 1000 * .8) * (time in hrs)

DISTANCE_THRESHOLD = 0  # miles
EDGE_THRESHOLD = 300  # miles - max edge length for connectivity

SF_COORDS = (37.7749, -122.4194)
LA_COORDS = (34.0522, -118.2437)

# Major West Coast cities (name, lat, lng)
cities = [
    ('Seattle', 47.6062, -122.3321),
    ('Portland', 45.5152, -122.6784),
    ('San Francisco', 37.7749, -122.4194),
    ('Los Angeles', 34.0522, -118.2437),
    ('San Diego', 32.7157, -117.1611),
    ('Sacramento', 38.5816, -121.4944),
    ('Las Vegas', 36.1699, -115.1398),
    ('Phoenix', 33.4484, -112.0740),
    ('Reno', 39.5296, -119.8138),
    ('Eugene', 44.0521, -123.0868),
]

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

# Build conflict graph: nodes connected if within 10 miles of each other
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

for i, node in enumerate(all_selected_nodes):
    if node in station_lookup:
        N.nodes[i]['charger_rate'] = station_lookup[node]
    else:
        N.nodes[i]['charger_rate'] = 0

# print 'charger_rate' for each node
print("\nNode charger rates:")
for i in N.nodes:
    print(f"Node {i} at {N.nodes[i]['pos']}: charger_rate = {N.nodes[i]['charger_rate']} kW")

RESULTS_DIR = 'results'
NODES_FILE = os.path.join(RESULTS_DIR, 'experiment_nodes.json')
sf_idx, sf_node, sf_dist = find_closest_node(SF_COORDS[0], SF_COORDS[1], all_selected_nodes)
la_idx, la_node, la_dist = find_closest_node(LA_COORDS[0], LA_COORDS[1], all_selected_nodes)
print(sf_dist, la_dist)
# Find closest nodes to SF and LA

def solve_clustering_mip(G, K, s, L=100, alpha=2.0):
    """
    Solve the station-clustering MIP for K clusters and source s.

    min  sum_i z_i + K
    s.t.
      sum_k x[i,k] + z[i] >= 1                                    for all i in I
      d[i,l]*(x[i,k]+x[l,k]-1) <=
          alpha*y[j] - alpha*y[i]*(x[i,k]+x[l,k]-1) + L*x[j,k]  for all i,j,l in I, k in [K]
      x[i,k] + x[j,k] <= 1                                        for all i,j in I: d[i,j] > L
      sum_k x[i,k] <= 1                                            for all i in I
      x[i,k] in {0,1},  z[i] >= 0

    where y[i] = shortest dH-path distance from s to i in G.

    The path-potential constraint expands to:
        (d[i,l] + alpha*y[i]) * (x[i,k] + x[l,k] - 1) <= alpha*y[j] + L*x[j,k]
    which is linear since d, y, alpha, L are all parameters.

    Parameters
    ----------
    G     : nx.DiGraph with 'dH' edge attribute
    K     : int, number of clusters (fixed input)
    s     : node id, source node
    L     : float, range limit (default 100, matching parameters['L'])
    alpha : float, scaling factor for path potentials (default 1.0)

    Returns
    -------
    x_val   : dict (i, k) -> float in {0, 1}  or None if infeasible
    z_val   : dict i -> float                  or None if infeasible
    obj_val : float                            or None if infeasible
    """
    import gurobipy as gp
    from gurobipy import GRB

    nodes = list(G.nodes())
    K_range = range(1, K + 1)

    # --- Shortest-path potentials from source s (dH weight) ---
    y = dict(nx.shortest_path_length(G, source=s, weight='dH'))
    big_y = (max(y.values()) + L + 1) if y else (L + 1)
    for v in nodes:
        if v not in y:
            y[v] = big_y  # unreachable nodes get a large potential

    # --- Direct edge distances d[i, l] ---
    d = {}
    for u, v, data in G.edges(data=True):
        d[u, v] = data.get('dH', L)

    # --- Model ---
    model = gp.Model("clustering_mip")
    model.setParam('OutputFlag', 1)

    x = model.addVars(nodes, K_range, vtype=GRB.BINARY, name='x')
    z = model.addVars(nodes, lb=0.0, name='z')

    # Objective: min sum_i z_i + K  (K is a constant, so this minimises z slack)
    model.setObjective(gp.quicksum(z[i] for i in nodes) + K, GRB.MINIMIZE)

    # C1: every node is covered by some cluster or pays a z penalty
    for i in nodes:
        model.addConstr(
            gp.quicksum(x[i, k] for k in K_range) + z[i] >= 1,
            name=f"cover_{i}"
        )

    # C2: path-potential constraints
    # (d[i,l] + alpha*y[i])*(x[i,k] + x[l,k] - 1) <= alpha*y[j] + L*x[j,k]
    # Only non-trivial when coeff > 0; when x[i,k]+x[l,k]-1 <= 0 the LHS <= 0
    # and RHS >= 0, so the constraint holds automatically — but Gurobi handles
    # this correctly regardless; we skip pairs with coeff == 0 to save memory.
    for k in K_range:
        for i in nodes:
            for l in nodes:
                coeff = d.get((i, l), L) + alpha * y[i]
                for j in nodes:
                    model.addConstr(
                        coeff * (x[i, k] + x[l, k] - 1) <= alpha * y[j] + L * x[j, k],
                        name=f"path_{i}_{j}_{l}_{k}"
                    )

    # C3: nodes farther apart than L cannot share a cluster
    edge_pairs = set()
    for u, v in G.edges():
        edge_pairs.add((min(u, v), max(u, v)))
    for (i, j) in edge_pairs:
        d_ij = min(d.get((i, j), L), d.get((j, i), L))
        if d_ij >= L:
            for k in K_range:
                model.addConstr(x[i, k] + x[j, k] <= 1, name=f"excl_{i}_{j}_{k}")

    # C4: each node in at most one cluster
    for i in nodes:
        model.addConstr(
            gp.quicksum(x[i, k] for k in K_range) <= 1,
            name=f"unique_{i}"
        )

    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"solve_clustering_mip: model status = {model.status} (not optimal)")
        return None, None, None

    x_val = {(i, k): x[i, k].X for i in nodes for k in K_range}
    z_val = {i: z[i].X for i in nodes}
    obj_val = model.ObjVal

    return x_val, z_val, obj_val


def run_experiment(n, config, mode='battery'):
    """Run a DDD experiment on a random subgraph of N with n nodes."""

    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_charger = 5
    n_non_charge = n - 5

    # 1. Randomly sample n nodes, keep all edges between them
    non_charger_inds = [v for v in N.nodes if N.nodes[v].get('charger_rate', 0) == 0]
    charger_inds = [v for v in N.nodes if N.nodes[v].get('charger_rate', 0) > 0]     
    print(f"{n_charger}/{len(charger_inds)}", f"{n_non_charge}/{len(non_charger_inds)}")
    sampled_indices = sorted(np.random.choice(charger_inds, size=n_charger, replace=False))
    sampled_indices += sorted(np.random.choice(non_charger_inds, size=n_non_charge, replace=False))
    print(f"Sampled {n_non_charge} non-charger nodes + {n_charger} charger nodes = {len(sampled_indices)} total nodes")

    subgraph = N.subgraph(sampled_indices).copy()

    # Relabel nodes to 0..n-1
    idx_map = {old: new for new, old in enumerate(sampled_indices)}
    subgraph = nx.relabel_nodes(subgraph, idx_map)

    # Convert to DiGraph (Instance_v2 expects directed graph)
    # Add both directions for each undirected edge
    G = nx.DiGraph()
    for node, data in subgraph.nodes(data=True):
        G.add_node(node, **data)
    for u, v, data in subgraph.edges(data=True):
        G.add_edge(u, v, **data)
        G.add_edge(v, u, **data)

    # Rename edge attribute 'weight' -> 'dH', and compute 'dL' and 'time'
    TRUCK_SPEED = 60  # miles per hour
    DL_FACTOR = 0.5   # dL = dH * factor (lighter load uses less battery)
    for u, v, data in G.edges(data=True):
        dist = data.pop('weight', 0)
        print(dist)
        data['dH'] = int(dist/190 * 100)  # convert to percentage of battery range
        data['dL'] = int(dist * DL_FACTOR)
        data['time'] = int(np.ceil(dist / TRUCK_SPEED))
    
    # print G edge properties

    # 2. Construct parameters
    charge_nodes = [idx_map[i] for i in sampled_indices
                    if N.nodes[i].get('charger_rate', 0) > 0]
    charge_rate = {idx_map[i]: N.nodes[i]['charger_rate']
                   for i in sampled_indices}



    # battery nodes are a random subset of non-charge nodes
    non_charge_nodes = [idx_map[i] for i in sampled_indices
                        if N.nodes[i].get('charger_rate', 0) == 0]
    if mode == 'battery':
        battery_nodes = non_charge_nodes
        tractor_nodes = []
        mobile_nodes = []
    else:
        battery_nodes = []
        tractor_nodes = non_charge_nodes
        mobile_nodes = []
    T = 80
    parameters = {
        'T': T,
        'L': 100,
        'D': 1,
        'step_size': 5,
        'MAX_ITER': 70,
        'charge_nodes': charge_nodes,
        'battery_nodes': battery_nodes,
        'tractor_nodes': tractor_nodes,
        'mobile_nodes': mobile_nodes,
        'charge_rate': charge_rate,
        'charge_cost': {(i, t): 1 for i in G.nodes for t in range(T)},
        'stat_cost': {i: 0 for i in G.nodes},
        'surplus_cost': {i: 0 for i in G.nodes},
        'N_tractors': 3,
        'N_batteries': 3,
        'N_chargers': 3,
        'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 100}},
        'bat_swap_time': 2,
        'tr_swap_time': 5,
    }

    # 3. Create Instance and run DDD
    instance = Instance_v2.Instance(G, parameters, config, network_only=True)
    
    # choose longest shortest path from (0, 100, 0, 4) 
    for e in instance.Ntl.Ntl.edges:
        u, v, __ = e
        instance.Ntl.Ntl.edges[e]['dH'] = instance.Ntl.edge_dists.get(e, 0)
        instance.Ntl.Ntl.edges[e]['time'] = instance.Ntl.edge_times.get(e, 0)
    src_nodes = [v for v in instance.Ntl.Ntl.nodes() if v[2] == 0]
    dest_nodes = [v for v in instance.Ntl.Ntl.nodes() if v[2] == T - 1]
    longest_path = -1
    s, t = None, None
    dest_set = set(dest_nodes)
    for src in src_nodes:
        lengths = nx.shortest_path_length(instance.Ntl.Ntl, source=src, weight='dH')
        for k, v in lengths.items():
            if k in dest_set and v > longest_path:
                s = src
                t = k
                longest_path = v
                print(src, k, v)
    print("Longest shortest path is from ", s, " to ", t, " with length ", longest_path, 'and time ', nx.shortest_path_length(instance.Ntl.Ntl, s, t, 'time'))
    if T < (longest_path/RANGE + 1) * nx.shortest_path_length(instance.Ntl.Ntl, s, t, 'time'):
        T = np.ceil(longest_path/RANGE + 1 * nx.shortest_path_length(instance.Ntl.Ntl, s, t, 'time'))
        print(T)
        parameters['T'] = int(T)
        instance = Instance_v2.Instance(G, parameters, config, network_only=True)

    parameters['source'] =  s[0]
    parameters['sink'] = t[0]
    _t0 = time.time()
    instance.Ntl.construct()
    instance.construct_model()
    time_taken = time.time() - _t0
    print("Time taken", time_taken)
    soln, val, props = instance.run_DDD(LB=LB, LP=False)

    # 4. Write results to JSON
    exp_id = str(uuid.uuid4())[:8]

    # Convert tuple keys to strings for JSON serialization
    def serialize_dict(d):
        return {str(k): v for k, v in d.items()}

    result = {
        'id': exp_id,
        'config': instance.config,
        'n': n,
        'objective': val,
        'initial_construct_time': time_taken,
        'properties': props,
        'solution': {k: serialize_dict(v) for k, v in soln.items()
                     if isinstance(v, dict)},
    }
    result_file = os.path.join(RESULTS_DIR, f'IP_experiment_{exp_id}-{config[0]}-{mode}.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    if 'heuristic' in config:
        _t0 = time.time()
        instance = Instance_v2.Instance(G, parameters, ['default'])
        time_taken = time.time() - _t0
        print("Time taken", time_taken)
        soln, val, props = instance.run_DDD(LP=False)
        result = {
            'id': exp_id,
            'config': instance.config,
            'n': n,
            'initial_construct_time': time_taken,
            'objective': val,
            'properties': props,
            'solution': {k: serialize_dict(v) for k, v in soln.items()
                        if isinstance(v, dict)}
        }
        result_file = os.path.join(RESULTS_DIR, f'IP_experiment_{exp_id}-default-{mode}.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
    # Write node coordinates to shared JSON file
    try:
        with open(NODES_FILE, 'r') as f:
            all_coords = json.load(f)
    except FileNotFoundError:
        all_coords = {}

    all_coords[exp_id] = [list(all_selected_nodes[i]) for i in sampled_indices]
    with open(NODES_FILE, 'w') as f:
        json.dump(all_coords, f, indent=2)

    print(f"Experiment {exp_id}: objective={val}, written to {result_file}")
    return exp_id


i = 0
while i < 5:
    print("Experiment ", i)
    try:
        run_experiment(10, ['heuristic'], 'tractor')
        i += 1
    except Exception as e:
        print(f"Experiment failed with error: {e}")
         
     