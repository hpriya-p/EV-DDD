#!/usr/bin/env python3
# coding: utf-8
"""
Test that run_DDD works on the c1fa89f5 instance from experiment_nodes.json.

Reproduces the base instance from Experiments.py restricted to the 10 nodes
stored under key "c1fa89f5" in results/experiment_nodes.json.
"""

import ast
import json
import unittest
from math import atan2, cos, radians, sin, sqrt

import networkx as nx
import numpy as np
import pandas as pd

import Instance_v2

# ── Constants matching Experiments.py ────────────────────────────────────────
PROCESSED_STATIONS_DF = "data/processed/stations_data.csv"
EDGE_THRESHOLD = 300   # miles – max edge length for connectivity
TRUCK_SPEED    = 60    # mph
DL_FACTOR      = 0.5  # dL = distance * DL_FACTOR
LB             = 50   # lower-bound passed to run_DDD


def haversine_miles(lat1, lng1, lat2, lng2):
    R = 3958.8
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def build_instance_from_coords(node_coords, config=['heuristic'], mode='tractor'):
    """
    Build an Instance from a list of (lat, lng) coordinates.

    Replicates the subgraph-construction logic of run_experiment() in
    Experiments.py: undirected edges for pairs within EDGE_THRESHOLD miles,
    dH/dL/time attributes, charger_rate from stations CSV, then directed.

    Parameters
    ----------
    node_coords : list of (lat, lng) tuples
    config      : Instance config string ('heuristic' or 'default')
    mode        : 'tractor' or 'battery'

    Returns
    -------
    instance   : Instance_v2.Instance  (network_only=True; no model yet)
    parameters : dict  (source/sink not yet set)
    """
    # ── Load station charger rates ────────────────────────────────────────────
    stations_df = pd.read_csv(PROCESSED_STATIONS_DF)
    stations_df['snap_to'] = stations_df['snap_to'].apply(ast.literal_eval)
    stations_df.reset_index(inplace=True)
    stations_df.index = stations_df['index']

    station_lookup = {
        (row['lat'], row['lng']): row['n_chargers'] * row['kw']
        for _, row in stations_df.iterrows()
    }

    # ── Build undirected graph ────────────────────────────────────────────────
    Und = nx.Graph()
    for i, coord in enumerate(node_coords):
        Und.add_node(i, pos=coord, charger_rate=station_lookup.get(coord, 0))

    for i in range(len(node_coords)):
        for j in range(i + 1, len(node_coords)):
            lat1, lng1 = node_coords[i]
            lat2, lng2 = node_coords[j]
            dist = haversine_miles(lat1, lng1, lat2, lng2)
            if dist <= EDGE_THRESHOLD:
                Und.add_edge(i, j, weight=dist)

    # ── Convert to directed graph with dH / dL / time ────────────────────────
    G = nx.DiGraph()
    for node, data in Und.nodes(data=True):
        G.add_node(node, **data)
    for u, v, data in Und.edges(data=True):
        dist = data['weight']
        attrs = {
            'dH':   int(dist / 190 * 100),
            'dL':   int(dist * DL_FACTOR),
            'time': int(np.ceil(dist / TRUCK_SPEED)),
        }
        G.add_edge(u, v, **attrs)
        G.add_edge(v, u, **attrs)

    # ── Parameters ───────────────────────────────────────────────────────────
    charge_nodes     = [i for i in G.nodes if G.nodes[i].get('charger_rate', 0) > 0]
    non_charge_nodes = [i for i in G.nodes if G.nodes[i].get('charger_rate', 0) == 0]
    charge_rate      = {i: G.nodes[i].get('charger_rate', 0) for i in G.nodes}

    if mode == 'tractor':
        battery_nodes = []
        tractor_nodes = non_charge_nodes
    else:
        battery_nodes = non_charge_nodes
        tractor_nodes = []

    T = 65
    parameters = {
        'T':            T,
        'L':            100,
        'D':            5,
        'step_size':    5,
        'MAX_ITER':     40,
        'charge_nodes': charge_nodes,
        'battery_nodes': battery_nodes,
        'tractor_nodes': tractor_nodes,
        'mobile_nodes': [],
        'charge_rate':  charge_rate,
        'charge_cost':  {(i, t): 1 for i in G.nodes for t in range(T)},
        'stat_cost':    {i: 0 for i in G.nodes},
        'surplus_cost': {i: 0 for i in G.nodes},
        'N_tractors':   3,
        'N_batteries':  3,
        'N_chargers':   3,
        'speed_curve':  {0: {'speed': 1, 'minbat': 0, 'maxbat': 100}},
        'bat_swap_time': 2,
        'tr_swap_time':  5,
    }

    # Build network only (source/sink unknown yet)
    instance = Instance_v2.Instance(G, parameters, config, network_only=True)
    return instance, parameters, T


def find_source_sink(instance, T):
    """
    Replicate Experiments.run_experiment: pick source/sink as endpoints of the
    longest shortest-path (by dH distance) across the time-expanded network.
    """
    Ntl = instance.Ntl.Ntl
    for e in Ntl.edges:
        Ntl.edges[e]['dH'] = instance.Ntl.edge_dists.get(e, 0)

    src_nodes  = [v for v in Ntl.nodes() if v[2] == 0]
    dest_nodes = set(v for v in Ntl.nodes() if v[2] == T - 1)

    longest, s, t = -1, None, None
    for src in src_nodes:
        lengths = nx.shortest_path_length(Ntl, source=src, weight='dH')
        for k, length in lengths.items():
            if k in dest_nodes and length > longest:
                longest, s, t = length, src, k

    return s, t


class TestC1fa89f5Instance(unittest.TestCase):
    """Run DDD on the c1fa89f5 instance stored in results/experiment_nodes.json."""

    def setUp(self):
        with open("results/experiment_nodes.json") as f:
            exp_nodes = json.load(f)
        self.node_coords = [tuple(c) for c in exp_nodes["c1fa89f5"]]

    def test_run_ddd_heuristic(self):
        """run_DDD should complete and return a finite objective value."""
        instance, parameters, T = build_instance_from_coords(
            self.node_coords, config=['heuristic'], mode='tractor'
        )

        s, t = find_source_sink(instance, T)
        self.assertIsNotNone(s, "Could not find a source node")
        self.assertIsNotNone(t, "Could not find a sink node")

        parameters['sources'] = [s[0]]
        parameters['sinks']   = [t[0]]
        print(f"Source: {s[0]}, Sink: {t[0]}")

        instance.construct_model()
        soln, val, props = instance.run_DDD(LB=LB)

        self.assertIsNotNone(soln, "run_DDD returned None solution")
        self.assertAlmostEqual(val, 125.75, places=2, msg=f"Expected objective 128.75, got {val}")
        

        print(f"c1fa89f5 — objective={val}, props={props}")
        return soln 

    

if __name__ == '__main__':
    unittest.main()
