#!/usr/bin/env python
# coding: utf-8
"""
Test swap functionality for the Instance class.

Tests DDD on a simple 3-node network with tractor swap at node 2.
"""

import unittest
import networkx as nx
from Instance_v2 import Instance


class TestSwap(unittest.TestCase):
    """Test swap operations in the DDD algorithm."""

    def setUp(self):
        """Set up test network and parameters."""
        self.N = nx.DiGraph()
        self.N.add_edge(1, 2, dH=5, dL=5, time=1)
        self.N.add_edge(2, 3, dH=5, dL=5, time=1)

        self.parameters = {
            'source': 1,
            'sink': 3,
            'T': 6,
            'L': 6,
            'D': 1,
            'charge_rate': {i: 1 for i in self.N.nodes},
            'battery_nodes': [],
            'charge_nodes': [],
            'mobile_nodes': [],
            'tractor_nodes': [2],
            'bat_swap_time': 1,
            'tr_swap_time': 1,
            'mobile_charge_rate': 0,
            'charge_cost': {(i, t): 10 for i in self.N.nodes for t in range(6)},
            'surplus_cost': {i: 0 for i in self.N.nodes},
            'stat_cost': {i: 0 for i in self.N.nodes},
            'rec_penalty': 1000,
            'step_size': 2,
            'MAX_ITER': 20,
            'N_tractors': 1,
            'N_chargers': 0,
            'N_batteries': 0,
            'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2}, 1: {'speed': 2, 'minbat': 2, 'maxbat': 4}, 2: {'speed': 1, 'minbat': 4, 'maxbat': 6}}
        }

    def test_ddd_swap_solution(self):
        """Test that DDD produces correct solution with tractor swap."""
        inst = Instance(self.N, self.parameters, ['default'])

        result, val, solve_prop = inst.run_DDD()

        expected_x_load = {
            ((1, 6, 0, inst.Ntl.get_q(1, 6)), (2, 1, 1, inst.Ntl.get_q(2, 1))): 1.0,
            ((2, 1, 1, inst.Ntl.get_q(2, 1)), (2, 6, 2, inst.Ntl.get_q(2, 6))): 1.0,
            ((2, 6, 2, inst.Ntl.get_q(2, 6)), (3, 1, 3, inst.Ntl.get_q(3, 1))): 1.0,
            ((3, 1, 3, inst.Ntl.get_q(3, 1)), (3, 1, 5)): 1.0,
        }
        expected_x_ener = {
            ((1, 6, 0, inst.Ntl.get_q(1, 6)), (2, 1, 1, inst.Ntl.get_q(2, 1))): 1.0,
            ((2, 1, 1, inst.Ntl.get_q(2, 1)), (2, 1, 5)): 1.0,
            ((2, 6, 0, inst.Ntl.get_q(2, 6)), (2, 6, 1, inst.Ntl.get_q(2, 6))): 1.0,
            ((2, 6, 1, inst.Ntl.get_q(2, 6)), (2, 6, 2, inst.Ntl.get_q(2, 6))): 1.0,
            ((2, 6, 2, inst.Ntl.get_q(2, 6)), (3, 1, 3, inst.Ntl.get_q(3, 1))): 1.0,
            ((3, 1, 3, inst.Ntl.get_q(3, 1)), (3, 1, 5, inst.Ntl.get_q(3, 1))): 1.0,
        }
        print("Result:", result)
        print("Solve properties:", solve_prop)
        print(expected_x_ener)
        e_check = ((2, 5, 2, 2), (3, 0, 3, 0), 0)
        print(f"\nEdge type of {e_check}: {inst.Ntl.edge_types.get(e_check, 'not found')}")
        print("\n--- e_load_ener constraint slacks (x_ener - x_load) ---")
        for e in inst.Ntl.Ntl.edges:
            if inst.Ntl.edge_types[e] == 'transit_H' and e[1][0] not in inst.param['battery_nodes']:
                x_e = result['x_ener'].get(e[:2], 0)
                x_l = result['x_load'].get(e[:2], 0)
                print(f"  e_load_ener[{e}]: slack={x_e - x_l}  (x_ener={x_e}, x_load={x_l})")
        # self.assertEqual(result['x_load'], expected_x_load)
        # self.assertEqual(result['x_ener'], expected_x_ener)
        self.assertEqual(result['n'][2], 1)
        self.assertEqual(val, 7)
    


if __name__ == '__main__':
    unittest.main()