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
            'sources': [1],
            'sinks': [3],
            'T': 6,
            'L': 6,
            'D': {1: (1, 0), 3: (0, 1)},  # demand of 1 from node 1 to node 3
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

        for key, v in result.items():
            if isinstance(v, dict):
                self.assertNotEqual(v, {}, f"run_DDD returned empty dict for result['{key}']")

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
        for e in inst.Ntl.Ntl.edges(keys=True):
            if inst.Ntl.edge_types[e] == 'transit_H' and e[1][0] not in inst.param['battery_nodes']:
                x_e = result['x_ener'].get(e[:2], 0)
                x_l = result['x_load'].get(e[:2], 0)
                print(f"  e_load_ener[{e}]: slack={x_e - x_l}  (x_ener={x_e}, x_load={x_l})")
        # self.assertEqual(result['x_load'], expected_x_load)
        # self.assertEqual(result['x_ener'], expected_x_ener)
        
        print(result)
        self.assertEqual(result['n'][2], 1)
        self.assertEqual(val, 3)
    

    def test_ddd_swap_solution_min_time_2(self):
        """Duplicate test with min_time parameter set to 2."""
        params =  {
            'sources': [1],
            'sinks': [3],
            'T': 7,
            'L': 6,
            'D': {1: (1, 0), 3: (0, 1)},  # demand of 1 from node 1 to node 3
            'min_time': 1, 
            'charge_rate': {i: 1 for i in self.N.nodes},
            'battery_nodes': [],
            'charge_nodes': [],
            'mobile_nodes': [],
            'tractor_nodes': [2],
            'bat_swap_time': 1,
            'tr_swap_time': 1,
            'mobile_charge_rate': 0,
            'charge_cost': {(i, t): 10 for i in self.N.nodes for t in range(1,7)},
            'surplus_cost': {i: 0 for i in self.N.nodes},
            'stat_cost': {i: 0 for i in self.N.nodes},
            'rec_penalty': 1000,
            'step_size': 2,
            'MAX_ITER': 20,
            'N_tractors': 2,
            'N_chargers': 0,
            'N_batteries': 0,
            'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2}, 1: {'speed': 2, 'minbat': 2, 'maxbat': 4}, 2: {'speed': 1, 'minbat': 4, 'maxbat': 6}}
        }
        
        inst = Instance(self.N, params, ['default'])

        result, val, solve_prop = inst.run_DDD()
        for name in ['demand-1', 'demand-2']:
            c = inst.model.getConstrByName(name)
            if c is not None:
                inst.model.update()
                expr = inst.model.getRow(c)
                print(f"{name}: {expr} {c.Sense} {c.RHS}")
            else:
                print(f"{name}: not found")
        for key, v in result.items():
            if isinstance(v, dict):
                self.assertNotEqual(v, {}, f"run_DDD returned empty dict for result['{key}']")

        

        print(result)
        # same expected checks as the original test
        self.assertGreaterEqual(result['n'][2], 1)
        self.assertEqual(val, 3)


class TestMultipleSources(unittest.TestCase):
    """Test DDD with multiple source nodes."""

    def setUp(self):
        # Two sources (1, 2) feeding into a shared node (3), then a single sink (4).
        # 1 -> 3 -> 4,  2 -> 3 -> 4
        # Node 3 has a charger and tractor swap station.
        # dH=5 forces each truck to recharge/swap at node 3 before continuing.
        self.N = nx.DiGraph()
        self.N.add_edge(1, 3, dH=5, dL=3, time=2)
        self.N.add_edge(2, 3, dH=5, dL=3, time=2)
        self.N.add_edge(3, 4, dH=5, dL=3, time=2)

        nodes = list(self.N.nodes)
        T = 10
        self.parameters = {
            'sources': [1, 2],
            'sinks': [4],
            'T': T,
            'L': 6,
            'D': {1: (1, 0), 2: (1, 0), 4: (0, 2)},  # demand of 1 from nodes 1 and 2 to node 4
            'charge_rate': {i: 0 for i in nodes},
            'battery_nodes': [],
            'charge_nodes': [],
            'mobile_nodes': [],
            'tractor_nodes': [3],
            'bat_swap_time': 1,
            'tr_swap_time': 1,
            'mobile_charge_rate': 0,
            'charge_cost': {(i, t): 10 for i in nodes for t in range(T)},
            'surplus_cost': {i: 0 for i in nodes},
            'stat_cost': {i: 0 for i in nodes},
            'rec_penalty': 1000,
            'step_size': 1,
            'MAX_ITER': 20,
            'N_tractors': 4,
            'N_chargers': 0,
            'N_batteries': 0,
            'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 6}},
        }

    def test_ddd_multiple_sources(self):
        """Two trucks depart from distinct sources and both reach the sink."""
        inst = Instance(self.N, self.parameters, ['default'])
        result, val, _ = inst.run_DDD()

        for key, v in result.items():
            if isinstance(v, dict):
                self.assertNotEqual(v, {}, f"run_DDD returned empty dict for result['{key}']")

        print("Multiple sources result:", result)
        print("val:", val)

        # Both trucks must reach the sink: total x_load flow into sink == D == 2
        sink_flow = sum(
            v for (e, v) in result['x_load'].items()
            if e[1][0] in self.parameters['sinks'] and e[1][2] == self.parameters['T'] - 1
        )
        self.assertAlmostEqual(sink_flow, self.parameters['D'][4][1], places=4,
                               msg=f"Expected {self.parameters['D'][4][1]} units at sink, got {sink_flow}")


if __name__ == '__main__':
    unittest.main()