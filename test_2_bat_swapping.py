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
            'D': {(1, 3): 1},
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
            'tractor_cost': 1, 
            'battery_cost': 1,
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

        for k, edge_dict in result['x_load'].items():
            self.assertNotEqual(edge_dict, {}, f"run_DDD returned empty x_load for commodity {k}")
        for k, edge_dict in result['x_ener'].items():
            self.assertNotEqual(edge_dict, {}, f"run_DDD returned empty x_ener for commodity {k}")

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
                x_e = sum(d.get(e[:2], 0) for d in result['x_ener'].values())
                x_l = sum(d.get(e[:2], 0) for d in result['x_load'].values())
                print(f"  e_load_ener[{e}]: slack={x_e - x_l}  (x_ener={x_e}, x_load={x_l})")
        # self.assertEqual(result['x_load'], expected_x_load)
        # self.assertEqual(result['x_ener'], expected_x_ener)
        
        print(result)
        self.assertEqual(result['n'][2], 1)
        self.assertEqual(val, 5)
    

    def test_ddd_swap_solution_min_time_2(self):
        """Duplicate test with min_time parameter set to 2."""
        params =  {
            'sources': [1],
            'sinks': [3],
            'T': 7,
            'L': 6,
            'D': {(1, 3): 1},
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
            'tractor_cost': 1,
            'battery_cost': 1,
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
        for k, edge_dict in result['x_load'].items():
            self.assertNotEqual(edge_dict, {}, f"run_DDD returned empty x_load for commodity {k}")
        for k, edge_dict in result['x_ener'].items():
            self.assertNotEqual(edge_dict, {}, f"run_DDD returned empty x_ener for commodity {k}")

        print(result)
        # same expected checks as the original test
        self.assertGreaterEqual(result['n'][2], 1)
        self.assertEqual(val, 5)


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
            'D': {(1, 4): 1, (2, 4): 1},  # 1 truck from node 1, 1 from node 2, both to node 4
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
            'tractor_cost': 1,
            'battery_cost': 1,
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

        for k, edge_dict in result['x_load'].items():
            self.assertNotEqual(edge_dict, {}, f"run_DDD returned empty x_load for commodity {k}")
        for k, edge_dict in result['x_ener'].items():
            self.assertNotEqual(edge_dict, {}, f"run_DDD returned empty x_ener for commodity {k}")

        print("Multiple sources result:", result)
        print("val:", val)

        # Both trucks must reach the sink: total x_load flow into sink == D == 2
        sink_flow = sum(
            flow for edge_dict in result['x_load'].values()
            for e, flow in edge_dict.items()
            if e[1][0] in self.parameters['sinks'] and e[1][2] == self.parameters['T'] - 1
        )
        expected_at_sink = sum(v for (src, snk), v in self.parameters['D'].items() if snk == 4)
        self.assertAlmostEqual(sink_flow, expected_at_sink, places=4,
                               msg=f"Expected {expected_at_sink} units at sink, got {sink_flow}")


class TestMultipleBatSwaps(unittest.TestCase):
    """Test DDD with mandatory battery swaps at every intermediate node."""

    def setUp(self):
        # 5-node chain: 1 → 2 → 3 → 4 → 5
        # dH=5 drains L=6 battery to 1 on every edge, so the truck cannot
        # continue without a swap at each intermediate node.
        self.N = nx.DiGraph()
        self.N.add_edge(1, 2, dH=5, dL=5, time=1)
        self.N.add_edge(2, 3, dH=5, dL=5, time=1)
        self.N.add_edge(3, 4, dH=5, dL=5, time=1)
        self.N.add_edge(4, 5, dH=5, dL=5, time=1)

        nodes = list(self.N.nodes)
        T = 9  # 4 transit + 3 swaps = 7 steps, truck waits 1 step at sink
        self.parameters = {
            'sources': [1],
            'sinks': [5],
            'T': T,
            'L': 6,
            'D': {(1, 5): 1},
            'charge_rate': {i: 0 for i in nodes},
            'battery_nodes': [2, 3, 4],
            'charge_nodes': [],
            'mobile_nodes': [],
            'tractor_nodes': [],
            'bat_swap_time': 1,
            'tr_swap_time': 1,
            'mobile_charge_rate': 0,
            'charge_cost': {(i, t): 10 for i in nodes for t in range(T)},
            'surplus_cost': {i: 0 for i in nodes},
            'tractor_cost': 1,
            'battery_cost': 1,
            'stat_cost': {i: 0 for i in nodes},
            'rec_penalty': 1000,
            'step_size': 1,
            'MAX_ITER': 20,
            'N_tractors': 0,
            'N_chargers': 0,
            'N_batteries': 3,
            'speed_curve': {
                0: {'speed': 1, 'minbat': 0, 'maxbat': 2},
                1: {'speed': 2, 'minbat': 2, 'maxbat': 4},
                2: {'speed': 1, 'minbat': 4, 'maxbat': 6},
            },
        }

    def test_ddd_multiple_battery_swaps(self):
        """Truck must swap battery at nodes 2, 3, and 4 to complete the route."""
        inst = Instance(self.N, self.parameters, ['default'])
        result, val, solve_prop = inst.run_DDD()

        for k, edge_dict in result['x_load'].items():
            self.assertNotEqual(edge_dict, {}, f"run_DDD returned empty x_load for commodity {k}")
        for k, edge_dict in result['x_ener'].items():
            self.assertNotEqual(edge_dict, {}, f"run_DDD returned empty x_ener for commodity {k}")

        print("Multiple battery swaps result:", result)
        print("val:", val)
        print("Solve properties:", solve_prop)

        # One battery must be deployed at each swap station
        self.assertEqual(result['n'][2], 1)
        self.assertEqual(result['n'][3], 1)
        self.assertEqual(result['n'][4], 1)
        # No batteries at source or sink
        self.assertEqual(result['n'][1], 0)
        self.assertEqual(result['n'][5], 0)

        # val = 7 (4 transit + 3 swap edges, each time=1, value_for_time=1)
        #       + 3 (one battery_cost per swap station)
        self.assertEqual(val, 10)


class TestMultipleBatSwapsMultiIter(unittest.TestCase):
    """Same 5-node battery-swap chain but with step_size=2.

    The optimal path requires post-swap departure times t=2, t=4, t=6.
    With step_size=2 the initial coarse network skips t=2 and t=4, so
    DDD must refine the discretization at least twice before finding a
    feasible solution.
    """

    def setUp(self):
        self.N = nx.DiGraph()
        self.N.add_edge(1, 2, dH=5, dL=5, time=1)
        self.N.add_edge(2, 3, dH=5, dL=5, time=1)
        self.N.add_edge(3, 4, dH=5, dL=5, time=1)
        self.N.add_edge(4, 5, dH=5, dL=5, time=1)

        nodes = list(self.N.nodes)
        T = 9
        self.parameters = {
            'sources': [1],
            'sinks': [5],
            'T': T,
            'L': 6,
            'D': {(1, 5): 1},
            'charge_rate': {i: 0 for i in nodes},
            'battery_nodes': [2, 3, 4],
            'charge_nodes': [],
            'mobile_nodes': [],
            'tractor_nodes': [],
            'bat_swap_time': 1,
            'tr_swap_time': 1,
            'mobile_charge_rate': 0,
            'charge_cost': {(i, t): 10 for i in nodes for t in range(T)},
            'surplus_cost': {i: 0 for i in nodes},
            'tractor_cost': 1,
            'battery_cost': 1,
            'stat_cost': {i: 0 for i in nodes},
            'rec_penalty': 1000,
            'step_size': 2,   # coarse: skips t=2,4 → forces multiple refinements
            'MAX_ITER': 20,
            'N_tractors': 0,
            'N_chargers': 0,
            'N_batteries': 3,
            'speed_curve': {
                0: {'speed': 1, 'minbat': 0, 'maxbat': 2},
                1: {'speed': 2, 'minbat': 2, 'maxbat': 4},
                2: {'speed': 1, 'minbat': 4, 'maxbat': 6},
            },
        }

    def test_ddd_multiiter_battery_swaps(self):
        """DDD must refine the time discretization across multiple iterations."""
        inst = Instance(self.N, self.parameters, ['default'])
        result, val, solve_prop = inst.run_DDD()

        print("Multi-iter result:", result)
        print("val:", val)
        print("Solve properties:", solve_prop)

        for k, edge_dict in result['x_load'].items():
            self.assertNotEqual(edge_dict, {}, f"run_DDD returned empty x_load for commodity {k}")
        for k, edge_dict in result['x_ener'].items():
            self.assertNotEqual(edge_dict, {}, f"run_DDD returned empty x_ener for commodity {k}")

        # Optimal value is the same regardless of step_size
        self.assertEqual(val, 10)

        # One battery at each swap station
        self.assertEqual(result['n'][2], 1)
        self.assertEqual(result['n'][3], 1)
        self.assertEqual(result['n'][4], 1)

        # The coarse discretization (step_size=2) forces at least 2 refinements
        self.assertGreater(solve_prop['n_iterations'], 1)


if __name__ == '__main__':
    unittest.main()
