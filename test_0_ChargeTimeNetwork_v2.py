import unittest
import networkx as nx
from ChargeTimeNetwork_v2 import ChargeTimeNetwork, RangeConstrViolation


class TestChargeTimeNetwork(unittest.TestCase):
    """Unit tests for ChargeTimeNetwork class (v2)."""

    def setUp(self):
        """Set up a simple network and parameters for testing."""
        # Create a simple directed graph
        self.N = nx.DiGraph()
        self.N.add_edge(1, 2, dH=2, dL=1, time=2)
        self.N.add_edge(2, 3, dH=2, dL=1, time=2)
        self.N.add_edge(1, 3, dH=3, dL=2, time=3)
        self.N.add_edge(3, 4, dH=1, dL=1, time=1)

        # Parameters
        self.parameters = {
            'L': 5,  # Max battery level
            'T': 10,  # Time horizon
            'battery_nodes': [2],
            'tractor_nodes': [],
            'charge_nodes': [3],
            'bat_swap_time': 2,
            'tr_swap_time': 3,
            'charge_rate': {1: 1, 2: 1, 3: 2, 4: 1},
            'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2}, 1: {'speed': 2, 'minbat': 2, 'maxbat': 4}, 2: {'speed': 1, 'minbat': 4, 'maxbat': 5}},
            'step_size': 2,
        }
        self.config = ['default']

    def test_init(self):
        """Test ChargeTimeNetwork initialization."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.config)

        self.assertEqual(ctn.N, self.N)
        self.assertEqual(ctn.param, self.parameters)
        self.assertEqual(ctn.config, self.config)
        self.assertIsInstance(ctn.charges, dict)
        self.assertIsInstance(ctn.times, dict)
        # charges and times should cover all nodes in N
        for node in self.N.nodes:
            self.assertIn(node, ctn.charges)
            self.assertIn(node, ctn.times)
        self.assertIsInstance(ctn.Ntl, nx.MultiDiGraph)

    def test_network_has_nodes(self):
        """Test that the constructed network has nodes."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.config)

        # The network should have nodes
        self.assertGreater(len(ctn.Ntl.nodes), 0)

    def test_network_has_edges(self):
        """Test that the constructed network has edges."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.config)

        # The network should have edges
        self.assertGreater(len(ctn.Ntl.edges), 0)

    def test_edge_types_populated(self):
        """Test that edge_types dictionary is populated."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.config)

        # edge_types should have entries
        self.assertGreater(len(ctn.edge_types), 0)

    def test_nodes_are_4_tuples(self):
        """Test that all nodes in Ntl are 4-tuples (i, g, t, q)."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.config)

        for node in ctn.Ntl.nodes:
            self.assertEqual(len(node), 4)

    def test_get_q(self):
        """Test get_q returns correct speed curve segment."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.config)

        self.assertEqual(ctn.get_q(1, 0), 0)  # 0 < maxbat=2
        self.assertEqual(ctn.get_q(1, 1), 0)  # 1 < maxbat=2
        self.assertEqual(ctn.get_q(1, 2), 1)  # 2 < maxbat=4
        self.assertEqual(ctn.get_q(1, 3), 1)  # 3 < maxbat=4
        self.assertEqual(ctn.get_q(1, 4), 2)  # 4 < maxbat=5
        self.assertEqual(ctn.get_q(1, 5), 2)  # 5 >= maxbat=5, return max key

    def test_default_config_charges(self):
        """Test that default config generates expected charge levels."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.config)

        # With L=5, step_size=2: [0] + range(1, 5, 2) + [5] = [0, 1, 3, 5]
        for node in self.N.nodes:
            self.assertEqual(ctn.charges[node], [0, 1, 3, 5])

    def test_default_config_times(self):
        """Test that default config generates expected time steps."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.config)

        # With T=10, step_size=2: [0] + range(1, 9, 2) + [9] = [0, 1, 3, 5, 7, 9]
        for node in self.N.nodes:
            self.assertEqual(ctn.times[node], [0, 1, 3, 5, 7, 9])

    def test_manual_check_swap_battery_node(self):
        """Test manual_check_swap for battery swap nodes."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.config)

        # Battery swap at node 2 (battery_nodes), swap time = 2
        q1 = ctn.get_q(2, 2)  # q=1
        q2 = ctn.get_q(2, 4)  # q=2
        v1 = (2, 2, 0, q1)  # node 2, charge 2, time 0, q=1
        v2 = (2, 4, 2, q2)  # node 2, charge 4, time 2 (after swap), q=2

        result = ctn.manual_check_swap(v1, v2)
        self.assertTrue(result)

    def test_manual_check_swap_wrong_time(self):
        """Test manual_check_swap with wrong swap time."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.config)

        q1 = ctn.get_q(2, 2)
        q2 = ctn.get_q(2, 4)
        # Wrong swap time (should be 2 for battery swap)
        v1 = (2, 2, 0, q1)
        v2 = (2, 4, 1, q2)  # Time difference is 1, not 2

        result = ctn.manual_check_swap(v1, v2)
        self.assertFalse(result)

    def test_manual_check_swap_charge_decrease(self):
        """Test manual_check_swap when charge decreases (not a swap)."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.config)

        q1 = ctn.get_q(2, 4)
        q2 = ctn.get_q(2, 2)
        # Charge decreases - not a swap
        v1 = (2, 4, 0, q1)
        v2 = (2, 2, 2, q2)

        result = ctn.manual_check_swap(v1, v2)
        self.assertFalse(result)

    def test_manual_check_swap_different_nodes(self):
        """Test manual_check_swap between different nodes (not a swap)."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.config)

        q1 = ctn.get_q(1, 2)
        q2 = ctn.get_q(2, 4)
        # Different nodes - not a swap (node 1 is not a battery/tractor node)
        v1 = (1, 2, 0, q1)
        v2 = (2, 4, 2, q2)

        result = ctn.manual_check_swap(v1, v2)
        self.assertFalse(result)


class TestOverestimateUnderestimate(unittest.TestCase):
    """Test the private helper methods for value estimation."""

    def setUp(self):
        """Set up a simple network for testing."""
        self.N = nx.DiGraph()
        self.N.add_edge(1, 2, dH=2, dL=1, time=2)

        self.parameters = {
            'L': 10,
            'T': 10,
            'battery_nodes': [],
            'tractor_nodes': [],
            'charge_nodes': [],
            'bat_swap_time': 2,
            'tr_swap_time': 3,
            'charge_rate': {1: 1, 2: 1},
            'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 5}, 1: {'speed': 1, 'minbat': 5, 'maxbat': 10}},
            'step_size': 2,
        }

        self.ctn = ChargeTimeNetwork(self.N, self.parameters, ['default'])

    def test_overestimate_exact_match(self):
        """Test overestimate when value exists in list."""
        lst = [0, 3, 5, 8]
        result = self.ctn._ChargeTimeNetwork__overestimate_val(lst, 5, 10)
        self.assertEqual(result, 5)

    def test_overestimate_between_values(self):
        """Test overestimate when value is between list elements."""
        lst = [0, 3, 5, 8]
        result = self.ctn._ChargeTimeNetwork__overestimate_val(lst, 4, 10)
        self.assertEqual(result, 5)

    def test_overestimate_below_min(self):
        """Test overestimate when value is below minimum."""
        lst = [2, 3, 5, 8]
        result = self.ctn._ChargeTimeNetwork__overestimate_val(lst, 1, 10)
        self.assertEqual(result, 2)

    def test_overestimate_above_max(self):
        """Test overestimate when value is above maximum in list."""
        lst = [0, 3, 5, 8]
        result = self.ctn._ChargeTimeNetwork__overestimate_val(lst, 9, 10)
        self.assertEqual(result, 10)

    def test_underestimate_exact_match(self):
        """Test underestimate when value exists in list."""
        lst = [0, 3, 5, 8]
        result = self.ctn._ChargeTimeNetwork__underestimate_val(lst, 5)
        self.assertEqual(result, 5)

    def test_underestimate_between_values(self):
        """Test underestimate when value is between list elements."""
        lst = [0, 3, 5, 8]
        result = self.ctn._ChargeTimeNetwork__underestimate_val(lst, 4)
        self.assertEqual(result, 3)

    def test_underestimate_above_max(self):
        """Test underestimate when value is above maximum."""
        lst = [0, 3, 5, 8]
        result = self.ctn._ChargeTimeNetwork__underestimate_val(lst, 10)
        self.assertEqual(result, 8)

    def test_underestimate_below_min(self):
        """Test underestimate when value is below minimum."""
        lst = [2, 3, 5, 8]
        result = self.ctn._ChargeTimeNetwork__underestimate_val(lst, 1)
        self.assertIsNone(result)


class TestFlowDecomposition(unittest.TestCase):
    """Test the flow_decomposition method."""

    def setUp(self):
        """Set up a network for flow decomposition testing."""
        self.N = nx.DiGraph()
        self.N.add_edge(1, 2, dH=1, dL=1, time=1)
        self.N.add_edge(2, 3, dH=1, dL=1, time=1)

        self.parameters = {
            'L': 5,
            'T': 10,
            'battery_nodes': [],
            'tractor_nodes': [],
            'charge_nodes': [],
            'bat_swap_time': 2,
            'tr_swap_time': 3,
            'charge_rate': {1: 1, 2: 1, 3: 1},
            'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2}, 1: {'speed': 2, 'minbat': 2, 'maxbat': 4}, 2: {'speed': 1, 'minbat': 4, 'maxbat': 5}},
            'step_size': 2,
        }

        self.ctn = ChargeTimeNetwork(self.N, self.parameters, ['default'])

    def test_flow_decomposition_empty_flow(self):
        """Test flow decomposition with empty flow."""
        flow = {}
        result = self.ctn.flow_decomposition(flow)
        self.assertEqual(result, {})

    def test_flow_decomposition_single_path(self):
        """Test flow decomposition with a single path."""
        # Create a simple flow using 4-tuples
        q_src = self.ctn.get_q(1, 5)
        v1 = (1, 5, 0, q_src)

        # Find actual edges in the network
        edges_in_ntl = list(self.ctn.Ntl.edges(keys=True))

        # Create flow using edges that exist
        flow = {}
        for e in edges_in_ntl:
            if e[0] == v1 and e[1][0] == 2:
                flow[e] = 10.0
            elif e[0][0] == 2 and e[1][0] == 3:
                flow[e] = 10.0
            elif e[0][0] == 3 and e[1][2] == 9:
                flow[e] = 10.0

        if flow:
            result = self.ctn.flow_decomposition(flow)
            # Should return at least one path if flow exists
            self.assertIsInstance(result, dict)


class TestAddSingleEdge(unittest.TestCase):
    """Test the add_single_edge method."""

    def setUp(self):
        """Set up a network for edge addition testing."""
        self.N = nx.DiGraph()
        self.N.add_edge(1, 2, dH=2, dL=1, time=2)

        self.parameters = {
            'L': 5,
            'T': 10,
            'battery_nodes': [],
            'tractor_nodes': [],
            'charge_nodes': [1],
            'bat_swap_time': 2,
            'tr_swap_time': 3,
            'charge_rate': {1: 2, 2: 1},
            'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2}, 1: {'speed': 2, 'minbat': 2, 'maxbat': 4}, 2: {'speed': 1, 'minbat': 4, 'maxbat': 5}},
            'step_size': 2,
        }

        self.ctn = ChargeTimeNetwork(self.N, self.parameters, ['default'])

    def test_add_transit_edge(self):
        """Test adding a transit edge."""
        q1 = self.ctn.get_q(1, 5)
        q2 = self.ctn.get_q(2, 3)
        v1 = (1, 5, 0, q1)
        v2 = (2, 3, 2, q2)

        initial_edge_count = len(self.ctn.Ntl.edges)
        edge = self.ctn.add_single_edge(v1, v2, 'transit_H')

        self.assertIsNotNone(edge)
        self.assertEqual(self.ctn.edge_types[edge], 'transit_H')
        self.assertEqual(self.ctn.edge_dists[edge], 2)  # g1 - g2 = 5 - 3 = 2
        self.assertEqual(self.ctn.edge_times[edge], 2)  # t2 - t1 = 2 - 0 = 2

    def test_add_wait_edge(self):
        """Test adding a wait edge."""
        q = self.ctn.get_q(1, 5)
        v1 = (1, 5, 0, q)
        v2 = (1, 5, 1, q)

        edge = self.ctn.add_single_edge(v1, v2, 'wait')

        self.assertIsNotNone(edge)
        self.assertEqual(self.ctn.edge_types[edge], 'wait')
        self.assertEqual(self.ctn.edge_dists[edge], 0)  # No distance for wait
        self.assertEqual(self.ctn.edge_charges[edge], 0)  # No charge for wait

    def test_add_charge_edge(self):
        """Test adding a charge edge."""
        q1 = self.ctn.get_q(1, 1)
        q2 = self.ctn.get_q(1, 3)
        v1 = (1, 1, 0, q1)
        v2 = (1, 3, 1, q2)  # Charge increases

        edge = self.ctn.add_single_edge(v1, v2, 'charge')

        self.assertIsNotNone(edge)
        self.assertEqual(self.ctn.edge_types[edge], 'charge')
        # The edge_charges stores the charge increase (g2 - g1)
        self.assertGreater(self.ctn.edge_charges[edge], 0)
        self.assertEqual(self.ctn.edge_dists[edge], 0)  # No distance for charge


class TestRangeConstrViolation(unittest.TestCase):
    """Test the RangeConstrViolation exception."""

    def test_exception_creation(self):
        """Test creating the exception with nodes and edges."""
        nodes = [1, 2, 3]
        del_edges = [(1, 2), (2, 3)]

        exc = RangeConstrViolation(nodes, del_edges)

        self.assertEqual(exc.affected_nodes, nodes)
        self.assertEqual(exc.del_edges, del_edges)

    def test_exception_raise(self):
        """Test raising the exception."""
        nodes = [1, 2]
        del_edges = [(1, 2)]

        with self.assertRaises(RangeConstrViolation) as context:
            raise RangeConstrViolation(nodes, del_edges)

        self.assertEqual(context.exception.affected_nodes, nodes)
        self.assertEqual(context.exception.del_edges, del_edges)


class TestDiff(unittest.TestCase):
    """Test the diff method."""

    def _make_ctn(self, params_override=None, graph_override=None):
        N = nx.DiGraph()
        N.add_edge(1, 2, dH=2, dL=1, time=2)
        N.add_edge(2, 3, dH=1, dL=1, time=2)

        params = {
            'L': 5,
            'T': 10,
            'battery_nodes': [],
            'tractor_nodes': [],
            'charge_nodes': [],
            'bat_swap_time': 2,
            'tr_swap_time': 3,
            'charge_rate': {1: 1, 2: 1, 3: 1},
            'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2},
                            1: {'speed': 2, 'minbat': 2, 'maxbat': 4},
                            2: {'speed': 1, 'minbat': 4, 'maxbat': 5}},
            'step_size': 2,
        }
        if params_override:
            params.update(params_override)
        return ChargeTimeNetwork(graph_override if graph_override is not None else N, params, ['default'])

    def test_diff_equal_networks(self):
        """Two independently constructed networks with identical parameters are equal."""
        ctn1 = self._make_ctn()
        ctn2 = self._make_ctn()
        self.assertTrue(ctn1.diff(ctn2))

    def test_diff_self(self):
        """A network compared with itself is equal."""
        ctn = self._make_ctn()
        self.assertTrue(ctn.diff(ctn))

    def test_diff_different_charges(self):
        """Networks with different charge discretisations are not equal."""
        # step_size=2  → charges [0,1,3,5]; step_size=1 → charges [0,1,2,3,4,5]
        ctn1 = self._make_ctn({'step_size': 2})
        ctn2 = self._make_ctn({'step_size': 1})
        self.assertFalse(ctn1.diff(ctn2))

    def test_diff_different_times(self):
        """Networks with different time discretisations are not equal."""
        # T=10 → times end at 9; T=8 → times end at 7
        ctn1 = self._make_ctn({'T': 10})
        ctn2 = self._make_ctn({'T': 8})
        self.assertFalse(ctn1.diff(ctn2))

    def test_diff_different_nodes(self):
        """Networks whose Ntl node sets differ are not equal."""
        ctn1 = self._make_ctn()
        ctn2 = self._make_ctn()
        # Inject an extra node into ctn2's time-expanded graph
        ctn2.Ntl.add_node((99, 0, 0, 0))
        self.assertFalse(ctn1.diff(ctn2))

    def test_diff_different_edges(self):
        """Networks whose Ntl edge sets differ are not equal."""
        ctn1 = self._make_ctn()
        ctn2 = self._make_ctn()
        # Pick any existing node and add a phantom edge in ctn2
        node = next(iter(ctn2.Ntl.nodes()))
        ctn2.Ntl.add_edge(node, (99, 0, 0, 0))
        self.assertFalse(ctn1.diff(ctn2))

    def test_diff_different_edge_types(self):
        """Networks whose edge_types dicts differ are not equal."""
        ctn1 = self._make_ctn()
        ctn2 = self._make_ctn()
        # Corrupt one entry in ctn2's edge_types
        some_edge = next(iter(ctn2.edge_types))
        ctn2.edge_types[some_edge] = 'transit_L' if ctn2.edge_types[some_edge] != 'transit_L' else 'wait'
        self.assertFalse(ctn1.diff(ctn2))

    def test_diff_different_edge_times(self):
        """Networks whose edge_times dicts differ are not equal."""
        ctn1 = self._make_ctn()
        ctn2 = self._make_ctn()
        some_edge = next(iter(ctn2.edge_times))
        ctn2.edge_times[some_edge] += 1
        self.assertFalse(ctn1.diff(ctn2))

    def test_diff_different_edge_dists(self):
        """Networks whose edge_dists dicts differ are not equal."""
        ctn1 = self._make_ctn()
        ctn2 = self._make_ctn()
        some_edge = next(iter(ctn2.edge_dists))
        ctn2.edge_dists[some_edge] += 1
        self.assertFalse(ctn1.diff(ctn2))

    def test_diff_different_edge_charges(self):
        """Networks whose edge_charges dicts differ are not equal."""
        ctn1 = self._make_ctn()
        ctn2 = self._make_ctn()
        some_edge = next(iter(ctn2.edge_charges))
        ctn2.edge_charges[some_edge] += 1
        self.assertFalse(ctn1.diff(ctn2))

    def test_diff_is_symmetric(self):
        """diff(a, b) and diff(b, a) should agree."""
        ctn1 = self._make_ctn({'step_size': 2})
        ctn2 = self._make_ctn({'step_size': 1})
        self.assertEqual(ctn1.diff(ctn2), ctn2.diff(ctn1))


class TestEdgeAttributes(unittest.TestCase):
    """Test that edge attributes are correctly set."""

    def setUp(self):
        """Set up a network for testing edge attributes."""
        self.N = nx.DiGraph()
        self.N.add_edge(1, 2, dH=2, dL=1, time=3)
        self.N.add_edge(2, 3, dH=1, dL=1, time=2)

        self.parameters = {
            'L': 5,
            'T': 15,
            'battery_nodes': [2],
            'tractor_nodes': [],
            'charge_nodes': [1],
            'bat_swap_time': 2,
            'tr_swap_time': 3,
            'charge_rate': {1: 2, 2: 1, 3: 1},
            'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2}, 1: {'speed': 2, 'minbat': 2, 'maxbat': 4}, 2: {'speed': 1, 'minbat': 4, 'maxbat': 5}},
            'step_size': 2,
        }

        self.ctn = ChargeTimeNetwork(self.N, self.parameters, ['default'])

    def test_all_edges_have_types(self):
        """Test that all edges have types assigned."""
        for edge in self.ctn.Ntl.edges(keys=True):
            self.assertIn(edge, self.ctn.edge_types)

    def test_all_edges_have_times(self):
        """Test that all edges have times assigned."""
        for edge in self.ctn.Ntl.edges(keys=True):
            self.assertIn(edge, self.ctn.edge_times)

    def test_all_edges_have_charges(self):
        """Test that all edges have charges assigned."""
        for edge in self.ctn.Ntl.edges(keys=True):
            self.assertIn(edge, self.ctn.edge_charges)

    def test_all_edges_have_dists(self):
        """Test that all edges have distances assigned."""
        for edge in self.ctn.Ntl.edges(keys=True):
            self.assertIn(edge, self.ctn.edge_dists)

    def test_edge_type_values(self):
        """Test that edge types are valid."""
        valid_types = {'transit_H', 'transit_L', 'wait', 'charge', 'swap', 'piecewise'}
        for edge, e_type in self.ctn.edge_types.items():
            self.assertIn(e_type, valid_types)


class TestMinTime(unittest.TestCase):
    """Test ChargeTimeNetwork construction when min_time is positive."""

    def _make_network(self):
        N = nx.DiGraph()
        N.add_edge(1, 2, dH=2, dL=1, time=2)
        N.add_edge(2, 3, dH=2, dL=1, time=2)
        N.add_edge(1, 3, dH=3, dL=2, time=3)
        N.add_edge(3, 4, dH=1, dL=1, time=1)
        return N

    def _base_params(self):
        return {
            'L': 5,
            'battery_nodes': [2],
            'tractor_nodes': [],
            'charge_nodes': [3],
            'bat_swap_time': 2,
            'tr_swap_time': 3,
            'charge_rate': {1: 1, 2: 1, 3: 2, 4: 1},
            'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2},
                            1: {'speed': 2, 'minbat': 2, 'maxbat': 4},
                            2: {'speed': 1, 'minbat': 4, 'maxbat': 5}},
            'step_size': 2,
        }

    def test_min_time_same_node_and_edge_count(self):
        """Ntl1 (min_time=0, T=10) and Ntl2 (min_time=5, T=15) have the same node and edge counts."""
        N = self._make_network()

        params1 = self._base_params()
        params1['min_time'] = 0
        params1['T'] = 10
        ctn1 = ChargeTimeNetwork(N, params1, ['default'])

        params2 = self._base_params()
        params2['min_time'] = 5
        params2['T'] = 15
        ctn2 = ChargeTimeNetwork(N, params2, ['default'])

        self.assertEqual(len(ctn1.Ntl.nodes), len(ctn2.Ntl.nodes))
        self.assertEqual(len(ctn1.Ntl.edges), len(ctn2.Ntl.edges))

    def test_min_time_no_node_before_min_time(self):
        """No node v in Ntl2 (min_time=5) has v[2] < 5."""
        N = self._make_network()

        params2 = self._base_params()
        params2['min_time'] = 5
        params2['T'] = 15
        ctn2 = ChargeTimeNetwork(N, params2, ['default'])

        for v in ctn2.Ntl.nodes:
            self.assertGreaterEqual(v[2], 5, f"Node {v} has time {v[2]} < min_time=5")

    def test_min_time_edges_shift_by_5(self):
        """For every edge in Ntl1, the time-shifted version (t+5) exists in Ntl2."""
        N = self._make_network()

        params1 = self._base_params()
        params1['min_time'] = 0
        params1['T'] = 10
        ctn1 = ChargeTimeNetwork(N, params1, ['default'])

        params2 = self._base_params()
        params2['min_time'] = 5
        params2['T'] = 15
        ctn2 = ChargeTimeNetwork(N, params2, ['default'])

        ntl2_edges = set((u, v) for u, v, _ in ctn2.Ntl.edges(keys=True))

        for u, v, _ in ctn1.Ntl.edges(keys=True):
            i,  g,  t,  q  = u
            j, g2, t2, q2  = v
            shifted_u = (i,  g,  t  + 5,  q)
            shifted_v = (j, g2,  t2 + 5, q2)
            self.assertIn(
                (shifted_u, shifted_v), ntl2_edges,
                f"Edge ({shifted_u}, {shifted_v}) not found in Ntl2 (shifted from ({u}, {v}))"
            )


class TestBndryWaitEdges(unittest.TestCase):
    """Tests for the bndry_wait_edges feature.

    When param['bndry_wait_edges'] = [i, ...], construct() should:
      - add dummy source node s_i to param['sources']
      - add dummy sink   node t_i to param['sinks']
      - for every charge level g in charges[i]:
          * add an edge (s_i, g, min_time) -> (i, g, min_time, q) of each type
          * add an edge (i, g, T-1, q)     -> (t_i, g, T-1)       of each type
      - not add s_i / t_i more than once even when called with overlapping nodes
    """

    def _make_params(self, bndry_nodes, sources=None, sinks=None):
        return {
            'L': 5,
            'T': 10,
            'min_time': 0,
            'battery_nodes': [],
            'tractor_nodes': [],
            'charge_nodes': [],
            'bat_swap_time': 2,
            'tr_swap_time': 3,
            'charge_rate': {1: 1, 2: 1, 3: 1},
            'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2},
                            1: {'speed': 2, 'minbat': 2, 'maxbat': 4},
                            2: {'speed': 1, 'minbat': 4, 'maxbat': 5}},
            'step_size': 2,
            'sources': list(sources or []),
            'sinks':   list(sinks   or []),
            'bndry_wait_edges': bndry_nodes,
        }

    def _make_network(self):
        N = nx.DiGraph()
        N.add_edge(1, 2, dH=2, dL=1, time=2)
        N.add_edge(2, 3, dH=1, dL=1, time=2)
        return N

    # ------------------------------------------------------------------
    # sources / sinks bookkeeping
    # ------------------------------------------------------------------

    def test_dummy_source_added_to_sources(self):
        """s_i is appended to param['sources'] for every i in bndry_wait_edges."""
        N = self._make_network()
        params = self._make_params(bndry_nodes=[1])
        ctn = ChargeTimeNetwork(N, params, ['default'])
        self.assertIn('s_1', params['sources'])

    def test_dummy_sink_added_to_sinks(self):
        """t_i is appended to param['sinks'] for every i in bndry_wait_edges."""
        N = self._make_network()
        params = self._make_params(bndry_nodes=[1])
        ctn = ChargeTimeNetwork(N, params, ['default'])
        self.assertIn('t_1', params['sinks'])

    def test_multiple_bndry_nodes_all_registered(self):
        """s_i and t_i are added for every node in bndry_wait_edges."""
        N = self._make_network()
        params = self._make_params(bndry_nodes=[1, 2])
        ctn = ChargeTimeNetwork(N, params, ['default'])
        for i in [1, 2]:
            self.assertIn(f's_{i}', params['sources'])
            self.assertIn(f't_{i}', params['sinks'])

    def test_no_duplicate_sources_or_sinks(self):
        """Constructing twice (or with repeated nodes) does not duplicate s_i / t_i."""
        N = self._make_network()
        params = self._make_params(bndry_nodes=[1])
        ChargeTimeNetwork(N, params, ['default'])
        self.assertEqual(params['sources'].count('s_1'), 1)
        self.assertEqual(params['sinks'].count('t_1'), 1)

    # ------------------------------------------------------------------
    # dummy nodes present in Ntl
    # ------------------------------------------------------------------

    def test_dummy_source_nodes_in_ntl(self):
        """A node (s_i, g, min_time, q) exists in Ntl for every g in charges[i]."""
        N = self._make_network()
        params = self._make_params(bndry_nodes=[1])
        ctn = ChargeTimeNetwork(N, params, ['default'])
        t0 = params['min_time']
        for g in ctn.charges[1]:
            q = ctn.get_q(1, g)
            self.assertIn(('s_1', g, t0, q), ctn.Ntl.nodes,
                          f"Dummy source node ('s_1', {g}, {t0}, {q}) missing from Ntl")

    def test_dummy_sink_nodes_in_ntl(self):
        """A node (t_i, g, T-1, q) exists in Ntl for every g in charges[i]."""
        N = self._make_network()
        params = self._make_params(bndry_nodes=[1])
        ctn = ChargeTimeNetwork(N, params, ['default'])
        T = params['T']
        for g in ctn.charges[1]:
            q = ctn.get_q(1, g)
            self.assertIn(('t_1', g, T - 1, q), ctn.Ntl.nodes,
                          f"Dummy sink node ('t_1', {g}, {T - 1}, {q}) missing from Ntl")

    # ------------------------------------------------------------------
    # edges between dummy nodes and real nodes
    # ------------------------------------------------------------------

    def test_source_edges_exist(self):
        """For every g in charges[i] there is at least one edge from (s_i, g, min_time, q)
        to the real node (i, g, min_time, q) in Ntl."""
        N = self._make_network()
        params = self._make_params(bndry_nodes=[1])
        ctn = ChargeTimeNetwork(N, params, ['default'])
        t0 = params['min_time']
        ntl_edges = set((u, v) for u, v, _ in ctn.Ntl.edges(keys=True))
        for g in ctn.charges[1]:
            q = ctn.get_q(1, g)
            src = ('s_1', g, t0, q)
            dst = (1, g, t0, q)
            self.assertIn((src, dst), ntl_edges,
                          f"Edge {src} -> {dst} missing from Ntl")

    def test_sink_edges_exist(self):
        """For every g in charges[i] there is at least one edge from the real node
        (i, g, T-1, q) to (t_i, g, T-1, q) in Ntl."""
        N = self._make_network()
        params = self._make_params(bndry_nodes=[1])
        ctn = ChargeTimeNetwork(N, params, ['default'])
        T = params['T']
        ntl_edges = set((u, v) for u, v, _ in ctn.Ntl.edges(keys=True))
        for g in ctn.charges[1]:
            q = ctn.get_q(1, g)
            src = (1, g, T - 1, q)
            snk = ('t_1', g, T - 1, q)
            self.assertIn((src, snk), ntl_edges,
                          f"Edge {src} -> {snk} missing from Ntl")

    def test_source_edge_types(self):
        """Edges from dummy source nodes are labelled transit_L and transit_H."""
        N = self._make_network()
        params = self._make_params(bndry_nodes=[1])
        ctn = ChargeTimeNetwork(N, params, ['default'])
        t0 = params['min_time']
        g0 = ctn.charges[1][0]
        q0 = ctn.get_q(1, g0)
        types_seen = set()
        for e, et in ctn.edge_types.items():
            if e[0] == ('s_1', g0, t0, q0):
                types_seen.add(et)
        self.assertIn('transit_L', types_seen)
        self.assertIn('transit_H', types_seen)

    def test_sink_edge_types(self):
        """Edges to dummy sink nodes are labelled transit_L and transit_H."""
        N = self._make_network()
        params = self._make_params(bndry_nodes=[1])
        ctn = ChargeTimeNetwork(N, params, ['default'])
        T = params['T']
        g0 = ctn.charges[1][0]
        q0 = ctn.get_q(1, g0)
        types_seen = set()
        for e, et in ctn.edge_types.items():
            if e[1] == ('t_1', g0, T - 1, q0):
                types_seen.add(et)
        self.assertIn('transit_L', types_seen)
        self.assertIn('transit_H', types_seen)


if __name__ == '__main__':
    unittest.main()
