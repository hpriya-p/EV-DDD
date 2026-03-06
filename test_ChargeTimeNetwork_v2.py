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
        self.config = 'default'

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

        self.ctn = ChargeTimeNetwork(self.N, self.parameters, 'default')

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

        self.ctn = ChargeTimeNetwork(self.N, self.parameters, 'default')

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

        self.ctn = ChargeTimeNetwork(self.N, self.parameters, 'default')

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
        return ChargeTimeNetwork(graph_override if graph_override is not None else N, params, 'default')

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

        self.ctn = ChargeTimeNetwork(self.N, self.parameters, 'default')

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


if __name__ == '__main__':
    unittest.main()
