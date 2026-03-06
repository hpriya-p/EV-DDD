import unittest
import networkx as nx
from ChargeTimeNetwork import ChargeTimeNetwork, RangeConstrViolation


class TestChargeTimeNetwork(unittest.TestCase):
    """Unit tests for ChargeTimeNetwork class."""

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
        # Initial charge and time lists for each node

    def test_init(self):
        """Test ChargeTimeNetwork initialization."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.init_L, self.init_T)

        self.assertEqual(ctn.N, self.N)
        self.assertEqual(ctn.param, self.parameters)
        self.assertEqual(ctn.charges, self.init_L)
        self.assertEqual(ctn.times, self.init_T)
        self.assertIsInstance(ctn.Ntl, nx.MultiDiGraph)

    def test_network_has_nodes(self):
        """Test that the constructed network has nodes."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.init_L, self.init_T)

        # The network should have nodes
        self.assertGreater(len(ctn.Ntl.nodes), 0)

    def test_network_has_edges(self):
        """Test that the constructed network has edges."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.init_L, self.init_T)

        # The network should have edges
        self.assertGreater(len(ctn.Ntl.edges), 0)

    def test_edge_types_populated(self):
        """Test that edge_types dictionary is populated."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.init_L, self.init_T)

        # edge_types should have entries
        self.assertGreater(len(ctn.edge_types), 0)

    def test_manual_check_swap_battery_node(self):
        """Test manual_check_swap for battery swap nodes."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.init_L, self.init_T)

        # Battery swap at node 2 (battery_nodes), swap time = 2
        v1 = (2, 2, 0)  # node 2, charge 2, time 0
        v2 = (2, 4, 2)  # node 2, charge 4, time 2 (after swap)

        result = ctn.manual_check_swap(v1, v2)
        self.assertTrue(result)

    def test_manual_check_swap_wrong_time(self):
        """Test manual_check_swap with wrong swap time."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.init_L, self.init_T)

        # Wrong swap time (should be 2 for battery swap)
        v1 = (2, 2, 0)
        v2 = (2, 4, 1)  # Time difference is 1, not 2

        result = ctn.manual_check_swap(v1, v2)
        self.assertFalse(result)

    def test_manual_check_swap_charge_decrease(self):
        """Test manual_check_swap when charge decreases (not a swap)."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.init_L, self.init_T)

        # Charge decreases - not a swap
        v1 = (2, 4, 0)
        v2 = (2, 2, 2)

        result = ctn.manual_check_swap(v1, v2)
        self.assertFalse(result)

    def test_manual_check_swap_different_nodes(self):
        """Test manual_check_swap between different nodes (not a swap)."""
        ctn = ChargeTimeNetwork(self.N, self.parameters, self.init_L, self.init_T)

        # Different nodes - not a swap
        v1 = (1, 2, 0)
        v2 = (2, 4, 2)

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
            'charge_rate': {1: 1, 2: 1}
        }

        self.init_L = {1: [0, 3, 5, 8], 2: [0, 2, 6, 10]}
        self.init_T = {1: [0, 2, 5, 9], 2: [0, 3, 7, 9]}

        self.ctn = ChargeTimeNetwork(self.N, self.parameters, self.init_L, self.init_T)

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
            'charge_rate': {1: 1, 2: 1, 3: 1}
        }

        self.init_L = {1: [0, 5], 2: [0, 4], 3: [0, 3]}
        self.init_T = {1: [0, 9], 2: [0, 1, 9], 3: [0, 2, 9]}

        self.ctn = ChargeTimeNetwork(self.N, self.parameters, self.init_L, self.init_T)

    def test_flow_decomposition_empty_flow(self):
        """Test flow decomposition with empty flow."""
        flow = {}
        result = self.ctn.flow_decomposition(flow)
        self.assertEqual(result, {})

    def test_flow_decomposition_single_path(self):
        """Test flow decomposition with a single path."""
        # Create a simple flow
        v1 = (1, 5, 0)
        v2 = (2, 4, 1)
        v3 = (3, 3, 2)
        v_sink = (3, 3, 9)

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
            'charge_rate': {1: 2, 2: 1}
        }

        self.init_L = {1: [0, 2, 5], 2: [0, 3, 5]}
        self.init_T = {1: [0, 2, 9], 2: [0, 4, 9]}

        self.ctn = ChargeTimeNetwork(self.N, self.parameters, self.init_L, self.init_T)

    def test_add_transit_edge(self):
        """Test adding a transit edge."""
        v1 = (1, 5, 0)
        v2 = (2, 3, 2)

        initial_edge_count = len(self.ctn.Ntl.edges)
        edge = self.ctn.add_single_edge(v1, v2, 'transit_H')

        self.assertIsNotNone(edge)
        self.assertEqual(self.ctn.edge_types[edge], 'transit_H')
        self.assertEqual(self.ctn.edge_dists[edge], 2)  # g1 - g2 = 5 - 3 = 2
        self.assertEqual(self.ctn.edge_times[edge], 2)  # t2 - t1 = 2 - 0 = 2

    def test_add_wait_edge(self):
        """Test adding a wait edge."""
        v1 = (1, 5, 0)
        v2 = (1, 5, 2)

        edge = self.ctn.add_single_edge(v1, v2, 'wait')

        self.assertIsNotNone(edge)
        self.assertEqual(self.ctn.edge_types[edge], 'wait')
        self.assertEqual(self.ctn.edge_dists[edge], 0)  # No distance for wait
        self.assertEqual(self.ctn.edge_charges[edge], 0)  # No charge for wait

    def test_add_charge_edge(self):
        """Test adding a charge edge."""
        v1 = (1, 2, 0)
        v2 = (1, 3, 1)  # Charge increases by 1

        edge = self.ctn.add_single_edge(v1, v2, 'charge')

        self.assertIsNotNone(edge)
        self.assertEqual(self.ctn.edge_types[edge], 'charge')
        # The edge_charges stores the charge increase (g2 - g1)
        self.assertGreaterEqual(self.ctn.edge_charges[edge], 1)
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
            'charge_rate': {1: 2, 2: 1, 3: 1}
        }

        self.init_L = {1: [0, 2, 5], 2: [0, 2, 4, 5], 3: [0, 3, 5]}
        self.init_T = {1: [0, 3, 14], 2: [0, 3, 6, 14], 3: [0, 5, 14]}

        self.ctn = ChargeTimeNetwork(self.N, self.parameters, self.init_L, self.init_T)

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
        valid_types = {'transit_H', 'transit_L', 'wait', 'charge', 'swap'}
        for edge, e_type in self.ctn.edge_types.items():
            self.assertIn(e_type, valid_types)


if __name__ == '__main__':
    unittest.main()
