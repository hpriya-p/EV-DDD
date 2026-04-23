"""
Unit tests for instances where speed_curve has more than 3 keys (4+ segments).

All speed curves here have >= 4 entries so that the piecewise segment logic
is exercised beyond the 3-segment case covered by the existing test suite.
"""
import pytest
import networkx as nx

try:
    import gurobipy
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from ChargeTimeNetwork_v2 import ChargeTimeNetwork
import Instance_v2

pytestmark = pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SPEED_CURVE_4 = {
    0: {'speed': 0.5, 'minbat': 0,  'maxbat': 3},
    1: {'speed': 1.0, 'minbat': 3,  'maxbat': 6},
    2: {'speed': 2.0, 'minbat': 6,  'maxbat': 9},
    3: {'speed': 1.5, 'minbat': 9,  'maxbat': 12},
}

SPEED_CURVE_5 = {
    0: {'speed': 0.5, 'minbat': 0,  'maxbat': 2},
    1: {'speed': 1.0, 'minbat': 2,  'maxbat': 4},
    2: {'speed': 2.0, 'minbat': 4,  'maxbat': 6},
    3: {'speed': 1.5, 'minbat': 6,  'maxbat': 8},
    4: {'speed': 0.5, 'minbat': 8,  'maxbat': 10},
}


def _make_simple_graph():
    """Linear graph: 1 -> 2 -> 3 with a charge node at 2."""
    N = nx.DiGraph()
    N.add_edge(1, 2, dH=2, dL=2, time=1)
    N.add_edge(2, 3, dH=2, dL=2, time=1)
    return N


def _base_params(N, speed_curve, T=15, L=12):
    nodes = list(N.nodes)
    return {
        'sources': [1],
        'sinks': [3],
        'T': T,
        'L': L,
        'D': {1: (1, 0), 3: (0, 1)},
        'step_size': 1,
        'MAX_ITER': 20,
        'charge_rate': {i: 2 for i in nodes},
        'battery_nodes': [],
        'charge_nodes': [2],
        'mobile_nodes': [],
        'tractor_nodes': [],
        'bat_swap_time': 1,
        'tr_swap_time': 1,
        'mobile_charge_rate': 0,
        'charge_cost': {(i, t): 1 for i in nodes for t in range(T)},
        'surplus_cost': {i: 0 for i in nodes},
        'stat_cost': {i: 0 for i in nodes},
        'rec_penalty': 1000,
        'N_tractors': 0,
        'N_chargers': 1,
        'N_batteries': 0,
        'speed_curve': speed_curve,
    }


def _make_ctn(speed_curve, T=15, L=12, step_size=1):
    N = _make_simple_graph()
    nodes = list(N.nodes)
    params = {
        'L': L,
        'T': T,
        'step_size': step_size,
        'battery_nodes': [],
        'tractor_nodes': [],
        'charge_nodes': [2],
        'bat_swap_time': 1,
        'tr_swap_time': 1,
        'charge_rate': {i: 2 for i in nodes},
        'speed_curve': speed_curve,
        'sources': [1],
        'sinks': [3],
    }
    return ChargeTimeNetwork(N, params, ['default'])


# ---------------------------------------------------------------------------
# ChargeTimeNetwork (CTN) tests — 4-segment speed curve
# ---------------------------------------------------------------------------

class TestCTN4Segments:
    """CTN structural tests with a 4-segment speed_curve."""

    def setup_method(self):
        self.ctn = _make_ctn(SPEED_CURVE_4, L=12)

    def test_network_has_nodes(self):
        assert len(self.ctn.Ntl.nodes) > 0

    def test_network_has_edges(self):
        assert len(self.ctn.Ntl.edges) > 0

    def test_nodes_are_4_tuples(self):
        for node in self.ctn.Ntl.nodes:
            assert len(node) == 4, f"Node {node} is not a 4-tuple"

    def test_node_q_values_within_speed_curve_keys(self):
        """The q component of every node must be a valid speed_curve key."""
        valid_qs = set(SPEED_CURVE_4.keys())
        for node in self.ctn.Ntl.nodes:
            _, _, _, q = node
            assert q in valid_qs, f"Node {node} has q={q} not in speed_curve keys"

    def test_edge_type_consistency(self):
        """edge_types, edge_times, edge_dists, edge_charges must cover all edges."""
        ntl_edges = set(self.ctn.Ntl.edges)
        assert set(self.ctn.edge_types.keys())   == ntl_edges
        assert set(self.ctn.edge_times.keys())   == ntl_edges
        assert set(self.ctn.edge_dists.keys())   == ntl_edges
        assert set(self.ctn.edge_charges.keys()) == ntl_edges

    def test_piecewise_edges_exist_at_all_inner_thresholds(self):
        """A piecewise edge should exist at each internal threshold (all but last segment)."""
        piecewise_edges = [e for e, t in self.ctn.edge_types.items() if t == 'piecewise']
        piecewise_g_values = {e[0][1] for e in piecewise_edges}
        for q in sorted(SPEED_CURVE_4.keys()):
            g_thresh = SPEED_CURVE_4[q]['maxbat']
            if g_thresh < 12:  # not the outermost boundary (L)
                assert g_thresh in piecewise_g_values, (
                    f"Expected piecewise edge at g={g_thresh} (threshold of segment {q})"
                )

    def test_charge_edges_do_not_cross_segment_boundaries(self):
        """Charge edges must start and end within the same speed_curve segment."""
        thresholds = sorted(SPEED_CURVE_4[q]['maxbat'] for q in SPEED_CURVE_4)
        charge_edges = [e for e, t in self.ctn.edge_types.items() if t == 'charge']
        for e in charge_edges:
            g_start = e[0][1]
            g_end   = e[1][1]
            q_start = self.ctn.get_q(e[0][0], g_start)
            ub = SPEED_CURVE_4[q_start]['maxbat']
            assert g_end <= ub, (
                f"Charge edge {e} crosses segment boundary: g_start={g_start}, "
                f"g_end={g_end}, segment UB={ub}"
            )


class TestGetQ4Segments:
    """get_q correctness for a 4-segment speed curve."""

    def setup_method(self):
        self.ctn = _make_ctn(SPEED_CURVE_4, L=12)

    @pytest.mark.parametrize("g,expected_q", [
        (0,  0),   # 0 < 3
        (1,  0),   # 1 < 3
        (2,  0),   # 2 < 3
        (3,  1),   # 3 < 6 (first segment maxbat=3, so g=3 is in segment 1)
        (4,  1),   # 4 < 6
        (5,  1),   # 5 < 6
        (6,  2),   # 6 < 9
        (7,  2),   # 7 < 9
        (8,  2),   # 8 < 9
        (9,  3),   # 9 < 12
        (10, 3),   # 10 < 12
        (11, 3),   # 11 < 12
        (12, 3),   # 12 >= 12, returns max key
    ])
    def test_get_q_mapping(self, g, expected_q):
        node = 1  # any node; get_q ignores node identity
        assert self.ctn.get_q(node, g) == expected_q, (
            f"get_q(node, {g}) expected {expected_q}, got {self.ctn.get_q(node, g)}"
        )


# ---------------------------------------------------------------------------
# ChargeTimeNetwork (CTN) tests — 5-segment speed curve
# ---------------------------------------------------------------------------

class TestCTN5Segments:
    """CTN structural tests with a 5-segment speed_curve."""

    def setup_method(self):
        self.ctn = _make_ctn(SPEED_CURVE_5, L=10)

    def test_network_has_nodes(self):
        assert len(self.ctn.Ntl.nodes) > 0

    def test_network_has_edges(self):
        assert len(self.ctn.Ntl.edges) > 0

    def test_node_q_values_within_speed_curve_keys(self):
        valid_qs = set(SPEED_CURVE_5.keys())
        for node in self.ctn.Ntl.nodes:
            _, _, _, q = node
            assert q in valid_qs, f"Node {node} has q={q} not in speed_curve keys"

    def test_piecewise_edges_exist_at_all_inner_thresholds(self):
        piecewise_edges = [e for e, t in self.ctn.edge_types.items() if t == 'piecewise']
        piecewise_g_values = {e[0][1] for e in piecewise_edges}
        for q in sorted(SPEED_CURVE_5.keys()):
            g_thresh = SPEED_CURVE_5[q]['maxbat']
            if g_thresh < 10:  # L=10
                assert g_thresh in piecewise_g_values, (
                    f"Expected piecewise edge at g={g_thresh} (threshold of segment {q})"
                )

    def test_get_q_all_segments(self):
        """Each segment's midpoint should map to that segment."""
        ctn = self.ctn
        # midpoints within each segment
        cases = [
            (0, 0),   # g=0 -> segment 0 (maxbat=2)
            (1, 0),
            (2, 1),   # g=2 -> segment 1 (maxbat=4)
            (3, 1),
            (4, 2),   # g=4 -> segment 2 (maxbat=6)
            (5, 2),
            (6, 3),   # g=6 -> segment 3 (maxbat=8)
            (7, 3),
            (8, 4),   # g=8 -> segment 4 (maxbat=10)
            (9, 4),
            (10, 4),  # g=10 >= maxbat, returns max key
        ]
        node = 1
        for g, expected_q in cases:
            assert ctn.get_q(node, g) == expected_q, (
                f"get_q(node, {g}) expected {expected_q}, got {ctn.get_q(node, g)}"
            )


# ---------------------------------------------------------------------------
# Instance (end-to-end) tests — 4-segment speed curve
# ---------------------------------------------------------------------------

class TestInstance4Segments:
    """Instance-level tests (build + solve) with a 4-segment speed_curve."""

    def _make_instance(self, **overrides):
        N = _make_simple_graph()
        params = _base_params(N, SPEED_CURVE_4, T=15, L=12)
        params.update(overrides)
        return Instance_v2.Instance(N.copy(), params, ['default'])

    def test_instance_builds_without_error(self):
        inst = self._make_instance()
        assert inst is not None
        assert len(inst.Ntl.Ntl.nodes) > 0

    def test_run_ddd_feasible(self):
        inst = self._make_instance()
        soln, val, _ = inst.run_DDD()
        assert val is not None and val < 1e9, f"run_DDD returned infeasible val={val}"

    def test_solution_dicts_nonempty(self):
        inst = self._make_instance()
        soln, val, _ = inst.run_DDD()
        for key, v in soln.items():
            if isinstance(v, dict):
                assert v != {}, f"run_DDD returned empty dict for result['{key}']"

    def test_x_load_vars_cover_all_network_edges(self):
        inst = self._make_instance()
        ntl_edges = set(inst.Ntl.Ntl.edges)
        assert set(inst.x_load.keys()) == ntl_edges

    def test_x_ener_vars_cover_all_network_edges(self):
        inst = self._make_instance()
        ntl_edges = set(inst.Ntl.Ntl.edges)
        assert set(inst.x_ener.keys()) == ntl_edges


# ---------------------------------------------------------------------------
# Instance (end-to-end) tests — 5-segment speed curve
# ---------------------------------------------------------------------------

class TestInstance5Segments:
    """Instance-level tests (build + solve) with a 5-segment speed_curve."""

    def _make_instance(self, **overrides):
        N = _make_simple_graph()
        params = _base_params(N, SPEED_CURVE_5, T=15, L=10)
        params.update(overrides)
        return Instance_v2.Instance(N.copy(), params, ['default'])

    def test_instance_builds_without_error(self):
        inst = self._make_instance()
        assert inst is not None
        assert len(inst.Ntl.Ntl.nodes) > 0

    def test_run_ddd_feasible(self):
        inst = self._make_instance()
        soln, val, _ = inst.run_DDD()
        assert val is not None and val < 1e9, f"run_DDD returned infeasible val={val}"

    def test_solution_conserves_flow_at_sinks(self):
        """Total x_load flow arriving at the sink must equal the demand."""
        N = _make_simple_graph()
        params = _base_params(N, SPEED_CURVE_5, T=15, L=10)
        inst = Instance_v2.Instance(N.copy(), params, ['default'])
        soln, val, _ = inst.run_DDD()
        T = params['T']
        sink_flow = sum(
            v for e, v in soln['x_load'].items()
            if e[1][0] == 3 and e[1][2] == T - 1
        )
        assert sink_flow == pytest.approx(params['D'][3][1], abs=1e-4), (
            f"Sink flow {sink_flow} != demand {params['D'][3][1]}"
        )



# ---------------------------------------------------------------------------
# Seed tests with 4-segment speed curve
# ---------------------------------------------------------------------------

def _make_seed_instance_4seg(step_size=3):
    """Return a (coarse instance, fine solution, N) triple using a 4-segment curve."""
    N = nx.DiGraph()
    N.add_edge(1, 2, dH=2, dL=2, time=1)
    N.add_edge(2, 3, dH=2, dL=2, time=1)
    nodes = list(N.nodes)
    T, L = 15, 12
    base_params = {
        'sources': [1], 'sinks': [3], 'T': T, 'L': L, 'D': {1: (1, 0), 3: (0, 1)},
        'step_size': 1,
        'MAX_ITER': 100,
        'charge_rate': {i: 2 for i in nodes},
        'battery_nodes': [], 'charge_nodes': [2], 'mobile_nodes': [], 'tractor_nodes': [],
        'bat_swap_time': 1, 'tr_swap_time': 1, 'mobile_charge_rate': 0,
        'charge_cost': {(i, t): 1 for i in nodes for t in range(T)},
        'surplus_cost': {i: 0 for i in nodes}, 'stat_cost': {i: 0 for i in nodes},
        'rec_penalty': 1000,
        'N_tractors': 0, 'N_chargers': 1, 'N_batteries': 0,
        'speed_curve': SPEED_CURVE_4,
    }
    inst_fine = Instance_v2.Instance(N.copy(), {**base_params, 'step_size': 1})
    soln, val, _ = inst_fine.run_DDD()
    inst = Instance_v2.Instance(N.copy(), {**base_params, 'step_size': step_size}, ['default'])
    soln2, val2, prop = inst.run_DDD()
    if val != val2:
        print(soln)
        print(soln2)
        print(prop)
    assert val == val2, f"Expected same objective value for step_size=1 and step_size={step_size}, got {val} vs {val2}"
    return inst, soln, N


def test_seed_with_4_segment_curve_breakpoints_inserted():
    """seed() must insert all (g, t) pairs from the fine solution into Ntl."""
    inst, soln, _ = _make_seed_instance_4seg()
    inst.seed(soln)
    for e in soln['x_load']:
        (i, g1, t1, _), (j, g2, t2, _) = e[0], e[1]
        assert g1 in inst.Ntl.charges[i]
        assert t1 in inst.Ntl.times[i]
        assert g2 in inst.Ntl.charges[j]
        assert t2 in inst.Ntl.times[j]


def test_seed_with_4_segment_curve_run_ddd_feasible():
    """run_DDD on a seeded coarse instance (4-segment curve) must be feasible."""
    inst, soln, _ = _make_seed_instance_4seg()
    inst.seed(soln)
    _, val, _ = inst.run_DDD()
    assert val is not None and val < 1e9, f"Seeded 4-segment run_DDD infeasible: {val}"


def test_seed_with_4_segment_curve_preserves_breakpoints():
    """seed() must not drop any charges/times that existed before the call."""
    inst, soln, N = _make_seed_instance_4seg()
    charges_before = {i: list(inst.Ntl.charges[i]) for i in N.nodes}
    times_before   = {i: list(inst.Ntl.times[i])   for i in N.nodes}
    inst.seed(soln)
    for i in N.nodes:
        for g in charges_before[i]:
            assert g in inst.Ntl.charges[i]
        for t in times_before[i]:
            assert t in inst.Ntl.times[i]
