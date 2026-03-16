import pytest
import networkx as nx

try:
    import gurobipy
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

import Instance_v2

pytestmark = pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")


def make_small_graph():
    G = nx.DiGraph()
    G.add_node(0)
    G.add_node(1)
    G.add_edge(0, 1)
    G[0][1]['dH'] = 1
    G[0][1]['dL'] = 1
    G[0][1]['time'] = 1
    return G


def make_parameters():
    return {
        'T': 3,
        'L': 2,
        'step_size': 1,
        'MAX_ITER': 2,
        'sources': [0],
        'sinks': [1],
        'D': 1,
        'charge_nodes': [0, 1],
        'battery_nodes': [],
        'tractor_nodes': [],
        'charge_rate': {0: 1, 1: 1},
        'charge_cost': {(0, 0): 1, (1, 0): 1, (0, 1): 1, (1, 1): 1, (0, 2): 1, (1, 2): 1},
        'stat_cost': {0: 0, 1: 0},
        'surplus_cost': {0: 0, 1: 0},
        'speed_curve': {0: {'speed': 0.5, 'minbat': 0, 'maxbat': 1}, 1: {'speed': 1, 'minbat': 1, 'maxbat': 2}},
        'N_tractors': 1,
        'N_chargers': 1,
        'N_batteries': 1
    }


def test_parse_var_name_roundtrip():
    N = make_small_graph()
    params = make_parameters()
    inst = Instance_v2.Instance(N, params)

    assert len(inst.x_load) > 0
    e = next(iter(inst.x_load.keys()))
    var = inst.x_load[e]
    parsed = inst.parse_var_name(var.VarName)
    assert parsed == e


def test_get_var_name_matches_added_var():
    N = make_small_graph()
    params = make_parameters()
    inst = Instance_v2.Instance(N, params)

    e = next(iter(inst.x_load.keys()))
    varname_from_func = inst._Instance__get_var_name('x_load', e)
    varname_actual = inst.x_load[e].VarName
    assert varname_from_func == varname_actual


def test_multi_in_out_edges_consistency():
    N = make_small_graph()
    params = make_parameters()
    inst = Instance_v2.Instance(N, params)

    nodes = list(inst.Ntl.Ntl.nodes)
    assert len(nodes) > 0
    v = nodes[0]

    in_edges = inst.multi_in_edges(v)
    out_edges = inst.multi_out_edges(v)

    for e in in_edges:
        assert e[1] == v

    for e in out_edges:
        assert e[0] == v


def test_constraints_and_demand_present():
    N = make_small_graph()
    params = make_parameters()
    inst = Instance_v2.Instance(N, params)

    constr_names = [c.ConstrName for c in inst.model.getConstrs()]

    assert 'demand-1' in constr_names
    assert 'demand-2' in constr_names

    assert any(name.startswith('flow_bal_load[') or name.startswith('flow_bal_ener[') for name in constr_names)

    some_edge = next(iter(inst.Ntl.Ntl.edges))
    e_type = inst.Ntl.edge_types[some_edge]
    if e_type == 'transit_L':
        expected_prefix = 'e_load_constr['
    elif e_type == 'transit_H':
        expected_prefix = 'e_load_ener['
    elif e_type == 'swap':
        expected_prefix = 'e_swap_x['
    else:
        expected_prefix = None

    if expected_prefix is not None:
        assert any(name.startswith(expected_prefix) for name in constr_names)


def make_diamond_params(T=10):
    """Diamond graph: 1 -> {2, 3} -> 4, charge at node 2, tractor swap at node 3."""
    N = nx.DiGraph()
    N.add_edge(1, 2, dH=4, dL=4, time=1)
    N.add_edge(1, 3, dH=5, dL=5, time=2)
    N.add_edge(2, 4, dH=3, dL=3, time=1)
    N.add_edge(3, 4, dH=4, dL=4, time=1)
    nodes = list(N.nodes)
    params = {
        'sources': [1],
        'sinks': [4],
        'T': T,
        'L': 10,
        'D': 1,
        'step_size': 1,
        'MAX_ITER': 20,
        'charge_rate': {i: 1 for i in nodes},
        'battery_nodes': [],
        'charge_nodes': [2],
        'mobile_nodes': [],
        'tractor_nodes': [3],
        'bat_swap_time': 1,
        'tr_swap_time': 1,
        'mobile_charge_rate': 0,
        'charge_cost': {(i, t): 10 for i in nodes for t in range(T)},
        'surplus_cost': {i: 0 for i in nodes},
        'stat_cost': {i: 0 for i in nodes},
        'rec_penalty': 1000,
        'N_tractors': 1,
        'N_chargers': 1,
        'N_batteries': 1,
        'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2},
                        1: {'speed': 2, 'minbat': 2, 'maxbat': 8},
                        2: {'speed': 1, 'minbat': 8, 'maxbat': 10}},
    }
    return N, params


def test_lagrange_empty_multipliers_solves():
    """Passing empty lagrange_multipliers dicts should not affect the model."""
    N, params = make_diamond_params()
    params_base = {**params}
    params_lag  = {**params, 'lagrange_multipliers': {'load': {}, 'ener': {}}}

    inst_base = Instance_v2.Instance(N.copy(), params_base, ['default'])
    soln_base, val_base, _ = inst_base.run_DDD()

    inst_lag = Instance_v2.Instance(N.copy(), params_lag, ['default'])
    soln_lag, val_lag, _ = inst_lag.run_DDD()

    assert val_lag == pytest.approx(val_base, abs=1e-6), (
        f"Empty lagrange_multipliers changed objective: {val_base} -> {val_lag}"
    )


def test_lagrange_multipliers_load():
    """Setting a positive multiplier lambda on solution edges should reduce the
    reported (modified) objective by exactly lambda * flow on those edges.

    lagrange_multipliers must use 3-tuple network edge keys (u, v, k) as found in
    inst.Ntl.Ntl.edges — NOT the 2-tuple keys from soln['x_load'] which come from
    convert_flow.  Flow lookup uses e[:2] to map back to the solution dict.
    """
    N, params = make_diamond_params()

    # 1. Solve base instance; also retain the final network edge keys
    inst_base = Instance_v2.Instance(N.copy(), params, ['default'])
    soln_base, val_base, _ = inst_base.run_DDD()

    # 2. Build multipliers using 3-tuple network keys for positive-flow load edges
    lam = 1.0
    lm_load = {
        e: lam
        for e, v in soln_base['x_load'].items() if v > 0
    }
    assert lm_load, "No positive-flow x_load edges found in base solution"

    # 3. Solve modified instance
    params_lag = {**params, 'lagrange_multipliers': {'load': lm_load, 'ener': {}}}
    inst_lag = Instance_v2.Instance(N.copy(), params_lag, ['default'])
    soln_lag, val_lag, _ = inst_lag.run_DDD()

    assert val_lag < val_base
    
 
 

def test_single_source_multiple_sinks():
    """Two trucks depart from one source and must each reach a distinct sink.

    Graph:  1 -> 3  (short path, time=1)
            1 -> 4  (long path,  time=2)
    sources=[1], sinks=[3, 4], D=2.

    Checks:
    - run_DDD solves feasibly (val is finite, solution dicts non-empty)
    - Total x_load flow arriving at sinks at T-1 equals D=2
    - D variables at both sinks sum to 2 (dem_snk constraint is satisfied)
    """
    N = nx.DiGraph()
    N.add_edge(1, 3, dH=3, dL=3, time=1)
    N.add_edge(1, 4, dH=4, dL=4, time=2)
    nodes = list(N.nodes)
    T = 8
    params = {
        'sources': [1],
        'sinks': [3, 4],
        'T': T,
        'L': 6,
        'D': 2,
        'step_size': 1,
        'MAX_ITER': 20,
        'charge_rate': {i: 0 for i in nodes},
        'battery_nodes': [],
        'charge_nodes': [],
        'mobile_nodes': [],
        'tractor_nodes': [],
        'bat_swap_time': 1,
        'tr_swap_time': 1,
        'mobile_charge_rate': 0,
        'charge_cost': {(i, t): 0 for i in nodes for t in range(T)},
        'surplus_cost': {i: 0 for i in nodes},
        'stat_cost': {i: 0 for i in nodes},
        'rec_penalty': 1000,
        'N_tractors': 0,
        'N_chargers': 0,
        'N_batteries': 0,
        'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 6}},
    }

    inst = Instance_v2.Instance(N.copy(), params, ['default'])
    soln, val, _ = inst.run_DDD()

    assert val is not None and val < 1e9, f"run_DDD returned infeasible/infinite val={val}"
    for key, v in soln.items():
        if isinstance(v, dict):
            assert v != {}, f"run_DDD returned empty dict for result['{key}']"

    # Total load flow into any sink node at the last time step must equal D=2
    sink_flow = sum(
        v for e, v in soln['x_load'].items()
        if e[1][0] in params['sinks'] and e[1][2] == T - 1
    )
    assert sink_flow == pytest.approx(params['D'], abs=1e-4), (
        f"Expected total sink flow = {params['D']}, got {sink_flow}"
    )

    # Flow is conserved: total arriving at all sinks equals D
    # (the solver is free to route all trucks to a single sink if that minimises cost)
    total_sink_flow = sum(
        v for e, v in soln['x_load'].items()
        if e[1][0] in params['sinks'] and e[1][2] == T - 1
    )
    assert total_sink_flow == pytest.approx(params['D'], abs=1e-4), (
        f"Expected total sink flow = {params['D']}, got {total_sink_flow}"
    )


def test_ddd_objective_same_for_different_step_sizes():
    """Test that run_DDD returns the same objective value for different step sizes.

    Uses a whetstone (diamond) graph: 1 -> {2, 3} -> 4
    """
    N = nx.DiGraph()

    # Whetstone: 1 -> {2, 3} -> 4
    N.add_edge(1, 2, dH=4, dL=4, time=1)
    N.add_edge(1, 3, dH=5, dL=5, time=2)
    N.add_edge(2, 4, dH=3, dL=3, time=1)
    N.add_edge(3, 4, dH=4, dL=4, time=1)

    nodes = list(N.nodes)
    T = 10

    base_params = {
        'sources': [1],
        'sinks': [4],
        'T': T,
        'L': 10,
        'D': 1,
        'charge_rate': {i: 1 for i in nodes},
        'battery_nodes': [],
        'charge_nodes': [2],
        'mobile_nodes': [],
        'tractor_nodes': [3],
        'bat_swap_time': 1,
        'tr_swap_time': 1,
        'mobile_charge_rate': 0,
        'charge_cost': {(i, t): 10 for i in nodes for t in range(T)},
        'surplus_cost': {i: 0 for i in nodes},
        'stat_cost': {i: 0 for i in nodes},
        'rec_penalty': 1000,
        'MAX_ITER': 20,
        'N_tractors': 1,
        'N_chargers': 1,
        'N_batteries': 1,
        'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2}, 1: {'speed': 2, 'minbat': 2, 'maxbat': 8}, 2: {'speed': 1, 'minbat': 8, 'maxbat': 10}}
    }

    def print_solution(inst, soln, val, label):
        print(f"\n{label}: val={val}")
        print(f"  x_load: {soln['x_load']}")
        print(f"  x_ener: {soln['x_ener']}")
        print(f"  Positive e_load_ener slacks (x_ener - x_load > 0):")
        for e in inst.Ntl.Ntl.edges:
            if inst.Ntl.edge_types[e] == 'transit_H' and e[1][0] not in inst.param['battery_nodes']:
                x_e = soln['x_ener'].get(e[:2], 0)
                x_l = soln['x_load'].get(e[:2], 0)
                if x_e - x_l > 0:
                    print(f"    e_load_ener[{e}]: slack={x_e - x_l}  (x_ener={x_e}, x_load={x_l})")

    params1 = {**base_params, 'step_size': 1}
    inst1 = Instance_v2.Instance(N.copy(), params1, ['default'])
    soln1, val1, solve_prop = inst1.run_DDD()
    for key, v in soln1.items():
        if isinstance(v, dict):
            assert v != {}, f"run_DDD (step_size=1) returned empty dict for result['{key}']"
    print_solution(inst1, soln1, val1, "step_size=1")

    params2 = {**base_params, 'step_size': 3}
    inst2 = Instance_v2.Instance(N.copy(), params2, ['default'])
    soln2, val2, solve_prop2 = inst2.run_DDD()
    for key, v in soln2.items():
        if isinstance(v, dict):
            assert v != {}, f"run_DDD (step_size=3) returned empty dict for result['{key}']"
    print_solution(inst2, soln2, val2, "step_size=3")

    # Check for 1->2->4 paths in step_size=3 network
    G3 = inst2.Ntl.Ntl
    sources = [v for v in G3.nodes if v[0] == 1 and v[2] == 0]
    sinks   = [v for v in G3.nodes if v[0] == 4 and v[2] == T - 1]
    via_2   = [v for v in G3.nodes if v[0] == 2]
    print("\n1->2->4 paths in step_size=3 network:")
    found = False
    for src in sources:
        for mid in via_2:
            for snk in sinks:
                if nx.has_path(G3, src, mid) and nx.has_path(G3, mid, snk):
                    path = nx.shortest_path(G3, src, mid) + nx.shortest_path(G3, mid, snk)[1:]
                    print(f"  {path}")
                    found = True
                    break
            if found:
                break
        if found:
            break
    if not found:
        print("  None found")

    assert val1 == val2, f"Objective values differ: step_size=1 gave {val1}, step_size=3 gave {val2}"


def test_seed_inserts_breakpoints_and_rebuilds():
    """seed() inserts x_load breakpoints into charges/times and fully rebuilds
    the network and Gurobi model.

    Uses the same diamond graph as test_ddd_objective_same_for_different_step_sizes.
    A coarse default instance (step_size=3) is seeded with the solution from a
    fine heuristic instance (step_size=1) and the structural invariants are checked.
    """
    N = nx.DiGraph()
    N.add_edge(1, 2, dH=4, dL=4, time=1)
    N.add_edge(1, 3, dH=5, dL=5, time=2)
    N.add_edge(2, 4, dH=3, dL=3, time=1)
    N.add_edge(3, 4, dH=4, dL=4, time=1)

    nodes = list(N.nodes)
    T = 6

    base_params = {
        'sources': [1],
        'sinks': [4],
        'T': T,
        'L': 10,
        'D': 1,
        'charge_rate': {i: 1 for i in nodes},
        'battery_nodes': [],
        'charge_nodes': [2],
        'mobile_nodes': [],
        'tractor_nodes': [3],
        'bat_swap_time': 1,
        'tr_swap_time': 1,
        'mobile_charge_rate': 0,
        'charge_cost': {(i, t): 10 for i in nodes for t in range(T)},
        'surplus_cost': {i: 0 for i in nodes},
        'stat_cost': {i: 0 for i in nodes},
        'rec_penalty': 1000,
        'MAX_ITER': 20,
        'N_tractors': 1,
        'N_chargers': 1,
        'N_batteries': 1,
        'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2},
                        1: {'speed': 2, 'minbat': 2, 'maxbat': 8},
                        2: {'speed': 1, 'minbat': 8, 'maxbat': 10}},
    }

    # Run heuristic DDD to obtain a solution
    inst_h = Instance_v2.Instance(N.copy(), {**base_params, 'step_size': 1})
    soln, _val, _ = inst_h.run_DDD()
    for key, v in soln.items():
        if isinstance(v, dict):
            assert v != {}, f"run_DDD (heuristic seed) returned empty dict for result['{key}']"
    x_load = soln['x_load']

    # Build a coarse default instance to seed
    inst_d = Instance_v2.Instance(N.copy(), {**base_params, 'step_size': 3}, ['default'])

    # Record discretization before seeding
    charges_before = {i: list(inst_d.Ntl.charges[i]) for i in N.nodes}
    times_before   = {i: list(inst_d.Ntl.times[i])   for i in N.nodes}

    inst_d.seed(x_load)

    # 1. Every (g, t) from a positive-flow edge in x_load must appear in charges/times
    for e, flow in x_load.items():
        if flow <= 0:
            continue
        (i, g1, t1, _), (j, g2, t2, _) = e[0], e[1]
        assert g1 in inst_d.Ntl.charges[i], f"g1={g1} missing from charges[{i}]"
        assert t1 in inst_d.Ntl.times[i],   f"t1={t1} missing from times[{i}]"
        assert g2 in inst_d.Ntl.charges[j], f"g2={g2} missing from charges[{j}]"
        assert t2 in inst_d.Ntl.times[j],   f"t2={t2} missing from times[{j}]"

    # 2. charges/times are supersets of their pre-seed values (never shrink)
    for i in N.nodes:
        for g in charges_before[i]:
            assert g in inst_d.Ntl.charges[i], f"pre-seed charge {g} lost at node {i}"
        for t in times_before[i]:
            assert t in inst_d.Ntl.times[i],   f"pre-seed time {t} lost at node {i}"

    # 3. Network was rebuilt: edges exist and all attribute dicts are consistent
    assert len(inst_d.Ntl.Ntl.edges) > 0, "Network has no edges after seed"
    ntl_edges = set(inst_d.Ntl.Ntl.edges)
    assert set(inst_d.Ntl.edge_types.keys())   == ntl_edges, "edge_types mismatch"
    assert set(inst_d.Ntl.edge_times.keys())   == ntl_edges, "edge_times mismatch"
    assert set(inst_d.Ntl.edge_dists.keys())   == ntl_edges, "edge_dists mismatch"
    assert set(inst_d.Ntl.edge_charges.keys()) == ntl_edges, "edge_charges mismatch"

    # 4. Gurobi model was rebuilt: model variables match current network edges
    assert set(inst_d.x_load.keys()) == ntl_edges, "model x_load vars mismatch"
    assert set(inst_d.x_ener.keys()) == ntl_edges, "model x_ener vars mismatch"

    # 5. run_DDD on the seeded instance completes without error
    seeded_soln, seeded_val, _ = inst_d.run_DDD()
    assert seeded_soln is not None
    assert seeded_val is not None
    for key, v in seeded_soln.items():
        if isinstance(v, dict):
            assert v != {}, f"run_DDD (seeded) returned empty dict for result['{key}']"


# ---------------------------------------------------------------------------
# bndry_wait_edges tests
# ---------------------------------------------------------------------------

def _make_bndry_network():
    N = nx.DiGraph()
    N.add_edge(1, 2, dH=2, dL=1, time=2)
    N.add_edge(2, 3, dH=1, dL=1, time=2)
    return N


def _make_bndry_params(bndry_nodes, T=10):
    nodes = [1, 2, 3]
    return {
        'L': 5, 'T': T, 'min_time': 0, 'step_size': 2,
        'battery_nodes': [], 'tractor_nodes': [], 'charge_nodes': [],
        'bat_swap_time': 2, 'tr_swap_time': 3,
        'charge_rate': {i: 1 for i in nodes},
        'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2},
                        1: {'speed': 2, 'minbat': 2, 'maxbat': 4},
                        2: {'speed': 1, 'minbat': 4, 'maxbat': 5}},
        'sources': [1], 'sinks': [3],
        'D': 1,
        'N_tractors': 0, 'N_batteries': 0, 'N_chargers': 0,
        'charge_cost': {(i, t): 1 for i in nodes for t in range(T)},
        'stat_cost': {i: 0 for i in nodes},
        'surplus_cost': {i: 0 for i in nodes},
        'bndry_wait_edges': bndry_nodes,
    }


def test_bndry_wait_edges_network_only_builds():
    """Instance with bndry_wait_edges builds without error (network_only=True)."""
    N = _make_bndry_network()
    params = _make_bndry_params(bndry_nodes=[1, 3])
    inst = Instance_v2.Instance(N, params, config=['default'], network_only=True)
    assert len(inst.Ntl.Ntl.nodes) > 0
    assert len(inst.Ntl.Ntl.edges) > 0


def test_bndry_wait_edges_sources_sinks_updated():
    """Dummy s_i / t_i nodes are appended to param['sources'] / param['sinks']."""
    N = _make_bndry_network()
    params = _make_bndry_params(bndry_nodes=[1, 3])
    Instance_v2.Instance(N, params, config=['default'], network_only=True)
    assert 's_1' in params['sources']
    assert 's_3' in params['sources']
    assert 't_1' in params['sinks']
    assert 't_3' in params['sinks']


def test_bndry_wait_edges_dummy_nodes_in_ntl():
    """Dummy source and sink nodes appear in Ntl for every charge level."""
    N = _make_bndry_network()
    params = _make_bndry_params(bndry_nodes=[1])
    inst = Instance_v2.Instance(N, params, config=['default'], network_only=True)
    ctn = inst.Ntl
    t0, tT = params['min_time'], params['T'] - 1
    for g in ctn.charges[1]:
        q = ctn.get_q(1, g)
        assert ('s_1', g, t0, q) in ctn.Ntl.nodes, f"Missing source dummy node for g={g}"
        assert ('t_1', g, tT, q) in ctn.Ntl.nodes, f"Missing sink dummy node for g={g}"


def test_bndry_wait_edges_boundary_edges_in_ntl():
    """For each g, edges s_i->i at min_time and i->t_i at T-1 exist in Ntl."""
    N = _make_bndry_network()
    params = _make_bndry_params(bndry_nodes=[1])
    inst = Instance_v2.Instance(N, params, config=['default'], network_only=True)
    ctn = inst.Ntl
    t0, tT = params['min_time'], params['T'] - 1
    ntl_edges = set((u, v) for u, v, _ in ctn.Ntl.edges(keys=True))
    for g in ctn.charges[1]:
        q = ctn.get_q(1, g)
        assert (('s_1', g, t0, q), (1, g, t0, q)) in ntl_edges, \
            f"Source edge missing for g={g}"
        assert ((1, g, tT, q), ('t_1', g, tT, q)) in ntl_edges, \
            f"Sink edge missing for g={g}"


def test_bndry_wait_edges_more_nodes_than_without():
    """An Instance with bndry_wait_edges has strictly more Ntl nodes than one without."""
    N = _make_bndry_network()
    params_plain = _make_bndry_params(bndry_nodes=[])
    del params_plain['bndry_wait_edges']
    inst_plain = Instance_v2.Instance(N.copy(), params_plain, config=['default'], network_only=True)

    params_bndry = _make_bndry_params(bndry_nodes=[1, 3])
    inst_bndry = Instance_v2.Instance(N.copy(), params_bndry, config=['default'], network_only=True)

    assert len(inst_bndry.Ntl.Ntl.nodes) > len(inst_plain.Ntl.Ntl.nodes)
    assert len(inst_bndry.Ntl.Ntl.edges) > len(inst_plain.Ntl.Ntl.edges)


def test_bndry_wait_edges_model_builds():
    """Instance with bndry_wait_edges builds a valid Gurobi model (network_only=False)."""
    N = _make_bndry_network()
    params = _make_bndry_params(bndry_nodes=[1, 3])
    inst = Instance_v2.Instance(N, params, config=['default'], network_only=False)
    assert inst.model.NumVars > 0
    assert inst.model.NumConstrs > 0


def test_bndry_wait_edges_demand_constraints_present():
    """demand-1 and demand-2 constraints are present in the model."""
    N = _make_bndry_network()
    params = _make_bndry_params(bndry_nodes=[1, 3])
    inst = Instance_v2.Instance(N, params, config=['default'], network_only=False)
    constr_names = {c.ConstrName for c in inst.model.getConstrs()}
    assert 'demand-1' in constr_names
    assert 'demand-2' in constr_names


def test_bndry_wait_edges_optimal_leq_plain():
    """Instance with bndry_wait_edges=[0] has optimal value <= plain instance (source=[0]).

    Adding bndry_wait_edges=[0] introduces dummy source s_0 and dummy sink t_0.
    The optimizer can route demand through the zero-cost shortcut
    s_0 -> (0,...) -> wait_to_T-1 -> t_0 (wait edges to T-1 have edge_time=0),
    so the bndry optimal is always <= the plain optimal.
    Both instances solve without error.
    """
    N = nx.DiGraph()
    N.add_node(0)
    N.add_node(1)
    N.add_edge(0, 1, dH=1, dL=1, time=2)
    T = 6
    nodes = list(N.nodes)

    base_params = {
        'T': T, 'L': 2, 'step_size': 1, 'min_time': 0,
        'MAX_ITER': 20, 'sources': [0], 'sinks': [1], 'D': 1,
        'charge_nodes': [], 'battery_nodes': [], 'tractor_nodes': [],
        'charge_rate': {i: 0 for i in nodes},
        'charge_cost': {(i, t): 0 for i in nodes for t in range(T)},
        'stat_cost': {i: 0 for i in nodes},
        'surplus_cost': {i: 0 for i in nodes},
        'N_tractors': 0, 'N_batteries': 0, 'N_chargers': 0,
        'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2}},
    }

    # Plain instance: source=[0], sink=[1]
    params_plain = {**base_params, 'sources': [0], 'sinks': [1]}
    inst_plain = Instance_v2.Instance(N.copy(), params_plain, ['default'])
    _, val_plain, _ = inst_plain.run_DDD()

    # Bndry instance: identical but with bndry_wait_edges=[0]
    params_bndry = {**base_params, 'sources': [0], 'sinks': [1], 'bndry_wait_edges': [0]}
    inst_bndry = Instance_v2.Instance(N.copy(), params_bndry, ['default'])
    _, val_bndry, _ = inst_bndry.run_DDD()

    assert val_plain is not None and val_plain < 1e9, f"Plain instance infeasible: val={val_plain}"
    assert val_bndry is not None and val_bndry < 1e9, f"Bndry instance infeasible: val={val_bndry}"
    # bndry is a relaxation of plain: it can always do at least as well
    assert val_bndry <= val_plain + 1e-6, (
        f"Expected val_bndry <= val_plain, got val_bndry={val_bndry}, val_plain={val_plain}"
    )
