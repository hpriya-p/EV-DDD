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
        'source': 0,
        'sink': 1,
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
    T = 6

    base_params = {
        'source': 1,
        'sink': 4,
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

    params1 = {**base_params, 'step_size': 1}
    inst1 = Instance_v2.Instance(N.copy(), params1)
    _, val1, solve_prop = inst1.run_DDD()

    params2 = {**base_params, 'step_size': 3}
    inst2 = Instance_v2.Instance(N.copy(), params2, 'default')
    _, val2, solve_prop2 = inst2.run_DDD()

    print("Heuristic properties", solve_prop, val1)
    print("Default properties", solve_prop2, val2)
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
        'source': 1,
        'sink': 4,
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
    x_load = soln['x_load']

    # Build a coarse default instance to seed
    inst_d = Instance_v2.Instance(N.copy(), {**base_params, 'step_size': 3}, 'default')

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

