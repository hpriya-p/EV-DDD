import pytest
import networkx as nx

GUROBI_AVAILABLE = True
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
        'D': {0: (1, 0), 1: (0, 1)},
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

    for k, v in params['D'].items():
        assert f"demand_{k}_src" in constr_names
        assert f"demand_{k}_dest" in constr_names

    assert any(name.startswith('flow_bal_load[') or name.startswith('flow_bal_ener[') for name in constr_names)

    some_edge = next(iter(inst.Ntl.Ntl.edges(keys=True)))
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
        'D': {1: (1, 0), 4: (0, 1)},
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
        'D': {1: (2, 0),  # supply of 2 at node 1
               3: (0, 1),  # demand of 1 at node 3
               4: (0, 1)}, # demand of 1 at node 4
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
    for k, v in params['D'].items():
        if k in params['sinks']:
            sink_flow = sum(
                v for e, v in soln['x_load'].items()
                if e[1][0] == k and e[1][2] == T - 1
            )
            assert sink_flow == pytest.approx(params['D'][k][1], abs=1e-4), (
                f"Expected sink {k} flow = {params['D'][k][1]}, got {sink_flow}"
            )
        if k in params['sources']:
            source_flow = sum(
                v for e, v in soln['x_load'].items()
                if e[0][0] == k and e[0][2] == 0
            )
            assert source_flow == pytest.approx(params['D'][k][0], abs=1e-4), (
                f"Expected source {k} flow = {params['D'][k][0]}, got {source_flow}"
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
        'D': {1: (1, 0),  # supply of 1 at node 1
               4: (0, 1)}, # demand of 1 at node 4
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
        for e in inst.Ntl.Ntl.edges(keys=True):
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
        'D': {1: (1, 0), 4: (0, 1)},
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

    inst_d.seed(soln)

    # 1. Every (g, t) from a positive-flow edge in x_load must appear in charges/times
    for e, flow in x_load.items():
        if flow <= 0:
            continue
        (i, g1, t1, _), (j, g2, t2, _) = e[0], e[1]
        assert g1 in inst_d.Ntl.charges[i], f"g1={g1} missing from charges[{i}]"
        assert t1 in inst_d.Ntl.times[i],   f"t1={t1} missing from times[{i}]"
        assert g2 in inst_d.Ntl.charges[j], f"g2={g2} missing from charges[{j}]"
        assert t2 in inst_d.Ntl.times[j],   f"t2={t2} missing from times[{j}]"

   
    for e, flow in soln['x_ener'].items():
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
# seed() unit tests
# ---------------------------------------------------------------------------

def _make_seed_instance(step_size=3):
    """Return a diamond-graph Instance and a soln dict from a fine-grained solve."""
    N = nx.DiGraph()
    N.add_edge(1, 2, dH=4, dL=4, time=1)
    N.add_edge(1, 3, dH=5, dL=5, time=2)
    N.add_edge(2, 4, dH=3, dL=3, time=1)
    N.add_edge(3, 4, dH=4, dL=4, time=1)
    nodes = list(N.nodes)
    T = 6
    base_params = {
        'sources': [1], 'sinks': [4], 'T': T, 'L': 10, 'D': {1: (1, 0), 4: (0, 1)},
        'charge_rate': {i: 1 for i in nodes}, 'battery_nodes': [],
        'charge_nodes': [2], 'mobile_nodes': [], 'tractor_nodes': [3],
        'bat_swap_time': 1, 'tr_swap_time': 1, 'mobile_charge_rate': 0,
        'charge_cost': {(i, t): 10 for i in nodes for t in range(T)},
        'surplus_cost': {i: 0 for i in nodes}, 'stat_cost': {i: 0 for i in nodes},
        'rec_penalty': 1000, 'MAX_ITER': 20,
        'N_tractors': 1, 'N_chargers': 1, 'N_batteries': 1,
        'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2},
                        1: {'speed': 2, 'minbat': 2, 'maxbat': 8},
                        2: {'speed': 1, 'minbat': 8, 'maxbat': 10}},
    }
    inst_fine = Instance_v2.Instance(N.copy(), {**base_params, 'step_size': 1})
    soln, _, _ = inst_fine.run_DDD()
    inst = Instance_v2.Instance(N.copy(), {**base_params, 'step_size': step_size}, ['default'])
    return inst, soln, N


def test_seed_x_load_breakpoints_inserted():
    """Every (g, t) from x_load edges must appear in Ntl.charges/times after seed."""
    inst, soln, _ = _make_seed_instance()
    inst.seed(soln)
    for e in soln['x_load']:
        (i, g1, t1, _), (j, g2, t2, _) = e[0], e[1]
        assert g1 in inst.Ntl.charges[i], f"g1={g1} missing from charges[{i}] (x_load)"
        assert t1 in inst.Ntl.times[i],   f"t1={t1} missing from times[{i}] (x_load)"
        assert g2 in inst.Ntl.charges[j], f"g2={g2} missing from charges[{j}] (x_load)"
        assert t2 in inst.Ntl.times[j],   f"t2={t2} missing from times[{j}] (x_load)"


def test_seed_x_ener_breakpoints_inserted():
    """Every (g, t) from x_ener edges must appear in Ntl.charges/times after seed."""
    inst, soln, _ = _make_seed_instance()
    inst.seed(soln)
    for e in soln['x_ener']:
        (i, g1, t1, _), (j, g2, t2, _) = e[0], e[1]
        assert g1 in inst.Ntl.charges[i], f"g1={g1} missing from charges[{i}] (x_ener)"
        assert t1 in inst.Ntl.times[i],   f"t1={t1} missing from times[{i}] (x_ener)"
        assert g2 in inst.Ntl.charges[j], f"g2={g2} missing from charges[{j}] (x_ener)"
        assert t2 in inst.Ntl.times[j],   f"t2={t2} missing from times[{j}] (x_ener)"


def test_seed_preserves_pre_seed_breakpoints():
    """seed() must not remove any charges/times that existed before the call."""
    inst, soln, N = _make_seed_instance()
    charges_before = {i: list(inst.Ntl.charges[i]) for i in N.nodes}
    times_before   = {i: list(inst.Ntl.times[i])   for i in N.nodes}
    inst.seed(soln)
    for i in N.nodes:
        for g in charges_before[i]:
            assert g in inst.Ntl.charges[i], f"pre-seed charge {g} lost at node {i}"
        for t in times_before[i]:
            assert t in inst.Ntl.times[i],   f"pre-seed time {t} lost at node {i}"


def test_seed_charges_times_remain_sorted_no_duplicates():
    """After seed(), every charges[i] and times[i] list is sorted with no duplicates."""
    inst, soln, N = _make_seed_instance()
    inst.seed(soln)
    for i in N.nodes:
        c = inst.Ntl.charges[i]
        assert c == sorted(set(c)), f"charges[{i}] not sorted/unique: {c}"
        t = inst.Ntl.times[i]
        assert t == sorted(set(t)), f"times[{i}] not sorted/unique: {t}"


def test_seed_rebuilds_network_edges():
    """After seed(), the charge-time network has edges and consistent attribute dicts."""
    inst, soln, _ = _make_seed_instance()
    inst.seed(soln)
    ntl_edges = set(inst.Ntl.Ntl.edges)
    assert len(ntl_edges) > 0, "Network has no edges after seed"
    assert set(inst.Ntl.edge_types.keys())   == ntl_edges, "edge_types mismatch after seed"
    assert set(inst.Ntl.edge_times.keys())   == ntl_edges, "edge_times mismatch after seed"
    assert set(inst.Ntl.edge_dists.keys())   == ntl_edges, "edge_dists mismatch after seed"
    assert set(inst.Ntl.edge_charges.keys()) == ntl_edges, "edge_charges mismatch after seed"


def test_seed_rebuilds_gurobi_model():
    """After seed(), x_load and x_ener model vars match the rebuilt network edges."""
    inst, soln, _ = _make_seed_instance()
    inst.seed(soln)
    ntl_edges = set(inst.Ntl.Ntl.edges)
    assert set(inst.x_load.keys()) == ntl_edges, "x_load vars mismatch after seed"
    assert set(inst.x_ener.keys()) == ntl_edges, "x_ener vars mismatch after seed"


def test_seed_run_ddd_feasible():
    """run_DDD() on a seeded instance should produce a feasible solution."""
    inst, soln, _ = _make_seed_instance()
    inst.seed(soln)
    seeded_soln, val, _ = inst.run_DDD()
    assert val is not None and val < 1e9, f"Seeded run_DDD returned infeasible val={val}"
    for key, v in seeded_soln.items():
        if isinstance(v, dict):
            assert v != {}, f"Seeded run_DDD returned empty dict for result['{key}']"


def test_seed_with_empty_solution():
    """seed() with empty x_load and x_ener should not crash and still rebuild."""
    inst, _, _ = _make_seed_instance()
    ntl_edges_before = len(inst.Ntl.Ntl.edges)
    inst.seed({'x_load': {}, 'x_ener': {}})
    assert len(inst.Ntl.Ntl.edges) > 0, "Network empty after seed with empty solution"
    # charges/times unchanged since no breakpoints were injected
    ntl_edges_after = len(inst.Ntl.Ntl.edges)
    assert ntl_edges_after == ntl_edges_before, (
        f"Edge count changed with empty seed: {ntl_edges_before} -> {ntl_edges_after}"
    )


def test_seed_idempotent():
    """Seeding twice with the same solution should not add duplicate breakpoints."""
    inst, soln, N = _make_seed_instance()
    inst.seed(soln)
    charges_after_first = {i: list(inst.Ntl.charges[i]) for i in N.nodes}
    times_after_first   = {i: list(inst.Ntl.times[i])   for i in N.nodes}
    inst.seed(soln)
    for i in N.nodes:
        assert inst.Ntl.charges[i] == charges_after_first[i], \
            f"charges[{i}] changed on second seed"
        assert inst.Ntl.times[i] == times_after_first[i], \
            f"times[{i}] changed on second seed"


def test_seed_preserves_optimal_value():
    """Seeding a coarse instance with a fine-grained solution should not change
    the optimal objective: the seeded coarse instance has all breakpoints needed
    to represent the fine solution, so its optimal must equal the fine optimal."""
    N = nx.DiGraph()
    N.add_edge(1, 2, dH=4, dL=4, time=1)
    N.add_edge(1, 3, dH=5, dL=5, time=2)
    N.add_edge(2, 4, dH=3, dL=3, time=1)
    N.add_edge(3, 4, dH=4, dL=4, time=1)
    nodes = list(N.nodes)
    T = 6
    base_params = {
        'sources': [1], 'sinks': [4], 'T': T, 'L': 10, 'D': {1: (1, 0), 4: (0, 1)},
        'charge_rate': {i: 1 for i in nodes}, 'battery_nodes': [],
        'charge_nodes': [2], 'mobile_nodes': [], 'tractor_nodes': [3],
        'bat_swap_time': 1, 'tr_swap_time': 1, 'mobile_charge_rate': 0,
        'charge_cost': {(i, t): 10 for i in nodes for t in range(T)},
        'surplus_cost': {i: 0 for i in nodes}, 'stat_cost': {i: 0 for i in nodes},
        'rec_penalty': 1000, 'MAX_ITER': 20,
        'N_tractors': 1, 'N_chargers': 1, 'N_batteries': 1,
        'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 2},
                        1: {'speed': 2, 'minbat': 2, 'maxbat': 8},
                        2: {'speed': 1, 'minbat': 8, 'maxbat': 10}},
    }

    # Fine-grained instance: reference optimal
    inst_fine = Instance_v2.Instance(N.copy(), {**base_params, 'step_size': 1})
    soln_fine, val_fine, _ = inst_fine.run_DDD()
    assert val_fine is not None and val_fine < 1e9, f"Fine instance infeasible: {val_fine}"

    # Coarse instance seeded with the fine solution
    inst_coarse = Instance_v2.Instance(N.copy(), {**base_params, 'step_size': 3}, ['default'])
    inst_coarse.seed(soln_fine)
    _, val_seeded, _ = inst_coarse.run_DDD()

    assert val_seeded == pytest.approx(val_fine, abs=1e-6), (
        f"Seeded coarse optimal {val_seeded} != fine optimal {val_fine}"
    )


def test_source_and_sink_cross_flow():
    """A node in both sources and sinks must receive transit trucks when its
    demand exceeds its local supply.

    Graph:  0 ↔ 1  (travel time=2, bidirectional)

    D = {0: (supply=1, demand=2), 1: (supply=2, demand=1)}
    sources = sinks = [0, 1]

    Node 0's own supply truck can contribute at most 1 to its arrival count
    at T-1, so the model must route at least 1 truck from node 1 to node 0
    in transit — it cannot satisfy demand solely via waiting arcs at node 0.
    """
    N = nx.DiGraph()
    N.add_edge(0, 1, dH=10, dL=5, time=2)
    N.add_edge(1, 0, dH=10, dL=5, time=2)
    nodes = list(N.nodes)
    T = 8
    params = {
        'sources': [0, 1],
        'sinks':   [0, 1],
        'T': T,
        'L': 100,
        'D': {0: (1, 2), 1: (2, 1)},
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
        'rec_penalty': 10000,
        'N_tractors': 0,
        'N_chargers': 0,
        'N_batteries': 0,
        'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 100}},
    }

    inst = Instance_v2.Instance(N.copy(), params, ['default'])
    soln, val, _ = inst.run_DDD()

    assert val is not None and val < 1e9, f"run_DDD infeasible: val={val}"
    for key, v in soln.items():
        if isinstance(v, dict):
            assert v != {}, f"run_DDD returned empty dict for result['{key}']"

    # Both demand constraints must be satisfied
    for node, (_, demand) in params['D'].items():
        inflow = sum(v for e, v in soln['x_load'].items()
                     if e[1][0] == node and e[1][2] == T - 1)
        assert inflow >= pytest.approx(demand, abs=1e-4), (
            f"Node {node}: demand={demand} not met, inflow at T-1={inflow}"
        )

    # Node 0 has supply=1 but demand=2: at least 1 truck must transit from
    # node 1 to node 0 (a waiting arc at node 0 covers at most 1 of the 2).
    transit_into_0 = sum(v for e, v in soln['x_load'].items()
                         if e[0][0] == 1 and e[1][0] == 0)
    assert transit_into_0 >= pytest.approx(1.0, abs=1e-4), (
        f"Expected >= 1 truck to transit from node 1 to node 0 "
        f"(supply=1 < demand=2 at node 0), got {transit_into_0}"
    )

