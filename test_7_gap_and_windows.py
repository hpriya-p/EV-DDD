"""Per-commodity load time windows, and upper bounds with stopping at a gap during Bi-DDD."""
import itertools

import networkx as nx
import pytest

import Instance_v2


def line_instance(T, step_size, trucks=(2, 1), **extra):
    """0 -> 1 -> 2 -> 3, each leg 4 of a 10-unit battery and one step; chargers at 1 and 2.
    Commodity 0 goes 0 -> 3, commodity 1 goes 1 -> 3."""
    N = nx.DiGraph()
    for i in range(3):
        N.add_edge(i, i + 1, dH=4, dL=2, time=1)
    p = {
        'T': T, 'L': 10, 'step_size': step_size, 'MAX_ITER': 300,
        'sources': [0, 1], 'sinks': [3], 'D': {(0, 3): trucks[0], (1, 3): trucks[1]},
        'charge_nodes': [1, 2], 'battery_nodes': [], 'tractor_nodes': [],
        'charge_rate': {i: 1 for i in N.nodes},
        'charge_cost': {(i, t): 1 for i in N.nodes for t in range(T)},
        'stat_cost': {i: 100 for i in N.nodes}, 'surplus_cost': {i: 0 for i in N.nodes},
        'tractor_cost': 0, 'battery_cost': 0, 'N_tractors': 0, 'value_for_time': 3,
        'speed_curve': {0: {'speed': 2, 'minbat': 0, 'maxbat': 10}},
        'bat_swap_time': 1, 'tr_swap_time': 1, 'verbose': False,
        'existing_chargers': {1: 1, 2: 1}, 'load_time_windows': True,
    }
    p.update(extra)
    inst = Instance_v2.Instance(N, p, ['heuristic', 'local_rebuild_update'])
    inst.model.setParam('OutputFlag', 0)
    inst.model.setParam('MIPGap', 0)
    return inst


def chain_instance(trucks, T, step_size, **extra):
    """0 -> 1 -> 2, legs of 6 of 10 and one step; one slow charger at 1 (2-step sessions)."""
    N = nx.DiGraph()
    N.add_edge(0, 1, dH=6, dL=3, time=1)
    N.add_edge(1, 2, dH=6, dL=3, time=1)
    p = {
        'T': T, 'L': 10, 'step_size': step_size, 'MAX_ITER': 300,
        'sources': [0], 'sinks': [2], 'D': {(0, 2): trucks},
        'charge_nodes': [1], 'battery_nodes': [], 'tractor_nodes': [],
        'charge_rate': {0: 1, 1: 1, 2: 1},
        'charge_cost': {(i, t): 1 for i in N.nodes for t in range(T)},
        'stat_cost': {i: 100 for i in N.nodes}, 'surplus_cost': {i: 0 for i in N.nodes},
        'tractor_cost': 0, 'battery_cost': 0, 'N_tractors': 0, 'value_for_time': 1,
        'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 10}},
        'bat_swap_time': 1, 'tr_swap_time': 1, 'verbose': False,
        'existing_chargers': {1: 1},
    }
    p.update(extra)
    inst = Instance_v2.Instance(N, p, ['heuristic', 'local_rebuild_update'])
    inst.model.setParam('OutputFlag', 0)
    inst.model.setParam('MIPGap', 0)
    return inst


def solve(inst):
    soln, val, props = inst.run_DDD(LB=0, LP=False)
    assert not props.get('timeout'), "DDD did not converge"
    return soln, val, props


# ---------------------------------------------------------------------------
# Load time windows
# ---------------------------------------------------------------------------

def test_time_windows_fix_some_load_variables():
    inst = line_instance(T=9, step_size=1)
    fixed = sum(1 for (e, k), x in inst.x_load.items() if x.UB == 0)
    off = line_instance(T=9, step_size=1, load_time_windows=False)
    fixed_off = sum(1 for (e, k), x in off.x_load.items() if x.UB == 0)
    assert fixed > fixed_off


def test_commodity_cannot_leave_before_reaching_station():
    # commodity 0 starts at 0 at time 0 and needs one step to reach 1
    inst = line_instance(T=9, step_size=1)
    for (e, k), x in inst.x_load.items():
        if k == 0 and e[0][0] == 1 and e[0][2] < 1:
            assert x.UB == 0


@pytest.mark.parametrize('T, step, trucks', list(itertools.product((7, 9), (1, 3), ((2, 1), (1, 2)))))
def test_time_windows_keep_the_optimum(T, step, trucks):
    _, with_windows, _ = solve(line_instance(T, step, trucks))
    _, without, _ = solve(line_instance(T, step, trucks, load_time_windows=False))
    assert with_windows == pytest.approx(without)


# ---------------------------------------------------------------------------
# Upper bounds and stopping at a gap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('trucks, T', [(2, 7), (3, 6), (3, 8)])
def test_gap_stop_brackets_the_optimum(trucks, T):
    _, opt, _ = solve(chain_instance(trucks, T, 1))
    inst = chain_instance(trucks, T, 3, upper_bound={'every': 1, 'gap': 1.0})
    soln, val, props = inst.run_DDD(LB=0, LP=False)
    if props.get('stopped_by') == 'gap':
        assert props['upper_bound'] >= opt - 1e-6          # a feasible solution
        assert props['lower_bound'] <= opt + 1e-6          # a valid bound
        assert val == props['upper_bound'] and soln['a_c'] is not None
    else:
        assert val == pytest.approx(opt)                   # converged before any bound


@pytest.mark.parametrize('trucks, T', [(2, 7), (3, 6), (3, 8)])
def test_zero_gap_reaches_the_optimum(trucks, T):
    _, opt, _ = solve(chain_instance(trucks, T, 1))
    _, val, props = solve(chain_instance(trucks, T, 3, upper_bound={'every': 1, 'gap': 0.0}))
    assert val == pytest.approx(opt, rel=1e-3)


def test_upper_bound_events_logged(tmp_path):
    log = tmp_path / 'ddd.jsonl'
    inst = chain_instance(3, 8, 3, upper_bound={'every': 1, 'gap': 0.0}, ddd_log=str(log))
    inst.run_DDD(LB=0, LP=False)
    events = [r for r in Instance_v2.read_ddd_log(str(log)) if r['event'] == 'upper_bound']
    assert events and all('tried' in r for r in events)


def test_time_windows_keep_lower_bound_valid_with_two_step_legs():
    def inst(step, windows):
        N = nx.DiGraph()
        N.add_edge(0, 1, dH=6, dL=3, time=2)
        N.add_edge(1, 2, dH=6, dL=3, time=2)
        T = 8
        p = {'T': T, 'L': 10, 'step_size': step, 'MAX_ITER': 300, 'sources': [0], 'sinks': [2],
             'D': {(0, 2): 1}, 'charge_nodes': [1], 'battery_nodes': [], 'tractor_nodes': [],
             'charge_rate': {0: 1, 1: 1, 2: 1}, 'charge_cost': {(i, t): 1 for i in N.nodes for t in range(T)},
             'stat_cost': {i: 100 for i in N.nodes}, 'surplus_cost': {i: 0 for i in N.nodes},
             'tractor_cost': 0, 'battery_cost': 0, 'N_tractors': 0, 'value_for_time': 10,
             'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 10}}, 'bat_swap_time': 1,
             'tr_swap_time': 1, 'verbose': False, 'existing_chargers': {1: 1}, 'load_time_windows': windows}
        i = Instance_v2.Instance(N, p, ['heuristic', 'local_rebuild_update'])
        i.model.setParam('OutputFlag', 0)
        i.model.setParam('MIPGap', 0)
        return i
    _, opt, _ = solve(inst(1, False))
    coarse = inst(3, True)
    coarse.model.optimize()
    assert coarse.model.ObjBound <= opt + 1e-6


@pytest.mark.parametrize('iters', [3, 5, 8])
def test_incremental_network_equals_fresh_build_on_directed_graph(iters):
    """local_rebuild_update must leave the network construct() would build, including the
    edges from predecessors into a station whose times changed."""
    inst = chain_instance(3, 11, 3)
    inst.param['MAX_ITER'] = iters
    inst.run_DDD(LB=0, LP=False)
    fresh = chain_instance(3, 11, 3, initial_discretization={'times': inst.Ntl.times,
                                                             'charges': inst.Ntl.charges})

    def sig(c):
        return sorted((tuple(map(int, e[0][:3])), tuple(map(int, e[1][:3])), c.edge_types[e],
                       int(c.edge_charges[e]), int(c.edge_times[e])) for e in c.Ntl.edges)
    assert sig(inst.Ntl) == sig(fresh.Ntl)


# ---------------------------------------------------------------------------
# Initial time points from the commodities' windows
# ---------------------------------------------------------------------------

def test_significant_times_on_line():
    from initial_discretization import significant_times
    inst = line_instance(T=9, step_size=1)
    pts = significant_times(inst.N, inst.param)
    # commodity 0 (0 -> 3): earliest at station 2 is 2 by travel and with charging (4 + 4 <= 10)
    assert 2 in pts[2]
    # latest departure from station 2 towards 3: T - 1 - 1 = 7
    assert 7 in pts[2]
    assert all(0 <= t <= 8 for ts in pts.values() for t in ts)


@pytest.mark.parametrize('T, trucks', [(7, (2, 1)), (9, (1, 2)), (9, (2, 1))])
def test_significant_times_keep_the_optimum(T, trucks):
    from initial_discretization import significant_times
    _, opt, _ = solve(line_instance(T, 1, trucks))
    probe = line_instance(T, T, trucks)
    pts = significant_times(probe.N, probe.param)
    _, val, _ = solve(line_instance(T, T, trucks, initial_discretization={'times': pts}))
    assert val == pytest.approx(opt)


def test_earliest_arrival_by_charging_and_by_swapping_separately():
    """0 -> 1 -> 2 -> 3 with legs of 6 of 10: the truck must recharge at 1 and at 2.
    Station 1 charges (rate 1, speed 1: 2 levels take 2 steps); station 2 swaps (2 steps)."""
    from initial_discretization import earliest_arrival, significant_times
    N = nx.DiGraph()
    for i in range(3):
        N.add_edge(i, i + 1, dH=6, dL=3, time=1)
    p = {'T': 20, 'L': 10, 'min_time': 0, 'D': {(0, 3): 1},
         'charge_nodes': [1, 2], 'battery_nodes': [2], 'tractor_nodes': [],
         'charge_rate': {i: 1 for i in N.nodes}, 'bat_swap_time': 2, 'tr_swap_time': 2,
         'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 10}}}
    by_charging = earliest_arrival(N, p, 0, charge=True, swap=False)
    by_swapping = earliest_arrival(N, p, 0, charge=False, swap=True)
    # charging only: 0 -> 1 at 1 with 4, charge 4 -> 6 (2 steps) until 3, -> 2 at 4 with 0,
    # charge 0 -> 6 (6 steps) until 10, -> 3 at 11
    assert by_charging[3] == 11
    # swapping only: no swap at 1, so the truck (4 left at 1) cannot reach 2
    assert 3 not in by_swapping and 2 not in by_swapping
    pts = significant_times(N, p)
    # combined search: charge at 1, swap at 2 (arrive 4 with 0, swap until 6), reach 3 at 7
    assert 7 in pts[3] and 4 in pts[2]


def test_latest_departures_and_departures_after_charging():
    """Same chain (T = 20): charge at 1 and 2, swap at 2 (2 steps); legs of 6 of 10."""
    from initial_discretization import latest_departure, significant_times
    N = nx.DiGraph()
    for i in range(3):
        N.add_edge(i, i + 1, dH=6, dL=3, time=1)
    p = {'T': 20, 'L': 10, 'min_time': 0, 'D': {(0, 3): 1},
         'charge_nodes': [1, 2], 'battery_nodes': [2], 'tractor_nodes': [],
         'charge_rate': {i: 1 for i in N.nodes}, 'bat_swap_time': 2, 'tr_swap_time': 2,
         'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 10}}}
    by_charging = latest_departure(N, p, 3, charge=True, swap=False)
    by_swapping = latest_departure(N, p, 3, charge=False, swap=True)
    # leaving 1 full: -> 2 (1 step) with 4, recharge 2 levels (2 steps) or swap (2 steps), -> 3 (1)
    assert by_charging[1] == 19 - 4 and by_swapping[1] == 19 - 4
    assert by_charging[2] == 18 and by_charging[3] == 19
    pts = significant_times(N, p)
    # at 1 the truck arrives at 1 with 4 and is full at 7 (only the full target is kept)
    assert {1, 7, 15}.issubset(pts[1]) and 3 not in pts[1]


def test_max_detour_keeps_points_near_fastest_route():
    """Diamond 0 -> {1, 2} -> 3: via 1 takes 2 steps, via 2 takes 6. With max_detour 0 only
    stations on the fastest schedule (0, 1, 3) get points; with 4, station 2 too."""
    from initial_discretization import significant_times, through_times
    N = nx.DiGraph()
    N.add_edge(0, 1, dH=2, dL=1, time=1)
    N.add_edge(1, 3, dH=2, dL=1, time=1)
    N.add_edge(0, 2, dH=2, dL=1, time=3)
    N.add_edge(2, 3, dH=2, dL=1, time=3)
    p = {'T': 20, 'L': 10, 'min_time': 0, 'D': {(0, 3): 1}, 'charge_nodes': [], 'battery_nodes': [],
         'tractor_nodes': [], 'charge_rate': {i: 1 for i in N.nodes}, 'bat_swap_time': 1, 'tr_swap_time': 1,
         'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 10}}}
    th = through_times(N, p, 0, 3)
    assert th[1] == 2 and th[2] == 6 and th[3] == 2
    assert 2 not in significant_times(N, p, max_detour=0)
    assert 2 in significant_times(N, p, max_detour=4)
    assert 2 in significant_times(N, p)          # no limit: every station with a window
