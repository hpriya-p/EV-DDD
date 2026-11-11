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


@pytest.mark.xfail(strict=True, reason=(
    "DDD waiting edges cost the full gap to the next retained time, more than the true wait "
    "of a schedule mapped onto the coarse network; time windows force such waits, so the "
    "lower bound exceeds the optimum (here 92 > 62)"))
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
