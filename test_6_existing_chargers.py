"""Existing chargers are free; only chargers beyond them (a_c) cost stat_cost each.

Path 0 -> 1 -> 2, each leg uses 6 of a 10-unit battery and takes one step. Every truck
must charge 4 -> 6 at node 1 during step 1 (T = 4 leaves no slack). A charger delivers
charge_rate * speed per step, so with rate 2 and speed 1 each truck needs one charger in
that step, and with speed 2 one charger serves two trucks.
"""
import networkx as nx
import pytest

import Instance_v2

STAT_COST = 100


def solve(trucks, existing_chargers=None, existing_charge_nodes=None, speed=1):
    N = nx.DiGraph()
    N.add_edge(0, 1, dH=6, dL=3, time=1)
    N.add_edge(1, 2, dH=6, dL=3, time=1)
    T = 4
    p = {
        'T': T, 'L': 10, 'step_size': 1, 'MAX_ITER': 50,
        'sources': [0], 'sinks': [2], 'D': {(0, 2): trucks},
        'charge_nodes': [1], 'battery_nodes': [], 'tractor_nodes': [],
        'charge_rate': {0: 2, 1: 2, 2: 2},
        'charge_cost': {(i, t): 1 for i in N.nodes for t in range(T)},
        'stat_cost': {i: STAT_COST for i in N.nodes},
        'surplus_cost': {i: 0 for i in N.nodes},
        'tractor_cost': 0, 'battery_cost': 0, 'N_tractors': 0,
        'speed_curve': {0: {'speed': speed, 'minbat': 0, 'maxbat': 10}},
        'bat_swap_time': 1, 'tr_swap_time': 1, 'verbose': False,
    }
    if existing_chargers is not None:
        p['existing_chargers'] = existing_chargers
    if existing_charge_nodes is not None:
        p['existing_charge_nodes'] = existing_charge_nodes
    inst = Instance_v2.Instance(N, p, ['heuristic', 'local_rebuild_update'])
    inst.model.setParam('OutputFlag', 0)
    inst.model.setParam('MIPGap', 0)
    soln, val, _ = inst.run_DDD(LB=0, LP=False)
    return round(soln['a_c'].get(1, 0)), val


@pytest.mark.parametrize('trucks, existing, new_chargers', [
    (2, 2, 0),   # two existing chargers serve both trucks for free
    (3, 2, 1),   # the third truck needs one new charger
    (2, 5, 0),   # spare existing chargers cost nothing
    (2, 0, 2),   # no existing chargers: both are new
])
def test_new_chargers_beyond_existing(trucks, existing, new_chargers):
    a_c, _ = solve(trucks, existing_chargers={1: existing})
    assert a_c == new_chargers


def test_only_new_chargers_are_paid_for():
    _, free = solve(2, existing_chargers={1: 2})
    _, one_new = solve(3, existing_chargers={1: 2})
    _, base3 = solve(3, existing_chargers={1: 3})
    assert one_new - base3 == pytest.approx(STAT_COST)
    assert free < one_new


def test_existing_charge_nodes_alone_count_one_charger():
    a_c, _ = solve(2, existing_charge_nodes=[1])
    assert a_c == 1


@pytest.mark.parametrize('trucks, existing, new_chargers', [
    (2, 1, 0),   # one charger delivers 2 * 2 = 4 per step: enough for two trucks
    (3, 1, 1),
    (4, 1, 1),
    (5, 1, 2),
])
def test_capacity_scales_with_speed(trucks, existing, new_chargers):
    a_c, _ = solve(trucks, existing_chargers={1: existing}, speed=2)
    assert a_c == new_chargers


# ---------------------------------------------------------------------------
# Sessions longer than one step (rate 1, speed 1: charging 4 -> 6 takes 2 steps)
# ---------------------------------------------------------------------------

def solve_two_step(trucks, existing, T, origins_by_truck=None):
    N = nx.DiGraph()
    N.add_edge(0, 1, dH=6, dL=3, time=1)
    N.add_edge(1, 2, dH=6, dL=3, time=1)
    p = {
        'T': T, 'L': 10, 'step_size': 1, 'MAX_ITER': 100,
        'sources': [0], 'sinks': [2], 'D': {(0, 2): trucks},
        'charge_nodes': [1], 'battery_nodes': [], 'tractor_nodes': [],
        'charge_rate': {0: 1, 1: 1, 2: 1},
        'charge_cost': {(i, t): 1 for i in N.nodes for t in range(T)},
        'stat_cost': {i: STAT_COST for i in N.nodes},
        'surplus_cost': {i: 0 for i in N.nodes},
        'tractor_cost': 0, 'battery_cost': 0, 'N_tractors': 0, 'value_for_time': 0,
        'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 10}},
        'bat_swap_time': 1, 'tr_swap_time': 1, 'verbose': False,
        'existing_chargers': {1: existing},
    }
    inst = Instance_v2.Instance(N, p, ['heuristic', 'local_rebuild_update'])
    inst.model.setParam('OutputFlag', 0)
    inst.model.setParam('MIPGap', 0)
    soln, val, props = inst.run_DDD(LB=0, LP=False)
    assert not props.get('timeout'), "DDD did not converge"
    return round(soln['a_c'].get(1, 0))


def test_one_truck_two_step_session_needs_one_charger():
    # the session spans steps 1 and 2; counting it all in step 1 would need 2 chargers
    assert solve_two_step(1, existing=1, T=5) == 0


def test_overlapping_two_step_sessions_need_two_chargers():
    # T = 5: both trucks must charge during steps 1-2, at the same time
    assert solve_two_step(2, existing=1, T=5) == 1


def test_consecutive_two_step_sessions_share_one_charger():
    # T = 7 leaves room for the second truck to charge in steps 3-4 (waiting is free)
    assert solve_two_step(2, existing=1, T=7) == 0


def test_min_overlap():
    mo = Instance_v2.Instance._min_overlap
    assert mo(0, 1, 2, 0, 1) == 1      # exact start: covers the step
    assert mo(0, 1, 2, 2, 3) == 0      # already over
    assert mo(0, 3, 2, 0, 3) == 1      # start in 0..2: [2, 4) covers only step 2 of [0, 3)
    assert mo(0, 3, 5, 3, 6) == 2      # start 0 covers 3, 4; start 2 covers 3, 4, 5
    assert mo(0, 5, 1, 0, 5) == 1      # one-step session always inside its own interval


@pytest.mark.parametrize('trucks, T', [(2, 7), (3, 8), (3, 6)])
def test_coarse_discretization_reaches_full_optimum(trucks, T):
    """DDD from a coarse grid (step_size 3) ends at the optimum of the full grid (step_size 1)."""
    def run(step_size):
        N = nx.DiGraph()
        N.add_edge(0, 1, dH=6, dL=3, time=1)
        N.add_edge(1, 2, dH=6, dL=3, time=1)
        p = {
            'T': T, 'L': 10, 'step_size': step_size, 'MAX_ITER': 200,
            'sources': [0], 'sinks': [2], 'D': {(0, 2): trucks},
            'charge_nodes': [1], 'battery_nodes': [], 'tractor_nodes': [],
            'charge_rate': {0: 1, 1: 1, 2: 1},
            'charge_cost': {(i, t): 1 for i in N.nodes for t in range(T)},
            'stat_cost': {i: STAT_COST for i in N.nodes},
            'surplus_cost': {i: 0 for i in N.nodes},
            'tractor_cost': 0, 'battery_cost': 0, 'N_tractors': 0, 'value_for_time': 1,
            'speed_curve': {0: {'speed': 1, 'minbat': 0, 'maxbat': 10}},
            'bat_swap_time': 1, 'tr_swap_time': 1, 'verbose': False,
            'existing_chargers': {1: 1},
        }
        inst = Instance_v2.Instance(N, p, ['heuristic', 'local_rebuild_update'])
        inst.model.setParam('OutputFlag', 0)
        inst.model.setParam('MIPGap', 0)
        _, val, props = inst.run_DDD(LB=0, LP=False)
        assert not props.get('timeout')
        return val
    assert run(3) == pytest.approx(run(1))
