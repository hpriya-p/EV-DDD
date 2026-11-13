"""Upper bound on the Bi-DDD (and heuristic Bi-DDD) optimum from a given discretization.

Given the retained charges and times at each node, build the charge-time network
with the opposite rounding to Bi-DDD: every edge's true head (j, g', t') is mapped
to the retained node with the largest charge <= g' and the smallest time >= t'
(edges whose head falls after T-1 are dropped). A vehicle following such an edge
really arrives no later and with no less charge than the node it lands on, so it
can wait until that time and carry the extra charge; waiting is always allowed
and extra charge never hurts:
  - transit and wait edges: immediate;
  - charging edges: the real vehicle starts at or above the modeled level; if it
    is in the same speed segment it charges at the same rate and needs no more
    energy, and if it is in a higher segment it already holds more than any
    target of the modeled (lower) segment;
  - swap edges go to exact retained levels.
Costs never undercount: travel time runs to the rounded-up arrival (waiting is
free, as in the full model), and the energy bought on a charging edge, for
its cost and for charger capacity, is the full amount to the true target.
Charger capacity is checked at every retained time for the sessions running then
(each drawing its charge evenly over its true duration); between retained times no
session starts, so the load is highest at those times and every step is within it.

So every feasible solution of the model on this network is feasible for the full
problem (and, with config 'heuristic', for the heuristic problem, which is a
restriction of it): its objective is an upper bound, even when the MIP is stopped
at a time limit before it proves optimality.

Use:
    from upper_bound import upper_bound, upper_bound_from_log
    res = upper_bound(N, parameters, config, times, charges, time_limit=600)
    res = upper_bound_from_log(N, parameters, config, 'ddd_log.jsonl')   # last iteration
    res['status'], res['upper_bound'], res['solution']
"""
import bisect
import time
from concurrent.futures import ProcessPoolExecutor

import gurobipy as gp
import networkx as nx
import numpy as np

import Instance_v2
from ChargeTimeNetwork_v2 import ChargeTimeNetwork


# ---------------------------------------------------------------------------
# Edge construction with pessimistic rounding (worker for a process pool)
# ---------------------------------------------------------------------------

_pw = {}


def _init_pess_worker(charges, times, param, config):
    _pw.update(charges=charges, times=times, param=param, config=config)


def _get_q(speed_curve, g):
    for q in sorted(speed_curve.keys()):
        if g < speed_curve[q]['maxbat']:
            return q
    return max(speed_curve.keys())


def _floor(lst, val):
    """Largest element of sorted lst that is <= val, or None."""
    k = bisect.bisect_right(lst, val)
    return lst[k - 1] if k > 0 else None


def _ceil(lst, val):
    """Smallest element of sorted lst that is >= val, or None."""
    k = bisect.bisect_left(lst, val)
    return lst[k] if k < len(lst) else None


def _compute_node_edges_pessimistic(task):
    """
    Edges out of station i, as ChargeTimeNetwork_v2._compute_node_edges builds them,
    except that heads are rounded pessimistically: charge down, time up.

    Returns (v1, v2, e_type, edge_charge, edge_time, edge_dist, g2_raw, t2_raw) per edge.
    Several true heads can round to the same node; the first is kept, which for
    charging edges (targets in increasing order) is the one needing the least energy.
    """
    i, times_i, charges_i, neighbors = task
    charges, times = _pw['charges'], _pw['times']
    param, config = _pw['param'], _pw['config']
    T, L = param['T'], param['L']
    speed_curve = param['speed_curve']
    get_q = lambda g: _get_q(speed_curve, g)
    seen, results = set(), []

    def emit(v1, j, g2_raw, t2_raw, e_type):
        _, g1, t1, _ = v1
        if e_type == 'wait':
            g2, t2 = g2_raw, t2_raw                      # consecutive retained times: exact
        else:
            if t2_raw > T - 1:
                return                                  # would arrive after the horizon
            g2 = _floor(charges[j], g2_raw)              # charge rounded down
            t2 = _ceil(times[j], t2_raw)                 # time rounded up
            if g2 is None or t2 is None:
                return
        v2 = (j, g2, t2, get_q(g2))
        key = (v1, v2, e_type)
        if key in seen:
            return
        seen.add(key)
        if e_type in ('transit_H', 'transit_L'):
            ec, et, ed = 0, t2 - t1, g1 - g2_raw        # time to the rounded-up arrival
        elif e_type == 'wait':
            et = (t2 - t1) if t2 != T - 1 else 0         # same convention as Bi-DDD
            ec, ed = 0, 0
        else:  # charge, swap, piecewise: energy to the true target, time to the rounded-up arrival
            ec, et, ed = g2_raw - g1, t2 - t1, 0
        results.append((v1, v2, e_type, ec, et, ed, g2_raw, t2_raw))

    for t in times_i:
        if t == T - 1:
            continue

        # ---- swap edges ----
        if i in param['battery_nodes'] or i in param['tractor_nodes']:
            if t + param['bat_swap_time'] < T:
                if 'heuristic' not in config:
                    stop_g = False
                    for g in charges_i:
                        if stop_g:
                            break
                        v1 = (i, g, t, get_q(g))
                        for g2 in charges_i:
                            if g2 <= g:
                                continue
                            if i in param['battery_nodes'] and t + param['bat_swap_time'] < T:
                                emit(v1, i, g2, t + param['bat_swap_time'], 'swap')
                            elif i in param['tractor_nodes'] and t + param['tr_swap_time'] < T:
                                emit(v1, i, g2, t + param['tr_swap_time'], 'swap')
                            else:
                                stop_g = True
                                break
                else:
                    thresholds = [speed_curve[q]['maxbat'] for q in sorted(speed_curve.keys())]
                    for g in charges_i:
                        v1 = (i, g, t, get_q(g))
                        if i in param['battery_nodes']:
                            for g2 in thresholds:
                                if g2 > g and t + param['bat_swap_time'] < T:
                                    emit(v1, i, g2, t + param['bat_swap_time'], 'swap')
                        else:  # tractor node
                            for g2 in charges_i:
                                if g2 > g and t + param['tr_swap_time'] < T:
                                    emit(v1, i, g2, t + int(param['tr_swap_time']), 'swap')

        # ---- charge + piecewise edges ----
        if i in param['charge_nodes']:
            for g in charges_i:
                q1 = get_q(g)
                v1 = (i, g, t, q1)
                UB = speed_curve[q1]['maxbat']
                if 'heuristic' not in config:
                    # rounded down (int), unlike the DDD network's ceil: a pessimistic edge must
                    # never charge faster than a real charger
                    for g2 in range(g + 1, int(min(UB + 1, g + param['charge_rate'][i] * speed_curve[q1]['speed'] + 1))):
                        if g2 <= L and t + 1 < T:
                            emit(v1, i, g2, t + 1, 'charge')
                else:
                    nb_dH = [dH for _, dH, _, _ in neighbors]
                    for g2 in sorted(set(charges_i + nb_dH)):
                        if g2 <= g or g2 > UB:
                            continue
                        t2 = int(np.ceil((g2 - g) / (param['charge_rate'][i] * speed_curve[q1]['speed'])))
                        if t + t2 < T:
                            emit(v1, i, g2, t + t2, 'charge')

            for q in sorted(speed_curve.keys()):
                g = speed_curve[q]['maxbat']
                if g < L:
                    emit((i, g, t, q), i, g, t, 'piecewise')

        # ---- transit edges ----
        for g in charges_i:
            v1 = (i, g, t, get_q(g))
            for j, dH, dL, travel_time in neighbors:
                if dH <= g and t + travel_time < T:
                    emit(v1, j, g - dH, t + travel_time, 'transit_H')
                if dL <= g and t + travel_time < T:
                    emit(v1, j, g - dL, t + travel_time, 'transit_L')

    # ---- wait edges ----
    for g in charges_i:
        q = get_q(g)
        for l in range(len(times_i) - 1):
            t, t_next = times_i[l], times_i[l + 1]
            v1 = (i, g, t, q)
            emit(v1, i, g, t_next, 'wait')
            if t_next < T - 1:
                emit(v1, i, g, T - 1, 'wait')

    return results


class PessimisticChargeTimeNetwork(ChargeTimeNetwork):
    """Charge-time network with heads rounded charge-down, time-up (see module docstring).

    Build it from an existing network (for N, parameters and configuration) with
    from_network(); edge_raw[e] keeps each edge's true head (g2_raw, t2_raw).

    Vehicles leave exactly at the retained times here (they arrive no later and wait),
    so Instance checks charger capacity step by step (exact_starts), with each charging
    session lasting its true duration.
    """

    exact_starts = True

    def charge_duration(self, e):
        """True duration of charging edge e: steps from its tail to its unrounded head."""
        return self.edge_raw[e][1] - e[0][2]

    @classmethod
    def from_network(cls, base, times, charges):
        net = cls.__new__(cls)
        net.__dict__.update({k: v for k, v in base.__dict__.items() if k != 'Ntl'})
        T, L, t_min = net.param['T'], net.param['L'], net.param['min_time']
        # the given points, plus the initial ones (always retained by Bi-DDD)
        net.times = {i: sorted({int(t) for t in base.times[i]}
                               | {int(t) for t in (times or {}).get(i, []) if t_min <= int(t) <= T - 1})
                     for i in base.N.nodes}
        net.charges = {i: sorted({int(g) for g in base.charges[i]}
                                 | {int(g) for g in (charges or {}).get(i, []) if 0 <= int(g) <= L})
                       for i in base.N.nodes}
        net.edge_raw = {}
        net.construct()
        return net

    def construct_parallel(self, n_workers=None):
        self.Ntl = nx.MultiDiGraph()
        self.edge_types, self.edge_charges, self.edge_times, self.edge_dists = {}, {}, {}, {}
        self.edge_raw = {}
        tasks = []
        for i in self.N.nodes:
            neighbors = [(j, self.N[i][j]['dH'], self.N[i][j]['dL'], self.N[i][j]['time'])
                         for j in self.N.neighbors(i)]
            tasks.append((i, self.times[i], self.charges[i], neighbors))
        import os
        with ProcessPoolExecutor(max_workers=n_workers or os.cpu_count(),
                                 initializer=_init_pess_worker,
                                 initargs=(self.charges, self.times, self.param, self.config)) as ex:
            all_results = list(ex.map(_compute_node_edges_pessimistic, tasks))
        for node_edges in all_results:
            for v1, v2, e_type, ec, et, ed, g2_raw, t2_raw in node_edges:
                key = self.Ntl.add_edge(v1, v2)
                e = (v1, v2, key)
                self.edge_types[e], self.edge_charges[e] = e_type, ec
                self.edge_times[e], self.edge_dists[e] = et, ed
                self.edge_raw[e] = (g2_raw, t2_raw)


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------

def _solution(inst):
    flows = {'x_load': {k: {} for k in range(inst.K)}, 'x_ener': {k: {} for k in range(inst.K)}}
    for name, var in (('x_load', inst.x_load), ('x_ener', inst.x_ener)):
        for (e, k), x in var.items():
            if x.X > 0.5:
                flows[name][k][(e[0], e[1], inst.Ntl.edge_types[e])] = round(x.X)
    get = lambda d: {int(i): round(x.X) for i, x in d.items() if x.X > 0.5}
    a_c = get(inst.a_c)
    return dict(flows, a_c=a_c, a_s=get(inst.a_s), n=get(inst.n), z=get(inst.z), a=a_c)  # a: older name of a_c


def upper_bound_on_instance(inst, times, charges, time_limit=None, mip_gap=1e-4, threads=0,
                            output=False, fixed_design=None, design_mode='exact'):
    """
    Upper bound for an existing Instance: its network is replaced by the pessimistic
    network of the given discretization and the model is rebuilt and solved once.
    Any settings that inst applies in construct_model (e.g. a fixed design) carry over.

    fixed_design: optional {'a_c': {i: v}, 'a_s': {...}, 'n': {...}, 'z': {...}} ('a' is read
    as a_c). With design_mode 'exact', every a_c, a_s, n, z is fixed to exactly that value (0
    where absent), so the solve asks whether that design can be operated feasibly (and at
    what cost); with 'at_least', the values are lower bounds and more may be installed.
    """
    assert design_mode in ('exact', 'at_least'), design_mode
    t0 = time.time()
    inst.Ntl = PessimisticChargeTimeNetwork.from_network(inst.Ntl, times, charges)
    inst.construct_model()
    if fixed_design is not None:
        for name, var in (('a_c', inst.a_c), ('a_s', inst.a_s), ('n', inst.n), ('z', inst.z)):
            given = fixed_design.get(name)
            if given is None and name == 'a_c':
                given = fixed_design.get('a')            # older name
            values = {int(i): v for i, v in (given or {}).items()}
            for i, x in var.items():
                v = values.get(int(i), 0)
                if design_mode == 'exact':
                    x.lb = x.ub = v
                else:
                    x.lb = max(x.lb, min(v, x.ub))
        inst.model.update()
    m = inst.model
    m.setParam('OutputFlag', 1 if output else 0)
    m.setParam('MIPGap', mip_gap)
    if threads:
        m.setParam('Threads', threads)
    if time_limit:
        m.setParam('TimeLimit', time_limit)
    build_time = time.time() - t0
    m.optimize()
    has_solution = m.SolCount > 0
    status = {gp.GRB.OPTIMAL: 'optimal', gp.GRB.TIME_LIMIT: 'time_limit', gp.GRB.INFEASIBLE: 'infeasible',
              gp.GRB.INF_OR_UNBD: 'infeasible'}.get(m.Status, f'gurobi_status_{m.Status}')
    return {
        'status': status,
        # any feasible solution of this model is feasible for the full problem
        'upper_bound': m.ObjVal if has_solution else None,
        'mip_bound': m.ObjBound if has_solution or m.Status == gp.GRB.TIME_LIMIT else None,
        'mip_gap': m.MIPGap if has_solution else None,
        'solution': _solution(inst) if has_solution else None,
        'network_size': [inst.Ntl.Ntl.number_of_nodes(), inst.Ntl.Ntl.number_of_edges()],
        'build_time': build_time, 'solve_time': m.Runtime,
    }


def upper_bound(N, parameters, config, times, charges, instance_class=Instance_v2.Instance,
                instance_kwargs=None, **solve_kwargs):
    """Upper bound from a discretization ({node: [times]}, {node: [charges]})."""
    params = dict(parameters)
    params.pop('initial_discretization', None)
    params.pop('ddd_log', None)
    inst = instance_class(N, params, list(config), network_only=True, **(instance_kwargs or {}))
    return upper_bound_on_instance(inst, times, charges, **solve_kwargs)


def upper_bound_from_log(N, parameters, config, ddd_log, iteration=None, **kwargs):
    """Upper bound from the discretization of one iteration of a ddd_log (default: the last).

    The result also carries that iteration's number and lower bound, and the gap.
    """
    disc = Instance_v2.discretization_from_log(ddd_log, iteration)
    res = upper_bound(N, parameters, config, disc['times'], disc['charges'], **kwargs)
    lb, ub = disc.get('lower_bound'), res['upper_bound']
    res.update(iteration=disc['iteration'], lower_bound=lb,
               gap=(ub - lb) / ub if (ub and lb is not None) else None)
    return res


# ---------------------------------------------------------------------------
# Extra points for the pessimistic network: true routes
# ---------------------------------------------------------------------------
# Any discretization gives a valid pessimistic network, so the upper-bound network may
# have more points than the relaxation's. On a coarse grid it can have no route at all
# (every arrival is rounded up to the next retained time, every charge down). These add
# the true (time, charge) points of whole routes, so that the network contains them
# exactly: first the relaxation's own vehicle routes replayed with true values, and, for
# a commodity none of whose routes fits in the horizon, its earliest-arrival route.
# Charging follows the pessimistic network's rule: one speed segment per charging edge,
# lasting ceil(levels / (rate * speed)) steps.

_TRANSIT = ('transit_H', 'transit_L')


def _num(x):
    return int(round(x)) if abs(x - round(x)) < 1e-9 else x


def _charge_pieces(param, i, g, target):
    """Charge at station i from g to target, one speed segment at a time:
    [(level, steps), ...], or None if i cannot charge."""
    curve, rate = param['speed_curve'], param['charge_rate'].get(i, 0)
    if rate <= 0:
        return None
    pieces = []
    while g < target:
        q = _get_q(curve, g)
        top = min(target, curve[q]['maxbat'], param['L'])
        if top <= g:
            return None
        pieces.append((top, int(np.ceil((top - g) / (rate * curve[q]['speed'])))))
        g = top
    return pieces


def replay_routes(inst, x_eners):
    """The relaxation's vehicle routes ({k: {edge: flow}} on inst.Ntl) followed with true
    travel times, energy use and charging: [(k, ok, [(station, charge, time), ...])].
    A route waits for its planned departure when it arrives early, and ends at its first
    arrival at its final station (it can wait there until T - 1; the relaxation, which
    rounds charge up, may add empty round trips after it that a real battery cannot
    make). ok is False when it runs out of battery, cannot charge as planned, or runs
    past T - 1. Charger capacity is not checked here: the MIP on the pessimistic network
    enforces it."""
    ctn, param, T = inst.Ntl, inst.param, inst.param['T']
    routes = []
    for k, flow in x_eners.items():
        flow = {e: x for e, x in flow.items() if x > 0.5}
        if not flow:
            continue
        for P in ctn.flow_decomposition(flow):
            i0, g, t = P[0][0][0], P[0][0][1], P[0][0][2]
            last = P[-1][1][0]
            states, ok = [(i0, g, t)], True
            for e in P:
                v1, v2, _ = e
                kind = ctn.edge_types[e]
                if v1[2] > t:                                   # wait for the planned departure
                    t = v1[2]
                    states.append((v1[0], g, t))
                if kind in _TRANSIT:
                    g, t = g - (ctn.edge_dists[e] - ctn.edge_charges[e]), t + ctn.edge_times[e]
                    ok = g >= 0
                    states.append((v2[0], g, t))
                elif kind == 'charge':
                    # charge to the level the relaxation plans after this edge (its head); the
                    # relaxation rounds charge up, so its later legs assume that level
                    target = max(v2[1], v1[1] - ctn.edge_dists[e] + ctn.edge_charges[e])
                    pieces = _charge_pieces(param, v1[0], g, target) if g < target else []
                    if pieces is None:
                        ok = False
                    for g, dt in pieces or []:
                        t += dt
                        states.append((v1[0], g, t))
                elif kind == 'swap':
                    g, t = v2[1], t + ctn.edge_times[e]
                    states.append((v2[0], g, t))
                if not ok or t > T - 1:
                    ok = False
                    break
                if kind in _TRANSIT and v2[0] == last:
                    break                                       # arrived: wait here until T - 1
            routes.append((k, ok, states))
    return routes


def earliest_route(inst, k):
    """Earliest-arrival route of commodity k's loaded truck (leaving its origin full at
    min_time; transit with consumption dH; charging as in the pessimistic network; swaps
    to a full battery where the mode allows them): [(station, charge, time), ...] up to
    its arrival at the destination, or None if it cannot arrive by T - 1."""
    import heapq
    param, N = inst.param, inst.N
    o, z = list(param['D'])[k]
    T, L, t0 = param['T'], param['L'], param.get('min_time', 0)
    curve = param['speed_curve']
    charge_nodes = set(param['charge_nodes'])
    swap_time = {i: param['bat_swap_time'] for i in param.get('battery_nodes', [])}
    swap_time.update({i: param['tr_swap_time'] for i in param.get('tractor_nodes', [])})
    succ = N.successors if N.is_directed() else N.neighbors
    best, prev = {(o, L): t0}, {}
    heap = [(t0, o, L)]

    def push(j, g2, t2, frm):
        if t2 <= T - 1 and t2 < best.get((j, g2), np.inf):
            best[(j, g2)], prev[(j, g2)] = t2, frm
            heapq.heappush(heap, (t2, j, g2))

    while heap:
        t, i, g = heapq.heappop(heap)
        if best.get((i, g), np.inf) < t:
            continue
        if i == z:
            states, s = [], (i, g)
            while s is not None:
                states.append((s[0], s[1], best[s]))
                s = prev.get(s)
            return states[::-1]
        for j in succ(i):
            dH = N[i][j]['dH']
            if dH <= g:
                push(j, _num(g - dH), t + int(N[i][j]['time']), (i, g))
        if i in charge_nodes and param['charge_rate'].get(i, 0) > 0:
            q = _get_q(curve, g)
            top = min(curve[q]['maxbat'], L)
            rate = param['charge_rate'][i] * curve[q]['speed']
            for g2 in range(int(np.floor(g)) + 1, int(top) + 1):
                push(i, g2, t + int(np.ceil((g2 - g) / rate)), (i, g))
        if i in swap_time and g < L:
            push(i, L, t + int(swap_time[i]), (i, g))
    return None


def route_points(inst, x_eners):
    """Points to add to the pessimistic network: the replayed routes that fit, plus the
    earliest-arrival route of every commodity with a route that does not.
    Returns ({station: {times}}, {station: {charges}}, summary)."""
    T, t_min, L = inst.param['T'], inst.param.get('min_time', 0), inst.param['L']
    times, charges = {}, {}

    def add(states):
        for i, g, t in states:
            g, t = _num(g), int(t)
            if t_min <= t <= T - 1:
                times.setdefault(i, set()).add(t)
            if isinstance(g, int) and 0 <= g <= L:
                charges.setdefault(i, set()).add(g)

    routes = replay_routes(inst, x_eners) if x_eners else []
    failed = set()
    for k, ok, states in routes:
        if ok:
            add(states)
        else:
            failed.add(k)
    covered = {k for k, ok, _ in routes if ok}
    fallback, unreachable = [], []
    for k in range(len(inst.param['D'])):
        if k in failed or k not in covered:
            route = earliest_route(inst, k)
            if route is None:
                unreachable.append(k)
            else:
                add(route)
                fallback.append(k)
    summary = {'routes': len(routes), 'replayed_ok': sum(ok for _, ok, _ in routes),
               'fallback': fallback, 'unreachable': unreachable,
               'points': sum(len(v) for v in times.values()) + sum(len(v) for v in charges.values())}
    return times, charges, summary
