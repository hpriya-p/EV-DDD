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
Costs never undercount: travel time runs to the rounded-up arrival (the real
wait also costs value_for_time), and the energy bought on a charging edge, for
its cost and for charger capacity, is the full amount to the true target.

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
                    for g2 in range(g + 1, int(min(UB + 1, g + param['charge_rate'][i] * speed_curve[q1]['speed'] + 1))):
                        if g2 <= L and t + 1 < T:
                            emit(v1, i, g2, t + 1, 'charge')
                else:
                    nb_dH = [dH for _, dH, _, _ in neighbors]
                    for g2 in sorted(set(charges_i + nb_dH)):
                        if g2 <= g or g2 > UB:
                            continue
                        t2 = int(np.ceil((g2 - g) / param['charge_rate'][i] * speed_curve[q1]['speed']))
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
    """

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
                            output=False, fixed_design=None):
    """
    Upper bound for an existing Instance: its network is replaced by the pessimistic
    network of the given discretization and the model is rebuilt and solved once.
    Any settings that inst applies in construct_model (e.g. a fixed design) carry over.

    fixed_design: optional {'a_c': {i: v}, 'a_s': {...}, 'n': {...}, 'z': {...}} ('a' is read
    as a_c); every a_c, a_s, n, z is then fixed to exactly that value (0 where absent), so the
    solve asks whether that design can be operated feasibly (and at what cost).
    """
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
                x.lb = x.ub = values.get(int(i), 0)
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
