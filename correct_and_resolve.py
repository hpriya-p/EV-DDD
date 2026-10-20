"""Correct the rounded edges of a Bi-DDD solution and solve again.

From the times and charges of one iteration of a ddd_log (default: the latest):
  1. build the Bi-DDD charge-time network (charge rounded up, time rounded down at
     edge heads: a relaxation) and the model, and solve it;
  2. find every rounded edge of the solution: an edge carrying load or energy flow
     whose true head (charge, arrival time) is not a retained node;
  3. add each such head's charge to the head station's charges list and its arrival
     time to the station's times list;
  4. rebuild the whole DDD network from the corrected lists (construct()) and the
     model, and solve again;
  5. if the new solution uses only exact edges it is feasible for the full problem,
     and since the network is still a relaxation it is also optimal (as in Bi-DDD's
     stopping rule): return it. Otherwise report that it is not feasible, with
     the rounded edges it still uses.
Steps 2-4 are repeated up to --rounds times (default 1).

After every solve (the first and each corrected one), three checks, in order:
  A. exact edges: the solution uses only exact edges -> feasible and optimal; stop.
  B. path replay (diagnostic only): each load and energy path of the solution is
     followed with true charges and arrival times, vehicles waiting when they arrive
     later than planned; reports the paths that run out of battery or arrive after
     T-1. Paths are checked one at a time (shared tractors and charger capacity are
     not), so this explains failures but does not certify feasibility.
  C. extension of the design: the solution's design (a, n, z) is fixed exactly and
     the model is solved on the pessimistic network of the same discretization
     (upper_bound.py), whose solutions are all feasible for the full problem.
     Feasible -> return that solution: the design can be extended to a feasible
     solution; it is optimal when its cost equals the relaxation's lower bound
     (within the MIP gap), otherwise its cost is an upper bound. Infeasible -> the
     design cannot be extended with this discretization; correct and continue.

With the 'heuristic' network, "optimal" is for the heuristic problem; the solution
is feasible for the full problem.

Use:
    from correct_and_resolve import correct_and_resolve
    res = correct_and_resolve(N, parameters, config, 'ddd_log.jsonl', rounds=1)
    res['status']   # 'feasible (exact edges)' | 'feasible (extended design)' | 'not feasible' | 'model infeasible'
    res['objective'], res['optimal'], res['solution'], res['rounded_edges'], res['history']

Command line, for experiment solves (labels as in progress.py):
    python correct_and_resolve.py exp11 tractor/MMM
    python correct_and_resolve.py exp11 battery/MHM --rounds 3 --json result.json
On Savio run it inside a job (srun ...), not on the login node.
"""
import argparse
import json
import os
import sys
import time

import gurobipy as gp

import Instance_v2
from upper_bound import _solution, upper_bound_on_instance

HERE = os.path.dirname(os.path.abspath(__file__))


def _solve(inst, mip_gap, time_limit, threads):
    m = inst.model
    m.setParam('OutputFlag', 0)
    m.setParam('MIPGap', mip_gap)
    if threads:
        m.setParam('Threads', threads)
    if time_limit:
        m.setParam('TimeLimit', time_limit)
    m.optimize()
    return m


def _rounded_edges(inst):
    """Edges with load or energy flow whose true head is not their head node."""
    used = {e for (e, k), x in list(inst.x_load.items()) + list(inst.x_ener.items()) if x.X > 0.5}
    return sorted((e for e in used if e in inst.Ntl.edge_types and not inst.Ntl.is_exact(e)), key=str)


def _true_head(ctn, e):
    v1, v2, _ = e
    return v2[0], v1[1] - ctn.edge_dists[e] + ctn.edge_charges[e], v1[2] + ctn.edge_times[e]


def _correct(ctn, edges):
    """Add the true heads of `edges` to the charges and times lists. Returns points added."""
    T, t_min = ctn.param['T'], ctn.param['min_time']
    added = []
    for e in edges:
        j, g, t = _true_head(ctn, e)
        g, t = int(g), int(t)
        if g not in ctn.charges[j]:
            ctn.charges[j] = sorted(set(ctn.charges[j]) | {g})
            added.append(('charge', j, g))
        if t_min <= t <= T - 1 and t not in ctn.times[j]:
            ctn.times[j] = sorted(set(ctn.times[j]) | {t})
            added.append(('time', j, t))
    return added


def _replay(inst):
    """Check B: follow every load and energy path with true values (diagnostic only).

    Returns {'paths', 'failing', 'max_delay', 'examples'}: a path fails when its real
    battery level goes below 0 or it would arrive after T-1. Vehicles wait when they
    arrive later than planned; a charging edge takes the time needed from the real
    charge level; a swap sets the charge to the exact level swapped to.
    """
    ctn, T = inst.Ntl, inst.param['T']
    curve, rate = inst.param['speed_curve'], inst.param['charge_rate']
    n_paths, failing, max_delay, examples = 0, 0, 0, []
    for name, var in (('load', inst.x_load), ('energy', inst.x_ener)):
        for k in range(inst.K):
            flow = {e: x.X for (e, kk), x in var.items() if kk == k and x.X > 0.5}
            for P, val in ctn.flow_decomposition(flow).items():
                n_paths += 1
                g, t = P[0][0][1], P[0][0][2]
                failure = None
                for e in P:
                    v1, v2, _ = e
                    kind = ctn.edge_types[e]
                    t = max(t, v1[2])                                    # wait for the planned departure
                    if kind in ('transit_H', 'transit_L'):
                        g -= ctn.edge_dists[e] - ctn.edge_charges[e]     # energy used on the leg
                        t += ctn.edge_times[e]
                        if g < 0:
                            failure = f"battery runs out on {kind} {v1[:3]} -> {v2[0]}"
                    elif kind == 'charge':
                        target = v1[1] - ctn.edge_dists[e] + ctn.edge_charges[e]
                        if g < target:
                            speed = curve[ctn.get_q(v1[0], g)]['speed']
                            t += int(-(-(target - g) // (rate[v1[0]] * speed)))   # ceil
                            g = target
                    elif kind == 'swap':
                        g, t = v2[1], t + ctn.edge_times[e]
                    elif kind == 'wait':
                        t = max(t, v2[2])
                    if failure is None and t > T - 1:
                        failure = f"arrives after T-1 at {v2[0]}"
                    if failure:
                        break
                    max_delay = max(max_delay, t - v2[2])
                if failure:
                    failing += 1
                    if len(examples) < 5:
                        examples.append(f"{name} commodity {k} (flow {val:g}): {failure}")
    return {'paths': n_paths, 'failing': failing, 'max_delay': max_delay, 'examples': examples}


def _extension(N, params, config, instance_class, instance_kwargs, inst, mip_gap, time_limit, threads):
    """Check C: the design of inst's solution fixed exactly, solved on the pessimistic network."""
    design = _solution(inst)
    design = {name: design[name] for name in ('a', 'n', 'z')}
    probe = instance_class(N, dict(params, initial_discretization=None), list(config),
                           network_only=True, **(instance_kwargs or {}))
    res = upper_bound_on_instance(probe, inst.Ntl.times, inst.Ntl.charges, time_limit=time_limit,
                                  mip_gap=mip_gap, threads=threads, fixed_design=design)
    res['design'] = design
    return res


def correct_and_resolve(N, parameters, config, ddd_log=None, iteration=None, rounds=1,
                        instance_class=Instance_v2.Instance, instance_kwargs=None,
                        mip_gap=1e-4, time_limit=None, threads=0, discretization=None):
    """See the module docstring. The discretization comes from ddd_log (one iteration,
    default the latest), or is given directly as discretization={'times': {node: [...]},
    'charges': {node: [...]}, 'iteration': ..., 'lower_bound': ...} (e.g. by a running solve)."""
    if discretization is not None:
        disc = {'iteration': discretization.get('iteration'), 'lower_bound': discretization.get('lower_bound'),
                'times': {int(i): [int(t) for t in ts] for i, ts in discretization['times'].items()},
                'charges': {int(i): [int(g) for g in gs] for i, gs in discretization['charges'].items()}}
    else:
        disc = Instance_v2.discretization_from_log(ddd_log, iteration)
    params = dict(parameters)
    params.pop('initial_discretization', None)
    params.pop('ddd_log', None)
    params['initial_discretization'] = {'times': disc['times'], 'charges': disc['charges']}
    t0 = time.time()
    inst = instance_class(N, params, list(config), **(instance_kwargs or {}))
    history = []

    common = {'iteration': disc['iteration'], 'lower_bound_from_log': disc.get('lower_bound')}
    for rnd in range(rounds + 1):
        m = _solve(inst, mip_gap, time_limit, threads)
        if m.SolCount == 0:
            return dict(common, status='model infeasible' if m.Status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD)
                        else f'no solution (gurobi status {m.Status})',
                        rounds=rnd, history=history, seconds=time.time() - t0)
        rounded = _rounded_edges(inst)
        lower = m.ObjBound          # proven lower bound of this relaxation
        step = {'round': rnd, 'objective': m.ObjVal, 'mip_bound': lower,
                'network_size': [inst.Ntl.Ntl.number_of_nodes(), inst.Ntl.Ntl.number_of_edges()],
                'rounded_edges': len(rounded)}
        history.append(step)

        # A. exact edges: feasible, and optimal since the network is a relaxation
        if not rounded:
            return dict(common, status='feasible (exact edges)', objective=m.ObjVal, lower_bound=lower,
                        optimal=m.Status == gp.GRB.OPTIMAL, solution=_solution(inst), rounded_edges=[],
                        rounds=rnd, history=history, seconds=time.time() - t0)

        # B. path replay (diagnostic)
        step['replay'] = _replay(inst)

        # C. can this design be extended to a feasible solution?
        ext = _extension(N, params, config, instance_class, instance_kwargs, inst, mip_gap, time_limit, threads)
        step['extension'] = {'status': ext['status'], 'objective': ext['upper_bound'],
                             'seconds': ext['solve_time'] + ext['build_time']}
        if ext['upper_bound'] is not None:
            ub = ext['upper_bound']
            return dict(common, status='feasible (extended design)', objective=ub, lower_bound=lower,
                        gap=(ub - lower) / ub if ub else 0.0,
                        optimal=ub - lower <= mip_gap * abs(ub) + 1e-6,
                        solution=ext['solution'], rounded_edges=_describe(inst, rounded),
                        rounds=rnd, history=history, seconds=time.time() - t0)

        if rnd == rounds:
            break
        # correct the lists, then recompute the whole network and the model
        step['points_added'] = len(_correct(inst.Ntl, rounded))
        inst.Ntl.construct()
        inst.construct_model()

    return dict(common, status='not feasible', objective=m.ObjVal, lower_bound=lower, optimal=False,
                solution=_solution(inst), rounded_edges=_describe(inst, rounded),
                rounds=len(history) - 1, history=history, seconds=time.time() - t0)


def _describe(inst, edges):
    return [{'tail': list(map(int, e[0])), 'head': list(map(int, e[1])), 'type': inst.Ntl.edge_types[e],
             'true_head': [int(x) for x in _true_head(inst.Ntl, e)]} for e in edges]


# ---------------------------------------------------------------------------
# Command line: experiment solves
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('words', nargs='+', help='label words of the experiment solve(s), e.g. exp11 tractor/MMM')
    ap.add_argument('--states', nargs='+', default=[''],
                    help='only solves in these progress.py states (prefix match; default: any)')
    ap.add_argument('--iteration', type=int, default=None, help='ddd_log iteration (default: latest)')
    ap.add_argument('--rounds', type=int, default=1, help='correction rounds (default 1)')
    ap.add_argument('--time-limit', type=float, default=1800, help='seconds per MIP solve')
    ap.add_argument('--mip-gap', type=float, default=1e-4)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--exp-dir', default=None, help='the case_study_experiments folder')
    ap.add_argument('--json', default=None, help='also write the results to this file')
    args = ap.parse_args()

    from upper_bounds_running import find_exp_dir
    exp_dir = find_exp_dir(args.exp_dir)
    os.environ.setdefault('EV_CODE_DIR', HERE)
    sys.path.insert(0, exp_dir)
    import progress
    import case_study as cs
    import solve_worker

    slurm = progress.slurm_tasks()
    selected = []
    for d in sorted(os.listdir(progress.RUNS)):
        run_dir = os.path.join(progress.RUNS, d)
        if not os.path.exists(os.path.join(run_dir, 'spec.json')):
            continue
        row = progress.row(run_dir, slurm)
        if any(row['state'].startswith(s) for s in args.states) and all(w in row['label'] for w in args.words):
            selected.append((run_dir, row))
    if not selected:
        print(f"No solves match {' '.join(args.words)} in state {' / '.join(args.states) or 'any'}.")
        return

    net = cs.build_network()
    results = []
    for run_dir, row in selected:
        with open(os.path.join(run_dir, 'spec.json')) as f:
            spec = json.load(f)
        log = os.path.join(run_dir, 'ddd_log.jsonl')
        entry = {'label': row['label'], 'hash': os.path.basename(run_dir), 'state': row['state']}
        if spec.get('kind') != 'ddd' or not os.path.exists(log):
            entry['status'] = 'no ddd_log' if spec.get('kind') == 'ddd' else 'not a DDD solve'
        else:
            N, params = cs.make_instance_inputs(net, **spec['scenario'])
            config = [spec.get('config', 'heuristic'), spec.get('policy', 'local_rebuild_update')]
            try:
                res = correct_and_resolve(
                    N, params, config, log, iteration=args.iteration, rounds=args.rounds,
                    instance_class=solve_worker.CaseStudyInstance,
                    instance_kwargs={'design': spec.get('design'), 'mip_gap': args.mip_gap,
                                     'threads': args.threads},
                    mip_gap=args.mip_gap, time_limit=args.time_limit, threads=args.threads)
                entry.update(res)
            except ValueError as e:      # e.g. no iteration logged yet
                entry['status'] = str(e)
        results.append(entry)

        fmt = lambda x: '--' if x is None else f"{x:,.2f}"
        hist = entry.get('history') or []
        print(f"{entry['label']}  ({entry['hash']}, ddd_log iteration {entry.get('iteration', '--')}, "
              f"its lower bound {fmt(entry.get('lower_bound_from_log'))})")
        for h in hist:
            print(f"  round {h['round']}: relaxation {fmt(h['objective'])} (lower bound {fmt(h['mip_bound'])}), "
                  f"network {h['network_size'][1]:,} edges, rounded edges used {h['rounded_edges']}"
                  + (f", {h['points_added']} points added" if 'points_added' in h else ''))
            if 'replay' in h:
                r = h['replay']
                print(f"    B. path replay: {r['failing']} of {r['paths']} paths fail with true values"
                      + (f" (max delay {r['max_delay']} steps)" if r['max_delay'] else '')
                      + ''.join(f"\n       - {x}" for x in r['examples'][:3]))
            if 'extension' in h:
                x = h['extension']
                print(f"    C. design fixed, pessimistic network: {x['status']}"
                      + (f", cost {fmt(x['objective'])}" if x['objective'] is not None else ''))
        status = entry.get('status', '')
        if status.startswith('feasible'):
            how = ('its solution uses only exact edges' if 'exact' in status
                   else "the relaxation's design can be operated feasibly")
            verdict = ('optimal' if entry.get('optimal')
                       else f"not proven optimal: gap {100 * entry.get('gap', 0):.2f}% to the lower bound"
                       if 'gap' in entry else 'not proven optimal (MIP stopped at its time limit)')
            print(f"  FEASIBLE ({how}; {verdict}): cost {fmt(entry['objective'])}; "
                  f"chargers a={entry['solution']['a']}, surplus n={entry['solution']['n']}")
        elif status == 'not feasible':
            print(f"  NOT FEASIBLE: the solution still uses {len(entry['rounded_edges'])} rounded edges and its "
                  f"design cannot be operated on the pessimistic network; e.g. "
                  + '; '.join(f"{r['type']} {r['tail'][:3]} -> {r['head'][:3]} (true head {r['true_head']})"
                              for r in entry['rounded_edges'][:3]))
        else:
            print(f"  {status}")
        print(f"  ({entry.get('seconds', 0):.0f}s)", flush=True)

    if args.json:
        with open(args.json, 'w') as f:
            # edge keys are tuples; _jsonable turns them into strings as in ddd_log
            json.dump(Instance_v2._jsonable(results), f, indent=2, default=str)
        print(f"wrote {args.json}")


if __name__ == '__main__':
    main()
