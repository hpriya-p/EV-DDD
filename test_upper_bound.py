"""Checks for upper_bound.py on small corridor instances of the case study.

    python test_upper_bound.py

1. The bound is valid: UB >= the known optimum and >= the iteration's lower bound,
   for the discretization of an early, a middle and the last Bi-DDD iteration.
2. The UB solution is implementable: on every edge it uses, the true head has at
   least the node's charge and arrives no later than the node's time.
3. Several OD pairs (K = 3): UB >= LB.
"""
import contextlib
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(HERE), 'case_study_experiments')
os.environ.setdefault('EV_CODE_DIR', HERE)
sys.path.insert(0, EXP)
sys.path.insert(0, HERE)
import case_study as cs
import solve_worker
import Instance_v2
from upper_bound import upper_bound_from_log, upper_bound


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def run_ddd(net, log, **inputs):
    N, p = cs.make_instance_inputs(net, **inputs)
    p['MAX_ITER'] = 10**6
    p['ddd_log'] = log
    with _quiet():
        inst = solve_worker.CaseStudyInstance(N, p, ['heuristic', 'local_rebuild_update'], mip_gap=1e-4)
        _, val, _ = inst.run_DDD(LB=0, LP=False)
    return N, p, val


def dominance_violations(res, N):
    """Edges of the UB solution whose true head is below/after the node they land on."""
    inst = res['_inst']
    bad = []
    for name in ('x_load', 'x_ener'):
        for k, flows in res['solution'][name].items():
            for (v1, v2, e_type) in flows:
                for key in inst.Ntl.Ntl[v1][v2]:
                    e = (v1, v2, key)
                    if inst.Ntl.edge_types[e] != e_type or e_type == 'wait':
                        continue
                    g_raw, t_raw = inst.Ntl.edge_raw[e]
                    if g_raw < v2[1] or t_raw > v2[2]:
                        bad.append((e, g_raw, t_raw))
    return bad


def main():
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)
    net = cs.build_network()
    ok = True

    # 1 + 2: single OD pair, known optimum
    corridor = dict(mode='tractor', T=96, ods=['phx_sf'], corridor_k=1, latest_origin_time=16)
    log = os.path.join(tmp, 'k1.jsonl')
    N, p, opt = run_ddd(net, log, **corridor)
    its = [r['iteration'] for r in Instance_v2.read_ddd_log(log) if r['event'] == 'iteration']
    for it in sorted({its[0], its[len(its) // 2], its[-1]}):
        N2, p2 = cs.make_instance_inputs(net, **corridor)
        with _quiet():
            res = upper_bound_from_log(N2, p2, ['heuristic', 'local_rebuild_update'], log, iteration=it,
                                       time_limit=300)
        ub, lb = res['upper_bound'], res['lower_bound']
        valid = ub is None or (ub >= opt - 1e-6 and ub >= lb - 1e-6)
        ok &= valid
        print(f"1. iteration {it:>3}: LB {lb:9.2f}  UB {('infeasible' if ub is None else f'{ub:9.2f}'):>10}  "
              f"optimum {opt:.2f}  {'ok' if valid else 'INVALID'}  ({res['status']})")

    # implementability of the UB solution at the last iteration
    disc = Instance_v2.discretization_from_log(log)
    N2, p2 = cs.make_instance_inputs(net, **corridor)
    inst = solve_worker.CaseStudyInstance(N2, p2, ['heuristic', 'local_rebuild_update'], network_only=True)
    from upper_bound import upper_bound_on_instance
    with _quiet():
        res = upper_bound_on_instance(inst, disc['times'], disc['charges'], time_limit=300)
    res['_inst'] = inst
    bad = dominance_violations(res, N2) if res['solution'] else []
    ok &= not bad
    print(f"2. UB solution edges whose true head is below or after their node: {len(bad)}")

    # 3: three OD pairs, stopped early
    log3 = os.path.join(tmp, 'k3.jsonl')
    three = dict(mode='tractor', T=240, corridor_k=1, latest_origin_time=16)
    N3, p3 = cs.make_instance_inputs(net, **three)
    p3['MAX_ITER'] = 16
    p3['ddd_log'] = log3
    with _quiet():
        inst3 = solve_worker.CaseStudyInstance(N3, p3, ['heuristic', 'local_rebuild_update'], mip_gap=1e-4)
        inst3.run_DDD(LB=0, LP=False)
    N3b, p3b = cs.make_instance_inputs(net, **three)
    with _quiet():
        res3 = upper_bound_from_log(N3b, p3b, ['heuristic', 'local_rebuild_update'], log3, time_limit=600)
    ub3, lb3 = res3['upper_bound'], res3['lower_bound']
    valid3 = ub3 is None or ub3 >= lb3 - 1e-6
    ok &= valid3
    print(f"3. K=3, iteration {res3['iteration']}: LB {lb3:.2f}  UB "
          f"{'infeasible' if ub3 is None else f'{ub3:.2f}'}  gap "
          f"{'--' if res3['gap'] is None else f'{100 * res3['gap']:.2f}%'}  "
          f"{'ok' if valid3 else 'INVALID'}  ({res3['status']})")
    print("ALL OK" if ok else "SOME CHECKS FAILED")


if __name__ == '__main__':
    main()
