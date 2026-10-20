"""Checks for correct_and_resolve.py on small corridor instances of the case study.

    python test_correct_and_resolve.py

1. One round from an early, a middle and the last Bi-DDD iteration: "feasible"
   comes with the known optimum; "not feasible" lists edges that really are rounded.
2. Repeated rounds from the first iteration reach a feasible solution at the optimum.
3. Three OD pairs: the objective never decreases from round to round (each
   correction only tightens the relaxation).
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
from correct_and_resolve import correct_and_resolve

CONFIG = ['heuristic', 'local_rebuild_update']


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def ddd_log(net, path, max_iter=10**6, **inputs):
    N, p = cs.make_instance_inputs(net, **inputs)
    p.update(MAX_ITER=max_iter, ddd_log=path)
    with quiet():
        inst = solve_worker.CaseStudyInstance(N, p, CONFIG, mip_gap=1e-4)
        _, val, _ = inst.run_DDD(LB=0, LP=False)
    return val


def run(net, log, inputs, **kw):
    N, p = cs.make_instance_inputs(net, **inputs)
    with quiet():
        return correct_and_resolve(N, p, CONFIG, log, instance_class=solve_worker.CaseStudyInstance,
                                   instance_kwargs={'mip_gap': 1e-4}, **kw)


def main():
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)
    net = cs.build_network()
    ok = True

    corridor = dict(mode='tractor', T=96, ods=['phx_sf'], corridor_k=1, latest_origin_time=16)
    log = os.path.join(tmp, 'k1.jsonl')
    opt = ddd_log(net, log, **corridor)
    its = [r['iteration'] for r in Instance_v2.read_ddd_log(log) if r['event'] == 'iteration']

    # 1. one round
    for it in sorted({its[0], its[len(its) // 2], its[-1]}):
        res = run(net, log, corridor, iteration=it, rounds=1)
        if res['status'] == 'feasible':
            good = abs(res['objective'] - opt) < 1e-4 * opt
        else:
            good = res['status'] == 'not feasible' and all(r['true_head'] != r['head'][:3] for r in res['rounded_edges'])
        ok &= good
        print(f"1. from iteration {it:>3}: {res['status']:<13} objective {res.get('objective', 0):9.2f} "
              f"(optimum {opt:.2f}), rounded edges left {len(res.get('rounded_edges', []))}  {'ok' if good else 'WRONG'}")

    # 2. repeated rounds from the first iteration
    res = run(net, log, corridor, iteration=its[0], rounds=60)
    good = res['status'] == 'feasible' and abs(res['objective'] - opt) < 1e-4 * opt
    ok &= good
    print(f"2. from iteration {its[0]} with up to 60 rounds: {res['status']} after {res['rounds']} rounds, "
          f"objective {res['objective']:.2f}  {'ok' if good else 'WRONG'}")

    # 3. three OD pairs, from a run stopped early
    three = dict(mode='tractor', T=240, corridor_k=1, latest_origin_time=16)
    log3 = os.path.join(tmp, 'k3.jsonl')
    ddd_log(net, log3, max_iter=12, **three)
    res = run(net, log3, three, rounds=3)
    objs = [h['objective'] for h in res['history']]
    good = all(b >= a * (1 - 1e-4) - 1e-6 for a, b in zip(objs, objs[1:]))
    ok &= good
    print(f"3. K=3: {res['status']} after {res['rounds']} rounds; objectives "
          f"{', '.join(f'{o:.2f}' for o in objs)}; rounded edges per round "
          f"{[h['rounded_edges'] for h in res['history']]}  {'ok' if good else 'WRONG'}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")


if __name__ == '__main__':
    main()
