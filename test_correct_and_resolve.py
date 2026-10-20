"""Checks for correct_and_resolve.py on small corridor instances of the case study.

    python test_correct_and_resolve.py

1. One round from an early, a middle and the last Bi-DDD iteration: a "feasible"
   result costs at least the known optimum, and exactly it when flagged optimal;
   "not feasible" lists edges that really are rounded; rounds that are not exact
   ran the path replay and the design-extension check.
2. Repeated rounds from the first iteration end feasible.
3. Three OD pairs: the lower bound never decreases from round to round, and a
   feasible result costs at least the lower bound.
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

    def judge(res):
        """A feasible result is a real solution: cost >= optimum, = optimum when flagged optimal."""
        st = res['status']
        if st.startswith('feasible'):
            good = res['objective'] >= opt - 1e-4 * opt and (not res['optimal'] or abs(res['objective'] - opt) < 1e-4 * opt)
        else:
            good = st == 'not feasible' and all(r['true_head'] != r['head'][:3] for r in res['rounded_edges'])
        checks_ran = all(('replay' in h and 'extension' in h) for h in res['history'] if h['rounded_edges'])
        return good and checks_ran

    # 1. one round
    for it in sorted({its[0], its[len(its) // 2], its[-1]}):
        res = run(net, log, corridor, iteration=it, rounds=1)
        good = judge(res)
        ok &= good
        h0 = res['history'][0]
        extra = (f"; replay: {h0['replay']['failing']}/{h0['replay']['paths']} paths fail; "
                 f"extension: {h0['extension']['status']}") if 'replay' in h0 else ''
        print(f"1. from iteration {it:>3}: {res['status']:<26} objective {res.get('objective', 0):9.2f} "
              f"optimal={res.get('optimal')} (optimum {opt:.2f}) after {res['rounds']} rounds{extra}  "
              f"{'ok' if good else 'WRONG'}")

    # 2. repeated rounds from the first iteration
    res = run(net, log, corridor, iteration=its[0], rounds=60)
    good = judge(res) and res['status'].startswith('feasible')
    ok &= good
    print(f"2. from iteration {its[0]} with up to 60 rounds: {res['status']} after {res['rounds']} rounds, "
          f"objective {res['objective']:.2f}, optimal={res['optimal']}  {'ok' if good else 'WRONG'}")

    # 3. three OD pairs, from a run stopped early
    three = dict(mode='tractor', T=240, corridor_k=1, latest_origin_time=16)
    log3 = os.path.join(tmp, 'k3.jsonl')
    ddd_log(net, log3, max_iter=12, **three)
    res = run(net, log3, three, rounds=3)
    lbs = [h['mip_bound'] for h in res['history']]
    good = all(b >= a * (1 - 1e-4) - 1e-6 for a, b in zip(lbs, lbs[1:]))
    if res['status'].startswith('feasible'):
        good &= res['objective'] >= res['lower_bound'] - 1e-6
    ok &= good
    gap = f", gap {100 * res['gap']:.2f}%" if 'gap' in res else ''
    print(f"3. K=3: {res['status']} after {res['rounds']} rounds, objective {res.get('objective', 0):.2f}, "
          f"lower bounds {', '.join(f'{x:.2f}' for x in lbs)}{gap}", end='')
    print(f"  {'ok' if good else 'WRONG'}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")


if __name__ == '__main__':
    main()
