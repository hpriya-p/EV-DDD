"""Print an upper bound for every running experiment solve.

For each solve that progress.py reports as running (on Savio: running according to
Slurm), take the time and charge discretization of the most recent iteration in its
ddd_log.jsonl, build the pessimistic charge-time network (upper_bound.py), solve the
model once and print the upper bound next to that iteration's lower bound.

Each solve is rebuilt from its spec.json exactly as the experiment ran it, including
a fixed base design (exp11's augmentation solves keep a, n, z >= the base design),
so the bound is for the same problem.

Each bound is one MIP of about the size of the solve's current DDD network, so on
Savio run this inside a job, not on the login node, e.g.
    srun --account=fc_evnetwork --partition=savio3_htc --qos=savio_normal -c 8 -t 4:00:00 \\
         python upper_bounds_running.py --threads 8 --time-limit 1800

Examples:
    python upper_bounds_running.py                      # every running solve
    python upper_bounds_running.py exp11 battery        # only labels containing these words
    python upper_bounds_running.py --states running stopped   # also solves that stopped
    python upper_bounds_running.py --json bounds.json   # also save the results
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def find_exp_dir(given=None):
    for d in (given, os.environ.get('EV_EXP_DIR'),
              os.path.join(os.path.dirname(HERE), 'case_study_experiments'),
              os.path.expanduser('~/case_study_experiments')):
        if d and os.path.exists(os.path.join(d, 'progress.py')):
            return os.path.abspath(d)
    raise SystemExit("case_study_experiments not found; pass --exp-dir")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('words', nargs='*', help='only solves whose label contains all of these')
    ap.add_argument('--states', nargs='+', default=['running'],
                    help='progress.py states to include (prefix match; default: running)')
    ap.add_argument('--time-limit', type=float, default=1800, help='seconds per upper-bound MIP')
    ap.add_argument('--mip-gap', type=float, default=1e-4)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--exp-dir', default=None, help='the case_study_experiments folder')
    ap.add_argument('--json', default=None, help='also write the results to this file')
    args = ap.parse_args()

    exp_dir = find_exp_dir(args.exp_dir)
    os.environ.setdefault('EV_CODE_DIR', HERE)
    sys.path.insert(0, exp_dir)
    sys.path.insert(0, HERE)
    import progress
    import case_study as cs
    import solve_worker
    from upper_bound import upper_bound_from_log
    import Instance_v2

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
        print(f"No solves in state {' / '.join(args.states)}"
              + (f" matching {' '.join(args.words)}" if args.words else '')
              + ("" if slurm is not None else " (squeue not available: states guessed from file times)"))
        return

    net = cs.build_network()
    results = []
    for run_dir, row in selected:
        with open(os.path.join(run_dir, 'spec.json')) as f:
            spec = json.load(f)
        log = os.path.join(run_dir, 'ddd_log.jsonl')
        entry = {'label': row['label'], 'hash': os.path.basename(run_dir), 'state': row['state']}
        if spec.get('kind') != 'ddd':
            entry['status'] = 'not a DDD solve'
        elif not os.path.exists(log):
            entry['status'] = 'no ddd_log yet'
        else:
            N, params = cs.make_instance_inputs(net, **spec['scenario'])
            config = [spec.get('config', 'heuristic'), spec.get('policy', 'local_rebuild_update')]
            t0 = time.time()
            try:
                res = upper_bound_from_log(
                    N, params, config, log,
                    instance_class=solve_worker.CaseStudyInstance,
                    instance_kwargs={'design': spec.get('design'), 'mip_gap': args.mip_gap,
                                     'threads': args.threads},
                    time_limit=args.time_limit, mip_gap=args.mip_gap, threads=args.threads)
                entry.update({k: res[k] for k in ('status', 'upper_bound', 'lower_bound', 'gap', 'iteration',
                                                  'mip_bound', 'network_size')})
                entry['design'] = {k: res['solution'][k] for k in ('a_c', 'a_s', 'n', 'z')} if res['solution'] else None
            except ValueError as e:          # e.g. no iteration logged yet
                entry['status'] = str(e)
            entry['seconds'] = time.time() - t0
        results.append(entry)
        fmt = lambda x: '--' if x is None else f"{x:,.2f}"
        gap = entry.get('gap')
        print(f"{entry['label']:<34} {entry['hash']}  iter {str(entry.get('iteration', '--')):>5}  "
              f"LB {fmt(entry.get('lower_bound')):>12}  UB {fmt(entry.get('upper_bound')):>12}  "
              f"gap {('--' if gap is None else f'{100 * gap:.2f}%'):>8}  {entry.get('status')}"
              f"  ({entry.get('seconds', 0):.0f}s)", flush=True)

    if args.json:
        with open(args.json, 'w') as f:
            # edge keys are tuples; _jsonable turns them into strings as in ddd_log
            json.dump(Instance_v2._jsonable(results), f, indent=2, default=str)
        print(f"wrote {args.json}")
    print("LB: the iteration's lower bound (from ddd_log). UB: objective of a feasible solution "
          "of the pessimistic network, valid even when the UB MIP stops at its time limit.")


if __name__ == '__main__':
    main()
