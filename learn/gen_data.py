#!/usr/bin/env python3
"""
Training data of the learned search (HANDOFF 16.25): variants of the base instance (learn/instances.py), each solved
by the exact model with a short time limit and checked by tools/check_solution.py; the checked designs are the labels.
Settings: configs/solver.json, learn.instances and learn.data.

  dse/bin/python learn/gen_data.py submit <out_dir> <n> [<first_seed>]   Slurm jobs over n seeds
  dse/bin/python learn/gen_data.py run <out_dir> <seed_from> <seed_to>    one job: its seeds in parallel processes
  dse/bin/python learn/gen_data.py solve <instance_dir>                   one variant: solve, save, check, label.json
  dse/bin/python learn/gen_data.py collect <out_dir>                      summary of all variants (index.json)
"""
import concurrent.futures
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, 'dse', 'bin', 'python')
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tools'))


def learn_cfg(cfg_dir=None):
    with open(os.path.join(cfg_dir or os.path.join(REPO, 'configs'), 'solver.json')) as f:
        return json.load(f).get('learn', {})


def solve(inst):
    """Solve one variant like main.py does (all uncertainty blocks on, its single rho level), without figures."""
    from config_reader import ConfigReader
    from architecture_loader import load_architecture_from_json
    from optimizer import AssignmentOptimizer, UNCERTAINTY_BLOCKS
    from check_solution import SolutionCheck
    inst = os.path.abspath(inst)
    cfg_dir, arch = os.path.join(inst, 'cfg'), os.path.join(inst, 'arch.json')
    cr = ConfigReader(cfg_dir)
    data = learn_cfg(cfg_dir).get('data', {})
    partitions = cr.get_partitions()
    partitions['hw_capacity'] = cr.get_hardware_capacities()
    unc = {k: True for k in UNCERTAINTY_BLOCKS}
    unc['params'] = cr.get_uncertainty_config()
    scs, comm, sensors, actuators, cable_types, locations = load_architecture_from_json(arch, config_reader=cr, num_locs=None)
    params = ','.join(p for p in (cr.get_solver_config().get('gurobi_params', ''), f"Threads={int(data.get('threads', 8))}") if p)
    t0 = time.time()
    sols = AssignmentOptimizer().optimize_growth_sweep(
        scs, locations, sensors, actuators, cable_types, comm, partitions=partitions, hardwares=cr.get_hardwares(),
        interfaces=cr.get_interfaces(), backbone=cr.get_backbone_config(), phase2_time=int(data.get('phase2_time', 30)),
        gurobi_params=params, lns=None, chains=cr.get_chains(), solve_time_limit=int(data.get('time_limit', 300)),
        uncertainty=unc)
    with open(os.path.join(inst, 'variant.json')) as f:
        label = json.load(f)
    label.update({'solve_s': round(time.time() - t0, 1), 'solved': bool(sols)})
    if sols:
        sol = sols[0]
        with open(os.path.join(inst, 'run_info.json'), 'w') as f:   # what the checker reads (main.py writes the same)
            json.dump({'args': {'config_dir': cfg_dir, 'architecture_json': arch, 'num_locs': None}}, f, indent=1)
        sol_path = os.path.join(inst, 'solution_1.json')
        with open(sol_path, 'w') as f:
            json.dump(sol, f, indent=1, default=str)
        chk = SolutionCheck(sol_path).run()
        print(chk.summary())
        ph1 = sol['solve_info']['phases'][0]
        label.update({'cost': float(sol['total_cost']), 'cable_length': float(sol.get('cable_length', 0.0)),
                      'phase1_status': ph1['status'], 'phase1_gap': ph1['mip_gap'], 'phase1_bound': ph1['bound'],
                      'check_ok': not chk.errors, 'check_errors': len(chk.errors),
                      'assignment': sol['assignment'],
                      'standby': {s: {'cell': sb['cell'], 'type': sb.get('type')} for s, sb in (sol.get('standby') or {}).items()}})
    with open(os.path.join(inst, 'label.json'), 'w') as f:
        json.dump(label, f, indent=1)


def run(out, a, b):
    """Make the variants a..b-1 (if missing) and solve them, data.parallel at a time, each in its own process."""
    from learn.instances import make_variant
    lc = learn_cfg()
    out = os.path.abspath(out)
    todo = []
    for seed in range(a, b):
        inst = os.path.join(out, f'v{seed:05d}')
        if os.path.exists(os.path.join(inst, 'label.json')):
            continue
        if not os.path.exists(os.path.join(inst, 'variant.json')):
            make_variant(seed, inst, lc.get('instances', {}))
        todo.append(inst)

    def one(inst):
        with open(os.path.join(inst, 'solve.log'), 'w') as log:
            return subprocess.run([PY, os.path.abspath(__file__), 'solve', inst], stdout=log, stderr=subprocess.STDOUT).returncode

    with concurrent.futures.ThreadPoolExecutor(max_workers=int(lc.get('data', {}).get('parallel', 4))) as ex:
        for inst, rc in zip(todo, ex.map(one, todo)):
            print(f"{os.path.basename(inst)}: exit {rc}", flush=True)


def submit(out, n, first=0):
    """One Slurm array task per data.per_job seeds; every task solves data.parallel variants at a time with data.threads
    Gurobi threads each. At most data.max_sessions solves run at once over all tasks (array throttle): every solve holds
    one session of the Gurobi WLS licence (academic baseline: 2 concurrent sessions, overage is tolerated only for a
    while)."""
    data = learn_cfg().get('data', {})
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    per_job, par, thr = int(data.get('per_job', 8)), int(data.get('parallel', 1)), int(data.get('threads', 32))
    tasks = -(-n // per_job)
    rounds = -(-per_job // par)
    minutes = int(rounds * (int(data.get('time_limit', 300)) + int(data.get('phase2_time', 30)) + 120) / 60) + 10
    throttle = max(1, int(data.get('max_sessions', 8)) // par)
    cmd = (f"cd {REPO} && a=$(( {first} + SLURM_ARRAY_TASK_ID * {per_job} )) && b=$(( a + {per_job} )) && "
           f"b=$(( b < {first + n} ? b : {first + n} )) && {PY} learn/gen_data.py run {out} $a $b")
    r = subprocess.run(['sbatch', '--parsable', '--job-name=gd', f'--array=0-{tasks - 1}%{throttle}', '--partition=cpu',
                        f'--cpus-per-task={par * thr}', '--mem=64G', f'--time={minutes // 60:02d}:{minutes % 60:02d}:00',
                        '--exclude=zits-slurmworker[2-3]', f'--output={out}/job_%a.txt', '--wrap', cmd],
                       capture_output=True, text=True)
    print(f"seeds {first}-{first + n - 1}: array job {r.stdout.strip() or r.stderr.strip()} ({tasks} tasks of {per_job} seeds, "
          f"at most {throttle * par} solves at once)")


def collect(out):
    """index.json: one row per variant (without the designs); printed summary by robustness level."""
    rows = []
    for p in sorted(glob.glob(os.path.join(out, 'v*', 'label.json'))):
        with open(p) as f:
            lb = json.load(f)
        rows.append({k: v for k, v in lb.items() if k not in ('assignment', 'standby')})
    with open(os.path.join(out, 'index.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    ok = [r for r in rows if r.get('check_ok')]
    print(f"{len(rows)} variants, {sum(1 for r in rows if r.get('solved'))} solved, {len(ok)} checked OK")
    for lo, hi in ((0.0, 0.0), (0.01, 0.4), (0.41, 0.7), (0.71, 1.0)):
        g = [r for r in ok if lo <= r['rho'] <= hi]
        if g:
            gaps = sorted(100 * (r['phase1_gap'] or 0.0) for r in g)
            print(f"  rho {lo:.2f}-{hi:.2f}: {len(g)} designs, proven {sum(1 for r in g if r['phase1_status'] == 'OPTIMAL')}, "
                  f"median gap {gaps[len(gaps) // 2]:.2f} %, worst {gaps[-1]:.2f} %")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'solve' and len(sys.argv) == 3:
        solve(sys.argv[2])
    elif cmd == 'run' and len(sys.argv) == 5:
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == 'submit' and len(sys.argv) in (4, 5):
        submit(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) == 5 else 0)
    elif cmd == 'collect' and len(sys.argv) == 3:
        collect(sys.argv[2])
    else:
        print(__doc__)
