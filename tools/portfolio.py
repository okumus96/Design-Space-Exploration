#!/usr/bin/env python3
"""
Seed portfolio for the robustness sweep (configs/solver.json: portfolio.seeds). The same instance is solved once per seed
in parallel Slurm jobs (one job per rho level and seed, so the levels run in parallel too); a dependent collect job then
keeps, per level, the cheapest checked design and writes the usual cost_vs_growth.dat / solution_<i>.json into <out>/best.
Rationale (2026-10-26 benchmark, 10 cells rho 1): the same model and settings reach gaps of 0.6 / 1.5 / 20 % with seeds
3 / 1 / 2 after 1 h; the best of three meets the 1 % target where a single run does not reliably.

  python tools/portfolio.py submit <out_dir> --architecture_json ... --num_locs N [--time_limit S] [--phase2_time S]
                                   [--config_dir configs] [main.py flags such as --uncertainty]
  python tools/portfolio.py collect <out_dir>          (run by the dependent job; can be rerun by hand)
"""
import glob
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, 'dse', 'bin', 'python')


def submit(out, main_args):
    cfg_dir = REPO + '/configs'
    for i, a in enumerate(main_args):
        if a == '--config_dir':
            cfg_dir = os.path.join(REPO, main_args[i + 1])
    solver = json.load(open(os.path.join(cfg_dir, 'solver.json')))
    seeds = [int(s) for s in solver.get('portfolio', {}).get('seeds', [1, 2, 3])]
    base_params = solver.get('gurobi_params', '')
    levels = json.load(open(os.path.join(cfg_dir, 'uncertainty.json')))['rho']['levels']
    out = os.path.join(REPO, out)
    os.makedirs(out, exist_ok=True)
    jobs = []
    for lv in levels:
        for seed in seeds:
            run = os.path.join(out, f'rho{lv:g}_seed{seed}')
            cfg = os.path.join(run, 'cfg')
            shutil.rmtree(run, ignore_errors=True)
            shutil.copytree(cfg_dir, cfg)
            u = json.load(open(os.path.join(cfg, 'uncertainty.json')))
            u['rho']['levels'] = [lv]
            json.dump(u, open(os.path.join(cfg, 'uncertainty.json'), 'w'), indent=2)
            args = [a for a in main_args]
            if '--config_dir' in args:
                args[args.index('--config_dir') + 1] = os.path.relpath(cfg, REPO)
            else:
                args += ['--config_dir', os.path.relpath(cfg, REPO)]
            params = ','.join(p for p in (base_params, f'Seed={seed}') if p)
            cmd = (f"cd {REPO} && {PY} main.py " + ' '.join(args) + f" --output_dir {os.path.relpath(run, REPO)}/out --gurobi_params '{params}'"
                   f" && for f in {run}/out/solution_*.json; do {PY} tools/check_solution.py $f | tail -n 2; done")
            r = subprocess.run(['sbatch', '--parsable', f'--job-name=pf_rho{lv:g}_s{seed}', '--partition=cpu', '--cpus-per-task=32', '--mem=64G',
                                '--time=02:00:00', '--exclude=zits-slurmworker[2-3]', f'--output={run}.txt', '--wrap', cmd],
                               capture_output=True, text=True)
            jid = r.stdout.strip().split(';')[0]
            jobs.append(jid)
            print(f'rho {lv:g} seed {seed}: job {jid or r.stderr.strip()}')
    dep = ':'.join(j for j in jobs if j)
    r = subprocess.run(['sbatch', '--parsable', '--job-name=pf_collect', '--partition=cpu', '--cpus-per-task=2', '--mem=4G', '--time=00:20:00',
                        f'--dependency=afterany:{dep}', f'--output={out}/collect.txt', '--wrap',
                        f"cd {REPO} && {PY} tools/portfolio.py collect {os.path.relpath(out, REPO)}"], capture_output=True, text=True)
    print('collect job', r.stdout.strip() or r.stderr.strip())


def collect(out):
    out = os.path.join(REPO, out)
    best_dir = os.path.join(out, 'best')
    shutil.rmtree(best_dir, ignore_errors=True)
    os.makedirs(best_dir)
    by_level = {}
    for run in sorted(glob.glob(os.path.join(out, 'rho*_seed*'))):
        log = run + '.txt'
        ok = os.path.exists(log) and '[check]' in open(log).read() and 'ERROR' not in open(log).read()
        for sol in glob.glob(os.path.join(run, 'out', 'solution_*.json')):
            s = json.load(open(sol))
            lv = s['solve_info'].get('rho')
            by_level.setdefault(lv, []).append((float(s['total_cost']), ok, sol, os.path.basename(run)))
    rows = ['rho cost units partitions seed gap_pct checked_ok']
    for i, lv in enumerate(sorted(k for k in by_level if k is not None), start=1):
        cands = sorted(by_level[lv], key=lambda c: (not c[1], c[0]))
        cost, ok, sol, run = cands[0]
        s = json.load(open(sol))
        shutil.copy(sol, os.path.join(best_dir, f'solution_{i}.json'))
        gap = s['solve_info']['phases'][0].get('mip_gap')
        rows.append(f"{lv:g} {cost:.2f} {s.get('num_locations_used', '-')} {s.get('num_partitions', '-')} {run.split('seed')[-1]} "
                    f"{(100 * gap) if gap is not None else float('nan'):.2f} {ok}")
        print(f"rho {lv:g}: best {cost:.2f} from {run} (gap {100 * gap if gap is not None else float('nan'):.2f} %, checker {'OK' if ok else 'NOT OK'}); all: "
              + ', '.join(f"{c:.0f}" for c, _o, _s, _r in sorted(by_level[lv])))
    with open(os.path.join(best_dir, 'cost_vs_rho_portfolio.dat'), 'w') as f:
        f.write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'submit':
        submit(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) >= 3 and sys.argv[1] == 'collect':
        collect(sys.argv[2])
    else:
        print(__doc__)
