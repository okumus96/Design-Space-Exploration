#!/usr/bin/env python3
"""
Speed benchmark of the optimiser (item A.3): fixed instances, the same measurement for every code variant.

  python tools/speed_bench.py submit <tag> [gurobi params, e.g. "MIPFocus=1"]
                                                snapshot the working tree (tracked files) into results/speed/<tag>/code
                                                and submit one Slurm job per instance (run from that snapshot, so later
                                                edits do not change a running measurement), each followed by the checker
  python tools/speed_bench.py report <tag> ...  table per tag: time to the first design, gap after 15 / 30 / 60 min,
                                                final status, candidates rejected by the lazy checks, checker verdict

Targets (HANDOFF, 2026-10-24): 10 cells rho 1 first design <= 5 min and gap <= 1 % within 1 h; rho 0.2 proven.
"""
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, 'dse', 'bin', 'python')
OUT = os.path.join(REPO, 'results', 'speed')
TIME_LIMIT = 3600
# name, architecture, cells, robustness level rho
INSTANCES = [
    ('c10_rho1', 'configs/full_architecture_v2.json', 10, 1.0),
    ('c10_rho02', 'configs/full_architecture_v2.json', 10, 0.2),
    ('c10_rho06', 'configs/full_architecture_v2.json', 10, 0.6),   # hardest point of the sweep (gap 10 % after 1 h)
    ('lite_rho1', 'configs/full_architecture_v2_lite.json', 4, 1.0),
]
MARKS = (900, 1800, 3600)


def snapshot(tag):
    code = os.path.join(OUT, tag, 'code')
    shutil.rmtree(code, ignore_errors=True)
    os.makedirs(code)
    # tracked files plus new, not ignored ones (a new module such as benders.py must travel with the snapshot)
    files = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.split('\n')
    for f in filter(None, files):
        src = os.path.join(REPO, f)
        if os.path.isfile(src):
            os.makedirs(os.path.join(code, os.path.dirname(f)), exist_ok=True)
            shutil.copy2(src, os.path.join(code, f))
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--short', '--untracked-files=no'], cwd=REPO, capture_output=True, text=True).stdout
    with open(os.path.join(OUT, tag, 'version.txt'), 'w') as f:
        f.write(f"HEAD {head}\n{dirty}")
    return code


def submit(tag, grb=None):
    code = snapshot(tag)
    if grb:
        with open(os.path.join(OUT, tag, 'version.txt'), 'a') as f:
            f.write(f"gurobi_params {grb}\n")
    for name, arch, cells, rho in INSTANCES:
        cfg = os.path.join(OUT, tag, f'cfg_{name}')
        shutil.rmtree(cfg, ignore_errors=True)
        shutil.copytree(os.path.join(code, 'configs'), cfg)
        p = os.path.join(cfg, 'uncertainty.json')
        u = json.load(open(p))
        u['rho']['levels'] = [rho]
        json.dump(u, open(p, 'w'), indent=2)
        run = os.path.join(OUT, tag, name)
        shutil.rmtree(run, ignore_errors=True)
        cmd = (f"cd {code} && {PY} main.py --architecture_json {arch} --num_locs {cells} --uncertainty "
               f"--time_limit {TIME_LIMIT} --phase2_time 60 --config_dir {cfg} --output_dir {run}" + (f" --gurobi_params '{grb}'" if grb else '') + " && "
               f"for f in {run}/solution_*.json; do {PY} tools/check_solution.py $f | tail -n 3; done")
        r = subprocess.run(['sbatch', f'--job-name=sb_{tag}_{name}'[:40], '--partition=cpu', '--cpus-per-task=32', '--mem=64G',
                            '--time=01:30:00', '--exclude=zits-slurmworker[2-3]', f'--output={run}.txt', '--wrap', cmd],
                           capture_output=True, text=True)
        print(name, r.stdout.strip() or r.stderr.strip())


NODE = re.compile(r'^\s*[H*]?\s*\d+\s+\d+\s')


def parse(log):
    """phase 1 (min cost) of a main.py log: first-design time, gap at the marks, final status, callback counts"""
    txt = open(log).read() if os.path.exists(log) else ''
    i = txt.find('Phase 1: minimize total_cost')
    j = txt.find('Phase 2', i + 1) if i >= 0 else -1
    ph1 = txt[i:j if j > 0 else None] if i >= 0 else ''
    first, pts = None, []
    for line in ph1.splitlines():
        if not NODE.match(line):
            continue
        tok = line.split()
        if not tok[-1].endswith('s'):
            continue
        try:
            t = float(tok[-1][:-1])
        except ValueError:
            continue
        gaps = [x for x in tok if x.endswith('%')]
        if gaps:
            pts.append((t, float(gaps[0][:-1])))
            first = t if first is None else first
    gap_at = {m: next((g for t, g in reversed(pts) if t <= m), None) for m in MARKS}
    fin = re.search(r'Best objective ([\d.e+-]+), best bound ([\d.e+-]+), gap ([\d.]+)%', ph1)
    runtime = re.search(r'Explored \d+ nodes .* in ([\d.]+) seconds', ph1)
    cb = re.search(r"after phase 1: \{'candidates': (\d+), 'rejected': (\d+)", txt)
    check = 'OK' if re.search(r'\[check\] .*: OK', txt) else ('errors' if '[check]' in txt else '-')
    return {'first_s': first, 'gap_at': gap_at, 'cost': float(fin.group(1)) if fin else None,
            'final_gap': float(fin.group(3)) if fin else None, 'runtime_s': float(runtime.group(1)) if runtime else None,
            'rejected': f"{cb.group(2)}/{cb.group(1)}" if cb else '-', 'check': check}


def report(tags):
    rows = ["tag        instance   first design  gap 15 min  gap 30 min  gap 60 min  final cost    final gap  runtime  rejected   check"]
    for tag in tags:
        for name, _a, _c, _r in INSTANCES:
            r = parse(os.path.join(OUT, tag, f'{name}.txt'))
            g = ['-' if r['gap_at'][m] is None else f"{r['gap_at'][m]:.2f} %" for m in MARKS]
            rows.append(f"{tag:10s} {name:10s} {('-' if r['first_s'] is None else f'{r_first(r)} s'):>12s}  {g[0]:>10s}  {g[1]:>10s}  {g[2]:>10s}  "
                        f"{('-' if r['cost'] is None else f'{r_cost(r)}'):>10s}  {('-' if r['final_gap'] is None else f'{r_gap(r)} %'):>9s}  "
                        f"{('-' if r['runtime_s'] is None else f'{r_run(r)} s'):>7s}  {r['rejected']:>9s}  {r['check']}")
    out = '\n'.join(rows)
    print(out)
    with open(os.path.join(OUT, 'report_' + '_'.join(tags) + '.txt'), 'w') as f:
        f.write(out + '\n')


def r_first(r):
    return f"{r['first_s']:.0f}"


def r_cost(r):
    return f"{r['cost']:.2f}"


def r_gap(r):
    return f"{r['final_gap']:.2f}"


def r_run(r):
    return f"{r['runtime_s']:.0f}"


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'submit':
        submit(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif len(sys.argv) >= 3 and sys.argv[1] == 'report':
        report(sys.argv[2:])
    else:
        print(__doc__)
