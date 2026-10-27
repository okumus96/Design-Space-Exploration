#!/usr/bin/env python3
"""
Gate test of the learned search (HANDOFF 16.25): the 65-component architecture on more candidate cells than the
training variants have, at rho 1 (failures on), the exact model with the default settings and with MIPFocus=1 against
the learned search, same time limit, every design checked. Settings: configs/solver.json, learn.gate.

  dse/bin/python learn/gate.py run <out_dir> <methods> [<policy.pt>]   in a Slurm job with 8 cores per run; methods:
                                                                      comma list of learn.gate.methods
  dse/bin/python learn/gate.py report <out_dir>                            best cost at the time marks, checker verdict
"""
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, 'dse', 'bin', 'python')
sys.path.insert(0, REPO)
from learn.gen_data import learn_cfg      # noqa: E402
from learn.instances import CFG_FILES     # noqa: E402

NODE = re.compile(r'^\s*[H*]?\s*\d+\s+\d+\s')
LNS = re.compile(r'^\[LNS\] iteration \d+: improved to ([\d.]+) at \d+ s \(total (\d+) s\)')


def run(out, methods, policy=None):
    """all (method, seed, cell count) runs of the gate at once in this job, each main.py run in its own process with
    the method's Gurobi parameters (learn.gate.methods, the same thread count for every method); a method of
    learn.gate.search_modes runs the guided search in that mode with the settings of learn.gate.search"""
    import concurrent.futures
    g = learn_cfg().get('gate', {})
    out = os.path.abspath(out)
    runs = []
    for method, seed in ((m, s) for m in methods for s in g.get('seeds', [None])):
        params = g['methods'][method] + (f",Seed={seed}" if seed is not None else '')
        mode = g.get('search_modes', {}).get(method)
        for cells in g.get('cells', [20, 33]):
            name = f"{method}_c{cells}" if seed is None else f"{method}_s{seed}_c{cells}"
            cfg = os.path.join(out, 'cfg_' + name)
            shutil.rmtree(cfg, ignore_errors=True)
            os.makedirs(cfg)
            for f in CFG_FILES:
                shutil.copy(os.path.join(REPO, 'configs', f), os.path.join(cfg, f))
            with open(os.path.join(cfg, 'uncertainty.json')) as f:
                u = json.load(f)
            u['rho']['levels'] = [float(g.get('rho', 1.0))]
            with open(os.path.join(cfg, 'uncertainty.json'), 'w') as f:
                json.dump(u, f, indent=2)
            if mode:
                with open(os.path.join(cfg, 'solver.json')) as f:
                    s = json.load(f)
                s['learn']['search'].update(g.get('search', {}), enabled=True, mode=mode)
                if seed is not None:
                    s['learn']['search']['seed'] = seed
                if policy:
                    s['learn']['search']['model'] = os.path.abspath(policy)
                with open(os.path.join(cfg, 'solver.json'), 'w') as f:
                    json.dump(s, f, indent=2)
            runs.append((name, [PY, os.path.join(REPO, 'main.py'), '--architecture_json', 'configs/full_architecture_v2.json',
                                '--num_locs', str(cells), '--uncertainty', '--time_limit', str(int(g.get('time_limit', 1800))),
                                '--phase2_time', str(int(g.get('phase2_time', 60))), '--config_dir', cfg,
                                '--output_dir', os.path.join(out, name), '--gurobi_params', params]))

    def one(r):
        with open(os.path.join(out, r[0] + '.txt'), 'w') as log:
            return r[0], subprocess.run(r[1], cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(runs)) as ex:
        for name, rc in ex.map(one, runs):
            print(f"{name}: exit {rc}", flush=True)


def curve(log):
    """(seconds since the start of phase 1, best cost) points: branch-and-bound lines of phase 1 and the improvements
    of the learned / LNS loop"""
    txt = open(log).read() if os.path.exists(log) else ''
    i = txt.find('Phase 1: minimize total_cost')
    j = txt.find('[Lexicographic] Phase 2', i + 1)
    ph = txt[i:j if j > 0 else None] if i >= 0 else ''
    pts, best = [], float('inf')
    for line in ph.splitlines():
        m = LNS.match(line)
        if m:
            best = min(best, float(m.group(1)))
            pts.append((float(m.group(2)), best))
            continue
        if NODE.match(line):
            tok = line.split()
            if tok[-1].endswith('s') and any(x.endswith('%') for x in tok):
                try:   # ... Incumbent BestBd Gap It/Node Time: the incumbent stands two columns before the gap
                    t = float(tok[-1][:-1])
                    inc = float(tok[next(k for k, x in enumerate(tok) if x.endswith('%')) - 2])
                except (ValueError, IndexError):
                    continue
                if not pts or t >= pts[-1][0]:   # phase-1 node lines (the LNS sub-solves restart Gurobi's clock)
                    best = min(best, inc)
                    pts.append((t, best))
    check = 'OK' if re.search(r'\[check\] .*: OK', txt) else ('errors' if '[check]' in txt else '-')
    final = re.search(r'TOTAL COST\s+\$([\d.]+)', txt)
    return pts, check, (float(final.group(1)) if final else None)


def report(out):
    g = learn_cfg().get('gate', {})
    marks = g.get('marks', [300, 600, 1200, 1800])
    rows = ['method          cells    first' + ''.join(f'{m:>10d}s' for m in marks) + '      final  check']
    for name in sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)) and not d.startswith('cfg_')):
        pts, check, final = curve(os.path.join(out, name + '.txt'))
        method, cells = name.rsplit('_c', 1)
        vals = [next((c for t, c in reversed(pts) if t <= m), None) for m in marks]
        first = f"{pts[0][0]:.0f}s" if pts else '-'   # seconds to the first design
        rows.append(f"{method:15s} {cells:>5s} {first:>8s}" + ''.join(f"{v:>11.1f}" if v is not None else f"{'-':>11s}" for v in vals)
                    + f" {final if final is not None else '-':>10} {check:>6s}")
    print('\n'.join(rows))
    with open(os.path.join(out, 'report.txt'), 'w') as f:
        f.write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'run' and len(sys.argv) in (4, 5):
        run(sys.argv[2], sys.argv[3].split(','), sys.argv[4] if len(sys.argv) == 5 else None)
    elif cmd == 'report' and len(sys.argv) == 3:
        report(sys.argv[2])
    else:
        print(__doc__)
