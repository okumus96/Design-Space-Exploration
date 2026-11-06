#!/usr/bin/env python3
"""
Polish lab (2026-11-04, HANDOFF 16.26): free the components of chosen cells of a finished design together and re-solve
with everything else fixed, to see whether the cells' partitions can be emptied (the cell polish frees one cell at a
time for polish_time_s; here several cells at once and a longer time).

  dse/bin/python tools/polish_cells_lab.py <run_dir> <out_dir> --cells KICK_R,COWL_R [--time 3600] [--params "..."]

run_dir: a finished run (run_info.json, architecture_used.json, solution_1.json); the design is rebuilt from
solution_1.json (every primary and standby cell fixed), then the components whose primary or standby sits on one of
the cells are freed. Written into <out_dir>: solution_1.json of the re-solve (when it found a design), a report, and
the checker's verdict.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from config_reader import ConfigReader                      # noqa: E402
from architecture_loader import load_architecture_from_json  # noqa: E402
from optimizer import AssignmentOptimizer, UNCERTAINTY_BLOCKS  # noqa: E402
import pieces                                                # noqa: E402


def partitions_of(sol):
    out = {}
    for k, n in sol['partition_counts'].items():
        cell, asil = k.rsplit('_asil', 1)
        out.setdefault(cell, {})[int(asil)] = n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir'); ap.add_argument('out_dir')
    ap.add_argument('--cells', required=True, help='comma-separated cell ids to free together')
    ap.add_argument('--time', type=float, default=3600.0)
    ap.add_argument('--params', default=None, help='Gurobi parameters (default: the run\'s)')
    a = ap.parse_args()
    run, out = os.path.abspath(a.run_dir), os.path.abspath(a.out_dir)
    os.makedirs(out, exist_ok=True)
    info = json.load(open(os.path.join(run, 'run_info.json')))
    args = info['args']
    arch = os.path.join(run, info.get('architecture_copy') or 'architecture_used.json')
    shutil.copy(arch, os.path.join(out, 'architecture_used.json'))
    sol0 = json.load(open(os.path.join(run, 'solution_1.json')))
    cr = ConfigReader(args['config_dir'])
    scs, comm, sens, acts, cables, locs = load_architecture_from_json(arch, cr, num_locs=args.get('num_locs'))
    partitions = cr.get_partitions()
    partitions['hw_capacity'] = cr.get_hardware_capacities()
    unc = {k: bool(args.get('uncertainty') or args.get(k)) for k in UNCERTAINTY_BLOCKS}
    unc['params'] = cr.get_uncertainty_config()
    rho = float(unc['params'].get('rho', {}).get('levels', [1.0])[0])
    unc['rho'] = unc['growth_alpha'] = rho
    cfg = {**pieces.DEFAULTS, **(cr.get_solver_config().get('pieces') or {})}
    gp = a.params or args.get('gurobi_params') or cr.get_solver_config().get('gurobi_params')
    common = dict(partitions=partitions, hardwares=cr.get_hardwares(), interfaces=cr.get_interfaces(), uncertainty=unc,
                  backbone=cr.get_backbone_config(), chains=cr.get_chains(),
                  gurobi_params=','.join(x for x in (gp, cfg.get('step_gurobi_params')) if x))
    placed_x = dict(sol0['assignment'])
    placed_r = {s: v['cell'] for s, v in (sol0.get('standby') or {}).items()}
    cells = [c.strip() for c in a.cells.split(',') if c.strip()]
    free = {s for s, c in placed_x.items() if c in cells} | {s for s, c in placed_r.items() if c in cells}
    print(f"[lab] {run}: design {sol0['total_cost']:.2f}, partitions {partitions_of(sol0)}", flush=True)
    print(f"[lab] freeing {len(free)} components of {cells} ({sum(1 for s in free if placed_x[s] in cells)} primaries, "
          f"{sum(1 for s in free if placed_r.get(s) in cells)} standbys), {a.time:.0f} s, gateway {common['backbone'].get('bus_gateway')}", flush=True)
    t0 = time.time()
    opt = AssignmentOptimizer()
    opt.memory = cr.get_solver_config().get('memory') or opt.memory
    sol = opt.optimize(scs, locs, sens, acts, cables, comm, time_limit=float(cfg['piece_time_s']), mip_gap=float(cfg['piece_gap']),
                       phase2_time=float(cfg['piece_phase2_s']), phase2_cap=float(cfg['piece_phase2_s']),
                       fixed={'x': placed_x, 'r': placed_r}, lns=None, reuse=True, **common)
    if sol is None:
        sys.exit('[lab] the design could not be rebuilt')
    print(f"[lab] rebuilt design: cost {sol['total_cost']:.2f} ({time.time() - t0:.0f} s)", flush=True)
    fixed = {'x': {s: c for s, c in placed_x.items() if s not in free}, 'r': {s: c for s, c in placed_r.items() if s not in free}}
    t1 = time.time()
    res = opt.optimize(scs, locs, sens, acts, cables, comm, time_limit=a.time, mip_gap=float(cfg['piece_gap']),
                       phase2_time=float(cfg['piece_phase2_s']), phase2_cap=float(cfg['piece_phase2_s']),
                       start=opt.last_values, fixed=fixed, lns=None, reuse=True, **common)
    lines = [f"design {sol['total_cost']:.2f}, partitions {partitions_of(sol)}",
             f"freed {len(free)} components of {cells}, {a.time:.0f} s"]
    if res is None:
        lines.append(f"no design in {time.time() - t1:.0f} s")
    else:
        ph = res['solve_info']['phases'][0]
        lines.append(f"re-solved: cost {res['total_cost']:.2f} (gap {100 * (ph.get('mip_gap') or 0):.2f} %, {time.time() - t1:.0f} s), "
                     f"partitions {partitions_of(res)}, units {sorted(set(res['assignment'].values()))}")
        moved = {s: (placed_x[s], c) for s, c in res['assignment'].items() if placed_x.get(s) != c}
        lines.append(f"moved {len(moved)} components: " + ", ".join(f"{s} {a_}->{b_}" for s, (a_, b_) in list(moved.items())[:12]) + (" ..." if len(moved) > 12 else ""))
        json.dump(res, open(os.path.join(out, 'solution_1.json'), 'w'), indent=2, default=str)
        json.dump({'args': {**args, 'output_dir': out}, 'architecture_copy': 'architecture_used.json', 'lab_of': run, 'cells': cells,
                   'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}, open(os.path.join(out, 'run_info.json'), 'w'), indent=2)
    open(os.path.join(out, 'lab_report.txt'), 'w').write("\n".join(lines) + "\n")
    print("\n".join(f"[lab] {l}" for l in lines), flush=True)
    if res is not None:
        subprocess.run([sys.executable, os.path.join(REPO, 'tools', 'check_solution.py'), os.path.join(out, 'solution_1.json')], check=False)


if __name__ == '__main__':
    main()
