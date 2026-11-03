#!/usr/bin/env python3
"""
Polish an existing group-by-group design again (HANDOFF 16.26): the placements of a finished build (the dump
pieces_polish.json of a run with pieces.dump_dir = its best polished design, when the run has one, else its last
pieces_step_<k>.json; --step K = that step's dump) are rebuilt into a full design with one short solve (every
primary and standby cell fixed), then the polish of pieces.py (pieces.polish_mode: cell, level or group) runs on it with
the run's own instance, config and Gurobi parameters, and, with pieces.final_network_s > 0, the final network clean-up
(pieces._final_network); the settings come from the run's solver.json (pieces), missing keys from pieces.DEFAULTS.
Written into <out_dir>: solution_1.json, run_info.json, architecture_used.json, pieces_report.txt, polish logs; the
checker is run at the end.

  dse/bin/python tools/pieces_polish.py <run_dir> <out_dir> [--step K]

run_dir: the run's output directory (run_info.json, architecture_used.json, pieces_step_*.json); default step: the last.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from config_reader import ConfigReader                      # noqa: E402
from architecture_loader import load_architecture_from_json  # noqa: E402
from optimizer import AssignmentOptimizer, UNCERTAINTY_BLOCKS  # noqa: E402
import pieces                                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--step', type=int, default=None)
    a = ap.parse_args()
    run, out = os.path.abspath(a.run_dir), os.path.abspath(a.out_dir)
    os.makedirs(out, exist_ok=True)
    info = json.load(open(os.path.join(run, 'run_info.json')))
    args = info['args']
    arch = os.path.join(run, info.get('architecture_copy') or 'architecture_used.json')
    shutil.copy(arch, os.path.join(out, 'architecture_used.json'))
    steps = sorted(glob.glob(os.path.join(run, 'pieces_step_*.json')), key=lambda f: int(re.search(r'_(\d+)\.json$', f).group(1)))
    polished = os.path.join(run, 'pieces_polish.json')
    dump_file = os.path.join(run, f'pieces_step_{a.step}.json') if a.step else (polished if os.path.exists(polished) else steps[-1])
    dump = json.load(open(dump_file))

    cr = ConfigReader(args['config_dir'])
    scs, comm, sens, acts, cables, locs = load_architecture_from_json(arch, cr, num_locs=args.get('num_locs'))
    partitions = cr.get_partitions()
    partitions['hw_capacity'] = cr.get_hardware_capacities()
    unc = {k: bool(args.get('uncertainty') or args.get(k)) for k in UNCERTAINTY_BLOCKS}
    unc['params'] = cr.get_uncertainty_config()
    rho = float(unc['params'].get('rho', {}).get('levels', [1.0])[0])
    unc['rho'] = unc['growth_alpha'] = rho
    if rho <= 0.0:
        unc['failures'] = unc['redundancy'] = False
    cfg = {**pieces.DEFAULTS, **(cr.get_solver_config().get('pieces') or {}), 'dump_dir': out}
    backbone = cr.get_backbone_config()
    if cfg.get('static_capacity', False):
        backbone = {**backbone, 'upfront_paths': True, 'cell_cut_capacity': True}
    gp = args.get('gurobi_params') or cr.get_solver_config().get('gurobi_params')
    common = dict(partitions=partitions, hardwares=cr.get_hardwares(), interfaces=cr.get_interfaces(), uncertainty=unc,
                  backbone=backbone, chains=cr.get_chains())

    print(f"[polish] {run}: design of {os.path.basename(dump_file)} ({len(dump['placed_x'])} components, cost {dump['cost']:.2f}), "
          f"polish {cfg['polish_rounds']} rounds by {cfg.get('polish_mode', 'cell')}, {cfg.get('polish_parallel', 1)} at a time, "
          f"{cfg['polish_time_s']} s", flush=True)
    t0 = time.time()
    opt = AssignmentOptimizer()
    opt.memory = cr.get_solver_config().get('memory') or opt.memory
    # the design itself: every primary and standby cell fixed, the rest (partitions, hardware, cables) re-decided;
    # reuse=True: this solve translates the whole-car model once, the polish rounds and the clean-up solve it again
    sol = opt.optimize(scs, locs, sens, acts, cables, comm, time_limit=float(cfg['piece_time_s']), mip_gap=float(cfg['piece_gap']),
                       phase2_time=float(cfg['piece_phase2_s']), phase2_cap=float(cfg['piece_phase2_s']),
                       gurobi_params=','.join(x for x in (gp, cfg.get('step_gurobi_params')) if x),
                       fixed={'x': dump['placed_x'], 'r': dump['placed_r']}, lns=None, reuse=True, **common)
    if sol is None:
        sys.exit('[polish] the dumped design could not be rebuilt')
    start_cost, rebuild_s = float(sol['total_cost']), time.time() - t0
    print(f"[polish] rebuilt design: cost {start_cost:.2f} ({rebuild_s:.0f} s)", flush=True)
    t1 = time.time()
    sol, rounds = pieces.polish_fn(cfg)(opt, scs, locs, sens, acts, cables, comm, gurobi_params=gp, cfg=cfg, pieces=dump['pieces'],
                                        steps=[], sol=sol, traffic={}, **common)
    polish_s = time.time() - t1
    final = None
    if float(cfg.get('final_network_s', 0) or 0) > 0:
        sol, final = pieces._final_network(opt, scs, locs, sens, acts, cables, comm, gurobi_params=gp, cfg=cfg, sol=sol, **common)
    sol['solve_info']['pieces'] = {'polish_only': True, 'from': dump_file, 'start_cost': start_cost, 'rebuild_s': round(rebuild_s, 1),
                                   'polish_s': round(polish_s, 1), 'polish': rounds, 'final_network': final,
                                   'settings': {k: v for k, v in cfg.items() if not k.startswith('_')}}
    lines = [f"polish only: {dump_file}", f"rebuilt design {start_cost:.2f} in {rebuild_s:.0f} s"]
    lines += [f"polish {r['round']} (pass {r.get('pass', 1)}, batch {r.get('batch', '-')}): {r.get('kind', 'cell')} {r['group']} ({r['free']} free): {r['before']:.1f} -> {r['after']:.1f} in {r['seconds']:.0f} s ({r['result']})" for r in rounds]
    if final:
        lines.append(f"final network: {final['before']:.1f} -> {final['after']:.1f} in {final['seconds']:.0f} s ({final['result']}; "
                     f"{final['status']}, gap {100 * (final['gap'] or 0):.3f} %), switches at {final['switches']}")
    lines.append(f"final cost {float(sol['total_cost']):.2f}; polish {polish_s / 60:.0f} min")
    open(os.path.join(out, 'pieces_report.txt'), 'w').write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    json.dump(sol, open(os.path.join(out, 'solution_1.json'), 'w'), indent=2, default=str)
    json.dump({'args': {**args, 'output_dir': out}, 'architecture_copy': 'architecture_used.json', 'polish_of': run,
               'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}, open(os.path.join(out, 'run_info.json'), 'w'), indent=2)
    subprocess.run([sys.executable, os.path.join(REPO, 'tools', 'check_solution.py'), os.path.join(out, 'solution_1.json')], check=False)


if __name__ == '__main__':
    main()
