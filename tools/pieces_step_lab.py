#!/usr/bin/env python3
"""
Reproduce one step of a group-by-group build on its own (HANDOFF 16.26): the partial design dumped after step k-1
(pieces.dump_dir: pieces_step_<k-1>.json) is loaded and step k is solved again with chosen variations, so a failing
step can be examined without re-running the build.

  dse/bin/python tools/pieces_step_lab.py <run_dir> <k> [--time S] [--free-standby] [--unlock N] [--params "..."]

run_dir: the run's output directory (pieces_step_*.json inside) with its config in <run_dir>_cfg; the architecture
and the number of cells come from run_info.json. --free-standby leaves the standby cells of the locked components
free (only their primary cells are locked); --unlock N frees the N previous groups as well.
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from config_reader import ConfigReader                      # noqa: E402
from architecture_loader import load_architecture_from_json  # noqa: E402
from optimizer import AssignmentOptimizer, UNCERTAINTY_BLOCKS  # noqa: E402
from pieces import sub_architecture                          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir'); ap.add_argument('k', type=int)
    ap.add_argument('--time', type=float, default=600.0); ap.add_argument('--free-standby', action='store_true')
    ap.add_argument('--unlock', type=int, default=0); ap.add_argument('--params', default='NoRelHeurTime=120')
    ap.add_argument('--no-static-capacity', action='store_true', help='switch the per-step cable-capacity rules off')
    ap.add_argument('--resume', type=float, default=0.0, help='after a no-design result: solve the kept model on for S more seconds (optimize resume=True)')
    a = ap.parse_args()
    run = os.path.abspath(a.run_dir); cfg_dir = run + '_cfg'
    info = json.load(open(os.path.join(run, 'run_info.json')))['args'] if os.path.exists(os.path.join(run, 'run_info.json')) else {}
    arch = info.get('architecture_json'); cells = info.get('num_locs')
    dump = json.load(open(os.path.join(run, f'pieces_step_{a.k - 1}.json')))
    cr = ConfigReader(cfg_dir)
    scs, comm, sens, acts, cables, locs = load_architecture_from_json(os.path.join(REPO, arch), cr, num_locs=cells)
    pieces = dump['pieces']
    k0 = max(0, a.k - 1 - a.unlock)
    keep = [s for p in pieces[:a.k] for s in p]
    locked = {s for p in pieces[:k0] for s in p}
    fixed = {'x': {s: c for s, c in dump['placed_x'].items() if s in locked},
             'r': {} if a.free_standby else {s: c for s, c in dump['placed_r'].items() if s in locked}}
    sub, comm_s, sens_s, acts_s, chs = sub_architecture(scs, comm, sens, acts, cr.get_chains(), keep)
    unc = {k: True for k in UNCERTAINTY_BLOCKS}; unc['params'] = cr.get_uncertainty_config()
    rho = float(unc['params'].get('rho', {}).get('levels', [1.0])[0]); unc['rho'] = rho; unc['growth_alpha'] = rho
    part = cr.get_partitions(); part['hw_capacity'] = cr.get_hardware_capacities()
    backbone = cr.get_backbone_config()
    if not a.no_static_capacity:
        backbone = {**backbone, 'upfront_paths': True, 'cell_cut_capacity': True}
    print(f"[lab] step {a.k}: {len(keep) - len(locked)} free ({len(pieces[a.k - 1])} new, {a.unlock} groups unlocked), {len(locked)} locked, "
          f"standbys {'free' if a.free_standby else 'locked'}, {len(chs)} chains, {len(sens_s) + len(acts_s)} devices, {a.time:.0f} s, {a.params}", flush=True)
    t0 = time.time()
    opt = AssignmentOptimizer()
    res = opt.optimize(sub, locs, sens_s, acts_s, cables, comm_s, partitions=part, hardwares=cr.get_hardwares(), interfaces=cr.get_interfaces(),
                       time_limit=a.time, mip_gap=0.01, uncertainty=unc, backbone=backbone, phase2_time=5, gurobi_params=a.params,
                       chains=chs, fixed=fixed, lns=None, reuse=True)
    if res is None:
        print(f"[lab] no design in {time.time() - t0:.0f} s", flush=True)
    if res is None and a.resume > 0:
        t1 = time.time()
        print(f"[lab] solving on for {a.resume:.0f} s more (model kept, resumable {getattr(opt, '_resumable', None)})", flush=True)
        res = opt.optimize(sub, locs, sens_s, acts_s, cables, comm_s, partitions=part, hardwares=cr.get_hardwares(), interfaces=cr.get_interfaces(),
                           time_limit=a.resume, mip_gap=0.01, uncertainty=unc, backbone=backbone, phase2_time=5, gurobi_params=a.params,
                           chains=chs, fixed=fixed, lns=None, reuse=True, resume=True)
        print(f"[lab] after solving on: {'no design' if res is None else 'design found'} in {time.time() - t1:.0f} s more", flush=True)
    if res is not None:
        ph = res['solve_info']['phases'][0]
        print(f"[lab] design: cost {res['total_cost']:.2f}, gap {100 * (ph.get('mip_gap') or 0):.2f} %, cells {len(set(res['assignment'].values()))}, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == '__main__':
    main()
