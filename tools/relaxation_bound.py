#!/usr/bin/env python3
"""
Placement-only relaxation bound of a run's instance (2026-11-04, HANDOFF 16.26): the exact model without its lazy
network rules (routing, capacity, latency, survival routes, bus length, packing), solved as a plain MILP with the
run's config; Gurobi's best bound is a lower bound on the cost of every design of the full model, next to the counting
bound of tools/design_bound.py. Written as <run_dir>/relaxation_bound.json.

  dse/bin/python tools/relaxation_bound.py <run_dir> [--time 7200] [--params "Threads=64"]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir'); ap.add_argument('--time', type=float, default=7200.0); ap.add_argument('--params', default=None)
    a = ap.parse_args()
    run = os.path.abspath(a.run_dir)
    info = json.load(open(os.path.join(run, 'run_info.json')))
    args = info['args']
    arch = os.path.join(run, info.get('architecture_copy') or 'architecture_used.json')
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
    gp = a.params or args.get('gurobi_params') or cr.get_solver_config().get('gurobi_params')
    design = None
    if os.path.exists(os.path.join(run, 'solution_1.json')):
        design = float(json.load(open(os.path.join(run, 'solution_1.json')))['total_cost'])
    print(f"[relaxation] {run}: {len(scs)} components, {len(locs)} cells, rho {rho}, design {design}, {a.time:.0f} s", flush=True)
    t0 = time.time()
    opt = AssignmentOptimizer()
    opt.memory = cr.get_solver_config().get('memory') or opt.memory
    res = opt.optimize(scs, locs, sens, acts, cables, comm, partitions=partitions, hardwares=cr.get_hardwares(), interfaces=cr.get_interfaces(),
                       uncertainty=unc, backbone=cr.get_backbone_config(), chains=cr.get_chains(), time_limit=a.time, mip_gap=1e-4,
                       gurobi_params=gp, lns=None, relax_network=True)
    out = {'run': run, 'design_cost': design, 'time_s': round(time.time() - t0, 1), 'rho': rho, **res}
    if design and res.get('bound'):
        out['gap_to_design'] = (design - res['bound']) / res['bound']
    json.dump(out, open(os.path.join(run, 'relaxation_bound.json'), 'w'), indent=1)
    print(f"[relaxation] bound {res.get('bound')}, relaxation optimum/incumbent {res.get('objective')} (gap {res.get('mip_gap')}), "
          f"partitions {res.get('n_partitions')}, units {res.get('units')}, hardware {res.get('hardware')}; design {design}"
          + (f", design gap to the bound {100 * out['gap_to_design']:.1f} %" if 'gap_to_design' in out else ''), flush=True)


if __name__ == '__main__':
    main()
