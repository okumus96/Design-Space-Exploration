#!/usr/bin/env python3
"""
Lower bound and optimality gap of a stored design (HANDOFF 16.26, 2026-10-30).

  dse/bin/python tools/design_bound.py results/<run>/solution_1.json [--config_dir configs] [--architecture <json>]
  dse/bin/python tools/design_bound.py results/<run>                 (every solution_<i>.json of the run)

The group-by-group build (pieces.py) reports no bound. This tool bounds the cost of the best design of the full model
from below with a relaxation that keeps only the counting rules of the model, on the checker's own demands (margins,
growth, derating, standbys; tools/check_solution.py, nothing imported from the optimiser):
  partitions  per ASIL level: the nominal load (every primary, plus the standby of every critical component whose
              FTTI is below the recovery time, which must be hot) over the partition caps, rounded up per resource;
              at least 2 where a critical component exists (its standby sits on another cell)
  units       the partitions over max_partitions per unit; the unit caps; at least 2 with a critical or redundant
              component; the partitions of one ASIL over max_per_asil_per_site
  hardware    per feature: the TOPS demand over TOPS per unit x the best derating; at least 1 unit where required,
              2 where a critical component requires it (the standby cell needs it too)
  ports       one port per device of a point-to-point interface (ETH: the cheaper of a port and a 16th of a switch),
              one port per shared-bus type with devices
  cables, bus lines, backbone routes, switch nodes: 0
Dropping rules can only lower the optimum, so the bound is valid; gap = (cost - bound) / cost as Gurobi defines it:
the design is at most that far above the best design of the model.
"""
import argparse
import glob
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_solution import SolutionCheck, RES, SHARED_BUSES, SWITCH_ETH_PORTS, REPO  # noqa: E402


def _cheapest_mix(types, load, h, n_min):
    """cheapest block counts {type: n} with sum n x cap >= load per resource (CPU and TOPS derated by h) and at least
    n_min blocks: a small integer program (scipy HiGHS); returns (mix, cost)"""
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    ids = sorted(types)
    res = [r for r in load if load[r] > 0]
    A = [[types[t][r] * (h if r in ('cpu', 'tops') else 1.0) for t in ids] for r in res] + [[1.0] * len(ids)]
    lb = [load[r] for r in res] + [float(n_min)]
    sol = milp(c=np.array([types[t]['cost'] for t in ids]), constraints=LinearConstraint(np.array(A), lb, np.inf),
               integrality=np.ones(len(ids)), bounds=Bounds(0, np.inf))
    if sol.x is None:
        return {}, float('inf')
    mix = {t: int(round(v)) for t, v in zip(ids, sol.x) if round(v) > 0}
    return mix, float(sum(types[t]['cost'] * n for t, n in mix.items()))


def _ceil(x):
    """round up, within the solver's feasibility tolerance"""
    return int(math.ceil(x - 1e-6 * max(1.0, abs(x))))


def design_bound(c):
    """lower bound of the total cost; c = SolutionCheck after load() and parse()"""
    h = max(c.derate(s) for s in c.cands)                       # the best (least derated) candidate cell
    ftti = {}
    for ch in c.chains:
        for s in ch.get('swcs', []):
            ftti[s] = min(ftti.get(s, float('inf')), float(ch.get('ftti_ms', float('inf'))))
    must_hot = {s for s in c.crit if ftti.get(s, float('inf')) < c.recovery}
    nominal = list(c.scs) + sorted(must_hot)                    # the instances with a nominal load
    out = {'h': h, 'critical': len(c.crit), 'must_hot': len(must_hot)}
    # partitions per ASIL level
    parts, rows, part_cost = {}, [], 0.0
    for a in sorted({sc['asil'] for sc in c.scs.values()}):
        inst = [s for s in nominal if c.scs[s]['asil'] == a]
        n, by = 1, {}
        n_min = 2 if any(s in c.crit for s in inst) else 1
        if c.types:
            # partition types (2026-11-04): the cheapest mix of blocks whose summed capacities (CPU and TOPS at the best
            # derating) cover the level's load, at least n_min blocks; the minimum block count separately (unit bound)
            load = {r: sum(c.demand(s, r) for s in inst) for r in c.res}
            mix, cost_a = _cheapest_mix(c.types, load, h, n_min)
            for r in c.res:
                top = max(t[r] for t in c.types.values()) * (h if r in ('cpu', 'tops') else 1.0)
                if load[r] > 0:
                    by[r] = load[r] / top
                    n = max(n, _ceil(by[r]))
            n = max(n, n_min)
            part_cost += cost_a
            parts[a] = n
            rows.append({'asil': a, 'components': sum(1 for s in c.scs.values() if s['asil'] == a),
                         'critical': sum(1 for s in c.crit if c.scs[s]['asil'] == a),
                         'must_hot': sum(1 for s in must_hot if c.scs[s]['asil'] == a), 'load': by, 'bound': n,
                         'mix': mix, 'mix_cost': cost_a})
            continue
        for r in RES:
            cap = float(c.part.get(f'{r}_cap', 0) or 0) * (h if r == 'cpu' else 1.0)
            if cap > 0:
                by[r] = sum(c.demand(s, r) for s in inst) / cap
                n = max(n, _ceil(by[r]))
        n = max(n, n_min)
        parts[a] = n
        part_cost += n * float(c.part.get('cost', 0) or 0)
        rows.append({'asil': a, 'components': sum(1 for s in c.scs.values() if s['asil'] == a),
                     'critical': sum(1 for s in c.crit if c.scs[s]['asil'] == a),
                     'must_hot': sum(1 for s in must_hot if c.scs[s]['asil'] == a), 'load': by, 'bound': n})
    out['partitions'], out['partition_rows'] = sum(parts.values()), rows
    # control units
    units = 1
    site_max = c.site_cfg.get('max_partitions')
    if site_max:
        units = max(units, _ceil(out['partitions'] / float(site_max)))
    for r in RES:
        cap = c.site_cfg.get(f'{r}_cap')
        if cap is not None:
            units = max(units, _ceil(sum(c.demand(s, r) for s in nominal) / (float(cap) * (h if r == 'cpu' else 1.0))))
    if c.crit or (c.flags.get('redundancy') and any(sc['redundant_with'] in c.scs for sc in c.scs.values())):
        units = max(units, 2)
    max_pa = int(c.part.get('max_per_asil_per_site', 3) or 0)   # null (no limit per ASIL) = 0: no unit bound from it
    if max_pa > 0:
        units = max(units, max(_ceil(n / float(max_pa)) for n in parts.values()))
    out['units'] = units
    # hardware features: at every cell of a primary or standby that needs them, TOPS within units x TOPS x derating
    hw = {}
    for f, price in c.hw_price.items():
        users = [s for s in c.scs if f in c.scs[s]['hw']]
        if not users:
            continue
        n = 2 if any(s in c.crit for s in users) else 1
        spec = c.hw_cap.get(f)
        if spec and spec.get('types'):   # GPU types (2026-11-05): the cheapest module mix covering the TOPS demand
            gtypes = {str(t['id']): {'tops': float(t['tops']), 'cost': float(t.get('cost') or 0)} for t in spec['types']}
            load = sum(c.demand(s, 'tops') for s in nominal if f in c.scs[s]['hw'])
            mix, mix_cost = _cheapest_mix(gtypes, {'tops': load}, h, n)
            units = sum(mix.values())
            hw[f] = (units, mix_cost / units if units else 0.0)   # (modules, mean price): n x p = the mix cost
            continue
        if spec:
            n = max(n, _ceil(sum(c.demand(s, 'tops') for s in nominal if f in c.scs[s]['hw']) / (float(spec['tops']) * h)))
        hw[f] = (n, price)
    out['hardware'] = hw
    # device ports: the devices driven by a component of the architecture
    by_if = defaultdict(int)
    for d in {d for sc in c.scs.values() for d in sc['devices']}:
        by_if[c.devices[d]['iface']] += 1
    ports = {}
    for t, k in sorted(by_if.items()):
        port = float(c.ct.get(t, {}).get('port', 0.0))
        if t in SHARED_BUSES:
            ports[t] = (1, port, port)
        elif t == 'ETH' and 'SWITCH' in c.hw_price:
            each = min(port, c.hw_price['SWITCH'] / SWITCH_ETH_PORTS)
            ports[t] = (k, each, k * each)
        else:
            ports[t] = (k, port, k * port)
    out['ports'] = ports
    out['cost'] = {'partitions': part_cost,
                   'units': units * float(c.site_cfg.get('fixed_cost', 0) or 0),
                   'hardware': sum(n * p for n, p in hw.values()),
                   'ports': sum(v[2] for v in ports.values()),
                   'network': 0.0}
    out['bound'] = sum(out['cost'].values())
    return out


def report(c, b):
    s = c.sol
    cost = float(s['total_cost'])
    gap = (cost - b['bound']) / cost if cost else float('inf')
    rel = os.path.relpath(c.path, REPO) if c.path.startswith(REPO) else c.path
    lines = [f"[bound] {rel}: cost {cost:.2f}, bound {b['bound']:.2f}, gap {100 * gap:.2f} % "
             f"(rho {c.rho:g}, {len(c.scs)} components, {len(c.cands)} cells, {b['critical']} critical of which "
             f"{b['must_hot']} must be hot, best derating {b['h']:g})"]
    design = {'partitions': (int(s.get('num_partitions', 0)), float(s.get('partition_cost', 0) or 0)),
              'units': (int(s.get('num_locations_used', 0)), float(s.get('site_cost', 0) or 0)),
              'hardware': (sum(c.hw_units.values()), float(s.get('hardware_cost', 0) or 0)),
              'ports': (sum(c.ports.values()), float(s.get('interface_cost', 0) or 0)),
              'network': (len(s.get('comm_links', [])), float(s.get('cable_cost', 0) or 0) + float(s.get('comm_cost', 0) or 0))}
    bound_n = {'partitions': b['partitions'], 'units': b['units'], 'hardware': sum(n for n, _ in b['hardware'].values()),
               'ports': sum(v[0] for v in b['ports'].values()), 'network': 0}
    lines.append(f"  {'part':<12}{'design':>8}{'cost':>12}{'bound':>8}{'cost':>12}{'slack':>10}")
    for k in ('partitions', 'units', 'hardware', 'ports', 'network'):
        n, v = design[k]
        lines.append(f"  {k:<12}{n:>8}{v:>12.2f}{bound_n[k]:>8}{b['cost'][k]:>12.2f}{v - b['cost'][k]:>10.2f}")
    lines.append(f"  {'total':<12}{'':>8}{cost:>12.2f}{'':>8}{b['bound']:>12.2f}{cost - b['bound']:>10.2f}")
    counts = defaultdict(int)
    for key, n in (s.get('partition_counts') or {}).items():
        m = re.match(r'^(?P<site>.+)_asil(?P<asil>\d+)$', key)
        counts[int(m['asil'])] += int(n)
    lines.append("  partitions per ASIL (bound = nominal load over the partition caps, rounded up):")
    lines.append(f"  {'ASIL':<6}{'comps':>7}{'crit':>6}{'hot':>5}{'cpu':>8}{'ram':>8}{'rom':>8}{'bound':>7}{'design':>8}")
    for r in b['partition_rows']:
        ld = r['load']
        lines.append(f"  {r['asil']:<6}{r['components']:>7}{r['critical']:>6}{r['must_hot']:>5}"
                     f"{ld.get('cpu', 0):>8.2f}{ld.get('ram', 0):>8.2f}{ld.get('rom', 0):>8.2f}{r['bound']:>7}{counts.get(r['asil'], 0):>8}"
                     + (f"   cheapest blocks {r['mix']} = {r['mix_cost']:.2f}" if 'mix' in r else ""))
    hw = ", ".join(f"{f} {n} x {p:g}" for f, (n, p) in b['hardware'].items()) or "none"
    ports = ", ".join(f"{t} {k} x {each:g}" for t, (k, each, _) in b['ports'].items()) or "none"
    lines.append(f"  hardware bound: {hw}; port bound: {ports}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='solution_<i>.json or a run directory')
    ap.add_argument('--config_dir', default=None, help='default: the run\'s config dir (run_info.json), else configs')
    ap.add_argument('--architecture', default=None, help='default: the run\'s architecture JSON (run_info.json)')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.path, 'solution_*.json')), key=lambda p: int(re.findall(r'(\d+)\.json$', p)[0])) \
        if os.path.isdir(a.path) else [a.path]
    for f in files:
        c = SolutionCheck(f, a.config_dir, a.architecture)
        c.load()
        c.parse()
        print(report(c, design_bound(c)), flush=True)
    return 0 if files else 1


if __name__ == '__main__':
    sys.exit(main())
