#!/usr/bin/env python3
"""
Regression test of the optimiser on the small instance with known proven optima.

  srun -p interactive -c 8 --mem=16G -t 01:30:00 dse/bin/python tools/regression_test.py

Solves fixed cases of configs/full_architecture_v2_lite.json (50 components) on 4 candidate cells (900 s per solve
at most), and
  1. compares cost and cable length of every solution with the proven values in CASES (relative tolerance 1e-4,
     the solver's optimality tolerance),
  2. runs the independent checker (tools/check_solution.py) on every solution: no rule may be violated,
  3. checks the checker itself: deliberately broken copies of the solutions (or of their configs) must be reported
     with the rule that was broken.
Outputs go to results/regression/ (overwritten on every run). Exit code 1 on any difference.
When a model change is meant to change a result, verify the new value and update CASES.
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from check_solution import SolutionCheck, check_path, _pk  # noqa: E402

OUT = os.path.join(REPO, 'results', 'regression')
ARCH = 'configs/full_architecture_v2_lite.json'
COMMON = ['--architecture_json', ARCH, '--num_locs', '4', '--time_limit', '900', '--phase2_time', '120']
TOL = 1e-4

# name, extra flags, growth levels written into the case's uncertainty.json, proven (cost, cable length) per solution
CASES = [
    ('nominal', ['--num_points', '1'], None,
     [(8101.0385, 56.2920)]),
    ('growth', ['--growth', '--sw_margin', '--derating', '--redundancy', '--ai_contention', '--robust_latency'], [0.0, 1.0],
     [(8491.7525, 48.9631), (8531.7405, 52.5428)]),
    ('failures', ['--uncertainty'], [0.0],
     [(11205.688, 58.2231)]),
]


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def copy_configs(dst, overrides=()):
    """JSON configs of configs/ into dst, with overrides [(file, function(dict))] applied"""
    os.makedirs(dst, exist_ok=True)
    src = os.path.join(REPO, 'configs')
    for f in os.listdir(src):
        if f.endswith('.json'):
            shutil.copy(os.path.join(src, f), dst)
    for fname, fn in overrides:
        p = os.path.join(dst, fname)
        with open(p) as fh:
            d = json.load(fh)
        fn(d)
        write_json(p, d)


def run_case(name, flags, levels):
    cfg = os.path.join(OUT, f'cfg_{name}')
    out = os.path.join(OUT, name)
    shutil.rmtree(cfg, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    over = [('uncertainty.json', lambda d: d['growth'].__setitem__('alpha_levels', levels))] if levels is not None else []
    copy_configs(cfg, over)
    os.makedirs(out)
    cmd = [sys.executable, os.path.join(REPO, 'main.py')] + COMMON + flags + \
          ['--config_dir', os.path.relpath(cfg, REPO), '--output_dir', os.path.relpath(out, REPO)]
    t0 = time.time()
    with open(os.path.join(OUT, f'{name}.txt'), 'w') as log:
        rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - t0, out


# ------------------------------------------------------------------------------------------------- mutations
def _first(seq):
    return next(iter(seq), None)


def m_unplaced(sol, chk):
    sc = sorted(sol['assignment'])[0]
    del sol['assignment'][sc]
    return 'placement', f"{sc} removed from the placement"


def m_partition_site(sol, chk):
    for sc, p in sorted(sol['partitions'].items()):
        other = _first(s for s in chk.cands if s != sol['assignment'][sc])
        if other:
            sol['partitions'][sc] = p.replace(sol['assignment'][sc] + '_', other + '_', 1)
            return 'partition', f"partition of {sc} moved to {other}"


def m_hardware(sol, chk):
    for sc, site in sorted(sol['assignment'].items()):
        for h in chk.scs[sc]['hw']:
            item = f"{h}@{site}"
            if item in sol['hw_features']:
                sol['hw_features'].remove(item)
                return 'hardware', f"{item} removed"


def m_redundancy(sol, chk):
    if not chk.flags.get('redundancy'):
        return None
    for sc, info in sorted(chk.scs.items()):
        p = info['redundant_with']
        if p in sol['assignment'] and sc in sol['assignment']:
            sol['assignment'][p] = sol['assignment'][sc]
            return 'redundancy', f"{p} moved next to {sc}"


def m_attachment(sol, chk):
    d = _first(sorted(sol.get('eth_sensor_attachments', {})))
    if d:
        del sol['eth_sensor_attachments'][d]
        return 'peripheral', f"attachment of {d} removed"


def m_ports(sol, chk):
    for s in chk.cands:
        need = sum(1 for d, a in chk.attach.items() if a == s) + sum(c for (a, b, n), c in chk.cables.items() if n == 'ETH' and s in (a, b))
        if need > 0:
            sol['interfaces'] = [i for i in sol['interfaces'] if not i.endswith(f"ETH@{s}")]
            sol['hw_features'] = [h for h in sol['hw_features'] if h != f"SWITCH@{s}"]
            return 'ports', f"Ethernet ports and switch at {s} removed"


def m_cable(sol, chk):
    for f in chk.flows:
        if f.get('path') and len(f['path']) >= 2 and not f.get('line'):
            a, b, n = f['path'][0], f['path'][1], f['iface']
            sol['comm_links'] = [lk for lk in sol['comm_links'] if not (_pk(lk['src_loc'], lk['dst_loc']) == _pk(a, b) and lk['iface'] == n)]
            return 'route', f"cables {a}-{b} {n} removed"


def m_line_member(sol, chk):
    for f in chk.flows:
        if f.get('line') and f.get('path'):
            line = next(b for b in sol['buses'] if b['id'] == f['line'])
            line['units'] = [u for u in line['units'] if u != f['path'][0]]
            return 'route', f"{f['path'][0]} taken off bus line {f['line']} that carries its traffic"


def m_extra_cable(sol, chk):
    act = sorted(chk.node_active)
    idle = _first(s for s in chk.cands if s not in chk.node_active)
    if act and idle:
        sol['comm_links'].append({'src_loc': act[0], 'dst_loc': idle, 'iface': 'ETH', 'count': 1})
        return 'backbone', f"cable {act[0]}-{idle} to an idle cell added"


def m_standby(sol, chk):
    sc = _first(sorted(sol.get('standby', {})))
    if sc:
        del sol['standby'][sc]
        return 'standby', f"standby of {sc} removed"


def m_standby_survival(sol, chk):
    sc = _first(sorted(sol.get('standby', {})))
    if sc:
        del sol['standby'][sc]
        return 'failure_survival', f"standby of {sc} removed (its primary's cell may fail)"


def m_standby_same_cell(sol, chk):
    sc = _first(sorted(sol.get('standby', {})))
    if sc:
        sol['standby'][sc]['cell'] = sol['assignment'][sc]
        return 'standby', f"standby of {sc} on its primary's cell"


def m_bus_drop(sol, chk):
    b = _first(b for b in sol.get('buses', []) if b['devices'])
    if b:
        d = b['devices'].pop(0)
        return 'bus', f"{d} taken off bus line {b['id']}"


def m_bus_length(sol, chk):
    b = _first(b for b in sol.get('buses', []) if b['steiner_m'] > 0)
    if b:
        b['length_m'] = 0.0
        return 'bus_length', f"cable of bus line {b['id']} set to 0 m"


def m_cost(sol, chk):
    sol['total_cost'] += 1.0
    return 'cost', 'total_cost + 1'


def m_report(sol, chk):
    sol['num_partitions'] += 1
    return 'report', 'num_partitions + 1'


def c_partition_cap(sol, chk):
    return 'partition_capacity', [('partitions.json', lambda d: d['partition'].__setitem__('cpu_cap', 1))], 'partition cpu_cap = 1'


def c_site_cap(sol, chk):
    return 'site_capacity', [('partitions.json', lambda d: d['site'].__setitem__('cpu_cap', 1))], 'site cpu_cap = 1'


def c_ai(sol, chk):
    feat = chk.unc.get('ai_contention', {}).get('ai_hw_feature')
    if not chk.flags.get('ai_contention') or not any(feat in chk.scs[sc]['hw'] for sc in chk.pos_x):
        return None
    return 'ai_contention', [('uncertainty.json', lambda d: d['ai_contention'].__setitem__('max_per_location', 0))], 'max_per_location = 0'


def c_bus_cap(sol, chk):
    b = _first(b for b in sol.get('buses', []) if b['device_load'] > 0)
    if b is None:
        return None
    n = b['type']
    return 'bus_capacity', [('buses.json', lambda d: d['cable_types'][n].__setitem__('capacity', 1e-3))], f"{n} capacity = 0.001"


def c_link_cap(sol, chk):
    if not chk.load:
        return None
    n = max(chk.load.items(), key=lambda kv: kv[1])[0][2]
    return 'link_capacity', [('buses.json', lambda d: d['cable_types'][n].__setitem__('capacity', 1e-3))], f"{n} capacity = 0.001"


def _with_crossing(chk):
    return any(f.get('path') for f in chk.flows if f['si'] in ('x', 'a') and f['di'] in ('x', 'a'))


def c_chain(sol, chk):
    if not (chk.chains and _with_crossing(chk)):
        return None
    return 'chain_latency', [('chains.json', lambda d: [c.__setitem__('net_budget_ms', 1e-3) for c in d['chains']])], 'chain budgets = 0.001 ms'


def c_degraded(sol, chk):
    if not (chk.flags.get('failures') and chk.chains and _with_crossing(chk)):
        return None
    return 'degraded_latency', [('chains.json', lambda d: [c.__setitem__('degraded_bound_ms', 1e-3) for c in d['chains']])], 'degraded bounds = 0.001 ms'


def c_cold(sol, chk):
    if not any(t.get('type') == 'cold' for t in sol.get('standby', {}).values()):
        return None
    return 'standby', [('uncertainty.json', lambda d: d['failures'].__setitem__('recovery_ms', 1e9))], 'recovery_ms = 1e9 (cold standbys inadmissible)'


def c_failure_cap(sol, chk):
    """a cell cap that holds every nominal load but not the load of one cell after a cell failure"""
    if not chk.cold:
        return None
    nominal = {s: (sum(chk.demand(sc, 'cpu') for sc, j in chk.pos_x.items() if j == s) +
                   sum(chk.demand(sc, 'cpu') for sc, j in chk.pos_r.items() if j == s and sc not in chk.cold)) / chk.derate(s)
               for s in chk.active}
    floor = max(nominal.values()) * 1.000001
    best = None
    for c in chk.active:
        add = {}
        for sc in chk.crit:
            if chk.pos_x.get(sc) == c and sc in chk.cold:
                j = chk.pos_r[sc]
                add[j] = add.get(j, 0.0) + chk.demand(sc, 'cpu') / chk.derate(j)
        for j, a in add.items():
            cap = max(floor, nominal[j] + 0.5 * a)
            if cap < (nominal[j] + a) / 1.001 and (best is None or cap < best):
                best = cap
    if best is None:
        return None
    return 'failure_capacity', [('partitions.json', lambda d: d['site'].__setitem__('cpu_cap', best))], f"site cpu_cap = {best:.0f}"


SOLUTION_MUTATIONS = [m_unplaced, m_partition_site, m_hardware, m_redundancy, m_attachment, m_ports, m_cable,
                      m_line_member, m_extra_cable, m_standby, m_standby_survival, m_standby_same_cell, m_bus_drop, m_bus_length, m_cost,
                      m_report]
CONFIG_MUTATIONS = [c_partition_cap, c_site_cap, c_ai, c_bus_cap, c_link_cap, c_chain, c_degraded, c_cold, c_failure_cap]


def run_mutations(sol_path):
    """every mutation must be reported with its rule; returns (passed, failed, skipped) lists"""
    base_dir = os.path.dirname(sol_path)
    base_info = json.load(open(os.path.join(base_dir, 'run_info.json')))
    ref = SolutionCheck(sol_path).run()
    passed, failed, skipped = [], [], []
    for k, fn in enumerate(SOLUTION_MUTATIONS + CONFIG_MUTATIONS):
        sol = copy.deepcopy(ref.sol)
        res = fn(sol, ref)
        if res is None:
            skipped.append(fn.__name__)
            continue
        d = os.path.join(OUT, 'mutations', f"{k:02d}_{fn.__name__}")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        info = copy.deepcopy(base_info)
        if fn in CONFIG_MUTATIONS:
            rule, overrides, what = res
            cfg = os.path.join(d, 'cfg')
            os.makedirs(cfg)
            for f in os.listdir(os.path.join(REPO, base_info['args']['config_dir'])):
                if f.endswith('.json'):
                    shutil.copy(os.path.join(REPO, base_info['args']['config_dir'], f), cfg)
            for fname, ofn in overrides:
                p = os.path.join(cfg, fname)
                data = json.load(open(p))
                ofn(data)
                write_json(p, data)
            info['args']['config_dir'] = os.path.relpath(cfg, REPO)
        else:
            rule, what = res
        write_json(os.path.join(d, 'run_info.json'), info)
        write_json(os.path.join(d, 'solution_1.json'), sol)
        chk = SolutionCheck(os.path.join(d, 'solution_1.json')).run()
        rules = {r for r, _m in chk.errors}
        (passed if rule in rules else failed).append((fn.__name__, rule, what, sorted(rules)))
    return passed, failed, skipped


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = True
    rows = []
    solutions = {}
    for name, flags, levels, expected in CASES:
        rc, secs, out = run_case(name, flags, levels)
        print(f"[case {name}] main.py exit {rc} after {secs:.0f} s (log results/regression/{name}.txt)")
        if rc != 0:
            ok = False
            rows.append((name, '-', 'main.py failed', '', '', 'FAIL'))
            continue
        checks = check_path(out)
        solutions[name] = [c.path for c in checks]
        if len(checks) != len(expected):
            ok = False
            rows.append((name, '-', f"{len(checks)} solutions, expected {len(expected)}", '', '', 'FAIL'))
        for idx, (c, (e_cost, e_cable)) in enumerate(zip(checks, expected), 1):
            cost, cable = float(c.sol['total_cost']), float(c.sol['cable_length'])
            proven = all(ph.get('status') == 'OPTIMAL' for ph in c.sol.get('solve_info', {}).get('phases', []))
            same = abs(cost - e_cost) <= TOL * abs(e_cost) and abs(cable - e_cable) <= TOL * abs(e_cable)
            verdict = 'OK' if (same and proven and not c.errors) else 'FAIL'
            ok = ok and verdict == 'OK'
            rows.append((name, idx, f"{cost:.4f} ({e_cost})", f"{cable:.4f} ({e_cable})",
                         ('proven' if proven else 'NOT PROVEN') + (', check OK' if not c.errors else f", {len(c.errors)} check errors"), verdict))
            if c.errors or c.warnings:
                print(c.summary())
    print("\ncase      sol  cost (expected)              cable (expected)          status                      verdict")
    for r in rows:
        print(f"{r[0]:9s} {str(r[1]):4s} {r[2]:28s} {r[3]:25s} {r[4]:27s} {r[5]}")
    # the checker must catch broken solutions: mutations of the failure case (all rule families) and of the nominal case
    for name in ('failures', 'nominal'):
        if not solutions.get(name):
            continue
        passed, failed, skipped = run_mutations(solutions[name][0])
        print(f"\n[checker self-test on {name}] {len(passed)} of {len(passed) + len(failed)} broken solutions detected"
              + (f", not applicable: {', '.join(skipped)}" if skipped else ''))
        for fn, rule, what, rules in failed:
            ok = False
            print(f"  MISSED {fn}: {what} -> expected [{rule}], reported {rules}")
    print("\nREGRESSION " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
