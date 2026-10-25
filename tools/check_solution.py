#!/usr/bin/env python3
"""
Independent check of optimisation results.

  dse/bin/python tools/check_solution.py results/<run>                     (every solution_<i>.json of the run)
  dse/bin/python tools/check_solution.py results/<run>/solution_1.json [--config_dir configs] [--architecture <json>]

Re-evaluates every rule of the model on a stored solution with plain Python and recomputes every cost and length:
placement, partitions and their capacities, control-unit caps, hardware, redundancy, AI contention, peripheral
wiring, bus lines (members, ASIL group, shared bit rate, Steiner length), ports, backbone cables and paths, link
capacities, chain latencies (with the bus crossings of the end devices), and with single
failures the standbys, survival in every scenario, degraded chain latencies, link capacity after every failure and
the cold-standby capacity.

It reads the raw config and architecture files itself and imports nothing from the optimiser (optimizer.py,
optimizer_utils.py, architecture_loader.py, config_reader.py, models.py), so an error in the model code cannot hide
here. The settings of a run (config dir, architecture, number of candidate sites) come from run_info.json next to
the solutions. Exit code 1 when a rule is violated or a reported number differs from its recomputation.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SHARED_BUSES = ('CAN', 'LIN', 'FLEXRAY')
SWITCH_ETH_PORTS = 16            # Ethernet ports provided by a SWITCH (constant of the model)
RES = ('cpu', 'ram', 'rom')
PART_RE = re.compile(r'^(?P<site>.+)_asil(?P<asil>\d+)_p(?P<k>\d+)$')
DRIVER_RE = re.compile(r'^SC_\d+_Driver_(.+)$')
VARIANT_ORDER = {'SC_SC': [('x', 'x'), ('x', 'r'), ('r', 'x'), ('r', 'r')],
                 'SENS_SC': [('a', 'x'), ('a', 'r')], 'SC_ACT': [('x', 'a'), ('r', 'a')]}
PACK_NODE_LIMIT = 200000


def _load(path):
    with open(path) as f:
        return json.load(f)


def _le(a, b):
    """a <= b within the solver's feasibility tolerance"""
    return a <= b + 1e-6 * max(1.0, abs(b)) + 1e-6


def _close(a, b, rel=1e-6, tol=1e-4):
    return abs(float(a) - float(b)) <= tol + rel * max(abs(float(a)), abs(float(b)))


def _pk(a, b):
    """canonical key of an undirected site pair"""
    return (a, b) if a <= b else (b, a)


def _repo_path(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def _pack(items, demand, cap, n_bins):
    """Vector bin packing: first-fit decreasing, then a depth-first search with a node limit.
    Returns True (packing found), False (proven impossible) or None (search limit reached)."""
    active = [r for r in cap if cap[r] > 0]
    if not items:
        return True
    if n_bins <= 0:
        return False
    order = sorted(items, key=lambda i: max((demand[i][r] / cap[r] for r in active), default=0.0), reverse=True)
    load = [{r: 0.0 for r in active} for _ in range(n_bins)]

    def fits(b, i):
        return all(_le(load[b][r] + demand[i][r], cap[r]) for r in active)

    for i in order:
        b = next((b for b in range(n_bins) if fits(b, i)), None)
        if b is None:
            break
        for r in active:
            load[b][r] += demand[i][r]
    else:
        return True
    load = [{r: 0.0 for r in active} for _ in range(n_bins)]
    nodes = [0]

    def dfs(k):
        if k == len(order):
            return True
        nodes[0] += 1
        if nodes[0] > PACK_NODE_LIMIT:
            raise TimeoutError
        i, seen = order[k], set()
        for b in range(n_bins):
            sig = tuple(round(load[b][r], 6) for r in active)
            if sig in seen or not fits(b, i):
                continue
            seen.add(sig)
            for r in active:
                load[b][r] += demand[i][r]
            if dfs(k + 1):
                return True
            for r in active:
                load[b][r] -= demand[i][r]
        return False

    try:
        return dfs(0)
    except TimeoutError:
        return None


class SolutionCheck:
    def __init__(self, solution_path, config_dir=None, architecture=None):
        self.path = solution_path
        self.sol = _load(solution_path)
        info_path = os.path.join(os.path.dirname(os.path.abspath(solution_path)), 'run_info.json')
        args = _load(info_path).get('args', {}) if os.path.exists(info_path) else {}
        self.config_dir = _repo_path(config_dir or args.get('config_dir') or 'configs')
        self.architecture = _repo_path(architecture or args.get('architecture_json') or 'configs/full_architecture_v2.json')
        self.num_locs = args.get('num_locs')
        self.errors, self.warnings = [], []
        self.checked = defaultdict(int)

    # ------------------------------------------------------------------------------------------------------ helpers
    def err(self, rule, msg):
        self.errors.append((rule, msg))

    def warn(self, rule, msg):
        self.warnings.append((rule, msg))

    def dist(self, a, b):
        if self.dmat is not None and a in self.didx and b in self.didx:
            return float(self.dmat[self.didx[a]][self.didx[b]])
        pa, pb = self.pos[a], self.pos[b]
        return abs(pa['x'] - pb['x']) + abs(pa['y'] - pb['y']) + abs(pa['z'] - pb['z'])

    def mult(self, sc):
        """demand multiplier: sw_margin (listed domains) x growth (1 + alpha * profile[domain])"""
        m = 1.0
        dom = str(sc['domain'] or '').strip()
        if self.flags.get('sw_margin'):
            cfg = self.unc.get('sw_margin', {})
            if dom.upper() in {str(d).strip().upper() for d in cfg.get('domains', [])}:
                margins = {int(k): float(v) for k, v in cfg.get('asil_margins', {}).items()}
                m_asil = margins.get(sc['asil'], float(cfg.get('default_asil_margin', 0.0)))
                m_ai = float(cfg.get('ai_margin', 0.0)) if cfg.get('ai_hw_feature') in sc['hw'] else 0.0
                m *= 1.0 + self.rho * (m_asil + m_ai)
        if self.flags.get('growth'):
            g = self.unc.get('growth', {})
            rc = self.unc.get('rho', {})
            rate = float(rc.get('growth_rate_ref', 0.20)) * float(g.get('profiles', {}).get(dom, g.get('default_profile', 0.0)))
            m *= (1.0 + rate) ** (float(rc.get('growth_years', 5)) * self.rho)
        return m

    def demand(self, sc_id, res):
        sc = self.scs[sc_id]
        return sc[res] * self.mult(sc)

    def derate(self, site):
        return max(0.0, 1.0 - self.rho * (1.0 - self.site_derating.get(site, 1.0))) if self.flags.get('derating') else 1.0

    # --------------------------------------------------------------------------------------------------- loading
    def load(self):
        cfg = lambda f: _load(os.path.join(self.config_dir, f))
        buses = cfg('buses.json')
        self.ct = {n: {'cost': float(s['cost_per_meter']), 'cap': float(s.get('capacity', float('inf'))),
                       'port': float(s.get('port_cost', 0)), 'hop': float(s.get('hop_latency_ms', 0.0)),
                       'frame': float(s.get('frame_kbit', float('inf')))}
                   for n, s in buses['cable_types'].items()}
        self.priorities = bool(buses.get('backbone', {}).get('priorities'))
        self._prio = None
        self.k_paths = max(1, int(buses.get('backbone', {}).get('candidate_paths', 3)))
        parts = cfg('partitions.json')
        self.part, self.site_cfg = parts.get('partition', {}), parts.get('site', {})
        self.hw_price = {h: float(s.get('cost', 0)) for h, s in cfg('hardwares.json').get('hardware_costs', {}).items()}
        self.hw_cap = {h: s for h, s in cfg('hardwares.json').get('hardware_costs', {}).items() if s.get('tops')}
        self.unc = cfg('uncertainty.json')
        chains_path = os.path.join(self.config_dir, 'chains.json')
        self.chains = _load(chains_path).get('chains', []) if os.path.exists(chains_path) else []
        self.devices, self.pos = {}, {}
        for kind, fname in (('sensor', 'sensors.json'), ('actuator', 'actuators.json')):
            for d in cfg(fname)[kind + 's']:
                self.devices[d['id']] = {'kind': kind, 'iface': d['interface'], 'volume': float(d.get('volume', 1.0)),
                                         'rate': float(d.get('rate_hz', 10.0) or 10.0)}
                self.pos[d['id']] = d['position']
        all_sites = cfg('sites.json')['sites']
        for s in all_sites:
            self.pos[s['id']] = s['position']
        self.site_derating = {s['id']: float(s.get('derating', 1.0)) for s in all_sites}
        if self.sol.get('candidate_sites'):
            self.cands = list(self.sol['candidate_sites'])
        else:
            n = self.num_locs
            if n is not None and 0 < n < len(all_sites):   # subset rule of the loader (evenly spread)
                step = len(all_sites) / float(n)
                self.cands = [all_sites[int(i * step)]['id'] for i in range(n)]
            else:
                self.cands = [s['id'] for s in all_sites]
            self.warn('format', 'candidate_sites not in the solution: derived from run_info num_locs')
        self.cand_set = set(self.cands)
        dpath = os.path.join(self.config_dir, 'distances.json')
        if os.path.exists(dpath):
            d = _load(dpath)
            self.didx, self.dmat = {i: k for k, i in enumerate(d['ids'])}, d['matrix']
        else:
            self.didx, self.dmat = {}, None
            self.warn('format', 'configs/distances.json missing: Manhattan distances')
        arch = _load(self.architecture)
        self.scs, self.sc_order = {}, []
        for c in arch.get('software_components', []):
            sc = {'id': c['id'], 'domain': c.get('domain', 'Unknown'), 'asil': int(c.get('asil_req', 0)),
                  'hw': list(c.get('hw_required', [])), 'redundant_with': c.get('redundant_with'),
                  'tops': float(c.get('tops_req', 0) or 0), 'fail_op': c.get('fail_operational'),
                  'rate': float(c.get('rate_hz', 10.0) or 10.0), 'type': c.get('type', ''),
                  'msg_kbit': (float(c['msg_kbit']) if c.get('msg_kbit') is not None else None),
                  'receives_from': list(c.get('receives_from', [])), 'devices': []}
            for r in RES:
                sc[r] = float(int(c.get(f'{r}_req', 0)))
            m = DRIVER_RE.match(sc['id'])
            if m and sc['type'] in ('sensor', 'actuator'):
                dev = self.devices.get(m.group(1))
                if dev is not None and dev['kind'] == sc['type']:
                    sc['devices'] = [m.group(1)]
            self.scs[sc['id']] = sc
            self.sc_order.append(sc['id'])
        si = self.sol.get('solve_info', {})
        self.flags = dict(si.get('uncertainty') or {})
        self.alpha = float(si.get('growth_alpha') or 0.0)
        self.rho = float(si['rho']) if si.get('rho') is not None else 1.0   # robustness level of the solve
        fcfg = self.unc.get('failures', {})
        self.asil_min = int(fcfg.get('critical_asil_min', 3))
        self.recovery = float(fcfg.get('recovery_ms', 50.0))
        self.classes = set(fcfg.get('classes', ['cell', 'switch', 'cable']))
        self.crit = {s for s, sc in self.scs.items() if (bool(sc['fail_op']) if sc.get('fail_op') is not None else sc['asil'] >= self.asil_min)} if self.flags.get('failures') else set()
        rl = self.unc.get('robust_latency', {})
        self.lat = {'robust': bool(self.flags.get('robust_latency')), 'factor': 1.0,
                    'sync': float(rl.get('sync_ms', 0.0)),
                    'platform': float(rl.get('platform_mean_ms', rl.get('platform_ms', 0.0))) + self.rho * (
                        float(rl.get('platform_p99_ms', rl.get('platform_mean_ms', rl.get('platform_ms', 0.0)))) - float(rl.get('platform_mean_ms', rl.get('platform_ms', 0.0))))}
        self.has_switch_hw = 'SWITCH' in self.hw_price
        lines_cfg = buses.get('backbone', {}).get('bus_lines', {})
        self.bus_umax = float(buses.get('backbone', {}).get('bus_max_utilization', 1.0))
        self.flow_types = [n for n in self.ct if n not in SHARED_BUSES or int(lines_cfg.get(n, 1)) > 0]

    # ------------------------------------------------------------------------------------------- solution parsing
    def parse(self):
        s = self.sol
        self.pos_x = dict(s.get('assignment', {}))
        self.part_of = dict(s.get('partitions', {}))
        self.standby = dict(s.get('standby') or {})
        self.pos_r = {sc: sb['cell'] for sc, sb in self.standby.items()}
        self.cold = {sc for sc, sb in self.standby.items() if sb.get('type') == 'cold'}
        self.hw, self.hw_units = set(), defaultdict(int)
        for item in s.get('hw_features', []):
            h, site = item.rsplit('@', 1)
            self.hw.add((site, h))
            self.hw_units[(site, h)] += 1   # the accelerator may have several units
        self.ports = defaultdict(int)
        for item in s.get('interfaces', []):
            cnt, rest = item.split('x', 1)
            n, site = rest.rsplit('@', 1)
            self.ports[(site, n)] += int(cnt)
        self.cables = defaultdict(int)
        for lk in s.get('comm_links', []):
            self.cables[_pk(lk['src_loc'], lk['dst_loc']) + (lk['iface'],)] += int(lk.get('count', 1))
        self.attach = dict(s.get('eth_sensor_attachments', {}))
        self.attach.update(s.get('eth_actuator_attachments', {}))
        self.buses = list(s.get('buses') or [])
        self.bus_by_id = {b['id']: b for b in self.buses}
        self.bus_of = defaultdict(list)                 # device -> ids of the bus lines it sits on
        for b in self.buses:
            for d in b.get('devices', []):
                self.bus_of[d].append(b['id'])
        if 'buses' not in s and (s.get('shared_sensor_attachments') or s.get('shared_actuator_attachments')):
            self.err('format', 'solution of the older shared-bus model (attachments + trunks): not checkable with this version')
        self._st_cache = {}
        self.switches = {site for (site, h) in self.hw if h == 'SWITCH'}
        self.active = set(self.pos_x.values()) | set(self.pos_r.values())       # sites with a control unit (y)
        self.node_active = self.active | self.switches                          # may terminate cables / peripherals

    # ---------------------------------------------------------------------------------------------- placement
    def check_placement(self):
        for sc in self.sc_order:
            self.checked['placement'] += 1
            site = self.pos_x.get(sc)
            if site is None:
                self.err('placement', f"{sc} is not placed")
            elif site not in self.cand_set:
                self.err('placement', f"{sc} placed at {site}, not a candidate site")
        for sc in self.pos_x:
            if sc not in self.scs:
                self.err('placement', f"unknown component {sc} in the solution")

    def check_partitions(self):
        """members of every partition, its ASIL and site; partition counts; capacities"""
        self.members = defaultdict(list)          # partition name -> [(sc, 'x' | 'r')]
        for sc, site in self.pos_x.items():
            self.checked['partition'] += 1
            self._member(sc, 'x', site, self.part_of.get(sc))
        for sc, sb in self.standby.items():
            self.checked['partition'] += 1
            self._member(sc, 'r', sb['cell'], sb.get('partition'))
        used = defaultdict(set)
        nmem = defaultdict(int)
        for p, mem in self.members.items():
            m = PART_RE.match(p)
            used[(m['site'], int(m['asil']))].add(int(m['k']))
            nmem[(m['site'], int(m['asil']))] += len(mem)
        counts = self.sol.get('partition_counts')
        if counts is None:
            counts = {f"{s}_asil{a}": max(ks) + 1 for (s, a), ks in used.items()}
            self.warn('format', 'partition_counts not in the solution: derived from the partition map')
        self.P = {}
        for key, n in counts.items():
            m = re.match(r'^(?P<site>.+)_asil(?P<asil>\d+)$', key)
            self.P[(m['site'], int(m['asil']))] = int(n)
        max_pa = int(self.part.get('max_per_asil_per_site', 3))
        for (s, a), n in self.P.items():
            self.checked['partition_count'] += 1
            if s not in self.cand_set:
                self.err('partition_count', f"partitions at {s}, not a candidate site")
            if n > max_pa:
                self.err('partition_count', f"{s} ASIL {a}: {n} partitions > {max_pa} per ASIL and site")
            if n > nmem[(s, a)]:
                self.err('partition_count', f"{s} ASIL {a}: {n} partitions for {nmem[(s, a)]} hosted instances (empty partition)")
        for (s, a), ks in used.items():
            if max(ks) + 1 > self.P.get((s, a), 0):
                self.err('partition_count', f"{s} ASIL {a}: partition p{max(ks)} used but {self.P.get((s, a), 0)} opened")
        per_site = defaultdict(int)
        for (s, a), n in self.P.items():
            per_site[s] += n
        site_max = self.site_cfg.get('max_partitions')
        for s in self.active | set(per_site):
            self.checked['partition_count'] += 1
            if s in self.active and per_site[s] < 1:
                self.err('partition_count', f"{s} hosts software but has no partition")
            if site_max is not None and per_site[s] > int(site_max):
                self.err('site_capacity', f"{s}: {per_site[s]} partitions > {site_max} per control unit")
        if int(self.sol.get('num_partitions', -1)) != sum(self.P.values()):
            self.err('report', f"num_partitions {self.sol.get('num_partitions')} != {sum(self.P.values())} opened")
        if int(self.sol.get('num_locations_used', -1)) != len(set(self.pos_x.values())):
            self.err('report', f"num_locations_used {self.sol.get('num_locations_used')} != {len(set(self.pos_x.values()))}")
        # nominal load: primaries and hot standbys (a cold standby uses nothing until a failure activates it)
        for p, mem in self.members.items():
            m = PART_RE.match(p)
            h = self.derate(m['site'])
            for r in RES:
                self.checked['partition_capacity'] += 1
                cap = float(self.part.get(f'{r}_cap', 0) or 0)
                if cap <= 0:
                    continue
                load = sum(self.demand(sc, r) for sc, inst in mem if inst == 'x' or sc not in self.cold)
                hr = h if r == 'cpu' else 1.0   # derating: processing capacity only
                if not _le(load, cap * hr):
                    self.err('partition_capacity', f"{p} {r} {load:.1f} > {cap * hr:.1f}")
        for s in self.active:
            h = self.derate(s)
            for r in RES:
                cap = self.site_cfg.get(f'{r}_cap')
                if cap is None:
                    continue
                self.checked['site_capacity'] += 1
                load = sum(self.demand(sc, r) for sc, site in self.pos_x.items() if site == s)
                load += sum(self.demand(sc, r) for sc, site in self.pos_r.items() if site == s and sc not in self.cold)
                hr = h if r == 'cpu' else 1.0
                if not _le(load, float(cap) * hr):
                    self.err('site_capacity', f"{s} {r} {load:.1f} > {float(cap) * hr:.1f} (cell cap x derating)")

    def _member(self, sc, inst, site, p):
        if sc not in self.scs:
            return
        what = sc if inst == 'x' else f"standby of {sc}"
        if p is None:
            self.err('partition', f"{what} has no partition")
            return
        m = PART_RE.match(p)
        if not m:
            self.err('partition', f"{what}: malformed partition name {p}")
            return
        if m['site'] != site:
            self.err('partition', f"{what} at {site} but its partition {p} is at {m['site']}")
        if int(m['asil']) != self.scs[sc]['asil']:
            self.err('partition', f"{what} (ASIL {self.scs[sc]['asil']}) in partition {p} of ASIL {m['asil']}")
        self.members[p].append((sc, inst))

    def check_hardware_rules(self):
        for sc, site in list(self.pos_x.items()) + list(self.pos_r.items()):
            for h in self.scs.get(sc, {}).get('hw', []):
                self.checked['hardware'] += 1
                if (site, h) not in self.hw:
                    self.err('hardware', f"{sc} needs {h} at {site}")
        if self.flags.get('redundancy'):
            for sc, info in self.scs.items():
                partner = info['redundant_with']
                if partner in self.scs and sc < partner:
                    self.checked['redundancy'] += 1
                    if self.pos_x.get(sc) is not None and self.pos_x.get(sc) == self.pos_x.get(partner):
                        self.err('redundancy', f"redundant pair {sc} / {partner} both at {self.pos_x[sc]}")
        # accelerator capacity (TOPS, always on; replaces the former contention limit K): primaries and hot standbys at
        # a site within units x TOPS per unit x derating
        for hf, spec in self.hw_cap.items():
            load = defaultdict(float)
            for sc, site in list(self.pos_x.items()) + [(sc, st) for sc, st in self.pos_r.items() if sc not in self.cold]:
                if hf in self.scs.get(sc, {}).get('hw', []):
                    load[site] += self.demand(sc, 'tops')
            for site, l in load.items():
                self.checked['tops_capacity'] += 1
                cap = self.hw_units.get((site, hf), 0) * float(spec['tops']) * self.derate(site)
                if not _le(l, cap):
                    self.err('tops_capacity', f"{site}: {hf} {l:.2f} TOPS > {cap:.2f} ({self.hw_units.get((site, hf), 0)} units)")

    # -------------------------------------------------------------------------------------------- peripherals
    def steiner(self, terminals):
        """minimum Steiner tree length of the terminals in the complete graph over the candidate sites and all
        devices with the channel cable lengths (classic Dreyfus-Wagner rooted at the first terminal)"""
        key = tuple(sorted(set(terminals)))
        if key in self._st_cache:
            return self._st_cache[key]
        if len(key) <= 1:
            val = 0.0
        elif len(key) == 2:
            val = self.dist(key[0], key[1])
        else:
            nodes = sorted(set(self.cands) | set(self.devices))
            d = {(a, b): min(self.dist(a, b), self.dist(b, a)) for a in nodes for b in nodes}
            root, rest = key[0], key[1:]
            best = {}                                    # (frozenset of terminals, node) -> tree joining them and the node
            for c in rest:
                for v in nodes:
                    best[frozenset([c]), v] = d[c, v]
            for size in range(2, len(rest) + 1):
                for combo in __import__('itertools').combinations(rest, size):
                    S = frozenset(combo)
                    members = sorted(S)
                    first, others = members[0], members[1:]
                    split = {}
                    for u in nodes:                      # join two parts at u (each split once: part with `first`)
                        val_u = float('inf')
                        for r in range(0, len(others)):
                            for part in __import__('itertools').combinations(others, r):
                                E = frozenset((first,) + part)
                                val_u = min(val_u, best[E, u] + best[S - E, u])
                        split[u] = val_u
                    for v in nodes:
                        best[S, v] = min(split[u] + d[u, v] for u in nodes)
            val = best[frozenset(rest), root]
        self._st_cache[key] = val
        return val

    def robust(self, base):
        return base * self.lat['factor'] + self.lat['sync'] + self.lat['platform'] if self.lat['robust'] else base

    def dev_kbit(self, d):
        return self.devices[d]['volume'] * 1000.0 / self.devices[d]['rate']

    def end_devices(self, ch):
        """bus devices at the ends of a chain: a bus sensor at its first and a bus actuator at its last component
        (per end the device with the longest own crossing)"""
        idx = [s for s in ch.get('swcs', []) if s in self.scs]
        ends = []
        for sc, kind in (((idx[0], 'sensor'), (idx[-1], 'actuator')) if idx else ()):
            cands = [d for d in self.scs[sc]['devices'] if self.devices[d]['kind'] == kind and self.devices[d]['iface'] in SHARED_BUSES]
            if cands:
                ends.append(max(cands, key=lambda d: self.dev_kbit(d) / self.ct[self.devices[d]['iface']]['cap'] + self.ct[self.devices[d]['iface']]['hop']))
        return ends

    def messages(self, flows, failed_line=None):
        """what waits on each resource: {resource: [(message, kbit)]}; resources are directed cables (a, b, type) and
        bus lines ('line', id); messages are the given flows and the devices of every line except a failed one"""
        msgs = defaultdict(list)
        for f in flows:
            if f['path'] is None:
                continue
            for res in self.resources(f):
                msgs[res].append((('f', id(f)), f['kbit']))
        for b in self.buses:
            if b['id'] != failed_line:
                for d in b['devices']:
                    if d in self.devices:
                        msgs[('line', b['id'])].append((('d', d), self.dev_kbit(d)))
        return msgs

    def resources(self, f):
        if f.get('line'):
            return [('line', f['line'])]
        return [(a, b, f['iface']) for a, b in zip(f['path'], f['path'][1:])]

    def rate(self, res):
        return self.ct[self.bus_by_id[res[1]]['type'] if res[0] == 'line' else res[2]]['cap']

    def prio(self, m):
        """priority key of a message (lower = served first): with backbone.priorities, a flow of a chain (any instance)
        has the smallest network budget of the chains it belongs to, so has an end device of a chain; all else last"""
        if not self.priorities:
            return 0.0
        if self._prio is None:
            by_base, by_dev = {}, {}
            for ch in self.chains:
                budget = float(ch['net_budget_ms'])
                for b in self.chain_flows(ch):
                    by_base[b] = min(by_base.get(b, float('inf')), budget)
                for d in self.end_devices(ch):
                    by_dev[d] = min(by_dev.get(d, float('inf')), budget)
            self._prio = ({id(f): by_base.get(f['base'], float('inf')) for f in self.flows}, by_dev)
        return self._prio[0].get(m[1], float('inf')) if m[0] == 'f' else self._prio[1].get(m[1], float('inf'))

    def queue(self, res, me, msgs):
        """wait in front of `me`: every other message of equal or higher priority once, and at most one frame of a
        lower-priority message (fixed priorities, non-preemptive); without priorities every other message once"""
        p, frame = self.prio(me), self.ct[self.bus_by_id[res[1]]['type'] if res[0] == 'line' else res[2]]['frame']
        wait = sum(k for m, k in msgs.get(res, []) if m != me and self.prio(m) <= p)
        block = max((min(k, frame) for m, k in msgs.get(res, []) if m != me and self.prio(m) > p), default=0.0)
        return (wait + block) / self.rate(res)

    def device_bus_ms(self, ch, msgs, failed_line=None):
        """bus crossings of the end devices of a chain, each with the queue of its line; a device with two lanes counts
        with its slowest surviving lane (each lane must meet the bound on its own)"""
        total = 0.0
        for d in self.end_devices(ch):
            t = self.ct[self.devices[d]['iface']]
            lines = [b for b in self.bus_of.get(d, []) if b != failed_line]
            q = max((self.queue(('line', b), ('d', d), msgs) for b in lines), default=0.0)
            total += self.robust(self.dev_kbit(d) / t['cap'] + t['hop'] + q)
        return total

    def check_peripherals(self):
        users = defaultdict(list)           # device -> [(sc, site)] of the primaries driving it
        drivers = defaultdict(list)         # device -> components of the architecture driving it
        for sc_id in self.sc_order:
            for d in self.scs[sc_id]['devices']:
                drivers[d].append(sc_id)
        for sc, site in self.pos_x.items():
            for d in self.scs.get(sc, {}).get('devices', []):
                users[d].append((sc, site))
        eth_cap = self.ct.get('ETH', {}).get('cap', float('inf'))
        for d, dev in self.devices.items():
            if dev['iface'] == 'ETH':
                self.checked['peripheral'] += 1
                att = self.attach.get(d)
                if att is None:
                    self.err('peripheral', f"Ethernet device {d} is not attached")
                    continue
                if att not in self.node_active:
                    self.err('peripheral', f"{d} attached at {att} without a control unit or switch")
                if not _le(dev['volume'], eth_cap):
                    self.err('peripheral', f"{d} volume {dev['volume']} > Ethernet capacity {eth_cap}")
        for d in self.attach:
            if d not in self.devices:
                self.err('peripheral', f"unknown device {d} attached")
        # bus lines: members, one ASIL group, shared bit rate, cable at least the Steiner tree of the members
        by_id = {}
        for b in self.buses:
            bid, n = b['id'], b['type']
            by_id[bid] = b
            self.checked['bus'] += 1
            if n not in SHARED_BUSES or n not in self.ct:
                self.err('bus', f"{bid}: {n} is not a bus type")
                continue
            if not b['units']:
                self.err('bus', f"{bid}: line without a control unit")
            for u in b['units']:
                if u not in self.cand_set:
                    self.err('bus', f"{bid}: member {u} is not a candidate site")
                elif u not in self.active:
                    self.err('bus', f"{bid}: member {u} has no control unit")
            groups = set()
            for d in b['devices']:
                dev = self.devices.get(d)
                if dev is None:
                    self.err('bus', f"{bid}: unknown device {d}")
                    continue
                if dev['iface'] != n:
                    self.err('bus', f"{bid}: {dev['iface']} device {d} on a {n} line")
                groups.add('HIGH' if any(self.scs[u]['asil'] >= 3 for u in drivers[d]) else 'LOW')
            if len(groups) > 1:
                self.err('bus', f"{bid}: devices of both ASIL groups on one line")
            elif groups and b.get('asil_group') not in groups:
                self.err('report', f"{bid}: reported ASIL group {b.get('asil_group')}, devices {groups.pop()}")
            load = sum(self.devices[d]['volume'] for d in b['devices'] if d in self.devices)
            self.checked['bus_capacity'] += 1
            if not _le(load, self.bus_umax * self.ct[n]['cap']):
                self.err('bus_capacity', f"{bid}: device load {load} > {self.bus_umax} x {self.ct[n]['cap']}")
            st = self.steiner(list(b['units']) + list(b['devices']))
            self.checked['bus_length'] += 1
            if not _le(st, float(b['length_m'])):   # within the solver's tolerances (binary members at 1 - IntFeasTol)
                self.err('bus_length', f"{bid}: cable {float(b['length_m']):.3f} m shorter than the Steiner tree of its members {st:.3f} m")
            elif float(b['length_m']) > st + 1e-4:
                self.warn('bus_length', f"{bid}: cable {float(b['length_m']):.3f} m longer than needed ({st:.3f} m)")
        for d, dev in self.devices.items():
            if dev['iface'] not in SHARED_BUSES:
                continue
            lines = self.bus_of.get(d, [])
            if not drivers[d]:
                if len(lines) > 1:
                    self.err('bus', f"{d} on {len(lines)} lines")
                continue
            self.checked['bus'] += 1
            # with failures on, a device of critical software has two lanes: two distinct lines of its type
            need = 2 if any(sc in self.crit for sc in drivers[d]) else 1
            if len(set(lines)) != len(lines) or len(lines) != need:
                self.err('bus', f"{dev['iface']} device {d} on {len(lines)} lines (needs exactly {need})")
                continue
            for line in lines:
                b = by_id[line]
                for sc, site in users.get(d, []):
                    if site not in b['units']:
                        self.err('bus', f"{d} on {b['id']} but its software {sc} runs at {site}, not a member")
                    if sc in self.pos_r and self.pos_r[sc] not in b['units']:
                        self.err('bus', f"{d} on {b['id']}: the standby of {sc} at {self.pos_r[sc]} is not a member")
        # ports per site and interface
        bb_ports = defaultdict(int)
        for (a, b, n), cnt in self.cables.items():
            bb_ports[(a, n)] += cnt
            bb_ports[(b, n)] += cnt
        bus_ports = defaultdict(int)        # (site, bus type) -> lines the unit belongs to
        for b in self.buses:
            for u in b['units']:
                bus_ports[(u, b['type'])] += 1
        direct = defaultdict(int)           # (site, iface) -> peripherals wired directly to the hosting unit
        for sc, site in list(self.pos_x.items()) + [(sc, st) for sc, st in self.pos_r.items() if sc in self.crit]:
            for d in self.scs.get(sc, {}).get('devices', []):
                n = self.devices[d]['iface']
                if n != 'ETH' and n not in SHARED_BUSES:
                    direct[(site, n)] += 1      # primaries, and standbys over their own cable
        for s in set(self.cands):
            for n, t in self.ct.items():
                have = self.ports.get((s, n), 0)
                if n == 'ETH':
                    need = sum(1 for d, a in self.attach.items() if a == s and self.devices.get(d, {}).get('iface') == 'ETH') + bb_ports[(s, n)]
                    cover = have + (SWITCH_ETH_PORTS if (self.has_switch_hw and s in self.switches) else 0)
                elif n in SHARED_BUSES:
                    need, cover = bus_ports[(s, n)] + bb_ports[(s, n)], have
                else:
                    need, cover = direct[(s, n)] + bb_ports[(s, n)], have
                if need > 0 or have > 0:
                    self.checked['ports'] += 1
                if need > cover:
                    self.err('ports', f"{s} {n}: {need} ports needed, {cover} provided")

    # ---------------------------------------------------------------------------------------------- backbone
    def build_flows(self):
        base = []
        for dst in self.sc_order:
            for src in self.scs[dst]['receives_from']:
                if src not in self.scs:
                    continue
                s = self.scs[src]
                volume, rate = 10.0, s['rate']
                if s['msg_kbit'] is not None:
                    volume = s['msg_kbit'] * rate / 1000.0
                elif s['devices']:
                    dev = self.devices[s['devices'][0]]
                    volume, rate = dev['volume'], dev['rate']
                if volume > 0:
                    base.append({'type': 'SC_SC', 'src': src, 'dst': dst, 'volume': volume, 'kbit': volume * 1000.0 / rate})
        for kind, ftype in (('sensor', 'SENS_SC'), ('actuator', 'SC_ACT')):
            for sc in self.sc_order:
                for d in self.scs[sc]['devices']:
                    dev = self.devices[d]
                    if dev['kind'] != kind or dev['iface'] != 'ETH' or dev['volume'] <= 0:
                        continue
                    src, dst = (d, sc) if kind == 'sensor' else (sc, d)
                    base.append({'type': ftype, 'src': src, 'dst': dst, 'volume': dev['volume'], 'kbit': dev['volume'] * 1000.0 / dev['rate']})
        self.flows = []
        for b, f in enumerate(base):
            si, di = {'SC_SC': ('x', 'x'), 'SENS_SC': ('a', 'x'), 'SC_ACT': ('x', 'a')}[f['type']]
            keys = [(si, di)]
            if self.crit:
                if f['type'] == 'SC_SC':
                    keys += [k for k, ok in ((('r', 'x'), f['src'] in self.crit), (('x', 'r'), f['dst'] in self.crit),
                                             (('r', 'r'), f['src'] in self.crit and f['dst'] in self.crit)) if ok]
                elif f['type'] == 'SENS_SC' and f['dst'] in self.crit:
                    keys.append(('a', 'r'))
                elif f['type'] == 'SC_ACT' and f['src'] in self.crit:
                    keys.append(('r', 'a'))
            for k in keys:
                self.flows.append({**f, 'base': b, 'si': k[0], 'di': k[1]})
        self.variants = defaultdict(dict)       # base -> {(si, di): flow}
        for f in self.flows:
            self.variants[f['base']][(f['si'], f['di'])] = f
        self.base = base
        # routes stored with the solution (flow_routes); older result files only have the routes per site pair
        stored = self.sol.get('flow_routes')
        if stored is not None:
            queue = defaultdict(list)
            for e in stored:
                queue[(e['src'], e['dst'], e['src_inst'], e['dst_inst'])].append(e)
            for f in self.flows:
                q = queue.get((f['src'], f['dst'], f['si'], f['di']))
                e = q.pop(0) if q else None
                f['iface'] = e['iface'] if e else None
                f['path'] = list(e['path']) if (e and e.get('path')) else None
                f['line'] = e.get('line') if e else None
                f['known'] = e is not None
                if e is None:
                    self.err('report', f"flow {f['src']} -> {f['dst']} ({f['si']}/{f['di']}) missing in flow_routes")
            for key, q in queue.items():
                for e in q:
                    self.err('report', f"flow_routes lists {key} which is no data flow of the architecture")
        else:
            self.warn('format', 'flow_routes not in the solution: routes taken per site pair from backbone_routes')
            by_pair = defaultdict(list)
            for r in self.sol.get('backbone_routes', []):
                by_pair[_pk(r['src_loc'], r['dst_loc'])].append(r)
            for f in self.flows:
                u, v = self.ends(f)
                f['iface'], f['path'], f['line'], f['known'] = None, None, None, True
                if u is None or v is None or u == v:
                    continue
                rs = by_pair.get(_pk(u, v), [])
                if len(rs) == 1:
                    path = list(rs[0]['path'])
                    f['iface'], f['path'] = rs[0]['iface'], (path if path[0] == u else path[::-1])
                elif len(rs) > 1:
                    f['known'] = False
        unknown = sum(1 for f in self.flows if not f['known'])
        if unknown and stored is None:
            self.warn('format', f"{unknown} flows on site pairs with several cable types: their type is unknown, not checked")

    def ends(self, f):
        def site(ent, inst):
            if inst == 'x':
                return self.pos_x.get(ent)
            if inst == 'r':
                return self.pos_r.get(ent)
            return self.attach.get(ent)
        return site(f['src'], f['si']), site(f['dst'], f['di'])

    def silent(self, f):
        """a flow to or from a cold standby carries no traffic in the nominal state (its path must still exist)"""
        return (f['si'] == 'r' and f['src'] in self.cold) or (f['di'] == 'r' and f['dst'] in self.cold)

    def check_backbone(self):
        for (a, b, n), cnt in self.cables.items():
            self.checked['backbone'] += 1
            for s in (a, b):
                if s not in self.cand_set:
                    self.err('backbone', f"cable {a}-{b} {n} ends at {s}, not a candidate site")
                elif s not in self.node_active:
                    self.err('backbone', f"cable {a}-{b} {n} ends at {s} without a control unit or switch")
            if n not in self.ct:
                self.err('backbone', f"cable {a}-{b} of unknown type {n}")
        for s in self.switches - self.active:
            self.checked['backbone'] += 1
            deg = sum(c for (a, b, _n), c in self.cables.items() if s in (a, b))
            if deg < 2:
                self.err('backbone', f"switch-only node {s} has {deg} cable(s), needs >= 2")
        caps = {n: t['cap'] for n, t in self.ct.items()}
        cand_index = {s: k for k, s in enumerate(self.cands)}
        path_of = {}
        self.load = defaultdict(float)
        self.line_traffic = defaultdict(float)
        for (a, b, n), cnt in self.cables.items():
            if n in SHARED_BUSES:
                self.err('backbone', f"point-to-point {n} cable {a}-{b}: units are joined by {n} bus lines only")
        for f in self.flows:
            u, v = self.ends(f)
            name = f"{f['type']} {f['src']} -> {f['dst']} ({f['si']}/{f['di']})"
            if u is None or v is None:
                self.err('route', f"{name}: an endpoint has no site")
                continue
            if u == v or not f['known']:
                continue
            self.checked['route'] += 1
            if f['path'] is None:
                self.err('route', f"{name}: {u} -> {v} without a backbone path")
                continue
            path, n = f['path'], f['iface']
            if path[0] != u or path[-1] != v:
                self.err('route', f"{name}: path {'-'.join(path)} does not join {u} and {v}")
                continue
            elig = [m for m in self.flow_types if f['volume'] <= caps[m]] or [max(self.flow_types, key=caps.get)]
            if n not in elig:
                self.err('route', f"{name}: {n} cannot carry {f['volume']}")
            if n in SHARED_BUSES:
                line = self.bus_by_id.get(f['line'])
                if line is None or line['type'] != n:
                    self.err('route', f"{name}: {n} flow without a {n} bus line ({f['line']})")
                    continue
                if len(path) != 2:
                    self.err('route', f"{name}: a bus line is one hop, path {'-'.join(path)}")
                for s_ in (u, v):
                    if s_ not in line['units']:
                        self.err('route', f"{name}: {s_} is not a member of bus line {line['id']}")
                key = _pk(u, v) + (n,)
                if path_of.setdefault(key, ('line', line['id'])) != ('line', line['id']):
                    self.err('route', f"{u}-{v} {n}: two different lines ({path_of[key][1]} and {line['id']})")
                if not self.silent(f):
                    self.line_traffic[line['id']] += f['volume']
                continue
            if len(path) == 3:
                relay = path[1]
                if n != 'ETH':
                    self.err('route', f"{name}: relayed {n} path (only Ethernet is switched)")
                if relay not in self.switches:
                    self.err('route', f"{name}: relay {relay} without a SWITCH")
                p, q = sorted((u, v), key=lambda s: cand_index.get(s, 0))
                ranked = sorted((r for r in self.cands if r not in (u, v)), key=lambda r: self.dist(p, r) + self.dist(r, q))
                if relay not in ranked[:self.k_paths - 1]:
                    self.err('route', f"{name}: relay {relay} not among the {self.k_paths - 1} candidate relays of {u}-{v}")
            elif len(path) != 2:
                self.err('route', f"{name}: path {'-'.join(path)} has {len(path) - 1} links (direct or one relay only)")
                continue
            for a, b in zip(path, path[1:]):
                if self.cables.get(_pk(a, b) + (n,), 0) < 1:
                    self.err('route', f"{name}: link {a}-{b} has no {n} cable")
            key = _pk(u, v) + (n,)
            canon = tuple(path) if path[0] == key[0] else tuple(path[::-1])
            if path_of.setdefault(key, canon) != canon:
                self.err('route', f"{u}-{v} {n}: two different paths ({'-'.join(path_of[key])} and {'-'.join(canon)})")
            if not self.silent(f):
                for a, b in zip(path, path[1:]):
                    self.load[(a, b, n)] += f['volume']
        for (a, b, n), ld in self.load.items():
            self.checked['link_capacity'] += 1
            cap = caps[n] * self.cables.get(_pk(a, b) + (n,), 0)
            if not _le(ld, cap):
                self.err('link_capacity', f"{a}->{b} {n}: load {ld:.1f} > {cap:.1f}")
        # a bus line: its devices and the unit-to-unit traffic of both directions share its bit rate
        for b in self.buses:
            self.checked['bus_capacity'] += 1
            total = sum(self.devices[d]['volume'] for d in b['devices'] if d in self.devices) + self.line_traffic.get(b['id'], 0.0)
            if b['type'] in self.ct and not _le(total, self.bus_umax * self.ct[b['type']]['cap']):
                self.err('bus_capacity', f"{b['id']}: devices + traffic {total:.1f} > {self.bus_umax} x {self.ct[b['type']]['cap']}")
            if not _close(b.get('traffic_load', 0.0), self.line_traffic.get(b['id'], 0.0), tol=1e-3):
                self.err('report', f"{b['id']}: reported traffic {b.get('traffic_load')}, recomputed {self.line_traffic.get(b['id'], 0.0)}")
        # reported peak load per link (both directions, nominal state)
        rep = self.sol.get('comm_link_peak_load', {})
        calc = defaultdict(float)
        for (a, b, n), ld in self.load.items():
            p = _pk(a, b)
            calc[f"{p[0]}|{p[1]}|{n}"] = max(calc[f"{p[0]}|{p[1]}|{n}"], ld)
        if not any(not f['known'] for f in self.flows):
            for k in set(rep) | set(calc):
                if not _close(rep.get(k, 0.0), calc.get(k, 0.0), tol=1e-3):
                    self.err('report', f"comm_link_peak_load {k}: reported {rep.get(k, 0.0)}, nominal load {calc.get(k, 0.0)}")

    # --------------------------------------------------------------------------------------------- latency
    def crossing_ms(self, f, msgs):
        """per hop: own transmission, per-hop bound, and the queue on the resource (see queue)"""
        t = self.ct[f['iface']]
        base = sum(f['kbit'] / t['cap'] + t['hop'] + self.queue(res, ('f', id(f)), msgs) for res in self.resources(f))
        return self.robust(base)

    def chain_flows(self, ch):
        """base flows of a chain: sensor -> first component, consecutive components, last component -> actuator"""
        into = {f['dst']: b for b, f in enumerate(self.base) if f['type'] == 'SENS_SC'}
        pair = {(f['src'], f['dst']): b for b, f in enumerate(self.base) if f['type'] == 'SC_SC'}
        out = {f['src']: b for b, f in enumerate(self.base) if f['type'] == 'SC_ACT'}
        idx = [s for s in ch.get('swcs', []) if s in self.scs]
        flows = [into[idx[0]]] if idx and idx[0] in into else []
        flows += [pair[a, b] for a, b in zip(idx, idx[1:]) if (a, b) in pair]
        flows += [out[idx[-1]]] if idx and idx[-1] in out else []
        return flows

    def check_chains(self):
        reported = self.sol.get('chain_latency', {})
        nominal = self.messages([f for f in self.flows if not self.silent(f) and f['known']])
        for ch in self.chains:
            total, unknown = 0.0, False
            for b in self.chain_flows(ch):
                f = self.variants[b][('x', 'x') if self.base[b]['type'] == 'SC_SC' else
                                     (('a', 'x') if self.base[b]['type'] == 'SENS_SC' else ('x', 'a'))]
                u, v = self.ends(f)
                if u is None or v is None or u == v:
                    continue
                if not f['known']:
                    unknown = True
                    continue
                if f['path'] is None:
                    continue                    # reported by the route check
                total += self.crossing_ms(f, nominal)
            if unknown:
                continue
            total += self.device_bus_ms(ch, nominal)
            self.checked['chain_latency'] += 1
            budget = float(ch['net_budget_ms'])
            if not _le(total, budget):
                self.err('chain_latency', f"{ch['id']}: {total:.3f} ms > budget {budget} ms")
            if ch['id'] in reported and not _close(reported[ch['id']]['latency_ms'], total, tol=2e-3):
                self.err('report', f"chain_latency {ch['id']}: reported {reported[ch['id']]['latency_ms']}, recomputed {total:.3f}")

    # ---------------------------------------------------------------------------------------------- failures
    def check_failures(self):
        if not self.flags.get('failures'):
            if self.standby:
                self.err('standby', f"{len(self.standby)} standbys although the failure block is off")
            return
        ftti = {}
        for ch in self.chains:
            for s in ch.get('swcs', []):
                ftti[s] = min(ftti.get(s, float('inf')), float(ch.get('ftti_ms', float('inf'))))
        for sc in sorted(self.crit):
            self.checked['standby'] += 1
            sb = self.standby.get(sc)
            if sb is None:
                self.err('standby', f"critical {sc} has no standby")
                continue
            if sb['cell'] == self.pos_x.get(sc):
                self.err('standby', f"standby of {sc} on the cell of its primary {sb['cell']}")
            if sb['cell'] not in self.cand_set:
                self.err('standby', f"standby of {sc} at {sb['cell']}, not a candidate site")
            if sb.get('type') == 'cold' and ftti.get(sc, float('inf')) < self.recovery:
                self.err('standby', f"cold standby of {sc}: FTTI {ftti[sc]} ms < recovery {self.recovery} ms")
            if sb.get('type') not in ('hot', 'cold'):
                self.err('standby', f"standby of {sc} of unknown type {sb.get('type')}")
        for sc in self.standby:
            if sc not in self.crit:
                self.err('standby', f"standby for the non-critical {sc}")
        if any(not f['known'] for f in self.flows):
            self.warn('failure_survival', 'routes of some flows unknown: failure scenarios not checked')
            return
        flows_of = defaultdict(list)            # critical SC -> [(base, role)]
        for b, f in enumerate(self.base):
            for role, ent in (('src', f['src']), ('dst', f['dst'])):
                if ent in self.crit:
                    flows_of[ent].append((b, role))
        scen = []
        if 'cell' in self.classes:
            scen += [('cell', c) for c in sorted(self.node_active)]
        if 'switch' in self.classes and self.has_switch_hw:
            scen += [('switch', c) for c in sorted(self.switches)]
        if 'cable' in self.classes:
            scen += [('cable', k) for k in sorted(self.cables)]
            scen += [('line', ln) for ln in sorted({f['line'] for f in self.flows if f.get('line') and f['path'] and self.ends(f)[0] != self.ends(f)[1]})]

        def alive(f, kind, elem):
            u, v = self.ends(f)
            if u is None or v is None or (kind == 'cell' and elem in (u, v)):
                return False
            if u == v:
                return True
            if f['path'] is None:
                return False
            path = f['path']
            if kind in ('cell', 'switch') and elem in path[1:-1]:
                return False
            if kind == 'switch' and f['iface'] == 'ETH' and elem in (u, v):
                return False                    # the Ethernet ports of a switched cell are the switch's ports
            if kind == 'cable' and f['iface'] not in SHARED_BUSES and elem[2] == f['iface'] and any(_pk(a, b) == elem[:2] for a, b in zip(path, path[1:])):
                return False
            if kind == 'line' and f.get('line') == elem:
                return False
            return True

        def served(sc, inst, kind, elem):
            cell = self.pos_x.get(sc) if inst == 'x' else self.pos_r.get(sc)
            if cell is None or (kind == 'cell' and elem == cell):
                return False
            for b, role in flows_of[sc]:
                f0 = self.base[b]
                if f0['type'] in ('SENS_SC', 'SC_ACT'):
                    dev = f0['src'] if f0['type'] == 'SENS_SC' else f0['dst']
                    if kind in ('cell', 'switch') and elem == self.attach.get(dev):
                        continue                # failure of the device's own attachment element: out of scope
                    keys = [('a', inst)] if f0['type'] == 'SENS_SC' else [(inst, 'a')]
                else:
                    partner = f0['dst'] if role == 'src' else f0['src']
                    if partner not in self.crit:
                        continue                # a non-critical partner may be lost (degraded mode)
                    keys = [((inst, pi) if role == 'src' else (pi, inst)) for pi in ('x', 'r')]
                if not any(alive(self.variants[b][k], kind, elem) for k in keys if k in self.variants[b]):
                    return False
            return True

        # lost functions: non-critical components (no standby) a failure makes unavailable: host cell failed, own bus
        # device on the failed line, a command to an Ethernet actuator without a surviving route, or an input not
        # delivered (no surviving variant, or its source component lost; transitive)
        non_crit = [sc for sc in self.sc_order if sc not in self.crit]
        inputs, act_out = defaultdict(list), defaultdict(list)
        for b, f in enumerate(self.base):
            (act_out[f['src']] if f['type'] == 'SC_ACT' else inputs[f['dst']]).append(b)

        def lost_set(kind, elem):
            dead = lambda b: not any(alive(f, kind, elem) for f in self.variants[b].values())
            lost = {sc for sc in non_crit if (kind == 'cell' and self.pos_x.get(sc) == elem)
                    or (kind == 'line' and any(self.bus_of.get(d) == [elem] for d in self.scs[sc]['devices']))
                    or any(dead(b) for b in act_out[sc])}
            grown = True
            while grown:
                grown = False
                for sc in non_crit:
                    if sc not in lost and any((self.base[b]['type'] == 'SC_SC' and self.base[b]['src'] in lost) or dead(b) for b in inputs[sc]):
                        lost.add(sc)
                        grown = True
            return lost

        def scen_key(kind, element):
            # the report names a cable "A-B TYPE" and a line "bus line ID"
            if kind == 'cable':
                pair, n = element.rsplit(' ', 1)
                return kind, frozenset(pair.split('-')), n
            return kind, element.replace('bus line ', '') if kind == 'line' else element

        reported_lost = {scen_key(e['kind'], e['element']): e['lost_functions']
                         for e in self.sol.get('failure_scenarios') or [] if 'lost_functions' in e}
        chain_flows = {ch['id']: self.chain_flows(ch) for ch in self.chains}
        for kind, elem in scen:
            label = f"{kind} {'-'.join(elem[:2]) + ' ' + elem[2] if kind == 'cable' else elem}"
            for sc in sorted(self.crit):
                self.checked['failure_survival'] += 1
                if not (served(sc, 'x', kind, elem) or served(sc, 'r', kind, elem)):
                    self.err('failure_survival', f"{label}: {sc} has no instance with all critical partners")
            key = (kind, frozenset(elem[:2]), elem[2]) if kind == 'cable' else (kind, elem)
            if key in reported_lost:
                self.checked['lost_functions'] += 1
                lost = lost_set(kind, elem)
                if set(reported_lost[key]) != lost:
                    self.err('report', f"{label}: lost functions reported {sorted(reported_lost[key])}, recomputed {sorted(lost)}")
            # traffic after the failure: the flows that keep running (alive, primaries and hot standbys) and, per data
            # flow, its first surviving variant (the variant rule of the degraded latency)
            carried = {id(f): f for f in self.flows if not self.silent(f) and alive(f, kind, elem)}
            for b in range(len(self.base)):
                pick = next((self.variants[b][k] for k in VARIANT_ORDER[self.base[b]['type']]
                             if k in self.variants[b] and alive(self.variants[b][k], kind, elem)), None)
                if pick is not None:
                    carried.setdefault(id(pick), pick)
            after = self.messages([f for f in carried.values() if self.ends(f)[0] != self.ends(f)[1]],
                                  failed_line=(elem if kind == 'line' else None))
            for ch in self.chains:
                total, broken = 0.0, False
                for b in chain_flows[ch['id']]:
                    pick = next((self.variants[b][k] for k in VARIANT_ORDER[self.base[b]['type']]
                                 if k in self.variants[b] and alive(self.variants[b][k], kind, elem)), None)
                    if pick is None:
                        broken = True           # a non-critical member is lost (survival checked above)
                        break
                    u, v = self.ends(pick)
                    if u != v:
                        total += self.crossing_ms(pick, after)
                if kind == 'line' and any(self.bus_of.get(d) == [elem] for d in self.end_devices(ch)):
                    broken = True               # the only line of a single-lane end device failed: out of scope
                if broken:
                    continue
                total += self.device_bus_ms(ch, after, failed_line=(elem if kind == 'line' else None))
                self.checked['degraded_latency'] += 1
                bound = float(ch.get('degraded_bound_ms', float(ch['net_budget_ms'])))
                if not _le(total, bound):
                    self.err('degraded_latency', f"{label}: {ch['id']} {total:.3f} ms > degraded bound {bound} ms")
            if kind == 'cell':
                self._failure_capacity(elem, label)
            # link capacity after the failure: the flows that keep running (alive, primaries and hot standbys) and,
            # per data flow, its first surviving variant (the variant rule of the degraded latency), which activates
            # the cold standbys it needs, against the remaining cables
            dl = defaultdict(float)
            for f in carried.values():
                if f['path'] is None:
                    continue
                if f.get('line'):
                    dl[('line', f['line'])] += f['volume']
                    continue
                for a, c2 in zip(f['path'], f['path'][1:]):
                    dl[(a, c2, f['iface'])] += f['volume']
            for key, ld in dl.items():
                self.checked['failure_link_capacity'] += 1
                if key[0] == 'line':
                    b = self.bus_by_id[key[1]]
                    ld += sum(self.devices[d]['volume'] for d in b['devices'] if d in self.devices)
                    cap = self.bus_umax * self.ct[b['type']]['cap']
                    if not _le(ld, cap):
                        self.err('failure_link_capacity', f"{label}: bus line {key[1]} devices + traffic {ld:.1f} > {cap:.1f} after the failure")
                    continue
                a, c2, n = key
                cap = self.ct[n]['cap'] * self.cables.get(_pk(a, c2) + (n,), 0)
                if not _le(ld, cap):
                    self.err('failure_link_capacity', f"{label}: {a}->{c2} {n} load {ld:.1f} > {cap:.1f} after the failure")
        # the lost-function metric of the solution: worst and mean count over the reported scenarios
        summary, entries = self.sol.get('lost_functions'), self.sol.get('failure_scenarios') or []
        if summary is not None and entries:
            self.checked['lost_functions'] += 1
            counts = [len(e.get('lost_functions', [])) for e in entries]
            calc = {'non_critical': len(non_crit), 'scenarios': len(counts), 'worst': max(counts), 'mean': sum(counts) / len(counts)}
            for k, v in calc.items():
                if not _close(float(summary.get(k, -1)), float(v), tol=1e-3):
                    self.err('report', f"lost_functions {k}: reported {summary.get(k)}, recomputed {v}")

    def _failure_capacity(self, c, label):
        """cold standbys activated by the failure of cell c must fit next to the load of their cell"""
        act = [sc for sc in self.crit if self.pos_x.get(sc) == c and sc in self.cold]
        by_site = defaultdict(list)
        for sc in act:
            by_site[self.pos_r[sc]].append(sc)
        for j, lst in by_site.items():
            h = self.derate(j)
            hot_here = [sc for sc, s in self.pos_r.items() if s == j and sc not in self.cold]
            prim = [sc for sc, s in self.pos_x.items() if s == j]
            for r in RES:
                cap = self.site_cfg.get(f'{r}_cap')
                if cap is None:
                    continue
                self.checked['failure_capacity'] += 1
                load = sum(self.demand(sc, r) for sc in prim + hot_here + lst)
                hr = h if r == 'cpu' else 1.0
                if not _le(load, float(cap) * hr):
                    self.err('failure_capacity', f"{label}: {j} {r} {load:.1f} > {float(cap) * hr:.1f} with the activated standbys")
            for hf, spec in self.hw_cap.items():   # accelerator TOPS with the activated standbys
                users = [sc for sc in prim + hot_here + lst if hf in self.scs[sc]['hw']]
                if users:
                    self.checked['failure_capacity'] += 1
                    load = sum(self.demand(sc, 'tops') for sc in users)
                    cap = self.hw_units.get((j, hf), 0) * float(spec['tops']) * h
                    if not _le(load, cap):
                        self.err('failure_capacity', f"{label}: {j} {hf} {load:.2f} TOPS > {cap:.2f} with the activated standbys")
            for a in sorted({self.scs[sc]['asil'] for sc in lst}):
                group = [sc for sc in prim + hot_here + lst if self.scs[sc]['asil'] == a]
                n = self.P.get((j, a), 0)
                caps = {r: float(self.part.get(f'{r}_cap', 0) or 0) * (h if r == 'cpu' else 1.0) for r in RES}
                for r in RES:
                    if caps[r] <= 0:
                        continue
                    self.checked['failure_capacity'] += 1
                    load = sum(self.demand(sc, r) for sc in group)
                    if not _le(load, n * caps[r]):
                        self.err('failure_capacity', f"{label}: {j} ASIL {a} {r} {load:.1f} > {n} x {caps[r]:.1f}")
                # beyond the model (it pools the partitions of one ASIL): can the group be packed into its partitions?
                dem = {sc: {r: self.demand(sc, r) for r in RES} for sc in group}
                ok = _pack(group, dem, caps, n)
                if ok is False:
                    self.warn('failure_packing', f"{label}: {j} ASIL {a}: {len(group)} instances fit pooled but not packed into {n} partition(s)")
                elif ok is None:
                    self.warn('failure_packing', f"{label}: {j} ASIL {a}: packing not decided within the search limit")

    # ----------------------------------------------------------------------------------------------- costs
    def check_costs(self):
        s = self.sol
        part_cost = sum(self.P.values()) * float(self.part.get('cost', 0))
        site_cost = len(self.active) * float(self.site_cfg.get('fixed_cost', 0) or 0)
        hw_cost = sum(self.hw_price.get(h, 0.0) * n for (_site, h), n in self.hw_units.items())
        for (_site, h) in self.hw:
            if h not in self.hw_price:
                self.err('cost', f"hardware {h} has no price in hardwares.json")
        if_cost = sum(cnt * self.ct[n]['port'] for (_site, n), cnt in self.ports.items())
        per_len, per_cost = 0.0, 0.0
        wired = list(self.pos_x.items()) + [(sc, st) for sc, st in self.pos_r.items() if sc in self.crit]
        for sc, site in wired:              # a critical component's standby has its own cable to a directly wired device
            for d in self.scs.get(sc, {}).get('devices', []):
                n = self.devices[d]['iface']
                if n == 'ETH' or n in SHARED_BUSES:
                    continue                    # Ethernet: attachment cable; bus types: bus line
                dl = self.dist(d, site)
                per_len += dl
                per_cost += dl * self.ct[n]['cost']
        for d, site in self.attach.items():
            if d not in self.devices:
                continue
            dl = self.dist(d, site)
            per_len += dl
            per_cost += dl * self.ct[self.devices[d]['iface']]['cost']
        for b in self.buses:
            if b['type'] in self.ct:
                per_len += float(b['length_m'])
                per_cost += float(b['length_m']) * self.ct[b['type']]['cost']
        bb_len = sum(cnt * self.dist(a, b) for (a, b, _n), cnt in self.cables.items())
        bb_cost = sum(cnt * self.dist(a, b) * self.ct[n]['cost'] for (a, b, n), cnt in self.cables.items())
        total = part_cost + site_cost + hw_cost + if_cost + per_cost + bb_cost
        for key, val in (('partition_cost', part_cost), ('site_cost', site_cost), ('hardware_cost', hw_cost),
                         ('interface_cost', if_cost), ('cable_cost', per_cost), ('comm_cost', bb_cost),
                         ('total_cost', total), ('peripheral_cable_length', per_len), ('backbone_length', bb_len),
                         ('cable_length', per_len + bb_len)):
            self.checked['cost'] += 1
            if key in s and not _close(s[key], val):
                self.err('cost', f"{key}: reported {s[key]}, recomputed {val:.4f}")
        si = s.get('solve_info', {})
        if si.get('cable_length_limit') is not None and not _le(per_len + bb_len, float(si['cable_length_limit'])):
            self.err('cost', f"cable length {per_len + bb_len:.4f} above the Pareto limit {si['cable_length_limit']}")
        if si.get('cost_limit') is not None and not _le(total, float(si['cost_limit'])):
            self.err('cost', f"cost {total:.4f} above the Pareto limit {si['cost_limit']}")
        for ph in si.get('phases', []):
            if ph.get('status') != 'OPTIMAL':
                gap = ph.get('mip_gap')
                self.warn('status', f"{ph.get('objective_name')}: {ph.get('status')}" + (f", gap {100 * gap:.3f} %" if gap is not None else ''))

    # ------------------------------------------------------------------------------------------------- run
    def run(self):
        self.load()
        self.parse()
        self.check_placement()
        self.check_partitions()
        self.check_hardware_rules()
        self.check_peripherals()
        self.build_flows()
        self.check_backbone()
        self.check_chains()
        self.check_failures()
        self.check_costs()
        return self

    def summary(self):
        head = f"[check] {os.path.relpath(self.path, REPO) if self.path.startswith(REPO) else self.path}: "
        head += ("OK" if not self.errors else f"{len(self.errors)} ERROR(S)") + \
                f" ({sum(self.checked.values())} checks in {len(self.checked)} rules, {len(self.warnings)} warning(s))"
        lines = [head]
        lines += [f"  error   [{r}] {m}" for r, m in self.errors[:200]]
        if len(self.errors) > 200:
            lines.append(f"  ... {len(self.errors) - 200} more errors")
        lines += [f"  warning [{r}] {m}" for r, m in self.warnings[:50]]
        return '\n'.join(lines)


def check_path(path, config_dir=None, architecture=None):
    """SolutionCheck objects of a solution file or of every solution_<i>.json in a run directory"""
    files = sorted(glob.glob(os.path.join(path, 'solution_*.json')), key=lambda p: int(re.findall(r'(\d+)\.json$', p)[0])) \
        if os.path.isdir(path) else [path]
    files = [f for f in files if re.search(r'solution_\d+\.json$', f)]
    return [SolutionCheck(f, config_dir, architecture).run() for f in files]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='solution_<i>.json or a run directory')
    ap.add_argument('--config_dir', default=None, help='default: the run\'s config dir (run_info.json), else configs')
    ap.add_argument('--architecture', default=None, help='default: the run\'s architecture JSON (run_info.json)')
    args = ap.parse_args()
    checks = check_path(args.path, args.config_dir, args.architecture)
    if not checks:
        print(f"[check] no solution files in {args.path}")
        return 1
    for c in checks:
        print(c.summary())
    return 1 if any(c.errors for c in checks) else 0


if __name__ == '__main__':
    sys.exit(main())
