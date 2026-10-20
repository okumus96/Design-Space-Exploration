#!/usr/bin/env python3
"""
Independent check of optimisation results.

  dse/bin/python tools/check_solution.py results/<run>                     (every solution_<i>.json of the run)
  dse/bin/python tools/check_solution.py results/<run>/solution_1.json [--config_dir configs] [--architecture <json>]

Re-evaluates every rule of the model on a stored solution with plain Python and recomputes every cost and length:
placement, partitions and their capacities, control-unit caps, hardware, redundancy, AI contention, peripheral
wiring, ports and shared-bus trunks, backbone cables and paths, link capacities, chain latencies, and with single
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
                m *= 1.0 + m_asil + m_ai
        if self.flags.get('growth'):
            g = self.unc.get('growth', {})
            m *= 1.0 + self.alpha * float(g.get('profiles', {}).get(dom, g.get('default_profile', 0.0)))
        return m

    def demand(self, sc_id, res):
        sc = self.scs[sc_id]
        return sc[res] * self.mult(sc)

    def derate(self, site):
        return max(0.0, self.site_derating.get(site, 1.0)) if self.flags.get('derating') else 1.0

    # --------------------------------------------------------------------------------------------------- loading
    def load(self):
        cfg = lambda f: _load(os.path.join(self.config_dir, f))
        buses = cfg('buses.json')
        self.ct = {n: {'cost': float(s['cost_per_meter']), 'cap': float(s.get('capacity', float('inf'))),
                       'port': float(s.get('port_cost', 0)), 'hop': float(s.get('hop_latency_ms', 0.0))}
                   for n, s in buses['cable_types'].items()}
        self.k_paths = max(1, int(buses.get('backbone', {}).get('candidate_paths', 3)))
        parts = cfg('partitions.json')
        self.part, self.site_cfg = parts.get('partition', {}), parts.get('site', {})
        self.hw_price = {h: float(s.get('cost', 0)) for h, s in cfg('hardwares.json').get('hardware_costs', {}).items()}
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
                  'rate': float(c.get('rate_hz', 10.0) or 10.0), 'type': c.get('type', ''),
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
        fcfg = self.unc.get('failures', {})
        self.asil_min = int(fcfg.get('critical_asil_min', 3))
        self.recovery = float(fcfg.get('recovery_ms', 50.0))
        self.classes = set(fcfg.get('classes', ['cell', 'switch', 'cable']))
        self.crit = {s for s, sc in self.scs.items() if sc['asil'] >= self.asil_min} if self.flags.get('failures') else set()
        rl = self.unc.get('robust_latency', {})
        self.lat = {'robust': bool(self.flags.get('robust_latency')), 'factor': float(rl.get('network_safety_factor', 1.0)),
                    'sync': float(rl.get('sync_ms', 0.0)), 'platform': float(rl.get('platform_ms', 0.0))}
        self.has_switch_hw = 'SWITCH' in self.hw_price

    # ------------------------------------------------------------------------------------------- solution parsing
    def parse(self):
        s = self.sol
        self.pos_x = dict(s.get('assignment', {}))
        self.part_of = dict(s.get('partitions', {}))
        self.standby = dict(s.get('standby') or {})
        self.pos_r = {sc: sb['cell'] for sc, sb in self.standby.items()}
        self.cold = {sc for sc, sb in self.standby.items() if sb.get('type') == 'cold'}
        self.hw = set()
        for item in s.get('hw_features', []):
            h, site = item.rsplit('@', 1)
            self.hw.add((site, h))
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
        self.shared_attach = dict(s.get('shared_sensor_attachments', {}))
        self.shared_attach.update(s.get('shared_actuator_attachments', {}))
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
                if not _le(load, cap * h):
                    self.err('partition_capacity', f"{p} {r} {load:.1f} > {cap * h:.1f}")
        for s in self.active:
            h = self.derate(s)
            for r in RES:
                cap = self.site_cfg.get(f'{r}_cap')
                if cap is None:
                    continue
                self.checked['site_capacity'] += 1
                load = sum(self.demand(sc, r) for sc, site in self.pos_x.items() if site == s)
                load += sum(self.demand(sc, r) for sc, site in self.pos_r.items() if site == s and sc not in self.cold)
                if not _le(load, float(cap) * h):
                    self.err('site_capacity', f"{s} {r} {load:.1f} > {float(cap) * h:.1f} (cell cap x derating)")

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
        if self.flags.get('ai_contention'):
            cfg = self.unc.get('ai_contention', {})
            per_site = defaultdict(int)
            for sc, site in self.pos_x.items():
                if cfg.get('ai_hw_feature') in self.scs.get(sc, {}).get('hw', []):
                    per_site[site] += 1
            for site, n in per_site.items():
                self.checked['ai_contention'] += 1
                if n > int(cfg['max_per_location']):
                    self.err('ai_contention', f"{site}: {n} components need {cfg['ai_hw_feature']} > {cfg['max_per_location']}")

    # -------------------------------------------------------------------------------------------- peripherals
    def check_peripherals(self):
        users = defaultdict(list)           # device -> [(sc, site)] of the primaries driving it
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
            elif dev['iface'] in SHARED_BUSES:
                att = self.shared_attach.get(d)
                for sc, site in users.get(d, []):
                    self.checked['peripheral'] += 1
                    if att != site:
                        self.err('peripheral', f"{dev['iface']} device {d} driven by {sc} at {site} but attached at {att}")
                if att is not None and att not in self.node_active:
                    self.err('peripheral', f"{d} attached at {att} without a control unit or switch")
        for d in list(self.attach) + list(self.shared_attach):
            if d not in self.devices:
                self.err('peripheral', f"unknown device {d} attached")
        # ports per site and interface
        bb_ports = defaultdict(int)
        for (a, b, n), cnt in self.cables.items():
            bb_ports[(a, n)] += cnt
            bb_ports[(b, n)] += cnt
        groups = defaultdict(set)           # (site, shared bus) -> ASIL groups of the components using it
        direct = defaultdict(int)           # (site, iface) -> peripherals wired directly to the hosting unit
        for sc, site in self.pos_x.items():
            for d in self.scs.get(sc, {}).get('devices', []):
                n = self.devices[d]['iface']
                if n in SHARED_BUSES:
                    groups[(site, n)].add('HIGH' if self.scs[sc]['asil'] >= 3 else 'LOW')
                elif n != 'ETH':
                    direct[(site, n)] += 1
        sites = set(self.cands)
        for s in sites:
            for n, t in self.ct.items():
                have = self.ports.get((s, n), 0)
                if n == 'ETH':
                    need = sum(1 for d, a in self.attach.items() if a == s and self.devices.get(d, {}).get('iface') == 'ETH') + bb_ports[(s, n)]
                    cover = have + (SWITCH_ETH_PORTS if (self.has_switch_hw and s in self.switches) else 0)
                elif n in SHARED_BUSES:
                    need = max(len(groups[(s, n)]), bb_ports[(s, n)])
                    cover = have
                    vol = sum(self.devices[d]['volume'] for d, a in self.shared_attach.items() if a == s and self.devices.get(d, {}).get('iface') == n)
                    if vol > 0 and t['cap'] != float('inf'):
                        self.checked['ports'] += 1
                        if not _le(vol, t['cap'] * have):
                            self.err('ports', f"{s} {n}: shared-bus load {vol} > {have} bus(es) x {t['cap']}")
                else:
                    need = direct[(s, n)] + bb_ports[(s, n)]
                    cover = have
                if need > 0 or have > 0:
                    self.checked['ports'] += 1
                if need > cover:
                    self.err('ports', f"{s} {n}: {need} ports needed, {cover} provided")
        # shared-bus trunks: the main segment reaches the farthest attached device; a second (ASIL-split) bus
        # repeats it when both ASIL groups use the bus at that site
        trunks = self.sol.get('shared_trunks')
        if trunks is None:
            self.warn('format', 'shared_trunks not in the solution: minimum trunk lengths assumed')
        self.trunk_len = {}
        for s in sites:
            for n in SHARED_BUSES:
                far = max((self.dist(d, s) for d, a in self.shared_attach.items() if a == s and self.devices.get(d, {}).get('iface') == n), default=None)
                split = len(groups[(s, n)]) == 2
                if trunks is None:
                    main = far or 0.0
                    extra = main if split else 0.0
                else:
                    t = trunks.get(f"{s}|{n}", {'main': 0.0, 'extra': 0.0})
                    main, extra = float(t['main']), float(t['extra'])
                if far is None:
                    if main > 1e-6 or extra > 1e-6:
                        self.err('trunk', f"{s} {n}: trunk {main:.3f} + {extra:.3f} m without an attached device")
                else:
                    self.checked['trunk'] += 1
                    if main < far - 1e-6:
                        self.err('trunk', f"{s} {n}: trunk {main:.3f} m shorter than the farthest device {far:.3f} m")
                    if split and extra < main - 1e-6:
                        self.err('trunk', f"{s} {n}: ASIL-split bus without its second trunk ({extra:.3f} < {main:.3f} m)")
                    if not split and extra > 1e-6:
                        self.err('trunk', f"{s} {n}: second trunk {extra:.3f} m without an ASIL split")
                if main > 1e-9 or extra > 1e-9:
                    self.trunk_len[(s, n)] = (main, extra)

    # ---------------------------------------------------------------------------------------------- backbone
    def build_flows(self):
        base = []
        for dst in self.sc_order:
            for src in self.scs[dst]['receives_from']:
                if src not in self.scs:
                    continue
                s = self.scs[src]
                volume, rate = 10.0, s['rate']
                if s['devices']:
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
                f['iface'], f['path'], f['known'] = None, None, True
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
            elig = [m for m in caps if f['volume'] <= caps[m]] or [max(caps, key=caps.get)]
            if n not in elig:
                self.err('route', f"{name}: {n} cannot carry {f['volume']}")
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
    def crossing_ms(self, f):
        hops = len(f['path']) - 1
        t = self.ct[f['iface']]
        base = hops * (f['kbit'] / t['cap'] + t['hop'])
        return base * self.lat['factor'] + self.lat['sync'] + self.lat['platform'] if self.lat['robust'] else base

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
                total += self.crossing_ms(f)
            if unknown:
                continue
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
            if kind == 'cable' and elem[2] == f['iface'] and any(_pk(a, b) == elem[:2] for a, b in zip(path, path[1:])):
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

        chain_flows = {ch['id']: self.chain_flows(ch) for ch in self.chains}
        for kind, elem in scen:
            label = f"{kind} {elem if kind != 'cable' else '-'.join(elem[:2]) + ' ' + elem[2]}"
            for sc in sorted(self.crit):
                self.checked['failure_survival'] += 1
                if not (served(sc, 'x', kind, elem) or served(sc, 'r', kind, elem)):
                    self.err('failure_survival', f"{label}: {sc} has no instance with all critical partners")
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
                        total += self.crossing_ms(pick)
                if broken:
                    continue
                self.checked['degraded_latency'] += 1
                bound = float(ch.get('degraded_bound_ms', 2.0 * float(ch['net_budget_ms'])))
                if not _le(total, bound):
                    self.err('degraded_latency', f"{label}: {ch['id']} {total:.3f} ms > degraded bound {bound} ms")
            if kind == 'cell':
                self._failure_capacity(elem, label)
            # link capacity after the failure: the flows that keep running (alive, primaries and hot standbys) and,
            # per data flow, its first surviving variant (the variant rule of the degraded latency), which activates
            # the cold standbys it needs, against the remaining cables
            carried = {id(f): f for f in self.flows if not self.silent(f) and alive(f, kind, elem)}
            for b in range(len(self.base)):
                pick = next((self.variants[b][k] for k in VARIANT_ORDER[self.base[b]['type']]
                             if k in self.variants[b] and alive(self.variants[b][k], kind, elem)), None)
                if pick is not None:
                    carried.setdefault(id(pick), pick)
            dl = defaultdict(float)
            for f in carried.values():
                if f['path'] is None:
                    continue
                for a, c2 in zip(f['path'], f['path'][1:]):
                    dl[(a, c2, f['iface'])] += f['volume']
            for (a, c2, n), ld in dl.items():
                self.checked['failure_link_capacity'] += 1
                cap = self.ct[n]['cap'] * self.cables.get(_pk(a, c2) + (n,), 0)
                if not _le(ld, cap):
                    self.err('failure_link_capacity', f"{label}: {a}->{c2} {n} load {ld:.1f} > {cap:.1f} after the failure")

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
                if not _le(load, float(cap) * h):
                    self.err('failure_capacity', f"{label}: {j} {r} {load:.1f} > {float(cap) * h:.1f} with the activated standbys")
            for a in sorted({self.scs[sc]['asil'] for sc in lst}):
                group = [sc for sc in prim + hot_here + lst if self.scs[sc]['asil'] == a]
                n = self.P.get((j, a), 0)
                caps = {r: float(self.part.get(f'{r}_cap', 0) or 0) * h for r in RES}
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
        hw_cost = sum(self.hw_price.get(h, 0.0) for (_site, h) in self.hw)
        for (_site, h) in self.hw:
            if h not in self.hw_price:
                self.err('cost', f"hardware {h} has no price in hardwares.json")
        if_cost = sum(cnt * self.ct[n]['port'] for (_site, n), cnt in self.ports.items())
        per_len, per_cost = 0.0, 0.0
        for sc, site in self.pos_x.items():
            for d in self.scs.get(sc, {}).get('devices', []):
                n = self.devices[d]['iface']
                if n == 'ETH' or n in SHARED_BUSES:
                    continue
                dl = self.dist(d, site)
                per_len += dl
                per_cost += dl * self.ct[n]['cost']
        for d, site in list(self.attach.items()) + list(self.shared_attach.items()):
            if d not in self.devices:
                continue
            dl = self.dist(d, site)
            per_len += dl
            per_cost += dl * self.ct[self.devices[d]['iface']]['cost']
        for (site, n), (main, extra) in self.trunk_len.items():
            per_len += main + extra
            per_cost += (main + extra) * self.ct[n]['cost']
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
