"""
Graph of an instance for the learned search (HANDOFF 16.25). Node types: comp (software component), cell (candidate
site), dev (sensor / actuator). Edge types: comp-comp messages, comp-dev driver relation, dev-cell and cell-cell cable
lengths, and the complete comp-cell candidate edges on which placements are scored.

Static features come from the same rules as the model at the instance's robustness level rho (demand multiplier,
derating, fail-operational rule), normalised by the partition capacities. The state of the search (the current design,
the components freed in the last iteration, the progress of the episode) is appended as dynamic features; without a
design they are zero. Deliberately left out: the domain label of a component and the zone label of a cell, so no
grouping by domain or zone can be learned from names; the domain acts only through the demand growth it implies.

The static part is built once per instance (InstanceGraph); a state graph only adds the dynamic tensors (vectorised),
so an iteration of the search stays cheap at thousands of components.
"""
import json
import math
import os
import sys

import torch
from torch_geometric.data import HeteroData

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from architecture_loader import load_architecture_from_json   # noqa: E402
from config_reader import ConfigReader                        # noqa: E402
from optimizer import UNCERTAINTY_BLOCKS                      # noqa: E402
from optimizer_utils import get_distance, is_fail_operational, site_derating, sw_multiplier   # noqa: E402

IFACES = ('ETH', 'CAN', 'LIN', 'FLEXRAY', 'SPI', 'UART')
LEN_SCALE = 5.0   # m, about the vehicle length
COMP_DYN, CELL_DYN, CAND_DYN = 5, 5, 2   # dynamic feature counts (see InstanceGraph.data)


def _lg(v):
    return math.log1p(max(0.0, float(v)))


def load_instance(cfg_dir, arch, rho, num_locs=None):
    """The instance as the optimiser sees it at robustness level rho (all uncertainty blocks on, as in main.py)."""
    cr = ConfigReader(cfg_dir)
    scs, comm, sensors, actuators, cable_types, locations = load_architecture_from_json(arch, config_reader=cr, num_locs=num_locs)
    unc = {k: True for k in UNCERTAINTY_BLOCKS}
    unc['params'] = cr.get_uncertainty_config()
    unc['rho'] = float(rho)
    if rho <= 0.0:   # as the robustness sweep: rho = 0 has no failure tolerance and no separation
        unc['failures'] = unc['redundancy'] = False
    return {'cr': cr, 'scs': scs, 'comm': comm, 'sensors': sensors, 'actuators': actuators, 'locations': locations,
            'unc': unc, 'rho': float(rho), 'chains': cr.get_chains(), 'partitions': cr.get_partitions(),
            'hw_cap': cr.get_hardware_capacities(), 'hardwares': list(cr.get_hardwares().keys())}


class InstanceGraph:
    """Static graph of one instance plus what the dynamic features need (effective demands, cell capacities, traffic)."""

    def __init__(self, inst):
        scs, locs, unc, rho = inst['scs'], inst['locations'], inst['unc'], inst['rho']
        part = inst['partitions']
        caps = {r: float(part.get(f'{r}_cap') or 1.0) for r in ('cpu', 'ram', 'rom')}
        hw = inst['hw_cap']
        tops_unit = float(next(iter(hw.values()))['tops']) if hw else 1.0
        tops_units = int(next(iter(hw.values())).get('max_units', 1)) if hw else 1
        fail_cfg = unc['params'].get('failures', {})
        fail_on = bool(unc.get('failures'))
        devices = list(inst['sensors']) + list(inst['actuators'])
        dev_kind = ['sensor'] * len(inst['sensors']) + ['actuator'] * len(inst['actuators'])
        dev_idx = {d.id: k for k, d in enumerate(devices)}
        sc_idx = {sc.id: i for i, sc in enumerate(scs)}
        glob = [rho, float(fail_on), len(locs) / 33.0, len(scs) / 100.0]
        self.comp_ids = [sc.id for sc in scs]
        self.cell_ids = [l.id for l in locs]
        self.comp_index, self.cell_index = sc_idx, {c: j for j, c in enumerate(self.cell_ids)}
        self.critical = [bool(fail_on and is_fail_operational(sc, fail_cfg)) for sc in scs]

        # chain context per component and per consecutive pair
        budget, ftti, pair_budget = {}, {}, {}
        for ch in inst['chains']:
            ids = [s for s in ch.get('swcs', []) if s in sc_idx]
            b, f = float(ch.get('net_budget_ms', 1e3)), float(ch.get('ftti_ms', 1e4))
            for s in ids:
                budget[s], ftti[s] = min(budget.get(s, 1e3), b), min(ftti.get(s, 1e4), f)
            for a, c in zip(ids, ids[1:]):
                pair_budget[a, c] = min(pair_budget.get((a, c), 1e3), b)
        msgs = [(sc_idx[m['src']], sc_idx[m['dst']], float(m['volume'])) for m in inst['comm']
                if m['src'] in sc_idx and m['dst'] in sc_idx and m['src'] != m['dst']]
        traffic_out, traffic_in = [0.0] * len(scs), [0.0] * len(scs)
        for a, b, v in msgs:
            traffic_out[a] += v
            traffic_in[b] += v

        comp_x, demand = [], []
        for i, sc in enumerate(scs):
            mult = sw_multiplier(sc, unc)
            d = [sc.cpu_req * mult / caps['cpu'], sc.ram_req * mult / caps['ram'], sc.rom_req * mult / caps['rom'],
                 float(sc.tops_req or 0.0) * mult / tops_unit]
            demand.append(d)
            asil = [0.0] * 5
            asil[max(0, min(4, int(sc.asil_req)))] = 1.0
            hwf = [float(h in (sc.hw_required or [])) for h in inst['hardwares']]
            drv = [float(bool(sc.sensors)), float(bool(sc.actuators))]
            comp_x.append(d + [mult - 1.0] + asil + hwf + drv +
                          [float(self.critical[i]), float(bool(sc.redundant_with)), _lg(sc.rate_hz) / 5.0,
                           _lg(traffic_out[i]) / 5.0, _lg(traffic_in[i]) / 5.0, float(sc.id in budget),
                           _lg(budget.get(sc.id, 1e3)) / 7.0, _lg(ftti.get(sc.id, 1e4)) / 10.0] + glob)
        self.demand = torch.tensor(demand, dtype=torch.float)          # per comp: cpu, ram, rom (partition units), tops (units)
        site = {r: part.get(f'site_{r}_cap') for r in ('cpu', 'ram', 'rom')}
        self.cell_cap = torch.tensor([[(float(site['cpu']) / caps['cpu'] if site['cpu'] else 2.0) * site_derating(l, unc),
                                       (float(site['ram']) / caps['ram'] if site['ram'] else 2.0),
                                       tops_units * site_derating(l, unc)] for l in locs], dtype=torch.float)
        # for the repair of a predicted placement (learn/search.py, not features): every resource per cell in
        # partition / hardware units, the ASIL of every component, the standbys that must be hot (chain FTTI below
        # the recovery time) and the partition limits of a cell
        self.cap_units = torch.tensor([[(float(site['cpu']) / caps['cpu'] if site['cpu'] else 2.0) * site_derating(l, unc),
                                        (float(site['ram']) / caps['ram'] if site['ram'] else 2.0),
                                        (float(site['rom']) / caps['rom'] if site['rom'] else 2.0),
                                        tops_units * site_derating(l, unc)] for l in locs], dtype=torch.float)
        self.asil = [max(0, min(4, int(sc.asil_req))) for sc in scs]
        recovery = float(fail_cfg.get('recovery_ms', 50.0))
        self.hot_forced = [self.critical[i] and ftti.get(sc.id, float('inf')) < recovery for i, sc in enumerate(scs)]
        self.redundant = [(i, sc_idx[sc.redundant_with]) for i, sc in enumerate(scs)
                          if sc.redundant_with in sc_idx and sc_idx[sc.redundant_with] != i]   # pairs that must be apart
        self.chains = [[sc_idx[s] for s in ch.get('swcs', []) if s in sc_idx] for ch in inst['chains']]   # members, in order
        self.max_per_asil = int(part.get('max_per_asil_per_site', 3))
        self.site_max_partitions = int(part['site_max_partitions']) if part.get('site_max_partitions') else None
        self.msg = torch.tensor([[a, b] for a, b, _ in msgs], dtype=torch.long).t() if msgs else torch.zeros((2, 0), dtype=torch.long)
        self.msg_vol = torch.tensor([v for _, _, v in msgs], dtype=torch.float)
        self.traffic = torch.tensor(traffic_out, dtype=torch.float) + torch.tensor(traffic_in, dtype=torch.float)

        cell_x = [[l.location.x / LEN_SCALE, l.location.y / LEN_SCALE, l.location.z / LEN_SCALE, site_derating(l, unc)] + glob
                  for l in locs]
        dev_x = [[float(getattr(d, 'interface', '') == n) for n in IFACES] + [float(k == 'sensor'), _lg(d.volume) / 5.0,
                 _lg(getattr(d, 'rate_hz', 10.0)) / 5.0, d.location.x / LEN_SCALE, d.location.y / LEN_SCALE,
                 d.location.z / LEN_SCALE] + glob for d, k in zip(devices, dev_kind)]
        self.x = {'comp': torch.tensor(comp_x, dtype=torch.float), 'cell': torch.tensor(cell_x, dtype=torch.float),
                  'dev': torch.tensor(dev_x, dtype=torch.float) if dev_x else torch.zeros((0, len(IFACES) + 6 + len(glob)))}

        self.edges = {}

        def put(rel, src, dst, attr, width):
            self.edges[rel] = (torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long),
                               torch.tensor(attr, dtype=torch.float) if src else torch.zeros((0, width)))

        s, d, a = [], [], []
        for m in inst['comm']:
            if m['src'] in sc_idx and m['dst'] in sc_idx and m['src'] != m['dst']:
                pb = pair_budget.get((m['src'], m['dst']))
                for u, v, fwd in ((sc_idx[m['src']], sc_idx[m['dst']], 1.0), (sc_idx[m['dst']], sc_idx[m['src']], 0.0)):
                    s.append(u); d.append(v)
                    a.append([_lg(m['volume']) / 5.0, _lg(m['kbit']) / 5.0, float(pb is not None), _lg(pb or 1e3) / 7.0, fwd])
        put(('comp', 'msg', 'comp'), s, d, a, 5)
        s, d, a = [], [], []
        for i, sc in enumerate(scs):
            for dv in list(sc.sensors or []) + list(sc.actuators or []):
                if dv in dev_idx:
                    s.append(i); d.append(dev_idx[dv]); a.append([_lg(devices[dev_idx[dv]].volume) / 5.0])
        put(('comp', 'drives', 'dev'), s, d, a, 1)
        put(('dev', 'driven_by', 'comp'), d, s, a, 1)
        s, d, a = [], [], []
        for k, dv in enumerate(devices):
            for j, l in enumerate(locs):
                s.append(k); d.append(j); a.append([get_distance(dv.location, l.location) / LEN_SCALE])
        put(('dev', 'wire', 'cell'), s, d, a, 1)
        put(('cell', 'wire_rev', 'dev'), d, s, a, 1)
        s, d, a = [], [], []
        for j, l in enumerate(locs):
            for k, l2 in enumerate(locs):
                if j != k:
                    s.append(j); d.append(k); a.append([get_distance(l.location, l2.location) / LEN_SCALE])
        put(('cell', 'link', 'cell'), s, d, a, 1)
        s, d, a = [], [], []
        for i, sc in enumerate(scs):
            devs = [devices[dev_idx[dv]] for dv in list(sc.sensors or []) + list(sc.actuators or []) if dv in dev_idx]
            for j, l in enumerate(locs):
                s.append(i); d.append(j); a.append([sum(get_distance(dv.location, l.location) for dv in devs) / LEN_SCALE, float(bool(devs))])
        put(('comp', 'cand', 'cell'), s, d, a, 2)
        put(('cell', 'cand_rev', 'comp'), d, s, a, 2)

    def cells_of(self, placement):
        """index tensor (per comp) of the cells of a {comp_id: cell_id} placement, -1 where missing"""
        return torch.tensor([self.cell_index.get(placement.get(c), -1) for c in self.comp_ids], dtype=torch.long)

    def data(self, design=None, freed=None, it_frac=0.0, label=None):
        """HeteroData of the state: static features plus the dynamic ones of design {'x': {comp: cell}, 'r': {comp:
        cell}} (cell loads, the traffic share leaving a component's cell, the components freed last, episode progress).
        With label (assignment / standby of a good design) the targets y_cell, y_standby and, with a design, y_moved
        (component sits elsewhere than in the good design)."""
        n, m = len(self.comp_ids), len(self.cell_ids)
        comp_dyn, cell_dyn = torch.zeros((n, COMP_DYN)), torch.zeros((m, CELL_DYN))
        cur = r_cur = None
        if design is not None:
            cur = self.cells_of(design.get('x', {}))
            r_cur = self.cells_of(design.get('r', {}))
            ok = cur >= 0
            load = torch.zeros((m, 3))
            load.index_add_(0, cur[ok], self.demand[ok][:, [0, 1, 3]])
            util = load / self.cell_cap.clamp(min=1e-6)
            cnt = torch.bincount(cur[ok], minlength=m).float()
            cell_dyn = torch.cat([(cnt > 0).float().unsqueeze(1), util, (cnt / 20.0).unsqueeze(1)], dim=1)
            cut = torch.zeros(n)
            if self.msg.shape[1]:
                a, b = self.msg
                crossing = ((cur[a] != cur[b]) & (cur[a] >= 0) & (cur[b] >= 0)).float() * self.msg_vol
                cut.index_add_(0, a, crossing)
                cut.index_add_(0, b, crossing)
            fr = torch.zeros(n)
            for c in (freed or ()):
                if c in self.comp_index:
                    fr[self.comp_index[c]] = 1.0
            here = torch.where(cur.unsqueeze(1) >= 0, util[cur.clamp(min=0)], torch.zeros((n, 3)))
            comp_dyn = torch.stack([fr, cut / self.traffic.clamp(min=1e-9), here[:, 0], here[:, 2],
                                    torch.full((n,), float(it_frac))], dim=1)
        data = HeteroData()
        data['comp'].x = torch.cat([self.x['comp'], comp_dyn], dim=1)
        data['cell'].x = torch.cat([self.x['cell'], cell_dyn], dim=1)
        data['dev'].x = self.x['dev']
        for rel, (ei, ea) in self.edges.items():
            data[rel].edge_index = ei
            if rel in (('comp', 'cand', 'cell'), ('cell', 'cand_rev', 'comp')):
                src_c = ei[0] if rel[0] == 'comp' else ei[1]
                dst_c = ei[1] if rel[0] == 'comp' else ei[0]
                dyn = torch.zeros((ei.shape[1], CAND_DYN))
                if cur is not None:
                    dyn[:, 0] = (cur[src_c] == dst_c).float()
                    dyn[:, 1] = (r_cur[src_c] == dst_c).float()
                ea = torch.cat([ea, dyn], dim=1)
            data[rel].edge_attr = ea
        data.comp_ids, data.cell_ids = self.comp_ids, self.cell_ids
        if label is not None:
            y = self.cells_of(label['assignment'])
            data['comp'].y_cell = y
            data['comp'].y_standby = self.cells_of({c: s['cell'] for c, s in (label.get('standby') or {}).items()})
            data['comp'].y_moved = ((cur != y) & (y >= 0)).float() if cur is not None else torch.zeros(n)
        return data


def load_variant(inst_dir):
    """(InstanceGraph, label) of a solved, checked training variant (learn/gen_data.py); None without a checked design."""
    with open(os.path.join(inst_dir, 'label.json')) as f:
        label = json.load(f)
    if not label.get('check_ok'):
        return None
    ig = InstanceGraph(load_instance(os.path.join(inst_dir, 'cfg'), os.path.join(inst_dir, 'arch.json'), label['rho']))
    ig.variant = os.path.basename(inst_dir)
    return ig, label
