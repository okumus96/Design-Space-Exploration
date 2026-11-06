#!/usr/bin/env python3
"""
Result figure of one synthesised design (HANDOFF 16.26, user 2026-11-02): the emergent architecture at a glance.
  panel A (top left): top view of the car (the render top_raw.png of tools/vehicle_viz/build_vehicle.py, faded), with
          the used ECUs drawn as boxes coloured by their emergent type, every routed cable by type (from
          <solution>_cables.json of tools/vehicle_layout/solution_paths.py) and the devices (sensor = circle,
          actuator = square); the car is only context
  panel B (right, only with --panel_b): deployment per ECU as a grid: one column per used ECU (header in the role colour with the
          accelerator / switch glyphs and the role name), the hosted components as a stacked bar (one segment per
          partition, fill = ASIL level), then the measures behind the role as bars with the threshold tick: largest
          functional domain, CPU load of the unit, nearby devices it connects to (of the devices in its zone), bus lines. Backbone links,
          bus lines and ports are printed as text tables for the paper, not drawn.
ECU types (a fixed rule applied to the result, stated in the legend): compute-heavy = an accelerator on the unit or
CPU load >= COMPUTE_CPU of the unit's cap; I/O-heavy (zonal) = connected to at least IO_LOCAL nearby devices; central compute = compute-heavy only, zonal compute = both, zonal I/O = I/O-heavy only,
small controller = neither.
Every number comes from the checker's reading of the solution (tools/check_solution.py).
  dse/bin/python tools/vehicle_viz/design_figure.py <solution_1.json> --config_dir <cfg> --architecture <arch.json>
      --render_dir <dir with top_raw.png and top_anchors.json> --out <file.pdf> [--title "..."]
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, Circle   # noqa: E402
from matplotlib import image as mpimg          # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tools'))
from check_solution import SolutionCheck, RES, PART_RE   # noqa: E402
from config_reader import ConfigReader                    # noqa: E402
from architecture_loader import load_architecture_from_json   # noqa: E402

# ECU roles (HANDOFF 16.26, 2026-11-02): the node roles of the E/E literature (vehicle computer / central compute,
# domain controller, zone controller with and without compute, edge ECU), assigned by three measured axes:
#   compute  = accelerator on the unit or CPU load >= COMPUTE_CPU of the unit cap
#   I/O      = the unit is connected to >= IO_LOCAL nearby devices: devices whose nearest used ECU it is (straight line
#              in the top view) and whose software it hosts or whose bus line it sits on (user 2026-11-03: a zone
#              controller must take the devices of its own area; hosted devices from elsewhere do not count)
#   domain   = the largest functional domain among the unit's primaries holds >= DOMAIN_SHARE of them
#              (ADAS and Sensing count as one domain, 'automated driving'; Platform software is left out)
# compute & domain -> domain controller; compute & I/O -> zone controller with compute; compute alone -> central
# compute; I/O alone -> zone controller; neither -> edge ECU.
COMPUTE_CPU, IO_LOCAL, DOMAIN_SHARE = 0.40, 4, 0.50
DOMAIN_GROUP = {'ADAS': 'automated driving', 'Sensing': 'automated driving', 'Platform': None}
TYPES = ['central compute', 'domain controller', 'zone controller + compute', 'zone controller', 'edge ECU']
# ECU colours (user 2026-11-06: the roles are taken out of the figure; every ECU its own colour, its endpoints in the
# same colour): the reference categorical hues in their fixed order, blue last (blue = the partitions and ETH);
# ECUs beyond eight are grey; every ECU and endpoint also carries its E-label
ECU_COLOURS = ['#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#7b3f00', '#6250d6', '#e34948', '#2a78d6']   # 2026-11-06: slot 5 dark brown, the green was too close to the aqua (user)
TYPE_COLOUR = {'central compute': '#eb6834', 'domain controller': '#6250d6', 'zone controller + compute': '#1baf7a',
               'zone controller': '#eda100', 'edge ECU': '#e87ba4'}
# ASIL = an ordinal level -> one hue, light to dark (dataviz palette, blue ramp steps 150 .. 550)
ASIL_FILL = {0: '#b7d3f6', 1: '#86b6ef', 2: '#5598e7', 3: '#2a78d6', 4: '#1c5cab'}
ASIL_TEXT = {0: '#0b0b0b', 1: '#0b0b0b', 2: '#0b0b0b', 3: '#ffffff', 4: '#ffffff'}
# cable / interface types = the colours of the 3-D render (tools/vehicle_viz/build_vehicle.py CABLE_COLOUR)
CABLE_COLOUR = {'ETH': '#1461fa', 'CAN': '#0db333', 'LIN': '#f2cc14', 'FLEXRAY': '#8c2ee0', 'UART': '#995c1a', 'SPI': '#149485'}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#dedcd6'
# hardware features on a unit: small tabs on the ECU box in panel A (user 2026-11-03: tabs, not symbols)
HW_ORDER = ['HW_ACC', 'SWITCH', 'DSP', 'HSM']
HW_COLOUR = {'HW_ACC': '#0b0b0b', 'SWITCH': '#1461fa', 'DSP': '#7d3c98', 'HSM': '#ffd166'}   # 2026-11-04: DSP violet, HSM gold (grey / white were not told apart); 2026-11-06: switch in the ETH blue (the GPU modules use the grey ramp)
HW_NAME = {'HW_ACC': 'accelerator (GPU)', 'SWITCH': 'Ethernet switch', 'DSP': 'DSP', 'HSM': 'security module'}
BAR_FILL = '#52514e'
# partition types and GPU modules (user 2026-11-06, second version: 'like a heatmap, darker as it grows'): one
# single-hue ramp each, light = small, dark = large, ordered by the catalogue (partition types by CPU, GPU modules by
# TOPS), so the colour of a type is the same in every figure of the same catalogue; partitions on the blue ramp (steps
# 150 .. 700), GPU modules on a grey ramp ending in black; neighbouring steps >= 14 OKLab x100 for normal vision and
# under protan / deutan / tritan simulation, the GPU greys >= 16 / 8 from the other hardware squares; the type name is
# written on every partition box, the legend lists the types small to large with their capacity
PART_RAMP = ['#b7d3f6', '#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b']
GPU_RAMP = ['#b5b4ae', '#6b6a65', '#0b0b0b']


def _ramp(ids, ramp):
    """the ids (small to large) spread evenly over the ramp, the first id on its lightest step and the last on its darkest"""
    n = len(ids)
    return {k: ramp[round(i * (len(ramp) - 1) / (n - 1))] if n > 1 else ramp[-1] for i, k in enumerate(ids)}


def _text_on(fill):
    """ink or white text on a fill, by its relative luminance"""
    r, g, b = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return '#0b0b0b' if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else '#ffffff'


def ptype_order(c):
    """the catalogue's partition types, smallest CPU first ([] without partition types)"""
    return sorted(c.types or {}, key=lambda k: (float(c.types[k].get('cpu', 0) or 0), k)) if c.types else []


def gpu_order(c):
    """the catalogue's GPU module types as (id, TOPS), smallest first ([] without GPU module types)"""
    tys = (getattr(c, 'hw_cap', {}).get('HW_ACC') or {}).get('types') or []
    return sorted(((str(t['id']), float(t['tops'])) for t in tys), key=lambda q: (q[1], q[0]))


def part_colours(c, p):
    """(fill, text colour) of a partition box: its partition type's ramp step, else (single-partition model) its ASIL level's"""
    if p.get('ptype'):
        fill = _ramp(ptype_order(c), PART_RAMP).get(p['ptype'], MUTED)
        return fill, _text_on(fill)
    return ASIL_FILL[p['asil']], ASIL_TEXT[p['asil']]


def gpu_modules(c, site):
    """the GPU modules of a cell as a list of type ids, largest first ([] without GPU module types)"""
    mix = getattr(c, 'hw_types', {}).get((site, 'HW_ACC'), {}) or {}
    rank = {k: i for i, (k, _t) in enumerate(gpu_order(c))}
    return [k for k in sorted(mix, key=lambda k: (-rank.get(k, -1), k)) for _ in range(int(mix[k]))]


def hw_glyphs(c, site, feats):
    """the squares of a unit's hardware, in HW_ORDER: the GPU as one square per module in its type's ramp step when the
    modules have types, else one square per feature"""
    fill = _ramp([k for k, _t in gpu_order(c)], GPU_RAMP)
    out = []
    for f in [f for f in HW_ORDER if f in feats] + sorted(f for f in feats if f not in HW_ORDER):
        mods = gpu_modules(c, site) if f == 'HW_ACC' else []
        out += [fill.get(m, HW_COLOUR['HW_ACC']) for m in mods] if mods else [HW_COLOUR.get(f, MUTED)]
    return out


def ptype_handles(c, ecus, suffix=' partition'):
    """legend entries of every partition type of the catalogue, smallest CPU first, with their CPU, unused ones
    marked; [] without partition types"""
    used = {p['ptype'] for parts in ecus.values() for p in parts if p.get('ptype')}
    fill = _ramp(ptype_order(c), PART_RAMP)
    return [Rectangle((0, 0), 1, 1, facecolor=fill.get(k, MUTED), edgecolor='none',
                      label=f"{k}{suffix}, {float(c.types[k].get('cpu', 0) or 0):,.0f} DMIPS" + ('' if k in used else ' (not used)'))
            for k in ptype_order(c)]


def read(solution, config_dir, architecture):
    c = SolutionCheck(solution, config_dir, architecture)
    c.load()
    c.parse()
    # partition types (2026-11-05): the type of every opened partition, as the checker's partition rule reads it
    c.ptype = dict(c.sol.get('partition_types') or {}) if c.types else {}
    return c


def cap_of(c, p, r, site):
    """capacity of partition p in resource r after derating (CPU and TOPS by the cell's thermal class): its partition
    type's capacity with partition types, else the single partition's; 0 when not given"""
    cap = c.part_cap(p, r) if c.types else float(c.part.get(f'{r}_cap', 0) or 0)
    return float(cap or 0.0) * (c.derate(site) if r in ('cpu', 'tops') else 1.0)


def partition_table(c):
    """per ECU: its partitions (sorted by ASIL, name) with level, primaries, hot and cold standbys, utilisation per
    resource (nominal load over the derated cap, as the checker's partition_capacity rule)"""
    members = defaultdict(list)
    for sc, p in c.part_of.items():
        members[p].append((sc, 'x'))
    for sc, sb in c.standby.items():
        if sb.get('partition'):
            members[sb['partition']].append((sc, 'r'))
    ecus = defaultdict(list)
    for p, mem in members.items():
        m = PART_RE.match(p)
        site, asil = m['site'], int(m['asil'])
        h = c.derate(site)
        prim = [sc for sc, inst in mem if inst == 'x']
        hot = [sc for sc, inst in mem if inst == 'r' and sc not in c.cold]
        cold = [sc for sc, inst in mem if inst == 'r' and sc in c.cold]
        util = {}
        for r in c.res:
            cap = cap_of(c, p, r, site)
            if cap > 0:
                util[r] = sum(c.demand(sc, r) for sc in prim + hot) / cap
        ecus[site].append({'name': p, 'asil': asil, 'prim': prim, 'hot': hot, 'cold': cold, 'util': util,
                           'ptype': c.ptype.get(p)})
    for site in ecus:
        ecus[site].sort(key=lambda q: (q['asil'], q['name']))
    return ecus


def ports_of(sol):
    out = defaultdict(lambda: defaultdict(int))
    for s in sol.get('interfaces', []):
        m = re.match(r'(\d+)x([A-Z]+)@(.+)$', s)
        if m:
            out[m.group(3)][m.group(2)] += int(m.group(1))
    return out


def cable_end(c, sol):
    """the unit a device's own cable ends at: ETH devices at their attachment unit (the model lets them attach to
    any active unit and routes the data over the backbone), every other device at the unit hosting its software"""
    end = {d: site for sc, site in c.pos_x.items() for d in c.scs[sc]['devices']}
    for key in ('eth_sensor_attachments', 'eth_actuator_attachments'):
        for d, u in (sol.get(key) or {}).items():
            end[d] = u
    for d, lanes in (sol.get('bus_gateways') or {}).items():   # gateway rule: the first lane's gateway unit
        if lanes:
            end[d] = lanes[sorted(lanes)[0]]
    return end


def local_devices(c, sol, ecus, cells, dev):
    """per used ECU: the devices in its zone (the used ECU nearest to the device, straight line in the top view), how
    many of them it is connected to (their cable ends at it, or it sits on their bus line) and how many devices' software
    it hosts in total"""
    used = {s: cells[s] for s in ecus}
    host = {d: site for sc, site in c.pos_x.items() for d in c.scs[sc]['devices']}
    conn = defaultdict(set)
    for d, h in cable_end(c, sol).items():
        conn[d].add(h)
    for b in sol.get('buses', []):
        for d in b['devices']:
            conn[d].update(b['units'])
    out = {s: {'zone': 0, 'served': 0, 'hosted': 0} for s in ecus}
    for d, (x, y) in dev.items():
        n = min(used, key=lambda s: math.hypot(used[s][0] - x, used[s][1] - y))
        out[n]['zone'] += 1
        if n in conn[d]:
            out[n]['served'] += 1
    for d, h in host.items():
        if h in out:
            out[h]['hosted'] += 1
    return out


def ecu_types(c, ecus, sol, local):
    """the emergent type of every used ECU (rule in the module docstring) and the measures behind it"""
    n_max = int(c.site_cfg.get('max_partitions', 4) or 4)
    if c.types:   # partition types: no cell cap; the role measure's cap = the cell full of the largest partition type
        site_cpu = n_max * max(t['cpu'] for t in c.types.values())
    else:
        site_cpu = float(c.site_cfg.get('cpu_cap', 0) or 0) or n_max * float(c.part.get('cpu_cap', 1))
    feats = defaultdict(set)
    for f in sol.get('hw_features', []):
        name, site = f.split('@')
        feats[site].add(name)
    if c.types and c.tops_feature:   # the accelerator is inside a partition type with TOPS (TDA4VH, ORIN)
        for site, parts in ecus.items():
            if any(c.types.get(p.get('ptype'), {}).get('tops', 0) > 0 for p in parts):
                feats[site].add(c.tops_feature)
    buses = defaultdict(int)
    for b in sol.get('buses', []):
        for u in b['units']:
            buses[u] += 1
    dev_sets = defaultdict(lambda: {'sensor': set(), 'actuator': set()})
    for sc, site in c.pos_x.items():
        for d in c.scs[sc]['devices']:
            dev_sets[site][c.devices[d]['kind']].add(d)
    devices = {site: {k: len(v) for k, v in q.items()} for site, q in dev_sets.items()}
    out = {}
    for site, parts in ecus.items():
        cpu = sum(c.demand(sc, 'cpu') for p in parts for sc in p['prim'] + p['hot']) / (site_cpu * c.derate(site))
        # utilisation of what is installed (user 2026-11-04): a partition is one compute block, so the unit's installed
        # capacity is its partitions times the block, not the cell's 4-block cap (the role rule keeps the cap measure)
        n_blocks = max(1, len(parts))
        inst = {r: sum(cap_of(c, p['name'], r, site) for p in parts) for r in ('cpu', 'ram', 'tops')}   # installed, derated
        load = {r: sum(c.demand(sc, r) for p in parts for sc in p['prim'] + p['hot']) for r in ('cpu', 'ram', 'tops')}
        cpu_inst = load['cpu'] / inst['cpu'] if inst['cpu'] > 0 else 0.0
        ram_inst = load['ram'] / inst['ram'] if inst['ram'] > 0 else 0.0
        tops_inst = load['tops'] / inst['tops'] if inst['tops'] > 0 else None   # partition types only
        doms = defaultdict(int)
        for p in parts:
            for sc in p['prim']:
                g = DOMAIN_GROUP.get(str(c.scs[sc].get('domain')), str(c.scs[sc].get('domain')))
                if g:
                    doms[g] += 1
        top_dom, top_n = max(doms.items(), key=lambda kv: kv[1]) if doms else ('-', 0)
        share = top_n / max(1, sum(doms.values()))
        compute = 'HW_ACC' in feats[site] or cpu >= COMPUTE_CPU
        io = local[site]['served'] >= IO_LOCAL
        domain = share >= DOMAIN_SHARE
        t = ('domain controller' if compute and domain else 'zone controller + compute' if compute and io
             else 'central compute' if compute else 'zone controller' if io else 'edge ECU')
        out[site] = {'type': t, 'cpu': cpu, 'cpu_inst': cpu_inst, 'ram_inst': ram_inst, 'tops_inst': tops_inst, 'blocks': n_blocks, 'devices': devices.get(site, {'sensor': 0, 'actuator': 0}), 'buses': buses[site],
                     'feats': feats[site], 'zone': local[site]['zone'], 'served': local[site]['served'], 'hosted': local[site]['hosted'],
                     'comp': sum(len(p['prim']) for p in parts), 'top_domain': top_dom, 'domain_share': share}
    return out


def top_view_map(render_dir, c, config_dir, architecture_copy, sol):
    """affine map car metres (x, y) -> render image (u, v) fitted on the anchors of the render (sites and devices);
    returns (A, cells, devices, cables)"""
    cr = ConfigReader(config_dir)
    _scs, _comm, _sens, _acts, _cables, locs = load_architecture_from_json(architecture_copy, cr, num_locs=c.num_locs)
    cells = {l.id: (float(l.location.x), float(l.location.y)) for l in locs}
    cab_path = os.path.splitext(os.path.abspath(sol_path_of(c)))[0] + '_cables.json'
    cab = json.load(open(cab_path))
    dev = {}
    for cb in cab['cables']:
        for end, pt in ((cb['src'], cb['points'][0]), (cb['dst'], cb['points'][-1])):
            if end not in cells:
                dev[end] = (float(pt[0]), float(pt[1]))
    anc = json.load(open(os.path.join(render_dir, 'top_anchors.json')))['anchors']
    P, Q = [], []
    for a in anc:
        name = a['id'].split(':')[0]
        m = cells.get(name) if a['kind'] == 'site' else dev.get(name)
        if m is not None:
            P.append([m[0], m[1], 1.0])
            Q.append([a['u'], a['v']])
    P, Q = np.array(P), np.array(Q)
    keep = np.ones(len(P), dtype=bool)
    for _ in range(3):   # a badge moved off its marker in the render is an outlier: drop and refit
        A, *_ = np.linalg.lstsq(P[keep], Q[keep], rcond=None)
        err = np.abs(P @ A - Q).max(axis=1)
        keep = err < 0.02
    return A, cells, dev, cab['cables']


def sol_path_of(c):
    return c.path if isinstance(getattr(c, 'path', None), str) else c._solution_path


def car_silhouette(render_dir):
    """the car's outline from the render: pixels that differ from the background, closed and hole-filled, the
    largest connected piece (wheels included); returns (mask, H, W)"""
    import scipy.ndimage as ndi
    img = mpimg.imread(os.path.join(render_dir, 'top_raw.png'))
    H, W = img.shape[0], img.shape[1]
    bg = img[0, 0, :3]
    m = np.abs(img[:, :, :3] - bg).sum(axis=2) > 0.2
    m = ndi.binary_fill_holes(ndi.binary_closing(m, iterations=8))
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m, H, W


def car_crop(mask, H, W):
    """the pixel box of the silhouette with a small margin: (y0, y1, x0, x1)"""
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    return (max(0, rows[0] - int(0.03 * H)), min(H, rows[-1] + int(0.03 * H)),
            max(0, cols[0] - int(0.015 * W)), min(W, cols[-1] + int(0.015 * W)))


def switch_nodes_of(sol, ecus):
    """cells that hold an Ethernet switch and no software: relay nodes of the backbone (site.switch_node_cost)"""
    return sorted({f.split('@')[1] for f in sol.get('hw_features', []) if f.startswith('SWITCH@') and f.split('@')[1] not in ecus})


def draw_panel_a(ax, render_dir, c, sol, types, labels, A, cells, dev, cables, titled=True, legend=True, ports=None):
    mask, H, W = car_silhouette(render_dir)
    y0, y1, x0, x1 = car_crop(mask, H, W)
    body = np.ones((H, W, 4))
    body[..., :3] = 1.0
    body[mask, :3] = (0.94, 0.94, 0.925)
    body[~mask, 3] = 0.0
    ax.imshow(body, zorder=0)
    ax.contour(mask.astype(float), levels=[0.5], colors=[INK2], linewidths=0.7, zorder=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis('off')

    def px(xy):
        u, v = np.array([xy[0], xy[1], 1.0]) @ A
        return u * W, (1.0 - v) * H

    for cb in sorted(cables, key=lambda q: (q['src'] in cells and q['dst'] in cells)):
        pts = np.array([px(p) for p in cb['points']])
        link = cb['src'] in cells and cb['dst'] in cells
        ax.plot(pts[:, 0], pts[:, 1], color=CABLE_COLOUR.get(cb['iface'], MUTED), lw=(2.0 if link else 1.0),
                alpha=(0.95 if link else 0.85), solid_capstyle='round', zorder=2)
    host = cable_end(c, sol)   # the unit a device's cable ends at (ETH: its attachment unit)
    for d, xy in dev.items():
        kind = c.devices.get(d, {}).get('kind', 'sensor')
        x, y = px(xy)
        fill = types[host[d]]['colour'] if host.get(d) in types else 'white'   # 2026-11-06: the colour of its ECU
        ax.plot([x], [y], marker=('o' if kind == 'sensor' else 's'), ms=13.0, mfc=fill, mec=INK, mew=1.1, zorder=3)
        if host.get(d) in labels:
            ax.text(x, y, labels[host[d]], fontsize=5.2, fontweight='bold', color=_text_on(fill) if fill != 'white' else INK,
                    ha='center', va='center', zorder=4)
    big = max(1, max(q['comp'] for q in types.values()))
    boxes = {}
    for site, t in types.items():
        x, y = px(cells[site])
        boxes[site] = [x, y, 0.046 * W * (0.7 + 0.45 * math.sqrt(t['comp'] / big))]
    for _ in range(50):   # push overlapping boxes apart (half a metre between cells is less than a box)
        moved = False
        items = list(boxes.items())
        for i_, (sa, a_) in enumerate(items):
            for sb, b_ in items[i_ + 1:]:
                dx, dy = b_[0] - a_[0], b_[1] - a_[1]
                ox = (a_[2] + b_[2]) / 2 + 0.01 * W - abs(dx)
                oy = (a_[2] + b_[2]) / 3 + 0.01 * W - abs(dy)
                if ox > 0 and oy > 0:
                    moved = True
                    if ox < oy:
                        sgn = 1 if dx >= 0 else -1
                        a_[0] -= sgn * ox / 2
                        b_[0] += sgn * ox / 2
                    else:
                        sgn = 1 if dy >= 0 else -1
                        a_[1] -= sgn * oy / 2
                        b_[1] += sgn * oy / 2
        if not moved:
            break
    for site, t in types.items():
        x, y, sz = boxes[site]
        tx, ty = px(cells[site])
        if math.hypot(tx - x, ty - y) > 0.004 * W:
            ax.plot([tx, x], [ty, y], color=INK, lw=1.0, zorder=3.5)
            ax.plot([tx], [ty], marker='o', ms=4, mfc=INK, mec=INK, zorder=3.5)
        ax.add_patch(FancyBboxPatch((x - sz / 2, y - sz / 3), sz, 2 * sz / 3, boxstyle='round,pad=0,rounding_size=7',
                                    facecolor=t['colour'], edgecolor=INK, lw=1.0, zorder=4))
        # ports: circles in the cable colour hanging outward from the bottom edge, at most 6 per row; hardware: small
        # squares inside the box, top-left (user 2026-11-03: ports outward, hardware inward, different shapes)
        pp = (ports or {}).get(site, {})
        plist = [k for k in ('ETH', 'CAN', 'LIN', 'FLEXRAY', 'UART', 'SPI') for _ in range(int(pp.get(k, 0)))]
        per_row = 6
        rows = [plist[r:r + per_row] for r in range(0, len(plist), per_row)]
        d = 0.15 * sz
        for ri, row in enumerate(rows):
            step = min(0.19 * sz, (sz - 0.1 * sz) / max(1, len(row)))
            cx0 = x - (len(row) - 1) * step / 2
            cy = y + sz / 3 + d * (0.35 + 1.05 * ri)
            for k, ptype in enumerate(row):
                ax.add_patch(Circle((cx0 + k * step, cy), d / 2, facecolor=CABLE_COLOUR.get(ptype, MUTED), edgecolor=INK, lw=0.6, zorder=5))
        glyphs = hw_glyphs(c, site, t['feats'])   # GPU modules: one square per module in its type's colour
        hs = min(0.17 * sz, 0.88 * sz / max(1, len(glyphs)) - 0.04 * sz)
        for k, colour in enumerate(glyphs):
            ax.add_patch(Rectangle((x - sz / 2 + 0.06 * sz + k * (hs + 0.04 * sz), y - sz / 3 + 0.05 * sz), hs, hs,
                                   facecolor=colour, edgecolor=INK, lw=0.6, zorder=6))
        ax.text(x, y, labels[site], fontsize=9, color=_text_on(t['colour']), ha='center', va='center', zorder=7, fontweight='bold')
    for site in switch_nodes_of(sol, types):   # switch-only relay nodes: a small grey box with the switch square
        x, y = px(cells[site])
        sz = 0.046 * W * 0.55
        ax.add_patch(FancyBboxPatch((x - sz / 2, y - sz / 3), sz, 2 * sz / 3, boxstyle='round,pad=0,rounding_size=5',
                                    facecolor='#e9e8e4', edgecolor=INK, lw=1.0, zorder=4))
        ax.add_patch(Rectangle((x - sz / 2 + 0.06 * sz, y - sz / 3 + 0.05 * sz), 0.22 * sz, 0.22 * sz, facecolor=HW_COLOUR['SWITCH'], edgecolor=INK, lw=0.6, zorder=6))
        ax.text(x, y + 0.05 * sz, labels.get(site, 'S'), fontsize=7.5, color=INK, ha='center', va='center', zorder=7, fontweight='bold')
    ax.text(x0 + 0.01 * (x1 - x0), y0 + 0.03 * (y1 - y0), 'front', fontsize=7, color=INK2, ha='left', va='top')
    if titled:
        ax.set_title('A  Placement (top view)', loc='left', fontsize=9, color=INK)
    ax.set_anchor('N')
    handles = []   # 2026-11-06: no role entries; the ECU colours are named by the E-labels on the boxes
    seen = sorted({f for q in types.values() for f in q['feats']}, key=lambda f: (HW_ORDER.index(f) if f in HW_ORDER else 9, f))
    if switch_nodes_of(sol, types):
        handles.append(Rectangle((0, 0), 1, 1, facecolor='#e9e8e4', edgecolor=INK, lw=0.8, label='switch node, no software'))
    gpu_used = {m for s_ in types for m in gpu_modules(c, s_)}   # GPU module types (2026-11-06): one entry per type
    for f in seen:
        if f == 'HW_ACC' and gpu_used:   # every GPU type of the catalogue, smallest first (user 2026-11-06: HAILO8 was missing)
            gfill = _ramp([k for k, _t in gpu_order(c)], GPU_RAMP)
            handles += [Rectangle((0, 0), 1, 1, facecolor=gfill.get(m, INK), edgecolor=INK, lw=0.8,
                                  label=f"GPU module {m}, {tp:,.0f} TOPS" + ('' if m in gpu_used else ' (not used)'))
                        for m, tp in gpu_order(c)]
        else:
            handles.append(Rectangle((0, 0), 1, 1, facecolor=HW_COLOUR.get(f, MUTED), edgecolor=INK, lw=0.8, label=HW_NAME.get(f, f)))
    used_types = {cb.get('iface') for cb in cables} | {b['type'] for b in sol.get('buses', [])}   # 2026-11-05: FlexRay was missing
    handles += [plt.Line2D([0], [0], color=CABLE_COLOUR[t], lw=2.0, label=t) for t in ('ETH', 'CAN', 'LIN', 'FLEXRAY', 'UART', 'SPI')
                if t in used_types or t in ('ETH', 'CAN', 'LIN', 'UART', 'SPI')]
    handles += [plt.Line2D([0], [0], marker='o', color='none', mfc=MUTED, mec=INK, markersize=6, label='port (circle, cable colour)')]
    handles += [plt.Line2D([0], [0], marker='o', color='none', mfc='white', mec=INK, markersize=9, label='sensor (colour + label = its ECU)'),
                plt.Line2D([0], [0], marker='s', color='none', mfc='white', mec=INK, markersize=9, label='actuator (colour + label = its ECU)')]
    if not legend:
        return handles
    ax.legend(handles=handles, loc='upper left', fontsize=7, ncol=4, frameon=False, bbox_to_anchor=(0.0, -0.01),
              handlelength=1.6, columnspacing=1.2)


def ecu_order(ecus, types):
    return sorted(ecus, key=lambda s: (-types[s]['comp'], s))   # 2026-11-06: most SWCs first (roles no longer used)


def draw_panel_b(ax, c, sol, ecus, ports, types, labels):
    """one column per ECU and rows aligned like a table: the header (role colour, E-label, cell, hardware glyphs) and
    the role name; the hosted components as one stacked bar (one segment per partition, fill = ASIL level, light to
    dark); then the measures behind the role, each as a bar with its threshold tick: largest functional domain, CPU
    load of the unit, attached sensors and actuators, shared bus lines"""
    sites = ecu_order(ecus, types)
    n = len(sites)
    col_w, gap = 1.0, 0.34
    x_of = {s: i * (col_w + gap) for i, s in enumerate(sites)}
    hdr, role_h, bar_h, row_h, bw = 0.46, 0.58, 2.3, 0.5, 0.62
    big = max(1, max(t['comp'] for t in types.values()))
    max_zone = max(1, max(t['zone'] for t in types.values()))
    max_bus = max(1, max(t['buses'] for t in types.values()))
    y_base = -(hdr + role_h + 0.5 + bar_h)
    rows = [('largest functional domain', 'domain', DOMAIN_SHARE), ('CPU load of the unit', 'cpu', COMPUTE_CPU),
            ('nearby devices it connects to\n(grey = nearby, not connected)', 'local', IO_LOCAL / max_zone), ('shared bus lines', 'bus', None)]
    y_row = {k: y_base - 0.42 - i * row_h for i, (_, k, _) in enumerate(rows)}
    lx = -0.2
    ax.text(lx, y_base + bar_h / 2, 'hosted components\none segment = one partition\nfill = ASIL level', fontsize=6.2,
            color=INK2, ha='right', va='center', linespacing=1.25)
    for name, k, _ in rows:
        ax.text(lx, y_row[k], name, fontsize=6.2, color=INK2, ha='right', va='center', linespacing=1.25)
    for s in sites:
        x, t = x_of[s], types[s]
        ax.add_patch(FancyBboxPatch((x, -hdr + 0.03), col_w, hdr - 0.06, boxstyle='round,pad=0.0,rounding_size=0.06',
                                    facecolor=t['colour'], edgecolor=INK, lw=0.8))
        ax.text(x + 0.07, -hdr / 2, labels[s], fontsize=8.5, fontweight='bold', color=INK, va='center', ha='left')
        gx, gy = x + col_w - 0.05, -hdr / 2
        if 'SWITCH' in t['feats']:
            ax.add_patch(Polygon([[gx - 0.12, gy], [gx - 0.06, gy + 0.10], [gx, gy], [gx - 0.06, gy - 0.10]],
                                 closed=True, facecolor=INK, edgecolor='none'))
            gx -= 0.16
        if 'HW_ACC' in t['feats']:
            ax.add_patch(Polygon([[gx - 0.13, gy - 0.09], [gx, gy - 0.09], [gx - 0.065, gy + 0.10]],
                                 closed=True, facecolor=INK, edgecolor='none'))
        ax.text(x + col_w / 2, -hdr - 0.05, f"{s}\n{t['type'].replace(' + ', chr(10) + '+ ')}", fontsize=5.8, color=INK2, ha='center', va='top', linespacing=1.3)
        # the stacked bar of hosted components, ASIL 0 at the bottom
        y = y_base
        for p in ecus[s]:
            h = bar_h * len(p['prim']) / big
            fill, tc = part_colours(c, p)
            ax.add_patch(Rectangle((x + 0.1, y), col_w - 0.2, max(0.0, h - 0.02), facecolor=fill, edgecolor='none'))
            if h >= 0.17:
                ax.text(x + col_w / 2, y + h / 2, str(len(p['prim'])), fontsize=5.8, color=tc,
                        ha='center', va='center')
            y += h
        ax.text(x + col_w / 2, y + 0.06, f"{t['comp']} · {len(ecus[s])} partitions", fontsize=5.8, color=INK, ha='center', va='bottom')
        # the role measures
        vals = {'domain': (t['domain_share'], f"{100 * t['domain_share']:.0f} %", re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', t['top_domain']).lower()),
                'cpu': (t['cpu'], f"{100 * t['cpu']:.0f} %", None),
                'local': (t['served'] / max_zone, f"{t['served']} of {t['zone']}", f"hosts software of {t['hosted']} devices"),
                'bus': (t['buses'] / max_bus, f"{t['buses']}", None)}
        for _, k, thr in rows:
            v, txt, sub = vals[k]
            yy = y_row[k]
            track = t['zone'] / max_zone if k == 'local' else 1.0
            ax.add_patch(Rectangle((x, yy - 0.07), bw * track, 0.14, facecolor=GRID, edgecolor='none'))
            ax.add_patch(Rectangle((x, yy - 0.07), bw * min(1.0, v), 0.14, facecolor=BAR_FILL, edgecolor='none'))
            if thr is not None:
                ax.plot([x + bw * thr] * 2, [yy - 0.11, yy + 0.11], color=INK2, lw=0.7, ls=(0, (1.5, 1.5)))
            ax.text(x + bw + 0.05, yy, txt, fontsize=5.8, color=INK, ha='left', va='center')
            if sub:
                ax.text(x, yy + 0.1, sub, fontsize=5.0, color=INK2, ha='left', va='bottom')
    y_min = min(y_row.values()) - 0.35
    ax.set_xlim(-1.75, n * (col_w + gap) - gap + 0.05)
    ax.set_ylim(y_min - 0.4, 0.08)
    ax.axis('off')
    ax.set_title('B  Deployment per ECU', loc='left', fontsize=9, color=INK)
    handles = ptype_handles(c, ecus, suffix='') or [Rectangle((0, 0), 1, 1, facecolor=ASIL_FILL[a], edgecolor='none', label=f'ASIL {a}')
                                                    for a in sorted(ASIL_FILL)]
    handles += [plt.Line2D([0], [0], marker='^', color='none', markerfacecolor=INK, markersize=7, label='accelerator (GPU)'),
                plt.Line2D([0], [0], marker='D', color='none', markerfacecolor=INK, markersize=6, label='Ethernet switch'),
                plt.Line2D([0], [0], color=INK2, lw=0.7, ls=(0, (1.5, 1.5)), label='threshold of the role rule')]
    ax.legend(handles=handles, loc='lower left', fontsize=6.0, ncol=8, frameon=False, bbox_to_anchor=(0.0, 0.0),
              handlelength=1.4, columnspacing=1.0)


def strip_units(ecus, types):
    """height of the partition strip in text lines: a three-line header and two lines per partition"""
    return max(3.1 + 1.95 * len(ecus[s]) for s in ecus)


def mini_bar(ax, x0, yc, w, frac, label, colour):
    """a small utilisation bar with its number: light track, fill always black (user 2026-11-04: white and black fills
    were confusing), the number in the text colour of its box"""
    ax.add_patch(Rectangle((x0, yc - 0.24), w, 0.48, facecolor='white', alpha=0.75, edgecolor='none'))
    ax.add_patch(Rectangle((x0, yc - 0.24), w * max(0.0, min(1.0, frac)), 0.48, facecolor=INK, edgecolor='none'))
    ax.text(x0 + w + 0.02, yc, label, fontsize=4.8, color=colour, ha='left', va='center')


def draw_partition_strip(ax, c, ecus, types, labels, tops_per_module, ports):
    """under the car (user 2026-11-03): one block per ECU, header in the role colour with its label, SWC count and I/O
    (2026-11-06: devices connected directly / via bus; no unit utilisation any more), then one box per partition in its
    ASIL colour (partition types: its type's ramp step, 2026-11-06) with the component count and its CPU / RAM use as bars"""
    sites = ecu_order(ecus, types)
    n = len(sites)
    ax.set_xlim(0, n)
    ax.set_ylim(-strip_units(ecus, types) - 0.3, 1.5)   # room above the headers for the port circles
    ax.axis('off')
    end, buses = cable_end(c, c.sol), c.sol.get('buses', [])
    on_bus = {d for b in buses for d in b['devices']}
    direct = {s: {d for d, u in end.items() if u == s and d not in on_bus} for s in sites}
    via_bus = {s: {d for b in buses if s in b['units'] for d in b['devices']} for s in sites}
    # I/O (user 2026-11-06, replaces the unit's CPU / RAM / GPU bars): the sensors and actuators connected to the ECU,
    # directly (their own cable or Ethernet link ends here) or over a bus line the ECU sits on; a device counts once
    io_label = {s: f"I/O: {len(direct[s] | via_bus[s])} ({len(direct[s])} direct, {len(via_bus[s])} via bus)" for s in sites}
    head_fs = min(6.5, 5.6 * 31 / max(len(v) for v in io_label.values()))   # one size for every header, fits the longest
    for i, s in enumerate(sites):
        t, x = types[s], i + 0.03
        ax.add_patch(FancyBboxPatch((x, -2.5), 0.94, 3.1, boxstyle='round,pad=0,rounding_size=0.08',
                                    facecolor=t['colour'], edgecolor=INK, lw=0.6))
        # header text (user 2026-11-06): the ECU label top left, then the SWC count, then the I/O, one bold font for all
        for yy, txt in ((0.15, labels[s]), (-0.85, f"{t['comp']} SWC"), (-1.85, io_label[s])):
            ax.text(x + 0.05, yy, txt, fontsize=head_fs, fontweight='bold', color=_text_on(t['colour']), ha='left', va='center')
        # the same symbols as on the car (user 2026-11-04): hardware as squares inside the header (right end of the title
        # line), ports as circles in the cable colour hanging outward from the header's top edge (on the car they hang
        # from the bottom edge; here the partitions are below); markers in points, so they stay round in this axis
        glyphs = hw_glyphs(c, s, t['feats'])   # GPU modules: one square per module in its type's colour
        for k, colour in enumerate(glyphs):
            ax.plot([x + 0.94 - 0.05 - (len(glyphs) - 1 - k) * 0.07], [0.15], marker='s', ms=5.2, mfc=colour,
                    mec=INK, mew=0.6, ls='none', zorder=6, clip_on=False)
        pp = (ports or {}).get(s, {})
        plist = [k for k in ('ETH', 'CAN', 'LIN', 'FLEXRAY', 'UART', 'SPI') for _ in range(int(pp.get(k, 0)))]
        if plist:
            step = min(0.07, 0.86 / len(plist))
            cx0 = x + 0.47 - (len(plist) - 1) * step / 2
            for k, ptype in enumerate(plist):
                ax.plot([cx0 + k * step], [0.6 + 0.3], marker='o', ms=5.0, mfc=CABLE_COLOUR.get(ptype, MUTED), mec=INK, mew=0.6,
                        ls='none', zorder=5, clip_on=False)
        y = -2.75
        for p in sorted(ecus[s], key=lambda q: -q['asil']):
            fill, tc = part_colours(c, p)   # partition type colour (2026-11-06), else the ASIL level's
            u = p['util']
            ax.add_patch(Rectangle((x, y - 1.75), 0.94, 1.7, facecolor=fill, edgecolor='white', lw=0.6))
            sb = (f" +{len(p['hot'])} hot" if p['hot'] else '') + (f" +{len(p['cold'])} cold" if p['cold'] else '')
            pt = f" {p['ptype']}" if p.get('ptype') else ''   # partition types: the type next to the level
            label = f"ASIL {p['asil']}{pt} · {len(p['prim'])} comp.{sb}"
            # font by length (2026-11-06): 31 characters fit the box at 5.0 pt, longer labels ran into the next ECU
            ax.text(x + 0.05, y - 0.5, label, fontsize=min(5.8, 5.0 * 31 / max(1, len(label))), color=tc, ha='left', va='center')
            for k, name in enumerate(('cpu', 'ram')):
                bx = x + 0.04 + k * 0.45
                ax.text(bx, y - 1.3, name.upper(), fontsize=4.8, color=tc, ha='left', va='center')
                mini_bar(ax, bx + 0.105, y - 1.3, 0.2, u.get(name, 0.0), f"{100 * u.get(name, 0.0):.0f}", tc)
            y -= 1.95


def print_ecu_table(c, sol, ecus, types, labels):
    """the deployment per ECU as a markdown table (the former panel B, for the paper), then the design-level numbers"""
    sites = ecu_order(ecus, types)
    peak = sol.get('comm_link_peak_load') or {}
    links = defaultdict(list)
    for l in sol.get('comm_links', []):
        key = f"{l['src_loc']}|{l['dst_loc']}|{l['iface']}"
        load = float(peak.get(key, peak.get(f"{l['dst_loc']}|{l['src_loc']}|{l['iface']}", 0.0)))
        cap = float(c.ct.get(l['iface'], {}).get('cap', 0.0) or 0.0)
        for u in (l['src_loc'], l['dst_loc']):
            links[u].append(load / cap if cap > 0 else 0.0)
    tot = len(c.scs)
    print('\nTable: deployment per ECU')
    print('| ECU | role | hardware | components | partitions (ASIL) | standbys hot / cold | unit CPU / RAM (of the installed blocks) | max partition CPU / RAM | hosted devices (I/O) | bus lines | backbone links (max load) | largest domain |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for s_ in sites:
        t, parts = types[s_], ecus[s_]
        hw = ', '.join(HW_NAME.get(f, f) for f in sorted(t['feats'], key=lambda f: HW_ORDER.index(f) if f in HW_ORDER else 9)) or '-'
        gmix = getattr(c, 'hw_types', {}).get((s_, 'HW_ACC'), {})
        if gmix:
            hw += ' (' + ', '.join(f"{n}x {k}" for k, n in sorted(gmix.items())) + ')'
        cnt = Counter((p['asil'], p.get('ptype') or '') for p in parts)
        asil = ', '.join(f"ASIL {a}" + (f" {pt}" if pt else '') + (f" x{n}" if n > 1 else '')
                         for (a, pt), n in sorted(cnt.items(), reverse=True))
        hot, cold = sum(len(p['hot']) for p in parts), sum(len(p['cold']) for p in parts)
        mcpu = max((p['util'].get('cpu', 0.0) for p in parts), default=0.0)
        mram = max((p['util'].get('ram', 0.0) for p in parts), default=0.0)
        lk = links.get(s_, [])
        dom = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', t['top_domain']).lower()
        links_txt = f"{len(lk)} ({100 * max(lk):.0f} %)" if lk else '0'
        print(f"| {labels[s_]} | {t['type']} | {hw} | {t['comp']} ({100 * t['comp'] / tot:.0f} %) | {len(parts)} ({asil}) | {hot} / {cold} | "
              f"{100 * t['cpu_inst']:.0f} % / {100 * t['ram_inst']:.0f} % | {100 * mcpu:.0f} % / {100 * mram:.0f} % | {t['hosted']} | {t['buses']} | {links_txt}", end='')
        print(f" | {dom} {100 * t['domain_share']:.0f} % |")
    pc = sol.get('partition_counts') or {}
    by_asil = defaultdict(int)
    for k, v in pc.items():
        by_asil[int(k.rsplit('_asil', 1)[1])] += int(v)
    feats = sol.get('hw_features', [])
    print('\nTable: the design')
    print('| measure | value |\n|---|---|')
    print(f"| total cost | {sol['total_cost']:.2f} |")
    print(f"| of which partitions / units / hardware / ports / cables / backbone links | {sol.get('partition_cost', 0):.0f} / {sol.get('site_cost', 0):.0f} / {sol.get('hw_cost', sol.get('hardware_cost', 0)):.0f} / {sol.get('interface_cost', 0):.0f} / {sol.get('cable_cost', 0):.0f} / {sol.get('comm_cost', 0):.0f} |")
    sw_nodes = switch_nodes_of(sol, ecus)
    print(f"| ECUs | {sol['num_ecus_used']} of {c.num_locs} candidate cells" + (f", plus {len(sw_nodes)} switch-only node{'s' if len(sw_nodes) > 1 else ''} ({', '.join(sw_nodes)})" if sw_nodes else '') + " |")
    print(f"| partitions | {sol['num_partitions']} ({', '.join(f'ASIL {a}: {n}' for a, n in sorted(by_asil.items(), reverse=True))}) |")
    n_acc = sum('HW_ACC' in t['feats'] for t in types.values())   # with partition types: units with a TOPS partition type
    print(f"| accelerators / switches | {n_acc} / {sum('SWITCH' in f for f in feats)} |")
    if c.types:
        tc = Counter(t for t in c.ptype.values())
        print(f"| partition types | {', '.join(f'{k} {tc[k]}' for k in c.types if tc.get(k))} |")
    print(f"| cable length (devices / backbone) | {sol['cable_length']:.1f} m ({float(sol.get('peripheral_cable_length', 0)):.1f} / {float(sol.get('backbone_length', 0)):.1f}) |")
    print(f"| backbone links / shared bus lines | {len(sol.get('comm_links', []))} / {len(sol.get('buses', []))} |")
    lf = sol.get('lost_functions') or {}
    print(f"| single-failure scenarios | {len(sol.get('failure_scenarios', []) or [])} (every critical function survives, checker rule); "
          f"non-critical functions lost: worst {lf.get('worst', '-')} of {lf.get('non_critical', '-')} ({lf.get('worst_scenario', '-')}), mean {float(lf.get('mean', 0)):.0f} |")


def print_tables(c, sol, sites, labels, ports):
    """the details that are not in the figure (for the text): backbone links, shared bus lines, ports per ECU"""
    peak = sol.get('comm_link_peak_load') or {}
    print('backbone links (pair, interface, peak load / capacity):')
    for l in sorted(sol.get('comm_links', []), key=lambda l: (labels.get(l['src_loc'], l['src_loc']), labels.get(l['dst_loc'], l['dst_loc']))):
        key = f"{l['src_loc']}|{l['dst_loc']}|{l['iface']}"
        load = float(peak.get(key, peak.get(f"{l['dst_loc']}|{l['src_loc']}|{l['iface']}", 0.0)))
        cap = float(c.ct.get(l['iface'], {}).get('cap', 0.0) or 0.0)
        print(f"  {labels.get(l['src_loc'], l['src_loc'])}-{labels.get(l['dst_loc'], l['dst_loc'])}  {l['iface']}  {100 * load / cap if cap else 0:.0f} %")
    print('shared bus lines (id, ECUs on it, devices, length, device load / capacity):')
    for b in sorted(sol.get('buses', []), key=lambda q: (q['type'], q['id'])):
        load = sum(c.devices[d]['volume'] for d in b['devices'] if d in c.devices)
        cap = float(c.ct.get(b['type'], {}).get('cap', 0.0) or 0.0)
        print(f"  {b['id']}  {' '.join(labels.get(u, u) for u in b['units'])}  {len(b['devices'])} dev  "
              f"{float(b['length_m']):.1f} m  {100 * load / cap if cap else 0:.0f} %")
    print('ports per ECU:')
    for s in sites:
        print(f"  {labels[s]}  " + ', '.join(f"{k} {v}" for k, v in sorted(ports.get(s, {}).items(), key=lambda kv: -kv[1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('solution')
    ap.add_argument('--config_dir', default=None)
    ap.add_argument('--architecture', default=None)
    ap.add_argument('--render_dir', required=True)
    ap.add_argument('--out', required=True, help='.pdf or .png; a .png twin is written next to a .pdf')
    ap.add_argument('--title', default=None)
    ap.add_argument('--panel_b', action='store_true', help='also draw the deployment grid (panel B); default: the top view only')
    a = ap.parse_args()
    c = read(a.solution, a.config_dir, a.architecture)
    c._solution_path = os.path.abspath(a.solution)
    sol = c.sol
    ecus = partition_table(c)
    ports = ports_of(sol)
    run_dir = os.path.dirname(os.path.abspath(a.solution))
    arch_copy = os.path.join(run_dir, 'architecture_used.json')
    A, cells, dev, cables = top_view_map(a.render_dir, c, a.config_dir or os.path.join(REPO, 'configs'),
                                         arch_copy if os.path.exists(arch_copy) else a.architecture, sol)
    types = ecu_types(c, ecus, sol, local_devices(c, sol, ecus, cells, dev))
    labels = {s: f'E{i + 1}' for i, s in enumerate(ecu_order(ecus, types))}
    for i, s in enumerate(ecu_order(ecus, types)):   # 2026-11-06: one colour per ECU (roles out of the figure)
        types[s]['colour'] = ECU_COLOURS[i] if i < len(ECU_COLOURS) else MUTED
    labels.update({s: f'S{i + 1}' for i, s in enumerate(switch_nodes_of(sol, ecus))})   # switch-only relay nodes
    for s_ in ecu_order(ecus, types):
        t = types[s_]
        print(f"{labels[s_]} = {s_}: {t['type']}; {t['comp']} components, CPU {100 * t['cpu_inst']:.0f} % of its {t['blocks']} installed block(s) ({100 * t['cpu']:.0f} % of the cell's cap, the role measure), "
              f"{', '.join(sorted(t['feats'])) or 'no hardware'}, connected to {t['served']} of the {t['zone']} nearby devices, "
              f"hosts the software of {t['hosted']} devices ({t['devices']['sensor']} sensors + {t['devices']['actuator']} actuators), "
              f"{t['buses']} bus lines, largest domain {t['top_domain']} {100 * t['domain_share']:.0f} %")

    if a.panel_b:
        fig = plt.figure(figsize=(18, 7.4), facecolor='white')
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[0.95, 1.05], left=0.02, right=0.99, top=0.9, bottom=0.03, wspace=0.06)
        ax_a = fig.add_subplot(gs[0, 0])
        draw_panel_b(fig.add_subplot(gs[0, 1]), c, sol, ecus, ports, types, labels)
    else:
        mask, H, W = car_silhouette(a.render_dir)
        y0, y1, x0, x1 = car_crop(mask, H, W)
        fig_w, img_h = 10.5, 10.5 * 0.96 * (y1 - y0) / (x1 - x0)     # the car fills the width
        top, legend_h = 0.45, 0.75                                     # inches for the title and the legend rows
        strip_h = 0.105 * strip_units(ecus, types) + 0.3                # the partition strip under the car
        fig_h = top + img_h + strip_h + legend_h
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
        ax_a = fig.add_axes([0.02, (legend_h + strip_h) / fig_h, 0.96, img_h / fig_h])
        hw_cfg = json.load(open(os.path.join(a.config_dir or os.path.join(REPO, 'configs'), 'hardwares.json')))['hardware_costs']
        draw_partition_strip(fig.add_axes([0.02, legend_h / fig_h, 0.96, (strip_h - 0.1) / fig_h]), c, ecus, types, labels,
                             float(hw_cfg.get('HW_ACC', {}).get('tops', 26.0) or 26.0), ports)
    handles = draw_panel_a(ax_a, a.render_dir, c, sol, types, labels, A, cells, dev, cables, titled=a.panel_b, legend=a.panel_b, ports=ports)
    if not a.panel_b:   # one legend for the car and the strip, centred at the bottom
        parts = ptype_handles(c, ecus) or [Rectangle((0, 0), 1, 1, facecolor=ASIL_FILL[k], edgecolor='none', label=f'ASIL {k} partition')
                                           for k in sorted(ASIL_FILL)]
        # grouped legend (2026-11-06): hardware, cables, devices, partitions, each group in whole columns of `rows`
        groups = [[h for h in handles if isinstance(h, Rectangle)],
                  [h for h in handles if isinstance(h, plt.Line2D) and h.get_marker() in (None, 'None', '')],
                  [h for h in handles if isinstance(h, plt.Line2D) and h.get_marker() not in (None, 'None', '')], parts]
        rows = max(4, math.ceil(len(parts) / 2))
        blank = lambda: Rectangle((0, 0), 1, 1, alpha=0.0, label=' ')
        handles = [h for g in groups if g for h in g + [blank() for _ in range(-len(g) % rows)]]
        fig.legend(handles=handles, loc='lower center', ncol=len(handles) // rows, fontsize=6.8, frameon=False,
                   bbox_to_anchor=(0.5, 0.02), handlelength=1.5, columnspacing=1.0)
        fig.text(0.5, 0.004, "partition bars: load with the rho 1 margins (primaries + hot standbys) over the partition's capacity, "
                 "CPU derated by the cell's thermal class; I/O: sensors and actuators connected directly or over a bus line of the ECU",
                 fontsize=6.2, color=INK, ha='center', va='bottom')
    print_ecu_table(c, sol, ecus, types, labels)
    print_tables(c, sol, ecu_order(ecus, types), labels, ports)
    title = a.title or f"{len(c.scs)} components, {c.num_locs} candidate cells, rho {c.rho:g}: {os.path.basename(run_dir)}"
    fig.suptitle(title, x=0.03, y=0.985, ha='left', fontsize=12, color=INK, fontweight='bold')
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    if out.endswith('.pdf'):
        fig.savefig(out[:-4] + '.png', dpi=160)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
