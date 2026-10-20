#!/usr/bin/env python3
"""
Technical architecture diagram of one optimisation result.

  dse/bin/python tools/architecture_diagram.py results/<run>/solution_1.json [--config_dir configs] [--out <path without extension>]

Draws every active control unit (candidate cell) as a box: header with cell id, zone, hardware (DSP, HW_ACC, SWITCH)
and ports; inside, one block per partition (ASIL, CPU/RAM/ROM utilisation) listing its software components coloured
by domain; below the box the peripherals attached to the unit with their interface; between the boxes the backbone
cables (type and count, relays over a switch as a path). A table of the cause-effect chains with their network
latency and budget closes the figure. Output: <out>.pdf and <out>.png (default: next to the solution file).
"""
import argparse
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

IFACE_COLOUR = {'ETH': '#3b8de0', 'CAN': '#2fb36a', 'LIN': '#d9c21c', 'UART': '#b08b45', 'SPI': '#3fb3aa', 'FLEXRAY': '#c060c0'}
DOMAIN_COLOUR = {'ADAS': '#dbe9fb', 'Driver': '#ececec', 'DriveTrain': '#fde3c8', 'BodyComfort': '#dcf3dc',
                 'Connectivity': '#e9dcf5', 'Infotainment': '#fbdde7'}
ASIL_NAME = {0: 'QM', 1: 'A', 2: 'B', 3: 'C', 4: 'D'}
HW_LABEL = {'HW_ACC': 'accelerator', 'DSP': 'DSP', 'HSM': 'HSM', 'SWITCH': 'switch'}


def load(args):
    sol = json.load(open(args.solution))
    cfg = lambda f: json.load(open(os.path.join(args.config_dir, f)))
    arch = {s['id']: s for s in json.load(open(args.architecture))['software_components']}
    sites = {s['id']: s for s in cfg('sites.json')['sites']}
    devices = {d['id']: d for d in cfg('sensors.json')['sensors'] + cfg('actuators.json')['actuators']}
    caps = cfg('partitions.json')['partition']
    return sol, arch, sites, devices, caps


def unit_data(sol, arch, devices):
    """Per active cell: partitions -> SWCs, hardware, ports, peripherals."""
    units = defaultdict(lambda: {'partitions': defaultdict(list), 'hw': [], 'ports': [], 'periph': []})
    for sc, part in sol['partitions'].items():
        site = sol['assignment'][sc]
        units[site]['partitions'][part].append(sc)
    for h in sol.get('hw_features', []):
        name, site = h.split('@')
        units[site]['hw'].append(name)
    for p in sol.get('interfaces', []):
        cnt, rest = p.split('x', 1)
        iface, site = rest.split('@')
        units[site]['ports'].append((iface, int(cnt)))
    for key in ('eth_sensor_attachments', 'eth_actuator_attachments', 'shared_sensor_attachments', 'shared_actuator_attachments'):
        for dev, site in sol.get(key, {}).items():
            units[site]['periph'].append(dev)
    for lk in sol.get('comm_links', []):
        units[lk['src_loc']]; units[lk['dst_loc']]
    return units


def part_label(part, scs, arch, caps):
    asil = int(part.split('_asil')[1].split('_')[0])
    cpu = sum(arch[s]['cpu_req'] for s in scs) / caps['cpu_cap'] * 100
    ram = sum(arch[s]['ram_req'] for s in scs) / caps['ram_cap'] * 100
    rom = sum(arch[s]['rom_req'] for s in scs) / caps['rom_cap'] * 100
    return f"partition ASIL {ASIL_NAME.get(asil, asil)}  ·  CPU {cpu:.0f} %  RAM {ram:.0f} %  ROM {rom:.0f} %"


def draw(sol, arch, sites, devices, caps, out, title):
    units = unit_data(sol, arch, devices)
    order = sorted(units, key=lambda s: (-sum(len(v) for v in units[s]['partitions'].values()), s))
    # geometry (inches): one column per unit; SWC lists in two columns when long
    LINE, COLW, GAP = 0.125, 4.5, 0.35
    header_h, part_head, periph_line = 0.62, 0.26, 0.15
    heights = {}
    for s in order:
        u = units[s]
        h = header_h
        for part, scs in sorted(u['partitions'].items()):
            rows = math.ceil(len(scs) / 2) if len(scs) > 8 else len(scs)
            h += part_head + rows * LINE + 0.12
        h += 0.1
        heights[s] = h
    box_h = max(heights.values()) if heights else 1.0
    periph_h = max((len(u['periph']) for u in units.values()), default=0) * periph_line + 0.35
    chains = sol.get('chain_latency', {})
    chain_h = (len(chains) + 1) * 0.16 + 0.4 if chains else 0.0
    link_h = 0.9 + 0.22 * max(0, len(sol.get('comm_links', [])) - 1)
    n = len(order)
    fig_w = max(8.0, n * (COLW + GAP) + GAP)
    fig_h = 1.0 + link_h + box_h + periph_h + chain_h + 0.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis('off')
    fig.text(0.5 * GAP / fig_w, (fig_h - 0.3) / fig_h, title, fontsize=11, fontweight='bold', ha='left', va='center')
    x0 = {s: GAP + i * (COLW + GAP) for i, s in enumerate(order)}
    top = fig_h - 1.0 - link_h
    # units
    for s in order:
        u = units[s]
        x, y_top = x0[s], top
        box = FancyBboxPatch((x, y_top - box_h), COLW, box_h, boxstyle='round,pad=0.02,rounding_size=0.08',
                             fc='#f7f9fc', ec='#2a3f6f', lw=1.4)
        ax.add_patch(box)
        zone = sites.get(s, {}).get('zone', '?')
        ax.text(x + 0.12, y_top - 0.17, f"{s}  ·  {zone}", fontsize=10, fontweight='bold', va='center')
        hw = ', '.join(HW_LABEL.get(h, h) for h in sorted(u['hw'])) or 'none'
        ax.text(x + 0.12, y_top - 0.36, f"hardware: {hw}", fontsize=7.5, va='center', color='#333')
        ports = '  '.join(f"{c}×{i}" for i, c in sorted(u['ports'], key=lambda t: -t[1])) or 'none'
        ax.text(x + 0.12, y_top - 0.52, f"ports: {ports}", fontsize=7.5, va='center', color='#333')
        y = y_top - header_h
        if not u['partitions']:
            ax.text(x + 0.12, y - 0.15, 'no software (network node)', fontsize=7.5, style='italic', va='center', color='#555')
        for part, scs in sorted(u['partitions'].items()):
            two = len(scs) > 8
            rows = math.ceil(len(scs) / 2) if two else len(scs)
            ph = part_head + rows * LINE + 0.12
            ax.add_patch(FancyBboxPatch((x + 0.1, y - ph), COLW - 0.2, ph - 0.04, boxstyle='round,pad=0.01,rounding_size=0.04',
                                        fc='white', ec='#8a2b2b', lw=1.0))
            ax.text(x + 0.18, y - 0.15, part_label(part, scs, arch, caps), fontsize=7.2, fontweight='bold', va='center', color='#8a2b2b')
            for k, sc in enumerate(sorted(scs, key=lambda i: (arch[i]['domain'], i))):
                col, row = (k % 2, k // 2) if two else (0, k)
                cx = x + 0.18 + col * (COLW - 0.36) / 2
                cy = y - part_head - 0.02 - row * LINE
                ax.add_patch(plt.Rectangle((cx, cy - LINE * 0.42), (COLW - 0.36) / 2 - 0.06, LINE * 0.84,
                                           fc=DOMAIN_COLOUR.get(arch[sc]['domain'], '#eee'), ec='none'))
                ax.text(cx + 0.04, cy, sc, fontsize=5.4, va='center', family='DejaVu Sans Mono')
            y -= ph
    # peripherals under each unit
    for s in order:
        u = units[s]
        x, y = x0[s], top - box_h - 0.2
        if not u['periph']:
            continue
        ax.text(x + 0.12, y, 'attached peripherals', fontsize=7.5, fontweight='bold', va='center', color='#333')
        for k, dev in enumerate(sorted(u['periph'])):
            iface = devices.get(dev, {}).get('interface', '?')
            yy = y - 0.17 - k * periph_line
            ax.add_patch(plt.Rectangle((x + 0.12, yy - 0.055), 0.11, 0.11, fc=IFACE_COLOUR.get(iface, '#999'), ec='none'))
            ax.text(x + 0.28, yy, f"{dev}  ({iface})", fontsize=6.3, va='center')
    # backbone links above the boxes
    links = sorted(sol.get('comm_links', []), key=lambda l: (order.index(l['src_loc']), order.index(l['dst_loc'])))
    relays = {(r['src_loc'], r['dst_loc'], r['iface']): r['path'] for r in sol.get('backbone_routes', []) if len(r['path']) > 2}
    for k, lk in enumerate(links):
        xa, xb = x0[lk['src_loc']] + COLW / 2, x0[lk['dst_loc']] + COLW / 2
        ya = top + 0.25 + k * 0.22
        c = IFACE_COLOUR.get(lk['iface'], '#777')
        ax.plot([xa, xa], [top + 0.02, ya], color=c, lw=1.2)
        ax.plot([xb, xb], [top + 0.02, ya], color=c, lw=1.2)
        ax.plot([xa, xb], [ya, ya], color=c, lw=2.2)
        label = f"{lk['count']}×{lk['iface']}"
        via = [p for key, p in relays.items() if {key[0], key[1]} == {lk['src_loc'], lk['dst_loc']}]
        if via:
            label += '  via ' + ' → '.join(via[0][1:-1])
        ax.text((xa + xb) / 2, ya + 0.07, label, fontsize=7, ha='center', va='bottom', color=c)
    if not links:
        ax.text(GAP, top + 0.3, 'no backbone cables (all software on one unit)', fontsize=7.5, style='italic', color='#555')
    # chain table
    if chains:
        y = top - box_h - periph_h - 0.25
        ax.text(GAP, y, 'cause-effect chains: network latency / budget (ms)', fontsize=8, fontweight='bold', va='center')
        for k, (cid, c) in enumerate(sorted(chains.items())):
            yy = y - 0.2 - k * 0.16
            cross = ', '.join(f"{x['from']}→{x['to']} {x['iface']} {x['hops']} hop {x['ms']:.1f}" for x in c['crossings']) or 'no network crossing'
            ok = 'ok' if c['ok'] else 'VIOLATED'
            name = cid.split('_Driver_ACT_')[-1].replace('_positioning', ' (from positioning sensor)')
            ax.text(GAP, yy, name, fontsize=6.6, va='center', family='DejaVu Sans Mono')
            ax.text(GAP + 3.6, yy, f"{c['latency_ms']:7.2f} / {c['budget_ms']:6.1f}  {ok}", fontsize=6.6, va='center', family='DejaVu Sans Mono',
                    color='#8a2b2b' if not c['ok'] else '#222')
            ax.text(GAP + 6.4, yy, cross, fontsize=6.2, va='center', color='#444')
    # legend
    lx, ly = GAP, fig_h - 0.62
    ax.text(lx, ly, 'domains:', fontsize=6.5, va='center', color='#333')
    for k, (dom, col) in enumerate(DOMAIN_COLOUR.items()):
        ax.add_patch(plt.Rectangle((lx + 0.7 + k * 1.15, ly - 0.06), 0.14, 0.12, fc=col, ec='#999', lw=0.4))
        ax.text(lx + 0.88 + k * 1.15, ly, dom, fontsize=6.5, va='center')
    for k, (iface, col) in enumerate(IFACE_COLOUR.items()):
        ax.add_patch(plt.Rectangle((lx + 0.7 + (len(DOMAIN_COLOUR) + 0.3 + k) * 1.15, ly - 0.06), 0.14, 0.12, fc=col, ec='none'))
        ax.text(lx + 0.88 + (len(DOMAIN_COLOUR) + 0.3 + k) * 1.15, ly, iface, fontsize=6.5, va='center')
    fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=170)
    plt.close(fig)
    return out + '.pdf', out + '.png'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('solution')
    ap.add_argument('--config_dir', default='configs')
    ap.add_argument('--architecture', default='configs/full_architecture_v2.json')
    ap.add_argument('--out', default=None, help='output path without extension (default: <solution>_architecture)')
    args = ap.parse_args()
    sol, arch, sites, devices, caps = load(args)
    out = args.out or os.path.splitext(args.solution)[0] + '_architecture'
    info = sol.get('solve_info', {})
    title = (f"{os.path.basename(os.path.dirname(os.path.abspath(args.solution)))} / {os.path.basename(args.solution)}:  "
             f"{sol['num_locations_used']} control units on {sol.get('num_candidate_sites', '?')} candidate cells, "
             f"{sol['num_partitions']} partitions, cost {sol['total_cost']:.0f}, cable {sol['cable_length']:.1f} m, {sol.get('status')}")
    files = draw(sol, arch, sites, devices, caps, out, title)
    print('written:', *files)


if __name__ == '__main__':
    main()
