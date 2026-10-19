#!/usr/bin/env python3
"""
Route the cables of an optimisation result through the installation space.

  dse/bin/python tools/vehicle_layout/solution_paths.py results/<run>/solution_1.json [--config_dir configs]

Reads the solution (assignment SC -> site, comm_links site <-> site with interface, peripheral attachments), computes
the shortest path of every cable on the voxel graph of configs/routing_space.npz (same graph as the distance table)
and writes <solution>_cables.json: a list of polylines {src, dst, iface, length_m, points [[x,y,z], ...]} plus the
list of used sites. tools/vehicle_viz/build_vehicle.py --solution draws it.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_layout import build_voxel_graph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('solution')
    ap.add_argument('--config_dir', default='configs')
    args = ap.parse_args()
    sol = json.load(open(args.solution))
    cfg = lambda f: json.load(open(os.path.join(args.config_dir, f)))
    channels, sensors, actuators = cfg('channels.json'), cfg('sensors.json')['sensors'], cfg('actuators.json')['actuators']
    iface_of = {it['id']: it['interface'] for it in sensors + actuators}

    g, pos, _kinds, _main, _ncomp, _vs = build_voxel_graph(os.path.join(args.config_dir, 'routing_space.npz'))
    attach = {ent: int(a['node'][1:]) for ent, a in channels['attachments'].items()}   # 'v123' -> 123

    # cables: backbone links + peripheral -> hosting site
    cables = []
    for lk in sol.get('comm_links', []):
        for _ in range(int(lk.get('count', 1))):
            cables.append((lk['src_loc'], lk['dst_loc'], lk.get('iface', 'ETH')))
    for key in ('eth_sensor_attachments', 'eth_actuator_attachments', 'shared_sensor_attachments', 'shared_actuator_attachments'):
        for dev, site in sol.get(key, {}).items():
            cables.append((dev, site, iface_of.get(dev, 'ETH')))
    # peripherals without an explicit attachment: wired to the site of their driver SC
    assigned = {dev for a, b, _ in cables for dev in (a, b)}
    for sc, site in sol.get('assignment', {}).items():
        for dev in iface_of:
            if sc.endswith('_' + dev) and dev not in assigned:
                cables.append((dev, site, iface_of[dev]))
                assigned.add(dev)

    sources = sorted({attach[a] for a, _, _ in cables if a in attach})
    dist, pred = dijkstra(g, directed=False, indices=sources, return_predecessors=True)
    row = {n: i for i, n in enumerate(sources)}
    out = []
    missing = []
    for a, b, iface in cables:
        if a not in attach or b not in attach:
            missing.append((a, b))
            continue
        u, v = attach[a], attach[b]
        pts, n = [], v
        while n != u and n >= 0:
            pts.append(pos[n].round(3).tolist())
            n = pred[row[u], n]
        pts.append(pos[u].round(3).tolist())
        out.append({'src': a, 'dst': b, 'iface': iface, 'length_m': round(float(dist[row[u], v]), 3), 'points': pts[::-1]})

    used_sites = sorted(set(sol.get('assignment', {}).values()))
    res = {'solution': os.path.abspath(args.solution), 'used_sites': used_sites, 'cables': out,
           'num_candidate_sites': sol.get('num_candidate_sites', len(json.load(open(os.path.join(args.config_dir, 'sites.json')))['sites'])),
           'total_routed_length_m': round(sum(c['length_m'] for c in out), 2)}
    dst = os.path.splitext(args.solution)[0] + '_cables.json'
    json.dump(res, open(dst, 'w'), indent=1)
    by = {}
    for c in out:
        by[c['iface']] = by.get(c['iface'], 0) + 1
    print(f"{len(out)} cables routed ({by}), total {res['total_routed_length_m']} m, {len(used_sites)} sites used; written {dst}")
    if missing:
        print(f"WARNING: {len(missing)} cables without attachment nodes (not drawn): {missing[:5]}")


if __name__ == '__main__':
    main()
