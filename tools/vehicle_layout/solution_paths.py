#!/usr/bin/env python3
"""
Route the cables of an optimisation result through the installation space.

  dse/bin/python tools/vehicle_layout/solution_paths.py results/<run>/solution_1.json [--config_dir configs]

Reads the solution (assignment SC -> site, comm_links site <-> site with interface, peripheral attachments, bus
lines, directly wired devices), computes the shortest path of every cable on the voxel graph of
configs/routing_space.npz (same graph as the distance table) and writes <solution>_cables.json: a list of polylines
{src, dst, iface, length_m, points [[x,y,z], ...]} plus the list of used sites. tools/vehicle_viz/build_vehicle.py
--solution draws it. Each polyline starts and ends at the unit / device itself: the straight drop to the nearest
channel point (e.g. into a door, where the space has no channel) is part of the cable, as in distances.json, so the
routed total equals the cable length the optimiser costs.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from generate_layout import build_voxel_graph  # noqa: E402
from check_solution import SHARED_BUSES, SolutionCheck  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('solution')
    ap.add_argument('--config_dir', default='configs')
    args = ap.parse_args()
    sol = json.load(open(args.solution))
    cfg = lambda f: json.load(open(os.path.join(args.config_dir, f)))
    channels, sensors, actuators = cfg('channels.json'), cfg('sensors.json')['sensors'], cfg('actuators.json')['actuators']
    iface_of = {it['id']: it['interface'] for it in sensors + actuators}
    where = {it['id']: it['position'] for it in cfg('sites.json')['sites'] + sensors + actuators}

    g, pos, _kinds, _main, _ncomp, _vs = build_voxel_graph(os.path.join(args.config_dir, 'routing_space.npz'))
    attach = {ent: int(a['node'][1:]) for ent, a in channels['attachments'].items()}   # 'v123' -> 123

    # cables: backbone links + peripheral -> hosting site
    cables = []
    for lk in sol.get('comm_links', []):
        for _ in range(int(lk.get('count', 1))):
            cables.append((lk['src_loc'], lk['dst_loc'], lk.get('iface', 'ETH')))
    for key in ('eth_sensor_attachments', 'eth_actuator_attachments'):
        for dev, site in sol.get(key, {}).items():
            cables.append((dev, site, iface_of.get(dev, 'ETH')))
    # bus lines: every edge of the line's tree (members and junction points) is one cable piece
    for b in sol.get('buses', []):
        for a, c in b.get('tree', []):
            cables.append((a, c, b['type']))
    # directly wired devices (SPI, UART, ...): one cable from the device to every component that reads or drives it,
    # and to the standby of a critical one; the checker's reading of the run's architecture, so it matches the cost
    chk = SolutionCheck(args.solution, config_dir=args.config_dir)
    chk.load()
    chk.parse()
    wired = list(chk.pos_x.items()) + [(sc, st) for sc, st in chk.pos_r.items() if sc in chk.crit]
    for sc, site in wired:
        for dev in chk.scs.get(sc, {}).get('devices', []):
            if iface_of[dev] != 'ETH' and iface_of[dev] not in SHARED_BUSES:
                cables.append((dev, site, iface_of[dev]))

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
        pts = pts[::-1]
        for end, at in ((a, 0), (b, len(pts))):   # the straight drop between the unit / device and its channel point
            p = [round(float(where[end][k]), 3) for k in 'xyz']
            if np.linalg.norm(np.subtract(p, pts[min(at, len(pts) - 1)])) > 0.005:
                pts.insert(at, p)
        length = channels['attachments'][a]['length'] + float(dist[row[u], v]) + channels['attachments'][b]['length']
        out.append({'src': a, 'dst': b, 'iface': iface, 'length_m': round(length, 3), 'points': pts})

    # every cell with a unit: ECUs (software placed there) and pure switch nodes (a switch, no software)
    used_sites = sorted(set(sol.get('assignment', {}).values())
                        | {str(h).split('@')[1] for h in sol.get('hw_features', []) if str(h).startswith('SWITCH@')})
    res = {'solution': os.path.abspath(args.solution), 'used_sites': used_sites, 'cables': out,
           'num_candidate_sites': sol.get('num_candidate_sites', len(json.load(open(os.path.join(args.config_dir, 'sites.json')))['sites'])),
           'total_routed_length_m': round(sum(c['length_m'] for c in out), 2)}
    dst = os.path.splitext(args.solution)[0] + '_cables.json'
    json.dump(res, open(dst, 'w'), indent=1)
    by = {}
    for c in out:
        by[c['iface']] = by.get(c['iface'], 0) + 1
    print(f"{len(out)} cables routed ({by}), total {res['total_routed_length_m']} m "
          f"(solution's cable length {sol.get('cable_length', float('nan')):.2f} m), {len(used_sites)} sites used; written {dst}")
    if missing:
        print(f"WARNING: {len(missing)} cables without attachment nodes (not drawn): {missing[:5]}")


if __name__ == '__main__':
    main()
