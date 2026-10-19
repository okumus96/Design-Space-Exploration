#!/usr/bin/env python3
"""
Generate candidate ECU sites and the cable channel graph from configs/vehicle.json.

Inputs (configs/):
  vehicle.json     fixed blocks, permissible ECU zones, cable channels (polylines)
  sensors.json     peripheral positions (metres, vehicle frame)
  actuators.json

Outputs (configs/, overwritten):
  sites.json       candidate ECU sites: grid points inside every ECU zone
  channels.json    channel graph: nodes, edges (with lengths) and the attachment of every site / peripheral
  distances.json   shortest-path cable length between every pair of sites and peripherals

Usage:  python tools/vehicle_layout/generate_layout.py [--config_dir configs] [--quiet]
"""
import argparse
import json
import math
import os
import sys
from itertools import product

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _node_key(p):
    """Nodes are identified by their rounded coordinates so that channels sharing an end point are joined."""
    return (round(p[0], 3), round(p[1], 3), round(p[2], 3))


def _junctions_on_segment(a, b, vertices, tol=1e-3):
    """Vertices of other channels lying on segment a-b (strictly inside), ordered along the segment."""
    ab = [b[i] - a[i] for i in range(3)]
    ab2 = sum(v * v for v in ab)
    hits = []
    for p in vertices:
        t = sum((p[i] - a[i]) * ab[i] for i in range(3)) / ab2
        if tol < t < 1 - tol:
            q = [a[i] + t * ab[i] for i in range(3)]
            if _dist(p, q) < tol:
                hits.append((t, p))
    return [p for _, p in sorted(hits)]


def build_channel_graph(channels_cfg):
    """Subdivide every channel polyline and return (graph, node_pos, node_channels, node_names).

    Where a vertex of one channel lies on a segment of another channel (e.g. a pillar starting on a sill),
    a junction node is inserted there so that touching channels are connected.
    """
    spacing = float(channels_cfg.get('node_spacing', 0.25))
    g = nx.Graph()
    node_pos = {}
    node_channels = {}
    all_vertices = [tuple(float(v) for v in p) for ch in channels_cfg['channels'] for p in ch['points']]

    for ch in channels_cfg['channels']:
        pts = [tuple(float(v) for v in p) for p in ch['points']]
        for a0, b0 in zip(pts, pts[1:]):
            stops = [a0] + _junctions_on_segment(a0, b0, all_vertices) + [b0]
            for a, b in zip(stops, stops[1:]):
                seg_len = _dist(a, b)
                n_sub = max(1, int(math.ceil(seg_len / spacing)))
                prev_key = None
                for k in range(n_sub + 1):
                    t = k / n_sub
                    p = tuple(a[i] + t * (b[i] - a[i]) for i in range(3))
                    key = _node_key(p)
                    node_pos.setdefault(key, p)
                    node_channels.setdefault(key, set()).add(ch['id'])
                    if prev_key is not None and prev_key != key:
                        g.add_edge(prev_key, key, length=_dist(node_pos[prev_key], node_pos[key]), channel=ch['id'], kind=ch.get('kind', ''))
                    prev_key = key

    # Stable node names
    names = {key: f"n{idx}" for idx, key in enumerate(sorted(node_pos))}
    return g, node_pos, node_channels, names


def build_voxel_graph(npz_path, reach=2):
    """Routing graph from the voxelised cavity space (routing_space.py): nodes = routable voxels, edges between voxels
    within `reach` voxels of each other (bridges one-voxel gaps), weight = straight-line length. Returns
    (csr_matrix, positions[n,3], kinds[n] names, main-component mask)."""
    d = np.load(npz_path, allow_pickle=False)
    free, kind, origin, s = d['free'], d['kind'], d['origin'], float(d['spacing'])
    kinds = [str(k) for k in d['kinds']]
    idx = np.argwhere(free)
    n = len(idx)
    pos = origin + idx * s
    node_of = -np.ones(free.shape, dtype=np.int64)
    node_of[idx[:, 0], idx[:, 1], idx[:, 2]] = np.arange(n)
    rows, cols, w = [], [], []
    offsets = [(dx, dy, dz) for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1) for dz in range(-reach, reach + 1)
               if (dx, dy, dz) > (0, 0, 0)]  # half of the neighbourhood; the matrix is symmetrised below
    for dx, dy, dz in offsets:
        a = idx + np.array([dx, dy, dz])
        ok = (a[:, 0] >= 0) & (a[:, 0] < free.shape[0]) & (a[:, 1] >= 0) & (a[:, 1] < free.shape[1]) & (a[:, 2] >= 0) & (a[:, 2] < free.shape[2])
        src = np.arange(n)[ok]
        dst = node_of[a[ok, 0], a[ok, 1], a[ok, 2]]
        keep = dst >= 0
        rows.append(src[keep]); cols.append(dst[keep]); w.append(np.full(keep.sum(), s * math.sqrt(dx * dx + dy * dy + dz * dz)))
    rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    g = sparse.coo_matrix((np.concatenate([w, w]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(n, n)).tocsr()
    ncomp, labels = connected_components(g, directed=False)
    main = labels == np.bincount(labels).argmax()
    node_kinds = [kinds[k] for k in kind[idx[:, 0], idx[:, 1], idx[:, 2]]]
    return g, pos, node_kinds, main, ncomp, s


def grid_points(bbox, spacing, max_sites):
    """Cell centres of a regular grid inside bbox; keep the max_sites points nearest to the bbox centre."""
    axes = []
    centre = []
    for ax in ('x', 'y', 'z'):
        lo, hi = float(bbox[ax][0]), float(bbox[ax][1])
        n = max(1, int(round((hi - lo) / spacing)))
        step = (hi - lo) / n
        axes.append([lo + (i + 0.5) * step for i in range(n)])
        centre.append((lo + hi) / 2.0)
    pts = [p for p in product(*axes)]
    pts.sort(key=lambda p: (_dist(p, centre), p))
    pts = pts[:max_sites]
    # Deterministic, readable order: left to right, front to rear, bottom to top
    pts.sort(key=lambda p: (p[0], -p[1], p[2]))
    return pts


def nearest_node(p, node_pos):
    best, best_d = None, float('inf')
    for key, q in node_pos.items():
        d = _dist(p, q)
        if d < best_d:
            best, best_d = key, d
    return best, best_d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config_dir', default='configs')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    cfg_dir = args.config_dir

    vehicle = json.load(open(os.path.join(cfg_dir, 'vehicle.json')))
    sensors = json.load(open(os.path.join(cfg_dir, 'sensors.json')))['sensors']
    actuators = json.load(open(os.path.join(cfg_dir, 'actuators.json')))['actuators']
    thermal = vehicle['thermal_classes']

    npz_path = os.path.join(cfg_dir, 'routing_space.npz')
    voxel_mode = os.path.exists(npz_path)
    if voxel_mode:
        return main_voxel(args, cfg_dir, vehicle, sensors, actuators, thermal, npz_path)

    # ---- channel graph from hand-defined polylines (fallback without the voxelised routing space)
    g, node_pos, node_channels, names = build_channel_graph(vehicle['cable_channels'])
    if not nx.is_connected(g):
        comps = list(nx.connected_components(g))
        sys.exit(f"ERROR: channel graph is not connected ({len(comps)} components). "
                 f"Smallest component: {sorted({c for k in min(comps, key=len) for c in node_channels[k]})}")

    # ---- candidate sites
    zones_cfg = vehicle['ecu_zones']
    spacing = float(zones_cfg.get('grid_spacing', 0.35))
    sites = []
    for z in zones_cfg['zones']:
        pts = grid_points(z['bbox'], spacing, int(z.get('max_sites', 1)))
        for k, p in enumerate(pts):
            sid = z['id'] if len(pts) == 1 else f"{z['id']}_{k + 1}"
            key, drop = nearest_node(p, node_pos)
            sites.append({
                'id': sid,
                'zone': z['id'],
                'thermal_class': z['thermal_class'],
                'derating': float(thermal[z['thermal_class']]['derating']),
                'position': {'x': round(p[0], 3), 'y': round(p[1], 3), 'z': round(p[2], 3)},
                'attach_node': names[key],
                'drop_length': round(drop, 3),
                'source': z.get('source', ''),
            })

    # ---- peripheral attachments
    attachments = {s['id']: {'node': s['attach_node'], 'length': s['drop_length']} for s in sites}
    peripherals = []
    for it in sensors + actuators:
        p = (it['position']['x'], it['position']['y'], it['position']['z'])
        key, drop = nearest_node(p, node_pos)
        attachments[it['id']] = {'node': names[key], 'length': round(drop, 3)}
        peripherals.append(it['id'])

    # ---- shortest-path distances between all attached entities
    ids = [s['id'] for s in sites] + peripherals
    key_of_name = {v: k for k, v in names.items()}
    attach_keys = {i: key_of_name[attachments[i]['node']] for i in ids}
    sp_cache = {}
    matrix = []
    for a in ids:
        ka = attach_keys[a]
        if ka not in sp_cache:
            sp_cache[ka] = nx.single_source_dijkstra_path_length(g, ka, weight='length')
        row = []
        for b in ids:
            kb = attach_keys[b]
            if a == b:
                row.append(0.0)
            else:
                row.append(round(attachments[a]['length'] + sp_cache[ka][kb] + attachments[b]['length'], 4))
        matrix.append(row)

    # ---- write outputs
    stamp = {'_generated_by': 'tools/vehicle_layout/generate_layout.py', 'vehicle': vehicle.get('name', ''),
             '_comment': 'Generated file, do not edit; change configs/vehicle.json and re-run the generator.'}
    json.dump({**stamp, 'grid_spacing': spacing, 'sites': sites},
              open(os.path.join(cfg_dir, 'sites.json'), 'w'), indent=2)
    json.dump({**stamp,
               'node_spacing': vehicle['cable_channels'].get('node_spacing', 0.25),
               'nodes': {names[k]: {'x': round(p[0], 3), 'y': round(p[1], 3), 'z': round(p[2], 3),
                                    'channels': sorted(node_channels[k])} for k, p in node_pos.items()},
               'edges': [{'u': names[u], 'v': names[v], 'length': round(d['length'], 4), 'channel': d['channel'], 'kind': d['kind']}
                         for u, v, d in g.edges(data=True)],
               'attachments': attachments},
              open(os.path.join(cfg_dir, 'channels.json'), 'w'), indent=2)
    json.dump({**stamp, 'unit': 'm', 'ids': ids, 'matrix': matrix},
              open(os.path.join(cfg_dir, 'distances.json'), 'w'), indent=1)

    if not args.quiet:
        total_len = sum(d['length'] for _, _, d in g.edges(data=True))
        print(f"channel graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, {total_len:.1f} m of channels")
        print(f"candidate sites: {len(sites)} in {len(zones_cfg['zones'])} zones (grid spacing {spacing} m)")
        for s in sites:
            p = s['position']
            print(f"  {s['id']:22s} ({p['x']:6.2f},{p['y']:6.2f},{p['z']:5.2f}) {s['thermal_class']:10s} drop {s['drop_length']:.2f} m -> {s['attach_node']}")
        far = sorted(((attachments[i]['length'], i) for i in peripherals), reverse=True)[:3]
        print("peripherals: {} attached; longest drops: {}".format(len(peripherals), ", ".join(f"{i} {d:.2f} m" for d, i in far)))
        n = len(sites)
        site_d = [matrix[i][j] for i in range(n) for j in range(n) if i != j]
        print(f"site-site cable length: min {min(site_d):.2f} m, mean {sum(site_d)/len(site_d):.2f} m, max {max(site_d):.2f} m")
        print(f"written: {cfg_dir}/sites.json, channels.json, distances.json")


def make_sites(vehicle, thermal):
    zones_cfg = vehicle['ecu_zones']
    spacing = float(zones_cfg.get('grid_spacing', 0.35))
    sites = []
    for z in zones_cfg['zones']:
        pts = grid_points(z['bbox'], spacing, int(z.get('max_sites', 1)))
        for k, p in enumerate(pts):
            sites.append({'id': z['id'] if len(pts) == 1 else f"{z['id']}_{k + 1}", 'zone': z['id'],
                          'thermal_class': z['thermal_class'], 'derating': float(thermal[z['thermal_class']]['derating']),
                          'position': {'x': round(p[0], 3), 'y': round(p[1], 3), 'z': round(p[2], 3)}, 'source': z.get('source', '')})
    return sites, spacing


def main_voxel(args, cfg_dir, vehicle, sensors, actuators, thermal, npz_path):
    """Sites + channel graph + distances with the voxelised routing space (cables inside the car structure)."""
    g, pos, node_kinds, main, ncomp, vs = build_voxel_graph(npz_path)
    n_main = int(main.sum())
    d = np.load(npz_path, allow_pickle=False)
    if 'sites' in d.files:   # candidate sites scanned from the ECU space of the reference model
        scanned = json.loads(str(d['sites']))
        params = json.loads(str(d['params']))
        spacing = params['ecu_spacing']
        sites = []
        for n, q in enumerate(scanned, 1):
            sites.append({'id': f"S{n:02d}", 'zone': q['thermal_class'], 'thermal_class': q['thermal_class'],
                          'derating': q['derating'], 'clearance': q['clearance'],
                          'position': {'x': q['x'], 'y': q['y'], 'z': q['z']},
                          'source': f"cell of the scanned ECU space (lattice {params['ecu_spacing']} m, clearance >= {params['ecu_clearance']} m, mount reach {params['mount_reach']} m)"})
    else:
        sites, spacing = make_sites(vehicle, thermal)
    tree = cKDTree(pos[main])
    main_ids = np.flatnonzero(main)

    ids, points = [], []
    for st in sites:
        ids.append(st['id']); points.append([st['position'][c] for c in 'xyz'])
    for it in sensors + actuators:
        ids.append(it['id']); points.append([it['position'][c] for c in 'xyz'])
    drops, nearest = tree.query(np.array(points))
    attach_nodes = main_ids[nearest]
    for st, d, a in zip(sites, drops[:len(sites)], attach_nodes[:len(sites)]):
        st['attach_node'], st['drop_length'] = f"v{a}", round(float(d), 3)
    attachments = {i: {'node': f"v{a}", 'length': round(float(d), 3)} for i, d, a in zip(ids, drops, attach_nodes)}

    # shortest paths between all attached entities (graph length + both drops)
    dist, pred = dijkstra(g, directed=False, indices=attach_nodes, return_predecessors=True)
    col = {a: k for k, a in enumerate(attach_nodes)}
    m = len(ids)
    matrix = [[0.0 if i == j else round(float(drops[i] + dist[i, attach_nodes[j]] + drops[j]), 4) for j in range(m)] for i in range(m)]

    # harness skeleton for visualisation: minimum spanning tree over all entities (graph lengths), i.e. one bundled
    # backbone connecting every site / peripheral, instead of the union of all pairwise shortest paths
    from scipy.sparse.csgraph import minimum_spanning_tree
    dm = np.array(matrix, dtype=float)
    dm[~np.isfinite(dm)] = 0.0
    mst = minimum_spanning_tree(sparse.csr_matrix(dm)).tocoo()
    used_edges, used_nodes = set(), set(attach_nodes.tolist())
    for i, j in zip(mst.row, mst.col):
        v = attach_nodes[j]
        while v != attach_nodes[i] and v >= 0:
            u = pred[i, v]
            if u < 0:
                break
            used_edges.add((min(u, v), max(u, v))); used_nodes.add(int(u)); used_nodes.add(int(v))
            v = u

    stamp = {'_generated_by': 'tools/vehicle_layout/generate_layout.py (voxel routing space)', 'vehicle': vehicle.get('name', ''),
             '_comment': 'Generated file, do not edit; change configs/vehicle.json / routing_space.npz and re-run the generator.'}
    json.dump({**stamp, 'grid_spacing': spacing, 'sites': sites}, open(os.path.join(cfg_dir, 'sites.json'), 'w'), indent=2)
    json.dump({**stamp, 'node_spacing': vs, 'routing_space': {'voxels': int(len(pos)), 'main_component': n_main, 'components': int(ncomp)},
               'nodes': {f"v{v}": {'x': round(float(pos[v, 0]), 3), 'y': round(float(pos[v, 1]), 3), 'z': round(float(pos[v, 2]), 3),
                                   'channels': [node_kinds[v]]} for v in sorted(used_nodes)},
               'edges': [{'u': f"v{u}", 'v': f"v{v}", 'length': round(float(g[u, v]), 4), 'channel': node_kinds[v], 'kind': node_kinds[v]}
                         for u, v in sorted(used_edges)],
               'attachments': attachments}, open(os.path.join(cfg_dir, 'channels.json'), 'w'), indent=1)
    json.dump({**stamp, 'unit': 'm', 'ids': ids, 'matrix': matrix}, open(os.path.join(cfg_dir, 'distances.json'), 'w'), indent=1)

    if not args.quiet:
        print(f"routing space: {len(pos)} routable voxels ({vs} m), {ncomp} components, main component {n_main} voxels")
        print(f"candidate sites: {len(sites)}; harness backbone (MST over {m} entities): {len(used_nodes)} nodes, {len(used_edges)} edges")
        for st in sites:
            p = st['position']
            print(f"  {st['id']:22s} ({p['x']:6.2f},{p['y']:6.2f},{p['z']:5.2f}) {st['thermal_class']:10s} drop {st['drop_length']:.2f} m")
        far = sorted(((attachments[i]['length'], i) for i in ids[len(sites):]), reverse=True)[:4]
        print("peripherals: longest drops: " + ", ".join(f"{i} {d:.2f} m" for d, i in far))
        ns = len(sites)
        sd = [matrix[i][j] for i in range(ns) for j in range(ns) if i != j]
        inf = sum(1 for x in sd if not np.isfinite(x))
        print(f"site-site cable length: min {min(sd):.2f} m, mean {np.mean(sd):.2f} m, max {max(sd):.2f} m; unreachable pairs: {inf}")
        print(f"written: {cfg_dir}/sites.json, channels.json, distances.json")


if __name__ == '__main__':
    main()
