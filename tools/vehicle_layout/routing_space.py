"""
Installation space of the vehicle, derived from the reference car model (runs inside Blender, headless):

  ~/blender/blender-headless --python tools/vehicle_layout/routing_space.py -- [--spacing 0.06] [--site_spacing 0.30]

One scan of the geometry gives two permissible spaces:

* CABLE space: the cavity between the body structure and the trim where harnesses run (behind the body skin incl. roof
  and rockers, inside the pillars, under the floor carpet, inside the dashboard, along the tunnel, behind trunk and
  engine-bay walls): a voxel is routable when it lies on the inner side of the nearest structure surface at a
  distance inside [band_min, band_max], inside the envelope, not inside an obstacle block or an engine-bay component.
  Inside the roof cavity the space is restricted to the rails / bows of vehicle.json 'roof_structure'.

* ECU space: where a control unit can be mounted: a free sphere of radius >= ecu_clearance fits (clearance to every
  solid part, block and glass), the point is enclosed by the car (surfaces within 1.5 m above, ahead, behind, left,
  right), it is mountable (within mount_reach of a structure surface and on its back side, or just above the floor
  carpet / trunk floor), and it is not an occupant seat. Candidate sites are the clearance maxima of the ECU space at
  least site_spacing apart; their thermal class follows the region of the car.

Output: configs/routing_space.npz (origin, spacing, free[nx,ny,nz] bool = cable space, kind[nx,ny,nz] uint8, kinds,
ecu_clear[nx,ny,nz] float = clearance where the ECU space holds else 0, sites = candidate points with clearance and
thermal class). The file is derived from the private reference model and is git-ignored; generate_layout.py builds
the channel graph and the site list from it.
"""
import argparse
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

REF_MODEL = 'assets/Hyundai_Ioniq_5_N_full_model.blend'
# model frame -> vehicle frame (see HANDOFF §15.1): rotate 180° about z, lift by ground clearance, centre lengthwise
MODEL_XF = Matrix.Translation((0, -0.06, 0.16)) @ Matrix.Rotation(math.pi, 4, 'Z')

# structure surfaces behind which a harness may run and on which a control unit may be mounted: name prefix -> kind
STRUCTURE = [
    ('doorJamb', 'pillar'), ('doorSill', 'sill'), ('floor_a', 'floor'), ('dashStructure', 'dash'), ('dashPad', 'dash'),
    ('centerConsoleStructure', 'tunnel'), ('trunkBay', 'trunk'), ('overheadConsole', 'roof'),
    # door cards are deliberately NOT structure: a door harness serves only that door's devices; the main harness never transits a door
    ('bumperF_a', 'bumper'), ('bumperR_a', 'bumper'), ('fenders_a', 'fender'), ('body_a', 'body'),
]
BAY_WALL_TOKENS = ('carPaint|enginebay', 'plastic_RGH|enginebay', 'rubber_rough|enginebay', 'Black|enginebay')
JUNK_TOKENS = ('_race_', 'rollcage', 'platejpn', 'paddleshifter', '_label_', 'seatbelt', 'backfacinggeo')
OBSTACLE_KINDS = ('battery', 'motor', 'wheel', 'seat', 'aux')
FLOOR_KINDS = ('floor', 'trunk')


def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_dir', default='configs')
    ap.add_argument('--spacing', type=float, default=0.06, help='voxel size in metres')
    # object sizes and mounting rules come from vehicle.json 'installation_space'
    ap.add_argument('--ecu_spacing', type=float, default=0.12, help='lattice spacing of the ECU space cells handed to the optimiser (multiple of --spacing)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--snap_peripherals', action='store_true',
                    help='move exterior sensors/actuators onto the nearest point of the car skin (1 cm inside) and rewrite their positions in sensors.json / actuators.json')
    return ap.parse_args(argv)


EXTERIOR_PREFIXES = ('body_a', 'glass', 'bumperF_a', 'bumperR_a', 'fenders_a', 'hood_a', 'trunk_a', 'wingMirror', 'headlight', 'taillight', 'doorLF_a', 'doorRF_a', 'bodyAntenna')
EXTERIOR_DEVICE_TYPES = ('CAMERA', 'LIDAR', 'GPS')


def snap_peripherals(config_dir, objs, inset=0.01):
    """Exterior devices sit on the body skin: snap each to the nearest exterior surface point, moved `inset` inside."""
    skin = bvh_for([o for o in objs if o.name.split()[0].startswith(EXTERIOR_PREFIXES) and 'int_' not in o.name.lower()])
    for fn, key in (('sensors.json', 'sensors'), ('actuators.json', 'actuators')):
        path = os.path.join(config_dir, fn)
        d = json.load(open(path))
        for it in d[key]:
            if it['type'].upper() not in EXTERIOR_DEVICE_TYPES:
                continue
            p = Vector((it['position']['x'], it['position']['y'], it['position']['z']))
            h = nearest(skin, p, 0.6)
            if h is None:
                print(f"  {it['id']}: no skin within 0.6 m, left unchanged")
                continue
            loc, nrm, dist = h
            q = loc - nrm.normalized() * inset
            it['position'] = {'x': round(q.x, 3), 'y': round(q.y, 3), 'z': round(q.z, 3)}
            it['snapped'] = f"moved {dist:.2f} m onto the car skin of the reference model"
            print(f"  {it['id']:22s} -> ({q.x:6.2f},{q.y:6.2f},{q.z:5.2f})  moved {dist:.2f} m")
        json.dump(d, open(path, 'w'), indent=2)


def centred_grid(lo, hi, s):
    """Grid of cell centres with spacing s, symmetric about (lo+hi)/2."""
    n = max(1, int(round((hi - lo) / s)))
    mid = (lo + hi) / 2.0
    return mid + (np.arange(n) - (n - 1) / 2.0) * s


def append_model():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(REF_MODEL, link=False) as (src, dst):
        dst.objects = list(src.objects)
    col = bpy.data.collections.new('Car model')
    bpy.context.scene.collection.children.link(col)
    objs = []
    for o in dst.objects:
        if o is not None and o.type == 'MESH' and not any(t in o.name.lower() for t in JUNK_TOKENS):
            col.objects.link(o)
            o.matrix_world = MODEL_XF @ o.matrix_world
            objs.append(o)
    return objs


def bvh_for(objs):
    verts, polys, off = [], [], 0
    for o in objs:
        m, mw = o.data, o.matrix_world
        verts.extend(mw @ v.co for v in m.vertices)
        polys.extend([i + off for i in p.vertices] for p in m.polygons)
        off += len(m.vertices)
    return BVHTree.FromPolygons(verts, polys) if polys else None


def nearest(tree, p, r):
    """(location, normal, distance) of the nearest surface within r, or None."""
    if tree is None:
        return None
    h = tree.find_nearest(p, r)
    return None if (h is None or h[0] is None) else (h[0], h[1], h[3])


def box_distance(p, b):
    """Distance from p to an axis-aligned box (0 inside)."""
    d = 0.0
    for c, ax in zip((p.x, p.y, p.z), ('x', 'y', 'z')):
        lo, hi = b[ax]
        if c < lo:
            d += (lo - c) ** 2
        elif c > hi:
            d += (c - hi) ** 2
    return math.sqrt(d)


def main():
    args = parse_args()
    t0 = time.time()
    vehicle = json.load(open(os.path.join(args.config_dir, 'vehicle.json')))
    dims = vehicle['dimensions']
    thermal = vehicle['thermal_classes']
    objs = append_model()
    print(f"appended {len(objs)} meshes in {time.time() - t0:.1f} s")
    if args.snap_peripherals:
        print("snapping exterior devices onto the car skin:")
        snap_peripherals(args.config_dir, objs)

    # --- classify meshes
    groups, bay_parts, solids = {}, [], []
    for o in objs:
        tok = o.name.split()[0]
        solids.append(o)
        if tok.startswith(('engineBay_a', 'engine_a')):
            (groups.setdefault('bay', []) if any(w in o.name for w in BAY_WALL_TOKENS) else bay_parts).append(o)
            continue
        for prefix, kind in STRUCTURE:
            if o.name.startswith(prefix):
                groups.setdefault(kind, []).append(o)
                break
    kinds = [k for _, k in STRUCTURE if k in groups] + (['bay'] if 'bay' in groups else [])
    kinds = list(dict.fromkeys(kinds))
    trees = {k: bvh_for(groups[k]) for k in kinds}
    part_tree = bvh_for(bay_parts)
    solid_tree = bvh_for(solids)
    kind_id = {k: i for i, k in enumerate(kinds)}
    print("structure groups:", {k: len(groups[k]) for k in kinds}, "| engine-bay components:", len(bay_parts))

    blocks = [b for b in vehicle['fixed_blocks'] if b['kind'] in OBSTACLE_KINDS]
    seats = [b['bbox'] for b in vehicle['fixed_blocks'] if b['kind'] == 'seat']
    rs = vehicle.get('roof_structure')

    # --- symmetric voxel grid over the body envelope
    s = args.spacing
    hx, hy = dims['width'] / 2, dims['length'] / 2
    xs = centred_grid(-hx, hx, s)
    ys = centred_grid(-hy, hy, s)
    zs = centred_grid(dims['ground_clearance'], dims['height'], s)
    nx, ny, nz = len(xs), len(ys), len(zs)
    free = np.zeros((nx, ny, nz), dtype=bool)
    kind_idx = np.full((nx, ny, nz), 255, dtype=np.uint8)
    ecu_clear = np.zeros((nx, ny, nz), dtype=np.float32)
    ins = vehicle['installation_space']
    r_cable = float(ins['cable_bundle_diameter']) / 2.0
    ecu_dims = sorted(float(v) for v in ins['ecu_size'])
    r_ecu = ecu_dims[1] / 2.0            # conservative sphere test: half of the middle dimension of the ECU box
    reach, floor_h = float(ins['mount_reach']), float(ins.get('floor_mount_height', 0.12))
    r_max = max(r_ecu, reach) + s
    clearance = np.zeros((nx, ny, nz), dtype=np.float32)   # > 0 where the voxel is mountable
    dirs = [Vector(d) for d in ((0, 0, 1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0))]
    print(f"installation space: cable radius {r_cable:.3f} m, ECU fit radius {r_ecu:.3f} m (box {ins['ecu_size']}), mount reach {reach} m")

    t1 = time.time()
    n_cable = n_ecu = n_mount = 0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                p = Vector((x, y, z))
                block_d = min((box_distance(p, b['bbox']) for b in blocks), default=1e9)
                if block_d < r_cable:
                    continue
                comp = nearest(part_tree, p, r_max)
                comp_d = 1e9 if comp is None else comp[2]
                if comp_d < r_cable:
                    continue   # inside or touching an engine-bay component

                # nearest structure surface (per kind): must be within reach and the point on its back side
                # (or just above the floor carpet / trunk floor)
                best = None
                for kname, tree in trees.items():
                    h = nearest(tree, p, reach)
                    if h and (best is None or h[2] < best[3]):
                        best = (h[0], h[1], kname, h[2])
                if best is None:
                    continue
                loc, nrm, kname, dist = best
                behind = nrm.dot(p - loc) < 0.0
                mountable = behind or (kname in FLOOR_KINDS and 0 < (p - loc).z <= floor_h and nrm.z > 0.5)
                if not mountable or any(box_distance(p, sb) == 0.0 for sb in seats):
                    continue
                if not all(solid_tree.ray_cast(p, d, 1.5)[0] is not None for d in dirs):
                    continue   # not enclosed by the car

                sol = nearest(solid_tree, p, r_max)
                clear = min(r_max if sol is None else sol[2], block_d, comp_d)
                if clear < r_cable:
                    continue
                clearance[i, j, k] = clear
                n_mount += 1
                # cable use: inside the roof cavity only along the roof structure strips
                on_strip = True
                if rs and z >= rs['z_min']:
                    on_strip = (abs(x) >= rs['rail_min_abs_x'] or abs(x) <= rs['centre_half_width']
                                or any(abs(y - by) <= rs['bow_half_width'] for by in rs['bows_y']))
                if on_strip:
                    free[i, j, k] = True
                    kind_idx[i, j, k] = kind_id['roof'] if (rs and z >= rs['z_min'] and 'roof' in kind_id) else kind_id[kname]
                    n_cable += 1
                if clear >= r_ecu:
                    ecu_clear[i, j, k] = clear
                    n_ecu += 1
    print(f"mountable voxels {n_mount}; of these cable-usable {n_cable}, ECU-usable {n_ecu}")
    # virtual header beams (windscreen / rear window): routable roof structure the reference model does not contain
    if rs:
        kid_roof = kind_id.setdefault('roof', len(kind_id))
        added = 0
        for hb in rs.get('virtual_headers', []):
            for i, x in enumerate(xs):
                if abs(x) > hb['half_span']:
                    continue
                for j, y in enumerate(ys):
                    if abs(y - hb['y']) > hb['half_width']:
                        continue
                    for k, z in enumerate(zs):
                        if not (hb['z'][0] <= z <= hb['z'][1]) or free[i, j, k]:
                            continue
                        p = Vector((x, y, z))
                        if min((box_distance(p, b['bbox']) for b in blocks), default=1e9) == 0.0:
                            continue
                        free[i, j, k] = True
                        kind_idx[i, j, k] = kid_roof
                        added += 1
        n_cable += added
        print(f"virtual header beams: {added} voxels added")

    # close one-voxel gaps in the cable band (e.g. where a pillar band meets the curved roof corner): a voxel that has
    # band voxels on opposite sides (along x, y or z) becomes routable too, unless it is inside an obstacle
    closed = 0
    for axis in range(3):
        a = np.roll(free, 1, axis=axis) & np.roll(free, -1, axis=axis) & ~free
        edge = [slice(None)] * 3
        edge[axis] = [0, -1]
        a[tuple(edge)] = False   # no wrap-around at the grid boundary
        for i, j, k in zip(*np.nonzero(a)):
            p = Vector((xs[i], ys[j], zs[k]))
            if min((box_distance(p, b['bbox']) for b in blocks), default=1e9) == 0.0:
                continue
            if part_tree is not None:
                near = part_tree.find_nearest(p, OBSTACLE_CLEARANCE if 'OBSTACLE_CLEARANCE' in globals() else 0.03)
                if near is not None and near[0] is not None:
                    continue
            free[i, j, k] = True
            src = (i + (axis == 0), j + (axis == 1), k + (axis == 2))
            kind_idx[i, j, k] = kind_idx[src]
            closed += 1
    n_cable += closed
    print(f"gap closing: {closed} voxels added")
    print(f"voxels {free.size}: cable space {n_cable} ({100.0 * n_cable / free.size:.1f} %), ECU space {n_ecu} "
          f"({100.0 * n_ecu / free.size:.1f} %) in {time.time() - t1:.1f} s")
    for kname, kid in kind_id.items():
        print(f"  {kname:8s} {int((kind_idx == kid).sum()):6d} cable voxels")

    # --- candidate sites: clearance maxima of the ECU space, non-maximum suppression at site_spacing
    def region(x, y, z):
        if y > 0.95 and z < 1.05:
            return 'engine_bay'
        if z >= 1.40:
            return 'roof'
        if y < -1.20:
            return 'trunk'
        if z < 0.36:
            return 'underbody'
        return 'cabin'

    # ECU space handed to the optimiser: EVERY feasible cell of a coarser lattice (step = ecu_spacing / spacing voxels),
    # no hand-picked points; the optimiser decides where control units go.
    step = max(1, int(round(args.ecu_spacing / s)))
    sites = []
    for i, j, k in zip(*np.nonzero(ecu_clear > 0)):
        if i % step or j % step or k % step:
            continue
        x, y, z = float(xs[i]), float(ys[j]), float(zs[k])
        r = region(x, y, z)
        sites.append({'x': round(x, 3), 'y': round(y, 3), 'z': round(z, 3), 'clearance': round(float(ecu_clear[i, j, k]), 3),
                      'thermal_class': r, 'derating': float(thermal[r]['derating'])})
    print(f"ECU space cells for the optimiser: {len(sites)} (lattice {step * s:.2f} m, clearance >= {r_ecu:.3f} m):",
          {r: sum(1 for q in sites if q['thermal_class'] == r) for r in ('engine_bay', 'cabin', 'roof', 'trunk', 'underbody')})

    out = args.out or os.path.join(args.config_dir, 'routing_space.npz')
    np.savez_compressed(out, origin=np.array([xs[0], ys[0], zs[0]]), spacing=s, free=free, kind=kind_idx,
                        kinds=np.array(kinds), clearance=clearance, ecu_clear=ecu_clear,
                        sites=np.array(json.dumps(sites)),
                        params=np.array(json.dumps({'ecu_clearance': r_ecu, 'cable_radius': r_cable, 'ecu_size': ins['ecu_size'], 'mount_reach': reach, 'ecu_spacing': step * s})),
                        source=np.array(['derived from the private IONIQ 5 N reference model; not redistributable']))
    print('written', out, f'({os.path.getsize(out) / 1e3:.0f} kB), total {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
