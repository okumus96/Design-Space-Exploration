"""
Phantom view of the vehicle E/E layout inside the Hyundai IONIQ 5 N reference model (clean technical style on white).

Runs inside Blender (5.2, headless) and, for the compositing step, re-invokes itself with a Python that has PIL:

  ~/blender/blender-headless --python tools/vehicle_viz/build_vehicle.py -- --render [--views iso,top,side] [--res 100]
      [--samples 48] [--labels numbers|names|none] [--band] [--outdir results/vehicle_layout/final/]

Inputs: configs/vehicle.json (axles, battery + motors), sites.json, channels.json, sensors.json, actuators.json, with --band also
configs/routing_space.npz, and the read-only car model assets/Hyundai_Ioniq_5_N_full_model.blend (appended, never saved to).
Outputs (in --outdir): layout.blend, <view>_raw.png (Cycles), <view>_anchors.json (projected marker positions),
<view>_labels.png (--labels names only), <view>.png (composited: raw + badges/labels + legend band + title), legend.png, legend.md.

Frame: x lateral (+right), y longitudinal (+front), z up, z=0 at the ground (configs/vehicle.json).
The car model has its front at -y and z=0 at the underbody; it is converted on import (verified transform). Its wheels are
misplaced and too small; fix_wheels() moves them onto the axles of vehicle.json and scales them to the real tyre radius.
"""
import argparse
import json
import math
import os
import subprocess
import sys

try:
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Matrix, Vector
except ImportError:  # compose step runs under dse/bin/python (PIL), where bpy does not exist
    bpy = None

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

# Linear RGB (Blender). Marker colours are prescribed by the styling direction; channel kinds get distinct saturated hues.
MARKER = {
    'site': (0.05, 0.80, 0.15), 'CAMERA': (0.08, 0.38, 0.98), 'LIDAR': (1.00, 0.45, 0.02), 'IMU': (0.00, 0.80, 0.80),
    'GPS': (0.00, 0.80, 0.80), 'actuator': (0.90, 0.06, 0.10), 'battery': (0.10, 0.45, 0.90), 'motor': (0.85, 0.30, 0.04), 'aux': (0.40, 0.58, 0.36),
    'attach': (0.30, 0.30, 0.33),
}
# Cables are coloured by their interface type only (configs/buses.json). Edges without a type (the example harness
# backbone before an optimisation result exists) are drawn in one neutral grey.
CABLE_TYPE_ORDER = ('ETH', 'CAN', 'LIN', 'FLEXRAY', 'UART', 'SPI')
CABLE_COLOUR = {'ETH': (0.08, 0.38, 0.98), 'CAN': (0.05, 0.70, 0.20), 'LIN': (0.95, 0.80, 0.08), 'FLEXRAY': (0.55, 0.18, 0.88),
                'UART': (0.60, 0.36, 0.10), 'SPI': (0.08, 0.58, 0.52), 'backbone': (0.22, 0.22, 0.26)}
CABLE_NAMES = {'backbone': 'harness backbone (example route, no type yet)'}
CODE_PREFIX = {'CAMERA': 'C', 'LIDAR': 'L', 'IMU': 'I', 'GPS': 'G'}
# Shape coding: ECU site = sphere, camera = clearly smaller sphere, IMU/GPS = small sphere, lidar = flat puck, actuator = outlined cube.
SITE_R, ACT_HALF, LIDAR_H = 0.07, 0.055, 0.05
SHOW_DROPS = False   # set from --show_drops; drops are hidden by default
SHOW_SITES = False   # set from --show_sites; the ECU space is a region, not points
SENSOR_R = {'CAMERA': 0.045, 'LIDAR': 0.07, 'IMU': 0.04, 'GPS': 0.04}
EDGE_RGB, BLOCK_ALPHA = (0.06, 0.06, 0.07), 0.35
LABEL_SIZE = {'site': 0.080, 'sensor': 0.072, 'actuator': 0.072}
TYRE_R = 0.365   # real tyre radius (m), 275/35 R21; the game model's tyres have r ~ 0.245 m

ANCHORS = []  # dicts: id, label, kind, pos, r (marker radius in m) -- every labelled marker, projected per camera
LABELS = []   # dicts: obj, anchor (Vector), r -- 3D text labels (--labels names), placed per camera


# ----------------------------------------------------------------------------------------------------------------- args
def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_dir', default=os.path.join(REPO, 'configs'))
    ap.add_argument('--car', default=os.path.join(REPO, 'assets', 'Hyundai_Ioniq_5_N_full_model.blend'))
    ap.add_argument('--outdir', default=os.path.join(REPO, 'results', 'vehicle_layout', 'final'))
    ap.add_argument('--out', default=None, help='.blend file (default: <outdir>/layout.blend)')
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--views', default='iso,top,side')
    ap.add_argument('--samples', type=int, default=48)
    ap.add_argument('--res', type=int, default=100, help='render resolution percentage (1800x1100 at 100)')
    ap.add_argument('--labels', choices=('numbers', 'names', 'none'), default='numbers',
                    help='numbers: badges with the legend No./code; names: ids as text; none: no labels')
    ap.add_argument('--band', action='store_true', help='draw the permissible cable-routing space (configs/routing_space.npz)')
    ap.add_argument('--show_drops', action='store_true', help='draw the short drops from devices to their channel node (hidden by default)')
    ap.add_argument('--show_sites', action='store_true', help='draw every ECU-space cell as a numbered sphere (default: the ECU space is drawn as a translucent region)')
    ap.add_argument('--python', default=os.path.join(REPO, 'dse', 'bin', 'python'), help='python with PIL for compositing')
    ap.add_argument('--compose', action='store_true', help='(internal) composite labels + legend; needs PIL')
    ap.add_argument('--solution', default=None, help='optimisation result (solution_<i>.json); draws its ECUs and routed cables (<solution>_cables.json from solution_paths.py) instead of the example backbone')
    args = ap.parse_args(argv)
    args.outdir = os.path.abspath(args.outdir)
    args.out = os.path.abspath(args.out) if args.out else os.path.join(args.outdir, 'layout.blend')
    args.views = [v.strip() for v in args.views.split(',') if v.strip()]
    return args


def load_configs(config_dir, solution=None):
    cfg = lambda f: json.load(open(os.path.join(config_dir, f)))
    C = {'vehicle': cfg('vehicle.json'), 'sites': cfg('sites.json')['sites'], 'channels': cfg('channels.json'),
         'sensors': cfg('sensors.json')['sensors'], 'actuators': cfg('actuators.json')['actuators'], 'solution': None, 'cables': None}
    if solution:
        C['solution'] = json.load(open(solution))
        cables = os.path.splitext(solution)[0] + '_cables.json'
        if not os.path.exists(cables):
            raise SystemExit(f'{cables} not found: run dse/bin/python tools/vehicle_layout/solution_paths.py {solution} first')
        C['cables'] = json.load(open(cables))
    return C


def peripheral_codes(sensors, actuators):
    """Short codes: cameras C1..Cn, lidars L1..Ln (sensors.json order), IMU 'I', GPS 'G' (numbered only if several), actuators A1..An."""
    codes, n_type = {}, {}
    for it in sensors:
        n_type[it['type']] = n_type.get(it['type'], 0) + 1
    count = {}
    for it in sensors:
        p = CODE_PREFIX.get(it['type'], 'S')
        count[p] = count.get(p, 0) + 1
        codes[it['id']] = p if n_type[it['type']] == 1 and p in ('I', 'G') else f'{p}{count[p]}'
    for i, it in enumerate(actuators, 1):
        codes[it['id']] = f'A{i}'
    return codes


def code_range(codes, prefix):
    n = sum(1 for c in codes.values() if c[0] == prefix)
    return prefix if n <= 1 else f'{prefix}1..{prefix}{n}'


def cable_type(e):
    return str(e.get('iface', 'backbone')).upper() if e.get('iface') else 'backbone'


def kinds_present(channels):
    present = {cable_type(e) for e in channels['edges']}
    return [k for k in CABLE_TYPE_ORDER if k in present] + sorted(present - set(CABLE_TYPE_ORDER))


def channel_colour(ctype):
    return CABLE_COLOUR.get(ctype, (0.5, 0.5, 0.5))


# ------------------------------------------------------------------------------------------------------------ materials
def mat_principled(name, rgb, alpha=1.0, rough=0.5, emit=0.0):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*rgb, 1.0)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Alpha'].default_value = alpha
    if emit > 0:
        b.inputs['Emission Color'].default_value = (*rgb, 1.0)
        b.inputs['Emission Strength'].default_value = emit
    if alpha < 1.0:
        m.surface_render_method = 'BLENDED'
    m.diffuse_color = (*rgb, alpha)
    return m


def mat_phantom(name, rgb, a_min, a_max, rough=0.45):
    """Transparent shell whose alpha rises towards grazing angles, so silhouettes and edges stay readable."""
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*rgb, 1.0)
    b.inputs['Roughness'].default_value = rough
    lw = nt.nodes.new('ShaderNodeLayerWeight')
    lw.inputs['Blend'].default_value = 0.5
    pw = nt.nodes.new('ShaderNodeMath')
    pw.operation, pw.inputs[1].default_value = 'POWER', 1.5
    ma = nt.nodes.new('ShaderNodeMath')
    ma.operation = 'MULTIPLY_ADD'
    ma.inputs[1].default_value, ma.inputs[2].default_value = a_max - a_min, a_min
    nt.links.new(lw.outputs['Facing'], pw.inputs[0])
    nt.links.new(pw.outputs[0], ma.inputs[0])
    nt.links.new(ma.outputs[0], b.inputs['Alpha'])
    m.surface_render_method = 'BLENDED'
    m.diffuse_color = (*rgb, a_min)
    return m


def mat_flat(name, rgb):
    """Unlit flat colour (labels, leader lines)."""
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = (*rgb, 1.0)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(em.outputs[0], out.inputs['Surface'])
    m.diffuse_color = (*rgb, 1.0)
    return m


# ------------------------------------------------------------------------------------------------------------ helpers
def collection(name):
    c = bpy.data.collections.get(name)
    if not c:
        c = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(c)
    return c


def link(obj, col):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)


def smooth(o):
    for p in o.data.polygons:
        p.use_smooth = True


def mesh_object(name, bm, mat, col, smooth_sides=False):
    """Object from a bmesh (the bmesh is freed). smooth_sides: smooth-shade faces that are not horizontal (cylinder walls)."""
    if smooth_sides:
        for f in bm.faces:
            f.smooth = abs(f.normal.z) < 0.5
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    col.objects.link(o)
    return o


def add_box(name, bbox, mat, col):
    """Axis-aligned box with the size baked into the mesh (object scale 1, so wireframe copies keep an even thickness)."""
    import bmesh
    (x0, x1), (y0, y1), (z0, z1) = bbox['x'], bbox['y'], bbox['z']
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(x1 - x0, y1 - y0, z1 - z0), verts=bm.verts)
    bmesh.ops.translate(bm, vec=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), verts=bm.verts)
    return mesh_object(name, bm, mat, col)


def add_puck(name, p, r, h, mat, col):
    """Short vertical cylinder (lidar marker)."""
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=48, radius1=r, radius2=r, depth=h)
    bmesh.ops.translate(bm, vec=p, verts=bm.verts)
    bm.normal_update()
    return mesh_object(name, bm, mat, col, smooth_sides=True)


def add_edges(o, thickness, col):
    """Dark edge wireframe of a mesh object (a copy with a Wireframe modifier), so translucent or small boxes keep a crisp outline."""
    e = bpy.data.objects.new(f'{o.name}_edges', o.data.copy())
    e.data.materials.clear()
    e.data.materials.append(mat_principled('edges', EDGE_RGB, rough=0.7))
    e.matrix_world = o.matrix_world.copy()
    w = e.modifiers.new('wire', 'WIREFRAME')
    w.thickness, w.use_even_offset, w.use_replace, w.use_boundary = thickness, True, True, True
    e.visible_shadow = False
    col.objects.link(e)
    return e


def add_sphere(name, p, r, mat, col, segments=24):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=max(6, segments // 2), radius=r, location=p)
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    smooth(o)
    link(o, col)
    return o


def add_curve(name, polylines, radius, mat, col, caps=True, resolution=4):
    """One curve object with one POLY spline per polyline, bevelled into a tube."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth, cu.bevel_resolution, cu.use_fill_caps = radius, resolution, caps
    for pts in polylines:
        sp = cu.splines.new('POLY')
        sp.points.add(len(pts) - 1)
        for i, p in enumerate(pts):
            sp.points[i].co = (p[0], p[1], p[2], 1.0)
    cu.materials.append(mat)
    o = bpy.data.objects.new(name, cu)
    col.objects.link(o)
    return o


def add_anchor(ident, label, kind, p, r):
    ANCHORS.append({'id': ident, 'label': label, 'kind': kind, 'pos': tuple(p), 'r': r})


def add_label(name, text, size, anchor, r, col, mat):
    cu = bpy.data.curves.new(f'lbl_{name}', 'FONT')
    cu.body, cu.size = text, size
    cu.align_x, cu.align_y = 'CENTER', 'CENTER'
    cu.materials.append(mat)
    o = bpy.data.objects.new(f'lbl_{name}', cu)
    col.objects.link(o)
    o.location = anchor
    LABELS.append({'obj': o, 'anchor': Vector(anchor), 'r': r})
    return o


def xyz(p):
    return (p['x'], p['y'], p['z'])


def world_bbox(objs):
    """Axis-aligned world bbox (lo, hi) of a list of objects."""
    corners = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    return Vector(map(min, zip(*corners))), Vector(map(max, zip(*corners)))


# ------------------------------------------------------------------------------------------------------------ car model
HIDE_TOKENS = ('_race', 'rollcage', 'platejpn', 'jpn_', 'paddleshifter', 'hoodliner', 'trunkliner', 'wiper', '_label', 'seatbelt')
SHELL_TOKENS = ('body', 'bumper', 'door', 'hood', 'trunk_', 'trunkbadge', 'fenders', 'wing', 'mirrorbase', 'headlight',
                'taillight', 'grille', 'skirt')


def classify(name):
    """Object name 'part_a_LODS0 material' -> 'shell' | 'glass' | 'interior' | 'wheel' | None (do not use)."""
    parts = name.split()
    tok, rest = parts[0].lower(), ' '.join(parts[1:]).lower()
    if any(h in tok for h in HIDE_TOKENS) or 'backfacinggeo' in rest:
        return None
    if tok.startswith('glass'):
        return None if 'int_' in tok else 'glass'          # glass*Int = interior duplicates of the windows
    if tok.startswith(('wheel', 'hyu_caliper', 'hyu_rotor')):
        return 'wheel'
    if tok.startswith(('doorcard', 'doorjamb', 'doorsill')):
        return 'interior'
    if tok.startswith(SHELL_TOKENS):
        return 'glass' if rest.startswith(('gls_', 'mirror')) else 'shell'
    return 'interior'


def import_car(path, col, axles):
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.objects = list(src.objects)
    objs = [o for o in dst.objects if o is not None]
    meshes = [o for o in objs if o.type == 'MESH']
    for o in meshes:                       # link first so parents (if any) are evaluated before we read matrices
        col.objects.link(o)
    bpy.context.view_layer.update()
    n_parent = sum(1 for o in meshes if o.parent is not None)
    xf = Matrix.Translation((0, -0.06, 0.16)) @ Matrix.Rotation(math.pi, 4, 'Z')
    mats = {'shell': mat_phantom('car_shell', (0.80, 0.81, 0.83), 0.10, 0.55),
            'glass': mat_phantom('car_glass', (0.78, 0.86, 0.94), 0.05, 0.30, rough=0.2),
            'interior': mat_principled('car_interior', (0.74, 0.74, 0.76), alpha=0.45, rough=0.9),
            'wheel': mat_principled('car_wheel', (0.16, 0.16, 0.17), rough=0.6)}
    counts, wheels = {}, []
    drop = [o for o in objs if o.type != 'MESH']
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = xf @ mw
        grp = classify(o.name)
        if grp is None:
            drop.append(o)
            counts['hidden'] = counts.get('hidden', 0) + 1
            continue
        if o.data.materials:
            o.data.materials[0] = mats[grp]
        else:
            o.data.materials.append(mats[grp])
        if grp != 'wheel':
            o.visible_shadow = False
        else:
            wheels.append(o)
        o['part_group'] = grp
        counts[grp] = counts.get(grp, 0) + 1
    for o in drop:
        bpy.data.objects.remove(o)
    print(f'car: {len(meshes)} meshes, {n_parent} with parents, groups={counts}')
    fix_wheels(wheels, axles)


def fix_wheels(wheels, axles):
    """The game model's wheels sit ~0.2 m off the axles and are ~1.5x too small. Group the wheel parts by the quadrant of their
    world bbox centre, move each group onto its axle (vehicle.json), scale it rigidly about its bbox centre so the tyre radius
    becomes TYRE_R, and set its lowest point on the ground (z = 0)."""
    groups = {}
    for o in wheels:
        lo, hi = world_bbox([o])
        c = (lo + hi) / 2
        groups.setdefault((c.y > 0, c.x > 0), []).append(o)
    for (front, right), members in sorted(groups.items(), reverse=True):
        lo, hi = world_bbox(members)
        c = (lo + hi) / 2
        y_axle = axles['front_y'] if front else axles['rear_y']
        s = TYRE_R / ((hi.z - lo.z) / 2)
        c_axle = Vector((c.x, y_axle, c.z))
        M = Matrix.Translation(c_axle) @ Matrix.Scale(s, 4) @ Matrix.Translation(-c_axle) @ Matrix.Translation((0, y_axle - c.y, 0))
        for o in members:
            o.matrix_world = M @ o.matrix_world
        lo, hi = world_bbox(members)
        for o in members:
            o.matrix_world = Matrix.Translation((0, 0, -lo.z)) @ o.matrix_world
        lo, hi = world_bbox(members)
        c = (lo + hi) / 2
        print(f"wheel {'F' if front else 'R'}{'R' if right else 'L'}: {len(members)} objects, scaled x{s:.3f}, "
              f'centre=({c.x:+.3f}, {c.y:+.3f}, {c.z:+.3f}), tyre radius {(hi.z - lo.z) / 2:.3f} m')


# ------------------------------------------------------------------------------------------------------------ layout
def build_layout(C, labels_mode):
    vehicle, sites, channels, sensors, actuators = C['vehicle'], C['sites'], C['channels'], C['sensors'], C['actuators']
    codes = peripheral_codes(sensors, actuators)
    col_blocks, col_sites, col_ch, col_att = collection('Blocks'), collection('Sites'), collection('Channels'), collection('Attachments')
    col_sen, col_act = collection('Sensors'), collection('Actuators')

    for b in vehicle['fixed_blocks']:   # translucent so markers behind them stay visible; dark edges keep the outline
        if b['kind'] in ('battery', 'motor', 'aux') and b.get('draw', True):
            o = add_box(b['id'], b['bbox'], mat_principled(f"blk_{b['kind']}", MARKER[b['kind']], alpha=BLOCK_ALPHA, rough=0.5, emit=0.25), col_blocks)
            o.visible_shadow = False
            add_edges(o, 0.012, col_blocks)

    if SHOW_SITES:   # individual cells only on request; by default the ECU space is drawn as a region (build_ecu_space)
        m_site = mat_principled('site', MARKER['site'], rough=0.4, emit=0.35)
        for i, s in enumerate(sites, 1):
            p = xyz(s['position'])
            add_sphere(f"site_{s['id']}", p, SITE_R, m_site, col_sites)
            add_anchor(s['id'], str(i), 'site', p, SITE_R)

    nodes = channels['nodes']
    if C.get('cables'):   # optimisation result: routed cables coloured by interface, ECU boxes at the used cells
        ecu = sorted(float(v) for v in vehicle['installation_space']['ecu_size'])
        hx, hy, hz = ecu[2] / 2, ecu[1] / 2, ecu[0] / 2   # longest side across the car, shortest side vertical
        pos_site = {s['id']: s['position'] for s in sites}
        col_ecu = collection('ECUs')
        for n, sid in enumerate(C['cables']['used_sites'], 1):
            p = pos_site[sid]
            o = add_box(f"ecu_{sid}", {'x': [p['x'] - hx, p['x'] + hx], 'y': [p['y'] - hy, p['y'] + hy], 'z': [p['z'] - hz, p['z'] + hz]},
                        mat_principled('ecu', MARKER['site'], alpha=0.9, rough=0.5, emit=0.2), col_ecu)
            add_edges(o, 0.008, col_ecu)
            add_anchor(sid, f"E{n}", 'site', xyz(p), SITE_R)
        for c in C['cables']['cables']:
            kind = cable_type(c)
            m = mat_principled(f'ch_{kind}', channel_colour(kind), rough=0.45, emit=0.3)
            add_curve(f"cable_{c['src']}__{c['dst']}", [[tuple(a), tuple(b)] for a, b in zip(c['points'], c['points'][1:])], 0.014, m, col_ch)
        channels = {'nodes': nodes, 'edges': [], 'attachments': channels['attachments']}
    by_channel, node_kind = {}, {}
    for e in channels['edges']:
        by_channel.setdefault(e['channel'], []).append(e)
        node_kind.setdefault(e['u'], cable_type(e))
        node_kind.setdefault(e['v'], cable_type(e))
    for ch_id, edges in by_channel.items():
        kind = cable_type(edges[0])
        m = mat_principled(f'ch_{kind}', channel_colour(kind), rough=0.45, emit=0.3)
        add_curve(f'ch_{ch_id}', [[xyz(nodes[e['u']]), xyz(nodes[e['v']])] for e in edges], 0.02, m, col_ch)
    for nid, kind in node_kind.items():   # small spheres at the graph nodes so tube joints look continuous
        add_sphere(f'node_{nid}', xyz(nodes[nid]), 0.021, mat_principled(f'ch_{kind}', channel_colour(kind)), col_ch, segments=12)

    positions = {s['id']: s['position'] for s in sites}
    positions.update({it['id']: it['position'] for it in sensors + actuators})
    att = [[xyz(positions[ent]), xyz(nodes[a['node']])] for ent, a in channels['attachments'].items() if a['length'] > 1e-6]
    if SHOW_DROPS and att:
        add_curve('attachments', att, 0.006, mat_principled('attach', MARKER['attach'], rough=0.8), col_att, resolution=2)

    for it in sensors:
        p, t = xyz(it['position']), it['type']
        m = mat_principled(f'sensor_{t}', MARKER.get(t, MARKER['IMU']), rough=0.4, emit=0.35)
        r = SENSOR_R.get(t, SENSOR_R['IMU'])
        if t == 'LIDAR':
            add_puck(it['id'], p, r, LIDAR_H, m, col_sen)
        else:
            add_sphere(it['id'], p, r, m, col_sen)
        add_anchor(it['id'], codes[it['id']], t, p, r)
    m_act = mat_principled('actuator', MARKER['actuator'], rough=0.4, emit=0.35)
    for it in actuators:
        p = xyz(it['position'])
        h = ACT_HALF
        o = add_box(it['id'], {'x': [p[0] - h, p[0] + h], 'y': [p[1] - h, p[1] + h], 'z': [p[2] - h, p[2] + h]}, m_act, col_act)
        add_edges(o, 0.008, col_act)
        add_anchor(it['id'], codes[it['id']], 'actuator', p, h * 1.4)

    if labels_mode == 'names':
        col_lab, m_label = collection('Labels'), mat_flat('label_black', (0.0, 0.0, 0.0))
        for a in ANCHORS:
            size = LABEL_SIZE[a['kind'] if a['kind'] in ('site', 'actuator') else 'sensor']
            add_label(a['id'], a['id'], size, a['pos'], a['r'], col_lab, m_label)


# -------------------------------------------------------------------------------------------------------- cable space
def voxel_surface(occ, origin, spacing):
    """Outer faces of the union of cubes (edge = spacing) centred on the True voxels of `occ`: (verts, quads).
    Faces between two occupied voxels are dropped, so a translucent material shows one shell per region, not every cube."""
    import numpy as np
    vid, verts, faces = {}, [], []

    def vert(i, j, k):
        key = (i, j, k)
        if key not in vid:
            vid[key] = len(verts)
            verts.append(tuple(origin[a] + (key[a] - 0.5) * spacing for a in range(3)))
        return vid[key]

    pad = np.pad(occ, 1)
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for s in (1, -1):
            nb = np.roll(pad, -s, axis=a)[1:-1, 1:-1, 1:-1]     # occupancy of the neighbour in direction s along axis a
            corners = ((0, 0), (1, 0), (1, 1), (0, 1)) if s > 0 else ((0, 0), (0, 1), (1, 1), (1, 0))   # outward winding
            for idx in zip(*np.nonzero(occ & ~nb)):
                base = list(idx)
                base[a] += 1 if s > 0 else 0
                quad = []
                for db, dc in corners:
                    v = list(base)
                    v[b] += db
                    v[c] += dc
                    quad.append(vert(*v))
                faces.append(quad)
    return verts, faces


def build_installation_space(path):
    """The one installation space (routing_space.npz): every mountable voxel, coloured by what fits there.
    grey = a cable bundle fits (clearance >= cable radius); green = a control unit fits as well (clearance >= ECU fit radius).
    One collection, one data set; the colours are two thresholds on the same clearance map."""
    import bmesh
    import numpy as np
    d = np.load(path)
    origin, spacing = [float(v) for v in d['origin']], float(d['spacing'])
    free = d['free']
    ecu = d['ecu_clear'] > 0 if 'ecu_clear' in d.files else np.zeros_like(free)
    col = collection('Installation space')
    classes = [('space_cable', free & ~ecu, (0.50, 0.50, 0.55), 0.12, 'cable bundle fits'),
               ('space_ecu', ecu, MARKER['site'], 0.35, 'control unit fits')]
    for name, occ, rgb, alpha, label in classes:
        if not occ.any():
            continue
        verts, faces = voxel_surface(occ, origin, spacing)
        bm = bmesh.new()
        bv = [bm.verts.new(v) for v in verts]
        for f in faces:
            bm.faces.new([bv[i] for i in f])
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mat_principled(name, rgb, alpha=alpha, rough=0.6, emit=0.35))
        o = bpy.data.objects.new(name, me)
        o.visible_shadow = False
        o['meaning'] = label
        col.objects.link(o)
        print(f'installation space / {label}: {int(occ.sum())} voxels, {len(faces)} outer faces')


# ------------------------------------------------------------------------------------------------------- environment
def build_env(scene, vehicle, args):
    col = collection('Environment')
    L = vehicle['dimensions']['length']
    bpy.ops.mesh.primitive_plane_add(size=18.0, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = 'ground'
    ground.is_shadow_catcher = True
    ground.data.materials.append(mat_principled('ground', (1.0, 1.0, 1.0), rough=1.0))
    link(ground, col)

    bpy.ops.object.empty_add(location=(0, 0.1, 0.40))   # low target: the car sits in the upper part of the iso frame
    target = bpy.context.active_object
    target.name = 'camera_target'
    link(target, col)

    sun_data = bpy.data.lights.new('sun', 'SUN')
    sun_data.energy, sun_data.angle = 0.9, math.radians(8)
    sun = bpy.data.objects.new('sun', sun_data)
    sun.location = (-5, 6, 10)
    c = sun.constraints.new('TRACK_TO')
    c.target, c.track_axis, c.up_axis = target, 'TRACK_NEGATIVE_Z', 'UP_Y'
    col.objects.link(sun)

    world = bpy.data.worlds.new('World')
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value, bg.inputs['Strength'].default_value = (1, 1, 1, 1), 1.0
    scene.world = world

    cams = {}
    # iso: perspective 3/4 view from front-left (x<0, y>0), ~27 deg above the horizon
    az, el, dist = math.radians(45), math.radians(27), 7.7
    d = Vector((-math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)))
    cam = bpy.data.objects.new('cam_iso', bpy.data.cameras.new('cam_iso'))
    cam.location = Vector(target.location) + d * dist
    cam.data.lens = 45
    c = cam.constraints.new('TRACK_TO')
    c.target, c.track_axis, c.up_axis = target, 'TRACK_NEGATIVE_Z', 'UP_Y'
    cams['iso'] = cam
    # top: orthographic, image right = -y -> FRONT ON THE LEFT; car shifted up to leave room for the legend
    cam = bpy.data.objects.new('cam_top', bpy.data.cameras.new('cam_top'))
    cam.location, cam.rotation_euler = (-0.18, 0.0, 10.0), (0, 0, math.radians(-90))
    cam.data.type, cam.data.ortho_scale = 'ORTHO', L + 1.0
    cams['top'] = cam
    # side: orthographic from +x (right side of the car), front on the right
    cam = bpy.data.objects.new('cam_side', bpy.data.cameras.new('cam_side'))
    cam.location, cam.rotation_euler = (10.0, 0.0, 0.90), (math.radians(90), 0, math.radians(90))
    cam.data.type, cam.data.ortho_scale = 'ORTHO', L + 0.85
    cams['side'] = cam
    for cam in cams.values():
        cam.data.clip_end = 100
        col.objects.link(cam)
    scene.camera = cams['iso']

    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    try:
        scene.cycles.denoising_use_gpu = True
    except Exception:
        pass
    scene.cycles.max_bounces, scene.cycles.transparent_max_bounces = 8, 128 if args.band else 64
    scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False
    scene.render.resolution_x, scene.render.resolution_y = 1800, 1100
    scene.render.resolution_percentage = args.res
    scene.render.film_transparent = False
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform, scene.view_settings.look = 'Standard', 'None'
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.refresh_devices()
        for dev in prefs.devices:
            dev.use = dev.type == 'CUDA'
        scene.cycles.device = 'GPU'
        print('cycles devices:', [(dev.name, dev.type, dev.use) for dev in prefs.devices])
    except Exception as e:
        print('Cycles GPU not available, using CPU:', e)
    return cams


# ------------------------------------------------------------------------------------------------------ label layout
def place_labels(scene, cam):
    """Billboard every 3D text label towards `cam` and push it to the least-crowded spot around its marker (image space)."""
    bpy.context.view_layer.update()
    R = cam.matrix_world.to_3x3()
    right, up = (R @ Vector((1, 0, 0))).normalized(), (R @ Vector((0, 1, 0))).normalized()
    rot = cam.matrix_world.to_euler()

    def rect(center, w, h):
        uv = [world_to_camera_view(scene, cam, center + right * (sx * w / 2) + up * (sy * h / 2)) for sx in (-1, 1) for sy in (-1, 1)]
        return (min(p.x for p in uv), max(p.x for p in uv), min(p.y for p in uv), max(p.y for p in uv))

    def overlap(a, b):
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[2], b[2]))

    def outside(a):
        return max(0.0, 0.01 - a[0]) + max(0.0, a[1] - 0.99) + max(0.0, 0.12 - a[2]) + max(0.0, a[3] - 0.99)

    for L in LABELS:
        L['obj'].rotation_euler = rot
    bpy.context.view_layer.update()
    markers = [rect(L['anchor'], 2 * L['r'], 2 * L['r']) for L in LABELS]
    placed, leaders = [], []
    order = sorted(range(len(LABELS)), key=lambda i: -(world_to_camera_view(scene, cam, LABELS[i]['anchor']).z))  # far first
    for i in order:
        L = LABELS[i]
        o, A, r = L['obj'], L['anchor'], L['r']
        w, h = o.dimensions.x, o.dimensions.y
        gap = 0.03
        ux, uy = r + gap + w / 2, r + gap + h / 2
        own = rect(A, w, h)
        own_area = max(1e-9, (own[1] - own[0]) * (own[3] - own[2]))   # overlaps are scored as a fraction of the label's own area
        ring_pen = 0.02                                                 # a 2 % overlap is preferred to moving one ring further out
        best, best_score = None, None
        for k in range(5):
            ring = [(0, uy + k * (h + 0.03)), (0, -(uy + k * (h + 0.03))), (ux + k * 0.16, 0), (-(ux + k * 0.16), 0),
                    (ux * 0.7 + k * 0.16, uy + k * (h + 0.03)), (-(ux * 0.7 + k * 0.16), uy + k * (h + 0.03)),
                    (ux * 0.7 + k * 0.16, -(uy + k * (h + 0.03))), (-(ux * 0.7 + k * 0.16), -(uy + k * (h + 0.03)))]
            for dx, dy in ring:
                C = A + right * dx + up * dy
                rc = rect(C, w, h)
                score = (sum(overlap(rc, p) for p in placed) + 2.0 * sum(overlap(rc, m) for j, m in enumerate(markers) if j != i)) / own_area
                score += outside(rc) * 50.0 + ring_pen * k
                if best_score is None or score < best_score:
                    best, best_score = (C, rc, dx, dy, k), score
                if score <= ring_pen * k + 1e-9:
                    break
            if best_score is not None and best_score <= ring_pen * best[4] + 1e-9:
                break
        C, rc, dx, dy, k = best
        o.location = C
        placed.append(rc)
        if k >= 1:  # label pushed away from its marker: draw a leader line to the nearest label edge
            edge = C - right * (w / 2 if dx > 1e-6 else -w / 2 if dx < -1e-6 else 0.0) - up * (h / 2 if dy > 1e-6 else -h / 2 if dy < -1e-6 else 0.0)
            leaders.append([tuple(A), tuple(edge)])
    old = bpy.data.objects.get('leaders')
    if old:
        cu = old.data
        bpy.data.objects.remove(old)
        bpy.data.curves.remove(cu)
    if leaders:
        add_curve('leaders', leaders, 0.005, mat_flat('leader_grey', (0.12, 0.12, 0.12)), bpy.data.collections['Labels'], resolution=1)


def project_anchors(scene, cam, length):
    """Marker positions in normalised image coordinates of `cam` (u right, v up, depth along the view) for the badge pass,
    plus two points just ahead of the front bumper (along +y) from which the composite draws the FRONT arrow."""
    bpy.context.view_layer.update()
    right = (cam.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized()
    out = []
    for a in ANCHORS:
        p = Vector(a['pos'])
        c = world_to_camera_view(scene, cam, p)
        e = world_to_camera_view(scene, cam, p + right * a['r'])
        out.append({'id': a['id'], 'label': a['label'], 'kind': a['kind'], 'u': c.x, 'v': c.y, 'depth': c.z, 'ru': abs(e.x - c.x)})
    f0 = world_to_camera_view(scene, cam, Vector((0, length / 2 + 0.15, 0.45)))
    f1 = world_to_camera_view(scene, cam, Vector((0, length / 2 + 0.65, 0.45)))
    return {'anchors': out, 'front': [[f0.x, f0.y], [f1.x, f1.y]]}


# ------------------------------------------------------------------------------------------------------------ render
def render_views(scene, cams, args, outdir, length):
    labels_col = bpy.data.collections.get('Labels')
    others = [c for c in scene.collection.children if c.name != 'Labels']
    ground = bpy.data.objects.get('ground')
    for view in args.views:
        cam = cams[view]
        scene.camera = cam
        with open(os.path.join(outdir, f'{view}_anchors.json'), 'w') as f:
            json.dump(dict(view=view, **project_anchors(scene, cam, length)), f, indent=1)
        if ground:
            ground.hide_render = view == 'top'   # the cast shadow next to the car only clutters the top view
        # main pass: everything but the labels
        if labels_col:
            labels_col.hide_render = True
        scene.render.film_transparent = False
        scene.cycles.samples, scene.cycles.use_denoising = args.samples, True
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.filepath = os.path.join(outdir, f'{view}_raw.png')
        bpy.ops.render.render(write_still=True)
        print('rendered', scene.render.filepath)
        if args.labels == 'names':
            # label pass: labels only on a transparent film (composited on top afterwards, never occluded by the car)
            place_labels(scene, cam)
            labels_col.hide_render = False
            for c in others:
                c.hide_render = True
            scene.render.film_transparent = True
            scene.cycles.samples, scene.cycles.use_denoising = 16, False
            scene.render.image_settings.color_mode = 'RGBA'
            scene.render.filepath = os.path.join(outdir, f'{view}_labels.png')
            bpy.ops.render.render(write_still=True)
            print('rendered', scene.render.filepath)
            for c in others:
                c.hide_render = False
            scene.render.film_transparent = False
            scene.render.image_settings.color_mode = 'RGB'
        if ground:
            ground.hide_render = False


def build(args):
    bad = [v for v in args.views if v not in ('iso', 'top', 'side')]
    if bad:
        raise SystemExit(f'unknown views {bad}; choose from iso, top, side')
    C = load_configs(args.config_dir, args.solution)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    import_car(args.car, collection('Car'), C['vehicle']['axles'])
    global SHOW_DROPS, SHOW_SITES
    SHOW_DROPS, SHOW_SITES = args.show_drops, args.show_sites
    build_layout(C, args.labels)
    band_path = os.path.join(args.config_dir, 'routing_space.npz')
    if os.path.exists(band_path):
        build_installation_space(band_path)   # always in the .blend; rendered only with --band
        band_col = bpy.data.collections.get('Installation space')
        if band_col is not None and not args.band:
            band_col.hide_render = True
    for c in [c for c in bpy.data.collections if len(c.objects) == 0 and len(c.children) == 0]:
        bpy.data.collections.remove(c)   # no empty collections (e.g. Attachments when drops are hidden)
    cams = build_env(scene, C['vehicle'], args)
    bpy.context.view_layer.update()
    if args.labels == 'names':
        place_labels(scene, cams['iso'])

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0   # no layout.blend1 backups
    bpy.ops.wm.save_as_mainfile(filepath=args.out)
    print('saved', args.out)
    if args.render:
        render_views(scene, cams, args, args.outdir, C['vehicle']['dimensions']['length'])
        cmd = [args.python, os.path.abspath(__file__), '--compose', '--outdir', args.outdir, '--views', ','.join(args.views),
               *(['--solution', args.solution] if args.solution else []),
               '--config_dir', args.config_dir, '--labels', args.labels] + (['--band'] if args.band else [])
        print('compose:', ' '.join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print('compose step failed with code', r.returncode)


# ------------------------------------------------------------------------------------------------ compose (PIL step)
def srgb(c):
    return tuple(int(round(255 * (1.055 * v ** (1 / 2.4) - 0.055 if v > 0.0031308 else 12.92 * v))) for v in c)


def hex_colour(c):
    return '#%02x%02x%02x' % srgb(c)


TITLES = {'iso': 'IONIQ 5 N E/E layout - perspective from front-left, above',
          'top': 'Top view - FRONT OF THE CAR ON THE LEFT (roof + glass transparent)',
          'side': 'Side view from the right (+x) - front on the right (shell transparent)'}
DARK, GREY_OUTLINE = (20, 20, 20), (40, 40, 40)


def fonts(size):
    from PIL import ImageFont
    try:
        import matplotlib
        d = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
        return ImageFont.truetype(os.path.join(d, 'DejaVuSans.ttf'), size), ImageFont.truetype(os.path.join(d, 'DejaVuSans-Bold.ttf'), size)
    except Exception:
        f = ImageFont.load_default(size=size)
        return f, f


def marker_entries(codes, labels_mode, n_sites, band, compact):
    """Colour-key rows (swatch kind, linear rgb, text) for the markers and blocks; swatch shapes follow the 3D shape coding."""
    numbered = labels_mode == 'numbers'
    if compact:
        site = ('ECU site (No. = legend)' if numbered else 'ECU site (label = id)' if labels_mode == 'names' else 'ECU site') if SHOW_SITES else 'installation space: green = ECU fits'
        cam, lid, imu, act = ('camera (C#)', 'lidar (L#)', 'IMU (I) / GPS (G)', 'actuator (A#)') if numbered else ('camera', 'lidar', 'IMU / GPS', 'actuator')
        bat, mot, car, space = 'battery', 'e-motor', 'car body / interior', 'installation space: grey = cable fits'
        aux = 'aux. unit'
    else:
        site = (f'candidate ECU site ({n_sites}; ' + ('badge = No. below)' if numbered else 'label = site id)' if labels_mode == 'names' else 'unlabelled)')) if SHOW_SITES else f'installation space, green where a control unit fits ({n_sites} cells handed to the optimiser)'
        cam, lid, imu, act = 'camera (small sphere)', 'lidar (puck)', 'IMU / GPS (small sphere)', 'actuator (outlined cube)'
        if numbered:
            cam, lid = f"camera (small sphere, {code_range(codes, 'C')})", f"lidar (puck, {code_range(codes, 'L')})"
            act, imu = f"actuator (outlined cube, {code_range(codes, 'A')})", f"IMU ({code_range(codes, 'I')}) / GPS ({code_range(codes, 'G')}), small spheres"
        bat, mot, car = 'battery pack (translucent box)', 'e-motor front / rear (translucent box)', 'IONIQ 5 N body / interior (transparent)'
        aux = 'auxiliary units: HVAC box, 12 V battery, charger (translucent box)'
        space = 'installation space, grey where only a cable bundle fits (same clearance map, lower threshold)'
    e = [('circle' if SHOW_SITES else 'block', MARKER['site'], site), ('small', MARKER['CAMERA'], cam), ('puck', MARKER['LIDAR'], lid), ('tiny', MARKER['IMU'], imu),
         ('square', MARKER['actuator'], act), ('block', MARKER['battery'], bat), ('block', MARKER['motor'], mot), ('block', MARKER['aux'], aux)]
    if SHOW_DROPS:
        e.append(('thin', MARKER['attach'], 'attachment' if compact else 'attachment to channel'))
    e.append(('ghost', None, car))
    if band:
        e.append(('ghost', None, space))
    return e


def draw_swatch(draw, kind, rgb, x, y, s):
    """Legend swatch of size s at (x, y); kinds mirror the 3D marker shapes."""
    if kind == 'text':
        return
    col = srgb(rgb) if rgb else None
    cx, cy = x + s / 2, y + s / 2
    if kind == 'circle':
        draw.ellipse([x, y, x + s, y + s], fill=col, outline=GREY_OUTLINE)
    elif kind == 'small':
        r = s * 0.33
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=GREY_OUTLINE)
    elif kind == 'tiny':
        r = s * 0.27
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=GREY_OUTLINE)
    elif kind == 'puck':
        draw.rounded_rectangle([x, cy - s * 0.22, x + s, cy + s * 0.22], radius=s * 0.2, fill=col, outline=GREY_OUTLINE)
    elif kind == 'square':
        draw.rectangle([x + 2, y + 2, x + s - 2, y + s - 2], fill=col, outline=GREY_OUTLINE, width=2)
    elif kind == 'block':
        pale = tuple(int(round(v * 0.35 + 255 * 0.65)) for v in srgb(rgb))
        draw.rectangle([x + 1, y + 1, x + s - 1, y + s - 1], fill=pale, outline=GREY_OUTLINE, width=2)
    elif kind == 'line':
        draw.line([x, cy, x + s, cy], fill=col, width=6)
    elif kind == 'thin':
        draw.line([x, cy, x + s, cy], fill=col, width=2)
    else:  # ghost
        draw.rectangle([x + 1, y + 1, x + s - 1, y + s - 1], fill=(225, 226, 230), outline=(150, 150, 155))


BADGE_D, BADGE_D2, BADGE_FONT = 26, 30, 17   # px at 1800 px width: >= 7 pt when printed 8.5 cm wide; 2-character codes get a wider ring


def badge_diameter(label, scale):
    return round((BADGE_D if len(label) < 2 else BADGE_D2) * scale)


def draw_badge(draw, cx, cy, text, outline, width, font, D):
    draw.ellipse([cx - D / 2, cy - D / 2, cx + D / 2, cy + D / 2], fill=(255, 255, 255, 255), outline=outline, width=width)
    draw.text((cx, cy + 0.5), text, fill=(0, 0, 0), font=font, anchor='mm')


def badge_style(kind, scale):
    """(outline colour, outline width): thin dark ring for sites, marker colour for sensors / actuators."""
    if kind == 'site':
        return GREY_OUTLINE, max(1, round(1.2 * scale))
    return srgb(MARKER.get(kind, MARKER['IMU'])), max(1, round(2.2 * scale))


def draw_badges(img, anchors, scale, keep_out):
    """Numbered badges centred on the projected markers; a badge that would collide is moved to a nearby free spot with a leader."""
    from PIL import ImageDraw
    W, H = img.size
    draw = ImageDraw.Draw(img, 'RGBA')
    font = fonts(max(8, round(BADGE_FONT * scale)))[0]
    items = [dict(a, x=a['u'] * W, y=(1 - a['v']) * H, D=badge_diameter(a['label'], scale), r=a['ru'] * W) for a in anchors]
    items.sort(key=lambda a: a['depth'])   # nearest first: they keep their marker, farther ones are pushed aside

    def score(it, cx, cy, placed):
        D, s = it['D'], 0.0
        for px, py, pD in placed:                                     # other badges
            s += max(0.0, (D + pD) / 2 + 2 * scale - math.hypot(cx - px, cy - py)) / D
        for o in items:                                               # other markers (half weight)
            if o is not it:
                s += 0.5 * max(0.0, D / 2 + o['r'] - math.hypot(cx - o['x'], cy - o['y'])) / D
        for x0, y0, x1, y1 in keep_out:                               # title box, legend band
            if cx + D / 2 > x0 and cx - D / 2 < x1 and cy + D / 2 > y0 and cy - D / 2 < y1:
                s += 10.0
        if cx - D / 2 < 2 or cx + D / 2 > W - 2 or cy - D / 2 < 2 or cy + D / 2 > H - 2:
            s += 10.0
        return s

    placed, leaders, badges = [], [], []
    for it in items:
        best = None
        for k in range(6):
            cands = [(it['x'], it['y'])] if k == 0 else [(it['x'] + it['D'] * (0.2 + 0.8 * k) * math.cos(t), it['y'] - it['D'] * (0.2 + 0.8 * k) * math.sin(t))
                                                         for t in (math.radians(90 + 45 * j) for j in range(8))]
            for cx, cy in cands:
                sc = score(it, cx, cy, placed) + 0.15 * k
                if best is None or sc < best[0]:
                    best = (sc, cx, cy, k)
            if best[0] <= 0.15 * k + 1e-9:
                break
        _, cx, cy, k = best
        placed.append((cx, cy, it['D']))
        if k > 0:
            d = math.hypot(cx - it['x'], cy - it['y'])
            ex, ey = cx - (cx - it['x']) / d * it['D'] / 2, cy - (cy - it['y']) / d * it['D'] / 2
            leaders.append(((it['x'], it['y']), (ex, ey)))
        badges.append((cx, cy, it))
    lw = max(1, round(1.3 * scale))
    for a, b in leaders:                 # leaders first so the badges cover their ends
        draw.line([a, b], fill=GREY_OUTLINE, width=lw)
    for cx, cy, it in badges:
        outline, width = badge_style(it['kind'], scale)
        draw_badge(draw, cx, cy, it['label'], outline, width, font, it['D'])
    print(f'badges: {len(badges)}, {len(leaders)} moved off their marker')


def band_rows(draw, entries, font, W, pad=14, s=18, gap_x=22, inner=8):
    """Flow layout of the legend strip: rows of (entry, width)."""
    widths = [(0 if k == 'text' else s + inner) + draw.textlength(t, font=font) for k, _, t in entries]
    rows, cur, cur_w = [], [], 0
    for e, w in zip(entries, widths):
        if cur and cur_w + w > W - 4 * pad:
            rows.append(cur)
            cur, cur_w = [], 0
        cur.append((e, w))
        cur_w += w + gap_x
    if cur:
        rows.append(cur)
    return rows


def draw_legend_band(img, rows, font, pad=14, row_h=30, s=18, gap_x=22, inner=8):
    """Legend strip at the bottom of the image."""
    from PIL import ImageDraw
    W, H = img.size
    draw = ImageDraw.Draw(img, 'RGBA')
    band_h = len(rows) * row_h + 2 * pad
    y0 = H - band_h - 10
    draw.rounded_rectangle([10, y0, W - 10, H - 10], radius=8, fill=(255, 255, 255, 235), outline=(170, 170, 175, 255))
    for ri, row in enumerate(rows):
        x = 10 + pad
        y = y0 + pad + ri * row_h
        for (kind, rgb, text), w in row:
            draw_swatch(draw, kind, rgb, x, y + (row_h - s) / 2 - 2, s)
            tx = x if kind == 'text' else x + s + inner
            draw.text((tx, y + (row_h - font.size) / 2 - 3), text, fill=DARK, font=font)
            x += w + gap_x


def title_box(draw, title, subtitle, font_bold, font_sub):
    w = max(draw.textlength(title, font=font_bold), draw.textlength(subtitle, font=font_sub))
    return (10, 10, 10 + w + 24, 10 + font_bold.size + font_sub.size + 24)


def draw_title(img, title, subtitle, font_bold, font_sub):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img, 'RGBA')
    box = title_box(draw, title, subtitle, font_bold, font_sub)
    draw.rounded_rectangle(box, radius=8, fill=(255, 255, 255, 235), outline=(170, 170, 175, 255))
    draw.text((22, 17), title, fill=DARK, font=font_bold)
    draw.text((22, 17 + font_bold.size + 6), subtitle, fill=(70, 70, 75), font=font_sub)


def draw_front(img, front, scale):
    """Small FRONT arrow just ahead of the front bumper, pointing along the car's +y as seen by this camera."""
    from PIL import ImageDraw
    W, H = img.size
    draw = ImageDraw.Draw(img, 'RGBA')
    (u0, v0), (u1, v1) = front
    x0, y0, x1, y1 = u0 * W, (1 - v0) * H, u1 * W, (1 - v1) * H
    d = math.hypot(x1 - x0, y1 - y0)
    dx, dy = (x1 - x0) / d, (y1 - y0) / d
    n = 58 * scale                                          # arrow length in px
    x0, y0 = min(max(x0, 12 + n), W - 12 - n), min(max(y0, 60 * scale), H - 60 * scale)   # keep inside the frame
    x1, y1 = x0 + dx * n, y0 + dy * n
    col, lw = (60, 60, 65), max(1, round(2 * scale))
    draw.line([(x0, y0), (x1, y1)], fill=col, width=lw)
    h = 7 * scale
    draw.polygon([(x1 + dx * h, y1 + dy * h), (x1 - dy * h * 0.6, y1 + dx * h * 0.6), (x1 + dy * h * 0.6, y1 - dx * h * 0.6)], fill=col)
    font = fonts(max(8, round(13 * scale)))[1]
    draw.text(((x0 + x1) / 2, (y0 + y1) / 2 - 12 * scale), 'FRONT', fill=col, font=font, anchor='mm')


def draw_table(draw, x, y, cols, rows, f, fb, badge_outlines=None, badge_font=None, row_h=32):
    """Table with header; cols = [(title, width)], rows = tuples of str. With badge_outlines the first column is drawn as a badge."""
    xs = [x]
    for _, w in cols:
        xs.append(xs[-1] + w)
    for (title, _), cx in zip(cols, xs):
        draw.text((cx + 8, y + 5), title, fill=DARK, font=fb)
    draw.line([x, y + row_h, xs[-1], y + row_h], fill=(120, 120, 125), width=2)
    for i, row in enumerate(rows):
        ry = y + row_h * (i + 1) + 2
        if i % 2:
            draw.rectangle([x, ry, xs[-1], ry + row_h], fill=(243, 243, 246))
        for j, (txt, cx) in enumerate(zip(row, xs)):
            if j == 0 and badge_outlines is not None:
                draw_badge(draw, cx + 8 + 15, ry + row_h / 2, txt, badge_outlines[i], 2, badge_font, badge_diameter(txt, 1.0))
            else:
                draw.text((cx + 8, ry + 5), txt, fill=DARK, font=f)
    return y + row_h * (len(rows) + 1) + 4


def write_legend(args, C, codes, present, outdir):
    """legend.png (colour key + tables) and legend.md with the same content."""
    from PIL import Image, ImageDraw
    sites, sensors, actuators = C['sites'], C['sensors'], C['actuators']
    key = marker_entries(codes, args.labels, len(sites), args.band, compact=False)
    key += [('line', channel_colour(k), CABLE_NAMES.get(k, f'cable: {k}')) for k in present]
    site_rows = [(str(i), s['id'], s['zone'], f"{s['thermal_class']} ({s.get('derating', 1.0):.2f})") for i, s in enumerate(sites, 1)]
    per_rows = [(codes[it['id']], it['id'], it['type']) for it in sensors + actuators]
    per_kinds = [it['type'] for it in sensors] + ['actuator'] * len(actuators)

    f, fb = fonts(20)
    f24b, f_badge = fonts(24)[1], fonts(BADGE_FONT)[0]
    W, row_h = 1800, 32
    n_key = (len(key) + 1) // 2
    y_tab = 70 + n_key * 32 + 40
    H = y_tab + 44 + (max(len(site_rows), len(per_rows)) + 1) * row_h + 40
    img = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 22), 'Legend - IONIQ 5 N E/E layout', fill=DARK, font=f24b)
    for i, (kind, rgb, text) in enumerate(key):
        x, y = 30 + (i % 2) * 880, 70 + (i // 2) * 32
        draw_swatch(draw, kind, rgb, x, y + 3, 22)
        draw.text((x + 34, y), text, fill=DARK, font=f)
    draw.text((30, y_tab), 'Candidate ECU sites', fill=DARK, font=f24b)
    draw.text((1130, y_tab), 'Sensors and actuators', fill=DARK, font=f24b)
    y = y_tab + 44
    draw_table(draw, 30, y, [('No.', 70), ('site id', 270), ('zone', 250), ('thermal class (derating)', 440)], site_rows, f, fb,
               [GREY_OUTLINE] * len(site_rows), f_badge, row_h)
    draw_table(draw, 1130, y, [('Code', 80), ('sensor / actuator id', 330), ('type', 230)], per_rows, f, fb,
               [badge_style(k, 1.0)[0] for k in per_kinds], f_badge, row_h)
    img.save(os.path.join(outdir, 'legend.png'))
    print('wrote', os.path.join(outdir, 'legend.png'))

    md = ['# Legend - IONIQ 5 N E/E layout', '', '## Colour key', '', '| Symbol | Colour | Meaning |', '|---|---|---|']
    shapes = {'circle': f'sphere r {SITE_R} m', 'small': f"sphere r {SENSOR_R['CAMERA']} m", 'tiny': f"sphere r {SENSOR_R['IMU']} m",
              'puck': f"puck r {SENSOR_R['LIDAR']} m, h {LIDAR_H} m", 'square': f'cube {2 * ACT_HALF:.2f} m, dark edges',
              'block': f'translucent box (alpha {BLOCK_ALPHA}), dark edges', 'line': 'tube (channel)', 'thin': 'thin line',
              'ghost': 'translucent shell'}
    md += [f"| {shapes[k]} | {hex_colour(rgb) if rgb else '-'} | {t} |" for k, rgb, t in key]
    md += ['', '## Candidate ECU sites', '', '| No. | site id | zone | thermal class (derating) |', '|---|---|---|---|']
    md += [f'| {r[0]} | {r[1]} | {r[2]} | {r[3]} |' for r in site_rows]
    md += ['', '## Sensors and actuators', '', '| Code | sensor/actuator id | type |', '|---|---|---|']
    md += [f'| {r[0]} | {r[1]} | {r[2]} |' for r in per_rows]
    with open(os.path.join(outdir, 'legend.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    print('wrote', os.path.join(outdir, 'legend.md'))


def compose(args):
    from PIL import Image, ImageDraw, ImageFilter
    outdir = args.outdir
    C = load_configs(args.config_dir, args.solution)
    codes = peripheral_codes(C['sensors'], C['actuators'])
    present = kinds_present(C['channels']) if C['cables'] is None else kinds_present({'edges': C['cables']['cables']})
    font, font_t, font_s = fonts(19)[0], fonts(16)[1], fonts(13)[0]
    ch = C['channels']
    if C['cables'] is not None:
        sol = C['solution']
        subtitle = (f"optimisation result: {len(C['cables']['used_sites'])} ECUs on {len(C['sites'])} candidate cells  |  "
                    f"{len(C['cables']['cables'])} cables, {C['cables']['total_routed_length_m']} m routed  |  "
                    f"total cost {sol.get('total_cost', 0):.0f}  |  {len(present)} cable types")
    else:
        subtitle = (f"{len(C['sites'])} candidate ECU sites  |  {len(C['sensors'])} sensors  |  {len(C['actuators'])} actuators  |  "
                    f"harness backbone: {len(ch['nodes'])} nodes, {len(ch['edges'])} edges, {len(present)} cable types")
    entries = marker_entries(codes, args.labels, len(C['sites']), args.band, compact=True) + [('text', None, 'cables:')]
    entries += [('line', channel_colour(k), CABLE_NAMES.get(k, k)) for k in present]
    for view in args.views:
        img = Image.open(os.path.join(outdir, f'{view}_raw.png')).convert('RGBA')
        W, H = img.size
        scale = W / 1800
        if args.labels == 'names':
            lab = Image.open(os.path.join(outdir, f'{view}_labels.png')).convert('RGBA')
            if lab.size != img.size:
                lab = lab.resize(img.size)
            halo = Image.new('RGBA', img.size, (255, 255, 255, 0))
            halo.putalpha(lab.getchannel('A').filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(1.0)))
            img = Image.alpha_composite(Image.alpha_composite(img, halo), lab)
        draw = ImageDraw.Draw(img, 'RGBA')
        rows = band_rows(draw, entries, font, W)
        band_h = len(rows) * 30 + 2 * 14
        title = TITLES.get(view, view)
        keep_out = [(10, H - band_h - 10, W - 10, H - 10), title_box(draw, title, subtitle, font_t, font_s)]
        proj = json.load(open(os.path.join(outdir, f'{view}_anchors.json')))
        draw_front(img, proj['front'], scale)
        if args.labels == 'numbers':
            draw_badges(img, proj['anchors'], scale, keep_out)
        draw_legend_band(img, rows, font)
        draw_title(img, title, subtitle, font_t, font_s)
        img.convert('RGB').save(os.path.join(outdir, f'{view}.png'))
        print('wrote', os.path.join(outdir, f'{view}.png'))
    write_legend(args, C, codes, present, outdir)


if __name__ == '__main__':
    a = parse_args()
    if a.compose:
        compose(a)
    elif bpy is None:
        raise SystemExit('run this script inside Blender (or with --compose under a Python that has PIL)')
    else:
        build(a)
