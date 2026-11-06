#!/usr/bin/env python3
"""
GPU (TOPS) demand for a schema-v3 source architecture (user 2026-11-05), driven by configs/gpu_demand.json:
  - every camera and every lidar of the car gets one detection network of per_sensor_tops raw TOPS (an Autoware
    reference design's TOPS per sensor, applied to the car's own sensors); a sensor that already has a measured network
    (camera_existing / lidar_existing) gets that network's TOPS set to per_sensor_tops, every other sensor gets a new
    component copied from the measured template of its kind (CPU, RAM, ROM, ASIL rule, fail-operational flag):
      camera  bound to its camera (bound_sensors, read by tools/arch_v3_prepare.py), detections to the tracker
      lidar   input from that lidar's preprocessing node, detections to the tracker
    the flows are copies of the template's flows (rate and size as measured or defaulted by the prepare step)
  - infotainment components get the TOPS of the given functions (equal split over the listed components) and the
    accelerator feature
  - chains unchanged (no new timing chains)
Writes configs/arch_v3/<out>/ (swc.json, flows.json, chains.json); then
  dse/bin/python tools/arch_v3_prepare.py --src configs/arch_v3/<out> --name <out>

  dse/bin/python tools/arch_v3_gpu_demand.py [configs/gpu_demand.json]
"""
import copy
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(p):
    with open(p) as f:
        return json.load(f)


def _dump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump(obj, f, indent=1)


def main(spec_path):
    spec = _load(spec_path)
    src = os.path.join(REPO, spec['source_dir'])
    out = os.path.join(REPO, 'configs', 'arch_v3', spec['out'])
    swc, flows, chains = (_load(os.path.join(src, f)) for f in ('swc.json', 'flows.json', 'chains.json'))
    comps = swc['components']
    by_id = {c['id']: c for c in comps}
    acc = spec.get('accelerator_feature', 'HW_ACC')
    tops = float(spec['per_sensor_tops'])
    ref = spec['reference']
    note = (f"gpu_demand (tools/arch_v3_gpu_demand.py, {os.path.relpath(spec_path, REPO)}): {ref['name']}, {ref['tops']:g} TOPS for "
            f"{ref['sensors']} sensors = {tops:g} TOPS per sensor, applied to this car's cameras and lidars; {ref['source']}")
    fl_of = {(f['src'], f['dst']): f for f in flows['flows']}

    def flow_copy(pair, src_id, dst_id):
        f = copy.deepcopy(fl_of[tuple(pair)])
        f['src'], f['dst'] = src_id, dst_id
        f['source'] = f"copy of the flow {pair[0]} -> {pair[1]} ({note})"
        return f

    def set_tops(c, value, why):
        c['tops_req'] = round(float(value), 6)
        c['tops_source'] = why
        if acc not in (c.get('hw_required') or []):
            c['hw_required'] = list(c.get('hw_required') or []) + [acc]

    added, changed, new_flows = [], [], []
    # cameras
    for cam in spec['cameras']:
        old = spec.get('camera_existing', {}).get(cam)
        if old:
            set_tops(by_id[old], tops, f"{note}; measured network of {cam}, its TOPS set to the per-sensor share (was {by_id[old].get('tops_req')})")
            changed.append(old)
            continue
        c = copy.deepcopy(by_id[spec['camera_template']])
        c['id'] = f"AW:/perception/object_recognition/detection/camera_dnn/{cam}"
        c['cpu_dmips'] = float(spec['camera_cpu_dmips'])
        c['cpu_dmips_source'] = f"copy of {spec['camera_template']}: {spec.get('camera_cpu_note', '')}"
        c['bound_sensors'] = [cam]
        c['source'] = f"added: camera detection network of {cam} ({note})"
        set_tops(c, tops, f"{note}; copy of {spec['camera_template']}")
        comps.append(c)
        added.append(c['id'])
        new_flows.append(flow_copy(spec['detections_flow_template'], c['id'], spec['detections_to']))
    # lidars
    for d, lidar in spec['lidars'].items():
        old = spec.get('lidar_existing', {}).get(d)
        if old:
            set_tops(by_id[old], tops, f"{note}; measured network (concatenated cloud), carries the share of {lidar}; was {by_id[old].get('tops_req')}")
            changed.append(old)
            continue
        c = copy.deepcopy(by_id[spec['lidar_template']])
        c['id'] = f"AW:/perception/object_recognition/detection/lidar_dnn/{lidar}"
        c['source'] = f"added: lidar detection network of {lidar} ({note})"
        set_tops(c, tops, f"{note}; copy of {spec['lidar_template']}")
        comps.append(c)
        added.append(c['id'])
        inp = spec['lidar_input'].format(dir=d)
        if inp not in by_id:
            sys.exit(f"[gpu_demand] ERROR: lidar input node {inp} not in {spec['source_dir']}")
        new_flows.append(flow_copy(spec['lidar_input_flow_template'], inp, c['id']))
        new_flows.append(flow_copy(spec['detections_flow_template'], c['id'], spec['detections_to']))
    # infotainment
    for item in spec.get('infotainment', []):
        ids = list(item.get('ids') or []) + sorted(c['id'] for c in comps if item.get('prefix') and c['id'].startswith(item['prefix']))
        if not ids:
            sys.exit(f"[gpu_demand] ERROR: no component for {item}")
        for cid in ids:
            set_tops(by_id[cid], float(item['tops_total']) / len(ids),
                     f"gpu_demand: {item['what']}, {item['tops_total']:g} TOPS over {len(ids)} component(s); {item['source']}")
            changed.append(cid)
    flows['flows'] += new_flows
    swc.setdefault('meta', {})['gpu_demand'] = {'spec': os.path.relpath(spec_path, REPO), 'reference': ref,
                                                'per_sensor_tops': tops, 'added': added, 'changed': changed}
    for name, obj in (('swc.json', swc), ('flows.json', flows), ('chains.json', chains)):
        _dump(obj, os.path.join(out, name))
    if spec.get('hw_catalogue'):   # the instance's hardware catalogue (read by tools/arch_v3_prepare.py from the source dir)
        _dump(_load(os.path.join(REPO, spec['hw_catalogue'])), os.path.join(out, 'hw_catalogue.json'))
    total = sum(float(c.get('tops_req') or 0) for c in comps)
    print(f"[gpu_demand] {len(added)} components added, {len(changed)} changed, {len(new_flows)} flows added; "
          f"total {total:.2f} TOPS raw over {sum(1 for c in comps if float(c.get('tops_req') or 0) > 0)} components; written to {out}")


if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'configs', 'gpu_demand.json')))
