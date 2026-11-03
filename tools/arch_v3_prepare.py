#!/usr/bin/env python3
"""
Prepare the schema-v3 architecture (configs/arch_v3: swc.json, flows.json, chains.json) for the optimiser and the
checker, which read the loader's fields (HANDOFF 16.26, 2026-10-28). Output per instance in configs/arch_v3/prepared/<name>/:
  arch.json   software_components with the loader's fields (cpu_req DMIPS, ram_req / rom_req kB, asil_req, ...),
              the devices of each driver (sensors / actuators lists) and the message flows per pair (flows), plus
              meta.assumptions = every value this script had to assume
  cfg/        the config files of the run: partitions.json with v3 capacities, sensors.json / actuators.json with the
              suite's devices plus the generated ones, chains.json with derived network budgets, the rest copied

  python tools/arch_v3_prepare.py                        -> prepared/full   (all components)
  python tools/arch_v3_prepare.py --subset 500 --seed 1  -> prepared/n500_s1: the whole Autoware part plus random
                                                            whole non-AD chains and their strongest partners
"""
import argparse
import json
import os
import random
import re
import shutil
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'configs', 'arch_v3')
CFG = os.path.join(REPO, 'configs')
OUT = os.path.join(SRC, 'prepared')
COPY_FILES = ('vehicle.json', 'buses.json', 'hardwares.json', 'uncertainty.json', 'sites.json', 'distances.json',
              'channels.json', 'solver.json', 'unit_classes.json', 'routing_space.npz')

# --- assumptions (every one is written to arch.json meta.assumptions) ---------------------------------------------
CORE_DMIPS = 12250          # one Cortex-A76-class core (configs/unit_classes.json cpu_currency)
PARTITION = {'cpu_cap': 49000, 'ram_cap': 16 * 1024 * 1024, 'rom_cap': 64 * 1024 * 1024}   # R-Car V4H class block
SITE_BLOCKS = 4             # a cell holds up to 4 partitions (partitions.json site.max_partitions)
BUS_LINES = {'CAN': 4, 'LIN': 6, 'FLEXRAY': 4}   # lines available per bus type (buses.json backbone.bus_lines; v2 car: 2 / 3 / 1)
AW_ASIL_DRIVING = ('/sensing/', '/localization/', '/perception/', '/planning/', '/control/', '/vehicle/', '/map/')
BUS_TO_IFACE = {'CAN': 'CAN', 'LIN': 'LIN', 'FLEXRAY': 'FLEXRAY', 'ETH': 'ETH', 'direct': 'UART'}
AUDIO_MBIT = {'amplifier': 1.4112, 'microphones': 0.256}   # CD audio; 16 kHz 16-bit mono
# endpoints of the non-AD functions (user 2026-10-29): the generated devices reduced to 16 common, physical sensors /
# actuators; duplicates of the suite merged into it, uncommon ones dropped. ENDPOINT_OF maps a generated device to
# its endpoint(s) (None = dropped; a list = one endpoint per side, driven by the same function); ENDPOINTS gives each new endpoint its position in the car frame (x lateral, + right;
# y longitudinal, + front; z up; m; left-hand drive) from the blocks of configs/vehicle.json, and its kind.
ENDPOINT_OF = {
    'gate_drivers': 'ACT_Motor_Front', 'brake_modulator': 'ACT_Brake_Front',
    'steering_torque_sensor': 'ACT_Steering', 'steering_angle_sensor': 'ACT_Steering', 'motor_position_sensor': 'ACT_Steering',
    'defrost_switch': 'ACT_HVAC_Cabin', 'temperature_sensor': 'ACT_HVAC_Cabin',
    'accelerator_pedal': 'accelerator_pedal', 'gear_selector': 'gear_selector', 'drive_mode_switch': 'gear_selector',
    'start_button': 'start_button', 'key_status': 'start_button', 'key_remote': 'keyless_receiver', 'rf_receiver': 'keyless_receiver',
    'door_lock_switches': ['door_lock_switches_left', 'door_lock_switches_right'],
    'window_switches': ['window_switches_left', 'window_switches_right'], 'pinch_sensors': ['window_switches_left', 'window_switches_right'],
    'seat_buttons': 'seat_controls', 'seat_position_sensors': 'seat_controls', 'tpms_receiver': 'tpms_receiver',
    'ultrasonic_sensors': 'parking_sensors', 'microphones': 'microphones', 'charge_port_sensors': 'charge_port',
    'pilot_signal': 'charge_port', 'horn': 'horn', 'mirror_motor_left': 'mirror_motor_left', 'mirror_motor_right': 'mirror_motor_right',
    'obc_control': 'onboard_charger', 'charger_contactors': 'onboard_charger', 'speakers': 'amplifier', 'amplifier': 'amplifier',
    'pedestrian_contact_sensors': None, 'hood_lifters': None, 'coupling_actuator': None, 'body_height_sensors': None,
    'accelerometers': None, 'chassis_sensor_pool': None, 'pdc_buzzer': None}
ENDPOINTS = {   # id: (kind, (x, y, z), placement)
    'accelerator_pedal': ('sensor', (-0.40, 0.90, 0.45), "driver footwell in front of the left seat, at the floor below the dashboard"),
    'gear_selector': ('sensor', (-0.30, 0.78, 0.88), "column-mounted drive selector right of the steering column (IONIQ 5)"),
    'start_button': ('sensor', (-0.22, 0.82, 0.92), "start / stop button on the dashboard right of the steering column"),
    'keyless_receiver': ('sensor', (0.00, -0.95, 1.50), "RF receiver in the rear headliner (common place of keyless / RF receivers)"),
    'door_lock_switches_left': ('sensor', (-0.85, 0.30, 0.88), "left doors: driver door inner panel"),
    'door_lock_switches_right': ('sensor', (0.85, 0.30, 0.88), "right doors: front passenger door inner panel"),
    'window_switches_left': ('sensor', (-0.82, 0.12, 0.76), "left doors: driver door armrest switch pack (master switches)"),
    'window_switches_right': ('sensor', (0.82, 0.12, 0.76), "right doors: front passenger door armrest switch"),
    'seat_controls': ('sensor', (-0.67, 0.20, 0.52), "outer side of the driver seat base"),
    'tpms_receiver': ('sensor', (0.00, -0.85, 0.42), "cabin centre under the rear seat cushion, between the wheels"),
    'parking_sensors': ('sensor', (0.35, -2.25, 0.50), "ultrasonic sensors in the rear bumper (one endpoint for the set)"),
    'microphones': ('sensor', (0.00, 0.30, 1.53), "overhead console behind the windscreen header"),
    'charge_port': ('sensor', (0.90, -1.75, 0.90), "charging inlet in the rear right quarter panel above the wheel arch (IONIQ 5)"),
    'horn': ('actuator', (-0.45, 2.12, 0.55), "behind the front bumper, left of the front lidar"),
    'mirror_motor_left': ('actuator', (-0.95, 0.88, 1.02), "left door mirror at the front upper corner of the door"),
    'mirror_motor_right': ('actuator', (0.95, 0.88, 1.02), "right door mirror at the front upper corner of the door"),
    'onboard_charger': ('actuator', (0.00, -1.925, 0.535), "integrated charging control unit / on-board charger (vehicle.json iccu_charger block)"),
    'amplifier': ('actuator', (0.35, -0.32, 0.40), "under the floor covering in front of the rear seat (IONIQ 5 N Bose amplifier, BRS installation manual)"),
}
# Autoware driver nodes of the sensor suite (tools/build_architecture.py replicates the kit's chains per device)
AW_LIDAR_DIRS = {'top': 'LIDAR_Front', 'left': 'LIDAR_Left_Pillar', 'right': 'LIDAR_Right_Pillar', 'rear': 'LIDAR_Rear',
                 'lidar_top': 'LIDAR_Top'}
AW_IMU, AW_GNSS = 'AW:/sensing/imu/imu_corrector', 'AW:/sensing/gnss/gnss_poser'
# the suite's actuators (configs/actuators.json) are driven by the last component of the matching generated function
FUNCTION_ACTUATORS = {'Chassis_SteeringSystemEps': ['ACT_Steering'], 'Chassis_BrakeByWireActuation': ['ACT_Brake_Front', 'ACT_Brake_Rear'],
                      'Powertrain_TractionInverterControl': ['ACT_Motor_Front'], 'BodyComfort_ClimateControlHVAC': ['ACT_HVAC_Cabin'],
                      'BodyComfort_ExteriorLight': ['ACT_Light_Front', 'ACT_Light_Rear']}


def suite_actuator_drivers(comps):
    """component id -> suite actuators it drives: the component with the highest index of each function"""
    last = {}
    for c in comps:
        m = re.match(r'^(.+)_(\d+)$', c['id'])
        if m and m.group(1) in FUNCTION_ACTUATORS and int(m.group(2)) >= last.get(m.group(1), (-1, None))[0]:
            last[m.group(1)] = (int(m.group(2)), c['id'])
    return {cid: FUNCTION_ACTUATORS[fn] for fn, (_i, cid) in last.items()}


def _load(name, base=SRC):
    with open(os.path.join(base, name)) as f:
        return json.load(f)


def _dump(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


def aw_asil(c):
    if c.get('fail_operational'):
        return 4
    return 2 if any(p in c['id'] for p in AW_ASIL_DRIVING) else 0


def processing_ms(c):
    p = c.get('processing_time_ms')
    if isinstance(p, dict):
        p = p.get('mean_ms')
    if p:
        return float(p)
    return sum(float(r['wcet_us']) for r in (c.get('runnables') or [])) / 1000.0


def prepare(name, subset=None, seed=1):
    comps = _load('swc.json')['components']
    flows = _load('flows.json')['flows']
    chains = _load('chains.json')['chains']
    suite_s = _load('sensors.json', CFG)
    suite_a = _load('actuators.json', CFG)
    cams = [s['id'] for s in suite_s['sensors'] if s['type'] == 'CAMERA']
    assumptions = []
    by_id = {c['id']: c for c in comps}
    act_drivers = suite_actuator_drivers(comps)
    assumptions.append("the suite's actuators are driven by the last component of the matching function: " +
                       ", ".join(f"{cid} -> {', '.join(a)}" for cid, a in sorted(act_drivers.items())))

    # --- 1. the subset: whole Autoware part + random whole non-AD chains + strongest partners -----------------------
    keep = set(by_id)
    if subset:
        rng = random.Random(seed)
        vol = defaultdict(float)
        for fl in flows:
            v = (fl['bytes'] or 737) * 8 * (fl['rate_hz'] or 40.0) / 1e6
            vol[fl['src'], fl['dst']] += v
            vol[fl['dst'], fl['src']] += v
        nbr = defaultdict(set)
        for a, b in vol:
            nbr[a].add(b)
        keep = {c['id'] for c in comps if c['id'].startswith('AW:')}
        # every endpoint of the car is in every subset (user 2026-10-29: the physical car is the same at every size):
        # the components driving a sensor / actuator endpoint (suite actuators and the 18 endpoints of the non-AD
        # functions) and the whole chains they belong to
        drivers = set(act_drivers)
        for c in comps:
            if any(ENDPOINT_OF.get(d, d) is not None for d in list(c.get('sensors') or []) + list(c.get('actuators') or [])):
                drivers.add(c['id'])
        for ch in chains:
            if any(s in drivers for s in ch['swcs']):
                keep.update(s for s in ch['swcs'] if s in by_id)
        keep.update(drivers)
        n_fixed = len(keep)
        non_ad = [ch for ch in chains if not ch['id'].startswith('AW')]
        rng.shuffle(non_ad)
        for ch in non_ad:
            if len(keep) >= subset * 0.8:
                break
            keep.update(s for s in ch['swcs'] if s in by_id)
        while len(keep) < subset:
            cand = defaultdict(float)
            for s in keep:
                for t in nbr[s]:
                    if t not in keep:
                        cand[t] += vol[s, t]
            if not cand:
                break
            keep.add(max(cand.items(), key=lambda kv: (kv[1], kv[0]))[0])
        assumptions.append(f"subset {name}: the whole Autoware part ({sum(1 for s in keep if s.startswith('AW:'))} nodes), every "
                           f"endpoint's driving components and their chains ({n_fixed} components fixed in every subset), "
                           f"random whole non-AD chains (seed {seed}) up to 80 % of {subset} components, then the components "
                           f"with the most traffic to the included ones until {subset}")
    comps = [c for c in comps if c['id'] in keep]

    # --- 2. flows: defaults for unmeasured topics, one entry per pair ------------------------------------------------
    meas_rate = statistics.median([f['rate_hz'] for f in flows if f['kind'] == 'ros2_topic' and f['rate_hz']])
    meas_bytes = statistics.median([f['bytes'] for f in flows if f['kind'] == 'ros2_topic' and f['bytes']])
    n_default = 0
    pair = {}
    for fl in flows:
        if fl['src'] not in keep or fl['dst'] not in keep or fl['src'] == fl['dst']:
            continue
        rate, size = fl['rate_hz'], fl['bytes']
        if rate is None or size is None:
            n_default += 1
            rate, size = rate or meas_rate, size or meas_bytes
        p = pair.setdefault((fl['src'], fl['dst']), {'src': fl['src'], 'dst': fl['dst'], 'volume': 0.0, 'kbit': 0.0, 'n': 0})
        p['volume'] += size * 8 * rate / 1e6
        p['kbit'] = max(p['kbit'], size * 8 / 1000.0)
        p['n'] += 1
    if n_default:
        assumptions.append(f"{n_default} Autoware topics without a measured rate or size: median of the measured topics "
                           f"({meas_rate:g} Hz, {meas_bytes:g} B)")
    assumptions.append("messages: several topics between the same pair are one flow with the summed volume and the largest message")
    receives = defaultdict(list)
    out_rate = defaultdict(float)
    for (s, d), p in pair.items():
        receives[d].append(s)
    for fl in flows:
        if fl['src'] in keep and fl['dst'] in keep:
            out_rate[fl['src']] = max(out_rate[fl['src']], fl['rate_hz'] or meas_rate)

    # --- 3. chains with derived network budgets, the tightest budget per component -----------------------------------
    pm = {c['id']: processing_ms(c) for c in comps}
    out_chains, tight_of = [], {}
    for ch in chains:
        members = [s for s in ch['swcs'] if s in keep]
        if len(members) < 2 or len(members) != len(ch['swcs']):
            continue
        nb = ch.get('net_budget_ms')
        if nb is None:
            nb = round(float(ch['e2e_bound_ms']) - sum(pm[m] for m in members), 3)
        entry = {'id': ch['id'], 'swcs': members, 'net_budget_ms': nb, 'e2e_bound_ms': ch.get('e2e_bound_ms'),
                 'source': ch.get('source')}
        if ch.get('ftti_ms'):
            entry['ftti_ms'] = float(ch['ftti_ms'])
        out_chains.append(entry)
        for m in members:
            tight_of[m] = min(tight_of.get(m, float('inf')), float(nb))
    assumptions.append("chain network budget = end-to-end bound minus the processing times of the members (measured mean "
                       "for Autoware nodes where known, sum of the runnables' WCET for generated components)")
    # a device on a bus costs its driver's chain one bus crossing (frame time + hop + robust platform term); a device
    # of a control loop whose budget cannot pay that crossing is wired directly to its unit (SPI, no bus crossing)
    cable = _load('buses.json', CFG)['cable_types']
    lat = _load('uncertainty.json', CFG)['robust_latency']
    def crossing_ms(iface, kbit):
        t = cable[iface]
        return kbit / float(t['capacity']) + float(t.get('hop_latency_ms', 0.0)) + float(lat['platform_p99_ms']) + float(lat.get('sync_ms', 0.0))
    direct = []

    # --- 4. components in the loader's fields -------------------------------------------------------------------------
    devices_s, devices_a = {}, {}
    n_capped, n_asil = 0, 0
    prepared = []
    for c in comps:
        cid = c['id']
        aw = cid.startswith('AW:')
        cpu = max(1, int(round(c['cpu_dmips'] or 0)))
        if 'traffic_light_node_container' in cid and cpu > CORE_DMIPS:
            cpu, n_capped = CORE_DMIPS, n_capped + 1
        asil = c.get('asil')
        if asil is None:
            asil, n_asil = aw_asil(c), n_asil + 1
        sensors, actuators = [], []
        if aw:
            m = re.search(r'/sensing/lidar/([^/]+)/velodyne_ros_wrapper_node$', cid)
            if m and m.group(1) in AW_LIDAR_DIRS:
                sensors = [AW_LIDAR_DIRS[m.group(1)]]
            m = re.search(r'/(camera(\d+))/detection/traffic_light_map_based_detector$', cid)
            if m:
                # traffic-light recognition runs on the front camera only (user 2026-11-03): camera6 = CAM_Front_Center
                sensors = ['CAM_Front_Center'] if 'CAM_Front_Center' in cams else cams[:1]
            if cid == AW_IMU:
                sensors = ['IMU_Top_Center']
            if cid == AW_GNSS:
                sensors = ['GPS_Center']
        else:
            sensors, actuators = [], list(act_drivers.get(cid, []))
            for d in list(c.get('sensors') or []) + list(c.get('actuators') or []):
                ends = ENDPOINT_OF.get(d, d)
                for e in ([] if ends is None else ends if isinstance(ends, list) else [ends]):
                    kind = 'actuator' if (e.startswith('ACT_') or ENDPOINTS.get(e, ('sensor',))[0] == 'actuator') else 'sensor'
                    lst = actuators if kind == 'actuator' else sensors
                    if e not in lst:
                        lst.append(e)
            period = float(c.get('period_ms') or 100.0)
            for kind, names, store in (('sensor', sensors, devices_s), ('actuator', actuators, devices_a)):
                for d in names:
                    if d.startswith('ACT_'):
                        continue
                    iface = BUS_TO_IFACE.get(c.get('bus'), 'UART')
                    rate = 1000.0 / period
                    volume = AUDIO_MBIT.get(d, 64 * rate / 1e6)   # one 8-byte frame per period, audio streams as sourced
                    if d in store:   # a second driver of a merged endpoint: its frames add to the endpoint's stream
                        store[d]['volume'] = round(store[d]['volume'] + volume, 6)
                        store[d]['rate_hz'] = max(store[d]['rate_hz'], rate)
                        continue
                    if iface in ('CAN', 'LIN', 'FLEXRAY') and tight_of.get(cid, float('inf')) < crossing_ms(iface, 0.064):
                        direct.append(f"{d} ({iface} -> SPI, chain budget {tight_of[cid]:g} ms)")
                        iface = 'SPI'
                    _kind, pos, where = ENDPOINTS[d]
                    store[d] = {'type': d.upper(), 'id': d, 'interface': iface, 'volume': round(volume, 6),
                                'position': {'x': pos[0], 'y': pos[1], 'z': pos[2]}, 'rate_hz': rate,
                                'source': f"assumption: endpoint of the non-AD functions ({c.get('bus')} bus of the function; "
                                          f"placed at: {where}; one 8-byte frame per {period:g} ms period per driver"
                                          + ('; audio stream as sourced' if d in AUDIO_MBIT else '') + ')'}
        rate_hz = 1000.0 / float(c['period_ms']) if c.get('period_ms') else (out_rate.get(cid) or 10.0)
        prepared.append({
            'id': cid, 'domain': c['domain'], 'function': c.get('function'),
            'type': 'sensor' if sensors else ('actuator' if actuators else c['role']),
            'cpu_req': cpu, 'ram_req': max(1, int(round(c['ram_kb'] or 0))), 'rom_req': max(1, int(round(c['rom_kb'] or 0))),
            'asil_req': int(asil), 'hw_required': list(c.get('hw_required') or []), 'tops_req': float(c.get('tops_req') or 0.0),
            'fail_operational': bool(c.get('fail_operational')), 'rate_hz': round(rate_hz, 3), 'redundant_with': None,
            'platform': c.get('platform') or 'posix', 'sensors': sensors, 'actuators': actuators,
            'receives_from': sorted(receives.get(cid, [])), 'source': c.get('source'),
            'asil_source': 'data' if c.get('asil') is not None else 'rule: fail-operational D, driving-chain node B, else QM',
        })
    if n_capped:
        assumptions.append(f"{n_capped} traffic_light_node_container nodes: the measured host CPU share (about 83,000 DMIPS, the "
                           f"CPU side of GPU inference on the A100 host) capped at one target core ({CORE_DMIPS} DMIPS); the "
                           f"inference load is their HW_ACC / TOPS demand")
    if n_asil:
        assumptions.append(f"{n_asil} Autoware nodes without ASIL: fail-operational -> ASIL D, nodes of the driving chain "
                           f"(sensing, localization, perception, planning, control, vehicle, map) -> ASIL B, else QM")
    assumptions.append("units: cpu_req = DMIPS (configs/unit_classes.json cpu_currency), ram_req / rom_req = kB; "
                       "a partition = one R-Car V4H-class block (49,000 DMIPS, 16 GB RAM, 64 GB flash), a cell up to 4 of them")
    assumptions.append("generated devices reduced to 18 endpoints (user 2026-10-29; window and lock switches one per side): duplicates of the suite merged into it "
                       "(inverter gate drivers -> ACT_Motor_Front, ESC brake modulator -> ACT_Brake_Front, EPS sensors -> "
                       "ACT_Steering, defrost / temperature -> ACT_HVAC_Cabin), parts of one unit merged (e.g. seat buttons + "
                       "position = seat controls), uncommon ones dropped (pedestrian sensors, active hood, AWD coupling, ride "
                       "height, accelerometers, chassis sensor pool, parking buzzer); positions from the blocks of vehicle.json")
    assumptions.append("generated devices with a 'direct' wiring are modelled as UART point-to-point cables")
    assumptions.append("Autoware sensor drivers: velodyne driver per lidar (kit top/left/right/rear = suite front/left/right/rear, "
                       "clone lidar_top = LIDAR_Top), traffic-light map-based detector per camera (camera6/7 = the first two "
                       "cameras of the suite, camera8-13 the others), imu_corrector = IMU, gnss_poser = GNSS")

    # --- 5. write ----------------------------------------------------------------------------------------------------
    out = os.path.join(OUT, name)
    cfg = os.path.join(out, 'cfg')
    os.makedirs(cfg, exist_ok=True)
    for f in COPY_FILES:
        if os.path.exists(os.path.join(CFG, f)):
            shutil.copy(os.path.join(CFG, f), os.path.join(cfg, f))
    part = _load('partitions.json', CFG)
    part['partition'].update(PARTITION)
    part['site'].update({k: v * SITE_BLOCKS for k, v in PARTITION.items()})
    part['_v3'] = ("capacities of the schema-v3 instances (tools/arch_v3_prepare.py): partition = one R-Car V4H-class block "
                   "(configs/unit_classes.json central_ad_soc: 49,000 DMIPS, 16 GB RAM, 64 GB flash assumed), site = "
                   f"{SITE_BLOCKS} blocks; costs unchanged")
    _dump(part, os.path.join(cfg, 'partitions.json'))
    used_s = {d for c in prepared for d in c['sensors']}
    used_a = {d for c in prepared for d in c['actuators']}
    driver_of = {d: c['id'] for c in prepared for d in c['sensors'] + c['actuators']}
    suite_acts = []
    for a in suite_a['actuators']:
        if a['id'] not in used_a:
            continue
        a = dict(a)
        if a['interface'] in ('CAN', 'LIN', 'FLEXRAY') and tight_of.get(driver_of[a['id']], float('inf')) < crossing_ms(a['interface'], 0.135):
            direct.append(f"{a['id']} ({a['interface']} -> SPI, chain budget {tight_of[driver_of[a['id']]]:g} ms)")
            a['source'] = (a.get('source') or '') + f"; direct wiring (SPI) assumed: driven by a control loop with a {tight_of[driver_of[a['id']]]:g} ms network budget"
            a['interface'] = 'SPI'
        suite_acts.append(a)
    if direct:
        assumptions.append("devices of control loops whose network budget cannot pay one bus crossing (frame time + hop + robust "
                           "platform term) are wired directly to their unit (SPI, no bus crossing), as in the engine / chassis "
                           "ECUs they come from: " + "; ".join(direct))
    suite_s['sensors'] = [s for s in suite_s['sensors'] if s['id'] in used_s] + [devices_s[d] for d in sorted(devices_s) if d in used_s]
    suite_a['actuators'] = suite_acts + [devices_a[d] for d in sorted(devices_a) if d in used_a]
    # consistency: no chain may need more for its end devices than its budget
    dev_if = {d['id']: (d['interface'], d['volume'] * 1000.0 / d['rate_hz']) for d in suite_s['sensors'] + suite_a['actuators']}
    comp_devs = {c['id']: c['sensors'] + c['actuators'] for c in prepared}
    for ch in out_chains:
        need = 0.0
        for end in (ch['swcs'][0], ch['swcs'][-1]):
            need += max((crossing_ms(dev_if[d][0], dev_if[d][1]) for d in comp_devs.get(end, []) if dev_if[d][0] in ('CAN', 'LIN', 'FLEXRAY')), default=0.0)
        if need > ch['net_budget_ms']:
            print(f"[prepare] WARNING: chain {ch['id']} needs {need:.2f} ms for its end devices, budget {ch['net_budget_ms']} ms")
    _dump(suite_s, os.path.join(cfg, 'sensors.json'))
    _dump(suite_a, os.path.join(cfg, 'actuators.json'))
    # bus lines: a line carries one ASIL group (drivers of ASIL >= 3 = high, else low); a device of critical
    # (fail-operational or ASIL >= 3) software sits on two lines of its type (two lanes), so a type needs at least
    # two high lines where a critical device exists plus one low line where a low-ASIL device exists
    crit_comp = {c['id'] for c in prepared if c['fail_operational']}          # critical = standby (the model's rule)
    high_comp = {c['id'] for c in prepared if c['asil_req'] >= 3}
    buses = _load('buses.json', CFG)
    lines = {}
    for t in ('CAN', 'LIN', 'FLEXRAY'):
        devs = [d for d in suite_s['sensors'] + suite_a['actuators'] if d['interface'] == t]
        need = 0
        for group in (True, False):   # high-ASIL lines and low-ASIL lines are separate
            members = [d for d in devs if (driver_of.get(d['id']) in high_comp) == group]
            need += 2 if any(driver_of.get(d['id']) in crit_comp for d in members) else (1 if members else 0)
        lines[t] = max(BUS_LINES[t], need)
    buses['backbone']['bus_lines'] = lines
    buses['backbone']['_bus_lines_v3'] = ("tools/arch_v3_prepare.py: lines available per type for the schema-v3 instances (assumption: "
                                          f"{BUS_LINES}; per ASIL group at least two lines where a critical device needs two lanes, else one "
                                          "where a device exists); the model decides how many are used")
    _dump(buses, os.path.join(cfg, 'buses.json'))
    assumptions.append(f"bus lines available: {lines} (v2 car: CAN 2, LIN 3, FlexRay 1): a line carries one ASIL group and a device of "
                       "critical software sits on two lines, so per ASIL group two lines where a critical device exists, else one")
    _dump({'_generated_by': 'tools/arch_v3_prepare.py', 'architecture': f'arch_v3/prepared/{name}', 'chains': out_chains},
          os.path.join(cfg, 'chains.json'))
    arch = {'schema': 'v3-prepared', 'meta': {'generated_by': 'tools/arch_v3_prepare.py', 'name': name, 'subset': subset,
                                              'seed': seed, 'assumptions': assumptions},
            'software_components': prepared, 'flows': [{k: v for k, v in p.items() if k != 'n'} for p in pair.values()]}
    _dump(arch, os.path.join(out, 'arch.json'))
    # attach every endpoint to the cable routing space: sites.json, channels.json, distances.json of the instance
    # (tools/vehicle_layout/generate_layout.py; the cells and their distances stay those of configs/)
    import subprocess
    subprocess.run([sys.executable, os.path.join(REPO, 'tools', 'vehicle_layout', 'generate_layout.py'), '--config_dir', cfg,
                    '--quiet'], check=True)
    dist_ids = set(_load('distances.json', cfg)['ids'])
    missing = [d['id'] for d in suite_s['sensors'] + suite_a['actuators'] if d['id'] not in dist_ids]
    if missing:
        sys.exit(f"[prepare] ERROR: endpoints without a cable distance: {missing}")

    cpu, ram, rom, tops = (sum(c[k] for c in prepared) for k in ('cpu_req', 'ram_req', 'rom_req', 'tops_req'))
    print(f"[prepare] {name}: {len(prepared)} components ({sum(1 for c in prepared if c['id'].startswith('AW:'))} Autoware), "
          f"{len(pair)} flows, {len(out_chains)} chains, {len(suite_s['sensors'])} sensors, {len(suite_a['actuators'])} actuators")
    print(f"[prepare] demand: {cpu:,} DMIPS (>= {cpu / PARTITION['cpu_cap']:.1f} partitions), RAM {ram / 1024 / 1024:.1f} GB "
          f"(>= {ram / PARTITION['ram_cap']:.1f}), ROM {rom / 1024 / 1024:.1f} GB (>= {rom / PARTITION['rom_cap']:.1f}), "
          f"{tops:.1f} TOPS; largest component {max(c['cpu_req'] for c in prepared):,} DMIPS")
    for a in assumptions:
        print(f"[prepare] assumption: {a}")
    print(f"[prepare] written to {out}")
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subset', type=int, default=None, help='number of components (default: all)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--name', default=None, help='output name (default: full or n<subset>_s<seed>)')
    a = ap.parse_args()
    prepare(a.name or ('full' if not a.subset else f'n{a.subset}_s{a.seed}'), a.subset, a.seed)
