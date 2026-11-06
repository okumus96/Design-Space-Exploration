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
SITE_BLOCKS = 4             # a cell holds up to 4 partitions (partitions.json site.max_partitions)
# partition types (building blocks, user 2026-11-04): each partition is one chip of a class of configs/unit_classes.json
# with its datasheet CPU / RAM / ROM / TOPS and its chip price (external memory not priced); any mix of up to
# SITE_BLOCKS blocks per cell; every block supports every ASIL; no housing cost and no per-cell resource caps
BLOCK_CLASSES = (('S32K344', 'ecu_small'), ('TC397', 'zone_mcu_classic'), ('S32G3', 'zone_gateway'),
                 ('TDA4VH', 'central_ad_soc_ti'), ('ORIN', 'central_ad_gpu'))
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
# extended endpoint set (user 2026-11-04, 'option A'): with the device bug of the generator fixed every catalogue function
# carries its devices; the 88 devices the old catalogues had lost get their real place. Merged only what sits in one
# physical unit or one module (sensors inside the inverter / battery / brake unit, the items of one door, the tailgate,
# the seat, a lamp cluster, the steering column module); the endpoint's kind wins where sensors and actuators share a
# unit (one cable either way). 45 new endpoints; prepare --endpoints core keeps the 33-endpoint set of 2026-10-29.
_DOORS = ['door_lock_switches_left', 'door_lock_switches_right', 'door_rear_left', 'door_rear_right']
_WINDOWS = ['window_switches_left', 'window_switches_right', 'door_rear_left', 'door_rear_right']
_SEATS = ['seat_controls', 'seat_controls_right']
_LAMPS = ['ACT_Light_Front', 'headlamp_left', 'ACT_Light_Rear', 'rear_lamp_left']
ENDPOINT_OF_EXT = {
    # body and comfort
    'rain_sensor': 'rain_light_sensor', 'light_sensor': 'rain_light_sensor', 'ambient_light_sensor': 'rain_light_sensor',
    'humidity_sensor': 'rain_light_sensor', 'wiper_stalk': 'column_switches', 'light_switch': 'column_switches',
    'steering_angle': 'column_switches', 'horn_button': 'steering_wheel_switches', 'wiper_motor_front': 'wiper_motor_front',
    'washer_pump': 'washer_pump', 'wiper_motor_rear': 'tailgate_module', 'rear_window_heater': 'tailgate_module',
    'trunk_lock': 'tailgate_module', 'mirror_switch': 'window_switches_left', 'mirror_heater': ['mirror_motor_left', 'mirror_motor_right'],
    'door_contacts': _DOORS, 'door_lock_actuators': _DOORS, 'door_handle_sensors': _DOORS, 'window_motors': _WINDOWS,
    'window_switches': _WINDOWS, 'pinch_sensors': _WINDOWS, 'interior_lamps': ['interior_lamp_front', 'interior_lamp_rear'],
    'intrusion_sensor': 'interior_lamp_front', 'seat_axis_motors': _SEATS, 'seat_temperature_sensors': _SEATS, 'seat_heaters': _SEATS,
    'seat_fans': _SEATS, 'headlamps': ['ACT_Light_Front', 'headlamp_left'], 'rear_lamps': ['ACT_Light_Rear', 'rear_lamp_left'],
    'indicators': _LAMPS, 'brake_lamps': ['ACT_Light_Rear', 'rear_lamp_left'], 'hazard_lamps': _LAMPS, 'siren': 'siren',
    'tilt_sensor': 'siren', 'clamp_relays': 'power_distribution_box', 'key_transponder': 'start_button', 'lf_antennas': 'keyless_receiver',
    'cabin_temperature_sensors': 'climate_sensors', 'sun_sensor': 'climate_sensors', 'hvac_blower': 'ACT_HVAC_Cabin',
    'flap_actuators': 'ACT_HVAC_Cabin', 'compressor': 'e_compressor', 'body_sensor_pool': None, 'keypad': None,
    # occupant safety
    'crash_sensors_front': ['crash_sensor_front_left', 'crash_sensor_front_right'],
    'crash_sensors_side': ['crash_sensor_side_left', 'crash_sensor_side_right'],
    'pressure_sensors': ['crash_sensor_side_left', 'crash_sensor_side_right'],
    'airbag_squibs': ['airbag_driver', 'airbag_passenger', 'airbag_side_left', 'airbag_side_right'],
    'belt_pretensioners': ['airbag_side_left', 'airbag_side_right'], 'seat_occupancy_sensors': 'seat_controls_right',
    'belt_buckle_switches': _SEATS, 'warning_lamp': 'cluster_display', 'chime': 'cluster_display',
    # chassis
    'wheel_speed_sensors': ['wheel_speed_FL', 'wheel_speed_FR', 'wheel_speed_RL', 'wheel_speed_RR'],
    'yaw_rate_sensor': 'chassis_inertial_sensor', 'lateral_acceleration': 'chassis_inertial_sensor', 'epb_switch': 'epb_switch',
    'parking_brake_actuators': 'ACT_Brake_Rear', 'eps_motor': 'ACT_Steering',
    'damper_valves': ['damper_FL', 'damper_FR', 'damper_RL', 'damper_RR'], 'brake_pedal_sensor': 'brake_pedal',
    'brake_pressure_sensors': None, 'brake_actuators_front': 'ACT_Brake_Front', 'brake_actuators_rear': 'ACT_Brake_Rear',
    # powertrain (sensors inside the inverter / the DC-DC unit are read by the unit itself: not separate endpoints)
    'phase_current_sensors': None, 'dc_link_voltage': None, 'resolver': None, 'inverter_temperature': None,
    'cell_voltage_monitors': 'battery_pack', 'pack_current_sensor': 'battery_pack', 'pack_temperature_sensors': 'battery_pack',
    'main_contactors': 'battery_pack', 'pre_charge_relay': 'battery_pack', 'coolant_temperature_sensors': 'coolant_module',
    'coolant_pumps': 'coolant_module', 'valves': 'coolant_module', 'radiator_fan': 'radiator_fan', 'lv_voltage_sensor': None,
    'dcdc_control': 'onboard_charger', 'park_lock_actuator': 'rear_drive_unit',
    # infotainment and connectivity
    'cluster_display': 'cluster_display', 'touch_panel': 'center_display', 'buttons': 'center_display',
    'center_display': 'center_display', 'hud_projector': 'hud', 'driver_camera': 'driver_camera',
    'cellular_modem': 'roof_antenna', 'gnss_receiver': 'roof_antenna', 'v2x_radio': 'roof_antenna',
}
ENDPOINTS_EXT = {   # id: (kind, (x, y, z), placement); positions from the blocks of configs/vehicle.json
    'rain_light_sensor': ('sensor', (0.00, 0.42, 1.43), "windscreen top behind the interior mirror: rain / light / humidity sensor"),
    'column_switches': ('sensor', (-0.37, 0.78, 0.84), "steering column module: wiper and light stalks, steering angle sensor"),
    'steering_wheel_switches': ('sensor', (-0.37, 0.64, 0.92), "steering wheel hub: horn pad and wheel buttons (clock spring)"),
    'wiper_motor_front': ('actuator', (-0.25, 1.30, 0.98), "cowl below the windscreen base: front wiper motor"),
    'washer_pump': ('actuator', (-0.70, 1.95, 0.45), "washer reservoir, front left behind the bumper"),
    'tailgate_module': ('actuator', (0.00, -2.15, 1.05), "tailgate below the rear glass: rear wiper, rear window heater, latch"),
    'interior_lamp_front': ('actuator', (0.00, 0.22, 1.54), "overhead console: front interior lamps, intrusion sensor"),
    'interior_lamp_rear': ('actuator', (0.00, -0.75, 1.55), "headliner above the rear seats: rear interior lamp"),
    'seat_controls_right': ('sensor', (0.67, 0.20, 0.52), "outer side of the passenger seat base: seat motors, heaters, occupancy, buckle"),
    'door_rear_left': ('actuator', (-0.85, -0.65, 0.80), "rear left door: window motor and switch, lock, contact, handle sensor"),
    'door_rear_right': ('actuator', (0.85, -0.65, 0.80), "rear right door: window motor and switch, lock, contact, handle sensor"),
    'headlamp_left': ('actuator', (-0.60, 1.98, 0.72), "front left lamp cluster (mirror of ACT_Light_Front)"),
    'rear_lamp_left': ('actuator', (-0.58, -2.00, 0.92), "rear left lamp cluster (mirror of ACT_Light_Rear)"),
    'siren': ('actuator', (0.55, 2.10, 0.55), "alarm siren with tilt sensor, front right behind the bumper"),
    'power_distribution_box': ('actuator', (-0.48, 1.45, 0.80), "front junction box next to the 12 V battery: terminal clamp relays"),
    'climate_sensors': ('sensor', (0.00, 1.05, 1.02), "dashboard top centre: sun sensor, in-car temperature sensor"),
    'e_compressor': ('actuator', (0.50, 1.95, 0.45), "electric A/C compressor, front bay right"),
    'crash_sensor_front_left': ('sensor', (-0.55, 2.25, 0.55), "front bumper beam, left: upfront crash sensor"),
    'crash_sensor_front_right': ('sensor', (0.55, 2.25, 0.55), "front bumper beam, right: upfront crash sensor"),
    'crash_sensor_side_left': ('sensor', (-0.86, -0.20, 0.45), "B-pillar base, left: side crash sensor (door pressure sensor)"),
    'crash_sensor_side_right': ('sensor', (0.86, -0.20, 0.45), "B-pillar base, right: side crash sensor (door pressure sensor)"),
    'airbag_driver': ('actuator', (-0.37, 0.60, 0.94), "steering wheel driver airbag (clock spring)"),
    'airbag_passenger': ('actuator', (0.40, 0.80, 0.95), "passenger airbag in the dashboard"),
    'airbag_side_left': ('actuator', (-0.70, -0.05, 0.75), "left: side and curtain airbags, belt pretensioner"),
    'airbag_side_right': ('actuator', (0.70, -0.05, 0.75), "right: side and curtain airbags, belt pretensioner"),
    'wheel_speed_FL': ('sensor', (-0.78, 1.44, 0.35), "front left wheel hub: wheel speed sensor"),
    'wheel_speed_FR': ('sensor', (0.78, 1.44, 0.35), "front right wheel hub: wheel speed sensor"),
    'wheel_speed_RL': ('sensor', (-0.78, -1.57, 0.35), "rear left wheel hub: wheel speed sensor"),
    'wheel_speed_RR': ('sensor', (0.78, -1.57, 0.35), "rear right wheel hub: wheel speed sensor"),
    'chassis_inertial_sensor': ('sensor', (0.00, -0.10, 0.38), "centre tunnel: yaw rate and acceleration sensor of the stability control"),
    'epb_switch': ('sensor', (0.05, 0.75, 0.80), "parking brake switch, front of the centre console"),
    'damper_FL': ('actuator', (-0.62, 1.44, 0.75), "front left strut top: adaptive damper valve"),
    'damper_FR': ('actuator', (0.62, 1.44, 0.75), "front right strut top: adaptive damper valve"),
    'damper_RL': ('actuator', (-0.62, -1.57, 0.70), "rear left damper top: adaptive damper valve"),
    'damper_RR': ('actuator', (0.62, -1.57, 0.70), "rear right damper top: adaptive damper valve"),
    'brake_pedal': ('sensor', (-0.30, 0.92, 0.45), "driver footwell: brake pedal travel sensor (brake-by-wire)"),
    'battery_pack': ('actuator', (0.00, 0.00, 0.25), "traction battery under the floor: management unit, contactors, pre-charge"),
    'coolant_module': ('actuator', (0.00, 1.95, 0.40), "front bay below the drive unit: coolant pumps, valves, temperature sensors"),
    'radiator_fan': ('actuator', (0.00, 2.05, 0.50), "radiator fan behind the front grille"),
    'rear_drive_unit': ('actuator', (0.20, -1.50, 0.50), "rear drive unit: park lock actuator"),
    'cluster_display': ('actuator', (-0.37, 0.85, 1.00), "instrument cluster behind the steering wheel: telltales, chime"),
    'center_display': ('actuator', (0.05, 0.80, 1.00), "centre display with touch panel and buttons"),
    'hud': ('actuator', (-0.37, 1.00, 1.03), "head-up display projector in the dashboard"),
    'driver_camera': ('sensor', (-0.30, 0.72, 0.98), "driver monitoring camera on the steering column cover"),
    'roof_antenna': ('sensor', (0.00, -1.40, 1.62), "roof shark fin: cellular, GNSS and V2X antennas"),
}
# interface of an endpoint where it is not the bus of the function that drives it: wheel-speed and pedal sensors are
# wired point to point to their controller (two-wire sensor lines, the UART class with its 5 m reach), the chassis
# inertial sensor is a CAN node
ENDPOINT_IFACE = {**{f'wheel_speed_{w}': 'UART' for w in ('FL', 'FR', 'RL', 'RR')}, 'brake_pedal': 'UART',
                  'chassis_inertial_sensor': 'CAN'}
# Autoware driver nodes of the sensor suite (tools/build_architecture.py replicates the kit's chains per device)
AW_LIDAR_DIRS = {'top': 'LIDAR_Front', 'left': 'LIDAR_Left_Pillar', 'right': 'LIDAR_Right_Pillar', 'rear': 'LIDAR_Rear',
                 'lidar_top': 'LIDAR_Top'}
AW_IMU, AW_GNSS = 'AW:/sensing/imu/imu_corrector', 'AW:/sensing/gnss/gnss_poser'
# the suite's actuators (configs/actuators.json) are driven by the last component of the matching generated function
FUNCTION_ACTUATORS = {'Chassis_SteeringSystemEps': ['ACT_Steering'], 'Chassis_BrakeByWireActuation': ['ACT_Brake_Front', 'ACT_Brake_Rear'],
                      'Powertrain_TractionInverterControl': ['ACT_Motor_Front'], 'BodyComfort_ClimateControlHVAC': ['ACT_HVAC_Cabin'],
                      'BodyComfort_ExteriorLight': ['ACT_Light_Front', 'ACT_Light_Rear']}


def suite_actuator_drivers(comps):
    """component id -> suite actuators it drives: the actuator driver of each function (role 'actuator' = the end of its
    safety path, 2026-11-04), else the component with the highest index (catalogues generated before the path rule)"""
    last = {}
    for c in comps:
        m = re.match(r'^(.+)_(\d+)$', c['id'])
        if not (m and m.group(1) in FUNCTION_ACTUATORS):
            continue
        key = (1 if c.get('role') == 'actuator' else 0, int(m.group(2)))
        if key >= last.get(m.group(1), ((-1, -1), None))[0]:
            last[m.group(1)] = (key, c['id'])
    return {cid: FUNCTION_ACTUATORS[fn] for fn, (_k, cid) in last.items()}


def _load(name, base=None):
    with open(os.path.join(base or SRC, name)) as f:
        return json.load(f)


def block_types():
    """the partition types (configs/unit_classes.json classes of BLOCK_CLASSES) in the units of the v3 demands:
    cpu DMIPS, ram / rom kB, tops rated INT8 TOPS, cost USD (chip price)"""
    uc = _load('unit_classes.json', CFG)
    out = []
    for tid, key in BLOCK_CLASSES:
        c = uc[key]
        out.append({'id': tid, 'class': key, 'chip': c['example'], 'cpu_cap': float(c['cpu_dmips']),
                    'ram_cap': round(float(c['ram_mb']) * 1024, 1), 'rom_cap': round(float(c['flash_mb']) * 1024, 1),
                    'tops_cap': float(c.get('tops') or 0), 'cost': float(c['cost_usd']), 'cost_source': c.get('cost_source')})
    return out


def _dump(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


def _generate_layout(cfg):
    """sites.json, channels.json, distances.json (and routing_space.npz) of the instance: every endpoint of cfg/sensors.json
    and cfg/actuators.json attached to the cable routing space (tools/vehicle_layout/generate_layout.py; the cells and
    their distances stay those of configs/)"""
    import subprocess
    subprocess.run([sys.executable, os.path.join(REPO, 'tools', 'vehicle_layout', 'generate_layout.py'), '--config_dir', cfg,
                    '--quiet'], check=True)


def _apply_reach(cfg, suite_s, suite_a, prepared, tight_of, cable, crossing_ms, orig_if):
    """Direct wiring within reach (see the call in prepare): for every driver, its direct devices (SPI, UART) are kept in
    catalogue order while one cell lies within reach of all of them (routed lengths of cfg/distances.json); a device
    without such a cell takes a bus: its function's bus if that crossing fits the driver's chain budget, else the cheapest
    bus whose crossing fits; none fits = error. Rewrites cfg/sensors.json and cfg/actuators.json when a device moved.
    Returns one line per moved device."""
    dist = _load('distances.json', cfg)
    idx = {k: i for i, k in enumerate(dist['ids'])}
    matrix = dist['matrix']
    cells = [s['id'] for s in _load('sites.json', cfg)['sites']]
    reach = {n: float(t.get('max_length_m') or float('inf')) for n, t in cable.items() if isinstance(t, dict)}
    bus_types = ('ETH', 'CAN', 'FLEXRAY', 'LIN')
    devs = {d['id']: d for d in suite_s['sensors'] + suite_a['actuators']}

    def length(d, cell):
        return float(matrix[idx[d['id']]][idx[cell]])

    def within(d, cell):
        return length(d, cell) <= reach.get(d['interface'], float('inf')) + 1e-9

    moved = []
    for c in prepared:
        kept = []
        for d in [devs[x] for x in c['sensors'] + c['actuators'] if x in devs and devs[x]['interface'] not in bus_types]:
            if any(all(within(k, cell) for k in kept + [d]) for cell in cells):
                kept.append(d)
                continue
            budget = tight_of.get(c['id'], float('inf'))
            kbit = d['volume'] * 1000.0 / d['rate_hz']
            fits = {i: crossing_ms(i, kbit) for i in bus_types if crossing_ms(i, kbit) <= budget}
            if not fits:
                sys.exit(f"[prepare] ERROR: {d['id']} ({d['interface']}, reach {reach.get(d['interface'])} m) of {c['id']} has no cell "
                         f"within reach shared with {[k['id'] for k in kept]}, and no bus crossing fits the chain budget {budget:g} ms")
            new = orig_if.get(d['id']) if orig_if.get(d['id']) in fits else min(fits, key=lambda i: (float(cable[i]['cost_per_meter']), fits[i]))
            nearest = min((length(d, cell), cell) for cell in cells)
            note = (f"no cell within {reach.get(d['interface']):g} m ({d['interface']}) of {d['id']} and of {', '.join(k['id'] for k in kept) or 'nothing else'}"
                    f" (nearest cell {nearest[1]} at {nearest[0]:.2f} m) -> {new}, crossing {fits[new]:.2f} ms within the {budget:g} ms budget"
                    + (f" (its bus {orig_if[d['id']]}: {crossing_ms(orig_if[d['id']], kbit):.2f} ms)" if orig_if.get(d['id']) and orig_if[d['id']] != new else ''))
            moved.append(note)
            d['source'] = (d.get('source') or '') + '; reach: ' + note
            d['interface'] = new
    if moved:
        _dump(suite_s, os.path.join(cfg, 'sensors.json'))
        _dump(suite_a, os.path.join(cfg, 'actuators.json'))
    return moved


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


def prepare(name, subset=None, seed=1, endpoints='full'):
    # endpoint set (2026-11-04): 'full' = the extended set (ENDPOINT_OF_EXT / ENDPOINTS_EXT), 'core' = the 33 endpoints of
    # 2026-10-29 (every device without an endpoint there is not modelled)
    if endpoints == 'full':
        ENDPOINT_OF.update(ENDPOINT_OF_EXT)
        ENDPOINTS.update(ENDPOINTS_EXT)
        FUNCTION_ACTUATORS['BodyComfort_ExteriorLight'] = _LAMPS   # all four lamp clusters are driven by the lighting function
    else:
        for d in ENDPOINT_OF_EXT:
            ENDPOINT_OF.setdefault(d, None)
    comps = _load('swc.json')['components']
    flows = _load('flows.json')['flows']
    chains = _load('chains.json')['chains']
    suite_s = _load('sensors.json', CFG)
    suite_a = _load('actuators.json', CFG)
    cams = [s['id'] for s in suite_s['sensors'] if s['type'] == 'CAMERA']
    assumptions = []
    by_id = {c['id']: c for c in comps}
    act_drivers = suite_actuator_drivers(comps)
    assumptions.append("the suite's actuators are driven by the actuator driver (end of the safety path) of the matching function: " +
                       ", ".join(f"{cid} -> {', '.join(a)}" for cid, a in sorted(act_drivers.items())))
    # one driver per suite actuator (2026-11-04, user: no fatal errors): a generated device merged into a suite actuator
    # that has its own driver function (FUNCTION_ACTUATORS; e.g. the ESC brake modulator = the front brake of brake-by-wire)
    # is not wired a second time; the other function sends its command to that driver (one 8-byte frame per period), as
    # in a brake system where ESC and brake-by-wire command one modulator through one control unit
    designated = {e: cid for cid, es in act_drivers.items() for e in es}
    redirected = []
    for c in comps:
        if c['id'].startswith('AW:'):
            continue
        kept_a = []
        for d in c.get('actuators') or []:
            ends = ENDPOINT_OF.get(d, d)
            ends = [] if ends is None else ends if isinstance(ends, list) else [ends]
            tgt = sorted({designated[e] for e in ends if e in designated and designated[e] != c['id']})
            if tgt and all(e in designated and designated[e] != c['id'] for e in ends):
                for t in tgt:
                    flows.append({'src': c['id'], 'dst': t, 'bytes': 8, 'rate_hz': round(1000.0 / float(c.get('period_ms') or 10.0), 3),
                                  'kind': 'command', 'source': f"actuator command {d} -> the driver of {', '.join(ends)} (one driver per actuator)"})
                redirected.append(f"{c['id']} ({d} -> command to {', '.join(tgt)})")
            else:
                kept_a.append(d)
        c['actuators'] = kept_a
    if redirected:
        assumptions.append("one driver per actuator: a generated device merged into a suite actuator with its own driver function is "
                           "commanded through that driver: " + "; ".join(redirected))

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
    direct, orig_if = [], {}   # orig_if: the bus a device had before the direct-wiring rule (the reach rule falls back to it)

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
            if c.get('bound_sensors'):   # added networks (tools/arch_v3_gpu_demand.py, 2026-11-05): their own sensor
                sensors = [s for s in c['bound_sensors'] if s in {d['id'] for d in suite_s['sensors']}]
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
                    iface = ENDPOINT_IFACE.get(d) or BUS_TO_IFACE.get(c.get('bus'), 'UART')
                    rate = 1000.0 / period
                    volume = AUDIO_MBIT.get(d, 64 * rate / 1e6)   # one 8-byte frame per period, audio streams as sourced
                    if d in store:   # a second driver of a merged endpoint: its frames add to the endpoint's stream
                        store[d]['volume'] = round(store[d]['volume'] + volume, 6)
                        store[d]['rate_hz'] = max(store[d]['rate_hz'], rate)
                        continue
                    if iface in ('CAN', 'LIN', 'FLEXRAY') and tight_of.get(cid, float('inf')) < crossing_ms(iface, 0.064):
                        direct.append(f"{d} ({iface} -> SPI, chain budget {tight_of[cid]:g} ms)")
                        orig_if[d] = iface
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
    part['partition'] = {
        'types': block_types(),
        'tops_feature': 'HW_ACC',
        'max_per_asil_per_site': None,
        'description': ("partition types (building blocks, user 2026-11-04): every partition is one block (chip) of a type "
                        "with its own CPU (DMIPS), RAM and ROM (kB, the units of the v3 demands), TOPS and price; the solver "
                        "chooses the type of every partition, any mix of up to site.max_partitions blocks per cell, one ASIL "
                        "per partition, every block supports every ASIL. TOPS is a block resource: the accelerator "
                        "requirement tops_feature (HW_ACC) is served by the TOPS of the component's block (no add-on "
                        "accelerator). Capacities and prices from configs/unit_classes.json (class), chip price only, external "
                        "memory not priced. max_per_asil_per_site null = no limit of its own (user 2026-11-04)."),
    }
    part['site'].update({'max_partitions': SITE_BLOCKS, 'cpu_cap': None, 'ram_cap': None, 'rom_cap': None,
                         'fixed_cost': 0, 'switch_node_cost': 0,
                         'cost_source': "no housing cost per cell (user 2026-11-04: 'there should be no such cost')"})
    part['_v3'] = ("partition types of the schema-v3 instances (tools/arch_v3_prepare.py BLOCK_CLASSES): " +
                   ", ".join(f"{t['id']} ({t['class']})" for t in part['partition']['types']) +
                   f"; a cell = up to {SITE_BLOCKS} blocks, no per-cell resource caps, no housing cost")
    hw = _load('hardwares.json', cfg)
    catalogue_path = os.path.join(SRC, 'hw_catalogue.json')
    if os.path.exists(catalogue_path):
        # hardware catalogue of the source (2026-11-05, configs/hw_catalogue.json): partitions = CPU / RAM / ROM only
        # (prices with their external memory), the GPU = per-cell modules of several types (HW_ACC with types)
        catalogue = _load('hw_catalogue.json')
        part['partition']['types'] = [{k: v for k, v in t.items()} | {'tops_cap': 0.0} for t in catalogue['partition_types']]
        part['partition'].pop('tops_feature', None)
        part['partition']['description'] = ("partition types (user 2026-11-05): CPU (DMIPS), RAM and ROM (kB) per partition, one ASIL "
                                            "per partition, up to site.max_partitions per cell, any mix; prices include the "
                                            "external memory; no TOPS (the GPU is per-cell hardware, hardwares.json HW_ACC types)")
        part['site']['max_partitions'] = int(catalogue.get('max_partitions_per_site', SITE_BLOCKS))
        gpu = catalogue['gpu']
        hw.setdefault('hardware_costs', {})[gpu['feature']] = {
            'description': "GPU modules per cell, shared by all partitions of the cell (user 2026-11-05); cost and TOPS per type",
            'cost': 0, 'types': gpu['types'], 'max_units_per_site': int(gpu['max_units_per_site']),
            'source': os.path.relpath(catalogue_path, REPO)}
        hw.pop('_partition_types', None)
        assumptions.append(f"hardware catalogue {os.path.relpath(catalogue_path, REPO)}: partition types " +
                           ", ".join(f"{t['id']} ({t['cost']:g} USD)" for t in catalogue['partition_types']) + "; GPU modules per cell " +
                           ", ".join(f"{t['id']} {t['tops']:g} TOPS {t['cost']:g} USD" for t in gpu['types']) +
                           f", at most {gpu['max_units_per_site']} per cell")
    elif hw.get('hardware_costs', {}).pop('HW_ACC', None) is not None:   # TOPS comes with the SoC blocks (user 2026-11-04)
        hw['_partition_types'] = "HW_ACC (Hailo-8 add-on accelerator) removed: TOPS is a resource of the TDA4VH / ORIN blocks (partitions.json)"
    _dump(part, os.path.join(cfg, 'partitions.json'))
    _dump(hw, os.path.join(cfg, 'hardwares.json'))
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
            orig_if[a['id']] = a['interface']
            a['interface'] = 'SPI'
        suite_acts.append(a)
    if direct:
        assumptions.append("devices of control loops whose network budget cannot pay one bus crossing (frame time + hop + robust "
                           "platform term) are wired directly to their unit (SPI, no bus crossing), as in the engine / chassis "
                           "ECUs they come from: " + "; ".join(direct))
    suite_s['sensors'] = [s for s in suite_s['sensors'] if s['id'] in used_s] + [devices_s[d] for d in sorted(devices_s) if d in used_s]
    suite_a['actuators'] = suite_acts + [devices_a[d] for d in sorted(devices_a) if d in used_a]
    # shared endpoints (user 2026-11-04): a generated endpoint used by several functions takes the fastest bus among its
    # users' buses (bit rate from buses.json), so every user's data fits; before, the first function processed set it
    # (the instrument cluster ended up on LIN, the steering column module on LIN although stability control reads it).
    # Interfaces fixed in ENDPOINT_IFACE and the suite's own devices (configs/sensors.json, actuators.json) are kept.
    users_of = defaultdict(list)
    for c in prepared:
        for d in c['sensors'] + c['actuators']:
            users_of[d].append(c['id'])
    rate_of = {n: float(t.get('capacity') or 0.0) for n, t in cable.items() if isinstance(t, dict)}
    generated = {d['id'] for d in list(devices_s.values()) + list(devices_a.values())}
    upgraded = []
    for d in suite_s['sensors'] + suite_a['actuators']:
        if d['id'] not in generated or d['id'] in ENDPOINT_IFACE:
            continue
        buses = sorted({BUS_TO_IFACE.get(by_id[u].get('bus'), 'UART') for u in users_of.get(d['id'], []) if u in by_id})
        if len(buses) < 2:
            continue
        fastest = max(buses, key=lambda b: rate_of.get(b, 0.0))
        current = orig_if.get(d['id'], d['interface'])   # the bus before a direct-wiring decision
        if fastest != current:
            upgraded.append(f"{d['id']} ({current} -> {fastest}; users' buses {', '.join(buses)})")
            d['interface'] = fastest
            orig_if.pop(d['id'], None)   # the direct-wiring decision is taken again below, over all users
            d['source'] = (d.get('source') or '') + f"; shared endpoint: the fastest bus among its users ({', '.join(buses)})"
    if upgraded:
        assumptions.append("shared endpoints take the fastest bus among their users' buses: " + "; ".join(upgraded))
    # bus capacity (2026-11-05, smoke IIS: window_switches_right fits on no LIN line): a bus device whose data rate exceeds
    # the usable capacity of one line of its bus (capacity x backbone.bus_max_utilization) cannot sit on that bus; it
    # takes the next faster bus type whose line can carry it (merged endpoints sum the frames of their users)
    umax = float(_load('buses.json', CFG)['backbone'].get('bus_max_utilization', 1.0) or 1.0)
    faster = sorted((n for n in ('LIN', 'CAN', 'FLEXRAY', 'ETH') if n in rate_of), key=lambda n: rate_of[n])
    overfull = []
    for d in suite_s['sensors'] + suite_a['actuators']:
        t = d['interface']
        if t not in ('LIN', 'CAN', 'FLEXRAY') or float(d['volume']) <= rate_of[t] * umax:
            continue
        new_t = next((n for n in faster if rate_of[n] > rate_of[t] and float(d['volume']) <= rate_of[n] * umax), None)
        if new_t is None:
            continue
        overfull.append(f"{d['id']} ({t} -> {new_t}: {float(d['volume']) * 1000:.1f} kbit/s > {rate_of[t] * umax * 1000:.1f} usable on one {t} line)")
        d['interface'] = new_t
        orig_if.pop(d['id'], None)
        d['source'] = (d.get('source') or '') + f"; bus capacity: more data than one {t} line carries, on {new_t}"
    if overfull:
        assumptions.append("bus capacity: a device with more data than one line of its bus carries takes the next faster bus: " + "; ".join(overfull))
    # tight loops over all users (2026-11-04): a bus device used by several components (e.g. ACT_Steering, read by the
    # sensor end and driven by the actuator end of the steering chain) is wired directly when ANY user's chain budget
    # cannot pay its bus crossing; the rules above saw only one user per device (the first creator / the last driver)
    for d in suite_s['sensors'] + suite_a['actuators']:
        # UART too (user 2026-11-04: split the direct devices between UART and SPI by the same latency rule): a point-to-
        # point line whose crossing a user's loop cannot pay becomes SPI, the fast board-level line (1 m reach)
        if d['interface'] not in ('CAN', 'LIN', 'FLEXRAY', 'UART') or not users_of.get(d['id']):
            continue
        tight = min(tight_of.get(u, float('inf')) for u in users_of[d['id']])
        kbit = d['volume'] * 1000.0 / d['rate_hz'] if d.get('rate_hz') else 0.135
        if tight < crossing_ms(d['interface'], kbit):
            direct.append(f"{d['id']} ({d['interface']} -> SPI, tightest user chain budget {tight:g} ms)")
            orig_if.setdefault(d['id'], d['interface'])
            d['source'] = (d.get('source') or '') + f"; direct wiring (SPI) assumed: a user's control loop has a {tight:g} ms network budget"
            d['interface'] = 'SPI'
            assumptions.append(f"direct wiring over all users: {d['id']} ({orig_if[d['id']]} -> SPI): the tightest chain budget of its users, "
                               f"{tight:g} ms, cannot pay one {orig_if[d['id']]} crossing")
    # reach (2026-11-03): a directly wired device (SPI, UART) must lie within its cable's reach (configs/buses.json
    # cable_types.*.max_length_m, routed length of the instance's channel graph) of the unit of its driver, one unit for
    # all the driver's direct devices. The routing space is generated here (positions are final, interfaces may still
    # change): where no cell is within reach of all the driver's direct devices, the devices in catalogue order after the
    # last one with a common cell cannot be wired directly; each takes its function's bus if that crossing fits the
    # driver's chain budget, else the cheapest bus whose crossing fits (the rear brake: Ethernet 3.5 ms against the 5.4 ms
    # budget, CAN 5.8, FlexRay 8.5), else the instance cannot be built (error).
    _dump(suite_s, os.path.join(cfg, 'sensors.json'))
    _dump(suite_a, os.path.join(cfg, 'actuators.json'))
    _generate_layout(cfg)
    moved = _apply_reach(cfg, suite_s, suite_a, prepared, tight_of, cable, crossing_ms, orig_if)
    if moved:
        assumptions.append("reach: a directly wired device lies within its cable's reach (buses.json max_length_m, routed length) of "
                           "the unit of its driver, one unit for all the driver's direct devices; where no cell is within reach of all "
                           "of them, the devices beyond the common cell of the first ones take the cheapest bus whose crossing fits "
                           "the chain budget: " + "; ".join(moved))
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
    lines, alone = {}, []
    for t in ('CAN', 'LIN', 'FLEXRAY'):
        devs = [d for d in suite_s['sensors'] + suite_a['actuators'] if d['interface'] == t]
        # lines of their own (2026-11-05, car_s4 build stopped at step 10): a device whose tightest user chain cannot
        # absorb one foreign frame on its line (budget - own crossing < one frame time; with chain priorities a frame of
        # the tightest chain waits at most one frame) shares no line, so its lanes are its own; e.g. ACT_Brake_Front:
        # 5.791 ms budget, 5.771 ms own CAN crossing, one CAN frame 0.27 ms
        frame_ms = float(cable[t].get('frame_kbit', 0.0) or 0.0) / float(cable[t]['capacity'])
        own = []
        for d in devs:
            tight = min((tight_of.get(u, float('inf')) for u in users_of.get(d['id'], [])), default=float('inf'))
            kbit = d['volume'] * 1000.0 / d['rate_hz'] if d.get('rate_hz') else 0.135
            if tight - crossing_ms(t, kbit) < frame_ms:
                own.append(d)
                alone.append(f"{d['id']} ({t}: budget {tight:g} ms, own crossing {crossing_ms(t, kbit):.3f} ms, frame {frame_ms:.3f} ms)")
        need = sum(2 if any(u in crit_comp for u in users_of.get(d['id'], [])) else 1 for d in own)
        for group in (True, False):   # high-ASIL lines and low-ASIL lines are separate
            members = [d for d in devs if d not in own and (driver_of.get(d['id']) in high_comp) == group]
            need += 2 if any(driver_of.get(d['id']) in crit_comp for d in members) else (1 if members else 0)
        lines[t] = max(BUS_LINES[t], need)
    buses['backbone']['bus_lines'] = lines
    # upfront bus-length bounds (2026-11-05): 'full' enumerates device groups up to 4 with every cell and every pair of
    # cells per line, about 24,000 groups x 466 at 28 CAN devices (car_s4) = 11 million Steiner trees per line, the
    # build never finishes (smoke r5_smoke: 2 h in tree_bound); 'pairs' keeps the pair and tour bounds, and the exact
    # Steiner cuts in the callback keep every line length exact
    n_bus_devs = max(sum(1 for d in suite_s['sensors'] + suite_a['actuators'] if d['interface'] == t) for t in ('CAN', 'LIN', 'FLEXRAY'))
    buses['backbone']['bus_tree_bounds'] = 'pairs'
    assumptions.append(f"bus-length bounds: 'pairs' (up to {n_bus_devs} devices on one bus type; 'full' enumerates groups of up to 4 "
                       "devices with every cell and pair of cells, millions of Steiner trees per line); the exact lazy cuts stay")
    buses['backbone']['_bus_lines_v3'] = ("tools/arch_v3_prepare.py: lines available per type for the schema-v3 instances (assumption: "
                                          f"{BUS_LINES}; per ASIL group at least two lines where a critical device needs two lanes, else one "
                                          "where a device exists; plus its own lines (two lanes if critical) for a device whose tightest chain "
                                          "cannot absorb one foreign frame); the model decides how many are used")
    _dump(buses, os.path.join(cfg, 'buses.json'))
    assumptions.append(f"bus lines available: {lines} (v2 car: CAN 2, LIN 3, FlexRay 1): a line carries one ASIL group and a device of "
                       "critical software sits on two lines, so per ASIL group two lines where a critical device exists, else one; "
                       "a device whose chain cannot absorb one foreign frame has its lines alone" + (": " + "; ".join(alone) if alone else ""))
    _dump({'_generated_by': 'tools/arch_v3_prepare.py', 'architecture': f'arch_v3/prepared/{name}', 'chains': out_chains},
          os.path.join(cfg, 'chains.json'))
    arch = {'schema': 'v3-prepared', 'meta': {'generated_by': 'tools/arch_v3_prepare.py', 'name': name, 'subset': subset,
                                              'seed': seed, 'assumptions': assumptions},
            'software_components': prepared, 'flows': [{k: v for k, v in p.items() if k != 'n'} for p in pair.values()]}
    _dump(arch, os.path.join(out, 'arch.json'))
    # the cable routing space was generated before the reach rule (positions unchanged since): every endpoint attached
    dist_ids = set(_load('distances.json', cfg)['ids'])
    missing = [d['id'] for d in suite_s['sensors'] + suite_a['actuators'] if d['id'] not in dist_ids]
    if missing:
        sys.exit(f"[prepare] ERROR: endpoints without a cable distance: {missing}")

    cpu, ram, rom, tops = (sum(c[k] for c in prepared) for k in ('cpu_req', 'ram_req', 'rom_req', 'tops_req'))
    print(f"[prepare] {name}: {len(prepared)} components ({sum(1 for c in prepared if c['id'].startswith('AW:'))} Autoware), "
          f"{len(pair)} flows, {len(out_chains)} chains, {len(suite_s['sensors'])} sensors, {len(suite_a['actuators'])} actuators")
    big = max(block_types(), key=lambda t: t['cpu_cap'])
    print(f"[prepare] demand: {cpu:,} DMIPS (>= {cpu / big['cpu_cap']:.1f} {big['id']} blocks), RAM {ram / 1024 / 1024:.1f} GB "
          f"(>= {ram / big['ram_cap']:.1f}), ROM {rom / 1024 / 1024:.1f} GB (>= {rom / big['rom_cap']:.1f}), "
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
    ap.add_argument('--src', default=None, help='directory of swc.json / flows.json / chains.json (default: configs/arch_v3)')
    ap.add_argument('--endpoints', choices=('full', 'core'), default='full', help='full: the extended endpoint set (2026-11-04); core: the 33 endpoints of 2026-10-29')
    a = ap.parse_args()
    if a.src:
        SRC = os.path.abspath(a.src)
    prepare(a.name or ('full' if not a.subset else f'n{a.subset}_s{a.seed}'), a.subset, a.seed, a.endpoints)
