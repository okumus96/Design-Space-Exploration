"""Build the schema-v3 software architecture (configs/arch_v3/: swc.json, flows.json, chains.json) from measured and
sourced inputs.

  1. Autoware, node by node (~/autoware_bench): the topic graph (publisher / subscriber nodes per topic, from
     `ros2 topic info -v`), the recorded per-node processing times, the process samples (CPU, RSS) and the topic
     rates / message sizes. Every ROS 2 node is one component; its links are its subscriptions.
  2. The sensor / actuator suite (configs/sensors.json, configs/actuators.json): Autoware's driver chains are
     replicated per device (one velodyne driver chain per lidar, one camera chain per camera, IMU, GNSS); actuators
     are driven by the vehicle interface components of the powertrain / chassis domains.
  3. The non-AD domains: the function catalogue tools/arch_v3_functions.json (AUTOSAR Application Interfaces
     clusters, each with a source) expanded into components, runnables, links and chains with the CARIAD
     distributions (Denzinger, Becker, Ulbrich, arXiv 2605.04837 / ECRTS 2026, Tables 1-4).

Every record carries `source`: 'measured' | 'datasheet' | 'standard' | 'literature' | 'generated:<rule>'. The generated
part is representative (its structure follows the standards, its numbers follow published distributions), not a real
vehicle's software. Deterministic for a given seed. `--scale` multiplies the catalogue's component counts (1 = the
documents' decompositions, about 300 non-AD components; 20 = about 6,000).
"""
import argparse
import json
import math
import os
import random
import re
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUR = json.load(open(os.path.join(REPO, 'configs', 'partitions.json')))['cpu_currency']   # one DMIPS currency (sources there)


# ----------------------------------------------------------------------------------------------- CARIAD distributions
CARIAD = 'Denzinger, Becker, Ulbrich, arXiv 2605.04837 (ECRTS 2026), one motion-and-drive ECU on AURIX TC39x'
ASIL_LEVELS = [0, 1, 2, 3, 4]
ASIL_SHARE = [0.41, 0.07, 0.14, 0.10, 0.28]                                      # Table 1 (QM, A, B, C, D)
ROM_KB = {0: (6, 255), 1: (37, 140), 2: (15, 228), 3: (18, 168), 4: (21, 252)}  # Table 1
RAM_KB = {0: (0.5, 29), 1: (1, 22), 2: (0.5, 17), 3: (1, 26), 4: (1, 23)}       # Table 1
PERIODS = [1, 5, 10, 20, 50, 100, 200, 500, 1000]                               # Table 2
PERIOD_SHARE = [0.05, 0.31, 0.25, 0.12, 0.08, 0.10, 0.02, 0.04, 0.03]
WCET_US = {1: (15, 290), 5: (10, 725), 10: (22, 1218), 20: (56, 1733), 50: (125, 1902),
           100: (228, 2447), 200: (191, 3988), 500: (403, 6176), 1000: (1209, 9200)}
CHAIN_LEN = [2, 3, 4, 5, 6]                                                      # Table 3 (components per chain)
CHAIN_LEN_SHARE = [0.40, 0.25, 0.20, 0.10, 0.05]
DATA_AGE_FACTOR = (1.8, 4.9)       # Table 3 text: data-age bound = chain hyperperiod x 1.8 ... 4.9

# Fault tolerant time interval (ISO 26262-1:2018, 3.61: minimum time-span from the occurrence of a fault in an item to
# a possible occurrence of a hazardous event if the safety mechanisms are not activated; FHTI = FDTI + FRTI <= FTTI).
# Published values are rare; per function: source where published, else derived with the HARA kinematic rule
# (time until the vehicle leaves a corridor: t = sqrt(2 d / a) for a lateral offset d at lateral acceleration a;
# SAE 2021-01-0066 ties the FTTI to an acceptable travel distance) and marked 'derived'.
FTTI_MS = {
    'ExteriorLight': (500.0, 'AUTOSAR EXP_SafetyUseCase (CP R19-11): total loss of low beam, FTT 500 ms'),
    'TractionInverterControl': (60.0, 'Infineon AN1101 (inverter, unintended torque, FTTI <= 60 ms; training material, not verified)'),
    'PowertrainCoordinator': (60.0, 'as traction inverter (unintended torque)'),
    'SteeringSystemEps': (240.0, 'Kaihara, Kimura, Ishihara, Nakamura, JTEKT Engineering Journal 1017E (2020), Safety Design of Steering Systems for Autonomous Driving: steering failure in automated driving, time from the failure to lane departure (FTTI, ISO 26262 Part 3) Tc = 240 ms for the tightest road condition (metropolitan expressway, curve radius 88 m, 100 km/h; expressway cases longer), confirmed on a simulator; main->sub switchover <= 10 ms'),
    'SteeringVehicleStabilizing': (240.0, 'as EPS (JTEKT 2020, lane-departure FTTI 240 ms)'),
    'BrakeByWireActuation': (100.0, 'derived: loss of braking at 1 s headway margin; trade figure 100 ms (not citable)'),
    'Esc': (100.0, 'as brake-by-wire'),
    'VehicleCrashDetection': (10.0, 'derived: airbag decision window ~10-30 ms after impact (airbag inflation ~30 ms, SAE 901136)'),
    'OccupantRestraintActivation': (10.0, 'as crash detection'),
    'BatteryManagement': (1000.0, 'derived: thermal runaway / contactor opening within seconds; conservative 1 s'),
}
FTTI_FAIL_OPERATIONAL_SWITCHOVER_MS = 50.0   # Becker, Schaetz, Buckl, Armbruster 2014: switchover <= 50 ms (design aim)
CAN_SIGNAL_BYTES = 8               # one classic CAN frame (ISO 11898); LIN frame 8 B too

# ----------------------------------------------------------------------------------------- Bosch WATERS 2015 benchmark
# Kramer, Ziegenbein, Hamann, "Real World Automotive Benchmarks For Free", WATERS 2015 (engine management; the full
# model in Hamann et al., ECRTS 2017: 1,250 runnables in 21 tasks / ISRs, 10,000 labels, 4 cores at 200 MHz).
# Used as the second, independent reference next to CARIAD: period mix (Table III), average execution times
# (Table IV), BCET / WCET factors (Table V), label sizes (Table I) and chain structure (Tables VI / VII; the
# end-to-end constraint of a chain is the sum of the periods of its activation patterns).
WATERS = 'Kramer, Ziegenbein, Hamann, WATERS 2015 (Bosch engine management; Hamann et al., ECRTS 2017)'
WATERS_PERIODS = [1, 2, 5, 10, 20, 50, 100, 200, 1000]
WATERS_PERIOD_SHARE = [0.03, 0.02, 0.02, 0.25, 0.25, 0.03, 0.20, 0.01, 0.04]        # + 15 % angle-synchronous (engine only)
WATERS_ACET_US = {1: (0.34, 5.00, 30.11), 2: (0.32, 4.20, 40.69), 5: (0.36, 11.04, 83.38), 10: (0.21, 10.09, 309.87),
                  20: (0.25, 8.74, 291.42), 50: (0.29, 17.56, 92.98), 100: (0.21, 10.53, 420.43), 200: (0.22, 2.56, 21.95),
                  1000: (0.37, 0.43, 0.46)}                                         # min / avg / max
WATERS_WCET_FACTOR = {1: (1.30, 29.11), 2: (1.54, 19.04), 5: (1.13, 18.44), 10: (1.06, 30.03), 20: (1.06, 15.61),
                      50: (1.13, 7.76), 100: (1.02, 8.88), 200: (1.03, 4.90), 1000: (1.84, 4.75)}   # WCET = ACET x f
WATERS_LABEL_BYTES = [1, 2, 4, 8, 16, 32, 64, 128]
WATERS_LABEL_SHARE = [0.35, 0.49, 0.13, 0.008, 0.013, 0.005, 0.002, 0.002]
WATERS_CHAIN_LEN = [2, 3, 4, 5]
WATERS_CHAIN_LEN_SHARE = [0.30, 0.40, 0.20, 0.10]
# ACET as a Weibull distribution per period class (DTU benchmark fingerprint 'WATERS-Bosch' fitted to Table IV:
# Craciunas, Becker, Pop, "Automotive Cause-Effect Chains in the Wild", RTSOPS 2026, github dtu-ese; shape ~1 = near
# exponential); 1000 ms class uniform in its range. Chains: 4 per 100 runnables (same fingerprint).
WATERS_ACET_WEIBULL = {1: (1.044, 4.8226), 2: (1.0607, 4.0265), 5: (1.0082, 10.7603), 10: (1.0098, 9.9291),
                       20: (1.0131, 8.5489), 50: (1.0032, 17.7928), 100: (1.0090, 10.3669), 200: (1.1571, 2.5481)}
CHAINS_PER_100_RUNNABLES = 4
# shared vehicle signals: sources of state that many functions read (speed, wheel speeds, steering angle, pedals,
# gear, SoC, door / lock / light states, outside temperature, ...). Real vehicles exchange thousands of such
# signals between ECUs (a luxury car in 2008: ~2,500 inter-ECU signals; Volvo: >7,000 external signals, IEEE Spectrum
# 'How software is eating the car'); the generator targets about 5,000 cross-function links.
SHARED_SIGNAL_SOURCES = {
    'Chassis_ChassisSensor': ['wheel_speeds', 'yaw_rate', 'lateral_acceleration', 'longitudinal_acceleration'],
    'Chassis_SteeringSystemEps': ['steering_angle', 'steering_torque'],
    'Chassis_Esc': ['vehicle_speed', 'brake_pressure', 'esc_active'],
    'Powertrain_DriverRequest': ['accelerator_pedal', 'drive_mode'],
    'Powertrain_PowertrainCoordinator': ['torque_actual', 'torque_request'],
    'Powertrain_GearParkLock': ['gear_position'],
    'Powertrain_BatteryManagement': ['soc', 'pack_voltage', 'pack_current', 'battery_temperature'],
    'Powertrain_ThermalManagement': ['coolant_temperature'],
    'BodyComfort_TerminalClampControl': ['ignition_state', 'clamp_15', 'clamp_30'],
    'BodyComfort_CentralLocking': ['lock_state', 'door_states'],
    'BodyComfort_ExteriorLight': ['light_state', 'indicator_state'],
    'BodyComfort_BodySensors': ['outside_temperature', 'ambient_light', 'rain_intensity'],
    'BodyComfort_ClimateControlHVAC': ['cabin_temperature'],
    'OccupantSafety_CrashStatus': ['crash_status'],
    'OccupantSafety_OccupantDetection': ['seat_occupancy'],
    'Connectivity_TelematicsControlUnit': ['vehicle_time', 'gnss_position'],
    'Infotainment_HeadUnitApplications': ['user_settings'],
}
SHARED_SIGNALS_PER_COMPONENT = (0, 2)   # how many shared signals a component reads (uniform), ~1 on average: ~5,500 cross-function links at scale 25 (reference 2,500-7,000+)
DMIPS_PER_US_AT_1 = 1.0 / 1000.0   # CPU demand = sum(WCET / period) of the runnables, expressed as a share of one
                                   # TC39x-class core; converted to DMIPS with the core's rating below
CORE_DMIPS = {'AURIX_TC39x': 700, 'S32K3': 685, 'A76': 20000}
# WCET measured on AURIX TC39x (CARIAD); a core of that class is rated about 700 DMIPS (Infineon: 2,700 DMIPS over
# 4 lockstep cores / S32K3 M7 685 DMIPS, NXP), so demand_DMIPS = load_share x 700


def pick(rng, values, shares):
    return rng.choices(values, weights=shares, k=1)[0]


def log_uniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


# --------------------------------------------------------------------------------------------- non-AD generation
def generate_non_ad(catalogue, scale, rng, devices):
    """components, flows and chains of the non-AD domains from the function catalogue and the CARIAD distributions"""
    comps, flows, chains = [], [], []
    for domain, functions in catalogue['domains'].items():
        for fn in functions:
            n_swc = max(2, round(fn['swc_count'] * scale))
            base_period = fn['period_ms']
            fid = f"{domain}_{fn['id']}"
            ids = []
            mem = fn.get('mem_mb')   # POSIX application: MB-scale memory of the whole function, shared by its components
            if mem:
                fn_rom_kb = log_uniform(rng, *mem['rom']) * 1024.0 / n_swc
                fn_ram_kb = log_uniform(rng, *mem['ram']) * 1024.0 / n_swc
            for k in range(n_swc):
                cid = f"{fid}_{k:03d}"
                # ASIL: the function's safety-critical share at the function's ASIL, the rest lower (CARIAD mix as
                # the prior); the first component of a critical function is always at the function's ASIL
                if fn['asil'] >= 3 and (k == 0 or rng.random() < 0.6):
                    asil = fn['asil']
                else:
                    asil = min(fn['asil'], pick(rng, ASIL_LEVELS, ASIL_SHARE))
                # period: the function's base period for the first component, CARIAD mix otherwise (clipped to the
                # neighbourhood of the base period: communication happens between equal / adjacent periods, Table 4)
                if k == 0:
                    period = min(PERIODS, key=lambda q: abs(math.log(q) - math.log(base_period)))   # nearest CARIAD class
                else:
                    cands = [p for p in PERIODS if 0.2 * base_period <= p <= 5 * base_period] or [base_period]
                    w = [PERIOD_SHARE[PERIODS.index(p)] for p in cands]
                    period = pick(rng, cands, w)
                n_run = rng.choice([1, 1, 1, 2, 2, 3])
                runnables = []
                for r in range(n_run):
                    if period in WATERS_ACET_US and rng.random() < 0.5:
                        lo, avg, hi = WATERS_ACET_US[period]
                        if period in WATERS_ACET_WEIBULL:
                            k, lam = WATERS_ACET_WEIBULL[period]
                            acet = min(hi, max(lo, rng.weibullvariate(lam, k)))
                        else:
                            acet = rng.uniform(lo, hi)
                        wc = acet * rng.uniform(*WATERS_WCET_FACTOR[period])
                        runnables.append({'period_ms': period, 'wcet_us': round(wc, 1), 'acet_us': round(acet, 2), 'source': 'WATERS Tables IV-V'})
                    else:
                        wc = log_uniform(rng, *WCET_US[period])
                        runnables.append({'period_ms': period, 'wcet_us': round(wc, 1), 'source': 'CARIAD Table 2'})
                # CPU demand from the average execution time (WATERS ACET where drawn, else WCET / typical WCET-ACET
                # factor of the period class); the worst-case load is kept as a separate field
                def acet_of(rr):
                    if 'acet_us' in rr:
                        return rr['acet_us']
                    f = WATERS_WCET_FACTOR.get(rr['period_ms'], (1.5, 5.0))
                    return rr['wcet_us'] / ((f[0] + f[1]) / 2)
                load_share = sum(acet_of(rr) / (rr['period_ms'] * 1000.0) for rr in runnables)
                wc_share = sum(rr['wcet_us'] / (rr['period_ms'] * 1000.0) for rr in runnables)
                # memory per component = per SW-C (CARIAD Table 1 ranges)
                if mem:   # POSIX / Android / Adaptive application: the function's share per component (sources in the catalogue)
                    rom = round(fn_rom_kb, 1)
                    ram = round(fn_ram_kb, 2)
                else:
                    rom = round(log_uniform(rng, *ROM_KB[asil]), 1)   # per SW-C (CARIAD Table 1); a component is one SW-C
                    ram = round(log_uniform(rng, *RAM_KB[asil]), 2)
                role = 'sensor' if (k == 0 and fn['sensors']) else ('actuator' if (k == n_swc - 1 and fn['actuators']) else 'processing')
                comps.append({'id': cid, 'domain': domain, 'function': fn['id'], 'role': role, 'asil': asil,
                              'period_ms': period, 'runnables': runnables,
                              'cpu_dmips': round(load_share * CORE_DMIPS['AURIX_TC39x'], 3), 'cpu_dmips_worst': round(wc_share * CORE_DMIPS['AURIX_TC39x'], 3), 'ram_kb': ram, 'rom_kb': rom,
                              'sensors': fn['sensors'] if role == 'sensor' else [], 'actuators': fn['actuators'] if role == 'actuator' else [],
                              'bus': fn['bus'], 'platform': (mem['platform'] if mem else 'classic'),
                              'source': f"generated:catalogue={fn['source']}; distributions={CARIAD}" + (f"; memory: {mem['source']}" if mem else '')})
                ids.append(cid)
            # links inside the function: a pipeline sensor -> processing ... -> actuator, plus random extra links
            # between components of equal / adjacent periods (Table 4); message = one CAN/LIN signal frame
            def add_flow(a, b, kind):
                pa = next(c for c in comps if c['id'] == a)['period_ms']
                lb = pick(rng, WATERS_LABEL_BYTES, WATERS_LABEL_SHARE)
                flows.append({'src': a, 'dst': b, 'bytes': max(CAN_SIGNAL_BYTES, lb), 'label_bytes': lb, 'rate_hz': round(1000.0 / pa, 3),
                              'kind': kind, 'source': 'generated: pipeline + CARIAD Table 4; label size WATERS Table I; carried in 8-byte CAN/LIN frames (ISO 11898)'})
            for a, b in zip(ids, ids[1:]):
                add_flow(a, b, 'pipeline')
            extra = max(0, round(n_swc * 0.5))
            for _ in range(extra):
                a, b = rng.sample(ids, 2)
                if not any(f['src'] == a and f['dst'] == b for f in flows):
                    add_flow(a, b, 'extra')
            # chains: one or two per function, lengths from Table 3, along the pipeline; bound = hyperperiod x factor
            n_run_fn = sum(len(x['runnables']) for x in comps if x['function'] == fn['id'] and x['domain'] == domain)
            n_chains = max(1, round(n_run_fn * CHAINS_PER_100_RUNNABLES / 100.0))
            for c in range(n_chains):
                L = min(pick(rng, CHAIN_LEN, CHAIN_LEN_SHARE) if rng.random() < 0.5 else pick(rng, WATERS_CHAIN_LEN, WATERS_CHAIN_LEN_SHARE), n_swc)
                start = rng.randrange(0, n_swc - L + 1)
                members = ids[start:start + L]
                periods = [next(x for x in comps if x['id'] == m)['period_ms'] for m in members]
                hyper = max(periods)
                # two published rules for the end-to-end constraint: WATERS (sum of the periods of the activation
                # patterns involved) and CARIAD (hyperperiod x 1.8 ... 4.9); the generator alternates and records which
                if c % 2 == 0:
                    bound = float(sum(sorted(set(periods))))
                    rule = f'WATERS: sum of the periods of the activation patterns ({WATERS})'
                else:
                    bound = hyper * rng.uniform(*DATA_AGE_FACTOR)
                    rule = f'CARIAD: hyperperiod x {DATA_AGE_FACTOR} ({CARIAD})'
                ftti, ftti_src = FTTI_MS.get(fn['id'], (None, None))
                if ftti is None and fn['asil'] >= 3:
                    ftti, ftti_src = 4.0 * bound, 'derived: no published value; FTTI = 4 x the chain end-to-end bound (to be replaced by a HARA value)'
                chains.append({'id': f"{fid}_chain{c}", 'swcs': members, 'e2e_bound_ms': round(bound, 1),
                               'ftti_ms': round(ftti, 1) if ftti else None, 'ftti_source': ftti_src,
                               'source': f"generated: lengths CARIAD Table 3 / WATERS Table VII; bound {rule}"})
    # shared vehicle signals: every component reads 0-4 signals from the source functions (one 8-byte frame each,
    # at the reader's period); this is where most inter-ECU signals of a real vehicle come from
    src_comp = {}
    for fid, signals in SHARED_SIGNAL_SOURCES.items():
        prod = [c for c in comps if f"{c['domain']}_{c['function']}" == fid]
        if prod:
            src_comp[fid] = (prod[-1]['id'], signals)
    n_shared = 0
    for c in comps:
        for _ in range(rng.randint(*SHARED_SIGNALS_PER_COMPONENT)):
            fid = rng.choice(list(src_comp))
            sid, signals = src_comp[fid]
            if sid == c['id'] or f"{c['domain']}_{c['function']}" == fid:
                continue
            flows.append({'src': sid, 'dst': c['id'], 'bytes': CAN_SIGNAL_BYTES, 'label_bytes': 2, 'rate_hz': round(1000.0 / c['period_ms'], 3),
                          'signal': rng.choice(signals), 'kind': 'shared_signal',
                          'source': 'generated: shared vehicle signal (inter-ECU signal counts: ~2,500 luxury car 2008, >7,000 Volvo, IEEE Spectrum); 8-byte frame'})
            n_shared += 1
    # memory calibration: the CARIAD per-SW-C ranges are wide (6-255 kB); drawn log-uniformly they sum to about
    # 3x the flash of a real ECU. Scale ROM and RAM so that 1,250 runnables (one engine ECU, ECRTS 2017) hold
    # 16 MB ROM (TC39x flash, Infineon) and 6.9 MB RAM minus the BSW share (0.6-1.3 MB ROM / core, CARIAD), keeping
    # every value inside its ASIL range. Recorded in the component source.
    classic = [c for c in comps if c['platform'] == 'classic']   # POSIX applications (MB-scale, own sources) are not calibrated
    n_run = sum(len(c['runnables']) for c in classic)
    units = n_run / 1250.0
    target_rom_kb = (16 * 1024 - 4 * 1000) * units       # 16 MB flash minus 4 cores x ~1 MB BSW
    target_ram_kb = (6.9 * 1024 - 4 * 200) * units       # 6.9 MB SRAM minus BSW / stack estimate
    f_rom = target_rom_kb / sum(c['rom_kb'] for c in classic)
    f_ram = target_ram_kb / sum(c['ram_kb'] for c in classic)
    for c in classic:
        lo, hi = ROM_KB[c['asil']]
        c['rom_kb'] = round(min(hi, max(lo, c['rom_kb'] * f_rom)), 1)
        lo, hi = RAM_KB[c['asil']]
        c['ram_kb'] = round(min(hi, max(lo, c['ram_kb'] * f_ram)), 2)
        c['source'] += f'; memory scaled x{f_rom:.2f} (ROM) / x{f_ram:.2f} (RAM) to a TC39x flash / SRAM budget per 1,250 runnables, within the CARIAD ranges'
    return comps, flows, chains


# ------------------------------------------------------------------------------------------------- cross-domain
def cross_domain_links(comps, flows):
    """links between domains that exist in Autoware's vehicle interface and in the AUTOSAR inter-domain interfaces"""
    by_fn = defaultdict(list)
    for c in comps:
        by_fn[c['function']].append(c['id'])

    def first(fn):
        return by_fn[fn][0] if by_fn.get(fn) else None

    def last(fn):
        return by_fn[fn][-1] if by_fn.get(fn) else None

    pairs = [
        # Autoware vehicle interface: control_cmd -> powertrain coordinator / brake / steering; status back
        ('AW:/control/vehicle_cmd_gate', first('PowertrainCoordinator'), 71, 33.0, 'autoware_control_msgs/Control (vehicle interface)'),
        ('AW:/control/vehicle_cmd_gate', first('BrakeByWireActuation'), 71, 33.0, 'autoware_control_msgs/Control'),
        ('AW:/control/vehicle_cmd_gate', first('SteeringSystemEps'), 71, 33.0, 'autoware_control_msgs/Control'),
        ('AW:/control/vehicle_cmd_gate', first('ExteriorLight'), 8, 10.0, 'turn indicators / hazard lights commands'),
        ('AW:/control/vehicle_cmd_gate', first('GearParkLock'), 8, 10.0, 'gear command'),
        (last('DriverRequest'), 'AW:/sensing/vehicle_velocity_converter', 8, 30.0, 'velocity_status (vehicle status report)'),
        (last('SteeringSystemEps'), 'AW:/sensing/vehicle_velocity_converter', 8, 30.0, 'steering_status'),
        (last('GearParkLock'), 'AW:/sensing/vehicle_velocity_converter', 8, 30.0, 'gear_status'),
        # AUTOSAR inter-domain interfaces (body <-> HMI, chassis -> body)
        (last('Esc'), first('ExteriorLight'), 8, 20.0, 'brake light request (AI_BODY 4.6.3 BrakeLightManager)'),
        (last('CentralLocking'), first('HeadUnitApplications'), 8, 10.0, 'lock status to HMI (AI_BODY 4.3.8)'),
        (last('ClimateControlHVAC'), first('HeadUnitApplications'), 8, 10.0, 'climate status to HMI'),
        (first('HeadUnitApplications'), first('ClimateControlHVAC'), 8, 10.0, 'climate set points from HMI'),
        (last('BatteryManagement'), first('InstrumentCluster'), 8, 10.0, 'state of charge to cluster'),
        (last('PowertrainCoordinator'), first('InstrumentCluster'), 8, 20.0, 'speed / torque to cluster'),
        (last('PostCrashInformation'), first('ECall'), 8, 10.0, 'crash information to eCall (AI_OPS 401)'),
        (last('TelematicsControlUnit'), first('OtaUpdateManager'), 125000, 10.0, 'OTA download (estimate 10 Mbit/s)'),
        (last('OtaUpdateManager'), first('CentralGateway'), 125000, 10.0, 'OTA package to the gateway'),
        (last('V2xGateway'), 'AW:/planning/scenario_planning/scenario_selector', 700, 10.0, 'external traffic signals / CAM (ETSI)'),
        (last('TirePressureMonitoring'), first('InstrumentCluster'), 8, 1.0, 'tire pressure warning'),
        (last('ParkingAid'), first('HeadUnitApplications'), 8, 20.0, 'PDC distances to HMI (AI_HMI PDC example)'),
    ]
    for src, dst, nbytes, hz, note in pairs:
        if src and dst:
            flows.append({'src': src, 'dst': dst, 'bytes': nbytes, 'rate_hz': hz, 'kind': 'cross_domain',
                          'source': f"standard/interface: {note}"})


# ----------------------------------------------------------------------------------------------- Autoware part
def parse_topic_info(path):
    """{topic: {'type': t, 'pubs': [nodes], 'subs': [nodes]}} from `ros2 topic info -v` dumps"""
    graph, topic, section = {}, None, None
    for line in open(path):
        s = line.rstrip('\n')
        if s.startswith('=== '):
            topic = s[4:].strip()
            graph[topic] = {'type': None, 'pubs': [], 'subs': []}
            section = None
            continue
        if topic is None:
            continue
        if s.startswith('Type:'):
            graph[topic]['type'] = s.split(':', 1)[1].strip()
        elif s.startswith('Publisher count'):
            section = 'pubs'
        elif s.startswith('Subscription count'):
            section = 'subs'
        elif s.startswith('Node name:') and section:
            graph[topic][section].append(s.split(':', 1)[1].strip())
        elif s.startswith('Node namespace:') and section:
            ns = s.split(':', 1)[1].strip()
            if graph[topic][section]:
                name = graph[topic][section][-1]
                graph[topic][section][-1] = (ns.rstrip('/') + '/' + name) if ns != '/' else '/' + name
    return graph


def load_measurements(bench):
    out = os.path.join(bench, 'out2_prev')   # the run with hz / bw files
    pt = {}
    # rosbag replay with perception: run 7 (job 1075, route set, planners active) if present, else run 6 (no route)
    p = next((x for x in (os.path.join(bench, 'pt_stats_run7.txt'), os.path.join(bench, 'pt_stats_run6.txt')) if os.path.exists(x)), '')
    if p:
        for line in open(p):
            m = re.match(r'(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if m:
                pt[m.group(1)] = {'n': int(m.group(2)), 'mean_ms': float(m.group(3)), 'p99_ms': float(m.group(4)), 'max_ms': float(m.group(5))}
    pt_psim = {}
    p = os.path.join(bench, 'pt_stats_psim_moving.txt')   # planning simulator with a route, driving window (job 1073)
    if os.path.exists(p):
        for line in open(p):
            m = re.match(r'(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if m:
                pt_psim[m.group(1)] = {'n': int(m.group(2)), 'mean_ms': float(m.group(3)), 'p99_ms': float(m.group(4)), 'max_ms': float(m.group(5))}
    cpu = {}
    p = os.path.join(bench, 'cpu_by_node.json')   # node path -> CPU / RSS of its process (real-time run + camera run)
    if os.path.exists(p):
        cpu = json.load(open(p))
    # planning / control nodes were idle in the rosbag runs (no route); the planning-simulator run (job 1073, route set,
    # vehicle driving) gives their real load -> it overrides the rosbag values for those nodes
    p = os.path.join(bench, 'cpu_by_node_psim.json')
    if os.path.exists(p):
        for k, v in json.load(open(p)).items():
            if k.startswith('/planning') or k.startswith('/control'):
                cpu[k] = v
    # container -> nodes loaded into it (launch log), so that a container process's CPU / RSS can be shared among
    # the nodes it hosts; a node's own process (``__node:=``) keeps its full load
    hosted = defaultdict(list)
    # launch logs of the rosbag runs (out2 of the latest run, or the kept copies of earlier runs) and of the simulator run
    for launch_log in (os.path.join(bench, d, 'launch.log') for d in ('out2', 'out2_run7_timeout', 'out2_prev', 'out_psim')):
        if not os.path.exists(launch_log):
            continue
        for line in open(launch_log):
            m = re.search(r"Loaded node '([^']+)' in container '([^']+)'", line)
            if m:
                cont = m.group(2) if m.group(2).startswith('/') else '/' + m.group(2)
                if m.group(1) not in hosted[cont]:
                    hosted[cont].append(m.group(1))
    for cont, nodes_in in hosted.items():
        c = cpu.get(cont)
        if c:
            for n in nodes_in:
                cpu.setdefault(n, {'cpu_mean_pct': round(c['cpu_mean_pct'] / len(nodes_in), 2),
                                   'cpu_max_pct': round(c['cpu_max_pct'] / len(nodes_in), 2),
                                   'rss_max_mb': round(c['rss_max_mb'] / len(nodes_in), 1), 'shared_container': cont})
    rates = {}
    topics_all = []
    tt = os.path.join(bench, 'out2', 'topics_types.txt')
    if os.path.exists(tt):
        topics_all = [l.split()[0] for l in open(tt) if l.startswith('/')]
    for f in os.listdir(out) if os.path.isdir(out) else []:
        if f.startswith('hz_'):
            stem = f[3:-4]
            t = next((x for x in topics_all if x.strip('/').replace('/', '_') == stem), '/' + stem.replace('_', '/'))
            txt = open(os.path.join(out, f)).read()
            hz = re.findall(r'average rate: ([\d.]+)', txt)
            bwf = os.path.join(out, 'bw_' + f[3:])
            size = re.findall(r'Message size mean: ([\d.]+) ([KM]?B)', open(bwf).read()) if os.path.exists(bwf) else []
            if hz:
                kb = None
                if size:
                    v, u = size[-1]
                    kb = float(v) * {'B': 1 / 1024, 'KB': 1, 'MB': 1024}[u]
                rates[t] = {'hz': float(hz[-1]), 'size_kb': kb}
    return pt, pt_psim, cpu, rates


NODE_TOPIC_ALIAS = {'/localization/pose_twist_fusion_filter/ekf_localizer': '/localization/pose_twist_fusion_filter',
                    '/localization/pose_estimator/ndt_scan_matcher': '/localization/pose_estimator'}


def build_autoware(bench, sensors, rng):
    topic_info = os.path.join(bench, 'out_graph', 'topic_info.txt')
    graph = parse_topic_info(topic_info) if os.path.exists(topic_info) else {}
    pt, pt_psim, cpu, rates = load_measurements(bench)

    def node_time(table, n):
        """The node's processing time per cycle from its debug topics: the total where several module times exist
        (largest mean; lateral + longitudinal controller summed), never a cyclic-time or pipeline-latency topic."""
        # some nodes publish their debug topics in the parent namespace (EKF: /localization/pose_twist_fusion_filter/debug/...,
        # NDT: /localization/pose_estimator/exe_time_ms); those namespaces hold a single processing node
        prefixes = [n + '/'] + [a + '/' for node, a in NODE_TOPIC_ALIAS.items() if node == n]
        cands = {k: v for k, v in table.items() if any(k.startswith(pf) for pf in prefixes) and 'cyclic_time' not in k and 'pipeline_latency' not in k}
        if not cands:
            return None
        lat = [v for k, v in cands.items() if '/lateral/' in k]
        lon = [v for k, v in cands.items() if '/longitudinal/' in k]
        if lat and lon:
            return {key: round(lat[0][key] + lon[0][key], 2) for key in ('mean_ms', 'p99_ms', 'max_ms')} | {'n': lat[0]['n'], 'topic': 'lateral + longitudinal'}
        k = max(cands, key=lambda x: cands[x]['mean_ms'])
        pf = next(x for x in prefixes if k.startswith(x))
        return dict(cands[k]) | {'topic': k[len(pf) - 1:]}
    nodes = set()
    for t, g in graph.items():
        nodes.update(g['pubs']); nodes.update(g['subs'])
    nodes = sorted(n for n in nodes if not re.search(r'transform_listener_impl|^/launch_ros|^/_ros2cli|rosbag2|/rviz', n))
    comps, flows = [], []
    for n in nodes:
        dom = 'ADAS'
        if n.startswith('/sensing/'):
            dom = 'Sensing'
        elif n.startswith('/system/') or n.startswith('/adapi/') or n.startswith('/map/'):
            dom = 'Platform'
        ptn, pt_src = node_time(pt, n), 'measured: rosbag replay with perception (run 7 if recorded, else run 6)'
        if ptn is None and node_time(pt_psim, n):
            ptn, pt_src = node_time(pt_psim, n), 'measured: planning simulator with a route, driving window (job 1073)'
        c = cpu.get(n) or next((v for k, v in cpu.items() if k.split('/')[-1] == n.split('/')[-1] and not k.startswith('/')), None)
        dmips = round(c['cpu_mean_pct'] / 100.0 * CUR['host_to_target_factor'] * CUR['reference_core_dmips'], 1) if c and c.get('cpu_mean_pct') is not None else None
        comps.append({'id': 'AW:' + n, 'domain': dom, 'function': 'autoware', 'role': 'processing', 'asil': None,
                      'processing_time_ms': ptn, 'processing_time_source': pt_src if ptn else None,
                      'cpu_mean_pct': c['cpu_mean_pct'] if c else None, 'rss_max_mb': c['rss_max_mb'] if c else None, 'cpu_run': c.get('run') if c else None,
                      'cpu_dmips': dmips, 'cpu_dmips_source': ('measured host core share x %.2f (host -> target, Betz et al. arXiv 2404.12683) x %d DMIPS (Cortex-A76 1.8 GHz, Renesas V4H flyer); configs/partitions.json cpu_currency' % (CUR['host_to_target_factor'], CUR['reference_core_dmips'])) if dmips is not None else None,
                      'shared_container': c.get('shared_container') if c else None,
                      'source': 'measured: Autoware universe-cuda-jazzy on an A100 node; rosbag replay (jobs 1034/1044/run 6) and planning simulator with a route (job 1073)'})
    skip = re.compile(r'^/(parameter_events|rosout|clock|tf|tf_static|diagnostics)')
    for t, g in graph.items():
        if skip.match(t) or '/debug/' in t:
            continue
        r = rates.get(t)
        for p in g['pubs']:
            for s in g['subs']:
                if p != s and 'AW:' + p in {c['id'] for c in comps} and 'AW:' + s in {c['id'] for c in comps}:
                    flows.append({'src': 'AW:' + p, 'dst': 'AW:' + s, 'topic': t, 'type': g['type'],
                                  'rate_hz': r['hz'] if r else None, 'bytes': int(r['size_kb'] * 1024) if (r and r['size_kb']) else None,
                                  'kind': 'ros2_topic', 'source': 'measured: topic graph (ros2 topic info -v); rate/size: ros2 topic hz/bw where measured'})
    # the measured Autoware end-to-end chain (lidar -> localization / perception -> planning -> control -> command),
    # as the node sequence of Betz et al. (arXiv 2404.12683) restricted to the nodes present; the network budget is
    # the total bound (100 ms, Lin et al. ASPLOS 2018) minus the measured computation of the chain nodes
    chain_nodes = ['/sensing/lidar/top/velodyne_ros_wrapper_node', '/sensing/lidar/concatenate_data',
                   '/localization/pose_estimator/ndt_scan_matcher', '/localization/pose_twist_fusion_filter/ekf_localizer',
                   '/perception/object_recognition/detection/centerpoint/lidar_centerpoint',
                   '/perception/object_recognition/tracking/multi_object_tracker',
                   '/perception/object_recognition/prediction/map_based_prediction',
                   '/planning/scenario_planning/lane_driving/behavior_planning/behavior_path_planner',
                   '/planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner',
                   '/planning/scenario_planning/velocity_smoother', '/planning/planning_validator',
                   '/control/trajectory_follower/controller_node_exe', '/control/vehicle_cmd_gate']
    ids = {c['id'] for c in comps}
    members = ['AW:' + n for n in chain_nodes if 'AW:' + n in ids]
    chains = []
    if members:
        comp_mean = sum((c['processing_time_ms'] or {}).get('mean_ms', 0.0) for c in comps if c['id'] in members)
        comp_p99 = sum((c['processing_time_ms'] or {}).get('p99_ms', 0.0) for c in comps if c['id'] in members)
        chains.append({'id': 'AW_lidar_to_command', 'swcs': members, 'e2e_bound_ms': 100.0, 'comp_mean_ms': round(comp_mean, 1), 'comp_p99_sum_ms': round(comp_p99, 1),
                       'net_budget_ms': round(100.0 - comp_mean, 1), 'ftti_ms': None,
                       'source': 'chain: Betz et al. arXiv 2404.12683 (Autoware sensor-to-command); bound 100 ms: Lin et al. ASPLOS 2018; '
                                 'computation: sum of the measured mean processing times (p99 sum recorded separately; Betz et al.: 52.8 ms mean computation on EPYC 7313P)'})
    return comps, flows, chains


def replicate_sensor_chains(aw_comps, aw_flows, sensors):
    """our suite has 5 lidars and 8 cameras: Autoware's per-lidar driver chain (velodyne driver, distortion corrector,
    crop box x2, ring outlier filter) exists for top/left/right/rear in the sample kit; it is replicated for the
    fifth lidar (front), and the traffic-light camera chain (camera6/7) for the remaining six cameras"""
    lidars = [s['id'] for s in sensors if s['type'] == 'LIDAR']
    cams = [s['id'] for s in sensors if s['type'] == 'CAMERA']
    comps, flows = list(aw_comps), list(aw_flows)
    tmpl_lidar = [c for c in aw_comps if c['id'].startswith('AW:/sensing/lidar/top/')]
    tmpl_cam = [c for c in aw_comps if '/camera6/' in c['id']]

    def clone(templates, old, new, tag):
        mapping = {}
        for c in templates:
            nc = dict(c)
            nc['id'] = c['id'].replace(old, new)
            nc['source'] = c['source'] + f'; replicated for {tag} (same node chain as the sample kit)'
            mapping[c['id']] = nc['id']
            comps.append(nc)
        for f in aw_flows:
            if f['src'] in mapping or f['dst'] in mapping:
                nf = dict(f)
                nf['src'] = mapping.get(f['src'], f['src'])
                nf['dst'] = mapping.get(f['dst'], f['dst'])
                flows.append(nf)
    kit_lidars = ['top', 'left', 'right', 'rear']
    for extra in lidars[len(kit_lidars):]:
        clone(tmpl_lidar, '/sensing/lidar/top/', f'/sensing/lidar/{extra.lower()}/', extra)
    for k, cam in enumerate(cams[2:], start=8):
        clone(tmpl_cam, '/camera6/', f'/camera{k}/', cam)
    return comps, flows


# ------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', default=os.path.expanduser('~/autoware_bench'))
    ap.add_argument('--functions', default=os.path.join(HERE, 'arch_v3_functions.json'))
    ap.add_argument('--scale', type=float, default=1.0, help='multiplier on the catalogue component counts')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(REPO, 'configs', 'arch_v3'))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    catalogue = json.load(open(args.functions))
    sensors = json.load(open(os.path.join(REPO, 'configs', 'sensors.json')))['sensors']
    aw_comps, aw_flows, aw_chains = build_autoware(args.bench, sensors, rng)
    aw_comps, aw_flows = replicate_sensor_chains(aw_comps, aw_flows, sensors)
    comps, flows, chains = generate_non_ad(catalogue, args.scale, rng, sensors)
    chains = aw_chains + chains
    cross_domain_links(comps + aw_comps, flows)
    os.makedirs(args.out, exist_ok=True)
    meta = {'generated': '2026-10-23', 'seed': args.seed, 'scale': args.scale,
            'counts': {'autoware_components': len(aw_comps), 'non_ad_components': len(comps), 'flows': len(flows) + len(aw_flows), 'chains': len(chains)},
            'sources': catalogue['sources']}
    json.dump({'meta': meta, 'components': aw_comps + comps}, open(os.path.join(args.out, 'swc.json'), 'w'), indent=1)
    json.dump({'meta': meta, 'flows': aw_flows + flows}, open(os.path.join(args.out, 'flows.json'), 'w'), indent=1)
    json.dump({'meta': meta, 'chains': chains}, open(os.path.join(args.out, 'chains.json'), 'w'), indent=1)
    by_dom = defaultdict(int)
    for c in aw_comps + comps:
        by_dom[c['domain']] += 1
    print('components per domain:', dict(by_dom))
    print('flows:', len(aw_flows), 'measured +', len(flows), 'generated;', 'chains:', len(chains))
    asil = defaultdict(int)
    for c in comps:
        asil[c['asil']] += 1
    tot = sum(asil.values()) or 1
    print('non-AD ASIL share:', {k: f"{v / tot:.0%}" for k, v in sorted(asil.items())})
    print('non-AD CPU demand total (DMIPS):', round(sum(c['cpu_dmips'] for c in comps)))


if __name__ == '__main__':
    main()
