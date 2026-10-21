import itertools
import os
import time
import numpy as np
os.environ.setdefault("GRB_LICENSE_FILE", os.path.expanduser("~/gurobi.lic"))
import gurobipy as gp
from gurobipy import GRB
from optimizer_utils import (get_distance, build_sensor_lookup, build_actuator_lookup,
                             build_cost_map, build_latency_map, extract_solution,
                             sw_multiplier, site_derating, pack_partitions, flow_routes, chain_latencies, steiner_tree,
                             resource_messages, message_kbit, resource_rate, crossing_ms, robust_ms, device_line, device_ms,
                             message_prio, frame_kbit, queue_ms)

# Uncertainty-aware constraint blocks, each switched on by its own CLI flag (parameters in configs/uncertainty.json)
UNCERTAINTY_BLOCKS = ('sw_margin', 'derating', 'redundancy', 'ai_contention', 'robust_latency', 'growth', 'failures')

# Packing check of the pooled (site, ASIL) capacity constraint: cuts added before giving up (see optimize())
MAX_PACKING_CUTS = 10


class AssignmentOptimizer:
    def __init__(self):
        print("Initialized AssignmentOptimizerNew with quadratic ECU-ECU costs and linear sensor/actuator costs")
        self.last_values = None   # variable values of the last solve, the start solution of the next one in a sweep

    def calculate_cable_expressions(self, x, scs, locations, sensors, actuators, cable_types, comm_matrix, feasible_locs_per_sc, comm=None, attach_s=None, attach_a=None, bus=None):
        """
        Cable cost and cable length as Gurobi linear expressions (not constraints): devices wired directly to the
        unit of their software, Ethernet device attachments, bus lines (CAN, LIN, FlexRay) and backbone cables.
        """
        cost_map = build_cost_map(cable_types)
        sensor_lookup = build_sensor_lookup(sensors)
        actuator_lookup = build_actuator_lookup(actuators)
        bus_types = set((bus or {}).get('types', ()))
        cost_terms, dist_terms = [], []

        # devices wired directly to the unit hosting their software (not Ethernet, not on a bus line)
        for i, sc in enumerate(scs):
            devs = [sensor_lookup.get(s_id) for s_id in (sc.sensors or [])] + [actuator_lookup.get(a_id) for a_id in (sc.actuators or [])]
            devs = [d for d in devs if d is not None and d.location and d.interface != 'ETH' and d.interface not in bus_types]
            if not devs:
                continue
            for j in feasible_locs_per_sc[i]:
                dist = sum(get_distance(d.location, locations[j].location) for d in devs)
                cost = sum(get_distance(d.location, locations[j].location) * cost_map.get(d.interface, 0.0) for d in devs)
                if dist > 0:
                    cost_terms.append((x[i, j], cost))
                    dist_terms.append((x[i, j], dist))

        # Ethernet devices: cable to their attachment unit
        for att, devs in ((attach_s, sensors), (attach_a, actuators)):
            for (di, j), av in (att or {}).items():
                if getattr(devs[di], 'location', None):
                    dist = get_distance(devs[di].location, locations[j].location)
                    cost_terms.append((av, dist * cost_map.get(getattr(devs[di], 'interface', 'ETH'), 0.0)))
                    dist_terms.append((av, dist))

        # bus lines: one cable through all members (length variable, set by the Steiner cuts)
        for b, lv in (bus or {}).get('len', {}).items():
            cost_terms.append((lv, cost_map.get(b[0], 0.0)))
            dist_terms.append((lv, 1.0))

        # backbone cables between units
        for (j1, j2, iface), comm_var in (comm or {}).items():
            dist = get_distance(locations[j1].location, locations[j2].location)
            cost_terms.append((comm_var, dist * cable_types[iface].cost_per_meter))
            dist_terms.append((comm_var, dist))

        cable_cost_expr = gp.quicksum(v * c for v, c in cost_terms) if cost_terms else 0
        cable_distance_expr = gp.quicksum(v * d for v, d in dist_terms) if dist_terms else 0
        return cable_cost_expr, cable_distance_expr, 0

    def _create_model_and_variables(self, n_sc, n_locs, scs, locations, unique_asils, all_hw, sensors, actuators, all_interfaces, infeasible_ij=None, comm_matrix=None, partitions=None, backbone=None, cable_types=None, uncertainty=None, chains=None):
        """
        Create Gurobi model and define all decision variables

        Returns:
            model: Gurobi model object
            x: SC placement variables x[i, j] (SC i hosted at site j; only latency-feasible pairs)
            P: Partition count variables P[j, a] (partitions of ASIL a at site j, integer)
            y: Site activation variables y[j] (site j hosts a control unit with >= 1 partition)
            hw_use, if_use, attachment, trunk, comm, traffic flow and flow variables
            feasible_locs_per_sc / feasible_scs_per_loc: feasible pairs after the latency precompute
            var_stats: Variable family counts
        """
        # Create model
        model = gp.Model("ECU_Assignment")
        model.setParam('OutputFlag', 1)
        model.setParam('MIPFocus', 2)
        #model.setParam('Symmetry', 1)
        model.setParam("Method", 6)
        model.setParam("Seed", 0)  # Fixed seed for reproducible results across runs
        #model.setParam("MIPGap", 0.001)  # 1% gap for faster solutions (adjust as needed)
        model.update()

        partitions = partitions or {}
        # Partitions of one ASIL at one site (configs/partitions.json: partition.max_per_asil_per_site)
        max_p_per_asil = int(partitions.get('max_per_asil_per_site', 3))

        # x[i,j] = SC i hosted at site j. No partition slot index: which partition an SC lands in is not a
        # decision (slots are interchangeable), so every design has exactly one representation.
        infeasible_ij = infeasible_ij or set()
        feasible_locs_per_sc = {i: [] for i in range(n_sc)}
        feasible_scs_per_loc = {j: [] for j in range(n_locs)}
        x = {}
        for i in range(n_sc):
            for j in range(n_locs):
                if (i, j) in infeasible_ij:
                    continue
                feasible_locs_per_sc[i].append(j)
                feasible_scs_per_loc[j].append(i)
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{scs[i].id}_{locations[j].id}")

        for i in range(n_sc):
            if not feasible_locs_per_sc[i]:
                raise ValueError(f"SC {scs[i].id} has no feasible locations after latency precompute")

        # P[j,a] = number of partitions of ASIL a at site j (bounded by the SCs of that ASIL that may be hosted there)
        P = {}
        for j in range(n_locs):
            for a in unique_asils:
                n_sc_a = sum(1 for i in feasible_scs_per_loc[j] if scs[i].asil_req == a)
                P[j, a] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=min(max_p_per_asil, n_sc_a),
                                       name=f"P_{locations[j].id}_asil{a}")

        # y[j] = site j hosts a control unit with at least one partition
        y = {j: model.addVar(vtype=GRB.BINARY, name=f"y_{locations[j].id}") for j in range(n_locs)}

        # --- Standby instances of the critical components (configs/uncertainty.json: failures) ---
        # r[i,j]: standby of SC i at cell j (another cell than the primary's); hot[i]: the standby runs permanently (1)
        # or is a cold standby activated after a failure (0); cold is admissible only when the smallest FTTI of the
        # chains of i covers the platform recovery time (hot[i] is None = hot forced). hr[i,j] = r[i,j] AND hot[i].
        fail_on = bool((uncertainty or {}).get('failures'))
        fail_cfg = (uncertainty or {}).get('params', {}).get('failures', {}) if fail_on else {}
        r, hot, hr, critical = {}, {}, {}, []
        if fail_on:
            asil_min = int(fail_cfg.get('critical_asil_min', 3))
            recovery = float(fail_cfg.get('recovery_ms', 50.0))
            critical = [i for i in range(n_sc) if int(scs[i].asil_req) >= asil_min]
            ftti = {}
            for ch in (chains or []):
                for sid in ch.get('swcs', []):
                    ftti[sid] = min(ftti.get(sid, float('inf')), float(ch.get('ftti_ms', float('inf'))))
            for i in critical:
                cold_ok = ftti.get(scs[i].id, float('inf')) >= recovery
                hot[i] = model.addVar(vtype=GRB.BINARY, name=f"hot_{scs[i].id}") if cold_ok else None
                for j in feasible_locs_per_sc[i]:
                    r[i, j] = model.addVar(vtype=GRB.BINARY, name=f"r_{scs[i].id}_{locations[j].id}")
                    hr[i, j] = (model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"hr_{scs[i].id}_{locations[j].id}")
                                if cold_ok else r[i, j])
            print(f"  Failures: {len(critical)} critical SCs (ASIL >= {asil_min}), cold standby admissible for "
                  f"{sum(1 for i in critical if hot[i] is not None)} of them (recovery {recovery:.0f} ms)")

        # hw_use[j,h] = HW feature h is used at location j (binary)
        hw_use = {}
        for j in range(n_locs):
            for h in all_hw:
                hw_use[j, h] = model.addVar(vtype=GRB.BINARY, name=f"hw_{locations[j].id}_{h}")
        
        # if_use[j,i_name] = Number of Interface i_name ports at location j (integer)
        if_use = {}
        for j in range(n_locs):
            for i_name in all_interfaces:
                if_use[j, i_name] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"if_{locations[j].id}_{i_name}")

        # Attachment variables for ETH sensors/actuators:
        # attach_s[s_idx, j] = 1 if ETH sensor s is physically connected to location j.
        # attach_a[a_idx, j] = 1 if ETH actuator a is physically connected to location j.
        sensor_id_to_idx = {s.id: si for si, s in enumerate(sensors)}
        actuator_id_to_idx = {a.id: ai for ai, a in enumerate(actuators)}

        eth_sensor_indices = [si for si, s in enumerate(sensors) if getattr(s, 'interface', None) == 'ETH']
        eth_actuator_indices = [ai for ai, a in enumerate(actuators) if getattr(a, 'interface', None) == 'ETH']

        attach_s = {}
        for si in eth_sensor_indices:
            for j in range(n_locs):
                attach_s[si, j] = model.addVar(vtype=GRB.BINARY, name=f"attachS_{sensors[si].id}_{locations[j].id}")

        attach_a = {}
        for ai in eth_actuator_indices:
            for j in range(n_locs):
                attach_a[ai, j] = model.addVar(vtype=GRB.BINARY, name=f"attachA_{actuators[ai].id}_{locations[j].id}")

        # Bus lines (CAN, LIN, FlexRay; configs/buses.json: backbone.bus_lines = lines available per type). A line b
        # has members: units (bus_unit[b, j]) and devices of its type (bus_dev[b, d]); bus_len[b] is its cable length,
        # bus_high[b] its ASIL group (devices of ASIL >= 3 software and of lower ASIL never share a line).
        bus_types = [n for n in ('CAN', 'LIN', 'FLEXRAY') if n in all_interfaces]
        bus_cfg = dict((backbone or {}).get('bus_lines', {}))
        bus = {'types': [], 'inst': [], 'act': {}, 'len': {}, 'high': {}, 'unit': {}, 'dev': {}, 'devices': {},
               'points': {**{loc.id: loc.location for loc in locations}, **{d.id: d.location for d in list(sensors) + list(actuators)}}}
        for kind, devs in (('sensor', sensors), ('actuator', actuators)):
            for dev in devs:
                if getattr(dev, 'interface', None) not in bus_types:
                    continue
                users = [i for i, sc in enumerate(scs) if dev.id in (sc.sensors or []) + (sc.actuators or [])]
                bus['devices'][dev.id] = {'iface': dev.interface, 'kind': kind, 'users': users,
                                          'volume': float(getattr(dev, 'volume', 0.0) or 0.0),
                                          'kbit': float(getattr(dev, 'volume', 0.0) or 0.0) * 1000.0 / float(getattr(dev, 'rate_hz', 10.0) or 10.0),
                                          'group': 'HIGH' if any(int(scs[i].asil_req) >= 3 for i in users) else 'LOW'}
        bus['all_types'] = list(bus_types)
        for w_if in bus_types:
            devs = [d for d, info in bus['devices'].items() if info['iface'] == w_if]
            n_lines = int(bus_cfg.get(w_if, max(1, len(devs))))
            if n_lines > 0:
                bus['types'].append(w_if)
            for k in range(n_lines):
                b = (w_if, k)
                bus['inst'].append(b)
                bus['act'][b] = model.addVar(vtype=GRB.BINARY, name=f"bus_{w_if}{k + 1}")
                bus['len'][b] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"busLen_{w_if}{k + 1}")
                bus['high'][b] = model.addVar(vtype=GRB.BINARY, name=f"busHigh_{w_if}{k + 1}")
                for j in range(n_locs):
                    bus['unit'][b, j] = model.addVar(vtype=GRB.BINARY, name=f"busUnit_{w_if}{k + 1}_{locations[j].id}")
                for d in devs:
                    bus['dev'][b, d] = model.addVar(vtype=GRB.BINARY, name=f"busDev_{w_if}{k + 1}_{d}")
        if bus['inst']:
            print(f"  Bus lines: " + ", ".join(f"{n} {sum(1 for b in bus['inst'] if b[0] == n)} line(s) for "
                                               f"{sum(1 for d in bus['devices'].values() if d['iface'] == n)} device(s)" for n in bus['types']))

        # comm[j1,j2,w] = number of cables of type w between sites j1 and j2 (integer). Every cable type of
        # configs/buses.json may connect two units; a site pair may carry cables of several types.
        # Bus types (CAN, LIN, FlexRay) connect units only through their bus lines; a unit-to-unit flow of a bus type
        # takes one line both units belong to (one hop). Types without a line cannot carry backbone traffic.
        bb_ifaces = [n for n in all_interfaces if n not in bus['all_types'] or n in bus['types']]
        cap_of = {n: float(getattr((cable_types or {}).get(n), 'capacity', float('inf'))) for n in bb_ifaces}
        comm = {}
        for j1 in range(n_locs):
            for j2 in range(j1 + 1, n_locs):
                for w_if in bb_ifaces:
                    if w_if in bus['types']:
                        continue
                    comm[j1, j2, w_if] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"comm_{locations[j1].id}_{locations[j2].id}_{w_if}")

        # --- TRAFFIC FLOW GENERATION ---
        traffic_flows = []
        t_idx = 0
        
        # 1. SC-to-SC Flows (Mevcut olan)
        sc_id_map = {sc.id: i for i, sc in enumerate(scs)}
        if comm_matrix:
            for link in comm_matrix:
                src_id = link.get('src')
                dst_id = link.get('dst')
                vol = link.get('volume', 0)
                if vol > 0 and src_id in sc_id_map and dst_id in sc_id_map:
                    traffic_flows.append({
                        'id': t_idx,
                        'type': 'SC_SC',  # Tipini belirtiyoruz
                        'src_idx': sc_id_map[src_id], # Kaynak SC index
                        'dst_idx': sc_id_map[dst_id], # Hedef SC index
                        'volume': vol,
                        'kbit': float(link.get('kbit', 0.0)),
                    })
                    t_idx += 1

        # 2. Sensor-to-SC Flows (ETH only)
        # ETH sensors can attach to a SWITCH or ECU node; their data is then routed over the backbone
        # to the destination SWC location.
        sensor_lookup = build_sensor_lookup(sensors)
        for dst_sc_idx, sc in enumerate(scs):
            for s_id in (getattr(sc, 'sensors', None) or []):
                s_obj = sensor_lookup.get(s_id)
                if not s_obj or getattr(s_obj, 'interface', None) != 'ETH':
                    continue
                vol = getattr(s_obj, 'volume', 0) or 0
                if vol <= 0:
                    continue
                s_idx = sensor_id_to_idx.get(s_id)
                if s_idx is None:
                    continue
                traffic_flows.append({
                    'id': t_idx,
                    'type': 'SENS_SC',
                    'sensor_idx': s_idx,
                    'dst_idx': dst_sc_idx,
                    'volume': vol,
                    'kbit': vol * 1000.0 / float(getattr(s_obj, 'rate_hz', 10.0) or 10.0),
                })
                t_idx += 1

        # 3. SC-to-Actuator Flows (ETH only)
        actuator_lookup = build_actuator_lookup(actuators)
        for src_sc_idx, sc in enumerate(scs):
            for a_id in (getattr(sc, 'actuators', None) or []):
                a_obj = actuator_lookup.get(a_id)
                if not a_obj or getattr(a_obj, 'interface', None) != 'ETH':
                    continue
                vol = getattr(a_obj, 'volume', 0) or 0
                if vol <= 0:
                    continue
                a_idx = actuator_id_to_idx.get(a_id)
                if a_idx is None:
                    continue
                traffic_flows.append({
                    'id': t_idx,
                    'type': 'SC_ACT',
                    'src_idx': src_sc_idx,
                    'act_idx': a_idx,
                    'volume': vol,
                    'kbit': vol * 1000.0 / float(getattr(a_obj, 'rate_hz', 10.0) or 10.0),
                })
                t_idx += 1

        # Standby flows (failures): a standby instance exchanges the flows of its primary with the primary partners
        # and, for critical partners, with their standbys. 'src_inst' / 'dst_inst': 'x' primary, 'r' standby,
        # 'a' peripheral attachment; the endpoints come from the matching variables (flow_ends).
        if critical:
            crit = set(critical)
            for tr in list(traffic_flows):
                variants = []
                if tr['type'] == 'SC_SC':
                    s, d = tr['src_idx'], tr['dst_idx']
                    if s in crit:
                        variants.append(('r', 'x'))
                    if d in crit:
                        variants.append(('x', 'r'))
                    if s in crit and d in crit:
                        variants.append(('r', 'r'))
                elif tr['type'] == 'SENS_SC' and tr['dst_idx'] in crit:
                    variants.append(('a', 'r'))
                elif tr['type'] == 'SC_ACT' and tr['src_idx'] in crit:
                    variants.append(('r', 'a'))
                for si, di in variants:
                    traffic_flows.append({**tr, 'id': t_idx, 'src_inst': si, 'dst_inst': di, 'base': tr['id']})
                    t_idx += 1
            print(f"  Standby flows: {sum(1 for tr in traffic_flows if 'base' in tr)} added to {sum(1 for tr in traffic_flows if 'base' not in tr)} base flows")

        # --- BACKBONE: aggregated traffic per site pair on candidate paths ---
        # The per-flow routing binaries of v1 are gone: all traffic of a site pair takes one of the pair's candidate
        # paths. The model holds no traffic variables: "a pair with traffic needs a selected path" and "link load <=
        # cables x capacity" are lazy constraints added in the callback of optimize(); the LP stays small and the
        # relaxation does not see the traffic.
        k_paths = max(1, int((backbone or {}).get('candidate_paths', 3)))
        # Interface choice per flow: a flow may use every backbone type whose capacity holds its volume (fallback:
        # the largest type); w[t,iface] selects it when there is more than one option.
        w, flow_elig = {}, {}
        for tr in traffic_flows:
            vol = float(tr['volume'])
            elig = [n for n in bb_ifaces if vol <= cap_of[n]] or [max(bb_ifaces, key=lambda n: cap_of[n])]
            flow_elig[tr['id']] = elig
            if len(elig) > 1:
                for n in elig:
                    w[tr['id'], n] = model.addVar(vtype=GRB.BINARY, name=f"w_{tr['id']}_{n}")
        pair_vol, flow_ends = {}, {}
        for tr in traffic_flows:
            si, di = tr.get('src_inst', 'x'), tr.get('dst_inst', 'x')
            if tr['type'] == 'SC_SC':
                src = {u: (r if si == 'r' else x)[tr['src_idx'], u] for u in feasible_locs_per_sc[tr['src_idx']]}
                dst = {v: (r if di == 'r' else x)[tr['dst_idx'], v] for v in feasible_locs_per_sc[tr['dst_idx']]}
            elif tr['type'] == 'SENS_SC':
                src = {u: attach_s[tr['sensor_idx'], u] for u in range(n_locs) if (tr['sensor_idx'], u) in attach_s}
                dst = {v: (r if di == 'r' else x)[tr['dst_idx'], v] for v in feasible_locs_per_sc[tr['dst_idx']]}
            else:  # SC_ACT
                src = {u: (r if si == 'r' else x)[tr['src_idx'], u] for u in feasible_locs_per_sc[tr['src_idx']]}
                dst = {v: attach_a[tr['act_idx'], v] for v in range(n_locs) if (tr['act_idx'], v) in attach_a}
            flow_ends[tr['id']] = (src, dst)
            for u in src:
                for v in dst:
                    if u != v:
                        pair_vol.setdefault((u, v), []).append(float(tr['volume']))

        # Candidate paths of every site pair with possible traffic: the direct link and the (k-1) shortest routes
        # over one relaying site (which then needs a SWITCH). sel[u,v,w,p] selects the path of the pair for cable type w.
        site_dist = {(u, v): get_distance(locations[u].location, locations[v].location)
                     for u in range(n_locs) for v in range(n_locs) if u != v}
        paths, sel = {}, {}
        for (u, v) in sorted({(min(a, b), max(a, b)) for (a, b) in pair_vol}):
            relays = sorted((r for r in range(n_locs) if r not in (u, v)), key=lambda r: site_dist[u, r] + site_dist[r, v])
            for w_if in bb_ifaces:
                # relaying over a switch exists for Ethernet only; other point-to-point types are a direct cable; a bus
                # type offers its lines (candidate p = line p, a one-hop path)
                if w_if in bus['types']:
                    paths[u, v, w_if] = [[u, v] for b in bus['inst'] if b[0] == w_if]
                else:
                    paths[u, v, w_if] = [[u, v]] + ([[u, r, v] for r in relays[:k_paths - 1]] if w_if == 'ETH' else [])
                for p in range(len(paths[u, v, w_if])):
                    sel[u, v, w_if, p] = model.addVar(vtype=GRB.BINARY, name=f"sel_{locations[u].id}_{locations[v].id}_{w_if}_p{p}")
        net = {'paths': paths, 'sel': sel, 'flow_ends': flow_ends, 'bus': bus,
               'bb_ifaces': bb_ifaces, 'cap_of': cap_of, 'w': w, 'flow_elig': flow_elig,
               'r': r, 'hot': hot, 'hr': hr, 'critical': critical, 'fail_cfg': fail_cfg,
               'x_vars': x, 'y_vars': y, 'switch_vars': {j: hw_use.get((j, 'SWITCH')) for j in range(n_locs)},
               'flow_meta': {tr['id']: tr for tr in traffic_flows},
               'k_paths': k_paths, 'volumes': {tr['id']: float(tr['volume']) for tr in traffic_flows},
               'kbit': {tr['id']: float(tr.get('kbit', 0.0)) for tr in traffic_flows},
               'site_ids': [loc.id for loc in locations],
               'hop_ms': {n: float(getattr((cable_types or {}).get(n), 'hop_latency_ms', 0.0)) for n in bb_ifaces},
               'path_link_cut': bool((backbone or {}).get('path_link_cut', False)),
               # a bus line may be loaded up to this share of its bit rate (the queue bound of the latency assumes
               # a bus that is not saturated; configs/buses.json: backbone.bus_max_utilization)
               'bus_umax': float((backbone or {}).get('bus_max_utilization', 1.0))}

        var_stats = {
            'x': len(x),
            'P': len(P),
            'y': len(y),
            'hw_use': len(hw_use),
            'if_use': len(if_use),
            'attach_s': len(attach_s),
            'attach_a': len(attach_a),
            'bus': len(bus['act']) + len(bus['len']) + len(bus['high']) + len(bus['unit']) + len(bus['dev']),
            'comm': len(comm),
            'traffic_flows': len(traffic_flows),
            'sel': len(sel),
            'w': len(w),
            'r': len(r),
            'hot': sum(1 for v in hot.values() if v is not None),
            'pairs_with_paths': len(paths),
        }
        var_stats['total'] = (
            var_stats['x']
            + var_stats['P']
            + var_stats['y']
            + var_stats['hw_use']
            + var_stats['if_use']
            + var_stats['attach_s']
            + var_stats['attach_a']
            + var_stats['bus']
            + var_stats['comm']
            + var_stats['sel']
            + var_stats['w']
            + var_stats['r']
            + var_stats['hot']
        )

        return model, x, P, y, hw_use, if_use, attach_s, attach_a, bus, comm, traffic_flows, net, feasible_locs_per_sc, feasible_scs_per_loc, var_stats

    def _add_constraints(self, model, x, P, y, hw_use, if_use, attach_s, attach_a, bus, comm, traffic_flows, net, scs, locations, sensors, actuators, 
                              cable_types, comm_matrix, partitions, unique_asils,
                              feasible_locs_per_sc, feasible_scs_per_loc, enable_comm_bw_constraints=True,
                              comm_bw_big_m=None, uncertainty=None):
        """
        Add all constraints to the optimization model
        """
        n_sc = len(scs)
        n_locs = len(locations)
        all_interfaces = list(cable_types.keys())
        all_hw = sorted(list(set(h for (j, h) in hw_use.keys())))
        
        print(f"\nAdding constraints...")

        cstats = {
            'c1a_unique_site': 0,
            'c1b_x_le_y': 0,
            'c1c_partitions_le_scs': 0,
            'c1d_site_caps': 0,
            'c1e_x_le_P': 0,
            'c1g_standby': 0,
            'c2_capacity': 0,
            'c3_hw_required': 0,
            'c4_if_required': 0,
            'c5_redundancy': 0,
            'c6_ai_contention': 0,
            'c9_if_activation': 0,
            'c9_attach': 0,
            'c11_bus': 0,
            'c10_pair_traffic': 0,
            'c10_path_choice': 0,
            'c10_relay_switch': 0,
        }
        
        def x_expr(i, j):
            return x.get((i, j), 0)

        # Uncertainty-aware constraint blocks: each one switched on separately (see configs/uncertainty.json)
        unc = uncertainty or {}
        unc_params = unc.get('params', {})
        print(f"  Uncertainty blocks: " + ", ".join(f"{k}={bool(unc.get(k))}" for k in UNCERTAINTY_BLOCKS))
        if unc.get('growth'):
            print(f"  Growth level alpha = {float(unc.get('growth_alpha', 0.0)):.2f} (profiles {unc.get('params', {}).get('growth', {}).get('profiles')})")

        # SW-based uncertainty (selective static margin, configs/uncertainty.json: sw_margin)
        def get_sw_multiplier(sc):
            return sw_multiplier(sc, unc)

        # Data-derived bounds (replace arbitrary Big-M values)
        # comm_ub: a backbone link never carries more than the total traffic volume, and extra parallel links only
        # add cost; 2 keeps the "switch-only node has degree >= 2" rule satisfiable with one neighbour.
        total_traffic_volume = sum(float(tr['volume']) for tr in traffic_flows)
        comm_ub = {n: max(2, int(np.ceil(total_traffic_volume / cap))) if cap < float('inf') else 2 for n, cap in net['cap_of'].items()}
        for (_j1, _j2, w_if), comm_var in comm.items():
            comm_var.UB = comm_ub[w_if]
        net['comm_ub'] = comm_ub
        print(f"  Backbone cable upper bounds per type: {comm_ub} (total traffic {total_traffic_volume:.1f})")

        # Constraint 1a: every SC is hosted at exactly one site
        for i in range(n_sc):
            model.addConstr(gp.quicksum(x[i, j] for j in feasible_locs_per_sc[i]) == 1, name=f"unique_site_{scs[i].id}")
            cstats['c1a_unique_site'] += 1

        # Constraint 1b: a hosting site is active, and an active site hosts at least one SC (no ghost sites)
        for j in range(n_locs):
            for i in feasible_scs_per_loc[j]:
                model.addConstr(x[i, j] <= y[j], name=f"x_le_y_{scs[i].id}_{locations[j].id}")
                cstats['c1b_x_le_y'] += 1
            model.addConstr(y[j] <= gp.quicksum(x[i, j] for i in feasible_scs_per_loc[j])
                            + gp.quicksum(net['r'][i, j] for i in net['critical'] if (i, j) in net['r']), name=f"y_le_sum_x_{locations[j].id}")
            cstats['c1b_x_le_y'] += 1

        # Constraint 1f: an active site has at least one partition. Implied for integer solutions (y = 1 => an SC is
        # hosted => its ASIL has a partition), but it is what makes the LP relaxation pay for opening a site: without
        # it y[j] = 1 is free at every cell and every peripheral attaches to its nearest cell (facility-location
        # strong linking; 33 cells: bound 7,262 at the root without it).
        for j in range(n_locs):
            model.addConstr(gp.quicksum(P[j, a] for a in unique_asils) >= y[j], name=f"site_has_partition_{locations[j].id}")
            cstats['c1b_x_le_y'] += 1

        # Constraint 1c: partitions of ASIL a at site j never exceed the SCs of ASIL a hosted there
        # (no partition without software; with c2 the solver opens P = ceil(demand / capacity))
        for j in range(n_locs):
            for a in unique_asils:
                model.addConstr(
                    P[j, a] <= gp.quicksum(x[i, j] for i in feasible_scs_per_loc[j] if scs[i].asil_req == a)
                    + gp.quicksum(net['r'][i, j] for i in net['critical'] if (i, j) in net['r'] and scs[i].asil_req == a),
                    name=f"partitions_le_scs_{locations[j].id}_asil{a}"
                )
                cstats['c1c_partitions_le_scs'] += 1

        # Constraint 1e: a hosted SC needs a partition of its ASIL at its site
        for (i, j), xv in x.items():
            model.addConstr(xv <= P[j, scs[i].asil_req], name=f"x_le_P_{scs[i].id}_{locations[j].id}")
            cstats['c1e_x_le_P'] += 1

        # Constraint 1d: per-site caps of the control unit (configs/partitions.json: site.max_partitions and
        # site.cpu_cap / ram_cap / rom_cap = the hosted demand of the whole unit over all its partitions, hot standbys
        # included as in constraint 2, times the site derating; null = unlimited, which reproduces the v1 model)
        site_max_p = partitions.get('site_max_partitions')
        for j in range(n_locs):
            p_sum = gp.quicksum(P[j, a] for a in unique_asils)
            if site_max_p is not None:
                model.addConstr(p_sum <= float(site_max_p) * y[j], name=f"site_max_partitions_{locations[j].id}")
                cstats['c1d_site_caps'] += 1
            H_j = site_derating(locations[j], unc)
            for res in ('cpu', 'ram', 'rom'):
                site_cap = partitions.get(f'site_{res}_cap')
                if site_cap is None:
                    continue
                demand = gp.quicksum(
                    x[i, j] * getattr(scs[i], f'{res}_req') * get_sw_multiplier(scs[i]) for i in feasible_scs_per_loc[j]
                ) + gp.quicksum(
                    net['hr'][i, j] * getattr(scs[i], f'{res}_req') * get_sw_multiplier(scs[i])
                    for i in feasible_scs_per_loc[j] if (i, j) in net['hr']
                )
                model.addConstr(demand <= float(site_cap) * H_j * y[j], name=f"site_{res}_cap_{locations[j].id}")
                cstats['c1d_site_caps'] += 1

        # Constraint 2: resource capacity per (site, ASIL): hosted demand <= partitions * partition capacity * derating.
        # Pooled over the partitions of one ASIL; exactness is checked after the solve (pack_partitions) and
        # enforced with cuts where a group cannot be packed.
        for j in range(n_locs):
            H_j = site_derating(locations[j], unc)
            for a in unique_asils:
                scs_ja = [i for i in feasible_scs_per_loc[j] if scs[i].asil_req == a]
                for res in ('cpu', 'ram', 'rom'):
                    demand = gp.quicksum(
                        x[i, j] * getattr(scs[i], f'{res}_req') * get_sw_multiplier(scs[i]) for i in scs_ja
                    ) + gp.quicksum(
                        net['hr'][i, j] * getattr(scs[i], f'{res}_req') * get_sw_multiplier(scs[i])
                        for i in scs_ja if (i, j) in net['hr']   # hot standbys consume their resources permanently
                    )
                    model.addConstr(
                        demand <= P[j, a] * partitions.get(f'{res}_cap', float('inf')) * H_j,
                        name=f"{res}_capacity_{locations[j].id}_asil{a}"
                    )
                    cstats['c2_capacity'] += 1

        # Constraint 1g (failures): one standby per critical SC on another cell, with a partition of its ASIL and its
        # hardware there; hr = r AND hot for the SCs that may use a cold standby
        for i in net['critical']:
            model.addConstr(gp.quicksum(net['r'][i, j] for j in feasible_locs_per_sc[i]) == 1, name=f"standby_{scs[i].id}")
            for j in feasible_locs_per_sc[i]:
                rv = net['r'][i, j]
                model.addConstr(rv + x[i, j] <= 1, name=f"standby_other_cell_{scs[i].id}_{locations[j].id}")
                model.addConstr(rv <= y[j], name=f"standby_site_{scs[i].id}_{locations[j].id}")
                model.addConstr(rv <= P[j, scs[i].asil_req], name=f"standby_partition_{scs[i].id}_{locations[j].id}")
                for h in (scs[i].hw_required or []):
                    model.addConstr(rv <= hw_use[j, h], name=f"standby_hw_{scs[i].id}_{h}_{locations[j].id}")
                if net['hot'][i] is not None:
                    hv = net['hr'][i, j]
                    model.addConstr(hv <= rv, name=f"hr_le_r_{scs[i].id}_{locations[j].id}")
                    model.addConstr(hv <= net['hot'][i], name=f"hr_le_hot_{scs[i].id}_{locations[j].id}")
                    model.addConstr(hv >= rv + net['hot'][i] - 1, name=f"hr_ge_{scs[i].id}_{locations[j].id}")
                cstats['c1g_standby'] += 1

        # Constraint 3: HW feature required
        for i in range(n_sc):
            for h in (scs[i].hw_required or []):
                for j in feasible_locs_per_sc[i]:
                    model.addConstr(
                        x_expr(i, j) <= hw_use[j, h],
                        name=f"hw_required_{scs[i].id}_{h}_{locations[j].id}"
                    )
                    cstats['c3_hw_required'] += 1
        
        # Constraint 4: Interface Port Counting and Switch Requirement
        # Total ports = Sum of ports required by SCs (sensors/actuators) + Backbone connections
        s_lookup = build_sensor_lookup(sensors)
        a_lookup = build_actuator_lookup(actuators)
        for j in range(n_locs):
            for i_name in all_interfaces:
                comm_port_demand = gp.LinExpr()
                for (j1, j2, iface), comm_var in comm.items():
                    if (j1 == j) and (iface == i_name):
                        comm_port_demand.add(comm_var, 1)
                    elif (j2 == j) and (iface == i_name):
                        comm_port_demand.add(comm_var, 1)

                # 4a. Bus types (CAN, LIN, FlexRay): one port per bus line the unit belongs to, plus one per
                # point-to-point cable of that type
                if i_name in bus['types']:
                    model.addConstr(
                        if_use[j, i_name] >= gp.quicksum(bus['unit'][b, j] for b in bus['inst'] if b[0] == i_name) + comm_port_demand,
                        name=f"bus_ports_{locations[j].id}_{i_name}"
                    )

                else:
                    # 4b. Point-to-point style counting (ETH and other non-shared interfaces)
                    sc_port_demand = gp.LinExpr()

                    for i in feasible_scs_per_loc[j]:
                        sc = scs[i]
                        # For ETH, peripheral port demand is modeled via attachment vars (attach_s/attach_a).
                        # Counting SC->peripheral ETH edges here would double-count the same physical ports.
                        if i_name == 'ETH' and (attach_s is not None or attach_a is not None):
                            num_sensors_if = 0
                            num_actuators_if = 0
                        else:
                            num_sensors_if = sum(
                                1
                                for s_id in (sc.sensors or [])
                                if s_lookup.get(s_id)
                                and s_lookup.get(s_id).interface == i_name
                            )
                            num_actuators_if = sum(
                                1
                                for a_id in (sc.actuators or [])
                                if a_lookup.get(a_id)
                                and a_lookup.get(a_id).interface == i_name
                            )

                        if num_sensors_if + num_actuators_if > 0:
                            sc_port_demand.add(x_expr(i, j), num_sensors_if + num_actuators_if)

                    # ETH peripherals connect to an attachment point (SWITCH or ECU node) via attach vars.
                    if i_name == 'ETH':
                        for (si, jj), av in (attach_s or {}).items():
                            if jj == j:
                                sc_port_demand.add(av, 1.0)
                        for (ai, jj), av in (attach_a or {}).items():
                            if jj == j:
                                sc_port_demand.add(av, 1.0)

                    # Total ports at location j for interface i_name
                    # Special case (ETH): if a SWITCH is installed at this location, it provides a limited
                    # number of physical ETH ports (capacity). Any remaining demand must be covered by
                    # explicit ETH ports (if_use), which can represent additional PHY/NIC ports.
                    if i_name == 'ETH' and 'SWITCH' in all_hw:
                        SWITCH_ETH_PORTS = 16
                        model.addConstr(
                            if_use[j, i_name] + SWITCH_ETH_PORTS * hw_use[j, 'SWITCH'] >= sc_port_demand + comm_port_demand,
                            name=f"port_count_{locations[j].id}_{i_name}_switch_covers"
                        )
                    else:
                        model.addConstr(
                            if_use[j, i_name] >= sc_port_demand + comm_port_demand,
                            name=f"port_count_{locations[j].id}_{i_name}"
                        )

                cstats['c4_if_required'] += 1
        
        if unc.get('redundancy'):
            # Constraint 5: Redundancy Constraints
            sc_id_to_idx = {sc.id: i for i, sc in enumerate(scs)}
            for i, sc in enumerate(scs):
                if hasattr(sc, 'redundant_with') and sc.redundant_with:
                    partner_id = sc.redundant_with
                    if partner_id in sc_id_to_idx:
                        partner_idx = sc_id_to_idx[partner_id]
                        if i < partner_idx:
                            common_js = set(feasible_locs_per_sc[i]).intersection(set(feasible_locs_per_sc[partner_idx]))
                            for j in common_js:
                                model.addConstr(
                                    x_expr(i, j)
                                    + x_expr(partner_idx, j)
                                    <= 1,
                                    name=f"redundancy_{sc.id}_{partner_id}_{locations[j].id}"
                                )
                                cstats['c5_redundancy'] += 1

        if unc.get('ai_contention'):
            # Constraint 6: AI accelerator contention limit per location
            ai_cfg = unc_params.get('ai_contention', {})
            K_THRESHOLD = int(ai_cfg['max_per_location'])
            ai_sc_indices = [
                i for i, sc in enumerate(scs)
                if ai_cfg['ai_hw_feature'] in (getattr(sc, 'hw_required', None) or [])
            ]
            ai_sc_set = set(ai_sc_indices)
            if ai_sc_indices:
                for j in range(n_locs):
                    ai_sum_at_loc = gp.quicksum(
                        x_expr(i, j)
                        for i in feasible_scs_per_loc[j]
                        if i in ai_sc_set
                    )
                    model.addConstr(
                        ai_sum_at_loc <= K_THRESHOLD,
                        name=f"ai_contention_limit_loc_{locations[j].id}"
                    )
                    cstats['c6_ai_contention'] += 1
        
        # Constraint 9a: ECU-to-ECU Communication - Endpoint Activation (Big-M)
        # A backbone link can only touch an "active" node. This prevents the optimizer from
        # opening comm links to completely unused locations.
        #
        # Here, "active" means:
        # - at least one open partition (y=1) OR
        # - a dedicated SWITCH placed at that location (switch-only network node)
        #
        # Note: This is grouped near interface-activation constraints because all of these
        # constraints directly gate the same decision variable family: comm[j1,j2,*].
        all_interfaces = list(cable_types.keys())

        # Build an "active node" expression per location.
        if 'SWITCH' in all_hw:
            loc_active = {j: y[j] + hw_use[j, 'SWITCH'] for j in range(n_locs)}
        else:
            loc_active = {j: 1.0 * y[j] for j in range(n_locs)}

        # Constraint 9-ATT: ETH sensor/actuator attachments must connect to an active node,
        # and must respect peripheral max-latency on the physical cable to that attachment point.
        # (We do not model end-to-end network latency here; this only constrains the local cable.)
        eth_cap_obj = cable_types.get('ETH') if cable_types else None
        eth_cap = getattr(eth_cap_obj, 'capacity', None) if eth_cap_obj is not None else None
        try:
            eth_cap = float(eth_cap) if eth_cap is not None else None
        except Exception:
            eth_cap = None
        if attach_s:
            for (si, j), av in attach_s.items():
                model.addConstr(av <= loc_active[j], name=f"attachS_active_{sensors[si].id}_{locations[j].id}")
                cstats['c9_attach'] += 1

                # ETH peripheral link bandwidth (sensor -> attachment point)
                vol = float(getattr(sensors[si], 'volume', 0.0) or 0.0)
                if eth_cap is not None and eth_cap > 0 and eth_cap != float('inf') and vol > 0:
                    model.addConstr(
                        vol * av <= eth_cap,
                        name=f"attachS_bw_{sensors[si].id}_{locations[j].id}"
                    )
                    cstats['c9_attach'] += 1


            for si in {k[0] for k in attach_s.keys()}:
                model.addConstr(
                    gp.quicksum(attach_s[si, j] for j in range(n_locs) if (si, j) in attach_s) == 1,
                    name=f"attachS_one_{sensors[si].id}"
                )
                cstats['c9_attach'] += 1

        if attach_a:
            for (ai, j), av in attach_a.items():
                model.addConstr(av <= loc_active[j], name=f"attachA_active_{actuators[ai].id}_{locations[j].id}")
                cstats['c9_attach'] += 1

                # ETH peripheral link bandwidth (attachment point -> actuator)
                vol = float(getattr(actuators[ai], 'volume', 0.0) or 0.0)
                if eth_cap is not None and eth_cap > 0 and eth_cap != float('inf') and vol > 0:
                    model.addConstr(
                        vol * av <= eth_cap,
                        name=f"attachA_bw_{actuators[ai].id}_{locations[j].id}"
                    )
                    cstats['c9_attach'] += 1


            for ai in {k[0] for k in attach_a.keys()}:
                model.addConstr(
                    gp.quicksum(attach_a[ai, j] for j in range(n_locs) if (ai, j) in attach_a) == 1,
                    name=f"attachA_one_{actuators[ai].id}"
                )
                cstats['c9_attach'] += 1

        # Constraint 11 (bus lines, CAN / LIN / FlexRay): every bus device with software sits on exactly one line of its
        # type, together with the unit of every component driving it and, with failures on, the unit of that
        # component's standby (the device stays reachable when the primary's cell fails); one ASIL group per line;
        # the devices of a line share its bit rate; members are units that exist; the cable length of a line is at
        # least the Steiner tree of its members (pairs here, the exact tree as lazy cuts in optimize()).
        for d, info in bus['devices'].items():
            lines = [b for b in bus['inst'] if b[0] == info['iface']]
            total = gp.quicksum(bus['dev'][b, d] for b in lines)
            model.addConstr(total == 1 if info['users'] else total <= 1, name=f"bus_device_{d}")
            cstats['c11_bus'] += 1
            for b in lines:
                dv = bus['dev'][b, d]
                model.addConstr(dv <= bus['act'][b], name=f"bus_dev_act_{d}_{b[0]}{b[1] + 1}")
                if info['group'] == 'HIGH':
                    model.addConstr(dv <= bus['high'][b], name=f"bus_dev_high_{d}_{b[0]}{b[1] + 1}")
                else:
                    model.addConstr(dv <= 1 - bus['high'][b], name=f"bus_dev_low_{d}_{b[0]}{b[1] + 1}")
                cstats['c11_bus'] += 2
                for i in info['users']:
                    for j in feasible_locs_per_sc[i]:
                        model.addConstr(bus['unit'][b, j] >= dv + x[i, j] - 1, name=f"bus_driver_{d}_{scs[i].id}_{b[0]}{b[1] + 1}_{locations[j].id}")
                        cstats['c11_bus'] += 1
                        if (i, j) in net['r']:
                            model.addConstr(bus['unit'][b, j] >= dv + net['r'][i, j] - 1, name=f"bus_standby_{d}_{scs[i].id}_{b[0]}{b[1] + 1}_{locations[j].id}")
                            cstats['c11_bus'] += 1
        for b in bus['inst']:
            name = f"{b[0]}{b[1] + 1}"
            devs = [d for d in bus['devices'] if (b, d) in bus['dev']]
            for j in range(n_locs):
                model.addConstr(bus['unit'][b, j] <= y[j], name=f"bus_unit_exists_{name}_{locations[j].id}")
                model.addConstr(bus['unit'][b, j] <= bus['act'][b], name=f"bus_unit_act_{name}_{locations[j].id}")
                cstats['c11_bus'] += 2
            line_sel = [sv for (u, v, w_if, p), sv in net['sel'].items() if w_if == b[0] and p == b[1]]
            model.addConstr(bus['act'][b] <= gp.quicksum(bus['dev'][b, d] for d in devs) + gp.quicksum(line_sel), name=f"bus_has_use_{name}")
            cap = float(getattr(cable_types.get(b[0]), 'capacity', float('inf'))) * net['bus_umax']
            if cap < float('inf'):
                model.addConstr(gp.quicksum(bus['devices'][d]['volume'] * bus['dev'][b, d] for d in devs) <= cap, name=f"bus_capacity_{name}")
                cstats['c11_bus'] += 1
            members = [(locations[j].id, bus['unit'][b, j]) for j in range(n_locs)] + [(d, bus['dev'][b, d]) for d in devs]
            # length >= half the sum, over the members, of the distance to the nearest other possible member (an Euler
            # tour of the line passes every member and is twice its length)
            nn = {e: min((get_distance(bus['points'][e], bus['points'][f]) for f, _v in members if f != e), default=0.0) for e, _v in members}
            model.addConstr(bus['len'][b] >= 0.5 * gp.quicksum(nn[e] * v for e, v in members), name=f"bus_len_tour_{name}")
            cstats['c11_bus'] += 1
            for k1 in range(len(members)):
                for k2 in range(k1 + 1, len(members)):
                    (ida, va), (idb, vb) = members[k1], members[k2]
                    dist = get_distance(bus['points'][ida], bus['points'][idb])
                    if dist > 0:
                        model.addConstr(bus['len'][b] >= dist * (va + vb - 1), name=f"bus_len_pair_{name}_{k1}_{k2}")
                        cstats['c11_bus'] += 1
            nxt = (b[0], b[1] + 1)
            if nxt in bus['act']:
                model.addConstr(bus['act'][b] >= bus['act'][nxt], name=f"bus_order_{name}")   # interchangeable lines: open in order
                cstats['c11_bus'] += 1
            # the lines of one type are interchangeable: the k-th device of the type may only use lines 1..k
            allowed = []
            for k_dev, d in enumerate(sorted(devs)):
                if b[1] > k_dev:
                    model.addConstr(bus['dev'][b, d] == 0, name=f"bus_symmetry_{name}_{d}")
                    cstats['c11_bus'] += 1
                else:
                    allowed.append(d)
            # exact trees of the small member sets up front (device groups up to 4 alone, with one unit and with two
            # units; up to 12 candidate cells also unit triples): valid because the Steiner tree only grows with its
            # members; larger sets come as lazy cuts
            def tree_bound(members, label):
                st_len = steiner_tree([e for e, _v in members], bus['points'])[0]
                if st_len > 0:
                    model.addConstr(bus['len'][b] >= st_len * (gp.quicksum(v for _e, v in members) - (len(members) - 1)), name=f"bus_tree_{name}_{label}")
                    cstats['c11_bus'] += 1
            units = [(locations[j].id, bus['unit'][b, j]) for j in range(n_locs)]
            for size in range(1, min(4, len(allowed)) + 1):
                for group in itertools.combinations(allowed, size):
                    gm = [(d, bus['dev'][b, d]) for d in group]
                    if size >= 2:
                        tree_bound(gm, '_'.join(group))
                    for u1 in range(n_locs):
                        tree_bound(gm + [units[u1]], f"{'_'.join(group)}_{units[u1][0]}")
                        for u2 in range(u1 + 1, n_locs):
                            tree_bound(gm + [units[u1], units[u2]], f"{'_'.join(group)}_{units[u1][0]}_{units[u2][0]}")
            if n_locs <= 12:
                for trio in itertools.combinations(range(n_locs), 3):
                    tree_bound([units[k] for k in trio], '_'.join(units[k][0] for k in trio))

        for (j1, j2, iface), comm_var in comm.items():
            M = comm_ub[iface]  # a cable count never exceeds its data-derived upper bound
            model.addConstr(
                comm_var <= M * loc_active[j1],
                name=f"comm_active_j1_{locations[j1].id}_{locations[j2].id}_{iface}"
            )
            cstats['c9_if_activation'] += 1
            model.addConstr(
                comm_var <= M * loc_active[j2],
                name=f"comm_active_j2_{locations[j1].id}_{locations[j2].id}_{iface}"
            )
            cstats['c9_if_activation'] += 1

            # Constraint 9b: ECU-to-ECU Communication - Interface Activation (Big-M)
            # If comm[j1,j2,iface] > 0, then the endpoint must provide that interface.
            # Interface activation: for ETH, allow either explicit ETH ports OR a SWITCH at the endpoint.
            if iface == 'ETH' and 'SWITCH' in all_hw:
                model.addConstr(
                    comm_var <= M * (if_use[j1, iface] + hw_use[j1, 'SWITCH']),
                    name=f"if_act_j1_{locations[j1].id}_{locations[j2].id}_{iface}_or_switch"
                )
                cstats['c9_if_activation'] += 1
                model.addConstr(
                    comm_var <= M * (if_use[j2, iface] + hw_use[j2, 'SWITCH']),
                    name=f"if_act_j2_{locations[j1].id}_{locations[j2].id}_{iface}_or_switch"
                )
                cstats['c9_if_activation'] += 1
            else:
                model.addConstr(
                    comm_var <= M * if_use[j1, iface],
                    name=f"if_act_j1_{locations[j1].id}_{locations[j2].id}_{iface}"
                )
                cstats['c9_if_activation'] += 1
                model.addConstr(
                    comm_var <= M * if_use[j2, iface],
                    name=f"if_act_j2_{locations[j1].id}_{locations[j2].id}_{iface}"
                )
                cstats['c9_if_activation'] += 1

        # Constraint 9c: Switch-only node must have meaningful connectivity
        # If a location has a SWITCH but no open partitions, it should act as a real
        # network junction (not a dangling node). We approximate node degree by the
        # total number of incident comm links.
        #
        # Enforced logic (without introducing extra binaries):
        #   if hw_use[j,'SWITCH']=1 and sum_p,a y[j,a,p]=0  => incident_comm(j) >= 2
        #   otherwise (if any partition is open), the constraint becomes non-binding.
        if 'SWITCH' in all_hw:
            for j in range(n_locs):
                part_sum_j = y[j]
                incident_comm_j = gp.LinExpr()
                for (j1, j2, _iface), comm_var in comm.items():
                    if j1 == j or j2 == j:
                        incident_comm_j.add(comm_var, 1.0)

                # incident_comm_j >= 2 when switch-only; relaxed when part_sum_j >= 1
                model.addConstr(
                    incident_comm_j >= 2 * hw_use[j, 'SWITCH'] - 2 * part_sum_j,
                    name=f"switch_only_degree_ge2_{locations[j].id}"
                )
        

        
        # C-NET: aggregated backbone traffic on candidate paths (replaces the per-flow routing binaries of v1). The
        # traffic itself (a pair with traffic needs a selected path, link load within cables x capacity) is enforced
        # by the lazy constraints of the callback in optimize().
        paths, sel = net['paths'], net['sel']
        has_switch = 'SWITCH' in all_hw

        # C-NET-0: every flow with a choice uses exactly one backbone cable type
        for t, elig in net['flow_elig'].items():
            if len(elig) > 1:
                model.addConstr(gp.quicksum(net['w'][t, n] for n in elig) == 1, name=f"flow_iface_{t}")
                cstats['c10_pair_traffic'] += 1

        # C-NET-2: per site pair and cable type at most one candidate path (both directions)
        for (u, v, w_if), plist in paths.items():
            model.addConstr(gp.quicksum(sel[u, v, w_if, p] for p in range(len(plist))) <= 1, name=f"one_path_{locations[u].id}_{locations[v].id}_{w_if}")
            cstats['c10_path_choice'] += 1
            # Optional (configs/buses.json: backbone.path_link_cut): a selected path has a cable on each of its links.
            # Implied for integer solutions; in the LP it makes a fractional placement pay a proportional share of a
            # whole cable instead of load / capacity.
            if w_if in bus['types']:
                # a bus line carries a unit-to-unit flow only when both units are members of it, and it is then at least
                # as long as the cable between them (the bus counterpart of the path-link cut: keeps the LP honest)
                d_uv = get_distance(locations[u].location, locations[v].location)
                for p in range(len(plist)):
                    for j in (u, v):
                        model.addConstr(sel[u, v, w_if, p] <= bus['unit'][(w_if, p), j], name=f"line_member_{locations[u].id}_{locations[v].id}_{w_if}{p + 1}_{locations[j].id}")
                        cstats['c10_path_choice'] += 1
                    model.addConstr(bus['len'][(w_if, p)] >= d_uv * sel[u, v, w_if, p], name=f"line_length_{locations[u].id}_{locations[v].id}_{w_if}{p + 1}")
                    cstats['c10_path_choice'] += 1
            elif net.get('path_link_cut'):
                for p, path in enumerate(plist):
                    for a, b in zip(path, path[1:]):
                        key = (a, b, w_if) if a < b else (b, a, w_if)
                        model.addConstr(comm[key] >= sel[u, v, w_if, p], name=f"path_cable_{locations[u].id}_{locations[v].id}_{w_if}_p{p}_{locations[a].id}_{locations[b].id}")
                        cstats['c10_path_choice'] += 1
            # C-NET-3: a relaying site needs a SWITCH
            for p, path in enumerate(plist):
                for r in path[1:-1]:
                    if has_switch:
                        model.addConstr(sel[u, v, w_if, p] <= hw_use[r, 'SWITCH'], name=f"relay_switch_{locations[u].id}_{locations[v].id}_p{p}_{locations[r].id}")
                    else:
                        model.addConstr(sel[u, v, w_if, p] == 0, name=f"relay_impossible_{locations[u].id}_{locations[v].id}_p{p}")
                    cstats['c10_relay_switch'] += 1

        cstats['total'] = sum(v for k, v in cstats.items() if k != 'total')
        return model, cstats

    def _build_objective_function(self, model, x, P, y, hw_use, if_use, comm, scs, locations, sensors, actuators, 
                                  cable_types, comm_matrix, partitions, hardwares, interfaces,
                                  n_locs, unique_asils, all_hw, all_interfaces,
                                  feasible_locs_per_sc, attach_s=None, attach_a=None, bus=None,
                                  minimize_cable_length=False, cable_length_limit=None, cost_limit=None):
        """
        Build and set the objective function for the optimization model
        
        Objectives:
            - Default: minimize total cost
            - If minimize_cable_length=True: minimize total cable length
        """
        print(f"Building objective function...")
        
        # Partition cost
        partition_cost_expr = gp.quicksum(partitions.get('cost', 0) * P[j, a] for j in range(n_locs) for a in unique_asils)

        # Fixed activation cost of a site (configs/partitions.json: site.fixed_cost; 0 = free, as in v1)
        site_fixed_cost = float(partitions.get('site_fixed_cost', 0) or 0)
        site_cost_expr = gp.quicksum(site_fixed_cost * y[j] for j in range(n_locs))
        
        # HW cost
        hw_cost_expr = gp.quicksum(
            hardwares.get(h, 0) * hw_use[j, h]
            for j in range(n_locs)
            for h in all_hw
        )
        
        # Interface port cost
        if_cost_expr = gp.quicksum(
            interfaces[i_name].port_cost * if_use[j, i_name]
            for j in range(n_locs)
            for i_name in all_interfaces
        )
        
        # Cable cost (sensor/actuator cables + ECU-to-ECU communication backbone)
        cable_cost_expr, cable_distance_expr, _ = self.calculate_cable_expressions(
            x, scs, locations, sensors, actuators, cable_types, comm_matrix,
            feasible_locs_per_sc, comm=comm, attach_s=attach_s, attach_a=attach_a, bus=bus
        )

        total_cost = partition_cost_expr + site_cost_expr + hw_cost_expr + if_cost_expr + cable_cost_expr

        # Optional epsilon-constraint on total cable length for Pareto search
        if cable_length_limit is not None:
            model.addConstr(
                cable_distance_expr <= float(cable_length_limit),
                name="eps_cable_length_limit"
            )

        # Optional epsilon-constraint on total cost for reverse Pareto sweep
        if cost_limit is not None:
            model.addConstr(
                total_cost <= float(cost_limit),
                name="eps_cost_limit"
            )
        
        # Lexicographic objectives (solved in two phases in optimize()):
        # the primary objective is minimized first, then the secondary one with the primary fixed at its optimum.
        cost_obj = ('total_cost', gp.LinExpr() + total_cost)
        cable_obj = ('cable_length', gp.LinExpr() + cable_distance_expr)
        objectives = [cable_obj, cost_obj] if minimize_cable_length else [cost_obj, cable_obj]
        print(f"Objective: {objectives[0][0]}, then {objectives[1][0]} (lexicographic)")

        return model, objectives

    def optimize(self, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=None, hardwares=None, interfaces=None, enable_comm_bw_constraints=True, comm_bw_big_m=10000, minimize_cable_length=False, cable_length_limit=None, cost_limit=None, time_limit=None, mip_gap=None, lp_path=None, uncertainty=None, backbone=None, phase2_time=600, gurobi_params=None, chains=None, start=None):
        """
        Dynamic selection of partitions + HW + interfaces + cables.
        
        Args:
            scs: List of SoftwareComponent objects
            locations: List of Location objects (CandidateECU/Point with id, location attributes)
            sensors: List of Sensor objects
            actuators: List of Actuator objects
            cable_types: Dict of Interface objects by name
            comm_matrix: List of communication links
            partitions: Dict with partition config (cost, cpu_cap, etc.) from config_reader
            hardwares: Dict with HW feature costs (e.g., {'SSE_CPU': 200, 'DSP': 300})
            interfaces: Dict of Interface objects (from config_reader)
        
        Returns:
            List with single solution dict
        """
        print("=" * 80)
        print("Gurobi OPTIMIZATION")
        print("=" * 80)
        
        # ======================= SETUP =======================
        n_sc = len(scs)
        n_locs = len(locations)
        
        # Collect all unique ASIL levels and HW features
        unique_asils = sorted(list(set(sc.asil_req for sc in scs)))
        all_hw = list(hardwares.keys()) if hardwares else []
        all_interfaces = list(interfaces.keys()) if interfaces else list(cable_types.keys())
        
        print(f"\nProblem size:")
        print(f"  SCs: {n_sc}, Locations: {n_locs}")
        print(f"  Unique ASIL levels: {unique_asils}")
        print(f"  HW features available: {all_hw}")
        print(f"  Interfaces available: {all_interfaces}")
        print(f"  Partition config: cost={partitions.get('cost', 0)}, cpu_cap={partitions.get('cpu_cap', 0)}, ram_cap={partitions.get('ram_cap', 0)}, rom_cap={partitions.get('rom_cap', 0)}")
        print(f"  Site config: max partitions per ASIL={partitions.get('max_per_asil_per_site', 3)}, site max partitions={partitions.get('site_max_partitions')}, "
              f"site caps cpu/ram/rom={partitions.get('site_cpu_cap')}/{partitions.get('site_ram_cap')}/{partitions.get('site_rom_cap')}, fixed cost={partitions.get('site_fixed_cost', 0)}")
        
        # ======================= PRECOMPUTE FEASIBILITY =======================
        infeasible_ij = set()   # placement restrictions from timing are enforced through the chain constraints

        # ======================= CREATE MODEL & VARIABLES =======================
        model, x, P, y, hw_use, if_use, attach_s, attach_a, bus, comm, traffic_flows, net, feasible_locs_per_sc, feasible_scs_per_loc, var_stats = self._create_model_and_variables(
            n_sc, n_locs, scs, locations, unique_asils, all_hw, sensors, actuators, all_interfaces, infeasible_ij=infeasible_ij, comm_matrix=comm_matrix,
            partitions=partitions, backbone=backbone, cable_types=cable_types, uncertainty=uncertainty, chains=chains
        )
        
        # ======================= ADD CONSTRAINTS =======================
        model, cstats = self._add_constraints(
            model, x, P, y, hw_use, if_use, attach_s, attach_a, bus, comm, traffic_flows, net, scs, locations, sensors, actuators,
            cable_types, comm_matrix, partitions, unique_asils,
            feasible_locs_per_sc, feasible_scs_per_loc,
            enable_comm_bw_constraints=enable_comm_bw_constraints,
            comm_bw_big_m=comm_bw_big_m,
            uncertainty=uncertainty
        )

        print("\n[DEBUG] Variable breakdown:")
        print(f"  total vars: {var_stats['total']}")
        print(f"  x: {var_stats['x']}")
        print(f"  P: {var_stats['P']}")
        print(f"  y: {var_stats['y']}")
        print(f"  hw_use: {var_stats['hw_use']}")
        print(f"  if_use: {var_stats['if_use']}")
        print(f"  attach_s: {var_stats.get('attach_s', 0)}")
        print(f"  attach_a: {var_stats.get('attach_a', 0)}")
        print(f"  bus lines: {var_stats.get('bus', 0)}")
        print(f"  comm: {var_stats['comm']} ({(100.0 * var_stats['comm'] / max(1, var_stats['total'])):.1f}%)")
        print(f"  traffic_flows: {var_stats['traffic_flows']}")
        print(f"  sel: {var_stats['sel']}  (site pairs with candidate paths: {var_stats['pairs_with_paths']})")

        print("\n[DEBUG] Constraint breakdown:")
        for k, v in cstats.items():
            print(f"  {k}: {v}")

        # ======================= BUILD OBJECTIVE =======================
        model, objectives = self._build_objective_function(
            model, x, P, y, hw_use, if_use, comm, scs, locations, sensors, actuators,
            cable_types, comm_matrix, partitions, hardwares, interfaces,
            n_locs, unique_asils, all_hw, all_interfaces,
            feasible_locs_per_sc, attach_s=attach_s, attach_a=attach_a, bus=bus,
            minimize_cable_length=minimize_cable_length, cable_length_limit=cable_length_limit,
            cost_limit=cost_limit
        )

        model.update()
        print("\n[DEBUG] Gurobi model size (final):")
        print(f"  NumVars: {model.NumVars}")
        print(f"  NumConstrs: {model.NumConstrs}")

        # Start solution: the values of the previous solve of a sweep (growth level, Pareto point), matched by variable
        # name. Only a starting incumbent for the branch-and-bound: Gurobi checks it (through the lazy callback) and
        # drops it if it violates a constraint of this model; the proven optimum is not affected.
        if start:
            matched = 0
            for v in model.getVars():
                val = start.get(v.VarName)
                if val is not None:
                    v.Start = val
                    matched += 1
            print(f"  Start solution: {matched} of {model.NumVars} variables taken from the previous solve")

        if time_limit is not None:
            try:
                tl = float(time_limit)
                if tl > 0:
                    model.setParam('TimeLimit', tl)
                    print(f"  Time limit per solve: {tl:.1f} s")
            except Exception:
                print(f"  [WARN] Ignoring invalid time_limit value: {time_limit}")
        if mip_gap is not None:
            model.setParam('MIPGap', float(mip_gap))
            print(f"  MIPGap: {float(mip_gap)}")
        # Extra Gurobi parameters for experiments, e.g. "MIPFocus=3,Cuts=2" (--gurobi_params)
        for item in (gurobi_params or '').split(','):
            if '=' in item:
                k, v = item.split('=', 1)
                model.setParam(k.strip(), float(v) if '.' in v else int(v))
                print(f"  Gurobi parameter {k.strip()} = {v.strip()}")
        # ======================= SOLVE (lexicographic, two phases) =======================
        print(f"\nSolving...")
        (primary_name, primary_expr), (secondary_name, secondary_expr) = objectives
        if lp_path:
            model.setObjective(primary_expr, GRB.MINIMIZE)
            model.write(lp_path)
            print(f"  Model written to: {lp_path}")

        tl = float(time_limit) if (time_limit is not None and float(time_limit) > 0) else None

        # Lazy traffic constraints: every integer solution is checked (also every start solution and repair);
        # a site pair with traffic but no selected path, or a link carrying more than its cables allow, adds a cut.
        # Chain latency (configs/chains.json): per chain the flows between its consecutive components, including the
        # sensor-to-driver and driver-to-actuator links of Ethernet peripherals; budget = network time allowed.
        unc_lat = (uncertainty or {}).get('params', {}).get('robust_latency', {})
        net['lat'] = {'robust': bool((uncertainty or {}).get('robust_latency')),
                      'factor': float(unc_lat.get('network_safety_factor', 1.0)),
                      'sync': float(unc_lat.get('sync_ms', 0.0)), 'platform': float(unc_lat.get('platform_ms', 0.0))}
        sc_index = {sc.id: i for i, sc in enumerate(scs)}
        base_flows = [tr for tr in traffic_flows if 'base' not in tr]
        flow_of_pair = {(tr['src_idx'], tr['dst_idx']): tr['id'] for tr in base_flows if tr['type'] == 'SC_SC'}
        flow_into = {tr['dst_idx']: tr['id'] for tr in base_flows if tr['type'] == 'SENS_SC'}
        flow_from = {tr['src_idx']: tr['id'] for tr in base_flows if tr['type'] == 'SC_ACT'}
        # a device on a bus line reaches the unit of its software in one bus crossing (fixed per device: kbit / bit
        # rate + per-hop bound, robust like any crossing); a chain adds it at its first (sensor) and last (actuator) end
        dev_end = {}       # (component, sensor/actuator) -> the bus device it drives with the longest own crossing
        for d, info in net['bus']['devices'].items():
            t_if = cable_types[info['iface']]
            ms = robust_ms(net, info['kbit'] / float(t_if.capacity) + float(getattr(t_if, 'hop_latency_ms', 0.0)))
            for i in info['users']:
                if ms > dev_end.get((i, info['kind']), (None, -1.0))[1]:
                    dev_end[i, info['kind']] = (d, ms)
        net['chains'] = []
        for ch in (chains or []):
            idx = [sc_index[s] for s in ch['swcs'] if s in sc_index]
            flows = [flow_into[idx[0]]] if idx and idx[0] in flow_into else []
            flows += [flow_of_pair[a, b] for a, b in zip(idx, idx[1:]) if (a, b) in flow_of_pair]
            flows += [flow_from[idx[-1]]] if idx and idx[-1] in flow_from else []
            ends = [dev_end[k] for k in ((idx[0], 'sensor'), (idx[-1], 'actuator')) if idx and k in dev_end]
            const = sum(ms for _d, ms in ends)   # own crossings of the end devices (their queues come in the cuts)
            net['chains'].append({'id': ch['id'], 'budget': float(ch['net_budget_ms']), 'const': const, 'dev_ends': [d for d, _ms in ends],
                                  'degraded': float(ch.get('degraded_bound_ms', 2.0 * float(ch['net_budget_ms']))), 'flows': flows})
            if const > float(ch['net_budget_ms']):
                print(f"  WARNING: chain {ch['id']} cannot meet its budget: its bus crossings alone take {const:.2f} ms")
        if net['chains']:
            print(f"  Chains: {len(net['chains'])} (network budgets {min(c['budget'] for c in net['chains']):.1f} - "
                  f"{max(c['budget'] for c in net['chains']):.1f} ms, robust latency {net['lat']['robust']})")

        # Priorities (configs/buses.json: backbone.priorities): fixed-priority, non-preemptive queues, as CAN arbitrates
        # by identifier and Ethernet switches serve IEEE 802.1Q traffic classes. A chain message gets the tightest
        # network budget of its chains (deadline-monotonic), a standby copy that of its base flow, the end device of a
        # chain that of the chain; all other traffic comes last. A message waits for every message of equal or higher
        # priority once and for at most one frame of a lower-priority one (cable_types.frame_kbit).
        net['frame_kbit'] = {n: float(getattr((cable_types or {}).get(n), 'frame_kbit', float('inf'))) for n in net['bb_ifaces']}
        if (backbone or {}).get('priorities'):
            pf, pd = {}, {}
            for ch in net['chains']:
                for t in ch['flows']:
                    pf[t] = min(pf.get(t, float('inf')), ch['budget'])
                for d in ch['dev_ends']:
                    pd[d] = min(pd.get(d, float('inf')), ch['budget'])
            for t, tr in net['flow_meta'].items():
                if tr.get('base') in pf:
                    pf[t] = pf[tr['base']]
            net['prio'] = {'f': pf, 'd': pd}
            print(f"  Priorities: {len(set(pf.values()) | set(pd.values()))} chain classes for {len(pf)} flows and "
                  f"{len(pd)} devices, all other traffic last")

        # Chain latency bound in the model itself (valid; the exact rule with the queues stays in the callback): a
        # chain flow whose two ends are on different sites crosses the network at least once, at no less than its own
        # transmission and per-hop bound on its fastest eligible type (robust constants included). Without it the
        # solver sees latency only through rejected candidates. cross[t] >= src_j - dst_j bounds [ends on different
        # sites] from below.
        # After the failure of cell c, for chains whose flows all join two critical SCs (a design that survives never
        # loses such a chain): a flow a -> b with the primary of a at c runs from the standby of a to an instance of b
        # and crosses for sure when the standby of a is on neither instance's site:
        # z >= x[a,c] + 2 r[a,j] - x[b,j] - r[b,j] - 2, and the same with a and b exchanged.
        def min_crossing(t):
            return min(robust_ms(net, net['kbit'][t] / net['cap_of'][n] + net['hop_ms'][n]) for n in net['flow_elig'][t])

        cross, cross_c, n_nom, n_fail = {}, {}, 0, 0
        for ch in net['chains']:
            terms = []
            for t in ch['flows']:
                if t not in cross:
                    src, dst = net['flow_ends'][t]
                    cross[t] = model.addVar(lb=0.0, ub=1.0, name=f"cross_{t}")
                    for j, var in src.items():
                        model.addConstr(cross[t] >= var - (dst[j] if j in dst else 0), name=f"cross_{t}_{j}")
                terms.append(min_crossing(t) * cross[t])
            model.addConstr(gp.quicksum(terms) <= ch['budget'] - ch['const'], name=f"chain_lb_{ch['id']}")
            n_nom += 1
        crit = set(net['critical'])
        for ch in net['chains']:
            meta = [net['flow_meta'][t] for t in ch['flows']]
            if not meta or any(tr['type'] != 'SC_SC' or tr['src_idx'] not in crit or tr['dst_idx'] not in crit for tr in meta):
                continue
            for c in range(n_locs):
                terms = []
                for t, tr in zip(ch['flows'], meta):
                    a, b = tr['src_idx'], tr['dst_idx']
                    if (a, c) not in x and (b, c) not in x:
                        continue
                    if (t, c) not in cross_c:
                        z = cross_c[t, c] = model.addVar(lb=0.0, ub=1.0, name=f"cross_{t}_fail_{locations[c].id}")
                        for p, q in ((a, b), (b, a)):
                            if (p, c) not in x:
                                continue
                            for j in feasible_locs_per_sc[p]:
                                if (p, j) in net['r']:
                                    model.addConstr(z >= x[p, c] + 2 * net['r'][p, j] - x.get((q, j), 0) - net['r'].get((q, j), 0) - 2,
                                                    name=f"cross_{t}_fail_{locations[c].id}_{scs[p].id}_{locations[j].id}")
                    terms.append(min_crossing(t) * cross_c[t, c])
                if terms:
                    model.addConstr(gp.quicksum(terms) <= ch['degraded'] - ch['const'], name=f"chain_lb_{ch['id']}_fail_{locations[c].id}")
                    n_fail += 1
        if net['chains']:
            print(f"  Chain latency bounds in the model: {n_nom} nominal, {n_fail} after a cell failure")

        model.setParam('LazyConstraints', 1)
        watched = []
        for t, (src, dst) in net['flow_ends'].items():
            watched += list(src.values()) + list(dst.values())
        watched += list(net['sel'].values()) + list(net['w'].values())
        watched += list(x.values()) + list(y.values()) + list(hw_use.values()) + list(P.values())
        watched += list(net['r'].values()) + [v for v in net['hot'].values() if v is not None] + list(net['hr'].values())
        watched += list(attach_s.values()) + list(attach_a.values())
        watched += list(bus['act'].values()) + list(bus['len'].values()) + list(bus['unit'].values()) + list(bus['dev'].values())
        seen_ids, cb_vars = set(), []
        for var in watched:
            if id(var) not in seen_ids:
                seen_ids.add(id(var)); cb_vars.append(var)
        comm_keys = list(comm.keys())
        comm_vars = [comm[k] for k in comm_keys]
        cap_of = net['cap_of']
        model._lazy_cuts = 0
        model._fail_cuts = 0
        # failure scenarios: the variants of every base flow and the base flows of every critical SC
        base_variants, flows_of = {}, {i: [] for i in net['critical']}
        for tr in traffic_flows:
            si = tr.get('src_inst', 'a' if tr['type'] == 'SENS_SC' else 'x')
            di = tr.get('dst_inst', 'a' if tr['type'] == 'SC_ACT' else 'x')
            base_variants.setdefault(tr.get('base', tr['id']), {})[(si, di)] = tr['id']
            if 'base' not in tr:
                for role, key in (('src', 'src_idx'), ('dst', 'dst_idx')):
                    if tr.get(key) in flows_of:
                        flows_of[tr[key]].append((tr['id'], role))
        net['base_variants'], net['flows_of'] = base_variants, flows_of

        model._repair = None
        model._repairs_used = 0
        # callback statistics (printed every 60 s and stored in solve_info): integer candidates from Gurobi, how many
        # were rejected by a lazy check, repair attempts, valid repairs, and the stage where a repair failed
        model._cb_stats = {'candidates': 0, 'rejected': 0, 'repair_tries': 0, 'repair_valid': 0, 'repair_fail': {},
                           'rejected_by': {}, 'cuts_by': {}}
        model._cb_last_print = 0.0
        has_switch_hw = 'SWITCH' in all_hw
        # Steiner cuts of the bus lines found in the callback ({(line, member names): (member variables, length)});
        # Gurobi drops callback cuts after each optimize, but these are valid for every solution, so they are kept as
        # model constraints between the phases (a start solution's bus lengths are recomputed by Gurobi's LP and
        # would otherwise fall below the trees and be rejected)
        model._st_cuts, model._st_added = {}, set()

        def keep_steiner_cuts():
            for key, (members, st_len) in model._st_cuts.items():
                if key not in model._st_added:
                    b = key[0]
                    model.addConstr(bus['len'][b] >= st_len * (gp.quicksum(members) - (len(members) - 1)),
                                    name=f"steiner_{b[0]}{b[1] + 1}_{len(model._st_added)}")
                    model._st_added.add(key)

        # a start solution brings its bus lines: their trees are added right away for the same reason
        if start:
            for b in bus['inst']:
                if start.get(bus['act'][b].VarName, 0.0) < 0.5:
                    continue
                mem = [(locations[j].id, bus['unit'][b, j]) for j in range(n_locs) if start.get(bus['unit'][b, j].VarName, 0.0) > 0.5]
                mem += [(d, bus['dev'][b, d]) for d in bus['devices'] if (b, d) in bus['dev'] and start.get(bus['dev'][b, d].VarName, 0.0) > 0.5]
                if len(mem) > 1:
                    model._st_cuts[(b, tuple(sorted(e for e, _v in mem)))] = ([v for _e, v in mem], steiner_tree([e for e, _v in mem], bus['points'])[0])
            keep_steiner_cuts()
        cont_ids = {id(v) for v in cb_vars if v.VType == GRB.CONTINUOUS}

        def lazy_cb(m, where):
            # Heuristic solutions repaired in MIPSOL are handed to Gurobi at the next node
            if where == GRB.Callback.MIPNODE:
                if m._repair is not None and m.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                    rv, rvals = zip(*m._repair.items())
                    m.cbSetSolution(list(rv), list(rvals))
                    obj = m.cbUseSolution()
                    m._repair = None
                    if obj < GRB.INFINITY:
                        m._repairs_used += 1
                return
            if where != GRB.Callback.MIPSOL:
                return
            cuts_before = m._lazy_cuts
            m._cb_stats['candidates'] += 1
            vals = dict(zip((id(v) for v in cb_vars), m.cbGetSolution(cb_vars)))
            cv = dict(zip(comm_keys, m.cbGetSolution(comm_vars)))
            getv = lambda var: vals[id(var)]
            routes = flow_routes(net, getv)
            hot_val = {i: (1.0 if net['hot'][i] is None else getv(net['hot'][i])) for i in net['critical']}

            def silent(t):
                # a standby flow of a cold standby carries no traffic in the nominal state (path must still exist)
                tr = net['flow_meta'][t]
                for inst, key in ((tr.get('src_inst', 'x'), 'src_idx'), (tr.get('dst_inst', 'x'), 'dst_idx')):
                    if inst == 'r' and hot_val.get(tr[key], 1.0) < 0.5:
                        return True
                return False

            def indicator(t, u, v, w_if, p):
                # linear lower bound of [flow t starts at u, ends at v, uses type w_if on path p]
                src, dst = net['flow_ends'][t]
                pair = (min(u, v), max(u, v))
                expr = src[u] + dst[v] + net['sel'][pair + (w_if, p)] - 2
                if len(net['flow_elig'][t]) > 1:
                    expr = expr + net['w'][t, w_if] - 1
                return expr

            def traffic_indicator(t, u, v, w_if, p):
                # as indicator, and the flow carries traffic in the nominal state: a standby endpoint that may be cold
                # must be hot (the flows of a cold standby are silent until a failure activates it)
                expr = indicator(t, u, v, w_if, p)
                tr = net['flow_meta'][t]
                for inst, key in ((tr.get('src_inst', 'x'), 'src_idx'), (tr.get('dst_inst', 'x'), 'dst_idx')):
                    if inst == 'r' and net['hot'].get(tr[key]) is not None:
                        expr = expr + net['hot'][tr[key]] - 1
                return expr

            def tally(name, before):
                # which check rejected the candidate (added cuts) and how many cuts it added
                d = m._lazy_cuts - before
                if d > 0:
                    m._cb_stats['cuts_by'][name] = m._cb_stats['cuts_by'].get(name, 0) + d
                    m._cb_stats['rejected_by'][name] = m._cb_stats['rejected_by'].get(name, 0) + 1
                return m._lazy_cuts

            _c = m._lazy_cuts
            # --- bus lines: the cable length of an open line is at least the exact Steiner tree of its members (the
            # tree only grows with more members, so the cut stays valid for every superset)
            bus_members = {}
            for b in bus['inst']:
                if getv(bus['act'][b]) < 0.5:
                    continue
                mem = [(locations[j].id, bus['unit'][b, j]) for j in range(n_locs) if getv(bus['unit'][b, j]) > 0.5]
                mem += [(d, bus['dev'][b, d]) for d in bus['devices'] if (b, d) in bus['dev'] and getv(bus['dev'][b, d]) > 0.5]
                st_len = steiner_tree([e for e, _v in mem], bus['points'])[0]
                bus_members[b] = st_len
                if getv(bus['len'][b]) < st_len - 1e-6:
                    m.cbLazy(bus['len'][b] >= st_len * (gp.quicksum(v for _e, v in mem) - (len(mem) - 1)))
                    m._lazy_cuts += 1
                    m._st_cuts[(b, tuple(sorted(e for e, _v in mem)))] = ([v for _e, v in mem], st_len)

            def resources(u, v, w_if, p):
                # what a routed flow loads: the directed links of its path (point-to-point cables) or its bus line (the
                # two directions and the line's devices share the line's bit rate)
                if w_if in bus['types']:
                    return [('line', (w_if, p))]
                pair = (min(u, v), max(u, v))
                path = net['paths'][pair + (w_if,)][p]
                seq = path if (u, v) == pair else path[::-1]
                return [(a, b2, w_if) for a, b2 in zip(seq, seq[1:])]

            def capacity_left(res, counts):
                # bit rate available to unit-to-unit traffic on a resource in the current solution
                if res[0] == 'line':
                    ln = res[1]
                    dev_load = sum(bus['devices'][d]['volume'] for d in bus['devices'] if (ln, d) in bus['dev'] and getv(bus['dev'][ln, d]) > 0.5)
                    return cap_of[ln[0]] * net['bus_umax'] - dev_load
                a, b2, w_if = res
                return cap_of[w_if] * counts[(a, b2, w_if) if a < b2 else (b2, a, w_if)]

            def capacity_cut(res, terms):
                # capacity >= routed volume (lower-bound terms) [+ the device volumes of a line]
                if res[0] == 'line':
                    ln = res[1]
                    devs = [(bus['dev'][ln, d], bus['devices'][d]['volume']) for d in bus['devices'] if (ln, d) in bus['dev']]
                    m.cbLazy(cap_of[ln[0]] * net['bus_umax'] >= gp.quicksum(vol * dv for dv, vol in devs) + gp.quicksum(vol * ind for ind, vol in terms))
                else:
                    a, b2, w_if = res
                    key = (a, b2, w_if) if a < b2 else (b2, a, w_if)
                    m.cbLazy(cap_of[w_if] * comm[key] >= gp.quicksum(vol * ind for ind, vol in terms))
                m._lazy_cuts += 1

            _c = tally('bus length', _c)
            # --- traffic: a pair with traffic needs a selected path; cables and lines carry the routed volume
            load, contrib = {}, {}
            for t, rt in routes.items():
                if rt is None:
                    continue
                u, v, w_if, p = rt
                pair = (min(u, v), max(u, v))
                plist = net['paths'][pair + (w_if,)]
                if p is None:
                    src, dst = net['flow_ends'][t]
                    rhs = src[u] + dst[v] - 1
                    if len(net['flow_elig'][t]) > 1:
                        rhs = rhs + net['w'][t, w_if] - 1
                    m.cbLazy(gp.quicksum(net['sel'][pair + (w_if, q)] for q in range(len(plist))) >= rhs)
                    m._lazy_cuts += 1
                    continue
                if silent(t):
                    continue
                vol = net['volumes'][t]
                for res in resources(u, v, w_if, p):
                    load[res] = load.get(res, 0.0) + vol
                    contrib.setdefault(res, []).append((traffic_indicator(t, u, v, w_if, p), vol))
            for res, ld in load.items():
                if ld > capacity_left(res, cv) + 1e-6:
                    capacity_cut(res, contrib[res])

            _c = tally('paths and capacity', _c)
            # --- chain latency: the network time of a chain must stay within its budget. A crossing waits for every
            # other message on each of its resources, so the cut has, besides the own crossing of a flow, one term per
            # message sharing a resource with it: (its kbit / rate) * [both there], lower-bounded by ind_t + ind_m - 1
            nom_msgs = resource_messages(net, routes, getv)
            qf = net['lat']['factor'] if net['lat']['robust'] else 1.0

            def msg_ind(msg, res, term_of_flow):
                return term_of_flow(msg[1]) if msg[0] == 'f' else bus['dev'][res[1], msg[1]]

            def queue_terms(me, ind_me, res, msgs, term_of_flow):
                # cut terms of the wait of `me` on res (see queue_ms): every message of equal or higher priority,
                # (its kbit / rate) * [both there]; of the lower-priority ones the longest blocking frame
                pm, rate, out, block = message_prio(net, me), resource_rate(net, res), [], None
                for msg in msgs.get(res, []):
                    if msg == me:
                        continue
                    if message_prio(net, msg) <= pm:
                        out.append(qf * message_kbit(net, msg) / rate * (ind_me + msg_ind(msg, res, term_of_flow) - 1))
                    else:
                        kb = min(message_kbit(net, msg), frame_kbit(net, res))
                        if block is None or kb > block[0]:
                            block = (kb, msg)
                if block is not None:
                    out.append(qf * block[0] / rate * (ind_me + msg_ind(block[1], res, term_of_flow) - 1))
                return out

            def chain_terms(ch, picks, msgs, term_of_flow, add_terms, routes_now):
                # latency of a chain in one state (picks: flow -> its lower-bound indicator) and the cut terms
                total, terms = ch['const'], []
                for t, ind_t in picks:
                    rt = routes_now[t]
                    total += crossing_ms(net, t, rt, msgs)
                    if not add_terms:
                        continue
                    u, v, w_if, p = rt
                    own = robust_ms(net, sum(net['kbit'][t] / net['cap_of'][w_if] + net['hop_ms'][w_if] for _r in resources(u, v, w_if, p)))
                    terms.append(own * ind_t)
                    for res in resources(u, v, w_if, p):
                        terms += queue_terms(('f', t), ind_t, res, msgs, term_of_flow)
                for d in ch['dev_ends']:
                    line = device_line(net, d, getv)
                    if line is None:
                        continue
                    res = ('line', line)
                    total += qf * queue_ms(net, res, ('d', d), msgs)
                    if add_terms:
                        terms += queue_terms(('d', d), bus['dev'][line, d], res, msgs, term_of_flow)
                return total, terms

            def nominal_term(k):
                rt = routes[k]
                return traffic_indicator(k, rt[0], rt[1], rt[2], rt[3])

            for ch in net['chains']:
                picks = []
                for t in ch['flows']:
                    rt = routes.get(t)
                    if rt is None or rt[3] is None:
                        continue
                    picks.append((t, indicator(t, *rt)))
                total, _ = chain_terms(ch, picks, nom_msgs, nominal_term, False, routes)
                if total > ch['budget'] + 1e-6:
                    _t, terms = chain_terms(ch, picks, nom_msgs, nominal_term, True, routes)
                    m.cbLazy(gp.quicksum(terms) <= ch['budget'] - ch['const'])
                    m._lazy_cuts += 1

            _c = tally('chain latency', _c)
            # --- single failures: every critical SC keeps a served instance under every scenario
            placement_ok = True
            pos_x = {i: next((j for j in feasible_locs_per_sc[i] if getv(x[i, j]) > 0.5), None) for i in range(n_sc)}
            pos_r = {i: next((j for j in feasible_locs_per_sc[i] if getv(net['r'][i, j]) > 0.5), None) for i in net['critical']}

            def ends_of(t):
                src, dst = net['flow_ends'][t]
                return (next((k for k, var in src.items() if getv(var) > 0.5), None),
                        next((k for k, var in dst.items() if getv(var) > 0.5), None))

            def flow_alive(t, kind, elem, routes_):
                # (alive, variables that fix this outcome, path missing). The variables are what a valid cut needs:
                # the endpoint at the failed cell, or else the whole configuration of the flow (both endpoints, the
                # cable type choice, the path). A missing path is a traffic violation handled by the path cuts, never
                # a reason to cut the placement.
                u, v = ends_of(t)
                if u is None or v is None:
                    return False, [], False
                src, dst = net['flow_ends'][t]
                if kind == 'cell' and elem in (u, v):
                    return False, [src[u] if u == elem else dst[v]], False
                if u == v:
                    return True, [src[u], dst[v]], False
                rt = routes_.get(t)
                if rt is None or rt[3] is None:
                    return False, [], True
                _u, _v, w_if, p = rt
                pair = (min(u, v), max(u, v))
                path = net['paths'][pair + (w_if,)][p]
                cfg = [src[u], dst[v], net['sel'][pair + (w_if, p)]]
                if len(net['flow_elig'][t]) > 1:
                    cfg.append(net['w'][t, w_if])
                if kind in ('cell', 'switch') and elem in path[1:-1]:
                    return False, cfg, False
                if kind == 'switch' and w_if == 'ETH' and elem in (u, v):
                    return False, cfg, False   # the ETH ports of a switched cell are the switch's ports
                if kind == 'cable' and w_if not in bus['types']:
                    a, b, wtype = elem
                    if wtype == w_if and any({pa, pb} == {a, b} for pa, pb in zip(path, path[1:])):
                        return False, cfg, False
                if kind == 'line' and (w_if, p) == elem:
                    return False, cfg, False
                return True, cfg, False

            def instance_served(i, inst, kind, elem, routes_):
                # instance inst ('x' primary, 'r' standby) survives and every flow of i to a critical partner is
                # served by an alive partner instance; non-critical partners may be lost; peripheral devices are
                # single, the failure of their own attachment element is out of scope
                cell = pos_x[i] if inst == 'x' else pos_r[i]
                if cell is None or (kind == 'cell' and elem == cell):
                    return False, [], False
                involved, missing = [], False
                for t_base, role in flows_of[i]:
                    tr = net['flow_meta'][t_base]
                    if tr['type'] in ('SENS_SC', 'SC_ACT'):
                        keys = [('a', inst)] if tr['type'] == 'SENS_SC' else [(inst, 'a')]
                        att = ends_of(t_base)[0 if tr['type'] == 'SENS_SC' else 1]
                        if kind in ('cell', 'switch') and elem == att:
                            continue
                    else:
                        partner = tr['dst_idx'] if role == 'src' else tr['src_idx']
                        if partner not in pos_r:
                            continue
                        keys = [((inst, pi) if role == 'src' else (pi, inst)) for pi in ('x', 'r')]
                    served = False
                    for key in keys:
                        t = base_variants[t_base].get(key)
                        if t is None:
                            continue
                        ok, vs, miss = flow_alive(t, kind, elem, routes_)
                        involved += vs
                        missing = missing or miss
                        if ok:
                            served = True
                            break
                    if not served:
                        return False, involved, missing
                return True, involved, False

            def scenario_list(routes_):
                classes = set(net['fail_cfg'].get('classes', ['cell', 'switch', 'cable']))
                has_sw = 'SWITCH' in all_hw
                scenarios = []
                if 'cell' in classes:
                    scenarios += [('cell', c) for c in range(n_locs) if getv(y[c]) > 0.5 or (has_sw and getv(hw_use[c, 'SWITCH']) > 0.5)]
                if 'switch' in classes and has_sw:
                    scenarios += [('switch', c) for c in range(n_locs) if getv(hw_use[c, 'SWITCH']) > 0.5]
                if 'cable' in classes:
                    if routes_ is routes:
                        scenarios += [('cable', key) for key in comm_keys if cv[key] > 0.5]
                    else:
                        used = {(min(u, v), max(u, v), w_if) for rt in routes_.values() if rt for u, v, w_if, _p in [rt] if w_if not in bus['types']}
                        scenarios += [('cable', key) for key in used]
                    scenarios += [('line', ln) for ln in sorted({(rt[2], rt[3]) for rt in routes_.values()
                                                                if rt and rt[2] in bus['types'] and rt[3] is not None})]
                return scenarios

            VARIANT_ORDER = {'SC_SC': [('x', 'x'), ('x', 'r'), ('r', 'x'), ('r', 'r')],
                             'SENS_SC': [('a', 'x'), ('a', 'r')], 'SC_ACT': [('x', 'a'), ('r', 'a')]}

            def degraded_check(states, routes_, add_cuts):
                """chain network latency in every failure scenario over the surviving instances, with the queues of the
                traffic after the failure; returns the number of chains over their degraded bound (with add_cuts: one
                cut per violation)"""
                violations = 0
                for st in states:
                    msgs = None
                    for ch in net['chains']:
                        picks, broken = [], False
                        for t_base in ch['flows']:
                            pick = st['picks'].get(t_base)
                            if pick is None or pick == 'missing':
                                broken = True   # a lost non-critical member, or a survival violation cut elsewhere
                                break
                            t, reasons = pick
                            u, v = ends_of(t)
                            if u == v:
                                continue
                            ind = indicator(t, *routes_[t])
                            for rv in reasons:
                                ind = ind + rv - 1
                            picks.append((t, ind))
                        if st['kind'] == 'line' and any(device_line(net, d, getv) == st['elem'] for d in ch['dev_ends']):
                            broken = True       # the end device's own line failed: out of scope (devices are single)
                        if broken:
                            continue
                        if msgs is None:
                            msgs = resource_messages(net, routes_, getv, carried=set(st['carried']),
                                                     failed_line=(st['elem'] if st['kind'] == 'line' else None))
                        term = (lambda k, st=st: degraded_term(k, st['carried'][k], routes_))
                        total, _ = chain_terms(ch, picks, msgs, term, False, routes_)
                        if total > ch['degraded'] + 1e-6:
                            violations += 1
                            if add_cuts:
                                _t, terms = chain_terms(ch, picks, msgs, term, True, routes_)
                                m.cbLazy(gp.quicksum(terms) <= ch['degraded'] - ch['const'])
                                m._lazy_cuts += 1
                                m._fail_cuts += 1
                return violations

            def scenario_check(routes_, add_cuts):
                """number of (scenario, SC) violations; with add_cuts the first violation per scenario is cut"""
                violations = 0
                scenarios = scenario_list(routes_)
                for kind, elem in scenarios:
                    for i in net['critical']:
                        ok_x, vs_x, miss_x = instance_served(i, 'x', kind, elem, routes_)
                        if ok_x:
                            continue
                        ok_r, vs_r, miss_r = instance_served(i, 'r', kind, elem, routes_)
                        if ok_r:
                            continue
                        if miss_x or miss_r:
                            continue   # undecided until the missing paths are selected (path cuts pending)
                        violations += 1
                        if add_cuts:
                            vars_ = [x[i, pos_x[i]], net['r'][i, pos_r[i]]] + vs_x + vs_r
                            for t_base, role in flows_of[i]:
                                tr = net['flow_meta'][t_base]
                                k = tr.get('dst_idx' if role == 'src' else 'src_idx')
                                if k is None:
                                    continue
                                if pos_x.get(k) is not None:
                                    vars_.append(x[k, pos_x[k]])
                                if pos_r.get(k) is not None:
                                    vars_.append(net['r'][k, pos_r[k]])
                            uniq = list({id(v): v for v in vars_}.values())
                            m.cbLazy(gp.quicksum(uniq) <= len(uniq) - 1)
                            m._lazy_cuts += 1
                            m._fail_cuts += 1
                        break
                return violations

            def scenario_states(routes_):
                """Per single-failure scenario: the traffic that keeps running (alive flows of primaries and hot
                standbys, reasons None) and, per data flow, its first surviving variant of the degraded latency rule,
                which activates the cold standbys it needs (with the reasons its earlier variants are dead)."""
                states = []
                for kind, elem in scenario_list(routes_):
                    carried, picks = {}, {}
                    for t in net['flow_meta']:
                        if not silent(t):
                            ok, _vs, _miss = flow_alive(t, kind, elem, routes_)
                            if ok:
                                carried[t] = None
                    for t_base, variants in base_variants.items():
                        reasons, pick = [], None
                        for key in VARIANT_ORDER[net['flow_meta'][t_base]['type']]:
                            t = variants.get(key)
                            if t is None:
                                continue
                            ok, vs, miss = flow_alive(t, kind, elem, routes_)
                            if miss:
                                pick = 'missing'
                                break
                            if ok:
                                pick = (t, reasons)
                                carried.setdefault(t, reasons)
                                break
                            reasons = reasons + vs
                        picks[t_base] = pick
                    states.append({'kind': kind, 'elem': elem, 'carried': carried, 'picks': picks})
                return states

            def degraded_traffic(states, routes_):
                """load and members per resource after every failure: [(load {res: volume}, members {res: [(t, reasons)]})]"""
                out = []
                for st in states:
                    load, members = {}, {}
                    for t, reasons in st['carried'].items():
                        u, v = ends_of(t)
                        if u == v:
                            continue
                        _u, _v, w_if, p = routes_[t]
                        for res in resources(u, v, w_if, p):
                            load[res] = load.get(res, 0.0) + net['volumes'][t]
                            members.setdefault(res, []).append((t, reasons))
                    out.append((load, members))
                return out

            def degraded_term(t, reasons, routes_):
                # lower bound of [flow t carries traffic on its path in the scenario]
                u, v = ends_of(t)
                _u, _v, w_if, p = routes_[t]
                if reasons is None:
                    return traffic_indicator(t, u, v, w_if, p)
                ind = indicator(t, u, v, w_if, p)
                for rv in reasons:
                    ind = ind + rv - 1
                return ind

            if net['critical']:
                states = scenario_states(routes)
                # link capacity after a failure (cables can be added, so this never rejects the placement)
                for load, members in degraded_traffic(states, routes):
                    for res, ld in load.items():
                        if ld > capacity_left(res, cv) + 1e-6:
                            capacity_cut(res, [(degraded_term(t, reasons, routes), net['volumes'][t]) for t, reasons in members[res]])
                            m._fail_cuts += 1
                _c = tally('capacity after failure', _c)
                if scenario_check(routes, True) > 0:
                    placement_ok = False
                _c = tally('failure survival', _c)
                if net['chains'] and degraded_check(states, routes, True) > 0:
                    placement_ok = False
                _c = tally('degraded latency', _c)
                # cold standbys activated by a cell failure must fit next to the nominal load of their cell
                for c in range(n_locs):
                    if getv(y[c]) < 0.5:
                        continue
                    activated = [i for i in net['critical'] if pos_x[i] == c and pos_r[i] is not None and hot_val[i] < 0.5]
                    groups = {}
                    for i in activated:
                        groups.setdefault((pos_r[i], scs[i].asil_req), []).append(i)
                    for (j, a), act in groups.items():
                        H = site_derating(locations[j], uncertainty)
                        for res in ('cpu', 'ram', 'rom'):
                            cap = float(partitions.get(f'{res}_cap', 0) or 0)
                            if cap <= 0:
                                continue
                            dem = lambda i, res=res: getattr(scs[i], f'{res}_req') * sw_multiplier(scs[i], uncertainty)
                            nominal = sum(dem(i) for i in feasible_scs_per_loc[j] if scs[i].asil_req == a and getv(x[i, j]) > 0.5)
                            hot_here = sum(dem(i) for i in net['critical'] if (i, j) in net['hr'] and scs[i].asil_req == a and getv(net['hr'][i, j]) > 0.5)
                            if nominal + hot_here + sum(dem(i) for i in act) <= getv(P[j, a]) * cap * H + 1e-6:
                                continue
                            placement_ok = False
                            expr = gp.quicksum(dem(i) * x[i, j] for i in feasible_scs_per_loc[j] if scs[i].asil_req == a)
                            expr += gp.quicksum(dem(i) * net['hr'][i, j] for i in net['critical'] if (i, j) in net['hr'] and scs[i].asil_req == a)
                            for i in act:
                                term = x[i, c] + net['r'][i, j] - 1
                                if net['hot'][i] is not None:
                                    term = term - net['hot'][i]
                                expr += dem(i) * term
                            m.cbLazy(expr <= P[j, a] * cap * H)
                            m._lazy_cuts += 1
                            m._fail_cuts += 1
                    # the same for the whole unit (constraint 1d: site caps over all ASILs)
                    by_site = {}
                    for i in activated:
                        by_site.setdefault(pos_r[i], []).append(i)
                    for j, act in by_site.items():
                        H = site_derating(locations[j], uncertainty)
                        for res in ('cpu', 'ram', 'rom'):
                            site_cap = partitions.get(f'site_{res}_cap')
                            if site_cap is None:
                                continue
                            dem = lambda i, res=res: getattr(scs[i], f'{res}_req') * sw_multiplier(scs[i], uncertainty)
                            nominal = sum(dem(i) for i in feasible_scs_per_loc[j] if getv(x[i, j]) > 0.5)
                            hot_here = sum(dem(i) for i in net['critical'] if (i, j) in net['hr'] and getv(net['hr'][i, j]) > 0.5)
                            if nominal + hot_here + sum(dem(i) for i in act) <= float(site_cap) * H + 1e-6:
                                continue
                            placement_ok = False
                            expr = gp.quicksum(dem(i) * x[i, j] for i in feasible_scs_per_loc[j])
                            expr += gp.quicksum(dem(i) * net['hr'][i, j] for i in net['critical'] if (i, j) in net['hr'])
                            for i in act:
                                term = x[i, c] + net['r'][i, j] - 1
                                if net['hot'][i] is not None:
                                    term = term - net['hot'][i]
                                expr += dem(i) * term
                            m.cbLazy(expr <= float(site_cap) * H)
                            m._lazy_cuts += 1
                            m._fail_cuts += 1

            _c = tally('standby capacity after failure', _c)

            def repair_routes():
                """routes for the repair: (routes, ) or None when some flow has no admissible option"""
                chain_flows = set()
                for ch in net['chains']:
                    for tb in ch['flows']:
                        chain_flows.update(base_variants.get(tb, {}).values())
                chain_devs = {d for ch in net['chains'] for d in ch['dev_ends']}
                kb = net['kbit']
                placed, line_load = {}, {}     # resource -> [(priority key, kbit, chain message?)] routed so far
                for (ln, d), var in bus['dev'].items():
                    if getv(var) > 0.5:
                        placed.setdefault(('line', ln), []).append((message_prio(net, ('d', d)), bus['devices'][d]['kbit'], d in chain_devs))
                        line_load[ln] = line_load.get(ln, 0.0) + bus['devices'][d]['volume']

                def wait_on(r, pm):
                    # queue in front of a message of priority pm on r (as queue_ms)
                    msgs_r = placed.get(r, [])
                    block = max((min(k, frame_kbit(net, r)) for p_, k, _c in msgs_r if p_ > pm), default=0.0)
                    return sum(k for p_, k, _c in msgs_r if p_ <= pm) + block

                def delay_to_chains(r, pm, k):
                    # what a new message adds to the queues of the chain messages already on r
                    return sum((k if pm <= p_ else min(k, frame_kbit(net, r))) for p_, _k, c in placed.get(r, []) if c)
                switch_at = {j for j in range(n_locs) if has_switch_hw and getv(hw_use[j, 'SWITCH']) > 0.5}
                pair_path, out = {}, {}
                flows = list(net['flow_meta'])
                flows.sort(key=lambda t: (0 if t in chain_flows else 1, -kb[t]))
                for t in flows:
                    u, v = ends_of(t)
                    if u is None or v is None or u == v:
                        out[t] = None
                        continue
                    pair = (min(u, v), max(u, v))
                    quiet = silent(t)
                    best = None
                    for n in net['flow_elig'][t]:
                        plist = net['paths'].get(pair + (n,), [])
                        cands = [pair_path[pair + (n,)]] if pair + (n,) in pair_path else range(len(plist))
                        for p in cands:
                            if n in bus['types']:
                                ln = (n, p)
                                if getv(bus['unit'][ln, u]) < 0.5 or getv(bus['unit'][ln, v]) < 0.5:
                                    continue
                                if not quiet and line_load.get(ln, 0.0) + net['volumes'][t] > cap_of[n] * net['bus_umax'] + 1e-9:
                                    continue
                            elif any(r not in switch_at for r in plist[p][1:-1]):
                                continue
                            rate = cap_of[n]
                            res_list = resources(u, v, n, p)
                            pm = message_prio(net, ('f', t))
                            own = sum(kb[t] / rate + net['hop_ms'][n] + wait_on(r, pm) / rate for r in res_list)
                            impact = 0.0 if quiet else sum(delay_to_chains(r, pm, kb[t]) / rate for r in res_list)
                            score = (own if t in chain_flows else 1e-3 * own) + impact
                            if best is None or score < best[0] - 1e-12:
                                best = (score, n, p)
                    if best is None:
                        return None
                    _score, n, p = best
                    pair_path.setdefault(pair + (n,), p)
                    out[t] = (u, v, n, p)
                    if not quiet:
                        for r in resources(u, v, n, p):
                            placed.setdefault(r, []).append((message_prio(net, ('f', t)), kb[t], t in chain_flows))
                        if n in bus['types']:
                            line_load[(n, p)] = line_load.get((n, p), 0.0) + net['volumes'][t]
                return (out,)

            # --- repair: the placement is kept, the traffic is routed by repair_routes, the cable counts follow the
            # loads; handed to Gurobi as a heuristic solution if it passes all checks
            if m._lazy_cuts > cuts_before:
                m._cb_stats['rejected'] += 1
            if not placement_ok:
                m._cb_stats['repair_fail']['placement'] = m._cb_stats['repair_fail'].get('placement', 0) + 1
            if m._lazy_cuts > cuts_before and placement_ok and m._repair is None:
                m._cb_stats['repair_tries'] += 1
                stage = None
                fix = {var: (getv(var) if id(var) in cont_ids else round(getv(var))) for var in cb_vars}
                for var in net['sel'].values():
                    fix[var] = 0
                for b in bus['inst']:
                    fix[bus['len'][b]] = bus_members.get(b, 0.0)
                # queue-aware routing: chain messages first, then the heavy ones; every flow takes the option
                # (cable type, path or bus line) that adds the least delay to itself and to the chain messages
                # already placed on the same resources; one path per unit pair and type; bus lines within their
                # utilization limit; relays only over cells that have a switch
                routed = repair_routes()
                feasible = routed is not None
                stage = None if feasible else 'no route option'
                rep_routes, rload, need = (routed[0] if routed else {}), {}, set()
                for t, rt in rep_routes.items():
                    if rt is None:
                        continue
                    u, v, w_if, p = rt
                    pair = (min(u, v), max(u, v))
                    if len(net['flow_elig'][t]) > 1:
                        for n in net['flow_elig'][t]:
                            fix[net['w'][t, n]] = 1 if n == w_if else 0
                    fix[net['sel'][pair + (w_if, p)]] = 1
                    if w_if not in bus['types']:
                        need.add(pair + (w_if,))
                    if not silent(t):
                        for res in resources(u, v, w_if, p):
                            key = res if res[0] == 'line' else (min(res[0], res[1]), max(res[0], res[1]), res[2])
                            rload[key] = rload.get(key, 0.0) + net['volumes'][t]   # both directions: conservative
                rep_states = scenario_states(rep_routes) if (feasible and net['critical']) else []
                if feasible and net['critical']:
                    for load, _members in degraded_traffic(rep_states, rep_routes):
                        for res, ld in load.items():
                            key = res if res[0] == 'line' else (min(res[0], res[1]), max(res[0], res[1]), res[2])
                            rload[key] = max(rload.get(key, 0.0), ld)
                for key in comm_keys:
                    fix[comm[key]] = 0
                for key in need:
                    n_cab = max(1, int(np.ceil(rload.get(key, 0.0) / cap_of[key[2]] - 1e-9)))
                    if n_cab > net['comm_ub'][key[2]]:
                        feasible, stage = False, stage or 'cable count'
                    fix[comm[key]] = n_cab
                for key, ld in rload.items():   # a bus line cannot be widened: its traffic must fit as it is
                    if key[0] == 'line' and ld > capacity_left(key, None) + 1e-6:
                        feasible, stage = False, stage or 'bus line capacity'
                if feasible and net['chains']:
                    feasible = all(c['ok'] for c in chain_latencies(net, rep_routes, getv).values())
                    stage = stage or (None if feasible else 'chain latency')
                if feasible and net['critical']:
                    feasible = scenario_check(rep_routes, False) == 0
                    stage = stage or (None if feasible else 'failure survival')
                if feasible and net['critical'] and net['chains']:
                    feasible = degraded_check(rep_states, rep_routes, False) == 0
                    stage = stage or (None if feasible else 'degraded latency')
                if feasible:
                    m._repair = fix
                    m._cb_stats['repair_valid'] += 1
                else:
                    m._cb_stats['repair_fail'][stage] = m._cb_stats['repair_fail'].get(stage, 0) + 1
            now = time.time()
            if now - m._cb_last_print >= 60.0:
                m._cb_last_print = now
                st = m._cb_stats
                print(f"[callback] candidates {st['candidates']}, rejected {st['rejected']}, repair tries {st['repair_tries']}, "
                      f"valid repairs {st['repair_valid']}, used {m._repairs_used}, repair failures {st['repair_fail']}, "
                      f"rejected by {st['rejected_by']}, cuts by {st['cuts_by']}", flush=True)

        def _solve_lexicographic():
            """Phase 1: primary objective. Phase 2: secondary objective with the primary fixed at its phase-1 value."""
            if tl is not None:
                model.setParam('TimeLimit', tl)
            model.setObjective(primary_expr, GRB.MINIMIZE)
            phases = []
            print(f"\n[Lexicographic] Phase 1: minimize {primary_name}")
            model.optimize(lazy_cb)
            phases.append(self._phase_info(model, primary_name))
            print(f"[callback] after phase 1: {model._cb_stats}, repairs used {model._repairs_used}", flush=True)
            lex_constr = None
            if model.SolCount > 0 and model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                primary_val = primary_expr.getValue()
                all_vars = model.getVars()
                phase1 = model.getAttr('X', all_vars)
                keep_steiner_cuts()
                # Phase 2 always runs: a phase-1 solution alone has an arbitrary secondary objective (e.g. a min-cable
                # design with junk cost). It gets the remaining time, but at least phase2_time seconds; the phase-1
                # solution stays feasible and is Gurobi's start solution.
                # tolerance 0.01 (far below the 1e-4 optimality gap): the start design's bus lengths and integers are
                # recomputed by Gurobi within its feasibility tolerances and must still meet the bound
                lex_constr = model.addConstr(
                    primary_expr <= primary_val + max(1e-2, 1e-6 * abs(primary_val)),
                    name=f"lex_fix_{primary_name}"
                )
                model.setObjective(secondary_expr, GRB.MINIMIZE)
                model.setAttr('Start', all_vars, phase1)
                if tl is not None:
                    model.setParam('TimeLimit', max(tl - model.Runtime, float(phase2_time)))
                print(f"\n[Lexicographic] Phase 2: minimize {secondary_name} with {primary_name} <= {primary_val:.6f}")
                model.optimize(lazy_cb)
                phases.append(self._phase_info(model, secondary_name))
                print(f"[callback] after phase 2: {model._cb_stats}, repairs used {model._repairs_used}", flush=True)
                if model.SolCount == 0:
                    # never lose the phase-1 design: re-accept it under the primary objective (start solution)
                    print(f"\n[Lexicographic] Phase 2 found no solution in its time; the phase-1 design is kept")
                    model.remove(lex_constr)
                    lex_constr = None
                    keep_steiner_cuts()
                    model.setObjective(primary_expr, GRB.MINIMIZE)
                    model.setAttr('Start', all_vars, phase1)
                    model.setParam('TimeLimit', float(phase2_time))
                    model.optimize(lazy_cb)
                    phases[-1]['status'] = 'NO_SOLUTION'
            return phases, lex_constr

        # The pooled capacity constraint c2 is exact unless the SCs of one (site, ASIL) group cannot be packed into
        # the P[j,a] partitions opened for them. Every solution is checked; a failing group gets a cut and the
        # model is solved again.
        partition_map = {}
        packing_cuts = 0
        while True:
            phases, lex_constr = _solve_lexicographic()
            if not (model.SolCount > 0 and model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT)):
                break
            partition_map, failed = pack_partitions(x, P, scs, locations, partitions, uncertainty, standby=net['hr'])
            if not failed:
                break
            if packing_cuts >= MAX_PACKING_CUTS:
                print(f"\n[Packing] WARNING: {len(failed)} (site, ASIL) group(s) still not packable after {packing_cuts} cuts; keeping the solution")
                break
            for (j, a, members, p_val) in failed:
                # If all SCs of the group are hosted at site j, at least p_val + 1 partitions are needed
                model.addConstr(
                    P[j, a] >= (p_val + 1) * (gp.quicksum(var for _key, var in members) - len(members) + 1),
                    name=f"packing_cut_{packing_cuts}_{locations[j].id}_asil{a}"
                )
                packing_cuts += 1
                print(f"\n[Packing] {len(members)} SCs of ASIL {a} at {locations[j].id} do not fit into {p_val} partition(s); cut added")
            if lex_constr is not None:
                model.remove(lex_constr)
            keep_steiner_cuts()
            print(f"[Packing] Solving again with {packing_cuts} cut(s)")

        # ======================= RETURN FORMATTED SOLUTION =======================
        if model.SolCount > 0 and model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            time_limited = any(ph['status'] != 'OPTIMAL' for ph in phases)
            if time_limited:
                print(f"\n⚠ Time limit reached; keeping best solution found (gaps: " + ", ".join(
                    f"{ph['objective_name']} " + (f"{100.0 * ph['mip_gap']:.4f}%" if ph['mip_gap'] is not None else ph['status']) for ph in phases) + ")")
            else:
                print(f"\n✓ Optimal solution found!")
            # Extract and return formatted solution
            formatted_solution = extract_solution(
                x, P, y, hw_use, if_use, comm, scs, locations, sensors, actuators,
                cable_types, partitions, hardwares, interfaces, comm_matrix=comm_matrix,
                traffic_flows=traffic_flows, net=net, attach_s=attach_s, attach_a=attach_a, bus=bus,
                partition_map=partition_map, standby=(net['r'], net['hot']), uncertainty=uncertainty
            )
            formatted_solution['solve_info'] = {
                'objective_type': 'min_cable_length' if minimize_cable_length else 'min_cost',
                'objectives': [primary_name, secondary_name],
                'cable_length_limit': cable_length_limit,
                'cost_limit': cost_limit,
                'phases': phases,
                'packing_cuts': packing_cuts,
                'backbone_candidate_paths': net['k_paths'],
                'backbone_interfaces': net['bb_ifaces'],
                'lazy_cuts': int(getattr(model, '_lazy_cuts', 0)),
                'chains': len(net.get('chains', [])),
                'critical': len(net['critical']),
                'failure_cuts': int(getattr(model, '_fail_cuts', 0)),
                'repairs_used': int(getattr(model, '_repairs_used', 0)),
                'callback_stats': dict(getattr(model, '_cb_stats', {})),
                'gurobi_params': gurobi_params,
                'runtime_s': sum(ph['runtime_s'] for ph in phases),
                'num_vars': int(model.NumVars),
                'num_constrs': int(model.NumConstrs),
                'num_bin_vars': int(model.NumBinVars),
                'num_int_vars': int(model.NumIntVars),
                'lp_file': lp_path,
                'uncertainty': {k: bool((uncertainty or {}).get(k)) for k in UNCERTAINTY_BLOCKS},
                'growth_alpha': float((uncertainty or {}).get('growth_alpha', 0.0)) if (uncertainty or {}).get('growth') else None,
            }
            formatted_solution['status'] = 'TIME_LIMIT' if time_limited else 'OPTIMAL'
            # values of this solution, the start of the next solve of a sweep (names are unique: checked once here)
            names = model.getAttr('VarName', model.getVars())
            if len(set(names)) != len(names):
                print(f"  [WARN] {len(names) - len(set(names))} duplicate variable names; start solutions may be incomplete")
            self.last_values = dict(zip(names, model.getAttr('X', model.getVars())))
            return formatted_solution
        elif model.status == GRB.TIME_LIMIT:
            print(f"\n✗ Optimization reached time limit without any feasible solution. Skipping this Pareto point.")
            return None
        else:
            print(f"\n✗ Optimization failed. Status: {model.status}")
            return None

    @staticmethod
    def _phase_info(model, objective_name):
        """Solver statistics of one lexicographic phase."""
        status_names = {GRB.OPTIMAL: 'OPTIMAL', GRB.TIME_LIMIT: 'TIME_LIMIT', GRB.INFEASIBLE: 'INFEASIBLE',
                        GRB.INF_OR_UNBD: 'INF_OR_UNBD', GRB.UNBOUNDED: 'UNBOUNDED'}
        has_sol = model.SolCount > 0
        return {
            'objective_name': objective_name,
            'status': status_names.get(model.status, str(model.status)),
            'objective': float(model.ObjVal) if has_sol else None,
            'bound': float(model.ObjBound) if has_sol else None,
            'mip_gap': float(model.MIPGap) if has_sol else None,
            'runtime_s': float(model.Runtime),
            'node_count': float(model.NodeCount),
        }

    def optimize_growth_sweep(self, scs, locations, sensors, actuators, cable_types, comm_matrix,
                              partitions=None, hardwares=None, interfaces=None, solve_time_limit=None, mip_gap=None,
                              lp_dir=None, uncertainty=None, backbone=None, phase2_time=600, gurobi_params=None,
                              chains=None, **_ignored):
        """
        Cost against software growth: for every growth level of configs/uncertainty.json (growth.alpha_levels) the
        min-cost design (second phase: cable length). Returns the solutions in the order of the levels; each carries
        solve_info.growth_alpha.
        """
        levels = [float(a) for a in (uncertainty or {}).get('params', {}).get('growth', {}).get('alpha_levels', [0.0])]
        print("=" * 80)
        print(f"GROWTH SWEEP (min cost per growth level): alpha in {levels}")
        print("=" * 80)
        solutions = []
        self.last_values = None
        for alpha in levels:
            unc = dict(uncertainty or {})
            unc['growth'] = True
            unc['growth_alpha'] = alpha
            print(f"\n[Growth] alpha = {alpha:.2f}")
            sol = self.optimize(
                scs, locations, sensors, actuators, cable_types, comm_matrix,
                partitions=partitions, hardwares=hardwares, interfaces=interfaces,
                time_limit=solve_time_limit, mip_gap=mip_gap,
                lp_path=(os.path.join(lp_dir, f"model_alpha_{alpha:g}.lp.gz") if lp_dir else None),
                uncertainty=unc, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params,
                chains=chains, start=self.last_values)
            if sol:
                solutions.append(sol)
            else:
                print(f"[Growth] alpha = {alpha:.2f}: no design found (infeasible, or no feasible design within the time limit); sweep stopped")
                break
        print(f"Growth sweep: {len(solutions)} of {len(levels)} levels solved")
        return solutions

    def optimize_pareto_epsilon_constraint(self, scs, locations, sensors, actuators, cable_types, comm_matrix,
                                           partitions=None, hardwares=None, interfaces=None,
                                           enable_comm_bw_constraints=True, comm_bw_big_m=10000, num_points=5,
                                           solve_time_limit=None, mip_gap=None, lp_dir=None, uncertainty=None, backbone=None, phase2_time=600, gurobi_params=None, chains=None):
        """
        Build Pareto front between total_cost and cable_length via epsilon-constraint.

        Auto-selects sweep axis: uses whichever objective (cost or cable_length)
        has the wider relative range to maximize the chance of feasible intermediate points.

        Steps:
        1) Solve min total_cost (gives one extreme)
        2) Solve min cable_length (gives other extreme)
        3) Determine sweep axis based on relative range width
        4) Sweep the chosen axis for intermediate points
        """
        print("=" * 80)
        print("PARETO OPTIMIZATION (total_cost vs cable_length)")
        print("=" * 80)

        def _lp(name):
            return os.path.join(lp_dir, f"model_{name}.lp.gz") if lp_dir else None

        self.last_values = None
        # Extreme 1: minimum cost
        sol_min_cost = self.optimize(
            scs, locations, sensors, actuators, cable_types, comm_matrix,
            partitions=partitions, hardwares=hardwares, interfaces=interfaces,
            enable_comm_bw_constraints=enable_comm_bw_constraints,
            comm_bw_big_m=comm_bw_big_m,
            time_limit=solve_time_limit,
            mip_gap=mip_gap,
            lp_path=_lp("min_cost"),
            uncertainty=uncertainty, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params, chains=chains,
            start=self.last_values
        )

        if num_points is not None and int(num_points) == 1:
            print("[Pareto] num_points = 1: only the min-cost extreme is solved")
            return [sol_min_cost] if sol_min_cost else []

        # Extreme 2: minimum cable length
        sol_min_len = self.optimize(
            scs, locations, sensors, actuators, cable_types, comm_matrix,
            partitions=partitions, hardwares=hardwares, interfaces=interfaces,
            enable_comm_bw_constraints=enable_comm_bw_constraints,
            comm_bw_big_m=comm_bw_big_m,
            minimize_cable_length=True,
            time_limit=solve_time_limit,
            mip_gap=mip_gap,
            lp_path=_lp("min_cable"),
            uncertainty=uncertainty, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params, chains=chains,
            start=self.last_values
        )

        pareto_solutions = []
        seen = set()

        def _add_unique(sol):
            if not sol or sol.get('status') not in ('OPTIMAL', 'TIME_LIMIT'):
                return
            key = (round(float(sol.get('total_cost', 0.0)), 4), round(float(sol.get('cable_length', 0.0)), 4))
            if key in seen:
                return
            seen.add(key)
            pareto_solutions.append(sol)

        _add_unique(sol_min_cost)
        _add_unique(sol_min_len)

        if not sol_min_cost or not sol_min_len:
            return pareto_solutions

        # Extract objective values for both extremes
        cost_at_min_cost  = float(sol_min_cost.get('total_cost', 0.0))
        cable_at_min_cost = float(sol_min_cost.get('cable_length', 0.0))
        cost_at_min_cable  = float(sol_min_len.get('total_cost', 0.0))
        cable_at_min_cable = float(sol_min_len.get('cable_length', 0.0))

        if num_points is None:
            num_points = 5
        num_points = max(2, int(num_points))

        # Compute relative range for each axis to decide sweep direction
        cable_range = abs(cable_at_min_cost - cable_at_min_cable)
        cost_range  = abs(cost_at_min_cable - cost_at_min_cost)
        cable_mid   = max(1e-9, (cable_at_min_cost + cable_at_min_cable) / 2.0)
        cost_mid    = max(1e-9, (cost_at_min_cost + cost_at_min_cable) / 2.0)

        cable_relative = cable_range / cable_mid
        cost_relative  = cost_range / cost_mid

        print(f"\n[Pareto] Cable range: {cable_at_min_cable:.2f} - {cable_at_min_cost:.2f} (span={cable_range:.2f}, relative={cable_relative:.2%})")
        print(f"[Pareto] Cost  range: {cost_at_min_cost:.2f} - {cost_at_min_cable:.2f} (span={cost_range:.2f}, relative={cost_relative:.2%})")

        if num_points > 2:
            if cost_relative > cable_relative:
                # Sweep on COST axis (wider): constrain cost, minimize cable_length
                c_min = cost_at_min_cost
                c_max = cost_at_min_cable
                print(f"[Pareto] → Sweeping on COST axis (relative {cost_relative:.2%} > {cable_relative:.2%})")
                print(f"[Pareto]   cost_min={c_min:.2f}, cost_max={c_max:.2f}")
                eps_values = np.linspace(c_min, c_max, num_points)[1:-1]
                for k, eps_cost in enumerate(eps_values, 1):
                    print(f"\n[Pareto] Solving with cost_limit={eps_cost:.2f} (minimize cable_length)...")
                    sol_eps = self.optimize(
                        scs, locations, sensors, actuators, cable_types, comm_matrix,
                        partitions=partitions, hardwares=hardwares, interfaces=interfaces,
                        enable_comm_bw_constraints=enable_comm_bw_constraints,
                        comm_bw_big_m=comm_bw_big_m,
                        minimize_cable_length=True,
                        cost_limit=float(eps_cost) + 1e-2,
                        time_limit=solve_time_limit,
                        mip_gap=mip_gap,
                        lp_path=_lp(f"eps_cost_{k}"),
                        uncertainty=uncertainty, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params, chains=chains,
                        start=self.last_values
                    )
                    _add_unique(sol_eps)
            else:
                # Sweep on CABLE axis (wider or equal): constrain cable, minimize cost
                l_max = cable_at_min_cost
                l_min = cable_at_min_cable
                print(f"[Pareto] → Sweeping on CABLE axis (relative {cable_relative:.2%} >= {cost_relative:.2%})")
                print(f"[Pareto]   cable_min={l_min:.2f}, cable_max={l_max:.2f}")
                eps_values = np.linspace(l_max, l_min, num_points)[1:-1]
                for k, eps_len in enumerate(eps_values, 1):
                    print(f"\n[Pareto] Solving with cable_length_limit={eps_len:.4f}...")
                    sol_eps = self.optimize(
                        scs, locations, sensors, actuators, cable_types, comm_matrix,
                        partitions=partitions, hardwares=hardwares, interfaces=interfaces,
                        enable_comm_bw_constraints=enable_comm_bw_constraints,
                        comm_bw_big_m=comm_bw_big_m,
                        cable_length_limit=float(eps_len) + 1e-6,
                        time_limit=solve_time_limit,
                        mip_gap=mip_gap,
                        lp_path=_lp(f"eps_cable_{k}"),
                        uncertainty=uncertainty, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params, chains=chains,
                        start=self.last_values
                    )
                    _add_unique(sol_eps)

        pareto_solutions.sort(key=lambda s: (float(s.get('total_cost', 0.0)), float(s.get('cable_length', 0.0))))   # solution_1 = min cost

        # Remove dominated solutions: a solution is dominated if another has both <= cost and <= cable_length
        if len(pareto_solutions) > 1:
            non_dominated = []
            for sol in pareto_solutions:
                c = float(sol.get('total_cost', 0.0))
                l = float(sol.get('cable_length', 0.0))
                dominated = False
                for other in pareto_solutions:
                    if other is sol:
                        continue
                    oc = float(other.get('total_cost', 0.0))
                    ol = float(other.get('cable_length', 0.0))
                    if oc <= c and ol <= l and (oc < c or ol < l):
                        dominated = True
                        break
                if not dominated:
                    non_dominated.append(sol)
            if non_dominated:
                removed = len(pareto_solutions) - len(non_dominated)
                if removed > 0:
                    print(f"[Pareto] Removed {removed} dominated solution(s)")
                pareto_solutions = non_dominated

        print(f"Pareto solutions found: {len(pareto_solutions)}")
        return pareto_solutions


