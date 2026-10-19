import os
import numpy as np
os.environ.setdefault("GRB_LICENSE_FILE", os.path.expanduser("~/gurobi.lic"))
import gurobipy as gp
from gurobipy import GRB
from optimizer_utils import (get_distance, build_sensor_lookup, build_actuator_lookup,
                             precompute_latency_infeasible_pairs, build_cost_map, build_latency_map, extract_solution,
                             sw_multiplier, site_derating, pack_partitions)

# Uncertainty-aware constraint blocks, each switched on by its own CLI flag (parameters in configs/uncertainty.json)
UNCERTAINTY_BLOCKS = ('sw_margin', 'derating', 'redundancy', 'ai_contention', 'robust_latency')

# Packing check of the pooled (site, ASIL) capacity constraint: cuts added before giving up (see optimize())
MAX_PACKING_CUTS = 10


class AssignmentOptimizer:
    def __init__(self):
        print("Initialized AssignmentOptimizerNew with quadratic ECU-ECU costs and linear sensor/actuator costs")
        pass

    def calculate_cable_expressions(self, x, scs, locations, sensors, actuators, cable_types, comm_matrix, feasible_locs_per_sc, comm=None, attach_s=None, attach_a=None, shared_attach_s=None, shared_attach_a=None, shared_trunk_len=None, shared_extra_trunk_len=None):
        """
        Calculate cable cost, distance, and latency as Gurobi linear expressions (NOT constraints).
        Includes: sensor/actuator cables and ECU-ECU/Location communication cables.
        
        Args:
            comm: Dict of ECU-ECU communication variables [j1, j2, iface] (LEGO model)
                  If None or empty, skips communication backbone calculation
        
        Returns:
            cable_cost_expr: Gurobi expression for total cable costs (includes all components)
            cable_distance_expr: Gurobi expression for total cable distance/length
            latency_expr: Gurobi expression for total latency (distance-based)
        """
        # Precompute lookups
        cost_map = build_cost_map(cable_types)
        latency_map = build_latency_map(cable_types)
        sensor_lookup = build_sensor_lookup(sensors)
        actuator_lookup = build_actuator_lookup(actuators)
        
        # Precompute cable cost, distance, and latency coefficients (only non-zero)
        cable_cost_terms = []  # List of (variable, cable_cost) tuples
        cable_distance_terms = []  # List of (variable, distance) tuples
        latency_terms = []  # List of (variable, latency) tuples
        
        n_sc = len(scs)
        shared_bus_ifaces = {'CAN', 'LIN', 'FLEXRAY'}

        for i in range(n_sc):
            sc = scs[i]
            for j in feasible_locs_per_sc[i]:
                cable_cost = 0.0
                cable_distance = 0.0
                latency = 0.0

                # Sum sensor cable costs, distances, and latencies
                for s_id in (sc.sensors or []):
                    sensor = sensor_lookup.get(s_id)
                    # If ETH attachments are modeled, ETH sensors are wired to their attachment point,
                    # not directly to the ECU hosting the SWC.
                    if attach_s is not None and sensor and sensor.interface == "ETH":
                        continue
                    # Shared bus interfaces are charged via shared attachment vars (single physical cable).
                    if shared_attach_s is not None and sensor and sensor.interface in shared_bus_ifaces:
                        continue
                    if sensor and sensor.location:
                        dist = get_distance(sensor.location, locations[j].location)
                        cable_cost += dist * cost_map.get(sensor.interface, 0.0)
                        cable_distance += dist
                        latency += dist * latency_map.get(sensor.interface, 0.0)

                # Sum actuator cable costs, distances, and latencies
                for a_id in (sc.actuators or []):
                    actuator = actuator_lookup.get(a_id)
                    # If ETH attachments are modeled, ETH actuators are wired to their attachment point.
                    if attach_a is not None and actuator and actuator.interface == "ETH":
                        continue
                    # Shared bus interfaces are charged via shared attachment vars (single physical cable).
                    if shared_attach_a is not None and actuator and actuator.interface in shared_bus_ifaces:
                        continue
                    if actuator and actuator.location:
                        dist = get_distance(actuator.location, locations[j].location)
                        cable_cost += dist * cost_map.get(actuator.interface, 0.0)
                        cable_distance += dist
                        latency += dist * latency_map.get(actuator.interface, 0.0)

                x_ij_expr = x[i, j]

                # Only add to expression if values are non-zero
                if cable_cost > 0:
                    cable_cost_terms.append((x_ij_expr, cable_cost))
                if cable_distance > 0:
                    cable_distance_terms.append((x_ij_expr, cable_distance))
                if latency > 0:
                    latency_terms.append((x_ij_expr, latency))

        # ECU-to-ECU/Location communication backbone (LEGO model with sparse comm variables)
        comm_cost_terms = []
        comm_distance_terms = []
        comm_latency_terms = []

        if comm:  # LEGO model: sparse communication variables
            for (j1, j2, iface), comm_var in comm.items():
                dist = get_distance(locations[j1].location, locations[j2].location)
                cost = dist * cable_types[iface].cost_per_meter
                lat = dist * latency_map.get(iface, 0.0)
                
                comm_cost_terms.append((comm_var, cost))
                comm_distance_terms.append((comm_var, dist))
                comm_latency_terms.append((comm_var, lat))

        # ETH sensor/actuator attachment cables (only when attachments are modeled)
        if attach_s:
            for (si, j), av in attach_s.items():
                s = sensors[si]
                if not getattr(s, 'location', None):
                    continue
                dist = get_distance(s.location, locations[j].location)
                iface = getattr(s, 'interface', 'ETH')
                cable_cost_terms.append((av, dist * cost_map.get(iface, 0.0)))
                cable_distance_terms.append((av, dist))
                latency_terms.append((av, dist * latency_map.get(iface, 0.0)))

        if attach_a:
            for (ai, j), av in attach_a.items():
                a = actuators[ai]
                if not getattr(a, 'location', None):
                    continue
                dist = get_distance(a.location, locations[j].location)
                iface = getattr(a, 'interface', 'ETH')
                cable_cost_terms.append((av, dist * cost_map.get(iface, 0.0)))
                cable_distance_terms.append((av, dist))
                latency_terms.append((av, dist * latency_map.get(iface, 0.0)))

        # Shared-bus (CAN/LIN/FLEXRAY) sensor/actuator attachment cables
        if shared_attach_s:
            for (si, j), av in shared_attach_s.items():
                s = sensors[si]
                if not getattr(s, 'location', None):
                    continue
                iface = getattr(s, 'interface', None)
                if iface not in shared_bus_ifaces:
                    continue
                dist = get_distance(s.location, locations[j].location)
                cable_cost_terms.append((av, dist * cost_map.get(iface, 0.0)))
                cable_distance_terms.append((av, dist))
                latency_terms.append((av, dist * latency_map.get(iface, 0.0)))

        if shared_attach_a:
            for (ai, j), av in shared_attach_a.items():
                aobj = actuators[ai]
                if not getattr(aobj, 'location', None):
                    continue
                iface = getattr(aobj, 'interface', None)
                if iface not in shared_bus_ifaces:
                    continue
                dist = get_distance(aobj.location, locations[j].location)
                cable_cost_terms.append((av, dist * cost_map.get(iface, 0.0)))
                cable_distance_terms.append((av, dist))
                latency_terms.append((av, dist * latency_map.get(iface, 0.0)))

        # Shared-bus trunk lengths (count shared main segment once per location/interface)
        if shared_trunk_len:
            for (j, iface), tvar in shared_trunk_len.items():
                cable_cost_terms.append((tvar, cost_map.get(iface, 0.0)))
                cable_distance_terms.append((tvar, 1.0))
                latency_terms.append((tvar, latency_map.get(iface, 0.0)))

        # Extra shared-bus trunks for ASIL-group split instances
        if shared_extra_trunk_len:
            for (j, iface), tvar in shared_extra_trunk_len.items():
                cable_cost_terms.append((tvar, cost_map.get(iface, 0.0)))
                cable_distance_terms.append((tvar, 1.0))
                latency_terms.append((tvar, latency_map.get(iface, 0.0)))
        
        # Create compact linear expressions
        if cable_cost_terms:
            cable_cost_expr = gp.quicksum(var * cost for var, cost in cable_cost_terms)
        else:
            cable_cost_expr = 0
        if comm_cost_terms:
            cable_cost_expr += gp.quicksum(var * cost for var, cost in comm_cost_terms)
            
        if cable_distance_terms:
            cable_distance_expr = gp.quicksum(var * dist for var, dist in cable_distance_terms)
        else:
            cable_distance_expr = 0
        if comm_distance_terms:
            cable_distance_expr += gp.quicksum(var * dist for var, dist in comm_distance_terms)
            
        if latency_terms:
            latency_expr = gp.quicksum(var * lat for var, lat in latency_terms)
        else:
            latency_expr = 0
        if comm_latency_terms:
            latency_expr += gp.quicksum(var * lat for var, lat in comm_latency_terms)
            
        return cable_cost_expr, cable_distance_expr, latency_expr
    
    def _create_model_and_variables(self, n_sc, n_locs, scs, locations, unique_asils, all_hw, sensors, actuators, all_interfaces, infeasible_ij=None, comm_matrix=None, partitions=None, backbone=None):
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

        # Shared-bus peripheral attachment variables (CAN/LIN/FLEXRAY):
        # shared_attach_s[s_idx, j] = 1 if non-ETH shared-bus sensor s is physically wired to location j.
        # shared_attach_a[a_idx, j] = 1 if non-ETH shared-bus actuator a is physically wired to location j.
        shared_bus_ifaces = {'CAN', 'LIN', 'FLEXRAY'}
        shared_sensor_indices = [si for si, s in enumerate(sensors) if getattr(s, 'interface', None) in shared_bus_ifaces]
        shared_actuator_indices = [ai for ai, a in enumerate(actuators) if getattr(a, 'interface', None) in shared_bus_ifaces]

        shared_attach_s = {}
        for si in shared_sensor_indices:
            for j in range(n_locs):
                shared_attach_s[si, j] = model.addVar(vtype=GRB.BINARY, name=f"sharedAttachS_{sensors[si].id}_{locations[j].id}")

        shared_attach_a = {}
        for ai in shared_actuator_indices:
            for j in range(n_locs):
                shared_attach_a[ai, j] = model.addVar(vtype=GRB.BINARY, name=f"sharedAttachA_{actuators[ai].id}_{locations[j].id}")

        # Shared-bus trunk length variables:
        # shared_trunk_len[j, iface] is the main shared segment length at location j for iface.
        shared_trunk_len = {}
        for j in range(n_locs):
            for iface in shared_bus_ifaces:
                shared_trunk_len[j, iface] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"sharedTrunkLen_{locations[j].id}_{iface}")

        # Additional trunk length variables for ASIL-group split shared buses.
        shared_extra_trunk_len = {}
        for j in range(n_locs):
            for iface in shared_bus_ifaces:
                shared_extra_trunk_len[j, iface] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"sharedExtraTrunkLen_{locations[j].id}_{iface}")
        
        # comm[j1,j2,'ETH'] = number of backbone cables between sites j1 and j2 (integer). Every site pair may carry
        # a cable: the candidate paths below use direct links and the two links of a relaying site.
        comm = {}
        for j1 in range(n_locs):
            for j2 in range(j1 + 1, n_locs):
                comm[j1, j2, "ETH"] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"comm_{locations[j1].id}_{locations[j2].id}_ETH")

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
                        'volume': vol
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
                })
                t_idx += 1

        # --- BACKBONE: aggregated traffic per site pair on candidate paths ---
        # lnk[t,u,v] (continuous, 0..1) is forced to 1 when flow t starts at site u and ends at site v; it only
        # consumes link capacity, so the LP sets it to max(0, start + end - 1). The per-flow routing binaries of v1
        # are gone: all traffic of a site pair takes one of the pair's candidate paths.
        k_paths = max(1, int((backbone or {}).get('candidate_paths', 3)))
        # lazy_traffic (configs/buses.json): no lnk / f variables; the traffic requirements are enforced as lazy
        # constraints in a callback (see optimize()). The LP is small, the relaxation does not see the traffic.
        lazy = bool((backbone or {}).get('lazy_traffic', False))
        lnk, lnk_terms, pair_vol, flow_ends = {}, [], {}, {}
        for tr in traffic_flows:
            if tr['type'] == 'SC_SC':
                src = {u: x[tr['src_idx'], u] for u in feasible_locs_per_sc[tr['src_idx']]}
                dst = {v: x[tr['dst_idx'], v] for v in feasible_locs_per_sc[tr['dst_idx']]}
            elif tr['type'] == 'SENS_SC':
                src = {u: attach_s[tr['sensor_idx'], u] for u in range(n_locs) if (tr['sensor_idx'], u) in attach_s}
                dst = {v: x[tr['dst_idx'], v] for v in feasible_locs_per_sc[tr['dst_idx']]}
            else:  # SC_ACT
                src = {u: x[tr['src_idx'], u] for u in feasible_locs_per_sc[tr['src_idx']]}
                dst = {v: attach_a[tr['act_idx'], v] for v in range(n_locs) if (tr['act_idx'], v) in attach_a}
            flow_ends[tr['id']] = (src, dst)
            for u, sv in src.items():
                for v, dv in dst.items():
                    if u == v:
                        continue
                    if lazy:
                        pair_vol.setdefault((u, v), []).append((None, float(tr['volume'])))
                        continue
                    lv = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"lnk_{tr['id']}_{locations[u].id}_{locations[v].id}")
                    lnk[tr['id'], u, v] = lv
                    lnk_terms.append((lv, sv, dv))
                    pair_vol.setdefault((u, v), []).append((lv, float(tr['volume'])))

        # Candidate paths of every site pair with possible traffic: the direct link and the (k-1) shortest routes
        # over one relaying site (which then needs a SWITCH). sel[u,v,p] selects the path of the pair, f[u,v,p,d]
        # is the volume routed on it in direction d (0: u->v, 1: v->u).
        site_dist = {(u, v): get_distance(locations[u].location, locations[v].location)
                     for u in range(n_locs) for v in range(n_locs) if u != v}
        paths, sel, f = {}, {}, {}
        for (u, v) in sorted({(min(a, b), max(a, b)) for (a, b) in pair_vol}):
            relays = sorted((w for w in range(n_locs) if w not in (u, v)), key=lambda w: site_dist[u, w] + site_dist[w, v])
            paths[u, v] = [[u, v]] + [[u, w, v] for w in relays[:k_paths - 1]]
            for p in range(len(paths[u, v])):
                sel[u, v, p] = model.addVar(vtype=GRB.BINARY, name=f"sel_{locations[u].id}_{locations[v].id}_p{p}")
                for d in ((0, 1) if not lazy else ()):
                    f[u, v, p, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"f_{locations[u].id}_{locations[v].id}_p{p}_d{d}")
        net = {'paths': paths, 'sel': sel, 'f': f, 'lnk': lnk, 'lnk_terms': lnk_terms, 'pair_vol': pair_vol, 'flow_ends': flow_ends,
               'k_paths': k_paths, 'lazy': lazy, 'volumes': {tr['id']: float(tr['volume']) for tr in traffic_flows},
               'conservation_cut': bool((backbone or {}).get('conservation_cut', False)),
               'path_link_cut': bool((backbone or {}).get('path_link_cut', False))}

        var_stats = {
            'x': len(x),
            'P': len(P),
            'y': len(y),
            'hw_use': len(hw_use),
            'if_use': len(if_use),
            'attach_s': len(attach_s),
            'attach_a': len(attach_a),
            'shared_attach_s': len(shared_attach_s),
            'shared_attach_a': len(shared_attach_a),
            'shared_trunk_len': len(shared_trunk_len),
            'shared_extra_trunk_len': len(shared_extra_trunk_len),
            'comm': len(comm),
            'traffic_flows': len(traffic_flows),
            'lnk': len(lnk),
            'sel': len(sel),
            'f': len(f),
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
            + var_stats['shared_attach_s']
            + var_stats['shared_attach_a']
            + var_stats['shared_trunk_len']
            + var_stats['shared_extra_trunk_len']
            + var_stats['comm']
            + var_stats['lnk']
            + var_stats['sel']
            + var_stats['f']
        )

        return model, x, P, y, hw_use, if_use, attach_s, attach_a, shared_attach_s, shared_attach_a, shared_trunk_len, shared_extra_trunk_len, comm, traffic_flows, net, feasible_locs_per_sc, feasible_scs_per_loc, var_stats

    def _add_constraints(self, model, x, P, y, hw_use, if_use, attach_s, attach_a, shared_attach_s, shared_attach_a, shared_trunk_len, shared_extra_trunk_len, comm, traffic_flows, net, scs, locations, sensors, actuators, 
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
            'c2_capacity': 0,
            'c3_hw_required': 0,
            'c4_if_required': 0,
            'c4_asil_bus_split': 0,
            'c4_shared_bus_bw': 0,
            'c5_redundancy': 0,
            'c6_ai_contention': 0,
            'c8_loc_loc_latency': 0,
            'c9_if_activation': 0,
            'c9_attach': 0,
            'c9_shared_attach': 0,
            'c9_shared_trunk': 0,
            'c10_pair_traffic': 0,
            'c10_path_choice': 0,
            'c10_relay_switch': 0,
            'c10_link_capacity': 0,
        }
        
        def x_expr(i, j):
            return x.get((i, j), 0)

        # Uncertainty-aware constraint blocks: each one switched on separately (see configs/uncertainty.json)
        unc = uncertainty or {}
        unc_params = unc.get('params', {})
        print(f"  Uncertainty blocks: " + ", ".join(f"{k}={bool(unc.get(k))}" for k in UNCERTAINTY_BLOCKS))

        # SW-based uncertainty (selective static margin, configs/uncertainty.json: sw_margin)
        def get_sw_multiplier(sc):
            return sw_multiplier(sc, unc)

        # Data-derived bounds (replace arbitrary Big-M values)
        # comm_ub: a backbone link never carries more than the total traffic volume, and extra parallel links only
        # add cost; 2 keeps the "switch-only node has degree >= 2" rule satisfiable with one neighbour.
        eth_capacity = float(cable_types['ETH'].capacity)
        total_traffic_volume = sum(float(tr['volume']) for tr in traffic_flows)
        comm_ub = max(2, int(np.ceil(total_traffic_volume / eth_capacity)))
        for comm_var in comm.values():
            comm_var.UB = comm_ub
        print(f"  Backbone link upper bound: comm <= {comm_ub} (total traffic {total_traffic_volume:.1f}, ETH capacity {eth_capacity:.1f})")

        # max_periph_dist[j, iface]: farthest peripheral of a shared bus from location j (upper bound of its trunk length)
        max_periph_dist = {}
        for j in range(n_locs):
            for iface in ('CAN', 'LIN', 'FLEXRAY'):
                dists = [get_distance(d.location, locations[j].location)
                         for d in list(sensors) + list(actuators)
                         if getattr(d, 'interface', None) == iface and getattr(d, 'location', None) is not None]
                max_periph_dist[j, iface] = max(dists, default=0.0)

        def asil_safety_group(sc):
            a = getattr(sc, 'asil_req', 0)
            if isinstance(a, str):
                a_norm = a.strip().upper()
                return 'HIGH' if a_norm in {'C', 'D'} else 'LOW'
            return 'HIGH' if int(a) >= 3 else 'LOW'

        # Constraint 1a: every SC is hosted at exactly one site
        for i in range(n_sc):
            model.addConstr(gp.quicksum(x[i, j] for j in feasible_locs_per_sc[i]) == 1, name=f"unique_site_{scs[i].id}")
            cstats['c1a_unique_site'] += 1

        # Constraint 1b: a hosting site is active, and an active site hosts at least one SC (no ghost sites)
        for j in range(n_locs):
            for i in feasible_scs_per_loc[j]:
                model.addConstr(x[i, j] <= y[j], name=f"x_le_y_{scs[i].id}_{locations[j].id}")
                cstats['c1b_x_le_y'] += 1
            model.addConstr(y[j] <= gp.quicksum(x[i, j] for i in feasible_scs_per_loc[j]), name=f"y_le_sum_x_{locations[j].id}")
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
                    P[j, a] <= gp.quicksum(x[i, j] for i in feasible_scs_per_loc[j] if scs[i].asil_req == a),
                    name=f"partitions_le_scs_{locations[j].id}_asil{a}"
                )
                cstats['c1c_partitions_le_scs'] += 1

        # Constraint 1e: a hosted SC needs a partition of its ASIL at its site
        for (i, j), xv in x.items():
            model.addConstr(xv <= P[j, scs[i].asil_req], name=f"x_le_P_{scs[i].id}_{locations[j].id}")
            cstats['c1e_x_le_P'] += 1

        # Constraint 1d: per-site caps of the control unit (configs/partitions.json: site.max_partitions,
        # site.cpu_cap / ram_cap / rom_cap; null = unlimited, which reproduces the v1 model)
        site_max_p = partitions.get('site_max_partitions')
        for j in range(n_locs):
            p_sum = gp.quicksum(P[j, a] for a in unique_asils)
            if site_max_p is not None:
                model.addConstr(p_sum <= float(site_max_p) * y[j], name=f"site_max_partitions_{locations[j].id}")
                cstats['c1d_site_caps'] += 1
            for res in ('cpu', 'ram', 'rom'):
                site_cap = partitions.get(f'site_{res}_cap')
                part_cap = float(partitions.get(f'{res}_cap', 0) or 0)
                if site_cap is not None and part_cap > 0:
                    model.addConstr(p_sum * part_cap <= float(site_cap) * y[j], name=f"site_{res}_cap_{locations[j].id}")
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
                    )
                    model.addConstr(
                        demand <= P[j, a] * partitions.get(f'{res}_cap', float('inf')) * H_j,
                        name=f"{res}_capacity_{locations[j].id}_asil{a}"
                    )
                    cstats['c2_capacity'] += 1

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
        shared_bus_ifaces = {'CAN', 'LIN', 'FLEXRAY'}
        
        for j in range(n_locs):
            for i_name in all_interfaces:
                comm_port_demand = gp.LinExpr()
                for (j1, j2, iface), comm_var in comm.items():
                    if (j1 == j) and (iface == i_name):
                        comm_port_demand.add(comm_var, 1)
                    elif (j2 == j) and (iface == i_name):
                        comm_port_demand.add(comm_var, 1)

                # 4a. Shared-bus interfaces (CAN/LIN/FLEXRAY):
                # A location needs only one interface endpoint if at least one assigned SC at this
                # location uses a peripheral on that bus.
                if i_name in shared_bus_ifaces:
                    low_group_use = model.addVar(vtype=GRB.BINARY, name=f"busGroupLow_{locations[j].id}_{i_name}")
                    high_group_use = model.addVar(vtype=GRB.BINARY, name=f"busGroupHigh_{locations[j].id}_{i_name}")
                    split_active = model.addVar(vtype=GRB.BINARY, name=f"busGroupSplit_{locations[j].id}_{i_name}")

                    for i in feasible_scs_per_loc[j]:
                        sc = scs[i]
                        uses_iface = any(
                            s_lookup.get(s_id) and s_lookup.get(s_id).interface == i_name
                            for s_id in (sc.sensors or [])
                        ) or any(
                            a_lookup.get(a_id) and a_lookup.get(a_id).interface == i_name
                            for a_id in (sc.actuators or [])
                        )
                        if uses_iface:
                            grp = asil_safety_group(sc)
                            bus_assign_var = model.addVar(
                                vtype=GRB.BINARY,
                                name=f"busAssign_{sc.id}_{locations[j].id}_{i_name}_{grp}"
                            )

                            # Explicit mapping: if SC is placed at location j, it is assigned to
                            # its ASIL-group candidate bus instance for this shared interface.
                            model.addConstr(
                                bus_assign_var == x_expr(i, j),
                                name=f"shared_bus_assign_eq_{locations[j].id}_{i_name}_{sc.id}"
                            )

                            model.addConstr(
                                if_use[j, i_name] >= x_expr(i, j),
                                name=f"shared_bus_if_use_{locations[j].id}_{i_name}_{sc.id}"
                            )

                            if grp == 'HIGH':
                                model.addConstr(
                                    high_group_use >= x_expr(i, j),
                                    name=f"shared_bus_group_high_{locations[j].id}_{i_name}_{sc.id}"
                                )
                                model.addConstr(
                                    bus_assign_var <= high_group_use,
                                    name=f"shared_bus_assign_to_high_{locations[j].id}_{i_name}_{sc.id}"
                                )
                            else:
                                model.addConstr(
                                    low_group_use >= x_expr(i, j),
                                    name=f"shared_bus_group_low_{locations[j].id}_{i_name}_{sc.id}"
                                )
                                model.addConstr(
                                    bus_assign_var <= low_group_use,
                                    name=f"shared_bus_assign_to_low_{locations[j].id}_{i_name}_{sc.id}"
                                )
                            cstats['c4_asil_bus_split'] += 1

                    # Enforce ASIL 0/1/2 and ASIL 3/4 segregation on shared buses.
                    # If both groups are present on the same location+interface, at least two bus instances are required.
                    model.addConstr(
                        if_use[j, i_name] >= low_group_use + high_group_use,
                        name=f"shared_bus_asil_split_count_{locations[j].id}_{i_name}"
                    )
                    cstats['c4_asil_bus_split'] += 1

                    # split_active = 1 iff both low and high groups are active on this shared bus.
                    model.addConstr(split_active <= low_group_use, name=f"shared_bus_split_le_low_{locations[j].id}_{i_name}")
                    model.addConstr(split_active <= high_group_use, name=f"shared_bus_split_le_high_{locations[j].id}_{i_name}")
                    model.addConstr(split_active >= low_group_use + high_group_use - 1, name=f"shared_bus_split_ge_sum_{locations[j].id}_{i_name}")
                    cstats['c4_asil_bus_split'] += 3

                    model.addConstr(
                        if_use[j, i_name] >= comm_port_demand,
                        name=f"shared_bus_comm_port_count_{locations[j].id}_{i_name}"
                    )

                    # Shared-bus bandwidth capacity at this location/interface.
                    # Without this, CAN/LIN/FLEXRAY trunks can be over-subscribed.
                    cap_obj = cable_types.get(i_name)
                    cap_val = getattr(cap_obj, 'capacity', None) if cap_obj is not None else None
                    try:
                        cap_val = float(cap_val) if cap_val is not None else None
                    except Exception:
                        cap_val = None

                    if cap_val is not None and cap_val > 0 and cap_val != float('inf'):
                        shared_bus_load = gp.LinExpr()
                        if shared_attach_s:
                            for (si, jj), av in shared_attach_s.items():
                                if jj != j:
                                    continue
                                if getattr(sensors[si], 'interface', None) != i_name:
                                    continue
                                vol = float(getattr(sensors[si], 'volume', 0.0) or 0.0)
                                if vol > 0:
                                    shared_bus_load.add(av, vol)
                        if shared_attach_a:
                            for (ai, jj), av in shared_attach_a.items():
                                if jj != j:
                                    continue
                                if getattr(actuators[ai], 'interface', None) != i_name:
                                    continue
                                vol = float(getattr(actuators[ai], 'volume', 0.0) or 0.0)
                                if vol > 0:
                                    shared_bus_load.add(av, vol)

                        model.addConstr(
                            shared_bus_load <= cap_val * if_use[j, i_name],
                            name=f"shared_bus_bw_{locations[j].id}_{i_name}"
                        )
                        cstats['c4_shared_bus_bw'] += 1

                    # If ASIL split is active, second shared trunk length equals main trunk length.
                    if (j, i_name) in shared_extra_trunk_len and (j, i_name) in shared_trunk_len:
                        split_m = max_periph_dist[j, i_name]
                        model.addConstr(
                            shared_extra_trunk_len[j, i_name] <= split_active * split_m,
                            name=f"shared_extra_trunk_gate_{locations[j].id}_{i_name}"
                        )
                        model.addConstr(
                            shared_extra_trunk_len[j, i_name] <= shared_trunk_len[j, i_name],
                            name=f"shared_extra_trunk_le_main_{locations[j].id}_{i_name}"
                        )
                        model.addConstr(
                            shared_extra_trunk_len[j, i_name] >= shared_trunk_len[j, i_name] - split_m * (1 - split_active),
                            name=f"shared_extra_trunk_eq_main_if_split_{locations[j].id}_{i_name}"
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
        
        # Constraint 6/7 (Sensor/Actuator max-latency): precomputed and enforced via sparse z creation.
        lat_cfg = unc_params.get('robust_latency', {})
        latency_map = build_latency_map(cable_types)

        def robust_link_latency(iface, dist):
            nominal = dist * latency_map.get(iface, 0.0)
            if not unc.get('robust_latency'):
                return nominal
            NETWORK_SAFETY_FACTOR = float(lat_cfg['network_safety_factor'])
            SYNC_ERROR_CONSTANT = float(lat_cfg['sync_error_constant'])
            PLATFORM_OVERHEAD = float(lat_cfg['platform_overhead'])
            return (nominal * NETWORK_SAFETY_FACTOR) + SYNC_ERROR_CONSTANT + PLATFORM_OVERHEAD
        
        # Constraint 8: Location-Location Communication Latency Constraints
        if comm_matrix:
            sc_id_map = {sc.id: i for i, sc in enumerate(scs)}
            for link in comm_matrix:
                src_id = link.get('src')
                dst_id = link.get('dst')
                max_lat = link.get('max_latency', float('inf'))
                
                if src_id not in sc_id_map or dst_id not in sc_id_map:
                    continue
                
                u = sc_id_map[src_id]
                v = sc_id_map[dst_id]
                
                for j1 in feasible_locs_per_sc[u]:
                    for j2 in feasible_locs_per_sc[v]:
                        if j1 == j2:
                            continue

                        dist = get_distance(locations[j1].location, locations[j2].location)
                        robust_lat = robust_link_latency('ETH', dist)
                        if robust_lat > max_lat:
                            model.addConstr(
                                x_expr(u, j1)
                                + x_expr(v, j2)
                                <= 1,
                                name=f"net_uncertainty_{u}_{v}_{j1}_{j2}"
                            )
                            cstats['c8_loc_loc_latency'] += 1
        
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
        M = comm_ub  # Big-M value: comm can never exceed its upper bound
        all_interfaces = list(cable_types.keys())

        # Build an "active node" expression per location.
        if 'SWITCH' in all_hw:
            loc_active = {j: y[j] + hw_use[j, 'SWITCH'] for j in range(n_locs)}
        else:
            loc_active = {j: 1.0 * y[j] for j in range(n_locs)}

        # Constraint 9-ATT: ETH sensor/actuator attachments must connect to an active node,
        # and must respect peripheral max-latency on the physical cable to that attachment point.
        # (We do not model end-to-end network latency here; this only constrains the local cable.)
        latency_map = build_latency_map(cable_types)
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

                max_lat = getattr(sensors[si], 'max_latency', None)
                if max_lat is not None and getattr(sensors[si], 'location', None) is not None:
                    dist = get_distance(sensors[si].location, locations[j].location)
                    lat = robust_link_latency('ETH', dist)
                    if lat > max_lat:
                        model.addConstr(av == 0, name=f"attachS_lat_infeas_{sensors[si].id}_{locations[j].id}")
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

                max_lat = getattr(actuators[ai], 'max_latency', None)
                if max_lat is not None and getattr(actuators[ai], 'location', None) is not None:
                    dist = get_distance(actuators[ai].location, locations[j].location)
                    lat = robust_link_latency('ETH', dist)
                    if lat > max_lat:
                        model.addConstr(av == 0, name=f"attachA_lat_infeas_{actuators[ai].id}_{locations[j].id}")
                        cstats['c9_attach'] += 1

            for ai in {k[0] for k in attach_a.keys()}:
                model.addConstr(
                    gp.quicksum(attach_a[ai, j] for j in range(n_locs) if (ai, j) in attach_a) == 1,
                    name=f"attachA_one_{actuators[ai].id}"
                )
                cstats['c9_attach'] += 1

        # Constraint 9-SHARED-ATT: CAN/LIN/FLEXRAY peripheral attachments
        # - Must connect to an active location
        # - Must satisfy local max-latency for its own interface
        # - A peripheral is physically attached to at most one location
        shared_bus_ifaces = {'CAN', 'LIN', 'FLEXRAY'}
        if shared_attach_s:
            for (si, j), av in shared_attach_s.items():
                iface = getattr(sensors[si], 'interface', None)
                if iface not in shared_bus_ifaces:
                    model.addConstr(av == 0, name=f"sharedAttachS_iface_inactive_{sensors[si].id}_{locations[j].id}")
                    cstats['c9_shared_attach'] += 1
                    continue
                model.addConstr(av <= loc_active[j], name=f"sharedAttachS_active_{sensors[si].id}_{locations[j].id}")
                cstats['c9_shared_attach'] += 1
                max_lat = getattr(sensors[si], 'max_latency', None)
                if max_lat is not None and getattr(sensors[si], 'location', None) is not None:
                    dist = get_distance(sensors[si].location, locations[j].location)
                    lat = robust_link_latency(iface, dist)
                    if lat > max_lat:
                        model.addConstr(av == 0, name=f"sharedAttachS_lat_infeas_{sensors[si].id}_{locations[j].id}")
                        cstats['c9_shared_attach'] += 1

            for si in {k[0] for k in shared_attach_s.keys()}:
                model.addConstr(
                    gp.quicksum(shared_attach_s[si, j] for j in range(n_locs) if (si, j) in shared_attach_s) <= 1,
                    name=f"sharedAttachS_one_{sensors[si].id}"
                )
                cstats['c9_shared_attach'] += 1

        if shared_attach_a:
            for (ai, j), av in shared_attach_a.items():
                iface = getattr(actuators[ai], 'interface', None)
                if iface not in shared_bus_ifaces:
                    model.addConstr(av == 0, name=f"sharedAttachA_iface_inactive_{actuators[ai].id}_{locations[j].id}")
                    cstats['c9_shared_attach'] += 1
                    continue
                model.addConstr(av <= loc_active[j], name=f"sharedAttachA_active_{actuators[ai].id}_{locations[j].id}")
                cstats['c9_shared_attach'] += 1
                max_lat = getattr(actuators[ai], 'max_latency', None)
                if max_lat is not None and getattr(actuators[ai], 'location', None) is not None:
                    dist = get_distance(actuators[ai].location, locations[j].location)
                    lat = robust_link_latency(iface, dist)
                    if lat > max_lat:
                        model.addConstr(av == 0, name=f"sharedAttachA_lat_infeas_{actuators[ai].id}_{locations[j].id}")
                        cstats['c9_shared_attach'] += 1

            for ai in {k[0] for k in shared_attach_a.keys()}:
                model.addConstr(
                    gp.quicksum(shared_attach_a[ai, j] for j in range(n_locs) if (ai, j) in shared_attach_a) <= 1,
                    name=f"sharedAttachA_one_{actuators[ai].id}"
                )
                cstats['c9_shared_attach'] += 1

        # Link shared peripheral attachments to SC placements:
        # if SC i uses a shared-bus peripheral and is placed at location j, that peripheral must be attached at j.
        sensor_id_to_idx = {s.id: si for si, s in enumerate(sensors)}
        actuator_id_to_idx = {a.id: ai for ai, a in enumerate(actuators)}

        for i in range(n_sc):
            sc = scs[i]
            for s_id in (getattr(sc, 'sensors', None) or []):
                si = sensor_id_to_idx.get(s_id)
                if si is None:
                    continue
                iface = getattr(sensors[si], 'interface', None)
                if iface not in shared_bus_ifaces:
                    continue
                for j in feasible_locs_per_sc[i]:
                    if (si, j) in shared_attach_s:
                        model.addConstr(
                            x_expr(i, j) <= shared_attach_s[si, j],
                            name=f"sharedAttachS_link_{sc.id}_{sensors[si].id}_{locations[j].id}"
                        )
                        cstats['c9_shared_attach'] += 1

            for a_id in (getattr(sc, 'actuators', None) or []):
                ai = actuator_id_to_idx.get(a_id)
                if ai is None:
                    continue
                iface = getattr(actuators[ai], 'interface', None)
                if iface not in shared_bus_ifaces:
                    continue
                for j in feasible_locs_per_sc[i]:
                    if (ai, j) in shared_attach_a:
                        model.addConstr(
                            x_expr(i, j) <= shared_attach_a[ai, j],
                            name=f"sharedAttachA_link_{sc.id}_{actuators[ai].id}_{locations[j].id}"
                        )
                        cstats['c9_shared_attach'] += 1

        # Constraint 9-SHARED-TRUNK:
        # shared_trunk_len[j, iface] is at least the farthest attached peripheral distance
        # for that location/interface, so the common trunk segment is counted once.
        if shared_trunk_len:
            for (j, iface), trunk_var in shared_trunk_len.items():
                if iface not in shared_bus_ifaces:
                    model.addConstr(trunk_var == 0.0, name=f"sharedTrunk_iface_zero_{locations[j].id}_{iface}")
                    cstats['c9_shared_trunk'] += 1
                    continue

                has_endpoint_expr = gp.LinExpr()

                if shared_attach_s:
                    for (si, jj), av in shared_attach_s.items():
                        if jj != j:
                            continue
                        s_iface = getattr(sensors[si], 'interface', None)
                        if s_iface != iface:
                            continue
                        dist = get_distance(sensors[si].location, locations[j].location) if getattr(sensors[si], 'location', None) is not None else 0.0
                        model.addConstr(
                            trunk_var >= dist * av,
                            name=f"sharedTrunk_lb_s_{locations[j].id}_{iface}_{sensors[si].id}"
                        )
                        cstats['c9_shared_trunk'] += 1
                        has_endpoint_expr.add(av, 1.0)

                if shared_attach_a:
                    for (ai, jj), av in shared_attach_a.items():
                        if jj != j:
                            continue
                        a_iface = getattr(actuators[ai], 'interface', None)
                        if a_iface != iface:
                            continue
                        dist = get_distance(actuators[ai].location, locations[j].location) if getattr(actuators[ai], 'location', None) is not None else 0.0
                        model.addConstr(
                            trunk_var >= dist * av,
                            name=f"sharedTrunk_lb_a_{locations[j].id}_{iface}_{actuators[ai].id}"
                        )
                        cstats['c9_shared_trunk'] += 1
                        has_endpoint_expr.add(av, 1.0)

                # If no shared endpoint is attached at this location/interface, trunk must be zero.
                # M is bounded by max possible distance in the vehicle envelope for this location.
                max_dist_loc = 0.0
                for s in sensors:
                    if getattr(s, 'interface', None) == iface and getattr(s, 'location', None) is not None:
                        max_dist_loc = max(max_dist_loc, get_distance(s.location, locations[j].location))
                for a in actuators:
                    if getattr(a, 'interface', None) == iface and getattr(a, 'location', None) is not None:
                        max_dist_loc = max(max_dist_loc, get_distance(a.location, locations[j].location))

                model.addConstr(
                    trunk_var <= max_dist_loc * has_endpoint_expr,
                    name=f"sharedTrunk_zero_if_no_ep_{locations[j].id}_{iface}"
                )
                cstats['c9_shared_trunk'] += 1

                # Extra split trunk length is bounded by the same physical envelope.
                if shared_extra_trunk_len and (j, iface) in shared_extra_trunk_len:
                    model.addConstr(
                        shared_extra_trunk_len[j, iface] <= trunk_var,
                        name=f"sharedExtraTrunk_le_main_{locations[j].id}_{iface}"
                    )
                    cstats['c9_shared_trunk'] += 1
            
        for (j1, j2, iface), comm_var in comm.items():
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
        

        
        # C-NET: aggregated backbone traffic on candidate paths (replaces the per-flow routing binaries of v1)
        paths, sel, f, lnk, pair_vol = net['paths'], net['sel'], net['f'], net['lnk'], net['pair_vol']
        eth_cap = float(cable_types['ETH'].capacity)
        has_switch = 'SWITCH' in all_hw

        # C-NET-1: lnk[t,u,v] >= start_t(u) + end_t(v) - 1
        for n_l, (lv, sv, dv) in enumerate(net['lnk_terms']):
            model.addConstr(lv >= sv + dv - 1, name=f"lnk_{n_l}")
            cstats['c10_pair_traffic'] += 1

        # C-NET-1b (optional, configs/buses.json: backbone.conservation_cut): aggregated conservation, valid and
        # raising the LP bound (root 7,411 -> 7,482 at 10 cells) but the heavier LP made the search slower in the
        # tests of 2026-10-19 (10 cells: 380 s -> time limit); off by default.
        for t, (src, dst) in (net['flow_ends'].items() if (net.get('conservation_cut') and not net['lazy']) else ()):
            for u, sv in src.items():
                outs = [lnk[t, u, v] for v in dst if v != u]
                if outs:
                    model.addConstr(gp.quicksum(outs) >= sv - dst.get(u, 0), name=f"lnk_out_{t}_{locations[u].id}")
                    cstats['c10_pair_traffic'] += 1
            for v, dv in dst.items():
                ins = [lnk[t, u, v] for u in src if u != v]
                if ins:
                    model.addConstr(gp.quicksum(ins) >= dv - src.get(v, 0), name=f"lnk_in_{t}_{locations[v].id}")
                    cstats['c10_pair_traffic'] += 1

        # C-NET-2: the traffic of a site pair is routed on exactly one of its candidate paths (both directions)
        arc_load = {}
        lazy = net['lazy']
        for (u, v), plist in paths.items():
            for p, path in enumerate(plist):
                for d in ((0, 1) if not lazy else ()):
                    seq = path if d == 0 else path[::-1]
                    for a, b in zip(seq, seq[1:]):
                        arc_load.setdefault((a, b), gp.LinExpr()).add(f[u, v, p, d], 1.0)
            for d, (s, t_) in (enumerate(((u, v), (v, u))) if not lazy else ()):
                vol_terms = pair_vol.get((s, t_), [])
                vmax = sum(vol for _, vol in vol_terms)
                model.addConstr(
                    gp.quicksum(f[u, v, p, d] for p in range(len(plist))) == gp.quicksum(lv * vol for lv, vol in vol_terms),
                    name=f"path_volume_{locations[s].id}_{locations[t_].id}"
                )
                for p in range(len(plist)):
                    model.addConstr(f[u, v, p, d] <= vmax * sel[u, v, p], name=f"path_gate_{locations[s].id}_{locations[t_].id}_p{p}")
                cstats['c10_path_choice'] += 1 + len(plist)
            model.addConstr(gp.quicksum(sel[u, v, p] for p in range(len(plist))) <= 1, name=f"one_path_{locations[u].id}_{locations[v].id}")
            cstats['c10_path_choice'] += 1
            # Optional (configs/buses.json: backbone.path_link_cut): a flow between the sites needs a selected path,
            # and a selected path has a cable on each of its links. Implied for integer solutions; in the LP it makes
            # a fractional placement pay a proportional share of a whole cable instead of load / capacity.
            if net.get('path_link_cut'):
                sel_sum = gp.quicksum(sel[u, v, p] for p in range(len(plist)))
                for (s, t_) in (((u, v), (v, u)) if not lazy else ()):
                    for lv, _vol in pair_vol.get((s, t_), []):
                        model.addConstr(sel_sum >= lv, name=f"path_needed_{len(net['lnk'])}_{cstats['c10_path_choice']}")
                        cstats['c10_path_choice'] += 1
                for p, path in enumerate(plist):
                    for a, b in zip(path, path[1:]):
                        key = (a, b, 'ETH') if a < b else (b, a, 'ETH')
                        model.addConstr(comm[key] >= sel[u, v, p], name=f"path_cable_{locations[u].id}_{locations[v].id}_p{p}_{locations[a].id}_{locations[b].id}")
                        cstats['c10_path_choice'] += 1
            # C-NET-3: a relaying site needs a SWITCH
            for p, path in enumerate(plist):
                for w in path[1:-1]:
                    if has_switch:
                        model.addConstr(sel[u, v, p] <= hw_use[w, 'SWITCH'], name=f"relay_switch_{locations[u].id}_{locations[v].id}_p{p}_{locations[w].id}")
                    else:
                        model.addConstr(sel[u, v, p] == 0, name=f"relay_impossible_{locations[u].id}_{locations[v].id}_p{p}")
                    cstats['c10_relay_switch'] += 1

        # C-NET-4: link capacity per direction: routed volume <= ETH capacity * number of cables
        for (a, b), load in arc_load.items():
            key = (a, b, 'ETH') if a < b else (b, a, 'ETH')
            model.addConstr(load <= eth_cap * comm[key], name=f"link_capacity_{locations[a].id}_{locations[b].id}")
            cstats['c10_link_capacity'] += 1

        cstats['total'] = sum(v for k, v in cstats.items() if k != 'total')
        return model, cstats

    def _build_objective_function(self, model, x, P, y, hw_use, if_use, comm, scs, locations, sensors, actuators, 
                                  cable_types, comm_matrix, partitions, hardwares, interfaces,
                                  n_locs, unique_asils, all_hw, all_interfaces,
                                  feasible_locs_per_sc, attach_s=None, attach_a=None, shared_attach_s=None, shared_attach_a=None, shared_trunk_len=None, shared_extra_trunk_len=None,
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
            feasible_locs_per_sc, comm=comm, attach_s=attach_s, attach_a=attach_a,
            shared_attach_s=shared_attach_s, shared_attach_a=shared_attach_a,
            shared_trunk_len=shared_trunk_len, shared_extra_trunk_len=shared_extra_trunk_len
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

    def optimize(self, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=None, hardwares=None, interfaces=None, enable_comm_bw_constraints=True, comm_bw_big_m=10000, minimize_cable_length=False, cable_length_limit=None, cost_limit=None, time_limit=None, mip_gap=None, lp_path=None, uncertainty=None, backbone=None, phase2_time=600, gurobi_params=None, branch_priority=False):
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
        infeasible_ij = precompute_latency_infeasible_pairs(scs, locations, sensors, actuators, cable_types)
        if infeasible_ij:
            print(f"  Precomputed infeasible SC-location pairs (latency): {len(infeasible_ij)}")

        # ======================= CREATE MODEL & VARIABLES =======================
        model, x, P, y, hw_use, if_use, attach_s, attach_a, shared_attach_s, shared_attach_a, shared_trunk_len, shared_extra_trunk_len, comm, traffic_flows, net, feasible_locs_per_sc, feasible_scs_per_loc, var_stats = self._create_model_and_variables(
            n_sc, n_locs, scs, locations, unique_asils, all_hw, sensors, actuators, all_interfaces, infeasible_ij=infeasible_ij, comm_matrix=comm_matrix,
            partitions=partitions, backbone=backbone
        )
        
        # ======================= ADD CONSTRAINTS =======================
        model, cstats = self._add_constraints(
            model, x, P, y, hw_use, if_use, attach_s, attach_a, shared_attach_s, shared_attach_a, shared_trunk_len, shared_extra_trunk_len, comm, traffic_flows, net, scs, locations, sensors, actuators,
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
        print(f"  shared_attach_s: {var_stats.get('shared_attach_s', 0)}")
        print(f"  shared_attach_a: {var_stats.get('shared_attach_a', 0)}")
        print(f"  shared_trunk_len: {var_stats.get('shared_trunk_len', 0)}")
        print(f"  shared_extra_trunk_len: {var_stats.get('shared_extra_trunk_len', 0)}")
        print(f"  comm: {var_stats['comm']} ({(100.0 * var_stats['comm'] / max(1, var_stats['total'])):.1f}%)")
        print(f"  traffic_flows: {var_stats['traffic_flows']}")
        print(f"  lnk: {var_stats['lnk']}  sel: {var_stats['sel']}  f: {var_stats['f']}  (site pairs with candidate paths: {var_stats['pairs_with_paths']})")

        print("\n[DEBUG] Constraint breakdown:")
        for k, v in cstats.items():
            print(f"  {k}: {v}")

        # ======================= BUILD OBJECTIVE =======================
        model, objectives = self._build_objective_function(
            model, x, P, y, hw_use, if_use, comm, scs, locations, sensors, actuators,
            cable_types, comm_matrix, partitions, hardwares, interfaces,
            n_locs, unique_asils, all_hw, all_interfaces,
            feasible_locs_per_sc, attach_s=attach_s, attach_a=attach_a,
            shared_attach_s=shared_attach_s, shared_attach_a=shared_attach_a,
            shared_trunk_len=shared_trunk_len, shared_extra_trunk_len=shared_extra_trunk_len,
            minimize_cable_length=minimize_cable_length, cable_length_limit=cable_length_limit,
            cost_limit=cost_limit
        )

        model.update()
        print("\n[DEBUG] Gurobi model size (final):")
        print(f"  NumVars: {model.NumVars}")
        print(f"  NumConstrs: {model.NumConstrs}")

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
        # Branch on the site decisions first (--branch_priority): closing a cell prunes all its placements at once
        if branch_priority:
            for var in y.values():
                var.BranchPriority = 10
            for var in list(P.values()) + list(hw_use.values()):
                var.BranchPriority = 5
            print("  Branch priority: y (10) > P, hw (5) > rest (0)")
        # ======================= SOLVE (lexicographic, two phases) =======================
        print(f"\nSolving...")
        (primary_name, primary_expr), (secondary_name, secondary_expr) = objectives
        if lp_path:
            model.setObjective(primary_expr, GRB.MINIMIZE)
            model.write(lp_path)
            print(f"  Model written to: {lp_path}")

        tl = float(time_limit) if (time_limit is not None and float(time_limit) > 0) else None

        # Lazy traffic constraints (configs/buses.json: backbone.lazy_traffic): every integer solution is checked;
        # a site pair with traffic but no selected path, or a link carrying more than its cables allow, adds a cut.
        lazy_cb = None
        if net['lazy']:
            model.setParam('LazyConstraints', 1)
            end_vars, end_index = [], {}
            for t, (src, dst) in net['flow_ends'].items():
                for var in list(src.values()) + list(dst.values()):
                    if id(var) not in end_index:
                        end_index[id(var)] = len(end_vars)
                        end_vars.append(var)
            sel_keys = list(net['sel'].keys())
            sel_vars = [net['sel'][k] for k in sel_keys]
            comm_keys = list(comm.keys())
            comm_vars = [comm[k] for k in comm_keys]
            eth_cap = float(cable_types['ETH'].capacity)
            model._lazy_cuts = 0

            def lazy_cb(m, where):
                if where != GRB.Callback.MIPSOL:
                    return
                ev = m.cbGetSolution(end_vars)
                sv = dict(zip(sel_keys, m.cbGetSolution(sel_vars)))
                cv = dict(zip(comm_keys, m.cbGetSolution(comm_vars)))
                load, contrib = {}, {}
                for t, (src, dst) in net['flow_ends'].items():
                    u = next((k for k, var in src.items() if ev[end_index[id(var)]] > 0.5), None)
                    v = next((k for k, var in dst.items() if ev[end_index[id(var)]] > 0.5), None)
                    if u is None or v is None or u == v:
                        continue
                    pair = (min(u, v), max(u, v))
                    plist = net['paths'][pair]
                    chosen = [p for p in range(len(plist)) if sv[pair + (p,)] > 0.5]
                    if not chosen:
                        m.cbLazy(gp.quicksum(net['sel'][pair + (p,)] for p in range(len(plist))) >= src[u] + dst[v] - 1)
                        m._lazy_cuts += 1
                        continue
                    p = chosen[0]
                    seq = plist[p] if (u, v) == pair else plist[p][::-1]
                    vol = net['volumes'][t]
                    for a, b in zip(seq, seq[1:]):
                        load[(a, b)] = load.get((a, b), 0.0) + vol
                        contrib.setdefault((a, b), []).append((src[u], dst[v], net['sel'][pair + (p,)], vol))
                for (a, b), ld in load.items():
                    key = (a, b, 'ETH') if a < b else (b, a, 'ETH')
                    if ld > eth_cap * cv[key] + 1e-6:
                        m.cbLazy(eth_cap * comm[key] >= gp.quicksum(vol * (s_var + d_var + sel_var - 2) for s_var, d_var, sel_var, vol in contrib[(a, b)]))
                        m._lazy_cuts += 1

        def _solve_lexicographic():
            """Phase 1: primary objective. Phase 2: secondary objective with the primary fixed at its phase-1 value."""
            if tl is not None:
                model.setParam('TimeLimit', tl)
            model.setObjective(primary_expr, GRB.MINIMIZE)
            phases = []
            print(f"\n[Lexicographic] Phase 1: minimize {primary_name}")
            model.optimize(lazy_cb) if lazy_cb else model.optimize()
            phases.append(self._phase_info(model, primary_name))
            lex_constr = None
            if model.SolCount > 0 and model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                primary_val = primary_expr.getValue()
                # Phase 2 always runs: a phase-1 solution alone has an arbitrary secondary objective (e.g. a min-cable
                # design with junk cost). It gets the remaining time, but at least phase2_time seconds; the phase-1
                # solution stays feasible and is Gurobi's start solution.
                lex_constr = model.addConstr(
                    primary_expr <= primary_val + max(1e-6, 1e-9 * abs(primary_val)),
                    name=f"lex_fix_{primary_name}"
                )
                model.setObjective(secondary_expr, GRB.MINIMIZE)
                if tl is not None:
                    model.setParam('TimeLimit', max(tl - model.Runtime, float(phase2_time)))
                print(f"\n[Lexicographic] Phase 2: minimize {secondary_name} with {primary_name} <= {primary_val:.6f}")
                model.optimize(lazy_cb) if lazy_cb else model.optimize()
                phases.append(self._phase_info(model, secondary_name))
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
            partition_map, failed = pack_partitions(x, P, scs, locations, partitions, uncertainty)
            if not failed:
                break
            if packing_cuts >= MAX_PACKING_CUTS:
                print(f"\n[Packing] WARNING: {len(failed)} (site, ASIL) group(s) still not packable after {packing_cuts} cuts; keeping the solution")
                break
            for (j, a, members, p_val) in failed:
                # If all SCs of the group are hosted at site j, at least p_val + 1 partitions are needed
                model.addConstr(
                    P[j, a] >= (p_val + 1) * (gp.quicksum(x[i, j] for i in members) - len(members) + 1),
                    name=f"packing_cut_{packing_cuts}_{locations[j].id}_asil{a}"
                )
                packing_cuts += 1
                print(f"\n[Packing] {len(members)} SCs of ASIL {a} at {locations[j].id} do not fit into {p_val} partition(s); cut added")
            if lex_constr is not None:
                model.remove(lex_constr)
            print(f"[Packing] Solving again with {packing_cuts} cut(s)")

        # ======================= RETURN FORMATTED SOLUTION =======================
        if model.SolCount > 0 and model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            time_limited = any(ph['status'] != 'OPTIMAL' for ph in phases)
            if time_limited:
                print(f"\n⚠ Time limit reached; keeping best solution found (gaps: " + ", ".join(f"{ph['objective_name']} {100.0 * ph['mip_gap']:.4f}%" for ph in phases) + ")")
            else:
                print(f"\n✓ Optimal solution found!")
            # Extract and return formatted solution
            formatted_solution = extract_solution(
                x, P, y, hw_use, if_use, comm, scs, locations, sensors, actuators,
                cable_types, partitions, hardwares, interfaces, comm_matrix=comm_matrix,
                traffic_flows=traffic_flows, net=net, attach_s=attach_s, attach_a=attach_a,
                shared_attach_s=shared_attach_s, shared_attach_a=shared_attach_a,
                shared_trunk_len=shared_trunk_len, shared_extra_trunk_len=shared_extra_trunk_len,
                partition_map=partition_map
            )
            formatted_solution['solve_info'] = {
                'objective_type': 'min_cable_length' if minimize_cable_length else 'min_cost',
                'objectives': [primary_name, secondary_name],
                'cable_length_limit': cable_length_limit,
                'cost_limit': cost_limit,
                'phases': phases,
                'packing_cuts': packing_cuts,
                'backbone_candidate_paths': net['k_paths'],
                'lazy_traffic': net['lazy'],
                'lazy_cuts': int(getattr(model, '_lazy_cuts', 0)),
                'gurobi_params': gurobi_params,
                'branch_priority': bool(branch_priority),
                'runtime_s': sum(ph['runtime_s'] for ph in phases),
                'num_vars': int(model.NumVars),
                'num_constrs': int(model.NumConstrs),
                'num_bin_vars': int(model.NumBinVars),
                'num_int_vars': int(model.NumIntVars),
                'lp_file': lp_path,
                'uncertainty': {k: bool((uncertainty or {}).get(k)) for k in UNCERTAINTY_BLOCKS},
            }
            formatted_solution['status'] = 'TIME_LIMIT' if time_limited else 'OPTIMAL'
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

    def optimize_pareto_epsilon_constraint(self, scs, locations, sensors, actuators, cable_types, comm_matrix,
                                           partitions=None, hardwares=None, interfaces=None,
                                           enable_comm_bw_constraints=True, comm_bw_big_m=10000, num_points=5,
                                           solve_time_limit=None, mip_gap=None, lp_dir=None, uncertainty=None, backbone=None, phase2_time=600, gurobi_params=None, branch_priority=False):
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

        # Extreme 1: minimum cost
        sol_min_cost = self.optimize(
            scs, locations, sensors, actuators, cable_types, comm_matrix,
            partitions=partitions, hardwares=hardwares, interfaces=interfaces,
            enable_comm_bw_constraints=enable_comm_bw_constraints,
            comm_bw_big_m=comm_bw_big_m,
            time_limit=solve_time_limit,
            mip_gap=mip_gap,
            lp_path=_lp("min_cost"),
            uncertainty=uncertainty, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params, branch_priority=branch_priority
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
            uncertainty=uncertainty, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params, branch_priority=branch_priority
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
                        uncertainty=uncertainty, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params, branch_priority=branch_priority
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
                        uncertainty=uncertainty, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params, branch_priority=branch_priority
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


