import os
import numpy as np
os.environ["GRB_LICENSE_FILE"] = "/home/okumus/gurobi.lic"
import gurobipy as gp
from gurobipy import GRB
from optimizer_utils import (get_distance, build_sensor_lookup, build_actuator_lookup, 
                             precompute_latency_infeasible_pairs, build_cost_map, build_latency_map, extract_solution)

# We need change direct link b/w the ECUs to some sort of network topology.
# We did not include uncertainty in overall system.
# We do not know numbers we gave is correct (i.e., cost, latency,  etc.)
# WE do not know our objective functions is correct. (semanatically, we can change distance with complexity model)



class AssignmentOptimizer:
    def __init__(self):
        print("Initialized AssignmentOptimizerNew with quadratic ECU-ECU costs and linear sensor/actuator costs")
        pass

    def calculate_cable_expressions(self, z, y, scs, locations, sensors, actuators, cable_types, comm_matrix, max_partitions_per_asil_per_loc, feasible_locs_per_sc, comm=None, attach_s=None, attach_a=None, shared_attach_s=None, shared_attach_a=None, shared_trunk_len=None, shared_extra_trunk_len=None):
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
            a = scs[i].asil_req
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

                x_ij_expr = gp.quicksum(z[i, j, a, p] for p in range(max_partitions_per_asil_per_loc) if (i, j, a, p) in z)

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
    
    def _create_model_and_variables(self, n_sc, n_locs, scs, locations, unique_asils, all_hw, sensors, actuators, all_interfaces, infeasible_ij=None, comm_matrix=None):
        """
        Create Gurobi model and define all decision variables
        
        Returns:
            model: Gurobi model object
            y: Partition opening variables
            z: SC-to-partition assignment variables
            hw_use: HW feature usage variables
            if_use: Interface usage variables
            comm: ECU-to-ECU communication variables
            max_partitions_per_asil_per_loc: Maximum partitions per location per ASIL
            feasible_locs_per_sc: Feasible location mapping per SC
            feasible_scs_per_loc: Feasible SC mapping per location
            var_stats: Variable family counts
        """
        # Create model
        model = gp.Model("ECU_Assignment")
        model.setParam('OutputFlag', 1)
        model.setParam('MIPFocus', 2)
        #model.setParam("MIPGap", 0.001)  # 1% gap for faster solutions (adjust as needed)
        model.update()
        
        # y[j,a,p] = Location j has partition for ASIL a (replica p)
        max_partitions_per_asil_per_loc = 3
        y = {}
        for j in range(n_locs):
            for a in unique_asils:
                for p in range(max_partitions_per_asil_per_loc):
                    y[j, a, p] = model.addVar(vtype=GRB.BINARY, name=f"y_{locations[j].id}_asil{a}_p{p}")
        
        # z[i,j,a,p] = SC i assigned to location j ASIL a partition p
        infeasible_ij = infeasible_ij or set()
        feasible_locs_per_sc = {i: [] for i in range(n_sc)}
        feasible_scs_per_loc = {j: [] for j in range(n_locs)}

        z = {}
        for i in range(n_sc):
            a = scs[i].asil_req
            for j in range(n_locs):
                if (i, j) in infeasible_ij:
                    continue
                feasible_locs_per_sc[i].append(j)
                feasible_scs_per_loc[j].append(i)
                for p in range(max_partitions_per_asil_per_loc):
                    z[i, j, a, p] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"z_{scs[i].id}_{locations[j].id}_asil{a}_p{p}"
                    )

        for i in range(n_sc):
            if not feasible_locs_per_sc[i]:
                raise ValueError(f"SC {scs[i].id} has no feasible locations after latency precompute")
        
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
        
        # comm[j1,j2,iface] = Number of iface links between location j1 and j2 (integer)
        # Sparse creation: only build comm vars for location pairs that can actually be required by SC comm links.
        required_loc_pairs = set()
        if comm_matrix:
            sc_id_to_idx = {sc.id: i for i, sc in enumerate(scs)}
            for link in comm_matrix:
                src_id = link.get('src')
                dst_id = link.get('dst')
                volume = link.get('volume', 0)
                if volume <= 0:
                    continue
                if src_id not in sc_id_to_idx or dst_id not in sc_id_to_idx:
                    continue

                i = sc_id_to_idx[src_id]
                k = sc_id_to_idx[dst_id]
                for j1 in feasible_locs_per_sc[i]:
                    for j2 in feasible_locs_per_sc[k]:
                        if j1 == j2:
                            continue
                        a, b = (j1, j2) if j1 < j2 else (j2, j1)
                        required_loc_pairs.add((a, b))

            # Also include potential pairs needed by ETH sensor/actuator traffic to avoid missing comm edges.
            # For each ETH sensor used by an SC, allow the attachment location to connect to any feasible
            # destination SC location. Similarly for ETH actuators.
            sensor_lookup = build_sensor_lookup(sensors)
            actuator_lookup = build_actuator_lookup(actuators)
            for sc_idx, sc in enumerate(scs):
                # Sensor -> SC
                for s_id in (getattr(sc, 'sensors', None) or []):
                    s_obj = sensor_lookup.get(s_id)
                    if not s_obj or getattr(s_obj, 'interface', None) != 'ETH':
                        continue
                    for j_attach in range(n_locs):
                        for j_sc in feasible_locs_per_sc[sc_idx]:
                            if j_attach == j_sc:
                                continue
                            a, b = (j_attach, j_sc) if j_attach < j_sc else (j_sc, j_attach)
                            required_loc_pairs.add((a, b))

                # SC -> Actuator
                for a_id in (getattr(sc, 'actuators', None) or []):
                    a_obj = actuator_lookup.get(a_id)
                    if not a_obj or getattr(a_obj, 'interface', None) != 'ETH':
                        continue
                    for j_attach in range(n_locs):
                        for j_sc in feasible_locs_per_sc[sc_idx]:
                            if j_attach == j_sc:
                                continue
                            a, b = (j_attach, j_sc) if j_attach < j_sc else (j_sc, j_attach)
                            required_loc_pairs.add((a, b))
        else:
            for j1 in range(n_locs):
                for j2 in range(j1 + 1, n_locs):
                    required_loc_pairs.add((j1, j2))

        comm = {}
        for j1, j2 in sorted(required_loc_pairs):
            #for iface in all_interfaces:
            #    comm[j1, j2, iface] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"comm_{locations[j1].id}_{locations[j2].id}_{iface}")
            # We assume all communication uses the same interface type for simplicity in this model (e.g., Ethernet)
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

        # Create flow
        flow = {}
        for tr in traffic_flows:
            t = tr['id']
            for (j1, j2, iface) in comm.keys():
                 # Directed flow variables
                 flow[t, j1, j2] = model.addVar(vtype=GRB.BINARY, name=f"f_{t}_{j1}_{j2}")
                 flow[t, j2, j1] = model.addVar(vtype=GRB.BINARY, name=f"f_{t}_{j2}_{j1}")


        var_stats = {
            'y': len(y),
            'z': len(z),
            'hw_use': len(hw_use),
            'if_use': len(if_use),
            'attach_s': len(attach_s),
            'attach_a': len(attach_a),
            'shared_attach_s': len(shared_attach_s),
            'shared_attach_a': len(shared_attach_a),
            'shared_trunk_len': len(shared_trunk_len),
            'shared_extra_trunk_len': len(shared_extra_trunk_len),
            'comm': len(comm),
            'comm_loc_pairs': len(required_loc_pairs),
            'traffic_flows': len(traffic_flows),
            'flow': len(flow)
        }
        var_stats['total'] = (
            var_stats['y']
            + var_stats['z']
            + var_stats['hw_use']
            + var_stats['if_use']
            + var_stats['attach_s']
            + var_stats['attach_a']
            + var_stats['shared_attach_s']
            + var_stats['shared_attach_a']
            + var_stats['shared_trunk_len']
            + var_stats['shared_extra_trunk_len']
            + var_stats['comm']
            + var_stats['traffic_flows']
            + var_stats['flow']
        )

        return model, y, z, hw_use, if_use, attach_s, attach_a, shared_attach_s, shared_attach_a, shared_trunk_len, shared_extra_trunk_len, comm, traffic_flows, flow, max_partitions_per_asil_per_loc, feasible_locs_per_sc, feasible_scs_per_loc, var_stats

    def _add_constraints(self, model, y, z, hw_use, if_use, attach_s, attach_a, shared_attach_s, shared_attach_a, shared_trunk_len, shared_extra_trunk_len, comm, traffic_flows, flow, scs, locations, sensors, actuators, 
                              cable_types, comm_matrix, partitions, unique_asils, max_partitions_per_asil_per_loc,
                              feasible_locs_per_sc, feasible_scs_per_loc, enable_comm_bw_constraints=True,
                              comm_bw_big_m=None):
        """
        Add all constraints to the optimization model
        """
        n_sc = len(scs)
        n_locs = len(locations)
        all_interfaces = list(cable_types.keys())
        all_hw = sorted(list(set(h for (j, h) in hw_use.keys())))
        
        print(f"\nAdding constraints...")

        cstats = {
            'c1b_unique_partition': 0,
            'c1d_z_y_relation': 0,
            'c1e_partition_used_only_if_assigned': 0,
            'c2_capacity': 0,
            'c3_hw_required': 0,
            'c4_if_required': 0,
            'c4_asil_bus_split': 0,
            'c5_redundancy': 0,
            'c6_ai_contention': 0,
            'c8_loc_loc_latency': 0,
            'c9_if_activation': 0,
            'c9_attach': 0,
            'c9_shared_attach': 0,
            'c9_shared_trunk': 0,
            'c10_network_load': 0,
            'c10_linearization': 0,
        }
        
        def x_expr(i, j):
            a_req = scs[i].asil_req
            return gp.quicksum(z[i, j, a_req, p] for p in range(max_partitions_per_asil_per_loc) if (i, j, a_req, p) in z)

        # SW-based uncertainty (selective static margin / VIP list)
        # Extra margin is applied only to ADAS and Infotainment domains.
        asil_margins = {
            0: 0.20,  # QM
            1: 0.15,  # A
            2: 0.10,  # B
            3: 0.05,  # C
            4: 0.00,  # D
            'QM': 0.20,
            'A': 0.15,
            'B': 0.10,
            'C': 0.05,
            'D': 0.00,
        }

        def get_sw_multiplier(sc):
            domain_norm = str(getattr(sc, 'domain', '') or '').strip().upper()
            if domain_norm not in {'ADAS', 'INFOTAINMENT'}:
                return 1.0

            asil_req = getattr(sc, 'asil_req', 0)
            if isinstance(asil_req, str):
                asil_key = asil_req.strip().upper()
            else:
                asil_key = int(asil_req)

            m_asil = asil_margins.get(asil_key, 0.20)
            m_ai = 0.25 if ('HW_ACC' in (getattr(sc, 'hw_required', None) or [])) else 0.0
            return 1.0 + m_asil + m_ai

        def asil_safety_group(sc):
            a = getattr(sc, 'asil_req', 0)
            if isinstance(a, str):
                a_norm = a.strip().upper()
                return 'HIGH' if a_norm in {'C', 'D'} else 'LOW'
            return 'HIGH' if int(a) >= 3 else 'LOW'

        # Constraint 1b: Every SC assigned to exactly one partition
        for i in range(n_sc):
            a = scs[i].asil_req
            model.addConstr(
                gp.quicksum(
                    z[i, j, a, p]
                    for j in feasible_locs_per_sc[i]
                    for p in range(max_partitions_per_asil_per_loc)
                    if (i, j, a, p) in z
                ) == 1,
                name=f"unique_partition_{scs[i].id}"
            )
            cstats['c1b_unique_partition'] += 1
        
        # Constraint 1d: z relationship with y (if SC at partition, partition is open)
        for i in range(n_sc):
            a = scs[i].asil_req
            for j in feasible_locs_per_sc[i]:
                for p in range(max_partitions_per_asil_per_loc):
                    if (i, j, a, p) not in z:
                        continue
                    model.addConstr(
                        z[i, j, a, p] <= y[j, a, p],
                        name=f"z_y_relation_{scs[i].id}_{locations[j].id}_p{p}"
                    )
                    cstats['c1d_z_y_relation'] += 1

        # Constraint 1e: A partition can be open only if at least one SC is assigned to it.
        # Prevents ghost partitions (y=1 without any z assignment).
        for j in range(n_locs):
            for a in unique_asils:
                for p in range(max_partitions_per_asil_per_loc):
                    assigned_sum = gp.quicksum(
                        z[i, j, a, p]
                        for i in feasible_scs_per_loc[j]
                        if scs[i].asil_req == a and (i, j, a, p) in z
                    )
                    model.addConstr(
                        y[j, a, p] <= assigned_sum,
                        name=f"partition_used_only_if_assigned_{locations[j].id}_asil{a}_p{p}"
                    )
                    cstats['c1e_partition_used_only_if_assigned'] += 1
        
        # Constraint 2: Capacity per location per ASIL partition
        for j in range(n_locs):
            H_j = max(0.0, float(getattr(locations[j], 'health_factor', 1.0)))
            for a in unique_asils:
                for p in range(max_partitions_per_asil_per_loc):
                    # CPU Capacity
                    sc_cpu_demand = gp.quicksum(
                        z[i, j, a, p] * scs[i].cpu_req * get_sw_multiplier(scs[i])
                        for i in feasible_scs_per_loc[j] if scs[i].asil_req == a and (i, j, a, p) in z
                    )
                    model.addConstr(
                        sc_cpu_demand <= y[j, a, p] * partitions.get('cpu_cap', float('inf')) * H_j,
                        name=f"cpu_capacity_uncertain_{locations[j].id}_asil{a}_p{p}"
                    )
                    
                    # RAM Capacity
                    sc_ram_demand = gp.quicksum(
                        z[i, j, a, p] * scs[i].ram_req * get_sw_multiplier(scs[i])
                        for i in feasible_scs_per_loc[j] if scs[i].asil_req == a and (i, j, a, p) in z
                    )
                    model.addConstr(
                        sc_ram_demand <= y[j, a, p] * partitions.get('ram_cap', float('inf')) * H_j,
                        name=f"ram_capacity_uncertain_{locations[j].id}_asil{a}_p{p}"
                    )
                    
                    # ROM Capacity
                    sc_rom_demand = gp.quicksum(
                        z[i, j, a, p] * scs[i].rom_req * get_sw_multiplier(scs[i])
                        for i in feasible_scs_per_loc[j] if scs[i].asil_req == a and (i, j, a, p) in z
                    )
                    model.addConstr(
                        sc_rom_demand <= y[j, a, p] * partitions.get('rom_cap', float('inf')) * H_j,
                        name=f"rom_capacity_uncertain_{locations[j].id}_asil{a}_p{p}"
                    )
                    cstats['c2_capacity'] += 3
        
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

                    # If ASIL split is active, second shared trunk length equals main trunk length.
                    if (j, i_name) in shared_extra_trunk_len and (j, i_name) in shared_trunk_len:
                        split_m = 1e6
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

        K_THRESHOLD = 2
        ai_sc_indices = [
            i for i, sc in enumerate(scs)
            if 'HW_ACC' in (getattr(sc, 'hw_required', None) or [])
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
        NETWORK_SAFETY_FACTOR = 1.10
        SYNC_ERROR_CONSTANT = 0.0005
        PLATFORM_OVERHEAD = 0.002
        latency_map = build_latency_map(cable_types)

        def robust_link_latency(iface, dist):
            nominal = dist * latency_map.get(iface, 0.0)
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
        M = 1000  # Big-M value
        all_interfaces = list(cable_types.keys())

        # Build an "active node" expression per location.
        if 'SWITCH' in all_hw:
            loc_active = {
                j: gp.quicksum(y[j, a, p] for a in unique_asils for p in range(max_partitions_per_asil_per_loc)) + hw_use[j, 'SWITCH']
                for j in range(n_locs)
            }
        else:
            loc_active = {
                j: gp.quicksum(y[j, a, p] for a in unique_asils for p in range(max_partitions_per_asil_per_loc))
                for j in range(n_locs)
            }

        # Constraint 9-ATT: ETH sensor/actuator attachments must connect to an active node,
        # and must respect peripheral max-latency on the physical cable to that attachment point.
        # (We do not model end-to-end network latency here; this only constrains the local cable.)
        latency_map = build_latency_map(cable_types)
        if attach_s:
            for (si, j), av in attach_s.items():
                model.addConstr(av <= loc_active[j], name=f"attachS_active_{sensors[si].id}_{locations[j].id}")
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
                part_sum_j = gp.quicksum(
                    y[j, a, p]
                    for a in unique_asils
                    for p in range(max_partitions_per_asil_per_loc)
                )
                incident_comm_j = gp.LinExpr()
                for (j1, j2, _iface), comm_var in comm.items():
                    if j1 == j or j2 == j:
                        incident_comm_j.add(comm_var, 1.0)

                # incident_comm_j >= 2 when switch-only; relaxed when part_sum_j >= 1
                model.addConstr(
                    incident_comm_j >= 2 * hw_use[j, 'SWITCH'] - 2 * part_sum_j,
                    name=f"switch_only_degree_ge2_{locations[j].id}"
                )
        

        
        # C-NET-1: (Flow Conservation)
        for tr in traffic_flows:
            t = tr['id']
            t_type = tr['type']
            
            for j in range(n_locs):
                # Incoming and outgoing flows (same)
                f_in = gp.quicksum(flow[t, k, j] for k in range(n_locs) if (t, k, j) in flow)
                f_out = gp.quicksum(flow[t, j, k] for k in range(n_locs) if (t, j, k) in flow)
                
                # --- SOURCE LOGIC ---
                is_src = 0
                if t_type == 'SC_SC':
                    src_i = tr['src_idx']
                    # Is the source SC at this location 'j'?
                    is_src = gp.quicksum(z[src_i, j, scs[src_i].asil_req, p] for p in range(max_partitions_per_asil_per_loc) if (src_i, j, scs[src_i].asil_req, p) in z)
                elif t_type == 'SENS_SC':
                    # ETH sensor source is its attachment location
                    s_idx = tr.get('sensor_idx')
                    if s_idx is not None and attach_s and (s_idx, j) in attach_s:
                        is_src = attach_s[s_idx, j]
                elif t_type == 'SC_ACT':
                    src_i = tr.get('src_idx')
                    if src_i is not None:
                        is_src = gp.quicksum(z[src_i, j, scs[src_i].asil_req, p] for p in range(max_partitions_per_asil_per_loc) if (src_i, j, scs[src_i].asil_req, p) in z)
                
                # --- DESTINATION LOGIC ---
                is_dest = 0
                if t_type == 'SC_SC':
                    dst_i = tr['dst_idx']
                    # Is the destination SC at this location 'j'?
                    is_dest = gp.quicksum(z[dst_i, j, scs[dst_i].asil_req, p] for p in range(max_partitions_per_asil_per_loc) if (dst_i, j, scs[dst_i].asil_req, p) in z)
                elif t_type == 'SENS_SC':
                    dst_i = tr.get('dst_idx')
                    if dst_i is not None:
                        is_dest = gp.quicksum(z[dst_i, j, scs[dst_i].asil_req, p] for p in range(max_partitions_per_asil_per_loc) if (dst_i, j, scs[dst_i].asil_req, p) in z)
                elif t_type == 'SC_ACT':
                    a_idx = tr.get('act_idx')
                    if a_idx is not None and attach_a and (a_idx, j) in attach_a:
                        is_dest = attach_a[a_idx, j]

                # EQUATION: Flow_Out - Flow_In = Source - Dest
                model.addConstr(f_out - f_in == is_src - is_dest, name=f"flow_bal_{t}_{j}")

        # C-NET-1b: Simple-path enforcement (avoid cycles/branching)
        # Without this, the binary flow model can create extra cyclic flow that has no objective penalty,
        # which then (via C-NET-3) can force SWITCH selection even when a direct link exists.
        # We enforce for each SC_SC traffic:
        # - Source node has no incoming flow
        # - Destination node has no outgoing flow
        # - Intermediate nodes have in-degree <= 1 and out-degree <= 1
        deg_M = n_locs  # big-M for degree constraints
        for tr in traffic_flows:
            if tr.get('type') not in ('SC_SC', 'SENS_SC', 'SC_ACT'):
                continue
            t = tr['id']
            for j in range(n_locs):
                f_in = gp.quicksum(flow[t, k, j] for k in range(n_locs) if (t, k, j) in flow)
                f_out = gp.quicksum(flow[t, j, k] for k in range(n_locs) if (t, j, k) in flow)

                if tr.get('type') == 'SC_SC':
                    src_i = tr['src_idx']
                    dst_i = tr['dst_idx']
                    is_src = gp.quicksum(
                        z[src_i, j, scs[src_i].asil_req, p]
                        for p in range(max_partitions_per_asil_per_loc)
                        if (src_i, j, scs[src_i].asil_req, p) in z
                    )
                    is_dest = gp.quicksum(
                        z[dst_i, j, scs[dst_i].asil_req, p]
                        for p in range(max_partitions_per_asil_per_loc)
                        if (dst_i, j, scs[dst_i].asil_req, p) in z
                    )
                elif tr.get('type') == 'SENS_SC':
                    s_idx = tr.get('sensor_idx')
                    dst_i = tr.get('dst_idx')
                    is_src = attach_s[s_idx, j] if (attach_s and s_idx is not None and (s_idx, j) in attach_s) else 0
                    is_dest = gp.quicksum(
                        z[dst_i, j, scs[dst_i].asil_req, p]
                        for p in range(max_partitions_per_asil_per_loc)
                        if (dst_i, j, scs[dst_i].asil_req, p) in z
                    ) if dst_i is not None else 0
                else:  # SC_ACT
                    src_i = tr.get('src_idx')
                    a_idx = tr.get('act_idx')
                    is_src = gp.quicksum(
                        z[src_i, j, scs[src_i].asil_req, p]
                        for p in range(max_partitions_per_asil_per_loc)
                        if (src_i, j, scs[src_i].asil_req, p) in z
                    ) if src_i is not None else 0
                    is_dest = attach_a[a_idx, j] if (attach_a and a_idx is not None and (a_idx, j) in attach_a) else 0

                # Source cannot have incoming
                model.addConstr(f_in <= deg_M * (1 - is_src), name=f"path_no_in_src_{t}_{j}")
                # Destination cannot have outgoing
                model.addConstr(f_out <= deg_M * (1 - is_dest), name=f"path_no_out_dest_{t}_{j}")

                # Intermediate degree limits
                model.addConstr(f_out <= 1 + (deg_M - 1) * is_src, name=f"path_out_deg_{t}_{j}")
                model.addConstr(f_in <= 1 + (deg_M - 1) * is_dest, name=f"path_in_deg_{t}_{j}")

        # C-NET-2: Capacity & Linking
        # If there is flow -> Comm (Cable) must exist
        for (u, v, iface), comm_var in comm.items():
            if iface != "ETH": continue
            
            # u->v direction
            load_uv = gp.LinExpr()
            # v->u direction
            load_vu = gp.LinExpr()
            
            for tr in traffic_flows:
                t = tr['id']
                vol = tr['volume']
                if (t, u, v) in flow: load_uv += flow[t, u, v] * vol
                if (t, v, u) in flow: load_vu += flow[t, v, u] * vol
            
            # If Ethernet is Full Duplex, handle separately; otherwise, total control is sufficient.
            # For simplicity, we link both directions to the cable capacity:
            model.addConstr(load_uv <= cable_types.get('ETH', None).capacity * comm_var, name=f"cap_uv_{u}_{v}")
            model.addConstr(load_vu <= cable_types.get('ETH', None).capacity * comm_var, name=f"cap_vu_{u}_{v}")

        # C-NET-3: Switch Permission (Hop Logic)
        # If you are not the source but are sending data out, Switch HW is required.
        for tr in traffic_flows:
            t = tr['id']
            
            for j in range(n_locs):
                f_out_sum = gp.quicksum(flow[t, j, k] for k in range(n_locs) if (t, j, k) in flow)

                # Determine whether node j is the SOURCE of this traffic flow
                is_src = 0
                if tr.get('type') == 'SC_SC':
                    src_i = tr['src_idx']
                    is_src = gp.quicksum(
                        z[src_i, j, scs[src_i].asil_req, p]
                        for p in range(max_partitions_per_asil_per_loc)
                        if (src_i, j, scs[src_i].asil_req, p) in z
                    )
                elif tr.get('type') == 'SENS_SC':
                    s_idx = tr.get('sensor_idx')
                    if s_idx is not None and attach_s and (s_idx, j) in attach_s:
                        is_src = attach_s[s_idx, j]
                elif tr.get('type') == 'SC_ACT':
                    src_i = tr.get('src_idx')
                    if src_i is not None:
                        is_src = gp.quicksum(
                            z[src_i, j, scs[src_i].asil_req, p]
                            for p in range(max_partitions_per_asil_per_loc)
                            if (src_i, j, scs[src_i].asil_req, p) in z
                        )
                
                # Flow_Out <= Is_Source + Has_Switch
                model.addConstr(f_out_sum <= is_src + hw_use[j, 'SWITCH'], name=f"sw_rule_{t}_{j}")


        cstats['total'] = sum(v for k, v in cstats.items() if k != 'total')
        return model, cstats

    def _build_objective_function(self, model, y, z, hw_use, if_use, comm, scs, locations, sensors, actuators, 
                                  cable_types, comm_matrix, partitions, hardwares, interfaces,
                                  n_locs, unique_asils, all_hw, all_interfaces, max_partitions_per_asil_per_loc,
                                  feasible_locs_per_sc, attach_s=None, attach_a=None, shared_attach_s=None, shared_attach_a=None, shared_trunk_len=None, shared_extra_trunk_len=None,
                                  minimize_cable_length=False, cable_length_limit=None):
        """
        Build and set the objective function for the optimization model
        
        Objectives:
            - Default: minimize total cost
            - If minimize_cable_length=True: minimize total cable length
        """
        print(f"Building objective function...")
        
        # Partition cost
        partition_cost_expr = gp.quicksum(
            partitions.get('cost', 0) * y[j, a, p]
            for j in range(n_locs)
            for a in unique_asils
            for p in range(max_partitions_per_asil_per_loc)
        )
        
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
            z, {}, scs, locations, sensors, actuators, cable_types, comm_matrix, max_partitions_per_asil_per_loc,
            feasible_locs_per_sc, comm=comm, attach_s=attach_s, attach_a=attach_a,
            shared_attach_s=shared_attach_s, shared_attach_a=shared_attach_a,
            shared_trunk_len=shared_trunk_len, shared_extra_trunk_len=shared_extra_trunk_len
        )

        # Optional epsilon-constraint on total cable length for Pareto search
        if cable_length_limit is not None:
            model.addConstr(
                cable_distance_expr <= float(cable_length_limit),
                name="eps_cable_length_limit"
            )
        
        if minimize_cable_length:
            model.setObjective(cable_distance_expr, GRB.MINIMIZE)
            print("Objective: Total Cable Length (all cables)")
        else:
            total_cost = partition_cost_expr + hw_cost_expr + if_cost_expr + cable_cost_expr
            model.setObjective(total_cost, GRB.MINIMIZE)
            print("Objective: Partition + HW + Interface + Cable + Communication costs")
        
        return model

    def optimize(self, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=None, hardwares=None, interfaces=None, enable_comm_bw_constraints=True, comm_bw_big_m=10000, minimize_cable_length=False, cable_length_limit=None):
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
        
        # ======================= PRECOMPUTE FEASIBILITY =======================
        infeasible_ij = precompute_latency_infeasible_pairs(scs, locations, sensors, actuators, cable_types)
        if infeasible_ij:
            print(f"  Precomputed infeasible SC-location pairs (latency): {len(infeasible_ij)}")

        # ======================= CREATE MODEL & VARIABLES =======================
        model, y, z, hw_use, if_use, attach_s, attach_a, shared_attach_s, shared_attach_a, shared_trunk_len, shared_extra_trunk_len, comm, traffic_flows, flows ,max_partitions_per_asil_per_loc, feasible_locs_per_sc, feasible_scs_per_loc, var_stats = self._create_model_and_variables(
            n_sc, n_locs, scs, locations, unique_asils, all_hw, sensors, actuators, all_interfaces, infeasible_ij=infeasible_ij, comm_matrix=comm_matrix
        )
        
        # ======================= ADD CONSTRAINTS =======================
        model, cstats = self._add_constraints(
            model, y, z, hw_use, if_use, attach_s, attach_a, shared_attach_s, shared_attach_a, shared_trunk_len, shared_extra_trunk_len, comm, traffic_flows, flows, scs, locations, sensors, actuators,
            cable_types, comm_matrix, partitions, unique_asils, max_partitions_per_asil_per_loc,
            feasible_locs_per_sc, feasible_scs_per_loc,
            enable_comm_bw_constraints=enable_comm_bw_constraints,
            comm_bw_big_m=comm_bw_big_m
        )

        print("\n[DEBUG] Variable breakdown:")
        print(f"  total vars: {var_stats['total']}")
        print(f"  y: {var_stats['y']}")
        print(f"  z: {var_stats['z']}")
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
        print(f"  flow: {var_stats['flow']}")
        print(f"  comm_loc_pairs: {var_stats['comm_loc_pairs']}")

        print("\n[DEBUG] Constraint breakdown:")
        print(f"  total constr: {cstats['total']}")
        print(f"  c1b_unique_partition: {cstats['c1b_unique_partition']}")
        print(f"  c1d_z_y_relation: {cstats['c1d_z_y_relation']}")
        print(f"  c1e_partition_used_only_if_assigned: {cstats['c1e_partition_used_only_if_assigned']}")
        print(f"  c2_capacity: {cstats['c2_capacity']}")
        print(f"  c3_hw_required: {cstats['c3_hw_required']}")
        print(f"  c4_if_required: {cstats['c4_if_required']}")
        print(f"  c4_asil_bus_split: {cstats.get('c4_asil_bus_split', 0)}")
        print(f"  c5_redundancy: {cstats['c5_redundancy']}")
        print(f"  c6_ai_contention: {cstats.get('c6_ai_contention', 0)}")
        print(f"  c8_loc_loc_latency: {cstats['c8_loc_loc_latency']}")
        print(f"  c9_if_activation: {cstats['c9_if_activation']}")
        print(f"  c9_attach: {cstats.get('c9_attach', 0)}")
        print(f"  c9_shared_attach: {cstats.get('c9_shared_attach', 0)}")
        print(f"  c9_shared_trunk: {cstats.get('c9_shared_trunk', 0)}")

        
        # ======================= BUILD OBJECTIVE =======================
        model = self._build_objective_function(
            model, y, z, hw_use, if_use, comm, scs, locations, sensors, actuators,
            cable_types, comm_matrix, partitions, hardwares, interfaces,
            n_locs, unique_asils, all_hw, all_interfaces, max_partitions_per_asil_per_loc,
            feasible_locs_per_sc, attach_s=attach_s, attach_a=attach_a,
            shared_attach_s=shared_attach_s, shared_attach_a=shared_attach_a,
            shared_trunk_len=shared_trunk_len, shared_extra_trunk_len=shared_extra_trunk_len,
            minimize_cable_length=minimize_cable_length, cable_length_limit=cable_length_limit
        )

        model.update()
        print("\n[DEBUG] Gurobi model size (final):")
        print(f"  NumVars: {model.NumVars}")
        print(f"  NumConstrs: {model.NumConstrs}")
        #return []
        # ======================= SOLVE =======================
        print(f"\nSolving...")
        model.write("model.lp")
        model.optimize()
        
        # ======================= RETURN FORMATTED SOLUTION =======================
        if model.status == GRB.OPTIMAL:
            print(f"\n✓ Optimal solution found!")
            # Extract and return formatted solution
            formatted_solution = extract_solution(
                y, z, hw_use, if_use, comm, scs, locations, sensors, actuators,
                cable_types, partitions, hardwares, interfaces, comm_matrix=comm_matrix,
                traffic_flows=traffic_flows, flow=flows, attach_s=attach_s, attach_a=attach_a,
                shared_attach_s=shared_attach_s, shared_attach_a=shared_attach_a,
                shared_trunk_len=shared_trunk_len, shared_extra_trunk_len=shared_extra_trunk_len
            )
            return formatted_solution
        else:
            print(f"\n✗ Optimization failed. Status: {model.status}")
            return None

    def optimize_pareto_epsilon_constraint(self, scs, locations, sensors, actuators, cable_types, comm_matrix,
                                           partitions=None, hardwares=None, interfaces=None,
                                           enable_comm_bw_constraints=True, comm_bw_big_m=10000, num_points=5):
        """
        Build Pareto front between total_cost and cable_length via epsilon-constraint.

        Steps:
        1) Solve min total_cost (gives one extreme)
        2) Solve min cable_length (gives other extreme)
        3) Sweep cable_length upper bound and minimize total_cost for intermediate points
        """
        print("=" * 80)
        print("PARETO OPTIMIZATION (total_cost vs cable_length)")
        print("=" * 80)

        # Extreme 1: minimum cost
        sol_min_cost = self.optimize(
            scs, locations, sensors, actuators, cable_types, comm_matrix,
            partitions=partitions, hardwares=hardwares, interfaces=interfaces,
            enable_comm_bw_constraints=enable_comm_bw_constraints,
            comm_bw_big_m=comm_bw_big_m
        )

        # Extreme 2: minimum cable length
        sol_min_len = self.optimize(
            scs, locations, sensors, actuators, cable_types, comm_matrix,
            partitions=partitions, hardwares=hardwares, interfaces=interfaces,
            enable_comm_bw_constraints=enable_comm_bw_constraints,
            comm_bw_big_m=comm_bw_big_m,
            minimize_cable_length=True
        )

        pareto_solutions = []
        seen = set()

        def _add_unique(sol):
            if not sol or sol.get('status') != 'OPTIMAL':
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

        l_max = float(sol_min_cost.get('cable_length', 0.0))
        l_min = float(sol_min_len.get('cable_length', 0.0))

        if num_points is None:
            num_points = 5
        num_points = max(2, int(num_points))

        # Epsilon sweep between extremes (skip endpoints; already solved)
        if l_max > l_min + 1e-9 and num_points > 2:
            eps_values = np.linspace(l_max, l_min, num_points)[1:-1]
            for eps_len in eps_values:
                sol_eps = self.optimize(
                    scs, locations, sensors, actuators, cable_types, comm_matrix,
                    partitions=partitions, hardwares=hardwares, interfaces=interfaces,
                    enable_comm_bw_constraints=enable_comm_bw_constraints,
                    comm_bw_big_m=comm_bw_big_m,
                    cable_length_limit=float(eps_len) + 1e-6
                )
                _add_unique(sol_eps)

        pareto_solutions.sort(key=lambda s: (float(s.get('cable_length', 0.0)), float(s.get('total_cost', 0.0))))
        print(f"Pareto solutions found: {len(pareto_solutions)}")
        return pareto_solutions


