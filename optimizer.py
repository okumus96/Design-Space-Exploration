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

    def calculate_cable_expressions(self, z, y, scs, locations, sensors, actuators, cable_types, comm_matrix, max_partitions_per_asil_per_loc, feasible_locs_per_sc, comm=None):
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
                    if sensor and sensor.location:
                        dist = get_distance(sensor.location, locations[j].location)
                        cable_cost += dist * cost_map.get(sensor.interface, 0.0)
                        cable_distance += dist
                        latency += dist * latency_map.get(sensor.interface, 0.0)

                # Sum actuator cable costs, distances, and latencies
                for a_id in (sc.actuators or []):
                    actuator = actuator_lookup.get(a_id)
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
    
    def _create_model_and_variables(self, n_sc, n_locs, scs, locations, unique_asils, all_hw, all_interfaces, infeasible_ij=None, comm_matrix=None):
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
        
        # if_use[j,i_name] = Interface i_name port is open at location j (binary)
        if_use = {}
        for j in range(n_locs):
            for i_name in all_interfaces:
                if_use[j, i_name] = model.addVar(vtype=GRB.BINARY, name=f"if_{locations[j].id}_{i_name}")
        
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
        else:
            for j1 in range(n_locs):
                for j2 in range(j1 + 1, n_locs):
                    required_loc_pairs.add((j1, j2))

        comm = {}
        for j1, j2 in sorted(required_loc_pairs):
            for iface in all_interfaces:
                comm[j1, j2, iface] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"comm_{locations[j1].id}_{locations[j2].id}_{iface}")
        
        # w[i, k, j1, j2] = 1 if SC i is at j1 AND SC k is at j2 (Linearization variable)
        # Create w variables for all communicating SC pairs and their feasible locations
        w = {}
        comm_req_pairs = set()
        #if comm_matrix:
        #    sc_id_to_idx = {sc.id: i for i, sc in enumerate(scs)}
        #    for link in comm_matrix:
        #        src_id = link.get('src')
        #        dst_id = link.get('dst')
        #        volume = link.get('volume', 0)
        #        if volume > 0 and src_id in sc_id_to_idx and dst_id in sc_id_to_idx:
        #            i = sc_id_to_idx[src_id]
        #            k = sc_id_to_idx[dst_id]
        #            # Store unique pairs (i, k) 
        #            comm_req_pairs.add((i, k))
        
        feasible_locs_sets = {i: set(feasible_locs_per_sc[i]) for i in range(n_sc)}
        #for (i, k) in comm_req_pairs:
            # We need w for both directions if the link is bidirectional in usage check,
            # but usually we normalize SC pairs. However, traffic aggregation check 
            # iterates over comm_req_map. 
            # Strict definition: w corresponds to existence of specific traffic flow segment.
            
            # Iterate feasible locations
        #    for j1 in feasible_locs_sets[i]:
        #        for j2 in feasible_locs_sets[k]:
        #            if j1 == j2: continue # Internal comms usually don't use network cables
                    
                    # Only create if this location pair connects via backbone
                    # (j1, j2) or (j2, j1) must be in required_loc_pairs
        #            pair = (j1, j2) if j1 < j2 else (j2, j1)
        #            if pair in required_loc_pairs:
        #                 w[i, k, j1, j2] = model.addVar(
        #                     vtype=GRB.BINARY, 
        #                     name=f"w_{scs[i].id}_{scs[k].id}_{locations[j1].id}_{locations[j2].id}"
        #                 )
        #NEW
        # Traffic flow.
        traffic_flows = []
        sc_id_map = {sc.id: i for i, sc in enumerate(scs)}
        t_idx = 0
        if comm_matrix:
            for link in comm_matrix:
                src_id = link.get('src')
                dst_id = link.get('dst')
                vol = link.get('volume', 0)
                # Sadece hacmi olan ve gecerli linkleri al
                if vol > 0 and src_id in sc_id_map and dst_id in sc_id_map:
                    traffic_flows.append({
                        'id': t_idx,
                        'src_idx': sc_id_map[src_id],
                        'dst_idx': sc_id_map[dst_id],
                        'volume': vol
                    })
                    t_idx += 1

        # Flow 
        flow = {}
        for tr in traffic_flows:
            t = tr['id']
            for u in range(n_locs):
                for v in range(n_locs):
                    if u == v: continue
                    flow[t, u, v] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"flow_{t}_{u}_{v}")

        print(f"\nVariables created:")
        print(f"  y: {len(y)} (partitions)")
        print(f"  z: {len(z)} (SC-to-partition)")
        print(f"  hw_use: {len(hw_use)} (HW selection)")
        print(f"  if_use: {len(if_use)} (Interface selection)")
        print(f"  comm: {len(comm)} (ECU-to-ECU communication)")
        print(f"  w: {len(w)} (Linearization vars)")
        print(f"  traffic_flows: {len(traffic_flows)} (Defined communication flows)")
        print(f"  flow: {len(flow)} (Traffic flow)")
        var_stats = {
            'y': len(y),
            'z': len(z),
            'hw_use': len(hw_use),
            'if_use': len(if_use),
            'comm': len(comm),
            'comm_loc_pairs': len(required_loc_pairs),
            'w': len(w),
            'traffic_flows': len(traffic_flows),
            'flow': len(flow),
        }
        var_stats['total'] = var_stats['y'] + var_stats['z'] + var_stats['hw_use'] + var_stats['if_use'] + var_stats['comm'] + var_stats['w'] + var_stats['traffic_flows'] + var_stats['flow']

        return model, y, z, hw_use, if_use, comm, w, traffic_flows, flow, max_partitions_per_asil_per_loc, feasible_locs_per_sc, feasible_scs_per_loc, var_stats

    def _add_constraints(self, model, y, z, hw_use, if_use, comm, w, traffic_flows, flow, scs, locations, sensors, actuators, 
                              cable_types, comm_matrix, partitions, unique_asils, max_partitions_per_asil_per_loc,
                              feasible_locs_per_sc, feasible_scs_per_loc, enable_comm_bw_constraints=True,
                              comm_bw_big_m=None):
        """
        Add all LEGO constraints to the optimization model
        """
        n_sc = len(scs)
        n_locs = len(locations)
        
        print(f"\nAdding constraints...")

        cstats = {
            'c1b_unique_partition': 0,
            'c1d_z_y_relation': 0,
            'c1e_partition_used_only_if_assigned': 0,
            'c2_capacity': 0,
            'c3_hw_required': 0,
            'c4_if_required': 0,
            'c5_redundancy': 0,
            'c8_loc_loc_latency': 0,
            'c9_if_activation': 0,
            'c10_network_load': 0,
            'c10_linearization': 0,
        }
        
        def x_expr(i, j):
            a_req = scs[i].asil_req
            return gp.quicksum(z[i, j, a_req, p] for p in range(max_partitions_per_asil_per_loc) if (i, j, a_req, p) in z)

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
            for a in unique_asils:
                for p in range(max_partitions_per_asil_per_loc):
                    sc_cpu_demand = gp.quicksum(
                        z[i, j, a, p] * scs[i].cpu_req
                        for i in feasible_scs_per_loc[j] if scs[i].asil_req == a and (i, j, a, p) in z
                    )
                    model.addConstr(
                        sc_cpu_demand <= y[j, a, p] * partitions.get('cpu_cap', float('inf')),
                        name=f"capacity_{locations[j].id}_asil{a}_p{p}"
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
        
        # Constraint 4: Interface required
        for i in range(n_sc):
            for i_name in (scs[i].interface_required or []):
                for j in feasible_locs_per_sc[i]:
                    model.addConstr(
                        x_expr(i, j) <= if_use[j, i_name],
                        name=f"if_required_{scs[i].id}_{i_name}_{locations[j].id}"
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
        
        # Constraint 6/7 (Sensor/Actuator max-latency): precomputed and enforced via sparse z creation.
        latency_map = build_latency_map(cable_types)
        
        # Constraint 8: Location-Location Communication Latency Constraints
        if comm_matrix:
            cost_map = build_cost_map(cable_types)
            loc_loc_latencies = {}
            all_interfaces = list(cable_types.keys())
            
            for j1 in range(n_locs):
                for j2 in range(j1 + 1, n_locs):
                    dist = get_distance(locations[j1].location, locations[j2].location)
                    if all_interfaces:
                        best_iface = max(all_interfaces, key=lambda x: cost_map.get(x, 0))
                        loc_loc_latencies[j1, j2] = dist * latency_map.get(best_iface, 0)
                    else:
                        loc_loc_latencies[j1, j2] = dist * 1e6
            
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
                        
                        a, b = (j1, j2) if j1 < j2 else (j2, j1)
                        lat = loc_loc_latencies.get((a, b), 0)
                        if lat > max_lat:
                            model.addConstr(
                                x_expr(u, j1)
                                + x_expr(v, j2)
                                <= 1,
                                name=f"loc_loc_lat_{u}_{v}_{j1}_{j2}"
                            )
                            cstats['c8_loc_loc_latency'] += 1
        
        # Constraint 9: ECU-to-ECU Communication - Interface Activation (Big-M)
        # If comm[j1,j2,iface] > 0, then if_use[j1,iface]=1 and if_use[j2,iface]=1
        M = 1000  # Big-M value
        all_interfaces = list(cable_types.keys())
        for (j1, j2, iface), comm_var in comm.items():
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
        
        # Constraint 10: Aggregated Traffic Capacity (Network Synthesis) - Linearized
        # To avoid Quadratic Constraints, we linearize the product x_expr(i, j1) * x_expr(k, j2)
        # We use the pre-created auxiliary variable w[i, k, j1, j2] = 1 iff SC i at j1 AND SC k at j2
        
        cstats['c10_network_load'] = 0
        cstats['c10_linearization'] = 0

        if enable_comm_bw_constraints and comm_matrix:
            print("  Building network traffic aggregation constraints (Linearized)...")
            
            # 1. Group SC communication by SC pairs
            sc_id_to_idx = {sc.id: i for i, sc in enumerate(scs)}
            comm_req_map = {} 
            for link in comm_matrix:
                src_id = link.get('src')
                dst_id = link.get('dst')
                volume = link.get('volume', 0)
                if volume > 0 and src_id in sc_id_to_idx and dst_id in sc_id_to_idx:
                    src_idx = sc_id_to_idx[src_id]
                    dst_idx = sc_id_to_idx[dst_id]
                    i_min, i_max = (src_idx, dst_idx) if src_idx < dst_idx else (dst_idx, src_idx)
                    comm_req_map[(i_min, i_max)] = comm_req_map.get((i_min, i_max), 0) + volume
            
            # Precompute feasible location sets for faster lookup
            feasible_locs_sets = {i: set(feasible_locs_per_sc[i]) for i in range(n_sc)}

            # Get all unique location pairs that have potential cables
            loc_pairs = set()
            for (j1, j2, iface) in comm.keys():
                loc_pairs.add((j1, j2))

            # 1. Add Linearization Constraints for all valid w variables
            # These are independent of traffic volume or cable capacity
            for (i, k, j1, j2), w_var in w.items():
                x_i_j1 = x_expr(i, j1)
                x_k_j2 = x_expr(k, j2)
                
                # McCormick envelopes
                model.addConstr(w_var <= x_i_j1, name=f"lin_w_leq_i_{i}_{k}_{j1}_{j2}")
                model.addConstr(w_var <= x_k_j2, name=f"lin_w_leq_k_{i}_{k}_{j1}_{j2}")
                model.addConstr(w_var >= x_i_j1 + x_k_j2 - 1, name=f"lin_w_geq_{i}_{k}_{j1}_{j2}")
                cstats['c10_linearization'] += 3

            # 2. Add Traffic Capacity Constraints per link
            for (j1, j2) in sorted(loc_pairs):
                # Left Side: Total Traffic Demand between j1 and j2
                traffic_expr = gp.LinExpr()
                has_traffic = False

                # Sum traffic for all SC pairs (i, k) that *could* result in traffic on this link
                for (i, k), volume in comm_req_map.items():
                    # Check forward direction: i->j1, k->j2
                    if (i, k, j1, j2) in w:
                         traffic_expr.add(w[i, k, j1, j2], volume)
                         has_traffic = True
                    
                    # Check reverse direction: k->j1, i->j2 (assuming full-duplex or shared medium capacity)
                    if (k, i, j1, j2) in w:
                         traffic_expr.add(w[k, i, j1, j2], volume)
                         has_traffic = True

                # Right Side: Total Physical Capacity provided by cables (sum of capacity of all cables)
                capacity_expr = gp.LinExpr()
                has_cables = False
                for iface in all_interfaces:
                    if (j1, j2, iface) in comm:
                         capacity_expr.add(comm[j1, j2, iface], cable_types[iface].capacity)
                         has_cables = True
                
                # Add constraint only if there is potential traffic and potential cables
                if has_traffic and has_cables:
                    model.addConstr(
                        traffic_expr <= capacity_expr,
                        name=f"c10_traffic_agg_{locations[j1].id}_{locations[j2].id}"
                    )
                    cstats['c10_network_load'] += 1
        
        # ETH network model constraints
        ##NEW
        # Her bir trafik akisi (t) ve her bir lokasyon (j) icin:
        for tr in traffic_flows:
            t_id = tr['id']
            src_idx = tr['src_idx']
            dst_idx = tr['dst_idx']
            
            for j in range(n_locs):
                # 1. Bu lokasyona GIREN toplam trafik (baska lokasyonlardan)
                flow_in = gp.quicksum(flow[t_id, k, j] for k in range(n_locs) if k != j)
                
                # 2. Bu lokasyondan CIKAN toplam trafik (baska lokasyonlara)
                flow_out = gp.quicksum(flow[t_id, j, k] for k in range(n_locs) if k != j)
                
                # 3. Bu lokasyon bu trafigin KAYNAGI mi? (SC i burada mi?)
                # (Tum partition ve asil olasiliklarini topluyoruz, sonuc 0 veya 1 olur)
                is_source = gp.quicksum(z[src_idx, j, a, p] 
                                    for a in unique_asils 
                                    for p in range(max_partitions_per_asil_per_loc)
                                    if (src_idx, j, a, p) in z)
                
                # 4. Bu lokasyon bu trafigin HEDEFI mi? (SC k burada mi?)
                is_dest = gp.quicksum(z[dst_idx, j, a, p] 
                                    for a in unique_asils 
                                    for p in range(max_partitions_per_asil_per_loc)
                                    if (dst_idx, j, a, p) in z)
                
                # DENKLEM: Giren + Uretilen = Cikan + Tuketinlen
                # Matematiksel olarak: Flow_Out - Flow_In = Source - Dest
                model.addConstr(flow_out - flow_in == is_source - is_dest, 
                                name=f"flow_conserv_{t_id}_{locations[j].id}")
                
        for (u, v, iface), comm_var in comm.items():
            # Sadece ETH kablolarini network routing icin kullaniyoruz
            if "ETH" not in iface: 
                continue

            # u -> v yonundeki trafik toplami
            traffic_u_v = gp.LinExpr()
            
            # v -> u yonundeki trafik toplami (Ethernet Full Duplex oldugu icin ayri ayri kontrol ediyoruz)
            traffic_v_u = gp.LinExpr()
            
            for tr in traffic_flows:
                t_id = tr['id']
                vol = tr['volume']
                
                # Eger bu flow degiskeni tanimliysa toplama ekle
                if (t_id, u, v) in flow:
                    traffic_u_v += flow[t_id, u, v] * vol
                    
                if (t_id, v, u) in flow:
                    traffic_v_u += flow[t_id, v, u] * vol
                    
            # KISIT 1: u->v trafigi kapasiteye sigmali
            model.addConstr(traffic_u_v <= cable_types[iface].capacity * comm_var, 
                            name=f"cap_uv_{u}_{v}")
                            
            # KISIT 2: v->u trafigi kapasiteye sigmali
            model.addConstr(traffic_v_u <= cable_types[iface].capacity * comm_var, 
                            name=f"cap_vu_{u}_{v}")
            
        for tr in traffic_flows:
            t_id = tr['id']
            src_idx = tr['src_idx']
            
            for j in range(n_locs):
                # Bu lokasyondan DISARI cikan trafik miktari (0 ile 1 arasi)
                flow_out_sum = gp.quicksum(flow[t_id, j, k] for k in range(n_locs) if k != j)
                
                # Bu lokasyon KAYNAK mi?
                is_source = gp.quicksum(z[src_idx, j, a, p] 
                                    for a in unique_asils 
                                    for p in range(max_partitions_per_asil_per_loc)
                                    if (src_idx, j, a, p) in z)
                
                # KISIT: Disari trafik cikabilmesi icin YA kaynak olmali YA DA switch olmali.
                # Flow_Out (max 1) <= Source (0/1) + Switch (0/1)
                model.addConstr(flow_out_sum <= is_source + hw_use[j, "SWITCH"], 
                                name=f"switch_perm_{t_id}_{locations[j].id}")


        cstats['total'] = sum(v for k, v in cstats.items() if k != 'total')
        return model, cstats

    def _build_objective_function(self, model, y, z, hw_use, if_use, comm, scs, locations, sensors, actuators, 
                                  cable_types, comm_matrix, partitions, hardwares, interfaces,
                                  n_locs, unique_asils, all_hw, all_interfaces, max_partitions_per_asil_per_loc,
                                  feasible_locs_per_sc):
        """
        Build and set the objective function for the optimization model
        
        Objective: Minimize total cost = partition_cost + hw_cost + interface_cost + cable_cost + comm_cost
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
        cable_cost_expr, _, _ = self.calculate_cable_expressions(
            z, {}, scs, locations, sensors, actuators, cable_types, comm_matrix, max_partitions_per_asil_per_loc,
            feasible_locs_per_sc, comm=comm
        )
        
        # Total cost
        total_cost = partition_cost_expr + hw_cost_expr + if_cost_expr + cable_cost_expr
        model.setObjective(total_cost, GRB.MINIMIZE)
        
        print(f"Objective: Partition + HW + Interface + Cable + Communication costs")
        
        return model

    def optimize(self, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=None, hardwares=None, interfaces=None, enable_comm_bw_constraints=True, comm_bw_big_m=10000):
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
        model, y, z, hw_use, if_use, comm, w, traffic_flows, flow, max_partitions_per_asil_per_loc, feasible_locs_per_sc, feasible_scs_per_loc, var_stats = self._create_model_and_variables(
            n_sc, n_locs, scs, locations, unique_asils, all_hw, all_interfaces, infeasible_ij=infeasible_ij, comm_matrix=comm_matrix
        )
        
        # ======================= ADD CONSTRAINTS =======================
        model, cstats = self._add_constraints(
            model, y, z, hw_use, if_use, comm, w, traffic_flows, flow, scs, locations, sensors, actuators,
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
        print(f"  comm: {var_stats['comm']} ({(100.0 * var_stats['comm'] / max(1, var_stats['total'])):.1f}%)")
        print(f"  w (linearization): {var_stats['w']}")
        print(f"  comm_loc_pairs: {var_stats['comm_loc_pairs']}")

        print("\n[DEBUG] Constraint breakdown:")
        print(f"  total constr: {cstats['total']}")
        print(f"  c1b_unique_partition: {cstats['c1b_unique_partition']}")
        print(f"  c1d_z_y_relation: {cstats['c1d_z_y_relation']}")
        print(f"  c1e_partition_used_only_if_assigned: {cstats['c1e_partition_used_only_if_assigned']}")
        print(f"  c2_capacity: {cstats['c2_capacity']}")
        print(f"  c3_hw_required: {cstats['c3_hw_required']}")
        print(f"  c4_if_required: {cstats['c4_if_required']}")
        print(f"  c5_redundancy: {cstats['c5_redundancy']}")
        print(f"  c8_loc_loc_latency: {cstats['c8_loc_loc_latency']}")
        print(f"  c9_if_activation: {cstats['c9_if_activation']}")
        print(f"  c10_network_load: {cstats['c10_network_load']} ({(100.0 * cstats['c10_network_load'] / max(1, cstats['total'])):.1f}%)")
        print(f"  c10_linearization: {cstats['c10_linearization']} ({(100.0 * cstats['c10_linearization'] / max(1, cstats['total'])):.1f}%)")
        
        # ======================= BUILD OBJECTIVE =======================
        model = self._build_objective_function(
            model, y, z, hw_use, if_use, comm, scs, locations, sensors, actuators,
            cable_types, comm_matrix, partitions, hardwares, interfaces,
            n_locs, unique_asils, all_hw, all_interfaces, max_partitions_per_asil_per_loc,
            feasible_locs_per_sc
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
                y, z, hw_use, if_use, comm, w, scs, locations, sensors, actuators,
                cable_types, partitions, hardwares, interfaces, comm_matrix=comm_matrix
            )
            return formatted_solution
        else:
            print(f"\n✗ Optimization failed. Status: {model.status}")
            return None


