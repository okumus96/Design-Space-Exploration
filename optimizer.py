import os
import numpy as np
os.environ["GRB_LICENSE_FILE"] = "/home/okumus/gurobi.lic"
import gurobipy as gp
from gurobipy import GRB

# Choosing our interface b/w ecus unintiutive.
# Interfaces right now have infinite capacity.
# We need change direct link b/w the ECUs to some sort of network topology.
# we need to get rid of the ECU types, model them as a legos of HWs and partitions.
# We did not include uncertainty in overall system.
# We do not know numbers we gave is correct (i.e., cost, latency,  etc.)
# WE do not know our objective functions is correct. (semanatically, we can change distance with complexity model)



class AssignmentOptimizer:
    def __init__(self):
        print("Initialized AssignmentOptimizerNew with quadratic ECU-ECU costs and linear sensor/actuator costs")
        pass
    
    def _extract_solution(self, x, y, z, hw_use, if_use, comm, scs, locations, sensors, actuators, 
                               cable_types, partitions, hardwares, interfaces):

        """Extract LEGO solution from optimized model"""
        assignment_map = {}
        partition_map = {}
        num_locations_used = 0
        num_partitions_opened = 0
        hw_opened = []
        if_opened = []
        
        # Extract SC assignments and partitions from z
        for (i, j, a, p), var in z.items():
            if var.X > 0.5:
                assignment_map[scs[i].id] = locations[j].id
                partition_map[scs[i].id] = f"{locations[j].id}_asil{a}_p{p}"
        
        # Extract partitions opened
        for (j, a, p), var in y.items():
            if var.X > 0.5:
                num_partitions_opened += 1
        
        # Extract HW opened
        for (j, h), var in hw_use.items():
            if var.X > 0.5:
                hw_opened.append(f"{h}@{locations[j].id}")
        
        # Extract interfaces opened
        for (j, i_name), var in if_use.items():
            if var.X > 0.5:
                if_opened.append(f"{i_name}@{locations[j].id}")
        
        # Count locations used
        locs_used = set()
        for (i, j, a, p), var in z.items():
            if var.X > 0.5:
                locs_used.add(j)
        num_locations_used = len(locs_used)
        
        # Calculate cable costs and distances
        cable_length = 0.0
        cable_cost_calc = 0.0
        cost_map = {name: ct.cost_per_meter for name, ct in cable_types.items()}
        sensor_lookup = {s.id: s for s in sensors}
        actuator_lookup = {a.id: a for a in actuators}
        
        for (i, j), var in x.items():
            if var.X > 0.5:
                sc = scs[i]
                ecu = locations[j]
                
                # Sensor cable costs and distances
                for s_id in sc.sensors:
                    sensor = sensor_lookup.get(s_id)
                    if sensor and sensor.location:
                        dist = self._get_distance(sensor.location, locations[j].location)
                        cable_length += dist
                        cable_cost_calc += dist * cost_map.get(sensor.interface, 0.0)
                
                # Actuator cable costs and distances
                for a_id in sc.actuators:
                    actuator = actuator_lookup.get(a_id)
                    if actuator and actuator.location:
                        dist = self._get_distance(actuator.location, locations[j].location)
                        cable_length += dist
                        cable_cost_calc += dist * cost_map.get(actuator.interface, 0.0)
        
        # Calculate costs
        partition_cost = num_partitions_opened * partitions.get('cost', 0)
        hw_cost = sum(hardwares.get(h.split('@')[0], 0) for h in hw_opened)
        if_cost = sum(interfaces[i.split('@')[0]].port_cost for i in if_opened)
        cable_cost = cable_cost_calc
        
        # Calculate communication cost (ECU-to-ECU backbone)
        comm_cost = 0.0
        for (j1, j2, iface), var in comm.items():
            if var.X > 0.5:
                num_links = int(round(var.X))
                dist = self._get_distance(locations[j1].location, locations[j2].location)
                cost_per_link = dist * cable_types[iface].cost_per_meter
                comm_cost += num_links * cost_per_link
        
        total_cost = partition_cost + hw_cost + if_cost + cable_cost + comm_cost
        
        solution = {
            'assignment': assignment_map,
            'partitions': partition_map,
            'hardware_cost': hw_cost,  # For visualizer compatibility
            'hw_cost': hw_cost,
            'interface_cost': if_cost,
            'cable_cost': cable_cost,
            'comm_cost': comm_cost,
            'cable_length': cable_length,
            'total_cost': total_cost,
            'num_locations_used': num_locations_used,
            'num_ecus_used': num_locations_used,  # For visualizer compatibility
            'num_partitions': num_partitions_opened,
            'hw_features': hw_opened,
            'interfaces': if_opened,
            'partition_cost': partition_cost,
            'objective': 'LEGO Total Cost',
            'status': 'OPTIMAL'
        }
        return solution

    def _get_distance(self, loc1, loc2):
        if not loc1 or not loc2:
            return 0.0
        dist_manhattan, _ = loc1.dist(loc2)
        return dist_manhattan
    
    def calculate_cable_expressions(self, x, y, scs, locations, sensors, actuators, cable_types, comm_matrix):
        """
        Calculate cable cost, distance, and latency as Gurobi linear expressions (NOT constraints).
        Includes: sensor cables and actuator cables only (no ECU-ECU cables).
        
        Returns:
            cable_cost_expr: Gurobi expression for total cable costs
            cable_distance_expr: Gurobi expression for total cable distance/length
            latency_expr: Gurobi expression for total latency (distance-based)
        """
        # Precompute lookups
        cost_map = {name: ct.cost_per_meter for name, ct in cable_types.items()}
        latency_map = {name: ct.latency_per_meter for name, ct in cable_types.items()}
        #print("Cable Types Cost Map:", cost_map)
        sensor_lookup = {s.id: s for s in sensors}
        actuator_lookup = {a.id: a for a in actuators}
        #print("Sensor Lookup:", sensor_lookup)
        #print("Actuator Lookup:", actuator_lookup)
        
        # Precompute cable cost, distance, and latency coefficients (only non-zero)
        cable_cost_terms = []  # List of (variable, cable_cost) tuples
        cable_distance_terms = []  # List of (variable, distance) tuples
        latency_terms = []  # List of (variable, latency) tuples
        
        for (i, j), _ in x.items():
            sc = scs[i]
            ecu = locations[j]
            cable_cost = 0.0
            cable_distance = 0.0
            latency = 0.0
            
            # Sum sensor cable costs, distances, and latencies
            for s_id in sc.sensors:
                sensor = sensor_lookup.get(s_id)
                if sensor and sensor.location:
                    dist = self._get_distance(sensor.location, locations[j].location)
                    cable_cost += dist * cost_map.get(sensor.interface, 0.0)
                    cable_distance += dist
                    latency += dist * latency_map.get(sensor.interface, 0.0)
            
            # Sum actuator cable costs, distances, and latencies
            for a_id in sc.actuators:
                actuator = actuator_lookup.get(a_id)
                if actuator and actuator.location:
                    dist = self._get_distance(actuator.location, locations[j].location)
                    cable_cost += dist * cost_map.get(actuator.interface, 0.0)
                    cable_distance += dist
                    latency += dist * latency_map.get(actuator.interface, 0.0)
            
            # Only add to expression if values are non-zero
            if cable_cost > 0:
                cable_cost_terms.append((x[i, j], cable_cost))
            if cable_distance > 0:
                cable_distance_terms.append((x[i, j], cable_distance))
            if latency > 0:
                latency_terms.append((x[i, j], latency))

        # ECU-ECU backbone contributions (quadratic in y)
        # Skip if y dictionary is empty (e.g., in LEGO model where partition-level partitioning is used)
        ecu_ecu_cost_terms = []
        ecu_ecu_distance_terms = []
        ecu_ecu_latency_terms = []

        if y:  # Only compute ECU-ECU backbone if y variables exist
            for j1 in range(len(locations)):
                for j2 in range(j1 + 1, len(locations)):
                    dist = self._get_distance(locations[j1].location, locations[j2].location)
                    if dist <= 0:
                        continue

                    common = set(locations[j1].interface_offered).intersection(set(locations[j2].interface_offered))
                    if not common:
                        cost = dist * 1e6
                        lat = dist * 1e6
                    else:
                        best_iface = max(common, key=lambda x: cost_map.get(x, 0))
                        cost = dist * cost_map.get(best_iface, 0)
                        lat = dist * latency_map.get(best_iface, 0)

                    ecu_ecu_distance_terms.append((y[j1], y[j2], dist))
                    ecu_ecu_cost_terms.append((y[j1], y[j2], cost))
                    ecu_ecu_latency_terms.append((y[j1], y[j2], lat))
        
        # Create compact linear expressions
        if cable_cost_terms:
            cable_cost_expr = gp.quicksum(var * cost for var, cost in cable_cost_terms)
        else:
            cable_cost_expr = 0
        if ecu_ecu_cost_terms:
            cable_cost_expr += gp.quicksum(y1 * y2 * cost for y1, y2, cost in ecu_ecu_cost_terms)
            
        if cable_distance_terms:
            cable_distance_expr = gp.quicksum(var * dist for var, dist in cable_distance_terms)
        else:
            cable_distance_expr = 0
        if ecu_ecu_distance_terms:
            cable_distance_expr += gp.quicksum(y1 * y2 * dist for y1, y2, dist in ecu_ecu_distance_terms)
            
        if latency_terms:
            latency_expr = gp.quicksum(var * lat for var, lat in latency_terms)
        else:
            latency_expr = 0
        if ecu_ecu_latency_terms:
            latency_expr += gp.quicksum(y1 * y2 * lat for y1, y2, lat in ecu_ecu_latency_terms)
            
        return cable_cost_expr, cable_distance_expr, latency_expr
    
    def _create_model_and_variables(self, n_sc, n_locs, scs, locations, unique_asils, all_hw, all_interfaces):
        """
        Create Gurobi model and define all decision variables
        
        Returns:
            model: Gurobi model object
            x: SC-to-location assignment variables
            y: Partition opening variables
            hw_use: HW feature usage variables
            if_use: Interface usage variables
            max_partitions_per_asil_per_loc: Maximum partitions per location per ASIL
        """
        # Create model
        model = gp.Model("LEGO_Assignment")
        model.setParam('OutputFlag', 1)
        model.setParam('MIPFocus', 2)
        model.update()
        
        # x[i,j] = SC i assigned to location j (binary)
        x = {}
        for i in range(n_sc):
            for j in range(n_locs):
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{scs[i].id}_{locations[j].id}")
        
        # y[j,a,p] = Location j has partition for ASIL a (replica p)
        max_partitions_per_asil_per_loc = 1
        y = {}
        for j in range(n_locs):
            for a in unique_asils:
                for p in range(max_partitions_per_asil_per_loc):
                    y[j, a, p] = model.addVar(vtype=GRB.BINARY, name=f"y_{locations[j].id}_asil{a}_p{p}")
        
        # z[i,j,a,p] = SC i assigned to location j ASIL a partition p
        z = {}
        for i in range(n_sc):
            a = scs[i].asil_req
            for j in range(n_locs):
                for p in range(max_partitions_per_asil_per_loc):
                    z[i, j, a, p] = model.addVar(vtype=GRB.BINARY, name=f"z_{scs[i].id}_{locations[j].id}_asil{a}_p{p}")
        
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
        comm = {}
        for j1 in range(n_locs):
            for j2 in range(j1 + 1, n_locs):
                for iface in all_interfaces:
                    comm[j1, j2, iface] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"comm_{locations[j1].id}_{locations[j2].id}_{iface}")
        
        print(f"\nVariables created:")
        print(f"  x: {len(x)} (SC-to-location)")
        print(f"  y: {len(y)} (partitions)")
        print(f"  z: {len(z)} (SC-to-partition)")
        print(f"  hw_use: {len(hw_use)} (HW selection)")
        print(f"  if_use: {len(if_use)} (Interface selection)")
        print(f"  comm: {len(comm)} (ECU-to-ECU communication)")
        
        return model, x, y, z, hw_use, if_use, comm, max_partitions_per_asil_per_loc

    def _add_constraints(self, model, x, y, z, hw_use, if_use, comm, scs, locations, sensors, actuators, 
                              cable_types, comm_matrix, partitions, unique_asils, max_partitions_per_asil_per_loc):
        """
        Add all LEGO constraints to the optimization model
        """
        n_sc = len(scs)
        n_locs = len(locations)
        
        print(f"\nAdding constraints...")
        
        # Constraint 1: Every SC assigned to exactly one location
        for i in range(n_sc):
            model.addConstr(
                gp.quicksum(x[i, j] for j in range(n_locs)) == 1,
                name=f"assign_{scs[i].id}"
            )
        
        # Constraint 1b: Every SC assigned to exactly one partition
        for i in range(n_sc):
            a = scs[i].asil_req
            model.addConstr(
                gp.quicksum(
                    z[i, j, a, p]
                    for j in range(n_locs)
                    for p in range(max_partitions_per_asil_per_loc)
                ) == 1,
                name=f"unique_partition_{scs[i].id}"
            )
        
        # Constraint 1c: z relationship with x (if SC at location, it's in some partition)
        for i in range(n_sc):
            a = scs[i].asil_req
            for j in range(n_locs):
                model.addConstr(
                    gp.quicksum(z[i, j, a, p] for p in range(max_partitions_per_asil_per_loc)) == x[i, j],
                    name=f"z_x_relation_{scs[i].id}_{locations[j].id}"
                )
        
        # Constraint 1d: z relationship with y (if SC at partition, partition is open)
        for i in range(n_sc):
            a = scs[i].asil_req
            for j in range(n_locs):
                for p in range(max_partitions_per_asil_per_loc):
                    model.addConstr(
                        z[i, j, a, p] <= y[j, a, p],
                        name=f"z_y_relation_{scs[i].id}_{locations[j].id}_p{p}"
                    )
        
        # Constraint 2: Capacity per location per ASIL partition
        for j in range(n_locs):
            for a in unique_asils:
                for p in range(max_partitions_per_asil_per_loc):
                    sc_cpu_demand = gp.quicksum(
                        z[i, j, a, p] * scs[i].cpu_req
                        for i in range(n_sc) if scs[i].asil_req == a
                    )
                    model.addConstr(
                        sc_cpu_demand <= y[j, a, p] * partitions.get('cpu_cap', float('inf')),
                        name=f"capacity_{locations[j].id}_asil{a}_p{p}"
                    )
        
        # Constraint 3: HW feature required
        for i in range(n_sc):
            for h in (scs[i].hw_required or []):
                for j in range(n_locs):
                    model.addConstr(
                        x[i, j] <= hw_use[j, h],
                        name=f"hw_required_{scs[i].id}_{h}"
                    )
        
        # Constraint 4: Interface required
        for i in range(n_sc):
            for i_name in (scs[i].interface_required or []):
                for j in range(n_locs):
                    model.addConstr(
                        x[i, j] <= if_use[j, i_name],
                        name=f"if_required_{scs[i].id}_{i_name}"
                    )
        
        # Constraint 5: Redundancy Constraints
        sc_id_to_idx = {sc.id: i for i, sc in enumerate(scs)}
        for i, sc in enumerate(scs):
            if hasattr(sc, 'redundant_with') and sc.redundant_with:
                partner_id = sc.redundant_with
                if partner_id in sc_id_to_idx:
                    partner_idx = sc_id_to_idx[partner_id]
                    if i < partner_idx:
                        for j in range(n_locs):
                            model.addConstr(
                                x[i, j] + x[partner_idx, j] <= 1,
                                name=f"redundancy_{sc.id}_{partner_id}_{locations[j].id}"
                            )
        
        # Constraint 6: Sensor Latency Constraints
        latency_map = {name: ct.latency_per_meter for name, ct in cable_types.items()}
        sensor_lookup = {s.id: s for s in sensors}
        for i in range(n_sc):
            for s_id in scs[i].sensors:
                sensor = sensor_lookup.get(s_id)
                if sensor and hasattr(sensor, 'max_latency') and sensor.max_latency:
                    for j in range(n_locs):
                        if sensor.location and locations[j].location:
                            dist = self._get_distance(sensor.location, locations[j].location)
                            latency = dist * latency_map.get(sensor.interface, 0.0)
                            if latency > sensor.max_latency:
                                model.addConstr(
                                    x[i, j] == 0,
                                    name=f"sensor_lat_{scs[i].id}_{sensor.id}_{locations[j].id}"
                                )
        
        # Constraint 7: Actuator Latency Constraints
        actuator_lookup = {a.id: a for a in actuators}
        for i in range(n_sc):
            for a_id in scs[i].actuators:
                actuator = actuator_lookup.get(a_id)
                if actuator and hasattr(actuator, 'max_latency') and actuator.max_latency:
                    for j in range(n_locs):
                        if actuator.location and locations[j].location:
                            dist = self._get_distance(actuator.location, locations[j].location)
                            latency = dist * latency_map.get(actuator.interface, 0.0)
                            if latency > actuator.max_latency:
                                model.addConstr(
                                    x[i, j] == 0,
                                    name=f"actuator_lat_{scs[i].id}_{actuator.id}_{locations[j].id}"
                                )
        
        # Constraint 8: Location-Location Communication Latency Constraints
        if comm_matrix:
            cost_map = {name: ct.cost_per_meter for name, ct in cable_types.items()}
            loc_loc_latencies = {}
            all_interfaces = list(cable_types.keys())
            
            for j1 in range(n_locs):
                for j2 in range(j1 + 1, n_locs):
                    dist = self._get_distance(locations[j1].location, locations[j2].location)
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
                
                for j1 in range(n_locs):
                    for j2 in range(n_locs):
                        if j1 == j2:
                            continue
                        
                        a, b = (j1, j2) if j1 < j2 else (j2, j1)
                        lat = loc_loc_latencies.get((a, b), 0)
                        if lat > max_lat:
                            model.addConstr(
                                x[u, j1] + x[v, j2] <= 1,
                                name=f"loc_loc_lat_{u}_{v}_{j1}_{j2}"
                            )
        
        # Constraint 9: ECU-to-ECU Communication - Interface Activation (Big-M)
        # If comm[j1,j2,iface] > 0, then if_use[j1,iface]=1 and if_use[j2,iface]=1
        M = 1000  # Big-M value
        all_interfaces = list(cable_types.keys())
        for j1 in range(n_locs):
            for j2 in range(j1 + 1, n_locs):
                for iface in all_interfaces:
                    model.addConstr(
                        comm[j1, j2, iface] <= M * if_use[j1, iface],
                        name=f"if_act_j1_{locations[j1].id}_{locations[j2].id}_{iface}"
                    )
                    model.addConstr(
                        comm[j1, j2, iface] <= M * if_use[j2, iface],
                        name=f"if_act_j2_{locations[j1].id}_{locations[j2].id}_{iface}"
                    )
        
        # Constraint 9b: Communication Bandwidth Requirement
        # Build a map of communication requirements (volume) between SC pairs
        sc_id_to_idx = {sc.id: i for i, sc in enumerate(scs)}
        comm_req_map = {}  # (sc_i_idx, sc_k_idx) -> volume (bandwidth requirement)
        for link in comm_matrix:
            src_id = link.get('src')
            dst_id = link.get('dst')
            volume = link.get('volume', 0)
            if volume > 0 and src_id in sc_id_to_idx and dst_id in sc_id_to_idx:
                src_idx = sc_id_to_idx[src_id]
                dst_idx = sc_id_to_idx[dst_id]
                # Normalize so src_idx < dst_idx
                i_min, i_max = (src_idx, dst_idx) if src_idx < dst_idx else (dst_idx, src_idx)
                comm_req_map[(i_min, i_max)] = volume
        
        # Add constraints: If SC i at j1 and SC k at j2 (j1 != j2), then ensure BW
        M_big = 100000
        for (i, k), bw_req in comm_req_map.items():
            for j1 in range(n_locs):
                for j2 in range(n_locs):
                    if j1 == j2:
                        continue
                    
                    # Determine comm direction: use min/max indices for lookup in comm dict
                    j_min, j_max = (j1, j2) if j1 < j2 else (j2, j1)
                    
                    # BW constraint with Big-M activation
                    # This constraint is active only when x[i,j1]=1 AND x[k,j2]=1
                    # Using: BW >= requirement × (x[i,j1] + x[k,j2] - 1)
                    # Rearranged: BW >= requirement - M×(2 - x[i,j1] - x[k,j2])
                    bw_expr = gp.quicksum(
                        comm[j_min, j_max, iface] * cable_types[iface].capacity
                        for iface in all_interfaces
                    )
                    model.addConstr(
                        bw_expr >= bw_req - M_big * (2 - x[i, j1] - x[k, j2]),
                        name=f"comm_bw_{scs[i].id}_{scs[k].id}_{locations[j1].id}_{locations[j2].id}"
                    )
        
        return model

    def _build_objective_function(self, model, x, y, hw_use, if_use, comm, scs, locations, sensors, actuators, 
                                  cable_types, comm_matrix, partitions, hardwares, interfaces,
                                  n_locs, unique_asils, all_hw, all_interfaces, max_partitions_per_asil_per_loc):
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
        
        # Cable cost (sensor/actuator cables only)
        cable_cost_expr, _, _ = self.calculate_cable_expressions(
            x, {}, scs, locations, sensors, actuators, cable_types, comm_matrix
        )
        
        # Communication cost (ECU-to-ECU backbone)
        comm_cost_expr = 0
        for j1 in range(n_locs):
            for j2 in range(j1 + 1, n_locs):
                dist = self._get_distance(locations[j1].location, locations[j2].location)
                for iface in all_interfaces:
                    cost_per_unit = dist * cable_types[iface].cost_per_meter
                    comm_cost_expr += comm[j1, j2, iface] * cost_per_unit
        
        # Total cost
        total_cost = partition_cost_expr + hw_cost_expr + if_cost_expr + cable_cost_expr + comm_cost_expr
        model.setObjective(total_cost, GRB.MINIMIZE)
        
        print(f"Objective: Partition + HW + Interface + Cable + Communication costs")
        
        return model

    def optimize(self, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=None, hardwares=None, interfaces=None):
        """
        LEGO-based optimization: Dynamic selection of partitions + HW + interfaces + cables.
        
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
        
        # ======================= CREATE MODEL & VARIABLES =======================
        model, x, y, z, hw_use, if_use, comm, max_partitions_per_asil_per_loc = self._create_model_and_variables(
            n_sc, n_locs, scs, locations, unique_asils, all_hw, all_interfaces
        )
        
        # ======================= ADD CONSTRAINTS =======================
        model = self._add_constraints(
            model, x, y, z, hw_use, if_use, comm, scs, locations, sensors, actuators,
            cable_types, comm_matrix, partitions, unique_asils, max_partitions_per_asil_per_loc
        )
        
        # ======================= BUILD OBJECTIVE =======================
        model = self._build_objective_function(
            model, x, y, hw_use, if_use, comm, scs, locations, sensors, actuators,
            cable_types, comm_matrix, partitions, hardwares, interfaces,
            n_locs, unique_asils, all_hw, all_interfaces, max_partitions_per_asil_per_loc
        )
        
        # ======================= SOLVE =======================
        print(f"\nSolving...")
        model.write("model.lp")
        model.optimize()
        solutions = []
        # ======================= EXTRACT SOLUTION =======================
        if model.status == GRB.OPTIMAL:
            print(f"\n✓ Optimal solution found!")
            sol = self._extract_solution(x, y, z, hw_use, if_use, comm, scs, locations, sensors, actuators, cable_types, partitions, hardwares, interfaces)
            solutions.append(sol)
            return solutions
        else:
            print(f"\n✗ Optimization failed. Status: {model.status}")
            return []


