import os
os.environ["GRB_LICENSE_FILE"] = "/home/frk/gurobi.lic"
import gurobipy as gp
from gurobipy import GRB


## Add container and container communication latency later
## Consider ASIL decomposition later
## Add load balancing later

## Add interface count (5 ETH) later



class AssignmentOptimizer:
    """
    SC -> ECU Assignment Optimization.
    
    Objectives: Minimize HW Cost and Cable Length (Pareto front)
    """
    
    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    def preprocess_communications(self, scs, comm_matrix):
        comm_pairs = []
        if comm_matrix:
            sc_id_map = {sc.id: i for i, sc in enumerate(scs)}
            for link in comm_matrix:
                u_id = link.get('src')
                v_id = link.get('dst')
                if u_id in sc_id_map and v_id in sc_id_map:
                    if sc_id_map[u_id] != sc_id_map[v_id]:
                        comm_pairs.append((sc_id_map[u_id], sc_id_map[v_id]))
            comm_pairs = list(set(comm_pairs))
        return comm_pairs

    def min_cost_objective(self, y_vars, ecus, x_vars=None, z_vars=None, 
                          sensor_ecu_costs=None, actuator_ecu_costs=None, ecu_ecu_costs=None, 
                          include_cable_cost=False):
        hw_cost = gp.quicksum(y_vars[j] * ecus[j].cost for j in range(len(ecus)))
        
        if not include_cable_cost:
            return hw_cost
        
        # Cable Costs (only if x_vars/z_vars and costs are provided)
        cable_cost = 0.0
        if x_vars and sensor_ecu_costs:
             cable_cost += gp.quicksum(x_vars[i, j] * sensor_ecu_costs.get((i, j), 0) for (i, j) in x_vars)
        
        if x_vars and actuator_ecu_costs:
             cable_cost += gp.quicksum(x_vars[i, j] * actuator_ecu_costs.get((i, j), 0) for (i, j) in x_vars)
             
        if z_vars and ecu_ecu_costs:
             cable_cost += gp.quicksum(z_vars[j1, j2] * ecu_ecu_costs.get((j1, j2), 0) for (j1, j2) in z_vars)
             
        return hw_cost + cable_cost
            
    def min_cable_objective(self, x_vars, z_vars,sensor_ecu_dists, actuator_ecu_dists, ecu_ecu_dists):
        # 1. Sensor Cables
        sensor_part = gp.quicksum(x_vars[i, j] * sensor_ecu_dists.get((i, j), 0) for (i, j) in x_vars)
        # 2. Actuator Cables
        actuator_part = gp.quicksum(x_vars[i, j] * actuator_ecu_dists.get((i, j), 0) for (i, j) in x_vars)
        # 3. Backbone Cables
        bb_part = gp.quicksum(z_vars[j1, j2] * ecu_ecu_dists.get((j1, j2), 0) for (j1, j2) in z_vars)
        
        return sensor_part + actuator_part + bb_part

    def _get_distance(self, loc1, loc2):
        """Calculates Euclidean distance between two points."""
        if not loc1 or not loc2:
            return 0.0
        _, dist_euclidean = loc1.dist(loc2)
        return dist_euclidean

    def _precompute_all_distances(self, scs, ecus, sensors, actuators, cable_types):
        """
        Calculates ALL potential cable lengths, costs, AND latencies ONCE before optimization.
        """
        sensor_ecu_dists = {}
        actuator_ecu_dists = {}
        ecu_ecu_dists = {}
        
        sensor_ecu_costs = {}
        actuator_ecu_costs = {}
        ecu_ecu_costs = {}
        
        sensor_ecu_latencies = {}
        actuator_ecu_latencies = {}
        ecu_ecu_latencies = {}
        
        # Map cable type name -> cost and latency per meter
        cost_map = {name: ct.cost_per_meter for name, ct in cable_types.items()}
        latency_map = {name: ct.latency_per_meter for name, ct in cable_types.items()}
        
        # Build lookup dicts for efficient access
        sensor_lookup = {s.id: s for s in sensors}
        actuator_lookup = {a.id: a for a in actuators}
        
        n_sc = len(scs)
        n_ecu = len(ecus)
        
        # A. SC -> ECU Cable Distances (Sensors + Actuators)
        for i in range(n_sc):
            for j in range(n_ecu):
                if not self._is_compatible(scs[i], ecus[j]):
                    continue
                
                # 1. Distance/Cost to Sensors
                dist_s = 0.0
                cost_s = 0.0
                lat_s = 0.0
                for s_id in scs[i].sensors:
                    sensor = sensor_lookup.get(s_id)
                    if sensor:
                        d = self._get_distance(sensor.location, ecus[j].location)
                        dist_s += d
                        cost_s += d * cost_map.get(sensor.interface, 0.0)
                        lat_s += d * latency_map.get(sensor.interface, 0.0)
                sensor_ecu_dists[i, j] = dist_s
                sensor_ecu_costs[i, j] = cost_s
                sensor_ecu_latencies[i, j] = lat_s
                
                # 2. Distance/Cost to Actuators
                dist_a = 0.0
                cost_a = 0.0
                lat_a = 0.0
                for a_id in scs[i].actuators:
                    actuator = actuator_lookup.get(a_id)
                    if actuator:
                        d = self._get_distance(actuator.location, ecus[j].location)
                        dist_a += d
                        cost_a += d * cost_map.get(actuator.interface, 0.0)
                        lat_a += d * latency_map.get(actuator.interface, 0.0)
                actuator_ecu_dists[i, j] = dist_a
                actuator_ecu_costs[i, j] = cost_a
                actuator_ecu_latencies[i, j] = lat_a

        # B. ECU <-> ECU Distances (Backbone)
        for j1 in range(n_ecu):
            for j2 in range(j1 + 1, n_ecu):
                dist = self._get_distance(ecus[j1].location, ecus[j2].location)
                ecu_ecu_dists[j1, j2] = dist
                
                common = set(ecus[j1].interface_offered).intersection(set(ecus[j2].interface_offered))
                
                if not common:
                    ecu_ecu_costs[j1, j2] = dist * 1e6
                    ecu_ecu_latencies[j1, j2] = dist * 1e6
                else:
                    best_iface = max(common, key=lambda x: cost_map.get(x, 0))
                    ecu_ecu_costs[j1, j2] = dist * cost_map.get(best_iface, 0)
                    ecu_ecu_latencies[j1, j2] = dist * latency_map.get(best_iface, 0)
                
        return (sensor_ecu_dists, actuator_ecu_dists, ecu_ecu_dists, 
                sensor_ecu_costs, actuator_ecu_costs, ecu_ecu_costs, 
                sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies)

    def _is_compatible(self, sc, ecu):
        """Checks if SC can run on this ECU (HW + Interface + ASIL)."""
        hw_ok = set(sc.hw_required).issubset(set(ecu.hw_offered))
        interface_ok = set(sc.interface_required).issubset(set(ecu.interface_offered)) if sc.interface_required else True
        asil_ok = ecu.asil_level >= sc.asil_req
        return hw_ok and interface_ok and asil_ok
    
    def _create_base_model(self, name, scs, ecus, sensors, actuators, 
                          comm_constraints_data=None, 
                          sensor_ecu_latencies=None, actuator_ecu_latencies=None, ecu_ecu_latencies=None, 
                          comm_matrix=None, enable_latency=False):
        """
        Creates variables and core constraints.
        Does NOT set the objective function.
        comm_constraints_data: List of (sc_idx_1, sc_idx_2) tuples representing required flows.
        sensor_ecu_latencies: Dict of latencies for Sensor-ECU connections.
        actuator_ecu_latencies: Dict of latencies for ECU-Actuator connections.
        ecu_ecu_latencies: Dict of latencies for ECU-ECU links.
        comm_matrix: Original communication matrix with volume and max_latency.
        enable_latency: Whether to add latency constraints.
        """
        model = gp.Model(name)
        model.setParam('OutputFlag', 0)
        
        n_sc = len(scs)
        n_ecu = len(ecus)
        
        # --- Create Variables ---
        # x[i,j] = 1 if SC_i is assigned to ECU_j
        x = {}
        for i in range(n_sc):
            for j in range(n_ecu):
                if self._is_compatible(scs[i], ecus[j]):
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        
        # y[j] = 1 if ECU_j is used (active)
        y = {j: model.addVar(vtype=GRB.BINARY, name=f"y_{j}") for j in range(n_ecu)}
        
        # z[j1,j2] = 1 if backbone exists between ECU_j1 and ECU_j2
        z = {}
        for j1 in range(n_ecu):
            for j2 in range(j1 + 1, n_ecu):
                z[j1, j2] = model.addVar(vtype=GRB.BINARY, name=f"z_{j1}_{j2}")
        
        # --- Constraints ---
        # 1. Assignment: Each SC must be assigned to exactly one ECU
        for i in range(n_sc):
            compat_ecus = [j for j in range(n_ecu) if (i, j) in x]
            if compat_ecus:
                model.addConstr(gp.quicksum(x[i, j] for j in compat_ecus) == 1, name=f"assign_{i}")
        
        # 2. Activation: If x[i,j]=1, then y[j] must be 1
        for j in range(n_ecu):
            for i in range(n_sc):
                if (i, j) in x:
                    # x <= y  -->  If assigned, ECU is active.
                    model.addConstr(x[i, j] <= y[j], name=f"activate_{i}_{j}")
        
        # 3. Capacity: Check CPU, RAM, ROM, Containers limits
        for j in range(n_ecu):
            compat_scs = [i for i in range(n_sc) if (i, j) in x]
            if compat_scs:
                model.addConstr(gp.quicksum(x[i, j] * scs[i].cpu_req for i in compat_scs) <= ecus[j].cpu_cap)
                model.addConstr(gp.quicksum(x[i, j] * scs[i].ram_req for i in compat_scs) <= ecus[j].ram_cap)
                model.addConstr(gp.quicksum(x[i, j] * scs[i].rom_req for i in compat_scs) <= ecus[j].rom_cap)
                model.addConstr(gp.quicksum(x[i, j] for i in compat_scs) <= ecus[j].max_containers)
        
        # 4. Backbone Logic
        if comm_constraints_data:
            # Connect everything at first
            # Ensure z matches activity 
            for j1 in range(n_ecu):
                for j2 in range(j1 + 1, n_ecu):
                    model.addConstr(z[j1, j2] <= y[j1])
                    model.addConstr(z[j1, j2] <= y[j2])
            
            # Now add traffic constraints (FORCE connection if traffic exists)
            # For every communicating pair (u, v), if u on j1 and v on j2, enforce z[j1,j2]=1
            # Constraint: z[j1, j2] >= x[u, j1] + x[v, j2] - 1
            # Pre-calculate possible ECUs for each SC to avoid useless loops
            sc_compat_ecus = {}
            for i in range(n_sc):
                sc_compat_ecus[i] = [j for j in range(n_ecu) if (i, j) in x]

            for (u, v) in comm_constraints_data:
                possible_j_u = sc_compat_ecus.get(u, [])
                possible_j_v = sc_compat_ecus.get(v, [])
                
                for j1 in possible_j_u:
                    for j2 in possible_j_v:
                        if j1 == j2: continue # Same ECU, no cable needed
                        
                        # Normalize order for z index
                        u_ecu, v_ecu = (j1, j2) if j1 < j2 else (j2, j1)
                        
                        model.addConstr(z[u_ecu, v_ecu] >= x[u, j1] + x[v, j2] - 1)

        else:
            # Fallback: All active ECUs connected (Mesh/Bus assumption)
            for j1 in range(n_ecu):
                for j2 in range(j1 + 1, n_ecu):
                    model.addConstr(z[j1, j2] <= y[j1])
                    model.addConstr(z[j1, j2] <= y[j2])
                    model.addConstr(z[j1, j2] >= y[j1] + y[j2] - 1)
        
        # 4.5. Redundancy Constraints
        # Redundant SC pairs must be placed on different ECUs for fault tolerance
        sc_id_to_idx = {sc.id: i for i, sc in enumerate(scs)}
        for i in range(n_sc):
            if scs[i].redundant_with:
                partner_id = scs[i].redundant_with
                if partner_id in sc_id_to_idx:
                    partner_idx = sc_id_to_idx[partner_id]
                    # Prevent both SCs from being on the same ECU
                    for j in range(n_ecu):
                        if (i, j) in x and (partner_idx, j) in x:
                            model.addConstr(x[i, j] + x[partner_idx, j] <= 1, 
                                          name=f"redundancy_{i}_{partner_idx}_{j}")
        
        # 5. Latency Constraints
        if enable_latency:
            # Build lookup dicts for efficient access
            sensor_lookup = {s.id: s for s in sensors}
            actuator_lookup = {a.id: a for a in actuators}
            
            # A. Sensor Latency Constraints
            if sensor_ecu_latencies:
                for i in range(n_sc):
                    for s_id in scs[i].sensors:
                        sensor = sensor_lookup.get(s_id)
                        if sensor and sensor.max_latency:
                            for j in range(n_ecu):
                                if (i, j) in x:
                                    latency = sensor_ecu_latencies.get((i, j), 0)
                                    if latency > sensor.max_latency:
                                        model.addConstr(x[i, j] == 0, name=f"sensor_lat_{i}_{j}")
            
            # B. Actuator Latency Constraints
            if actuator_ecu_latencies:
                for i in range(n_sc):
                    for a_id in scs[i].actuators:
                        actuator = actuator_lookup.get(a_id)
                        if actuator and actuator.max_latency:
                            for j in range(n_ecu):
                                if (i, j) in x:
                                    latency = actuator_ecu_latencies.get((i, j), 0)
                                    if latency > actuator.max_latency:
                                        model.addConstr(x[i, j] == 0, name=f"actuator_lat_{i}_{j}")
            
            # C. SC-to-SC Communication Latency Constraints
            if comm_matrix and ecu_ecu_latencies and comm_constraints_data:
                sc_id_map = {sc.id: i for i, sc in enumerate(scs)}
                added_count = 0
                
                for link in comm_matrix:
                    src_id = link.get('src')
                    dst_id = link.get('dst')
                    max_lat = link.get('max_latency', float('inf'))
                    
                    if src_id not in sc_id_map or dst_id not in sc_id_map:
                        continue
                    
                    u = sc_id_map[src_id]
                    v = sc_id_map[dst_id]
                    
                    # Pre-calculate possible ECUs
                    possible_j_u = [j for j in range(n_ecu) if (u, j) in x]
                    possible_j_v = [j for j in range(n_ecu) if (v, j) in x]
                    
                    if not possible_j_u or not possible_j_v:
                        continue
                    
                    has_feasible = False
                    for j1 in possible_j_u:
                        for j2 in possible_j_v:
                            if j1 != j2:
                                u_ecu, v_ecu = (j1, j2) if j1 < j2 else (j2, j1)
                                latency = ecu_ecu_latencies.get((u_ecu, v_ecu), 0)
                                if latency <= max_lat:
                                    has_feasible = True
                                    break
                        if has_feasible:
                            break
                    
                    if not has_feasible:
                        continue
                    
                    # Add constraints only for feasible combinations
                    for j1 in possible_j_u:
                        for j2 in possible_j_v:
                            if j1 == j2:
                                continue
                            
                            u_ecu, v_ecu = (j1, j2) if j1 < j2 else (j2, j1)
                            latency = ecu_ecu_latencies.get((u_ecu, v_ecu), 0)
                            
                            if latency > max_lat:
                                model.addConstr(
                                    x[u, j1] + x[v, j2] <= 1,
                                    name=f"latency_{u}_{v}_{j1}_{j2}"
                                )
                                added_count += 1
        
        return model, x, y, z

    def optimize(self, scs, ecus, sensors, actuators, cable_types, comm_matrix=None, num_points=5, include_cable_cost=False, enable_latency_constraints=False):
        """
        Main optimization routine. 
        Generates Pareto front for [HW Cost] vs [Cable Length].
        """
        print("\n" + "="*60)
        print("PARETO OPTIMIZATION: HW Cost vs Cable Length")
        print("="*60)

        # --- Convert comm link into gurobi language ---
        comm_pairs = self.preprocess_communications(scs, comm_matrix)

        # --- Step 0: Precompute Distances, Costs, and Latencies ---
        print("[0/3] Precomputing physical distances, costs, and latencies...")
        (sensor_ecu_dists, actuator_ecu_dists, ecu_ecu_dists,
         sensor_ecu_costs, actuator_ecu_costs, ecu_ecu_costs, 
         sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies) = self._precompute_all_distances(scs, ecus, sensors, actuators, cable_types)
        
        # --- Step 1: Find Min Cost ---
        print("\n[1/3] Finding Minimum Cost...")
        m1, x1, y1, z1 = self._create_base_model("MinCost", scs, ecus, sensors, actuators, comm_pairs, 
                                                 sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies, 
                                                 comm_matrix, enable_latency_constraints)
        
        # Set Objective with optional cable cost
        obj1 = self.min_cost_objective(y1, ecus, x1, z1, 
                                       sensor_ecu_costs, actuator_ecu_costs, ecu_ecu_costs, 
                                       include_cable_cost=include_cable_cost)
        m1.setObjective(obj1, GRB.MINIMIZE)
        m1.optimize()
        
        if m1.status != GRB.OPTIMAL:
            print("ERROR: No feasible solution found!")
            return []
            
        min_cost_val = m1.ObjVal
        print(f"Min Cost = ${min_cost_val:.0f}")
        
        print("\n[2/3] Finding Minimum Cable Length...")
        m2, x2, y2, z2 = self._create_base_model("MinCable", scs, ecus, sensors, actuators, comm_pairs, 
                                                 sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies, 
                                                 comm_matrix, enable_latency_constraints)
        m2.setObjective(self.min_cable_objective(x2, z2, sensor_ecu_dists, actuator_ecu_dists, ecu_ecu_dists), GRB.MINIMIZE)
        m2.optimize()
        
        hw_c = sum(ecus[j].cost for j in range(len(ecus)) if y2[j].X > 0.5)
        cable_c = 0.0
        if include_cable_cost:
            for (i, j), var in x2.items():
                if var.X > 0.5:
                    cable_c += sensor_ecu_costs.get((i, j), 0)
                    cable_c += actuator_ecu_costs.get((i, j), 0)
            for (j1, j2), var in z2.items():
                 if var.X > 0.5:
                    cable_c += ecu_ecu_costs.get((j1, j2), 0)
        
        cost_at_min_cable = hw_c + cable_c
        print(f"Cost at Min Cable = ${cost_at_min_cable:.0f}")
        
        # --- Step 3: Generate Pareto Points ---
        print(f"\n[3/3] Generating {num_points} Pareto solutions...")
        print("-" * 50)
        
        # Handle small numerical diffs
        if abs(cost_at_min_cable - min_cost_val) < 1.0:
            cost_levels = [min_cost_val]
            print("      (Min Cost == Cost at Min Cable. No trade-off range exists.)")
        else:
            step = (cost_at_min_cable - min_cost_val) / (num_points - 1)
            cost_levels = [min_cost_val + i * step for i in range(num_points)]
        
        solutions = []
        for idx, limit in enumerate(cost_levels):
            m, x, y, z = self._create_base_model(f"Pareto_{idx}", scs, ecus, sensors, actuators, comm_pairs, 
                                                 sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies, 
                                                 comm_matrix, enable_latency_constraints)
            
            # Constraint: Cost <= limit
            cost_expr = self.min_cost_objective(y, ecus, x, z, 
                                                sensor_ecu_costs, actuator_ecu_costs, ecu_ecu_costs, 
                                                include_cable_cost=include_cable_cost)
            m.addConstr(cost_expr <= limit, name="epsilon_constraint")
            
            # Objective: Minimize Cable Length
            m.setObjective(self.min_cable_objective(x, z, sensor_ecu_dists, actuator_ecu_dists, ecu_ecu_dists), GRB.MINIMIZE)
            m.optimize()
            
            if m.status == GRB.OPTIMAL:
                # Extract solution
                assignment = {}
                ecus_used = set()
                
                # Check Assignments
                for (i, j), var in x.items():
                    if var.X > 0.5:
                        assignment[scs[i].id] = ecus[j].id
                        ecus_used.add(j)
                
                # Calculate metrics
                hw_cost_sol = sum(ecus[j].cost for j in ecus_used)
                
                # Calculate Length
                cable_len_sol = 0.0
                cable_cost_sol = 0.0
                
                for (i, j), var in x.items():
                    if var.X > 0.5:
                        cable_len_sol += sensor_ecu_dists.get((i, j), 0)
                        cable_len_sol += actuator_ecu_dists.get((i, j), 0)
                        cable_cost_sol += sensor_ecu_costs.get((i, j), 0)
                        cable_cost_sol += actuator_ecu_costs.get((i, j), 0)
                        
                for (j1, j2), var in z.items():
                    if var.X > 0.5:
                        cable_len_sol += ecu_ecu_dists.get((j1, j2), 0)
                        cable_cost_sol += ecu_ecu_costs.get((j1, j2), 0)
                
                total_cost_sol = hw_cost_sol + cable_cost_sol

                solutions.append({
                    'assignment': assignment,
                    'hardware_cost': hw_cost_sol,
                    'cable_cost': cable_cost_sol,
                    'total_cost': total_cost_sol,
                    'cable_length': cable_len_sol,
                    'num_ecus_used': len(ecus_used)
                })
                print(f"  #{idx+1}: Cost=${total_cost_sol:.0f} (HW=${hw_cost_sol:.0f}, Cable=${cable_cost_sol:.0f}) | Len={cable_len_sol:.1f}m | ECUs={len(ecus_used)}")
                
        print("-" * 50)
        print(f"✓ Found {len(solutions)} Pareto solutions.")
        return solutions
