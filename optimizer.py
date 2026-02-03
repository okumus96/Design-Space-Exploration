import os
os.environ["GRB_LICENSE_FILE"] = "/home/okumus/gurobi.lic"
import gurobipy as gp
from gurobipy import GRB

## Uncertainties
    ## HW Failures
    ## OTA Updates/Scalability
    ## Performance Variations and Degradations in HW.
    ## Network Uncertainties (sporadic  events, etc.)
    ## SW Failures/wrong tests (bugs, crashes, memory leaks, etc.)

## POSSIBLE OBJECTIVES:
    ## Power Consumption
    ## ECU Resource Balancing
    ## Bus/Network Load Balancing

## Possible Constraints:
    ## Add container and container communication latency later
    ## Consider ASIL decomposition later
    ## Add interface count (5 ETH) later

## Other Features:
    ## There is  different can lines we do not put info and sc thing in same can line. 


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
        dist_manhattan, dist_euclidean = loc1.dist(loc2)
        return dist_manhattan

    def _precompute_all_metrics(self, scs, ecus, sensors, actuators, cable_types):
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

    def inject_warm_start(self, x_vars,greedy_sol):
        if greedy_sol:     # Helper to inject warm start
            # print("      -> Injecting Warm Start into Gurobi model...") # Optional logging
            for i_sc, j_ecu in greedy_sol.items():
                if (i_sc, j_ecu) in x_vars:
                    x_vars[i_sc, j_ecu].Start = 1
    
    def _generate_greedy_solution(self, scs, ecus, sensors, actuators, comm_matrix=None):
        """
        Generates a smart initial solution (Warm Start) considering:
        1. Constraints (Resources, ASIL, Interface)
        2. Physical Cabling (Sensors/Actuators)
        3. Communication Locality (SC-to-SC)
        4. Hardware Cost (Minimizing active ECUs)
        """
        solution = {} # sc_idx -> ecu_idx
        
        # Track remaining capacities of ECUs
        ecu_caps = []
        ecu_active = [False] * len(ecus)
        for e in ecus:
            ecu_caps.append({
                'cpu': e.cpu_cap,
                'ram': e.ram_cap,
                'rom': e.rom_cap,
                'cont': e.max_containers
            })
            
        sensor_lookup = {s.id: s for s in sensors}
        actuator_lookup = {a.id: a for a in actuators}
        sc_id_to_idx = {sc.id: i for i, sc in enumerate(scs)}

        # Precompute communication partners for locality score
        comm_neighbors = {i: [] for i in range(len(scs))}
        if comm_matrix:
            for link in comm_matrix:
                src = link.get('src')
                dst = link.get('dst')
                if src in sc_id_to_idx and dst in sc_id_to_idx:
                    u, v = sc_id_to_idx[src], sc_id_to_idx[dst]
                    comm_neighbors[u].append(v)
                    comm_neighbors[v].append(u)

        # Sort SCs: Combining Resource Heaviness and Connectivity Degree
        # "Hardest to place" first. 
        # Score = (CPU_req normalized) + (Connectivity Count)
        def urgency_score(i):
            res_norm = (scs[i].cpu_req / 500.0) # Approx weight
            conn_count = len(comm_neighbors[i])
            return res_norm + conn_count

        sorted_sc_indices = sorted(range(len(scs)), key=urgency_score, reverse=True)

        for i in sorted_sc_indices:
            sc = scs[i]
            best_ecu = -1
            best_score = float('inf')
            
            # Identify forbidden ECUs (redundancy)
            forbidden_ecus = set()
            if sc.redundant_with and sc.redundant_with in sc_id_to_idx:
                partner_idx = sc_id_to_idx[sc.redundant_with]
                if partner_idx in solution:
                    forbidden_ecus.add(solution[partner_idx])
            
            # Try all ECUs
            for j, ecu in enumerate(ecus):
                if j in forbidden_ecus: continue
                if not self._is_compatible(sc, ecu): continue
                
                # Check capacity
                cap = ecu_caps[j]
                if (cap['cpu'] >= sc.cpu_req and 
                    cap['ram'] >= sc.ram_req and 
                    cap['rom'] >= sc.rom_req and 
                    cap['cont'] >= 1):
                    
                    # --- SCORING (Lower is better) ---
                    
                    # 1. Distances to Peripherals (Sensors/Actuators)
                    dev_dist = 0.0
                    for s_id in sc.sensors:
                        s = sensor_lookup.get(s_id)
                        if s: dev_dist += self._get_distance(s.location, ecu.location)
                    for a_id in sc.actuators:
                        a = actuator_lookup.get(a_id)
                        if a: dev_dist += self._get_distance(a.location, ecu.location)
                    
                    # 2. SC-to-SC Communication Proximity
                    # Calculate distance to partners that are ALREADY placed.
                    comm_dist = 0.0
                    for partner_idx in comm_neighbors[i]:
                        if partner_idx in solution:
                            partner_ecu_idx = solution[partner_idx]
                            # Distance between candidate ECU and partner ECU
                            d = self._get_distance(ecu.location, ecus[partner_ecu_idx].location)
                            comm_dist += d
                            
                    # 3. Hardware Cost / Activation Penalty
                    # Heavily penalize opening a new ECU if not necessary.
                    # This drives the algorithm to pack SCs into fewer ECUs (Bin Packing behavior).
                    hw_penalty = 0.0
                    if not ecu_active[j]:
                        # Use ECU cost as penalty (or a multiplier of it)
                        hw_penalty = ecu.cost * 1.5 
                    
                    # Combined Weighted Score
                    # Weights can be tuned. Currently:
                    # - 1.0 for cable distance
                    # - 1.0 for comm distance
                    # - 1.5 * HW_Cost for using new ECU
                    current_score = dev_dist + comm_dist + hw_penalty
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_ecu = j
            
            if best_ecu != -1:
                solution[i] = best_ecu
                ecu_active[best_ecu] = True
                # Update caps
                ecu_caps[best_ecu]['cpu'] -= sc.cpu_req
                ecu_caps[best_ecu]['ram'] -= sc.ram_req
                ecu_caps[best_ecu]['rom'] -= sc.rom_req
                ecu_caps[best_ecu]['cont'] -= 1
            else:
                # Greedy search failed for at least one SC
                return None
                
        return solution
    
    def extract_solution(self,model, x_vars, y_vars, z_vars, scs, ecus, sensor_ecu_dists, actuator_ecu_dists, ecu_ecu_dists, sensor_ecu_costs, actuator_ecu_costs, ecu_ecu_costs):
            """Helper to extract solution data from solved model"""
            assignment_ = {}
            ecus_used_ = set()
            for (i_sc, j_ecu), var in x_vars.items():
                if var.X > 0.5:
                    assignment_[scs[i_sc].id] = ecus[j_ecu].id
                    ecus_used_.add(j_ecu)
            
            hw_cost_ = sum(ecus[j].cost for j in ecus_used_)
            
            len_ = 0.0
            cable_cost_ = 0.0
            for (i_xx, j_xx), var in x_vars.items():
                if var.X > 0.5:
                    len_ += sensor_ecu_dists.get((i_xx, j_xx), 0)
                    len_ += actuator_ecu_dists.get((i_xx, j_xx), 0)
                    cable_cost_ += sensor_ecu_costs.get((i_xx, j_xx), 0)
                    cable_cost_ += actuator_ecu_costs.get((i_xx, j_xx), 0)
            for (j1, j2), var in z_vars.items():
                if var.X > 0.5:
                    len_ += ecu_ecu_dists.get((j1, j2), 0)
                    cable_cost_ += ecu_ecu_costs.get((j1, j2), 0)
            
            return {
                'assignment': assignment_,
                'hardware_cost': hw_cost_,
                'cable_cost': cable_cost_,
                'total_cost': hw_cost_ + cable_cost_,
                'cable_length': len_,
                'num_ecus_used': len(ecus_used_),
                'status': "OPTIMAL" if model.status == GRB.OPTIMAL else f"GAP {model.MIPGap:.0%}"
            }

    def _create_base_model(self, name, scs, ecus, sensors, actuators, 
                          comm_constraints_data=None, 
                          sensor_ecu_latencies=None, actuator_ecu_latencies=None, ecu_ecu_latencies=None, 
                          comm_matrix=None, enable_latency=False, time_limit=60, mip_gap=0.01, verbose=False):
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
        # Enable output to monitor progress
        model.setParam('OutputFlag', 1 if verbose else 0)
        # MIPFocus=2: Focus on proving optimality (improves BestBound)
        model.setParam('MIPFocus', 2)
        if time_limit is not None:
            model.setParam('TimeLimit', time_limit)
        if mip_gap is not None:
            model.setParam('MIPGap', mip_gap)
        
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

        # 2.5. Location Exclusivity: At most one ECU active per physical location
        location_groups = {}
        for j in range(n_ecu):
            # Rounding to avoid float precision issues, though data generator uses exact values
            loc_key = (round(ecus[j].location.x, 3), round(ecus[j].location.y, 3))
            if loc_key not in location_groups:
                location_groups[loc_key] = []
            location_groups[loc_key].append(j)
        
        for loc, group in location_groups.items():
            if len(group) > 1:
                model.addConstr(gp.quicksum(y[j] for j in group) <= 1, name=f"one_ecu_at_{loc}")
        
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

    def optimize(self, scs, ecus, sensors, actuators, cable_types, comm_matrix=None, num_points=5, include_cable_cost=False, enable_latency_constraints=False, warm_start=False, time_limit=60, mip_gap=None, verbose=False):
        """
        Main optimization routine. 
        Generates Pareto front for [HW Cost] vs [Cable Length] using Bounds-Based Search.
        Strategy:
        1. Find Min Cost Solution -> Get Max Cable Length (L_max)
        2. Find Min Cable Length Solution -> Get Min Cable Length (L_min)
        3. Sweep Cable Length constraint from L_max to L_min to find Pareto points.
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
         sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies) = self._precompute_all_metrics(scs, ecus, sensors, actuators, cable_types)
        
        if warm_start:
            print("      Generating Warm Start solution...")
            greedy_sol = self._generate_greedy_solution(scs, ecus, sensors, actuators, comm_matrix)

        solutions = []

        # --- Step 1: Minimize Cost (Extremity 1) ---
        print("\n[1] Finding Extremity 1: Minimum Cost...")
        m1, x1, y1, z1 = self._create_base_model("MinCost", scs, ecus, sensors, actuators, comm_pairs, 
            sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies, comm_matrix, enable_latency_constraints, time_limit, mip_gap, verbose)
        if warm_start: self.inject_warm_start(x1, greedy_sol)
        
        c1 = self.min_cost_objective(y1, ecus, x1, z1, sensor_ecu_costs, actuator_ecu_costs, ecu_ecu_costs, include_cable_cost)
        #l1 = self.min_cable_objective(x1, z1, sensor_ecu_dists, actuator_ecu_dists, ecu_ecu_dists)
        m1.setObjective(c1, GRB.MINIMIZE)
        m1.optimize()
        
        if m1.SolCount == 0:
            print("Error: No feasible solution found for Min Cost!")
            return []
            
        sol1 = self.extract_solution(m1, x1, y1, z1, scs, ecus, sensor_ecu_dists, actuator_ecu_dists, ecu_ecu_dists, sensor_ecu_costs, actuator_ecu_costs, ecu_ecu_costs)
        L_max = sol1['cable_length']
        print(f"-> Status: {sol1['status']}, Min Cost: ${sol1['total_cost']:.0f}, Length: {L_max:.1f}m")

        # --- Step 2: Minimize Length (Extremity 2) ---
        print("\n[2] Finding Extremity 2: Minimum Cable Length...")
        m2, x2, y2, z2 = self._create_base_model("MinCable", scs, ecus, sensors, actuators, comm_pairs, 
            sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies, comm_matrix, enable_latency_constraints, time_limit, mip_gap, verbose)
        if warm_start: self.inject_warm_start(x2, greedy_sol)
        
        l2 = self.min_cable_objective(x2, z2, sensor_ecu_dists, actuator_ecu_dists, ecu_ecu_dists)
        m2.setObjective(l2, GRB.MINIMIZE)
        m2.optimize()
        
        if m2.SolCount == 0:
            print("Error: No feasible solution found for Min Cable!")
            # Use L_max as fallback if we can't find min cable (should only happen if inconsistent constraints)
            L_min = L_max 
        else:
            sol2 = self.extract_solution(m2, x2, y2, z2, scs, ecus, sensor_ecu_dists, actuator_ecu_dists, ecu_ecu_dists, sensor_ecu_costs, actuator_ecu_costs, ecu_ecu_costs)
            L_min = sol2['cable_length']
            print(f"-> Status: {sol2['status']}, Min Length: {L_min:.1f}m (Cost: ${sol2['total_cost']:.0f})")

        # --- Step 3: Grid Search ---
        print(f"\n[3] Searching Pareto Front from {L_max:.1f}m to {L_min:.1f}m using {num_points} intermediate points...")
        
        if abs(L_max - L_min) < 0.5:
            targets = [L_max]
            print("    -> Range is too small (Min Cost is already Min Length). Returning minimal set.")
        else:
            # We want num_points strictly BETWEEN L_max and L_min
            # Total steps including start (L_max) and end (L_min) would be num_points + 2
            # But user wants specific logic: Generate points between them.
            # Let's generate a full range including endpoints, then slice or just rely on duplicates checking.
            # Strategy: Generate (num_points + 2) targets linearly spaced.
            # This ensures we hit L_max, L_min, and exactly num_points in between.
            
            total_points = num_points + 2
            targets = [L_max - i * (L_max - L_min) / (total_points - 1) for i in range(total_points)]
            print(targets)
            
        print("-" * 110)
        print(f"| {'Iter':<4} | {'Limit (m)':<10} | {'Status':<10} | {'Total Cost':<12} | {'HW Cost':<10} | {'Cable Cost':<10} | {'Length (m)':<10} | {'ECUs':<4} |")
        print("-" * 110)
        
        processed_solutions = set() # Store hashes to avoid duplicates

        for idx, target_len in enumerate(targets):
            
            m, x, y, z = self._create_base_model(f"Grid_{idx}", scs, ecus, sensors, actuators, comm_pairs, 
                sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies, comm_matrix, enable_latency_constraints, time_limit, mip_gap, verbose)
            
            if warm_start: self.inject_warm_start(x, greedy_sol)
            
            # Constraint: Length <= Target
            l_expr = self.min_cable_objective(x, z, sensor_ecu_dists, actuator_ecu_dists, ecu_ecu_dists)
            m.addConstr(l_expr <= target_len, name="length_constraint")
            
            # Objective: Min Cost (+ epsilon length)
            c_expr = self.min_cost_objective(y, ecus, x, z, sensor_ecu_costs, actuator_ecu_costs, ecu_ecu_costs, include_cable_cost)
            m.setObjective(c_expr + 0.001 * l_expr, GRB.MINIMIZE)
            
            m.optimize()
            
            if m.SolCount > 0:
                sol = self.extract_solution(m, x, y, z, scs, ecus, sensor_ecu_dists, actuator_ecu_dists, ecu_ecu_dists, sensor_ecu_costs, actuator_ecu_costs, ecu_ecu_costs)
                
                # Check duplicate by exact cost and length tuple
                sol_key = (round(sol['total_cost'], 2), round(sol['cable_length'], 2))
                if sol_key not in processed_solutions:
                    processed_solutions.add(sol_key)
                    solutions.append(sol)
                    
                    print(f"| {idx+1:<4} | {target_len:<10.1f} | {sol['status']:<10} | ${sol['total_cost']:<11.0f} | ${sol['hardware_cost']:<9.0f} | ${sol['cable_cost']:<9.0f} | {sol['cable_length']:<10.1f} | {sol['num_ecus_used']:<4} |")
                else:
                    print(f"| {idx+1:<4} | {target_len:<10.1f} | {'DUPLICATE':<10} | {'-':<12} | {'-':<10} | {'-':<10} | {'-':<10} | {'-':<4} |")
            else:
                 print(f"| {idx+1:<4} | {target_len:<10.1f} | {'INFEASIBLE':<10} | {'-':<12} | {'-':<10} | {'-':<10} | {'-':<10} | {'-':<4} |")

        print("-" * 110)
        return solutions
