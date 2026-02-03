
import math
from z3 import *

class AssignmentOptimizerZ3:
    """
    SC -> ECU Assignment Optimization using Z3 Solver.
    
    Objectives: Minimize HW Cost and Cable Length (Pareto front)
    """

    # =========================================================================
    # HELPER FUNCTIONS (Copied from optimizer.py)
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

    def _get_distance(self, loc1, loc2):
        """Calculates Euclidean distance between two points."""
        if not loc1 or not loc2:
            return 0.0
        _, dist_euclidean = loc1.dist(loc2)
        return dist_euclidean

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
        
        cost_map = {name: ct.cost_per_meter for name, ct in cable_types.items()}
        latency_map = {name: ct.latency_per_meter for name, ct in cable_types.items()}
        
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

    # =========================================================================
    # Z3 MODEL CREATION & OPTIMIZATION
    # =========================================================================

    def _create_base_model(self, scs, ecus, comm_constraints_data, 
                           sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies, 
                           comm_matrix, enable_latency, sensors, actuators, time_limit_ms=60000):
        
        opt = Optimize()
        
        # Set timeout (in milliseconds)
        opt.set("timeout", time_limit_ms)
        
        n_sc = len(scs)
        n_ecu = len(ecus)

        # Variables
        # x[i, j] is 1 if sc[i] is on ecu[j], else 0
        x = {} 
        for i in range(n_sc):
            for j in range(n_ecu):
                if self._is_compatible(scs[i], ecus[j]):
                    x[i, j] = Int(f"x_{i}_{j}")
                    opt.add(x[i, j] >= 0)
                    opt.add(x[i, j] <= 1)

        # y[j] is 1 if ecu[j] is active, else 0
        y = {}
        for j in range(n_ecu):
            y[j] = Int(f"y_{j}")
            opt.add(y[j] >= 0)
            opt.add(y[j] <= 1)

        # z[j1, j2] is 1 if connection needed between ecu[j1] and ecu[j2]
        z = {}
        for j1 in range(n_ecu):
            for j2 in range(j1 + 1, n_ecu):
                z[j1, j2] = Int(f"z_{j1}_{j2}")
                opt.add(z[j1, j2] >= 0)
                opt.add(z[j1, j2] <= 1)

        # Constraints
        
        # 1. Assignment: Each SC assigned to exactly one ECU
        for i in range(n_sc):
            possible_ecus = [x[i, j] for j in range(n_ecu) if (i, j) in x]
            if possible_ecus:
                opt.add(Sum(possible_ecus) == 1)
            else:
                # Impossible assignment (should effectively make model UNSAT if critical)
                pass 

        # 2. Activation: x[i, j] == 1 => y[j] == 1
        for j in range(n_ecu):
             for i in range(n_sc):
                 if (i, j) in x:
                     opt.add(x[i, j] <= y[j])

        # 3. Capacity
        for j in range(n_ecu):
            possible_scs_vars = []
            cpu_reqs = []
            ram_reqs = []
            rom_reqs = []
            # Gather vars for this ECU
            for i in range(n_sc):
                if (i, j) in x:
                    possible_scs_vars.append(x[i, j])
                    cpu_reqs.append(x[i, j] * scs[i].cpu_req)
                    ram_reqs.append(x[i, j] * scs[i].ram_req)
                    rom_reqs.append(x[i, j] * scs[i].rom_req)
            
            if possible_scs_vars:
                opt.add(Sum(cpu_reqs) <= ecus[j].cpu_cap)
                opt.add(Sum(ram_reqs) <= ecus[j].ram_cap)
                opt.add(Sum(rom_reqs) <= ecus[j].rom_cap)
                opt.add(Sum(possible_scs_vars) <= ecus[j].max_containers)

        # 4. Backbone Logic
        # z[j1, j2] should be 1 if traffic flows between j1 and j2
        # Constraint: z[j1, j2] >= x[u, j1] + x[v, j2] - 1
        if comm_constraints_data:
            # First force Z to respect Y (optional but cleaner)
            for key in z:
                j1, j2 = key
                opt.add(z[j1, j2] <= y[j1])
                opt.add(z[j1, j2] <= y[j2])

            for (u, v) in comm_constraints_data:
                # Find compatible ECUs for u and v
                compat_u = [j for j in range(n_ecu) if (u, j) in x]
                compat_v = [j for j in range(n_ecu) if (v, j) in x]
                
                for j1 in compat_u:
                    for j2 in compat_v:
                        if j1 == j2: continue
                        
                        u_ecu, v_ecu = (j1, j2) if j1 < j2 else (j2, j1)
                        if (u_ecu, v_ecu) in z:
                            # If u is on j1 AND v is on j2, then connection needed
                            opt.add(z[u_ecu, v_ecu] >= x[u, j1] + x[v, j2] - 1)
        else:
            # Full mesh for active ECUs
            for j1 in range(n_ecu):
                for j2 in range(j1 + 1, n_ecu):
                    opt.add(z[j1, j2] >= y[j1] + y[j2] - 1)

        # 4.5 Redundancy
        sc_id_to_idx = {sc.id: i for i, sc in enumerate(scs)}
        for i in range(n_sc):
            if scs[i].redundant_with:
                partner_id = scs[i].redundant_with
                if partner_id in sc_id_to_idx:
                    partner_idx = sc_id_to_idx[partner_id]
                    for j in range(n_ecu):
                        if (i, j) in x and (partner_idx, j) in x:
                            opt.add(x[i, j] + x[partner_idx, j] <= 1)

        # 5. Latency Constraints
        if enable_latency:
            sensor_lookup = {s.id: s for s in sensors}
            actuator_lookup = {a.id: a for a in actuators}

            # A. Sensor Latency
            if sensor_ecu_latencies:
                for i in range(n_sc):
                    for s_id in scs[i].sensors:
                        sensor = sensor_lookup.get(s_id)
                        if sensor and sensor.max_latency:
                            for j in range(n_ecu):
                                if (i, j) in x:
                                    lat = sensor_ecu_latencies.get((i, j), 0)
                                    if lat > sensor.max_latency:
                                        # Forbidden assignment
                                        opt.add(x[i, j] == 0)

            # B. Actuator Latency
            if actuator_ecu_latencies:
                for i in range(n_sc):
                    for a_id in scs[i].actuators:
                        actuator = actuator_lookup.get(a_id)
                        if actuator and actuator.max_latency:
                            for j in range(n_ecu):
                                if (i, j) in x:
                                    lat = actuator_ecu_latencies.get((i, j), 0)
                                    if lat > actuator.max_latency:
                                        opt.add(x[i, j] == 0)

            # C. SC-SC Latency (Simplistic check like Gurobi)
            if comm_matrix and ecu_ecu_latencies and comm_constraints_data:
                sc_id_map = {sc.id: i for i, sc in enumerate(scs)}
                for link in comm_matrix:
                    src_id = link.get('src')
                    dst_id = link.get('dst')
                    max_lat = link.get('max_latency', float('inf'))
                    
                    if src_id in sc_id_map and dst_id in sc_id_map:
                        u = sc_id_map[src_id]
                        v = sc_id_map[dst_id]
                        
                        compat_u = [j for j in range(n_ecu) if (u, j) in x]
                        compat_v = [j for j in range(n_ecu) if (v, j) in x]

                        for j1 in compat_u:
                            for j2 in compat_v:
                                if j1 == j2: continue
                                u_ecu, v_ecu = (j1, j2) if j1 < j2 else (j2, j1)
                                if ecu_ecu_latencies.get((u_ecu, v_ecu), 0) > max_lat:
                                    # Forbidden pair
                                    opt.add(x[u, j1] + x[v, j2] <= 1)

        return opt, x, y, z

    def optimize(self, scs, ecus, sensors, actuators, cable_types, comm_matrix=None, num_points=5, include_cable_cost=False, enable_latency_constraints=False, warm_start=False, time_limit=60, mip_gap=None):
        """
        Z3-based Pareto Optimization.
        """
        print("\n" + "="*60)
        print("Z3 PARETO OPTIMIZATION: HW Cost vs Cable Length")
        print("="*60)
        if mip_gap is not None:
             print(f"      (Warning: mip_gap={mip_gap} ignored for Z3 solver)")

        # Convert time limit to ms
        time_limit_ms = int(time_limit * 1000)

        comm_pairs = self.preprocess_communications(scs, comm_matrix)

        print("[0/3] Precomputing metrics (Z3)...")
        (sensor_ecu_dists, actuator_ecu_dists, ecu_ecu_dists,
         sensor_ecu_costs, actuator_ecu_costs, ecu_ecu_costs, 
         sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies) = self._precompute_all_metrics(scs, ecus, sensors, actuators, cable_types)

        # --- Helper for Objectives ---
        def get_hw_cost_expr(y_vars, ecus_data):
            # Sum(y[j] * cost)
            # Z3 works better with sums if types match. ecus[j].cost is float.
            # Convert to Real or Int? Optimization with Real is fine.
            return Sum([y_vars[j] * ecus_data[j].cost for j in range(len(ecus_data))])

        def get_cable_cost_expr(x_vars, z_vars):
            terms = []
            if x_vars:
                # Sensor/Actuator Cost
                for (i, j), var in x_vars.items():
                    c_s = sensor_ecu_costs.get((i, j), 0)
                    c_a = actuator_ecu_costs.get((i, j), 0)
                    if c_s + c_a > 0:
                        terms.append(var * (c_s + c_a))
            if z_vars:
                # Backbone Cost
                for (j1, j2), var in z_vars.items():
                    c = ecu_ecu_costs.get((j1, j2), 0)
                    if c > 0:
                        terms.append(var * c)
            return Sum(terms)

        def get_cable_len_expr(x_vars, z_vars):
            terms = []
            # Sensor/Actuator Len
            for (i, j), var in x_vars.items():
                d_s = sensor_ecu_dists.get((i, j), 0)
                d_a = actuator_ecu_dists.get((i, j), 0)
                if d_s + d_a > 0:
                    terms.append(var * (d_s + d_a))
            # Backbone Len
            for (j1, j2), var in z_vars.items():
                d = ecu_ecu_dists.get((j1, j2), 0)
                if d > 0:
                    terms.append(var * d)
            return Sum(terms)

        # --- Step 1: Minimize Total Cost (HW [+ Cable]) ---
        print("\n[1/3] Finding Minimum Cost...")
        opt1, x1, y1, z1 = self._create_base_model(scs, ecus, comm_pairs, 
                                                   sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies, 
                                                   comm_matrix, enable_latency_constraints, sensors, actuators, time_limit_ms)
        
        cost_obj_expr = get_hw_cost_expr(y1, ecus)
        if include_cable_cost:
            cost_obj_expr = cost_obj_expr + get_cable_cost_expr(x1, z1)
            
        opt1.minimize(cost_obj_expr)
        
        res1 = opt1.check()
        min_cost_val = 0.0
        
        if res1 == sat:
            m1 = opt1.model()
            # Evaluate objective manually from model if needed, but Z3 optimizes it.
            # Z3 evaluation might return fractional/rational. Convert to float.
            
            # Helper to eval sum
            def eval_expr(model, expr):
                val = model.eval(expr)
                if isinstance(val, RatNumRef):
                    return float(val.numerator_as_long()) / float(val.denominator_as_long())
                if isinstance(val, IntNumRef):
                    return float(val.as_long())
                # Fallback for simple Algebraics or Reals
                try: 
                    return float(val.as_decimal(10).replace('?','')) 
                except: 
                    return 0.0

            min_cost_val = eval_expr(m1, cost_obj_expr)
            print(f"Min Cost (Z3) = ${min_cost_val:.2f}")
        else:
            print(f"ERROR: No feasible solution found (Result: {res1})")
            return []

        # --- Step 2: Minimize Cable Length (to find upper bound of cost) ---
        print("\n[2/3] Finding Minimum Cable Length...")
        opt2, x2, y2, z2 = self._create_base_model(scs, ecus, comm_pairs, 
                                                   sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies, 
                                                   comm_matrix, enable_latency_constraints, sensors, actuators, time_limit_ms)
        
        cable_len_expr = get_cable_len_expr(x2, z2)
        opt2.minimize(cable_len_expr)
        
        res2 = opt2.check()
        max_cost_val = min_cost_val # Fallback
        
        if res2 == sat:
            m2 = opt2.model()
            # Calculate cost at this point
            hw_c = 0.0
            for j in range(len(ecus)):
                if m2.eval(y2[j]).as_long() > 0:
                    hw_c += ecus[j].cost
            
            cab_c = 0.0
            if include_cable_cost:
                # Manual sum evaluation for cost
                # Re-construct expressions or just iterate x2, z2
                # Iterating variables is safer than expressions for evaluation
                for (i, j), var in x2.items():
                    if m2.eval(var).as_long() > 0:
                        cab_c += sensor_ecu_costs.get((i, j), 0)
                        cab_c += actuator_ecu_costs.get((i, j), 0)
                for (j1, j2), var in z2.items():
                    if m2.eval(var).as_long() > 0:
                         cab_c += ecu_ecu_costs.get((j1, j2), 0)
            
            max_cost_val = hw_c + cab_c
            print(f"Cost at Min Cable (Z3) = ${max_cost_val:.2f}")

        # --- Step 3: Pareto Points ---
        print(f"\n[3/3] Generating Pareto solutions...")
        
        cost_levels = []
        if abs(max_cost_val - min_cost_val) < 1.0:
            cost_levels = [min_cost_val]
        else:
            step = (max_cost_val - min_cost_val) / (num_points - 1)
            cost_levels = [min_cost_val + k * step for k in range(num_points)]

        solutions = []
        for idx, limit in enumerate(cost_levels):
            print(f"  Solving for Cost <= {limit:.2f}...")
            opt, x, y, z = self._create_base_model(scs, ecus, comm_pairs, 
                                                   sensor_ecu_latencies, actuator_ecu_latencies, ecu_ecu_latencies, 
                                                   comm_matrix, enable_latency_constraints, sensors, actuators, time_limit_ms)
            
            # Objective: Min Cable Len
            # Constraint: Total Cost <= limit
            
            c_expr = get_hw_cost_expr(y, ecus)
            if include_cable_cost:
                c_expr = c_expr + get_cable_cost_expr(x, z)
            
            opt.add(c_expr <= limit)
            opt.minimize(get_cable_len_expr(x, z))
            
            if opt.check() == sat:
                m = opt.model()
                
                # Extract Solution
                assignment = {}
                ecus_used = set()
                
                for (i, j), var in x.items():
                    if m.eval(var).as_long() > 0:
                        assignment[scs[i].id] = ecus[j].id
                        ecus_used.add(j)
                
                # Recalculate exact values
                hw_cost_sol = sum(ecus[j].cost for j in ecus_used)
                cable_cost_sol = 0.0
                cable_len_sol = 0.0
                
                for (i, j), var in x.items():
                    if m.eval(var).as_long() > 0:
                        cable_cost_sol += sensor_ecu_costs.get((i, j), 0)
                        cable_cost_sol += actuator_ecu_costs.get((i, j), 0)
                        cable_len_sol += sensor_ecu_dists.get((i, j), 0)
                        cable_len_sol += actuator_ecu_dists.get((i, j), 0)
                        
                for (j1, j2), var in z.items():
                    if m.eval(var).as_long() > 0:
                        cable_cost_sol += ecu_ecu_costs.get((j1, j2), 0)
                        cable_len_sol += ecu_ecu_dists.get((j1, j2), 0)
                
                total_cost_sol = hw_cost_sol + cable_cost_sol
                
                solutions.append({
                    'assignment': assignment,
                    'hardware_cost': hw_cost_sol,
                    'cable_cost': cable_cost_sol,
                    'total_cost': total_cost_sol,
                    'cable_length': cable_len_sol,
                    'num_ecus_used': len(ecus_used)
                })
                print(f"    -> Found: Cost=${total_cost_sol:.0f} | Len={cable_len_sol:.1f}m")
            else:
                print("    -> No solution found for this point.")
                
        return solutions
