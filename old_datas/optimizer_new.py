import os
import numpy as np
from pyexpat import model
os.environ["GRB_LICENSE_FILE"] = "/home/okumus/gurobi.lic"
import gurobipy as gp
from gurobipy import GRB

# _is_compatible and Location Exclusivity make things worse significantly.
# _is_compatible enable us to skip creating variables for incompatible pairs, which reduces variable count and constraints significantly.
# this lead to quick drop in gap but then the solver struggles to find better solutions or prove optimality.
# Location Exclusivity creates many constraints, but helps reduce feasible space. Give guidance to choose one ecu one location. Reduce symmetry, branches.

# Sensor/Actuator Latency Constraints makes things simpler and faster by eliminating infeasible assignments upfront probably. 
# Instead of creating variables and then relying on solver to find out that certain assignments lead to high latency, we proactively add constraints to prevent those assignments.
# This reduces the search space significantly, allowing the solver to focus on feasible regions of the solution space. In practice, this can lead to much faster convergence and better solutions.


# Choosing our interface b/w ecus unintiutive.
# Interfaces right now have infinite capacity.
# We need change direct link b/w the ECUs to some sort of network topology.
# we need to get rid of the ECU types, model them as a legos of HWs and partitions.
# We did not include uncertainty in overall system.
# We do not know numbers we gave is correct (i.e., cost, latency,  etc.)
# WE do not know our objective functions is correct. (semanatically, we can change distance with complexity model)



class AssignmentOptimizerNew:
    def __init__(self):
        print("Initialized AssignmentOptimizerNew with quadratic ECU-ECU costs and linear sensor/actuator costs")
        pass
    
    def _extract_solution_from_model(self, x, y, scs, ecus, sensors, actuators, cable_types, objective_name):
        """Extract solution from current model state"""
        assignment_map = {}
        for (i, j), var in x.items():
            if var.X > 0.5:
                assignment_map[scs[i].id] = ecus[j].id
        
        used_ecus = sum(1 for j in range(len(ecus)) if y[j].X > 0.5)
        
        # Calculate metrics
        cable_length = 0.0
        cable_cost_val = 0.0
        hw_cost_val = 0.0
        
        cost_map = {name: ct.cost_per_meter for name, ct in cable_types.items()}
        sensor_lookup = {s.id: s for s in sensors}
        actuator_lookup = {a.id: a for a in actuators}
        
        for (i, j), var in x.items():
            if var.X > 0.5:
                sc = scs[i]
                ecu = ecus[j]
                
                for s_id in sc.sensors:
                    sensor = sensor_lookup.get(s_id)
                    if sensor and sensor.location and ecu.location:
                        dist = self._get_distance(sensor.location, ecu.location)
                        cable_length += dist
                        cable_cost_val += dist * cost_map.get(sensor.interface, 0.0)
                
                for a_id in sc.actuators:
                    actuator = actuator_lookup.get(a_id)
                    if actuator and actuator.location and ecu.location:
                        dist = self._get_distance(actuator.location, ecu.location)
                        cable_length += dist
                        cable_cost_val += dist * cost_map.get(actuator.interface, 0.0)

        # ECU-ECU backbone contributions (same logic as objective)
        for j1 in range(len(ecus)):
            if y[j1].X <= 0.5:
                continue
            for j2 in range(j1 + 1, len(ecus)):
                if y[j2].X <= 0.5:
                    continue

                dist = self._get_distance(ecus[j1].location, ecus[j2].location)
                if dist <= 0:
                    continue

                common = set(ecus[j1].interface_offered).intersection(set(ecus[j2].interface_offered))
                if not common:
                    cable_cost_val += dist * 1e6
                    cable_length += dist
                else:
                    best_iface = max(common, key=lambda x: cost_map.get(x, 0))
                    cable_cost_val += dist * cost_map.get(best_iface, 0)
                    cable_length += dist
        
        hw_cost_val = sum(ecus[j].cost for j in range(len(ecus)) if y[j].X > 0.5)
        
        solution = {
            'assignment': assignment_map,
            'hardware_cost': hw_cost_val,
            'cable_cost': cable_cost_val,
            'total_cost': hw_cost_val + cable_cost_val,
            'cable_length': cable_length,
            'num_ecus_used': used_ecus,
            'objective': objective_name,
            'status': "OPTIMAL"
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

    def _is_compatible(self, sc, ecu):
        """Checks if SC can run on this ECU (HW + Interface + ASIL)."""
        hw_ok = set(sc.hw_required).issubset(set(ecu.hw_offered))
        interface_ok = set(sc.interface_required).issubset(set(ecu.interface_offered)) if sc.interface_required else True
        asil_ok = ecu.asil_level >= sc.asil_req
        return hw_ok and interface_ok and asil_ok
        #return True

    def create_base_model(self, scs, ecus, sensors, actuators, comm_matrix, cable_types, verbose=False, time_limit=None, mip_gap=None):
        # Create Model
        model = gp.Model("ECU_Assignment")
        # Enable output to monitor progress
        model.setParam('OutputFlag', 1 if verbose else 0)
        # MIPFocus=2: Focus on proving optimality (improves BestBound)
        model.setParam('MIPFocus', 2)
        if time_limit is not None: model.setParam('TimeLimit', time_limit)
        if mip_gap is not None: model.setParam('MIPGap', mip_gap)

        # ==========================================
        # 1. Variables
        # ==========================================
        n_sc = len(scs)
        n_ecu = len(ecus)

        # x[i, j] = 1 if SC i is assigned to ECU j (only create if compatible)
        x = {}
        for i, sc in enumerate(scs):
            #print(f"Creating variables for SC {sc.id} ({i}/{len(scs)})...")
            for j, ecu in enumerate(ecus):
                if self._is_compatible(sc, ecu):
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{sc.id}_{ecu.id}")

        # y[j] = 1 if ECU j is used
        y = {}
        for j, ecu in enumerate(ecus):
            #print(f"Creating variable for ECU {ecu.id} ({j}/{len(ecus)})...")
            y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{ecu.id}")

        # p[j, a] = 1 if ECU j has partition for ASIL a
        all_asils = sorted(list(set(sc.asil_req for sc in scs)))
        #print(all_asils)
        p = {}
        for j, ecu in enumerate(ecus):
            for a in all_asils:
                if a <= ecu.asil_level:
                    p[j, a] = model.addVar(vtype=GRB.BINARY, name=f"p_{ecu.id}_asil_{a}")
                    #print(p[j, a])

        # Pre-compute compatibility mapping
        compat_ecus_per_sc = {}  # For each SC, which ECUs are compatible
        compat_scs_per_ecu = {}  # For each ECU, which SCs are compatible
        
        for i in range(n_sc):
            compat_ecus_per_sc[i] = [j for j in range(n_ecu) if (i, j) in x]
        for j in range(n_ecu):
            compat_scs_per_ecu[j] = [i for i in range(n_sc) if (i, j) in x]


        # ==========================================
        # 2. Structural Constraints
        # ==========================================

        # Constraint 1: Every SC must be assigned to exactly one ECU
        for i, sc in enumerate(scs):
            if compat_ecus_per_sc[i]:
                model.addConstr(
                    gp.quicksum(x[i, j] for j in compat_ecus_per_sc[i]) == 1,
                    name=f"assign_{sc.id}"
                )

        # Constraint 2: (If SC is on ECU j, ECU j is used)
        for j in range(n_ecu):
            for i in compat_scs_per_ecu[j]:
                model.addConstr(x[i, j] <= y[j], name=f"assign_scs{scs[i].id}_ecu{ecus[j].id}")

        # Constraint 3. Capacity: Check CPU, RAM, ROM
        for j, ecu in enumerate(ecus):
            if compat_scs_per_ecu[j]:

                # CPU Capacity:
                # Total Used = (Base System Overhead) + (Reserved Partitions)
                # Base Overhead applies if ECU is used (y[j])
                # Reserved Partitions = Sum(p[j,a] * Reservation_Size)
                # check partitions fit into ECUs.
                base_overhead = ecu.cpu_cap * ecu.system_overhead_percent
                ram_base_overhead = ecu.ram_cap * ecu.system_overhead_percent
                rom_base_overhead = ecu.rom_cap * ecu.system_overhead_percent
                cpu_res_size = ecu.cpu_cap * ecu.partition_reservation_percent
                ram_res_size = ecu.ram_cap * ecu.partition_reservation_percent
                rom_res_size = ecu.rom_cap * ecu.partition_reservation_percent
                # Global ECU Capacity Constraint
                # Total usage (Overhead + Reservations) must fit in ECU
                relevant_ps = [p[j, a] for a in all_asils if (j, a) in p]
                if relevant_ps:
                    # CPU Global
                    total_reserved_cpu = gp.quicksum(relevant_ps) * cpu_res_size
                    model.addConstr(
                        (y[j] * base_overhead) + total_reserved_cpu <= ecu.cpu_cap,
                        name=f"ecu_global_cap_{ecu.id}"
                    )
                    
                    # RAM Global (Overhead + Reservations)
                    total_reserved_ram = gp.quicksum(relevant_ps) * ram_res_size
                    model.addConstr(
                        (y[j] * ram_base_overhead) + total_reserved_ram <= ecu.ram_cap,
                        name=f"ecu_global_ram_{ecu.id}"
                    )

                    # ROM Global (Overhead + Reservations)
                    total_reserved_rom = gp.quicksum(relevant_ps) * rom_res_size
                    model.addConstr(
                        (y[j] * rom_base_overhead) + total_reserved_rom <= ecu.rom_cap,
                        name=f"ecu_global_rom_{ecu.id}"
                    )

                
                # Partition Capacity Constraint
                # SWs inside a partition must fit into that partition's reservation
                # For each ASIL level 'a' supported by ECU 'j':
                for a in all_asils:
                    if (j, a) in p:
                        # Find all SCs with this ASIL 'a' that are compatible with ECU 'j'
                        scs_in_partition = [i for i in compat_scs_per_ecu[j] if scs[i].asil_req == a]
                        if scs_in_partition:
                            # 1. CPU Partition Capacity
                            model.addConstr(
                               gp.quicksum(x[i, j] * scs[i].cpu_req for i in scs_in_partition) <= cpu_res_size * p[j, a],
                               name=f"partition_cpu_cap_{ecu.id}_asil_{a}"
                            )
                            # 2. RAM Partition Capacity
                            ram_reservation_size = ecu.ram_cap * ecu.partition_reservation_percent
                            model.addConstr(
                               gp.quicksum(x[i, j] * scs[i].ram_req for i in scs_in_partition) <= ram_reservation_size * p[j, a],
                               name=f"partition_ram_cap_{ecu.id}_asil_{a}"
                            )
                            # 3. ROM Partition Capacity
                            rom_reservation_size = ecu.rom_cap * ecu.partition_reservation_percent
                            model.addConstr(
                               gp.quicksum(x[i, j] * scs[i].rom_req for i in scs_in_partition) <= rom_reservation_size * p[j, a],
                               name=f"partition_rom_cap_{ecu.id}_asil_{a}"
                            )


        # Constraint 4. Location Exclusivity: At most one ECU per location
        location_groups = {}
        for j, ecu in enumerate(ecus):
            if ecu.location:
                loc_key = (ecu.location.x, ecu.location.y)
                if loc_key not in location_groups:
                    location_groups[loc_key] = []
                location_groups[loc_key].append(j)
        for loc, group in location_groups.items():
            if len(group) > 1:
                model.addConstr(gp.quicksum(y[j] for j in group) <= 1, 
                              name=f"one_ecu_at_{loc}")

        # Constraint 5. Redundancy Constraints
        # Redundant SC pairs must be placed on different ECUs
        sc_id_to_idx = {sc.id: i for i, sc in enumerate(scs)}
        for i, sc in enumerate(scs):
            if sc.redundant_with:
                partner_id = sc.redundant_with
                if partner_id in sc_id_to_idx:
                    partner_idx = sc_id_to_idx[partner_id]
                    # Ensure we only add constraint once per pair (i < partner_idx)
                    if i < partner_idx:
                        # Only add constraint for ECUs compatible with BOTH SCs
                        common_ecus = set(compat_ecus_per_sc[i]).intersection(set(compat_ecus_per_sc[partner_idx]))
                        for j in common_ecus:
                            model.addConstr(x[i, j] + x[partner_idx, j] <= 1, 
                                          name=f"redundancy_{sc.id}_{partner_id}_{j}")

        # Constraint 6. Sensor Latency Constraints
        latency_map = {name: ct.latency_per_meter for name, ct in cable_types.items()}
        sensor_lookup = {s.id: s for s in sensors}
        for i in range(n_sc):
            for s_id in scs[i].sensors:
                sensor = sensor_lookup.get(s_id)
                if sensor and hasattr(sensor, 'max_latency') and sensor.max_latency:
                    for j in range(n_ecu):
                        if (i, j) in x:
                            if sensor.location and ecus[j].location:
                                dist = self._get_distance(sensor.location, ecus[j].location)
                                latency = dist * latency_map.get(sensor.interface, 0.0)
                                if latency > sensor.max_latency:
                                    model.addConstr(x[i, j] == 0, name=f"sensor_lat_{scs[i].id}_{sensor.id}_{ecus[j].id}")

        # Constraint 7. Actuator Latency Constraints
        actuator_lookup = {a.id: a for a in actuators}
        for i in range(n_sc):
            for a_id in scs[i].actuators:
                actuator = actuator_lookup.get(a_id)
                if actuator and hasattr(actuator, 'max_latency') and actuator.max_latency:
                    for j in range(n_ecu):
                        if (i, j) in x:
                            if actuator.location and ecus[j].location:
                                dist = self._get_distance(actuator.location, ecus[j].location)
                                latency = dist * latency_map.get(actuator.interface, 0.0)
                                if latency > actuator.max_latency:
                                    model.addConstr(x[i, j] == 0, name=f"actuator_lat_{scs[i].id}_{actuator.id}_{ecus[j].id}")

        # Constraint 8. ECU-ECU Communication Latency Constraints
        if comm_matrix:
            # Precompute ECU-ECU latencies
            cost_map = {name: ct.cost_per_meter for name, ct in cable_types.items()}
            latency_map = {name: ct.latency_per_meter for name, ct in cable_types.items()}
            ecu_ecu_latencies = {}
            for j1 in range(n_ecu):
                for j2 in range(j1 + 1, n_ecu):
                    dist = self._get_distance(ecus[j1].location, ecus[j2].location)
                    common = set(ecus[j1].interface_offered).intersection(set(ecus[j2].interface_offered))
                    if not common:
                        ecu_ecu_latencies[j1, j2] = dist * 1e6
                    else:
                        best_iface = max(common, key=lambda x: cost_map.get(x, 0))
                        ecu_ecu_latencies[j1, j2] = dist * latency_map.get(best_iface, 0)

            sc_id_map = {sc.id: i for i, sc in enumerate(scs)}
            for link in comm_matrix:
                src_id = link.get('src')
                dst_id = link.get('dst')
                max_lat = link.get('max_latency', float('inf'))

                if src_id not in sc_id_map or dst_id not in sc_id_map:
                    continue

                u = sc_id_map[src_id]
                v = sc_id_map[dst_id]

                for j1 in range(n_ecu):
                    for j2 in range(n_ecu):
                        if j1 == j2:
                            continue
                        if (u, j1) not in x or (v, j2) not in x:
                            continue

                        a, b = (j1, j2) if j1 < j2 else (j2, j1)
                        lat = ecu_ecu_latencies.get((a, b), 0)
                        if lat > max_lat:
                            model.addConstr(
                                x[u, j1] + x[v, j2] <= 1,
                                name=f"ecu_ecu_lat_{u}_{v}_{j1}_{j2}"
                            )
        

        # Constraint 9. Partition Logic
        # SC needs partition of its ASIL type
        for i, sc in enumerate(scs):
            a = sc.asil_req
            for j in compat_ecus_per_sc[i]:
                if (j, a) in p:
                    model.addConstr(x[i, j] <= p[j, a], name=f"link_partition_{sc.id}_{ecus[j].id}")
        
        model.write(f"model_ECU_Assignment.lp")
        print(f"Model written: model_ECU_Assignment.lp")

        return model, x, y       

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
        print("LEGO-BASED OPTIMIZATION (Partition + HW + Interface Costs)")
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
        print(f"  Partition config: cost={partitions.get('cost', 0)}, cpu_cap={partitions.get('cpu_cap', 0)}")
        
        # ======================= CREATE MODEL =======================
        model = gp.Model("LEGO_ECU_Assignment")
        model.setParam('OutputFlag', 1)
        model.setParam('MIPFocus', 2)
        model.update()
        
        # ======================= VARIABLES =======================
        # x[i,j] = SC i assigned to location j (binary)
        x = {}
        for i in range(n_sc):
            for j in range(n_locs):
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{scs[i].id}_{locations[j].id}")
        
        # y[j,a,p] = Location j has partition for ASIL a (replica p)
        # p ranges from 0 to max_partitions_per_asil_per_location
        max_partitions_per_asil_per_loc = 5  # Reasonable limit
        y = {}
        for j in range(n_locs):
            for a in unique_asils:
                for p in range(max_partitions_per_asil_per_loc):
                    y[j, a, p] = model.addVar(vtype=GRB.BINARY, name=f"y_{locations[j].id}_asil{a}_p{p}")
        
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
        
        print(f"\nVariables created:")
        print(f"  x: {len(x)} (SC-to-location)")
        print(f"  y: {len(y)} (partitions)")
        print(f"  hw_use: {len(hw_use)} (HW selection)")
        print(f"  if_use: {len(if_use)} (Interface selection)")
        
        # ======================= CONSTRAINTS =======================
        print(f"\nAdding constraints...")
        
        # Constraint 1: Every SC assigned to exactly one location
        for i in range(n_sc):
            model.addConstr(
                gp.quicksum(x[i, j] for j in range(n_locs)) == 1,
                name=f"assign_{scs[i].id}"
            )
        
        # Constraint 2: If SC assigned to location, matching ASIL partition must open
        for i in range(n_sc):
            a = scs[i].asil_req
            for j in range(n_locs):
                # x[i,j] = 1 → at least one partition of ASIL a must open
                model.addConstr(
                    x[i, j] <= gp.quicksum(y[j, a, p] for p in range(max_partitions_per_asil_per_loc)),
                    name=f"partition_required_{scs[i].id}_{locations[j].id}"
                )
        
        # Constraint 3: Capacity per location per ASIL partition
        for j in range(n_locs):
            for a in unique_asils:
                # Total CPU demand of SCs (ASIL a) assigned to location j
                sc_cpu_demand = gp.quicksum(
                    x[i, j] * scs[i].cpu_req
                    for i in range(n_sc) if scs[i].asil_req == a
                )
                # Number of partitions opened for ASIL a at location j
                num_partitions_a_j = gp.quicksum(y[j, a, p] for p in range(max_partitions_per_asil_per_loc))
                
                # Capacity constraint: total demand must fit in available partitions
                model.addConstr(
                    sc_cpu_demand <= num_partitions_a_j * partitions.get('cpu_cap', float('inf')),
                    name=f"capacity_{locations[j].id}_asil{a}"
                )
        
        # Constraint 4: HW feature required
        for i in range(n_sc):
            for h in (scs[i].hw_required or []):
                for j in range(n_locs):
                    model.addConstr(
                        x[i, j] <= hw_use[j, h],
                        name=f"hw_required_{scs[i].id}_{h}"
                    )
        
        # Constraint 5: Interface required
        for i in range(n_sc):
            for i_name in (scs[i].interface_required or []):
                for j in range(n_locs):
                    model.addConstr(
                        x[i, j] <= if_use[j, i_name],
                        name=f"if_required_{scs[i].id}_{i_name}"
                    )
        
        # Constraint 6: Redundancy Constraints
        # Redundant SC pairs must be placed on different locations
        sc_id_to_idx = {sc.id: i for i, sc in enumerate(scs)}
        for i, sc in enumerate(scs):
            if hasattr(sc, 'redundant_with') and sc.redundant_with:
                partner_id = sc.redundant_with
                if partner_id in sc_id_to_idx:
                    partner_idx = sc_id_to_idx[partner_id]
                    # Only add constraint once per pair (i < partner_idx)
                    if i < partner_idx:
                        for j in range(n_locs):
                            model.addConstr(
                                x[i, j] + x[partner_idx, j] <= 1,
                                name=f"redundancy_{sc.id}_{partner_id}_{locations[j].id}"
                            )
        
        # Constraint 7: Sensor Latency Constraints
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
        
        # Constraint 8: Actuator Latency Constraints
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
        
        # Constraint 9: Location-Location Communication Latency Constraints
        if comm_matrix:
            # Precompute location-location latencies
            cost_map = {name: ct.cost_per_meter for name, ct in cable_types.items()}
            loc_loc_latencies = {}
            for j1 in range(n_locs):
                for j2 in range(j1 + 1, n_locs):
                    dist = self._get_distance(locations[j1].location, locations[j2].location)
                    # Use the best (cheapest) interface available in cable_types
                    if all_interfaces:
                        best_iface = max(all_interfaces, key=lambda x: cost_map.get(x, 0))
                        loc_loc_latencies[j1, j2] = dist * latency_map.get(best_iface, 0)
                    else:
                        loc_loc_latencies[j1, j2] = dist * 1e6  # Fallback if no interfaces available
            
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
        
        # ======================= OBJECTIVE =======================
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
        
        # Total cost
        total_cost = partition_cost_expr + hw_cost_expr + if_cost_expr + cable_cost_expr
        model.setObjective(total_cost, GRB.MINIMIZE)
        
        print(f"Objective: Partition + HW + Interface + Cable costs")
        
        # ======================= SOLVE =======================
        print(f"\nSolving...")
        model.write("model_LEGO.lp")
        model.optimize()
        
        # ======================= EXTRACT SOLUTION =======================
        if model.status == GRB.OPTIMAL:
            print(f"\n✓ Optimal solution found!")
            sol = self._extract_solution_lego(x, y, hw_use, if_use, scs, locations, sensors, actuators, cable_types, partitions, hardwares, interfaces)
            return [sol]
        else:
            print(f"\n✗ Optimization failed. Status: {model.status}")
            return []

    def _extract_solution_lego(self, x, y, hw_use, if_use, scs, locations, sensors, actuators, cable_types, partitions, hardwares, interfaces):
        """Extract LEGO solution from optimized model"""
        assignment_map = {}
        num_locations_used = 0
        num_partitions_opened = 0
        hw_opened = []
        if_opened = []
        
        # Extract SC assignments
        for (i, j), var in x.items():
            if var.X > 0.5:
                assignment_map[scs[i].id] = locations[j].id
        
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
        for (i, j), var in x.items():
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
        
        total_cost = partition_cost + hw_cost + if_cost + cable_cost
        
        solution = {
            'assignment': assignment_map,
            'hardware_cost': hw_cost,  # For visualizer compatibility
            'hw_cost': hw_cost,
            'interface_cost': if_cost,
            'cable_cost': cable_cost,
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
