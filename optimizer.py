import gurobipy as gp
from gurobipy import GRB
from models import Point, CableType
import math

class AssignmentOptimizer():
    def __init__(self, base_cost_per_meter=10.0):
        """
        Initialize Optimizer.
        base_cost_per_meter: Base unit cost for cabling scaling.
        """
        # Define detailed Cable Types
        self.cable_types = {
            'ETH': CableType(name='ETH', cost_per_meter=base_cost_per_meter * 5.0, latency_per_meter=0.005, weight_per_meter=50),
            'FLEXRAY': CableType(name='FLEXRAY', cost_per_meter=base_cost_per_meter * 3.0, latency_per_meter=0.005, weight_per_meter=30),
            'CAN': CableType(name='CAN', cost_per_meter=base_cost_per_meter * 1.0, latency_per_meter=0.005, weight_per_meter=20),
            'LIN': CableType(name='LIN', cost_per_meter=base_cost_per_meter * 0.5, latency_per_meter=0.005, weight_per_meter=10),
            'default': CableType(name='default', cost_per_meter=base_cost_per_meter * 1.0, latency_per_meter=0.005, weight_per_meter=20)
        }
    
    def get_cable_type(self, interface_type):
        return self.cable_types.get(interface_type, self.cable_types['default'])

    def calculate_cable_metrics_for_assignment(self, assignment, ecus, sensors, actuators, scs, 
                                            comm_matrix=None, distance_metric="euclidean"):
        """Calculate total cable cost, latency, and weight including breakdown."""
        missing_locs = [ecu.id for ecu in ecus if ecu.location is None]
        if missing_locs:
            return {
                'total_cable_cost': 0, 'total_latency': 0, 'total_weight': 0,
                'sensor_cable_cost': 0, 'actuator_cable_cost': 0, 'ecu_ecu_cable_cost': 0,
                'sensor_latency': 0, 'actuator_latency': 0, 'ecu_ecu_latency': 0,
                'sensor_weight': 0, 'actuator_weight': 0, 'ecu_ecu_weight': 0
            }
        
        # Initialize metrics
        metrics = {
            'sensor': {'cost': 0.0, 'latency': 0.0, 'weight': 0.0},
            'actuator': {'cost': 0.0, 'latency': 0.0, 'weight': 0.0},
            'ecu_ecu': {'cost': 0.0, 'latency': 0.0, 'weight': 0.0}
        }
        
        def add_metrics(category, p1, p2, interface):
            if p1 and p2:
                 _, dist = p1.dist(p2)
                 cable = self.get_cable_type(interface)
                 metrics[category]['cost'] += dist * cable.cost_per_meter
                 metrics[category]['latency'] += dist * cable.latency_per_meter
                 metrics[category]['weight'] += dist * cable.weight_per_meter

        # SENSOR → ECU
        for sc_id, ecu_id in assignment.items():
            sc = next((s for s in scs if s.id == sc_id), None)
            ecu = next((e for e in ecus if e.id == ecu_id), None)
            if not sc or not ecu: continue
            for sensor_id in sc.sensors:
                sensor = next((s for s in sensors if s.id == sensor_id), None)
                if sensor and sensor.location:
                     add_metrics('sensor', ecu.location, sensor.location, sensor.interface)
        
        # ACTUATOR → ECU
        for sc_id, ecu_id in assignment.items():
            sc = next((s for s in scs if s.id == sc_id), None)
            ecu = next((e for e in ecus if e.id == ecu_id), None)
            if not sc or not ecu: continue
            for actuator_id in sc.actuators:
                actuator = next((a for a in actuators if a.id == actuator_id), None)
                if actuator and actuator.location:
                    add_metrics('actuator', ecu.location, actuator.location, actuator.interface)
        
        # ECU ↔ ECU (Backbone)
        if comm_matrix:
            connected_pairs = set()
            # comm_matrix is list of dicts: {'src': sc_id, 'dst': sc_id, ...}
            for link in comm_matrix:
                sc_src_id = link['src']
                sc_dst_id = link['dst']
                ecu_src_id = assignment.get(sc_src_id)
                ecu_dst_id = assignment.get(sc_dst_id)
                
                if ecu_src_id and ecu_dst_id and ecu_src_id != ecu_dst_id:
                     # Use set to avoid double counting same link if undirected, 
                     # but comm_matrix is usually directed. Visualizer treats as undirected for drawing.
                     # Here physically it's one cable? Or separate? 
                     # Assume full duplex backbone or just one cable run.
                     # Let's count unique ECU pairs
                     connected_pairs.add(tuple(sorted((ecu_src_id, ecu_dst_id))))
            
            for ecu_id_a, ecu_id_b in connected_pairs:
                ecu_a = next((e for e in ecus if e.id == ecu_id_a), None)
                ecu_b = next((e for e in ecus if e.id == ecu_id_b), None)
                if ecu_a and ecu_b:
                     add_metrics('ecu_ecu', ecu_a.location, ecu_b.location, "ETH")

        total_cost = sum(m['cost'] for m in metrics.values())
        total_latency = sum(m['latency'] for m in metrics.values())
        total_weight = sum(m['weight'] for m in metrics.values())

        return {
            'total_cable_cost': total_cost,
            'total_latency': total_latency,
            'total_weight': total_weight,
            'sensor_cable_cost': metrics['sensor']['cost'],
            'actuator_cable_cost': metrics['actuator']['cost'],
            'ecu_ecu_cable_cost': metrics['ecu_ecu']['cost'],
            'sensor_latency': metrics['sensor']['latency'],
            'actuator_latency': metrics['actuator']['latency'],
            'ecu_ecu_latency': metrics['ecu_ecu']['latency'],
            'sensor_weight': metrics['sensor']['weight'],
            'actuator_weight': metrics['actuator']['weight'],
            'ecu_ecu_weight': metrics['ecu_ecu']['weight']
        }

    def _calculate_sc_cable_cost(self, sc_idx, ecu_idx, scs, ecus, sensors, actuators):
        """Helper to calculate cable cost potential if SC i is assigned to ECU j"""
        sc = scs[sc_idx]
        ecu = ecus[ecu_idx]
        if not ecu.location: return 0.0
        
        cost = 0.0
        # Sensors
        for s_id in sc.sensors:
            sensor = next((s for s in sensors if s.id == s_id), None)
            if sensor and sensor.location:
                _, dist = ecu.location.dist(sensor.location)
                cable = self.get_cable_type(sensor.interface)
                cost += dist * cable.cost_per_meter
        
        # Actuators
        for a_id in sc.actuators:
            actuator = next((a for a in actuators if a.id == a_id), None)
            if actuator and actuator.location:
                _, dist = ecu.location.dist(actuator.location)
                cable = self.get_cable_type(actuator.interface)
                cost += dist * cable.cost_per_meter
                
        return cost

    def pre_optimization_analysis(self, scs, ecus):
        """
        Perform pre-optimization analysis to check HW compatibility and capacity feasibility.
        
        Args:
            scs: List of SoftwareComponent objects
            ecus: List of CandidateECU objects
        """
        print("\n=== Pre-Optimization Analysis ===")
        print(f"Total SW Components: {len(scs)}")
        print(f"Total ECUs: {len(ecus)}")
        
        # Check ASIL compatibility
        print(f"\nASIL Levels Available:")
        asil_counts = {}
        for ecu in ecus:
            asil_counts[ecu.asil_level] = asil_counts.get(ecu.asil_level, 0) + 1
        for level in sorted(asil_counts.keys()):
            print(f"  ASIL-{level}: {asil_counts[level]} ECUs")
        
        print(f"\nASIL Requirements:")
        sc_asil_counts = {}
        for sc in scs:
            sc_asil_counts[sc.asil_req] = sc_asil_counts.get(sc.asil_req, 0) + 1
        for level in sorted(sc_asil_counts.keys()):
            print(f"  ASIL-{level}: {sc_asil_counts[level]} SCs")
        
        # Check HW feasibility
        infeasible_scs = []
        for sc in scs:
            compatible_ecus = [e for e in ecus if set(sc.hw_required).issubset(set(e.hw_offered)) and e.asil_level >= sc.asil_req]
            if not compatible_ecus:
                infeasible_scs.append(sc)
                hw_match = [e for e in ecus if set(sc.hw_required).issubset(set(e.hw_offered))]
                asil_match = [e for e in ecus if e.asil_level >= sc.asil_req]
                print(f"{sc.id}: HW={len(hw_match)} ECUs, ASIL>={sc.asil_req}={len(asil_match)} ECUs, compatible={len(compatible_ecus)}")
        
        if infeasible_scs:
            print(f"\nFound {len(infeasible_scs)} HW-incompatible SCs!")
            return {}
        
        # Check capacity feasibility
        total_cpu_demand = sum(sc.cpu_req for sc in scs)
        total_ram_demand = sum(sc.ram_req for sc in scs)
        total_rom_demand = sum(sc.rom_req for sc in scs)
        
        total_cpu_capacity = sum(e.cpu_cap for e in ecus)
        total_ram_capacity = sum(e.ram_cap for e in ecus)
        total_rom_capacity = sum(e.rom_cap for e in ecus)
        
        print(f"\nCapacity Analysis:")
        print(f"  CPU Demand: {total_cpu_demand} / Capacity: {total_cpu_capacity}")
        print(f"  RAM Demand: {total_ram_demand} / Capacity: {total_ram_capacity}")
        print(f"  ROM Demand: {total_rom_demand} / Capacity: {total_rom_capacity}")
        
        if (total_cpu_demand > total_cpu_capacity or 
            total_ram_demand > total_ram_capacity or 
            total_rom_demand > total_rom_capacity):
            print("Total demand exceeds total capacity!")
            return {}
        
        print("Pre-optimization checks passed!")

    def optimize_sw_to_ecu_assignment(self, scs, ecus, priority="ecu_count"):
        """
        Optimize SW-to-ECU assignment prioritizing ECU consolidation.
        
        Args:
            scs: List of SoftwareComponent objects
            ecus: List of CandidateECU objects
            priority: "ecu_count" (minimize ECU count), "cost" (minimize cost), "balanced" (both)
            
        Returns:
            Dictionary with assignment results: {sc_id: ecu_id}
        """
        
        print("\n=== Compact Assignment Optimization ===")
        print(f"Priority: {priority}")
        
        # Pre-optimization analysis
        self.pre_optimization_analysis(scs, ecus)

        # Create model
        model = gp.Model("SW_ECU_Assignment_Compact")
        model.setParam('OutputFlag', 0)
        
        # Assignment variable: x[i,j] = 1 if SC i assigned to ECU j
        x = {}
        for i, sc in enumerate(scs):
            for j, ecu in enumerate(ecus):
                if set(sc.hw_required).issubset(set(ecu.hw_offered)):
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        
        # ECU usage variable: y[j] = 1 if ECU j is used
        y = {}
        for j in range(len(ecus)):
            y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
        
        # Each SC must be assigned to exactly one ECU
        for i, sc in enumerate(scs):
            feasible_ecus = [j for j in range(len(ecus)) if (i, j) in x]
            if feasible_ecus:
                model.addConstr(gp.quicksum(x[i, j] for j in feasible_ecus) == 1, name=f"assign_{i}")
        
        # ECU usage constraint: if SC assigned to ECU, ECU must be marked as used
        for j in range(len(ecus)):
            for i in range(len(scs)):
                if (i, j) in x:
                    model.addConstr(x[i, j] <= y[j], name=f"usage_{i}_{j}")
        
        # ASIL constraint: ECU ASIL level must meet or exceed SC ASIL requirement
        for i, sc in enumerate(scs):
            for j, ecu in enumerate(ecus):
                if (i, j) in x:
                    model.addConstr(x[i, j] * ecu.asil_level >= x[i, j] * sc.asil_req, 
                                  name=f"asil_{i}_{j}")
        
        # Container constraint: number of SCs assigned to ECU must not exceed max_containers
        for j, ecu in enumerate(ecus):
            model.addConstr(gp.quicksum(x[i, j] for i in range(len(scs)) if (i, j) in x) <= ecu.max_containers,
                          name=f"containers_{j}")
        
        # Capacity constraints: CPU, RAM, ROM usage must not exceed ECU capacity
        for j, ecu in enumerate(ecus):
            cpu_constraint = gp.quicksum(x[i, j] * scs[i].cpu_req 
                                        for i in range(len(scs)) if (i, j) in x) <= ecu.cpu_cap
            model.addConstr(cpu_constraint, name=f"cpu_{j}")
            
            ram_constraint = gp.quicksum(x[i, j] * scs[i].ram_req 
                                        for i in range(len(scs)) if (i, j) in x) <= ecu.ram_cap
            model.addConstr(ram_constraint, name=f"ram_{j}")
            
            rom_constraint = gp.quicksum(x[i, j] * scs[i].rom_req 
                                        for i in range(len(scs)) if (i, j) in x) <= ecu.rom_cap
            model.addConstr(rom_constraint, name=f"rom_{j}")
        
        # Multi-objective approach
        cost_expr = gp.quicksum(x[i, j] * ecus[j].cost for i, j in x)
        ecu_count_expr = gp.quicksum(y[j] for j in range(len(ecus)))
        
        if priority == "ecu_count":
            # Heavily prioritize minimizing ECU count
            objective = 1000 * ecu_count_expr + 0.1 * cost_expr
            print("→ Minimizing ECU count (cost is secondary)")
        elif priority == "cost":
            # Minimize cost (original approach)
            objective = cost_expr
            print("→ Minimizing cost")
        else:  # balanced
            # Balance between cost and ECU count
            objective = 100 * ecu_count_expr + cost_expr
            print("→ Balanced approach")
        
        model.setObjective(objective, GRB.MINIMIZE)
        
        # Optimize
        model.optimize()
        
        # Extract solution
        assignments = {}
        if model.status == GRB.OPTIMAL:
            print(f"\n Optimization Status: OPTIMAL")
            
            # Calculate metrics
            total_cost = 0
            ecus_used = set()
            for i, j in x:
                if x[i, j].X > 0.5:
                    assignments[scs[i].id] = ecus[j].id
                    ecus_used.add(j)
                    total_cost += ecus[j].cost
            
            print(f"   Total Cost: ${total_cost:.2f}")
            print(f"   ECUs Used: {len(ecus_used)}")
            
        elif model.status == GRB.INFEASIBLE:
            print(f"\n Optimization Status: INFEASIBLE")
            model.computeIIS()
            print("Infeasible constraints:")
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"  - {c.ConstrName}")
        else:
            print(f"\n Optimization Status: {model.status}")
        
        return assignments

    def optimize_pareto_epsilon_constraint(self, scs, ecus, max_ecu_count=None, num_points=5):
        """
        Generate Pareto front using Epsilon-Constraint method.
        Solves multiple times, each time optimizing one objective while constraining others.
        
        This method guarantees to find ALL Pareto-optimal solutions.
        
        Args:
            scs: List of SoftwareComponent objects
            ecus: List of CandidateECU objects
            max_ecu_count: Maximum ECU count to consider (if None, auto-calculated)
            num_points: Number of points on Pareto front to generate
            
        Returns:
            List of Pareto solutions with different trade-offs
        """

        ### TODO: Implement Redundancy Isolation Constraint
        ### TODO: Implement Container-based Resource
        ### TODO:
        print("\n=== Pareto Front via Epsilon-Constraint Method ===")
        self.pre_optimization_analysis(scs, ecus)
        
        if max_ecu_count is None:
            # Auto-calculate minimum possible ECUs
            max_ecu_count = len(ecus)
        
        pareto_solutions = []
        epsilon_values = []
        
        # First pass: Find minimum and maximum ECU count possible
        print("\nPhase 1: Finding ECU count bounds...")
        
        # Min ECUs: Optimize for ECU minimization
        model_min = gp.Model("Find_Min_ECUs")
        model_min.setParam('OutputFlag', 0)
        
        x_min = {}
        for i, sc in enumerate(scs):
            for j, ecu in enumerate(ecus):
                if set(sc.hw_required).issubset(set(ecu.hw_offered)): # HW compatibility is enforced here
                    x_min[i, j] = model_min.addVar(vtype=GRB.BINARY)
        
        y_min = {}
        for j in range(len(ecus)):
            y_min[j] = model_min.addVar(vtype=GRB.BINARY)
        
        for i in range(len(scs)):
            feasible = [j for j in range(len(ecus)) if (i, j) in x_min]
            if feasible:
                model_min.addConstr(gp.quicksum(x_min[i, j] for j in feasible) == 1)
        
        for j in range(len(ecus)):
            for i in range(len(scs)):
                if (i, j) in x_min:
                    model_min.addConstr(x_min[i, j] <= y_min[j])
            
            # ASIL constraint: ECU ASIL level must meet SC requirement
            for i, sc in enumerate(scs):
                if (i, j) in x_min:
                    model_min.addConstr(x_min[i, j] * ecus[j].asil_level >= x_min[i, j] * sc.asil_req)
            
            # Container constraint: number of SCs <= max_containers
            model_min.addConstr(gp.quicksum(x_min[i, j] for i in range(len(scs)) if (i, j) in x_min) <= ecus[j].max_containers)
            
            cpu = gp.quicksum(x_min[i, j] * scs[i].cpu_req 
                             for i in range(len(scs)) if (i, j) in x_min) <= ecus[j].cpu_cap
            ram = gp.quicksum(x_min[i, j] * scs[i].ram_req 
                             for i in range(len(scs)) if (i, j) in x_min) <= ecus[j].ram_cap
            rom = gp.quicksum(x_min[i, j] * scs[i].rom_req 
                             for i in range(len(scs)) if (i, j) in x_min) <= ecus[j].rom_cap
            model_min.addConstr(cpu)
            model_min.addConstr(ram)
            model_min.addConstr(rom)
        
        model_min.setObjective(gp.quicksum(y_min[j] for j in range(len(ecus))), GRB.MINIMIZE)
        model_min.optimize()
        
        min_ecu_count = int(model_min.ObjVal) if model_min.status == GRB.OPTIMAL else 1
        print(f"  Minimum ECUs needed: {min_ecu_count}")
        print(f"  Maximum ECUs available: {max_ecu_count}")
        
        # Generate epsilon values (ECU count constraints)
        epsilon_values = []
        for i in range(num_points):
            alpha = i / max(1, num_points - 1)
            ecu_count = int(min_ecu_count + alpha * (max_ecu_count - min_ecu_count))
            epsilon_values.append(ecu_count)
        
        print(f"\nPhase 2: Exploring Pareto front...")
        print(f"  ε values (ECU constraints): {epsilon_values}")
        
        # Second pass: For each epsilon value, minimize cost
        for idx, ecu_limit in enumerate(epsilon_values):
            print(f"\n  Solving with ECU ≤ {ecu_limit}...")
            
            model = gp.Model(f"Pareto_Epsilon_{ecu_limit}")
            model.setParam('OutputFlag', 0)
            
            x = {}
            for i, sc in enumerate(scs):
                for j, ecu in enumerate(ecus):
                    if set(sc.hw_required).issubset(set(ecu.hw_offered)):
                        x[i, j] = model.addVar(vtype=GRB.BINARY)
            
            y = {}
            for j in range(len(ecus)):
                y[j] = model.addVar(vtype=GRB.BINARY)
            
            # Each SC assigned to exactly one ECU
            for i in range(len(scs)):
                feasible = [j for j in range(len(ecus)) if (i, j) in x]
                if feasible:
                    model.addConstr(gp.quicksum(x[i, j] for j in feasible) == 1)
            
            # ECU usage tracking: if SC assigned to ECU, ECU marked as used
            for j in range(len(ecus)):
                for i in range(len(scs)):
                    if (i, j) in x:
                        model.addConstr(x[i, j] <= y[j])
            
            # ASIL constraint: ECU ASIL level must meet or exceed SC ASIL requirement
            for i, sc in enumerate(scs):
                for j, ecu in enumerate(ecus):
                    if (i, j) in x:
                        model.addConstr(x[i, j] * ecu.asil_level >= x[i, j] * sc.asil_req)
            
            # Epsilon constraint: limit total ECUs for Pareto trade-off analysis
            model.addConstr(gp.quicksum(y[j] for j in range(len(ecus))) <= ecu_limit, 
                          name=f"ecu_limit_{ecu_limit}")
            
            # Capacity constraints: CPU, RAM, ROM must fit in ECU
            for j, ecu in enumerate(ecus):
                cpu = gp.quicksum(x[i, j] * scs[i].cpu_req 
                                 for i in range(len(scs)) if (i, j) in x) <= ecu.cpu_cap
                ram = gp.quicksum(x[i, j] * scs[i].ram_req 
                                 for i in range(len(scs)) if (i, j) in x) <= ecu.ram_cap
                rom = gp.quicksum(x[i, j] * scs[i].rom_req 
                                 for i in range(len(scs)) if (i, j) in x) <= ecu.rom_cap
                model.addConstr(cpu)
                model.addConstr(ram)
                model.addConstr(rom)
            
            # Minimize cost given the ECU limit constraint
            cost_expr = gp.quicksum(x[i, j] * ecus[j].cost for i, j in x)
            model.setObjective(cost_expr, GRB.MINIMIZE)
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                assignments = {}
                ecus_used = set()
                total_cost = 0
                
                for i, j in x:
                    if x[i, j].X > 0.5:
                        assignments[scs[i].id] = ecus[j].id
                        ecus_used.add(j)
                        total_cost += ecus[j].cost
                
                solution_data = {
                    'assignment': assignments,
                    'total_cost': total_cost,
                    'num_ecus_used': len(ecus_used),
                    'ecu_limit': ecu_limit,
                    'method': 'epsilon_constraint'
                }
                pareto_solutions.append(solution_data)
                
                print(f"    ✓ Cost: ${total_cost:.2f}, ECUs: {len(ecus_used)}/{ecu_limit}")
            else:
                print(f"    ✗ Infeasible with ECU limit {ecu_limit}")
        
        print(f"\n✓ Found {len(pareto_solutions)} Pareto-optimal solutions")
        return pareto_solutions

    def optimize_pareto_cost_vs_loadbalance(self, scs, ecus, sensors, actuators, comm_matrix=None, num_points=5):
        """
        Generate Pareto front: Cost vs Load Balancing using Epsilon-Constraint.
        
        Objectives:
        - Cost: minimize total ECU cost
        - Load Balancing: minimize maximum ECU utilization
        
        Args:
            scs: List of SoftwareComponent objects
            ecus: List of CandidateECU objects
            num_points: Number of points on Pareto front
            
        Returns:
            List of Pareto solutions with cost/load-balance trade-offs
        """
        print("\n=== Pareto Front: Cost vs Load Balancing ===")
        self.pre_optimization_analysis(scs, ecus)
        
        pareto_solutions = []
        
        # Phase 1: Find min and max max_utilization possible
        print("\nPhase 1: Finding load balancing bounds...")
        
        # Find minimum max_utilization (best load balancing)
        model_min = gp.Model("Find_Min_MaxUtil")
        model_min.setParam('OutputFlag', 0)
        
        x_min = {}
        for i, sc in enumerate(scs):
            for j, ecu in enumerate(ecus):
                if set(sc.hw_required).issubset(set(ecu.hw_offered)):
                    x_min[i, j] = model_min.addVar(vtype=GRB.BINARY)
        
        y_min = {}
        for j in range(len(ecus)):
            y_min[j] = model_min.addVar(vtype=GRB.BINARY)
        
        max_util_var = model_min.addVar(vtype=GRB.CONTINUOUS, name="max_util")
        
        # Assignment constraints
        for i in range(len(scs)):
            feasible = [j for j in range(len(ecus)) if (i, j) in x_min]
            if feasible:
                model_min.addConstr(gp.quicksum(x_min[i, j] for j in feasible) == 1)
        
        # ECU usage tracking
        for j in range(len(ecus)):
            for i in range(len(scs)):
                if (i, j) in x_min:
                    model_min.addConstr(x_min[i, j] <= y_min[j])
        
        # Capacity constraints
        for j, ecu in enumerate(ecus):
            cpu = gp.quicksum(x_min[i, j] * scs[i].cpu_req 
                             for i in range(len(scs)) if (i, j) in x_min) <= ecu.cpu_cap
            ram = gp.quicksum(x_min[i, j] * scs[i].ram_req 
                             for i in range(len(scs)) if (i, j) in x_min) <= ecu.ram_cap
            rom = gp.quicksum(x_min[i, j] * scs[i].rom_req 
                             for i in range(len(scs)) if (i, j) in x_min) <= ecu.rom_cap
            model_min.addConstr(cpu)
            model_min.addConstr(ram)
            model_min.addConstr(rom)
        
        # Max utilization constraints: for each ECU, utilization <= max_util
        for j, ecu in enumerate(ecus):
            cpu_demand = gp.quicksum(x_min[i, j] * scs[i].cpu_req 
                                    for i in range(len(scs)) if (i, j) in x_min)
            ram_demand = gp.quicksum(x_min[i, j] * scs[i].ram_req 
                                    for i in range(len(scs)) if (i, j) in x_min)
            rom_demand = gp.quicksum(x_min[i, j] * scs[i].rom_req 
                                    for i in range(len(scs)) if (i, j) in x_min)
            
            # Utilization = (cpu_demand/cpu_cap + ram_demand/ram_cap + rom_demand/rom_cap) / 3
            # Simplified: track max component utilization
            total_capacity = ecu.cpu_cap + ecu.ram_cap + ecu.rom_cap
            total_demand = cpu_demand + ram_demand + rom_demand
            
            model_min.addConstr(total_demand <= max_util_var * total_capacity)
        
        model_min.setObjective(max_util_var, GRB.MINIMIZE)
        model_min.optimize()
        
        min_util = model_min.ObjVal if model_min.status == GRB.OPTIMAL else 0.5
        max_util = 1.0
        
        print(f"  Minimum max-utilization: {min_util:.1%}")
        print(f"  Maximum max-utilization: {max_util:.1%}")
        
        # Generate epsilon values (load balancing levels)
        util_levels = []
        for i in range(num_points):
            alpha = i / max(1, num_points - 1)
            util = min_util + alpha * (max_util - min_util)
            util_levels.append(util)
        
        print(f"\nPhase 2: Exploring Pareto front...")
        print(f"  Load balancing levels: {[f'{u:.1%}' for u in util_levels]}")
        
        # Phase 2: For each load balancing level, minimize cost (Hardware + Weighted Cable)
        for util_limit in util_levels:
            print(f"\n  Solving with max-utilization ≤ {util_limit:.1%}...")
            
            model = gp.Model(f"Pareto_LoadBalance_{util_limit:.2f}")
            model.setParam('OutputFlag', 0)
            
            x = {}
            for i, sc in enumerate(scs):
                for j, ecu in enumerate(ecus):
                    if set(sc.hw_required).issubset(set(ecu.hw_offered)):
                        x[i, j] = model.addVar(vtype=GRB.BINARY)
            
            y = {}
            for j in range(len(ecus)):
                y[j] = model.addVar(vtype=GRB.BINARY)
            
            max_util = model.addVar(vtype=GRB.CONTINUOUS, name="max_util")
            
            # Assignment constraints
            for i in range(len(scs)):
                feasible = [j for j in range(len(ecus)) if (i, j) in x]
                if feasible:
                    model.addConstr(gp.quicksum(x[i, j] for j in feasible) == 1)
            
            # ECU usage tracking
            for j in range(len(ecus)):
                for i in range(len(scs)):
                    if (i, j) in x:
                        model.addConstr(x[i, j] <= y[j])
            
            # ASIL constraint
            for i, sc in enumerate(scs):
                for j, ecu in enumerate(ecus):
                    if (i, j) in x:
                        model.addConstr(x[i, j] * ecu.asil_level >= x[i, j] * sc.asil_req)
            
            # Container constraint: number of SCs <= max_containers
            for j, ecu in enumerate(ecus):
                model.addConstr(gp.quicksum(x[i, j] for i in range(len(scs)) if (i, j) in x) <= ecu.max_containers)
            
            # Capacity constraints
            for j, ecu in enumerate(ecus):
                cpu = gp.quicksum(x[i, j] * scs[i].cpu_req 
                                 for i in range(len(scs)) if (i, j) in x) <= ecu.cpu_cap
                ram = gp.quicksum(x[i, j] * scs[i].ram_req 
                                 for i in range(len(scs)) if (i, j) in x) <= ecu.ram_cap
                rom = gp.quicksum(x[i, j] * scs[i].rom_req 
                                 for i in range(len(scs)) if (i, j) in x) <= ecu.rom_cap
                model.addConstr(cpu)
                model.addConstr(ram)
                model.addConstr(rom)
            
            # Load balancing constraint: max utilization <= limit
            for j, ecu in enumerate(ecus):
                cpu_demand = gp.quicksum(x[i, j] * scs[i].cpu_req 
                                        for i in range(len(scs)) if (i, j) in x)
                ram_demand = gp.quicksum(x[i, j] * scs[i].ram_req 
                                        for i in range(len(scs)) if (i, j) in x)
                rom_demand = gp.quicksum(x[i, j] * scs[i].rom_req 
                                        for i in range(len(scs)) if (i, j) in x)
                
                total_capacity = ecu.cpu_cap + ecu.ram_cap + ecu.rom_cap
                total_demand = cpu_demand + ram_demand + rom_demand
                
                model.addConstr(total_demand <= util_limit * total_capacity)
            
            # Objective: Minimize HW Cost + Weighted Cable Cost
            hw_cost = gp.quicksum(x[i, j] * ecus[j].cost for i, j in x)
            
            # Add cable costs (pre-calculated with weights)
            cable_cost_expr = 0
            for i, j in x:
                c_cost = self._calculate_sc_cable_cost(i, j, scs, ecus, sensors, actuators)
                cable_cost_expr += x[i, j] * c_cost
            
            model.setObjective(hw_cost + cable_cost_expr, GRB.MINIMIZE)
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                assignments = {}
                ecus_used = set()
                total_cost = 0
                
                for i, j in x:
                    if x[i, j].X > 0.5:
                        assignments[scs[i].id] = ecus[j].id
                        ecus_used.add(j)
                        total_cost += ecus[j].cost
                
                # Calculate actual max utilization
                actual_max_util = 0
                for j, ecu in enumerate(ecus):
                    if j in ecus_used:
                        cpu_used = sum(scs[i].cpu_req for i in range(len(scs)) 
                                      if (i, j) in x and x[i, j].X > 0.5)
                        ram_used = sum(scs[i].ram_req for i in range(len(scs)) 
                                      if (i, j) in x and x[i, j].X > 0.5)
                        rom_used = sum(scs[i].rom_req for i in range(len(scs)) 
                                      if (i, j) in x and x[i, j].X > 0.5)
                        total_capacity = ecu.cpu_cap + ecu.ram_cap + ecu.rom_cap
                        total_used = cpu_used + ram_used + rom_used
                        util = total_used / total_capacity if total_capacity > 0 else 0
                        actual_max_util = max(actual_max_util, util)
                
                # Calculate cable cost
                cable_info = self.calculate_cable_metrics_for_assignment(
                    assignments, ecus, sensors, actuators, scs, comm_matrix, "euclidean"
                )
                total_cost_with_cable = total_cost + cable_info['total_cable_cost']
                
                solution_data = {
                    'assignment': assignments,
                    'hardware_cost': total_cost,
                    'cable_cost': cable_info,
                    'total_cost': total_cost_with_cable,
                    'num_ecus_used': len(ecus_used),
                    'max_utilization': actual_max_util,
                    'load_balance_limit': util_limit,
                    'method': 'pareto_cost_vs_loadbalance'
                }
                pareto_solutions.append(solution_data)
                
                print(f"    ✓ HW: ${total_cost:.2f} | Cable: ${cable_info['total_cable_cost']:.2f} | Latency: {cable_info['total_latency']*1000:.2f}us | Util: {actual_max_util:.1%}")
            else:
                print(f"    ✗ Infeasible with load balance limit {util_limit:.1%}")
        
        print(f"\n✓ Found {len(pareto_solutions)} Pareto-optimal solutions")
        return pareto_solutions
