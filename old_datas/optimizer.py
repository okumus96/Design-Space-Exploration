import os
os.environ["GRB_LICENSE_FILE"] = "/home/frk/gurobi.lic"
import gurobipy as gp
from gurobipy import GRB
from models import Point, CableType
import math

class AssignmentOptimizer():
    def __init__(self):
        """
        Initialize Optimizer.
        """
        pass

    def _precompute_cable_data(self, scs, ecus, sensors, actuators, comm_matrix=None):
        """
        Precompute all cable length coefficients for optimization.
        
        Returns:
            cable_coeffs: dict {(sc_idx, ecu_idx): length} for sensor+actuator cables
            ecu_pair_distances: dict {(ecu_idx1, ecu_idx2): distance} for backbone
            comm_links: list of (src_sc_idx, dst_sc_idx) tuples from comm_matrix
        """
        # Sensor + Actuator coefficients for each (SC, ECU) pair
        cable_coeffs = {}
        for i, sc in enumerate(scs):
            for j, ecu in enumerate(ecus):
                if not ecu.location: continue
                coeff = 0.0
                for s_id in sc.sensors:
                    sensor = next((s for s in sensors if s.id == s_id), None)
                    if sensor and sensor.location:
                        _, dist = ecu.location.dist(sensor.location)
                        coeff += dist
                for a_id in sc.actuators:
                    actuator = next((a for a in actuators if a.id == a_id), None)
                    if actuator and actuator.location:
                        _, dist = ecu.location.dist(actuator.location)
                        coeff += dist
                cable_coeffs[(i, j)] = coeff
        
        # ECU pair distances for backbone
        ecu_pair_distances = {}
        for j1 in range(len(ecus)):
            for j2 in range(j1 + 1, len(ecus)):
                ecu1, ecu2 = ecus[j1], ecus[j2]
                if ecu1.location and ecu2.location:
                    _, dist = ecu1.location.dist(ecu2.location)
                    ecu_pair_distances[(j1, j2)] = dist
        
        # Communication links (SC indices)
        comm_links = []
        if comm_matrix:
            for link in comm_matrix:
                sc_src_id = link['src']
                sc_dst_id = link['dst']
                src_idx = next((i for i, s in enumerate(scs) if s.id == sc_src_id), None)
                dst_idx = next((i for i, s in enumerate(scs) if s.id == sc_dst_id), None)
                if src_idx is not None and dst_idx is not None:
                    comm_links.append((src_idx, dst_idx))
        
        return cable_coeffs, ecu_pair_distances, comm_links
    
    def _calculate_total_cable_length(self, assignment_indices, cable_coeffs, ecu_pair_distances, comm_links):
        """
        Calculate total cable length for a given assignment using precomputed data.
        
        Args:
            assignment_indices: dict {sc_idx: ecu_idx}
            cable_coeffs: precomputed (sc_idx, ecu_idx) -> length
            ecu_pair_distances: precomputed (ecu_idx1, ecu_idx2) -> distance
            comm_links: list of (src_sc_idx, dst_sc_idx)
        
        Returns:
            float: total cable length
        """
        total = 0.0
        
        # Sensor + Actuator cables
        for sc_idx, ecu_idx in assignment_indices.items():
            total += cable_coeffs.get((sc_idx, ecu_idx), 0.0)
        
        # ECU-ECU backbone
        connected_pairs = set()
        for src_idx, dst_idx in comm_links:
            ecu_src = assignment_indices.get(src_idx)
            ecu_dst = assignment_indices.get(dst_idx)
            if ecu_src is not None and ecu_dst is not None and ecu_src != ecu_dst:
                key = (min(ecu_src, ecu_dst), max(ecu_src, ecu_dst))
                connected_pairs.add(key)
        
        for key in connected_pairs:
            total += ecu_pair_distances.get(key, 0.0)
        
        return total

    def _check_interface_compatibility(self, sc, ecu):
        """Check if ECU offers all interfaces required by SC."""
        if not sc.interface_required:
            return True
        return set(sc.interface_required).issubset(set(ecu.interface_offered))

    def _check_full_compatibility(self, sc, ecu):
        """Check HW + Interface + ASIL compatibility."""
        hw_ok = set(sc.hw_required).issubset(set(ecu.hw_offered))
        iface_ok = self._check_interface_compatibility(sc, ecu)
        asil_ok = ecu.asil_level >= sc.asil_req
        return hw_ok and iface_ok and asil_ok

    def optimize(self, scs, ecus, sensors, actuators, cable_types, comm_matrix=None, num_points=5):
        """
        Generate Pareto front: HW Cost vs Cable Length using Epsilon-Constraint.
        
        Objectives:
        - Minimize HW Cost (ECU hardware)
        - Minimize Cable Length
        
        Constraints:
        - HW compatibility, Interface compatibility, ASIL compatibility
        - Capacity (CPU, RAM, ROM), Container limit
        
        Returns:
            List of Pareto-optimal solutions
        """
        print("\n=== Pareto Front: HW Cost vs Cable Length ===")
        
        # Precompute all cable data once
        cable_coeffs, ecu_pair_distances, comm_links = self._precompute_cable_data(
            scs, ecus, sensors, actuators, comm_matrix
        )
        
        # Phase 1: Find min HW cost (ignoring cable)
        print("\nPhase 1: Finding HW cost bounds...")
        
        model_cost = gp.Model("Min_HW_Cost")
        model_cost.setParam('OutputFlag', 0)
        
        x = {}
        for i, sc in enumerate(scs):
            for j, ecu in enumerate(ecus):
                if self._check_full_compatibility(sc, ecu):
                    x[i, j] = model_cost.addVar(vtype=GRB.BINARY)
        
        y = {}
        for j in range(len(ecus)):
            y[j] = model_cost.addVar(vtype=GRB.BINARY)
        
        # Constraints
        for i in range(len(scs)):
            feasible = [j for j in range(len(ecus)) if (i, j) in x]
            if feasible:
                model_cost.addConstr(gp.quicksum(x[i, j] for j in feasible) == 1)
        
        for j in range(len(ecus)):
            for i in range(len(scs)):
                if (i, j) in x:
                    model_cost.addConstr(x[i, j] <= y[j])
        
        for j, ecu in enumerate(ecus):
            model_cost.addConstr(gp.quicksum(x[i, j] * scs[i].cpu_req for i in range(len(scs)) if (i, j) in x) <= ecu.cpu_cap)
            model_cost.addConstr(gp.quicksum(x[i, j] * scs[i].ram_req for i in range(len(scs)) if (i, j) in x) <= ecu.ram_cap)
            model_cost.addConstr(gp.quicksum(x[i, j] * scs[i].rom_req for i in range(len(scs)) if (i, j) in x) <= ecu.rom_cap)
            model_cost.addConstr(gp.quicksum(x[i, j] for i in range(len(scs)) if (i, j) in x) <= ecu.max_containers)
        
        model_cost.setObjective(gp.quicksum(y[j] * ecus[j].cost for j in range(len(ecus))), GRB.MINIMIZE)
        model_cost.optimize()
        
        if model_cost.status != GRB.OPTIMAL:
            print("✗ No feasible solution!")
            return []
        
        min_cost = model_cost.ObjVal
        
        # Phase 2: Minimize cable length to explore the other Pareto extreme
        model_cable = gp.Model("Min_Cable")
        model_cable.setParam('OutputFlag', 0)
        
        x2 = {}
        for i, sc in enumerate(scs):
            for j, ecu in enumerate(ecus):
                if self._check_full_compatibility(sc, ecu):
                    x2[i, j] = model_cable.addVar(vtype=GRB.BINARY)
        
        y2 = {}
        for j in range(len(ecus)):
            y2[j] = model_cable.addVar(vtype=GRB.BINARY)
        
        for i in range(len(scs)):
            feasible = [j for j in range(len(ecus)) if (i, j) in x2]
            if feasible:
                model_cable.addConstr(gp.quicksum(x2[i, j] for j in feasible) == 1)
        
        for j in range(len(ecus)):
            for i in range(len(scs)):
                if (i, j) in x2:
                    model_cable.addConstr(x2[i, j] <= y2[j])
        
        for j, ecu in enumerate(ecus):
            model_cable.addConstr(gp.quicksum(x2[i, j] * scs[i].cpu_req for i in range(len(scs)) if (i, j) in x2) <= ecu.cpu_cap)
            model_cable.addConstr(gp.quicksum(x2[i, j] * scs[i].ram_req for i in range(len(scs)) if (i, j) in x2) <= ecu.ram_cap)
            model_cable.addConstr(gp.quicksum(x2[i, j] * scs[i].rom_req for i in range(len(scs)) if (i, j) in x2) <= ecu.rom_cap)
            model_cable.addConstr(gp.quicksum(x2[i, j] for i in range(len(scs)) if (i, j) in x2) <= ecu.max_containers)
        
        # Backbone variables for Phase 2
        backbone_vars = {}
        for src_idx, dst_idx in comm_links:
            for j1 in range(len(ecus)):
                for j2 in range(len(ecus)):
                    if j1 == j2: continue
                    if (src_idx, j1) not in x2 or (dst_idx, j2) not in x2: continue
                    key = (min(j1, j2), max(j1, j2))
                    if key in backbone_vars or key not in ecu_pair_distances: continue
                    
                    z = model_cable.addVar(vtype=GRB.BINARY)
                    backbone_vars[key] = z
                    model_cable.addConstr(z <= y2[key[0]])
                    model_cable.addConstr(z <= y2[key[1]])
                    model_cable.addConstr(z >= y2[key[0]] + y2[key[1]] - 1)
        
        # Build cable objective using precomputed data
        cable_expr = gp.LinExpr()
        for (i, j), coeff in cable_coeffs.items():
            if (i, j) in x2:
                cable_expr += x2[i, j] * coeff
        for key, z in backbone_vars.items():
            cable_expr += z * ecu_pair_distances[key]
        
        model_cable.setObjective(cable_expr, GRB.MINIMIZE)
        model_cable.optimize()
        
        max_cost = sum(ecus[j].cost for j in range(len(ecus)) if y2[j].X > 0.5)
        print(f"  HW Cost Range: ${min_cost:.0f} to ${max_cost:.0f}")
        
        # Phase 3: Generate Pareto points
        cost_levels = [min_cost + i * (max_cost - min_cost) / (num_points - 1) for i in range(num_points)]
        print(f"\nPhase 3: Exploring {num_points} Pareto points...")
        
        pareto_solutions = []
        
        for cost_limit in cost_levels:
            model = gp.Model(f"Pareto_{cost_limit:.0f}")
            model.setParam('OutputFlag', 0)
            
            xp = {}
            for i, sc in enumerate(scs):
                for j, ecu in enumerate(ecus):
                    if self._check_full_compatibility(sc, ecu):
                        xp[i, j] = model.addVar(vtype=GRB.BINARY)
            
            yp = {}
            for j in range(len(ecus)):
                yp[j] = model.addVar(vtype=GRB.BINARY)
            
            for i in range(len(scs)):
                feasible = [j for j in range(len(ecus)) if (i, j) in xp]
                if feasible:
                    model.addConstr(gp.quicksum(xp[i, j] for j in feasible) == 1)
            
            for j in range(len(ecus)):
                for i in range(len(scs)):
                    if (i, j) in xp:
                        model.addConstr(xp[i, j] <= yp[j])
            
            for j, ecu in enumerate(ecus):
                model.addConstr(gp.quicksum(xp[i, j] * scs[i].cpu_req for i in range(len(scs)) if (i, j) in xp) <= ecu.cpu_cap)
                model.addConstr(gp.quicksum(xp[i, j] * scs[i].ram_req for i in range(len(scs)) if (i, j) in xp) <= ecu.ram_cap)
                model.addConstr(gp.quicksum(xp[i, j] * scs[i].rom_req for i in range(len(scs)) if (i, j) in xp) <= ecu.rom_cap)
                model.addConstr(gp.quicksum(xp[i, j] for i in range(len(scs)) if (i, j) in xp) <= ecu.max_containers)
            
            # Cost constraint (epsilon)
            model.addConstr(gp.quicksum(yp[j] * ecus[j].cost for j in range(len(ecus))) <= cost_limit)
            
            # Backbone variables for this iteration
            backbone_vars_p = {}
            for src_idx, dst_idx in comm_links:
                for j1 in range(len(ecus)):
                    for j2 in range(len(ecus)):
                        if j1 == j2: continue
                        if (src_idx, j1) not in xp or (dst_idx, j2) not in xp: continue
                        key = (min(j1, j2), max(j1, j2))
                        if key in backbone_vars_p or key not in ecu_pair_distances: continue
                        
                        z = model.addVar(vtype=GRB.BINARY)
                        backbone_vars_p[key] = z
                        model.addConstr(z <= yp[key[0]])
                        model.addConstr(z <= yp[key[1]])
                        model.addConstr(z >= yp[key[0]] + yp[key[1]] - 1)
            
            # Build cable objective using precomputed data
            cable = gp.LinExpr()
            for (i, j), coeff in cable_coeffs.items():
                if (i, j) in xp:
                    cable += xp[i, j] * coeff
            for key, z in backbone_vars_p.items():
                cable += z * ecu_pair_distances[key]
            
            model.setObjective(cable, GRB.MINIMIZE)
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                assignment_indices = {}
                ecus_used = set()
                for i, j in xp:
                    if xp[i, j].X > 0.5:
                        assignment_indices[i] = j
                        ecus_used.add(j)
                
                hw_cost = sum(ecus[j].cost for j in ecus_used)
                cable_length = self._calculate_total_cable_length(
                    assignment_indices, cable_coeffs, ecu_pair_distances, comm_links
                )
                
                # Convert to SC/ECU IDs for output
                assignments = {scs[i].id: ecus[j].id for i, j in assignment_indices.items()}
                
                pareto_solutions.append({
                    'assignment': assignments,
                    'hardware_cost': hw_cost,
                    'cable_length': cable_length,
                    'num_ecus_used': len(ecus_used)
                })
                
                print(f"HW: ${hw_cost:.0f} | Cable: {cable_length:.1f}m")
        
        print(f"\Found {len(pareto_solutions)} Pareto solutions")
        return pareto_solutions
