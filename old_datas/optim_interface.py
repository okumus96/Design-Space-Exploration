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


# We need change direct link b/w the ECUs to some sort of network topology.
# We did not include uncertainty in overall system.
# We do not know numbers we gave is correct (i.e., cost, latency,  etc.)
# WE do not know our objective functions is correct. (semanatically, we can change distance with complexity model)
# we need to get rid of the ECU types, model them as a legos of HWs and partitions.


class OptimizationInterface:
    def __init__(self):
        pass

    def _comm_volume(self, link):
        """Traffic demand for a comm link (same unit as CableType.capacity)."""
        try:
            return float(link.get('volume', 1.0))
        except Exception:
            return 1.0
    
    def _extract_solution_from_model(self, x, y, z, scs, ecus, sensors, actuators, cable_types, objective_name):
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

        # ECU-ECU link contributions based on chosen interface (z)
        ecu_links = {}
        for (j1, j2, iface), var in z.items():
            if var.X > 0.5:
                dist = self._get_distance(ecus[j1].location, ecus[j2].location)
                if dist > 0:
                    cable_cost_val += dist * cost_map.get(iface, 0.0)
                    cable_length += dist
                ecu_links[(ecus[j1].id, ecus[j2].id)] = iface
        
        hw_cost_val = sum(ecus[j].cost for j in range(len(ecus)) if y[j].X > 0.5)
        
        solution = {
            'assignment': assignment_map,
            'hardware_cost': hw_cost_val,
            'cable_cost': cable_cost_val,
            'total_cost': hw_cost_val + cable_cost_val,
            'cable_length': cable_length,
            'num_ecus_used': used_ecus,
            'objective': objective_name,
            'ecu_links': ecu_links,
            'status': "OPTIMAL"
        }
        return solution

    def _get_distance(self, loc1, loc2):
        if not loc1 or not loc2:
            return 0.0
        dist_manhattan, _ = loc1.dist(loc2)
        return dist_manhattan
    
    def calculate_cable_expressions(self, x, y, z, scs, ecus, sensors, actuators, cable_types, comm_matrix):
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
            ecu = ecus[j]
            cable_cost = 0.0
            cable_distance = 0.0
            latency = 0.0
            
            # Sum sensor cable costs, distances, and latencies
            for s_id in sc.sensors:
                sensor = sensor_lookup.get(s_id)
                if sensor and sensor.location and ecu.location:
                    dist = self._get_distance(sensor.location, ecu.location)
                    cable_cost += dist * cost_map.get(sensor.interface, 0.0)
                    cable_distance += dist
                    latency += dist * latency_map.get(sensor.interface, 0.0)
            
            # Sum actuator cable costs, distances, and latencies
            for a_id in sc.actuators:
                actuator = actuator_lookup.get(a_id)
                if actuator and actuator.location and ecu.location:
                    dist = self._get_distance(actuator.location, ecu.location)
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

        # ECU-ECU link contributions (linear in chosen interface z)
        ecu_ecu_cost_terms = []
        ecu_ecu_distance_terms = []
        ecu_ecu_latency_terms = []

        for (j1, j2, iface), zvar in z.items():
            dist = self._get_distance(ecus[j1].location, ecus[j2].location)
            if dist <= 0:
                continue
            ecu_ecu_cost_terms.append((zvar, dist * cost_map.get(iface, 0.0)))
            ecu_ecu_distance_terms.append((zvar, dist))
            ecu_ecu_latency_terms.append((zvar, dist * latency_map.get(iface, 0.0)))
        
        # Create compact linear expressions
        if cable_cost_terms:
            cable_cost_expr = gp.quicksum(var * cost for var, cost in cable_cost_terms)
        else:
            cable_cost_expr = 0
        if ecu_ecu_cost_terms:
            cable_cost_expr += gp.quicksum(var * cost for var, cost in ecu_ecu_cost_terms)
            
        if cable_distance_terms:
            cable_distance_expr = gp.quicksum(var * dist for var, dist in cable_distance_terms)
        else:
            cable_distance_expr = 0
        if ecu_ecu_distance_terms:
            cable_distance_expr += gp.quicksum(var * dist for var, dist in ecu_ecu_distance_terms)
            
        if latency_terms:
            latency_expr = gp.quicksum(var * lat for var, lat in latency_terms)
        else:
            latency_expr = 0
        if ecu_ecu_latency_terms:
            latency_expr += gp.quicksum(var * lat for var, lat in ecu_ecu_latency_terms)
            
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

        # z[j1,j2,iface] = 1 if interface 'iface' is selected for ECU-ECU link (link exists iff sum(z)=1)
        z = {}
        z_by_pair = {}
        cable_ifaces = set(cable_types.keys())

        for j1 in range(n_ecu):
            for j2 in range(j1 + 1, n_ecu):
                common = set(ecus[j1].interface_offered).intersection(set(ecus[j2].interface_offered)).intersection(cable_ifaces)
                if not common:
                    # No common physical interface -> cannot create a link between these two ECUs
                    continue

                zvars = []
                z_by_pair[j1, j2] = []
                for iface in sorted(common):
                    z[j1, j2, iface] = model.addVar(vtype=GRB.BINARY ,name=f"z_{ecus[j1].id}_{ecus[j2].id}_{iface}")
                    zvars.append(z[j1, j2, iface])
                    z_by_pair[j1, j2].append((iface, z[j1, j2, iface]))

                # Link existence is sum(zvars). Allow at most one interface.
                model.addConstr(gp.quicksum(zvars) <= 1, name=f"choose_iface_{j1}_{j2}")
                model.addConstr(gp.quicksum(zvars) <= y[j1], name=f"link_used1_{j1}_{j2}")
                model.addConstr(gp.quicksum(zvars) <= y[j2], name=f"link_used2_{j1}_{j2}")

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

        connectable_pairs = set(z_by_pair.keys())


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
                            # Assuming RAM supports reservation same way as CPU (using partition_reservation_percent)
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

        # ==========================================
        # Constraint 8 & 9: SMART CAPACITY (Loopback-Aware)
        # ==========================================
        
        model.update()

        # --- 1. HAZIRLIK: SC Trafiklerini Topla ---
        # Her SC'nin ürettiği toplam veri hacmi
        sc_volume_map = {sc.id: 0.0 for sc in scs}
        
        # Kritik linkler listesi
        CRITICAL_VOLUME_THRESHOLD = 0.0
        critical_links = []
        
        sc_id_map = {sc.id: i for i, sc in enumerate(scs)}
        cap_map = {name: ct.capacity for name, ct in cable_types.items()}
        MAX_ECU_THROUGHPUT = float('inf')  

        if comm_matrix:
            for link in comm_matrix:
                src = link['src']
                dst = link['dst']
                vol = self._comm_volume(link)
                
                # SC bazlı toplam çıkış
                if src in sc_volume_map: sc_volume_map[src] += vol
                
                # Kritik link
                if vol >= CRITICAL_VOLUME_THRESHOLD:
                    if src in sc_id_map and dst in sc_id_map:
                        critical_links.append({
                            'u': sc_id_map[src], 'v': sc_id_map[dst], 'vol': vol
                        })

        print(f"Adding Smart Capacity Constraints (Loopback Ignored)...")

        # --- 2. KISIT: Node-Based Interface Capacity ---
        # Sadece ECU DIŞINA çıkan trafiği kontrol et.
        # "Eğer SC_u ECU_j'de ise, ürettiği verinin hepsi kabloya gitmez."
        # "Sadece başka ECU'daki SC'lere giden veri kabloya gider."
        # Ancak bunu 'route' olmadan tam hesaplamak zordur.
        # BASİTLEŞTİRİLMİŞ ÇÖZÜM:
        # Total_SC_Output <= Interface_Capacity + (Aynı ECU'da kalma ihtimali için devasa bir gevşetme payı)
        # YERİNE DAHA İYİSİ: Slack Variable ile Soft Constraint yapalım.
        
        total_violation = 0

        for j in range(n_ecu):
            # 1. Bu ECU'ya atanan SC'lerin toplam veri üretimi
            total_generated = gp.quicksum(x[i, j] * sc_volume_map[scs[i].id] for i in range(n_sc) if (i, j) in x)
            
            # 2. Donanım (CPU/Bus) Limiti (Bu her türlü geçerli)
            # Slack ekliyoruz ki patlamasın, ne kadar aştığını görelim.
            s_cpu = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"slack_cpu_{j}")
            model.addConstr(total_generated <= MAX_ECU_THROUGHPUT * y[j] + s_cpu, name=f"cpu_lim_{j}")
            
            # 3. Arayüz Kapasitesi (Kablo Limiti)
            # Burası kritik: Interface'ler toplam trafiği taşımalı.
            # z_by_pair'den kapasiteyi doğru çekme (Unpacking Fix)
            
            # Çıkan Kablolar (j -> k)
            out_cap = gp.quicksum(
                z_var * cap_map.get(iface_name, 0)
                for k in range(n_ecu) if k > j 
                for iface_name, z_var in z_by_pair.get((j, k), [])
            )
            # Gelen Kablolar (k -> j) - Full Duplex varsayımıyla ikisini topluyoruz
            in_cap = gp.quicksum(
                z_var * cap_map.get(iface_name, 0)
                for k in range(n_ecu) if k < j 
                for iface_name, z_var in z_by_pair.get((k, j), [])
            )
            
            total_iface_cap = out_cap + in_cap
            
            # Slack variable for interface capacity
            s_iface = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"slack_iface_{j}")
            
            # KISIT:
            # Üretilen Trafik <= Kablo Kapasitesi + Slack + (Kendi içi trafik payı)
            # Kendi içi trafik payı için "Internal Bus" varsayıyoruz.
            # Eğer slack çok büyürse, demek ki kablo yetmiyor.
            model.addConstr(total_generated <= total_iface_cap + 1e5*(1-y[j]) + s_iface, name=f"iface_lim_{j}")
            
            total_violation += s_cpu + s_iface


        # --- 3. KISIT: Critical Link Check (Büyük Paketler) ---
        # Bu kısım kesinlikle "Kablo Seçimi" yaptırır.
        
        for idx, item in enumerate(critical_links):
            u, v, vol = item['u'], item['v'], item['vol']
            
            possible_j1 = compat_ecus_per_sc[u]
            possible_j2 = compat_ecus_per_sc[v]
            
            for j1 in possible_j1:
                for j2 in possible_j2:
                    if j1 == j2: continue # Aynı ECU ise kablo gerekmez, kısıt yok.
                    
                    if j1 < j2: pair = (j1, j2)
                    else: pair = (j2, j1)
                    
                    if pair not in connectable_pairs: continue

                    # Link Kapasitesi
                    link_cap = gp.quicksum(
                        z_var * cap_map.get(iface_name, 0)
                        for iface_name, z_var in z_by_pair.get(pair, [])
                    )
                    
                    s_crit = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"slack_crit_{idx}")
                    
                    # Eğer SC_u -> j1 ve SC_v -> j2 ise:
                    model.addConstr(
                        vol * (x[u, j1] + x[v, j2] - 1) <= link_cap + s_crit,
                        name=f"crit_check_{idx}_{j1}_{j2}"
                    )
                    total_violation += s_crit
        

        # Constraint 10. Partition Logic
        # SC needs partition of its ASIL type
        for i, sc in enumerate(scs):
            a = sc.asil_req
            for j in compat_ecus_per_sc[i]:
                if (j, a) in p:
                    model.addConstr(x[i, j] <= p[j, a], name=f"link_partition_{sc.id}_{ecus[j].id}")
        
        model.write(f"model_ECU_Assignment.lp")
        print(f"Model written: model_ECU_Assignment.lp")

        return model, x, y, z

    def optimize(self, scs, ecus, sensors, actuators, cable_types, comm_matrix, enable_latency=True):

        # 1. Create Base Model (Constraints)
        model, x, y, z = self.create_base_model(scs, ecus, sensors, actuators, comm_matrix, cable_types, verbose=True, mip_gap=0.0)

        # 2. Calculate Cable Costs, Distances, and Latency (sensor/actuator only)
        cable_cost_expr, cable_distance_expr, latency_expr = self.calculate_cable_expressions(
            x, y, z, scs, ecus, sensors, actuators, cable_types, comm_matrix
        )
        
        solutions = []
        # 3. Objective 1: HW Cost + Cable Cost
        print(f"\n[1] Finding Extremity 1: Minimum Cost...")
        hw_cost = gp.quicksum(y[j] * ecus[j].cost for j in range(len(ecus)))
        total_cost = hw_cost + cable_cost_expr
        model.setObjective(1, GRB.MINIMIZE)
        model.optimize()
        
        if model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            model.write("model_iis.ilp")
            print("IIS written to model_iis.ilp")
            return []
            
        if model.status == GRB.OPTIMAL:
            best_distance_at_cost = cable_distance_expr.getValue()
            #best_cost = model.objVal
            sol_1 = self._extract_solution_from_model(x, y, z, scs, ecus, sensors, actuators, cable_types, "Min Total Cost")
            solutions.append(sol_1)
        else:
            print(f"Optimization Failed for Objective 1. Status Code: {model.Status}")
            return []

        # 4. Objective 2: Cable Distance
        print(f"\n[2] Finding Extremity 2: Minimum Cable Distance...")
        model.setObjective(cable_distance_expr, GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            model.write("model_iis.ilp")
            print("IIS written to model_iis.ilp")
            return []

        if model.status == GRB.OPTIMAL:
            best_distance = model.objVal
            #best_cost_at_distance = total_cost.getValue()
            sol_2 = self._extract_solution_from_model(x, y, z, scs, ecus, sensors, actuators, cable_types, "Min Cable Distance")
            solutions.append(sol_2)
        else:
            print(f"Optimization Failed for Objective 2. Status Code: {model.Status}")
            return []
        
        # 5. Objective 3: Pareto Points Between Extremities
        print(f"\n[3] Finding Pareto Points Between Extremities...")
        eps_values = np.linspace(best_distance_at_cost, best_distance, 5)[1:-1]
        print(f" Epsilon values for Pareto exploration: {eps_values}")
        eps_constr = None

        for idx, eps in enumerate(eps_values, 1):
            if eps_constr is not None:
                model.remove(eps_constr)
                model.update()
            eps_constr = model.addConstr(cable_distance_expr <= eps, name="eps_dist")
            model.setObjective(total_cost, GRB.MINIMIZE)
            model.optimize()
            if model.status == GRB.OPTIMAL:
                sol_eps = self._extract_solution_from_model(
                    x, y, z, scs, ecus, sensors, actuators, cable_types, f"Eps {idx}"
                )
                solutions.append(sol_eps)
            else:
                print(f"-> No solution for eps={eps:.2f} (Status {model.Status})")

        return solutions
