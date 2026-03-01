import pandas as pd
from tabulate import tabulate

class ReportGenerator:
    def __init__(self):
        pass

    def display_data_summary(self, scs, sensors, actuators, cable_types, comm_matrix, locations=None, ecus=None):
        print(f"\n Generated Data Summary:")
        if locations is not None:
            print(f"   - Locations: {len(locations)}")
        if ecus is not None:
            print(f"   - ECUs: {len(ecus)}")
        print(f"   - Software Components: {len(scs)}")
        print(f"   - Sensors: {len(sensors)}")
        print(f"   - Actuators: {len(actuators)}")
        print(f"   - Communication Links: {len(comm_matrix)}")
        print(f"   - Bus Types: {len(cable_types)}")

    def display_assignments(self,assignments):
            # Summary by location
            print(f"\n   SWs assigned per Location:")
            print(f"   [RAW ASSIGNMENT DATA]: {assignments}")
            loc_assignments = {}
            for sc_id, loc_id in assignments.items():
                if loc_id not in loc_assignments:
                    loc_assignments[loc_id] = []
                loc_assignments[loc_id].append(sc_id)
            
            for loc_id in sorted(loc_assignments.keys()):
                assigned_scs = loc_assignments[loc_id]
                sc_list = ", ".join(assigned_scs)
                print(f"      - {loc_id}: {len(assigned_scs)} SWs → [{sc_list}]")

    def display_sensors(self,df_sensors):
        print("\n" + "="*80)
        print("SENSORS")
        print("="*80)
        print(tabulate(df_sensors, headers="keys", tablefmt="grid", showindex=True))

    def display_actuators(self,df_actuators):
        print("\n" + "="*80)
        print("ACTUATORS")
        print("="*80)
        print(tabulate(df_actuators, headers="keys", tablefmt="grid", showindex=True))

    def display_scs(self,df_sc):
        print("\n" + "="*80)
        print("SOFTWARE COMPONENTS - Resource Requirements")
        print("="*80)
        # Convert to DataFrame if it's a list
        if isinstance(df_sc, list):
            df_sc = pd.DataFrame([vars(s) for s in df_sc])
        
        # First table: Resource requirements
        resource_cols = ['id', 'domain', 'cpu_req', 'ram_req', 'rom_req', 'asil_req', 'hw_required', 'redundant_with']
        df_resources = df_sc[resource_cols]
        print(tabulate(df_resources, headers="keys", tablefmt="grid", showindex=False))
        
        # Second table: Connectivity
        print("\n" + "="*80)
        print("SOFTWARE COMPONENTS - Connectivity")
        print("="*80)
        connectivity_cols = ['id', 'interface_required', 'sensors', 'actuators']
        df_connectivity = df_sc[connectivity_cols]
        print(tabulate(df_connectivity, headers="keys", tablefmt="grid", showindex=False))

    def display_ECUs(self,df_ecu):
        print("\n" + "="*80)
        print("ECUs")
        print("="*80)
        print(tabulate(df_ecu, headers="keys", tablefmt="grid", showindex=True))

    def display_locations(self, locations):
        print("\n" + "="*80)
        print("LOCATIONS")
        print("="*80)
        rows = [[loc.id, loc.location.x, loc.location.y] for loc in locations]
        print(tabulate(rows, headers=["id", "x", "y"], tablefmt="grid", showindex=False))

    def display_solution_details(self, solution, scs, locations, sensors=None, actuators=None):
        """
        Detailed solution analysis with comprehensive cost breakdown and assignment details
        """
        print("\n" + "="*80)
        print("DETAILED SOLUTION ANALYSIS")
        print("="*80)
        
        assignment = solution.get('assignment', {})
        
        # 1. ASSIGNMENT SUMMARY
        print("\n" + "-"*80)
        print("1. SOFTWARE COMPONENT ASSIGNMENTS")
        print("-"*80)
        
        sc_lookup = {sc.id: sc for sc in scs}
        loc_lookup = {loc.id: loc for loc in locations}
        
        # Group SCs by location
        location_mapping = {}
        for sc_id, loc_id in assignment.items():
            if loc_id not in location_mapping:
                location_mapping[loc_id] = []
            location_mapping[loc_id].append(sc_id)
        
        # Print assignment table
        assignment_rows = []
        for sc_id in sorted(assignment.keys()):
            loc_id = assignment[sc_id]
            sc = sc_lookup.get(sc_id, None)
            asil_str = f"ASIL{sc.asil_req}" if sc else "N/A"
            cpu_req = f"{sc.cpu_req}" if sc else "N/A"
            assignment_rows.append([sc_id, loc_id, asil_str, cpu_req])
        
        print(tabulate(
            assignment_rows,
            headers=["Software Component", "Location", "ASIL Level", "CPU Req"],
            tablefmt="grid",
            showindex=False
        ))
        
        # 2. LOCATION-WISE SUMMARY
        print("\n" + "-"*80)
        print("2. LOCATION-WISE ASSIGNMENT SUMMARY")
        print("-"*80)
        
        total_cpu_per_location = {}
        
        # First pass: calculate totals
        for loc_id in sorted(location_mapping.keys()):
            scs_at_loc = location_mapping[loc_id]
            total_cpu = sum(sc_lookup[sc_id].cpu_req for sc_id in scs_at_loc if sc_id in sc_lookup)
            total_cpu_per_location[loc_id] = total_cpu
        
        # Parse HW and Interface by location for detailed view
        hw_by_location = {}
        if_by_location = {}
        
        hw_features = solution.get('hw_features', [])
        for hw_str in hw_features:
            parts = hw_str.rsplit('@', 1)
            if len(parts) == 2:
                hw_name, loc_id = parts
                if loc_id not in hw_by_location:
                    hw_by_location[loc_id] = []
                hw_by_location[loc_id].append(hw_name)
        
        interfaces = solution.get('interfaces', [])
        for if_str in interfaces:
            parts = if_str.rsplit('@', 1)
            if len(parts) == 2:
                if_name, loc_id = parts
                if loc_id not in if_by_location:
                    if_by_location[loc_id] = []
                if_by_location[loc_id].append(if_name)
        
        # Display each location with its details
        for loc_id in sorted(location_mapping.keys()):
            scs_at_loc = sorted(location_mapping[loc_id])
            loc = loc_lookup.get(loc_id, None)
            
            # Location header
            loc_coords = f"({loc.location.x:.2f}, {loc.location.y:.2f})" if loc else "N/A"
            print(f"\n▸ {loc_id} {loc_coords}")
            print(f"   Status: {len(scs_at_loc)} SCs assigned | Total CPU: {total_cpu_per_location[loc_id]}")
            
            if hw_by_location.get(loc_id):
                print(f"   HW Features: {', '.join(sorted(set(hw_by_location[loc_id])))}")
            if if_by_location.get(loc_id):
                print(f"   Interfaces: {', '.join(sorted(set(if_by_location[loc_id])))}")
            
            # SCs assigned to this location
            sc_details = []
            for sc_id in scs_at_loc:
                sc = sc_lookup.get(sc_id)
                if sc:
                    asil = f"ASIL{sc.asil_req}"
                    cpu = sc.cpu_req
                    domain = getattr(sc, 'domain', 'N/A')
                    hw_req = ", ".join(sc.hw_required) if sc.hw_required else "-"
                    if_req = ", ".join(sc.interface_required) if sc.interface_required else "-"
                    
                    sc_details.append([sc_id, asil, cpu, domain, hw_req, if_req])
            
            if sc_details:
                print(tabulate(
                    sc_details,
                    headers=["SC ID", "ASIL", "CPU", "Domain", "HW Req", "IF Req"],
                    tablefmt="simple",
                    showindex=False
                ))
        
        # 3. PARTITION DETAILS
        print("\n" + "-"*80)
        print("3. PARTITION OPENINGS")
        print("-"*80)
        
        hw_features = solution.get('hw_features', [])
        if hw_features:
            # Parse HW features by location
            hw_by_location = {}
            for hw_str in hw_features:
                parts = hw_str.rsplit('@', 1)
                if len(parts) == 2:
                    hw_name, loc_id = parts
                    if loc_id not in hw_by_location:
                        hw_by_location[loc_id] = []
                    hw_by_location[loc_id].append(hw_name)
            
            partition_rows = []
            for loc_id in sorted(location_mapping.keys()):
                num_partitions = solution.get('num_partitions', 0)  # This is global in current impl
                partition_cost = solution.get('partition_cost', 0)
                partition_rows.append([
                    loc_id,
                    num_partitions,
                    partition_cost,
                    " | ".join(sorted(set(hw_by_location.get(loc_id, []))))
                ])
            
            print("Partition Config:")
            print(f"  Total Partitions Opened: {solution.get('num_partitions', 0)}")
            print(f"  Partition Cost (per partition): ${solution.get('partition_cost', 0):.2f}")
        
        # 4. HW FEATURES SUMMARY
        print("\n" + "-"*80)
        print("4. HARDWARE FEATURES OPENED")
        print("-"*80)
        
        if hw_features:
            hw_by_location = {}
            for hw_str in hw_features:
                parts = hw_str.rsplit('@', 1)
                if len(parts) == 2:
                    hw_name, loc_id = parts
                    if loc_id not in hw_by_location:
                        hw_by_location[loc_id] = []
                    hw_by_location[loc_id].append(hw_name)
            
            hw_rows = []
            for loc_id in sorted(hw_by_location.keys()):
                for hw_name in sorted(set(hw_by_location[loc_id])):
                    hw_rows.append([loc_id, hw_name])
            
            if hw_rows:
                print(tabulate(
                    hw_rows,
                    headers=["Location", "HW Feature"],
                    tablefmt="grid",
                    showindex=False
                ))
            else:
                print("  No HW features opened")
        else:
            print("  No HW features opened")
        
        # 5. INTERFACES SUMMARY
        print("\n" + "-"*80)
        print("5. INTERFACES OPENED")
        print("-"*80)
        
        interfaces = solution.get('interfaces', [])
        if interfaces:
            if_by_location = {}
            for if_str in interfaces:
                parts = if_str.rsplit('@', 1)
                if len(parts) == 2:
                    if_name, loc_id = parts
                    if loc_id not in if_by_location:
                        if_by_location[loc_id] = []
                    if_by_location[loc_id].append(if_name)
            
            if_rows = []
            for loc_id in sorted(if_by_location.keys()):
                for if_name in sorted(set(if_by_location[loc_id])):
                    if_rows.append([loc_id, if_name])
            
            if if_rows:
                print(tabulate(
                    if_rows,
                    headers=["Location", "Interface"],
                    tablefmt="grid",
                    showindex=False
                ))
            else:
                print("  No interfaces opened")
        else:
            print("  No interfaces opened")
        
        # 6. COST BREAKDOWN
        print("\n" + "-"*80)
        print("6. COST BREAKDOWN")
        print("-"*80)
        
        cost_rows = [
            ["Partition Cost", f"${solution.get('partition_cost', 0):.2f}"],
            ["HW Features Cost", f"${solution.get('hw_cost', 0):.2f}"],
            ["Interfaces Cost", f"${solution.get('interface_cost', 0):.2f}"],
            ["Cable Cost", f"${solution.get('cable_cost', 0):.2f}"],
            ["Communication Cost", f"${solution.get('comm_cost', 0):.2f}"],
            ["─" * 20, "─" * 15],
            ["TOTAL COST", f"${solution.get('total_cost', 0):.2f}"],
        ]
        
        print(tabulate(cost_rows, tablefmt="plain", showindex=False))
        
        # 7. RESOURCE METRICS
        print("\n" + "-"*80)
        print("7. RESOURCE METRICS")
        print("-"*80)
        
        metrics_rows = [
            ["Locations Used", solution.get('num_locations_used', 0)],
            ["Total Cable Length", f"{solution.get('cable_length', 0):.2f} m"],
            ["Optimization Status", solution.get('status', 'UNKNOWN')],
        ]
        
        print(tabulate(metrics_rows, headers=["Metric", "Value"], tablefmt="grid", showindex=False))
        
        # 8. CPU UTILIZATION PER LOCATION
        print("\n" + "-"*80)
        print("8. CPU UTILIZATION PER LOCATION")
        print("-"*80)
        
        cpu_rows = []
        for loc_id in sorted(location_mapping.keys()):
            cpu_used = total_cpu_per_location.get(loc_id, 0)
            # Note: Max CPU per partition is typically in config, showing raw usage for now
            cpu_rows.append([loc_id, cpu_used])
        
        print(tabulate(
            cpu_rows,
            headers=["Location", "CPU Used"],
            tablefmt="grid",
            showindex=False
        ))
        
        print("\n" + "="*80 + "\n")

    def display_partition(self, partitions):
        print("\n" + "="*80)
        print("PARTITION")
        print("="*80)
        partitions = partitions or {}
        row = [
            partitions.get('cost', 0),
            partitions.get('cpu_cap', 0),
            partitions.get('ram_cap', 0),
            partitions.get('rom_cap', 0)
        ]
        print(tabulate([row], headers=["cost", "cpu_cap", "ram_cap", "rom_cap"], tablefmt="grid", showindex=False))

    def display_hw_features(self, hardwares):
        print("\n" + "="*80)
        print("HW FEATURES")
        print("="*80)
        rows = [[k, v] for k, v in sorted((hardwares or {}).items())]
        print(tabulate(rows, headers=["hw", "cost"], tablefmt="grid", showindex=False))

    def display_interfaces(self, interface_costs):
        print("\n" + "="*80)
        print("INTERFACES")
        print("="*80)
        interfaces = interface_costs or {}
        rows = []
        for name, spec in sorted(interfaces.items()):
            rows.append([
                name,
                spec.name,
                spec.cost_per_meter,
                spec.latency_per_meter,
                spec.weight_per_meter,
                spec.capacity,
                spec.port_cost
            ])
        print(tabulate(rows, headers=["interface", "name", "cost/meter", "latency/meter", "weight/meter", "capacity", "port_cost"], tablefmt="grid", showindex=False))

    def display_data(self, df_sensors, df_actuators, df_sc, df_ecu=None, locations=None, hardwares=None, interface_costs=None, partitions=None):

        self.display_sensors(df_sensors)
        self.display_actuators(df_actuators)
        if locations is not None:
            self.display_locations(locations)
        if df_ecu is not None:
            # Display summary of generated ECUs
            self.display_ECUs(df_ecu)
        if partitions is not None:
            self.display_partition(partitions)
        if hardwares is not None:
            self.display_hw_features(hardwares)
        if interface_costs is not None:
            self.display_interfaces(interface_costs)
        self.display_scs(df_sc)

