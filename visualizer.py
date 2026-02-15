import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import pandas as pd
import seaborn as sns
import os
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
from tabulate import tabulate


class Visualization:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory for results: {self.save_dir}")

    def save_plot(self, filename):
        if self.save_dir:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
            print(f"Saved plot to: {filepath}")
            plt.close()
        else:
            plt.show()

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
    

    def plot_charts(self, sc_list, sensor_list, actuator_list, ecu_list=None):
        """Convert lists to DataFrames and plot charts"""
        # Convert lists to DataFrames if not already
        if isinstance(sc_list, list):
            df_sc = pd.DataFrame([vars(s) for s in sc_list])
        else:
            df_sc = sc_list
            
        df_ecu = None
        if ecu_list is not None:
            if isinstance(ecu_list, list):
                df_ecu = pd.DataFrame([vars(e) for e in ecu_list])
            else:
                df_ecu = ecu_list
            
        if isinstance(sensor_list, list):
            df_sensors = pd.DataFrame([vars(s) for s in sensor_list])
        else:
            df_sensors = sensor_list
            
        if isinstance(actuator_list, list):
            df_actuators = pd.DataFrame([vars(a) for a in actuator_list])
        else:
            df_actuators = actuator_list
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Hardware vs Software Distribution', fontsize=16)

        # ECU Distribution (optional)
        if df_ecu is not None and 'type' in df_ecu.columns:
            sns.countplot(x='type', data=df_ecu, ax=axes[0, 0], palette='viridis')
            axes[0, 0].set_title('Distribution of Candidate ECU Types')
        else:
            axes[0, 0].axis('off')
            axes[0, 0].set_title('')

        #  SWC Distribution in terms of Domain
        df_sc['type_guess'] = df_sc['id'].apply(lambda x: x.split('_')[-1])
        sns.countplot(x='type_guess', data=df_sc, hue='type_guess', ax=axes[0, 1], palette='magma', legend=False)
        axes[0, 1].set_title('Distribution of SWC Workloads')

        # Sensor Type Distribution
        sns.countplot(x='type', data=df_sensors, hue='type', ax=axes[1, 0], palette='Set2', legend=False)
        axes[1, 0].set_title('Distribution of Sensor Types')
        axes[1, 0].set_xlabel('Sensor Type')
        axes[1, 0].set_ylabel('Count')

        # Actuator Type Distribution
        sns.countplot(x='type', data=df_actuators, hue='type', ax=axes[1, 1], palette='Set1', legend=False)
        axes[1, 1].set_title('Distribution of Actuator Types')
        axes[1, 1].set_xlabel('Actuator Type')
        axes[1, 1].set_ylabel('Count')

        plt.tight_layout()
        self.save_plot("hw_sw_distribution.png")

    def plot_sw_sensor_actuator_graph_final(self, scs, sensors, actuators, comm_matrix, filename="system_graph.png"):
        """
        SW-Sensor-Actuator connection graph visualization with custom layout:
        - SWs in outer ring (circular layout)
        - Sensors and Actuators inside the ring
        """
        import math
        
        plt.figure(figsize=(18, 14)) 
        ax = plt.gca()
        G = nx.Graph()
        
        # Sensor and actuator lookup dictionaries
        sensor_dict = {s.id: s for s in sensors}
        actuator_dict = {a.id: a for a in actuators}
        
        # Add Nodes
        for sc in scs:
            G.add_node(sc.id, node_type='SW', domain=sc.domain, hw_required=sc.hw_required)
        for s in sensors:
            G.add_node(s.id, node_type='Sensor', sensor_type=s.type, interface=s.interface, volume=s.volume)
        for a in actuators:
            G.add_node(a.id, node_type='Actuator', actuator_type=a.type, interface=a.interface, volume=a.volume)
            
        # Add Edges: SW -> Sensor/Actuator (with volume information)
        for sc in scs:
            for sid in getattr(sc, 'sensors', []):
                sensor_obj = sensor_dict.get(sid)
                if sensor_obj:
                    G.add_edge(sc.id, sid, edge_type='sensor', interface=sensor_obj.interface, volume=sensor_obj.volume)
            for aid in getattr(sc, 'actuators', []):
                actuator_obj = actuator_dict.get(aid)
                if actuator_obj:
                    G.add_edge(sc.id, aid, edge_type='actuator', interface=actuator_obj.interface, volume=actuator_obj.volume)
        
        # Add Edges: SW -> SW (communication links)
        for comm in comm_matrix:
            G.add_edge(comm['src'], comm['dst'], edge_type='comm', 
                    volume=comm['volume'], latency=comm['max_latency'])

        # Custom circular layout for SWs with sensors/actuators inside
        pos = {}
        
        # Place SWs in a circle (outer ring)
        sw_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'SW']
        num_sws = len(sw_nodes)
        ring_radius = 3.0  # Outer ring radius
        
        for i, sw_id in enumerate(sw_nodes):
            angle = 2 * math.pi * i / num_sws
            x = ring_radius * math.cos(angle)
            y = ring_radius * math.sin(angle)
            pos[sw_id] = (x, y)
        
        # Place sensors and actuators inside the ring
        sensor_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'Sensor']
        actuator_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'Actuator']
        
        inner_radius = 1.2  # Inner radius for sensors/actuators
        num_inner = len(sensor_nodes) + len(actuator_nodes)
        
        # Place sensors
        for i, sensor_id in enumerate(sensor_nodes):
            angle = 2 * math.pi * i / num_inner
            x = inner_radius * math.cos(angle)
            y = inner_radius * math.sin(angle)
            pos[sensor_id] = (x, y)
        
        # Place actuators (after sensors, continuing around)
        for i, actuator_id in enumerate(actuator_nodes):
            idx = len(sensor_nodes) + i
            angle = 2 * math.pi * idx / num_inner
            x = inner_radius * math.cos(angle)
            y = inner_radius * math.sin(angle)
            pos[actuator_id] = (x, y)

        node_type_map = {
            'SW': ('tab:blue', 'o'), 
            'Sensor': ('tab:green', 's'), 
            'Actuator': ('tab:orange', '^')
        }
        
        interface_colors = {
            'CAN':     '#382501',  
            'ETH':     '#4ECDC4', 
            'LIN':     "#2FB914",  
            'FLEXRAY': "#F0094E",
        }
        
        node_labels = {n: n for n in G.nodes()}

        # Draw nodes
        for ntype, (color, shape) in node_type_map.items():
            nodelist = [n for n, d in G.nodes(data=True) if d['node_type'] == ntype]
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_shape=shape,
                                node_color=color, label=ntype, alpha=0.9, node_size=1200)

        # Draw SW-Sensor and SW-Actuator edges
        for u, v, d in G.edges(data=True):
            if d.get('edge_type') in ['sensor', 'actuator']:
                interface = d.get('interface', 'CAN')
                color = interface_colors.get(interface, 'gray')
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, width=2.0, alpha=0.7)
        
        # SW-SW Communication edges
        for u, v, d in G.edges(data=True):
            if d.get('edge_type') == 'comm':
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='darkgray', 
                                    width=2.0, alpha=0.6, style='dashed')

        # Node labels
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')

        # Position labels: HW Required (for SWs only)
        for sc in scs:
            if sc.id in pos:
                node_pos = pos[sc.id]
                hw_text = '\n'.join(sc.hw_required) if sc.hw_required else 'No HW'
                ax.text(node_pos[0], node_pos[1] - 0.35, hw_text, 
                    fontsize=7, ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.5, edgecolor='orange', linewidth=1),
                    weight='bold')
        
        # Edge labels: Interface and Volume information
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            if d.get('edge_type') in ['sensor', 'actuator']:
                interface = d.get('interface', 'CAN')
                volume = d.get('volume', 0)
                edge_labels[(u, v)] = f"{interface}\n{volume}MB/s"
            elif d.get('edge_type') == 'comm':
                latency = d.get('latency', '')
                volume = d.get('volume', 0)
                edge_labels[(u, v)] = f"{latency}ms\n{volume}MB/s"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, font_color='darkred')

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='SW', markerfacecolor='tab:blue', markersize=15),
            Line2D([0], [0], marker='s', color='w', label='Sensor', markerfacecolor='tab:green', markersize=15),
            Line2D([0], [0], marker='^', color='w', label='Actuator', markerfacecolor='tab:orange', markersize=15),
            Line2D([0], [0], color="#382501", lw=2.5, label='CAN Interface'),
            Line2D([0], [0], color='#4ECDC4', lw=2.5, label='ETH Interface'),
            Line2D([0], [0], color="#2FB914", lw=2.5, label='LIN Interface'),
            Line2D([0], [0], color="#F0094E", lw=2.5, label='FLEXRAY Interface'),
            Line2D([0], [0], color='darkgray', lw=2, linestyle='dashed', label='SW-SW Communication'),
        ]
        
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
        plt.title('Complete System Graph: Ring Layout (SWs outer ring, Sensors/Actuators inside)', fontsize=14, weight='bold')
        plt.axis('off')
        plt.tight_layout()
        self.save_plot(filename)

    def visualize_pareto_front(self, pareto_solutions, filename="pareto_front_analysis.png"):
        """
        Create a visualization of the Pareto front showing trade-offs.
        
        Args:
            pareto_solutions: List of solutions from optimize_pareto_epsilon_constraint
        """
        if not pareto_solutions:
            print("No solutions to visualize")
            return
        
        # Extract data
        cable_lengths = [sol['cable_length'] for sol in pareto_solutions]
        total_costs = [sol['total_cost'] for sol in pareto_solutions]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 1, figsize=(14, 5))
        
        # Plot 1: Pareto Front (Cable Length vs Total Cost)
        ax1 = axes
        ax1.scatter(cable_lengths, total_costs, s=200, alpha=0.6, c='blue', edgecolors='black', linewidth=2)
        
        # Add solution numbers
        for i, (length, cost) in enumerate(zip(cable_lengths, total_costs), 1):
            ax1.annotate(f'S{i}', (length, cost), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
        
        # Add line connecting solutions (Sorted by cable length)
        sorted_indices = sorted(range(len(cable_lengths)), key=lambda i: cable_lengths[i])
        sorted_lengths = [cable_lengths[i] for i in sorted_indices]
        sorted_costs = [total_costs[i] for i in sorted_indices]
        ax1.plot(sorted_lengths, sorted_costs, 'b--', alpha=0.3, linewidth=1)
        
        ax1.set_xlabel('Total Cable Length (m)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
        ax1.set_title('Pareto Front: Cost vs Cable Length', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_plot(filename)

    def visualize_optimization_result(self,scs, ecus, sensors, actuators, assignments, filename="optimization_result.png"):
        """
        ECU'ları kutular olarak görselleştir, SW'leri içinde göster
        """
        
        # Lookup dictionaries
        sc_dict = {s.id: s for s in scs}
        ecu_dict = {e.id: e for e in ecus}
        sensor_dict = {s.id: s for s in sensors}
        actuator_dict = {a.id: a for a in actuators}
        
        if not assignments:
            print("Hiçbir atama yapılmadı. Lütfen optimize fonksiyonunu çalıştırın.")
            return
        
        # Create reverse mapping: ECU -> list of SWs
        ecu_to_sws = {}
        for sw_id, ecu_id in assignments.items():
            if ecu_id not in ecu_to_sws:
                ecu_to_sws[ecu_id] = []
            ecu_to_sws[ecu_id].append(sw_id)
    
        # ECU'ları filtreле (sadece atanan)
        assigned_ecus = [e for e in ecus if e.id in ecu_to_sws]
        
        fig, ax = plt.subplots(figsize=(20, 14))
        
        # ECU rengini tip'e göre belirle
        ecu_colors = {
            'HPC': '#E8F4F8',
            'ZONE': '#FFF4E6',
            'MCU': '#F0E8F4'
        }
        
        # Interface renkleri
        interface_colors = {
            'CAN': '#FF6B6B',
            'ETH': '#4ECDC4',
            'LIN': '#FFE66D',
            'FLEXRAY': '#95E1D3',
        }
        
        # Sensor/actuator dışarıda node'lar
        sensor_positions = {}
        actuator_positions = {}
        
        # Toplam SW'lerin sensor/actuatorlarını topla
        all_sw_sensors = {}
        all_sw_actuators = {}
        for ecu_id, sw_ids in ecu_to_sws.items():
            for sw_id in sw_ids:
                sc = sc_dict[sw_id]
                if hasattr(sc, 'sensors'):
                    for sid in sc.sensors:
                        if sid not in all_sw_sensors:
                            all_sw_sensors[sid] = []
                        all_sw_sensors[sid].append(sw_id)
                if hasattr(sc, 'actuators'):
                    for aid in sc.actuators:
                        if aid not in all_sw_actuators:
                            all_sw_actuators[aid] = []
                        all_sw_actuators[aid].append(sw_id)
        
        # ECU kutularını ve SW'leri çiz
        ecu_box_height = 8
        ecu_y_spacing = 10  # ECU'lar arasındaki dikey mesafe
        ecu_positions = {}
        
        for idx, ecu in enumerate(assigned_ecus):
            y_pos = (len(assigned_ecus) - 1) * ecu_y_spacing - idx * ecu_y_spacing
            x_pos = 5
            
            sw_ids = ecu_to_sws[ecu.id]
            
            # CPU/RAM kullanımını hesapla
            total_cpu = sum(sc_dict[sw_id].cpu_req for sw_id in sw_ids)
            total_ram = sum(sc_dict[sw_id].ram_req for sw_id in sw_ids)
            
            cpu_util = 100 * total_cpu / ecu.cpu_cap if ecu.cpu_cap > 0 else 0
            ram_util = 100 * total_ram / ecu.ram_cap if ecu.ram_cap > 0 else 0
            
            # ECU kutusunun genişliği SW sayısına göre
            box_width = max(8, len(sw_ids) * 2.5)
            
            # ECU kutusu çiz
            color = ecu_colors.get(ecu.type, '#EEEEEE')
            fancy_box = FancyBboxPatch((x_pos, y_pos), box_width, ecu_box_height,
                                    boxstyle="round,pad=0.2", 
                                    edgecolor='black', facecolor=color,
                                    linewidth=2, alpha=0.8)
            ax.add_patch(fancy_box)
            
            # ECU başlığı
            header_text = f"{ecu.id} ({ecu.type})\n"
            header_text += f"HW: {', '.join(ecu.hw_offered)}\n"
            header_text += f"CPU: {total_cpu}/{ecu.cpu_cap} ({cpu_util:.0f}%)  |  RAM: {total_ram}/{ecu.ram_cap} ({ram_util:.0f}%)"
            
            ax.text(x_pos + box_width/2, y_pos + ecu_box_height - 0.8, header_text,
                fontsize=9, ha='center', va='top', weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # SW'leri kutu içine çiz
            sw_x_start = x_pos + 0.5
            sw_spacing = (box_width - 1) / max(len(sw_ids), 1)
            
            for sw_idx, sw_id in enumerate(sw_ids):
                sc = sc_dict[sw_id]
                sw_x = sw_x_start + sw_idx * sw_spacing + 0.5
                sw_y = y_pos + 3.5
                
                # SW kutusu
                sw_box = FancyBboxPatch((sw_x - 1, sw_y - 1), 2, 2,
                                        boxstyle="round,pad=0.1",
                                        edgecolor='darkblue', facecolor='lightblue',
                                        linewidth=1.5, alpha=0.9)
                ax.add_patch(sw_box)
                
                # SW adı ve HW gereksinimi
                sw_text = f"{sw_id}\nHW: {', '.join(sc.hw_required)}"
                ax.text(sw_x, sw_y + 0.3, sw_text,
                    fontsize=7, ha='center', va='center', weight='bold')
                
                # Sensor/actuator bağlantılarını çiz
                if hasattr(sc, 'sensors'):
                    for s_idx, sensor_id in enumerate(sc.sensors):
                        if sensor_id not in sensor_positions:
                            # Sensor'ü kutu dışında konumlandır
                            sensor_idx = len(sensor_positions)
                            sensor_positions[sensor_id] = (x_pos + box_width + 2, y_pos + ecu_box_height - 1 - sensor_idx * 1.2)
                        
                        sensor_x, sensor_y = sensor_positions[sensor_id]
                        
                        # Sensor noktası
                        sensor_obj = sensor_dict[sensor_id]
                        ax.plot(sensor_x, sensor_y, 'gs', markersize=8)
                        ax.text(sensor_x + 0.3, sensor_y, f"{sensor_id}({sensor_obj.type})",
                            fontsize=6, va='center')
                        
                        # Bağlantı çizgisi (interface renginde)
                        interface = sensor_obj.interface
                        color = interface_colors.get(interface, 'gray')
                        ax.plot([sw_x + 1, sensor_x], [sw_y, sensor_y],
                            color=color, linewidth=2, alpha=0.6, linestyle='--')
                
                if hasattr(sc, 'actuators'):
                    for a_idx, actuator_id in enumerate(sc.actuators):
                        if actuator_id not in actuator_positions:
                            # Actuator'ü kutu dışında konumlandır
                            actuator_idx = len(actuator_positions)
                            actuator_positions[actuator_id] = (x_pos + box_width + 2, y_pos - 0.5 - actuator_idx * 1.2)
                        
                        actuator_x, actuator_y = actuator_positions[actuator_id]
                        
                        # Actuator noktası
                        actuator_obj = actuator_dict[actuator_id]
                        ax.plot(actuator_x, actuator_y, '^', color='orange', markersize=8)
                        ax.text(actuator_x + 0.3, actuator_y, f"{actuator_id}({actuator_obj.type})",
                            fontsize=6, va='center')
                        
                        # Bağlantı çizgisi (interface renginde)
                        interface = actuator_obj.interface
                        color = interface_colors.get(interface, 'gray')
                        ax.plot([sw_x + 1, actuator_x], [sw_y, actuator_y],
                            color=color, linewidth=2, alpha=0.6, linestyle='--')
            
            ecu_positions[ecu.id] = (x_pos, y_pos, box_width, ecu_box_height)
        
        # Legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=ecu_colors['HPC'], ec='black', linewidth=2, label='HPC ECU'),
            plt.Rectangle((0, 0), 1, 1, fc=ecu_colors['ZONE'], ec='black', linewidth=2, label='ZONE ECU'),
            plt.Rectangle((0, 0), 1, 1, fc=ecu_colors['MCU'], ec='black', linewidth=2, label='MCU ECU'),
            Line2D([0], [0], marker='s', color='w', label='Sensor', 
                markerfacecolor='green', markersize=10),
            Line2D([0], [0], marker='^', color='w', label='Actuator', 
                markerfacecolor='orange', markersize=10),
            Line2D([0], [0], color='#FF6B6B', lw=2, linestyle='--', label='CAN'),
            Line2D([0], [0], color='#4ECDC4', lw=2, linestyle='--', label='ETH'),
            Line2D([0], [0], color='#FFE66D', lw=2, linestyle='--', label='LIN'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        
        ax.set_xlim(-2, max(ecu_positions[e.id][0] + ecu_positions[e.id][2] for e in assigned_ecus) + 5)
        ax.set_ylim(-5, len(assigned_ecus) * 10)
        ax.axis('off')
        
        plt.title('ECU Assignment Visualization: SW Components Inside ECU Containers', 
                fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_vehicle_layout_topdown(self, sensors, actuators, assignments=None, ecus=None, locations=None, scs=None, comm_matrix=None, cable_types=None, vehicle_length=4.5, vehicle_width=1.8, filename="vehicle_layout.png"):
        """
        Bird's eye view of vehicle layout showing sensors, actuators, and optionally ECUs.
        Displays the physical dimensions and locations of all components.
        
        When assignments and scs are provided:
        - Highlights active locations (those with assigned SCs)
        - Shows connections from active locations to their connected sensors/actuators
        - Colors connections by bus type (ETH, CAN, FLEXRAY, LIN)
        - Shows ECU-to-ECU backbone connections with different line styles
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Draw vehicle outline (front at top, rear at bottom)
        vehicle_rect = patches.Rectangle(
            (-vehicle_width/2, vehicle_length/2),      # Top-left (front-left in 2D view)
            vehicle_width,
            -vehicle_length,                            # Negative to draw downward
            linewidth=3,
            edgecolor='black',
            facecolor='lightgray',
            alpha=0.3,
            label='Vehicle Body'
        )
        ax.add_patch(vehicle_rect)
        
        # Add FRONT and REAR labels
        ax.text(0, vehicle_length/2 + 0.3, 'REAR', fontsize=14, weight='bold', ha='center', va='bottom',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax.text(0, -vehicle_length/2 - 0.1, 'FRONT', fontsize=14, weight='bold', ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Draw vehicle centerline
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Vehicle Centerline')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Color maps for bus/interface types
        interface_colors = {
            'ETH': '#3498DB',        # Blue
            'CAN': '#E74C3C',        # Red
            'FLEXRAY': '#2ECC71',    # Green
            'LIN': '#F39C12'         # Orange
        }
        
        # Color maps for sensor/actuator types
        sensor_colors = {
            'CAMERA': '#2ECC71',      # Green
            'LIDAR': '#E74C3C',       # Red
            'RADAR': '#3498DB',       # Blue
            'IMU': '#F39C12',         # Orange
            'GPS': '#9B59B6'          # Purple
        }
        
        actuator_colors = {
            'BRAKE': '#E74C3C',       # Red
            'STEERING': '#3498DB',    # Blue
            'MOTOR': '#2ECC71',       # Green
            'HVAC': '#F39C12',        # Orange
            'LIGHT': '#FFD700'        # Gold
        }
        
        # Helper function to offset labels to avoid overlap
        label_offsets = {}
        
        # Plot sensors 
        sensor_types_plotted = set()
        for idx, sensor in enumerate(sensors):
            y_pos = -sensor.location.y
            # Check if within vehicle bounds
            if abs(sensor.location.x) <= vehicle_width/2 and abs(y_pos) <= vehicle_length/2:
                color = sensor_colors.get(sensor.type, '#95A5A6')
                label = f'Sensor: {sensor.type}' if sensor.type not in sensor_types_plotted else ''
                ax.scatter(sensor.location.x, y_pos, s=250, c=color, 
                          marker='o', edgecolor='black', linewidth=2, zorder=5, label=label)
                
                # Offset labels to avoid overlap
                offset_x = 0.08 if idx % 2 == 0 else -0.08
                offset_y = 0.08 if idx % 3 == 0 else -0.08
                short_id = sensor.id.replace('CAM_', 'C_').replace('LIDAR_', 'L_').replace('_', '')
                ax.annotate(short_id, (sensor.location.x + offset_x, y_pos + offset_y), 
                           fontsize=7, ha='center', va='center', fontweight='bold', color='black',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
                if sensor.type not in sensor_types_plotted:
                    sensor_types_plotted.add(sensor.type)
        
        # Plot actuators 
        actuator_types_plotted = set()
        for idx, actuator in enumerate(actuators):
            y_pos = -actuator.location.y
            # Check if within vehicle bounds
            if abs(actuator.location.x) <= vehicle_width/2 and abs(y_pos) <= vehicle_length/2:
                color = actuator_colors.get(actuator.type, '#95A5A6')
                label = f'Actuator: {actuator.type}' if actuator.type not in actuator_types_plotted else ''
                ax.scatter(actuator.location.x, y_pos, s=250, c=color, 
                          marker='^', edgecolor='black', linewidth=2, zorder=5, label=label)
                
                # Offset labels to avoid overlap
                offset_x = 0.08 if idx % 2 == 0 else -0.08
                offset_y = 0.08 if idx % 3 == 1 else -0.08
                short_id = actuator.id.replace('ACT_', 'A_').replace('_', '')
                ax.annotate(short_id, (actuator.location.x + offset_x, y_pos + offset_y), 
                           fontsize=7, ha='center', va='center', fontweight='bold', color='black',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
                if actuator.type not in actuator_types_plotted:
                    actuator_types_plotted.add(actuator.type)
        
        # Plot Locations or ECUs (if provided)
        if locations is not None and assignments is None:
            # Mode 1: Show all candidate locations without assignments
            for loc in locations:
                y_pos = -loc.location.y

                label = 'Candidate Site' if loc.id == 'LOC0' else ""
                ax.scatter(loc.location.x, y_pos, s=350, c='lightgray',
                          marker='s', edgecolor='black', linewidth=2, zorder=4,
                          label=label)

                ax.text(loc.location.x, y_pos, loc.id,
                       fontsize=8, ha='center', va='center', fontweight='bold',
                       color='black', zorder=10)
        
        elif locations is not None and assignments is not None:
            # Mode 2: Show locations with assignments and connections
            # Create a mapping of location to its coordinates
            loc_dict = {loc.id: loc for loc in locations}
            
            # Find active locations and their assigned SCs
            active_locations = {}
            for sc_id, loc_id in assignments.items():
                if loc_id not in active_locations:
                    active_locations[loc_id] = []
                active_locations[loc_id].append(sc_id)
            
            # Build SC to sensors/actuators mapping if scs provided
            sc_to_sensors = {}
            sc_to_actuators = {}
            if scs is not None:
                for sc in scs:
                    if sc.sensors:
                        sc_to_sensors[sc.id] = sc.sensors
                    if sc.actuators:
                        sc_to_actuators[sc.id] = sc.actuators
            
            # Create sensor/actuator lookup dictionaries
            sensor_dict = {s.id: s for s in sensors}
            actuator_dict = {a.id: a for a in actuators}
            
            # Draw connections from active locations to their connected sensors/actuators
            interfaces_drawn = set()
            for loc_id, sc_ids in active_locations.items():
                if loc_id not in loc_dict:
                    continue
                
                loc = loc_dict[loc_id]
                loc_y = -loc.location.y
                
                # Collect all connected sensors and actuators for this location
                connected_sensors = set()
                connected_actuators = set()
                
                for sc_id in sc_ids:
                    if sc_id in sc_to_sensors:
                        connected_sensors.update(sc_to_sensors[sc_id])
                    if sc_id in sc_to_actuators:
                        connected_actuators.update(sc_to_actuators[sc_id])
                
                # Draw connections to sensors
                for sensor_id in connected_sensors:
                    if sensor_id in sensor_dict:
                        sensor = sensor_dict[sensor_id]
                        sensor_y = -sensor.location.y
                        
                        # Get interface color
                        interface = getattr(sensor, 'interface', 'CAN')
                        line_color = interface_colors.get(interface, '#95A5A6')
                        
                        # Draw line with interface type as linestyle
                        ax.plot([loc.location.x, sensor.location.x], [loc_y, sensor_y],
                               color=line_color, linewidth=2, alpha=0.6, zorder=3)
                        
                        # Add small dot at connection point on location
                        ax.scatter(loc.location.x, loc_y, s=50, c=line_color, 
                                  marker='o', zorder=6)
                
                # Draw connections to actuators
                for actuator_id in connected_actuators:
                    if actuator_id in actuator_dict:
                        actuator = actuator_dict[actuator_id]
                        actuator_y = -actuator.location.y
                        
                        # Get interface color
                        interface = getattr(actuator, 'interface', 'CAN')
                        line_color = interface_colors.get(interface, '#95A5A6')
                        
                        # Draw line with interface type as linestyle
                        ax.plot([loc.location.x, actuator.location.x], [loc_y, actuator_y],
                               color=line_color, linewidth=2, alpha=0.6, zorder=3)
                        
                        # Add small dot at connection point on location
                        ax.scatter(loc.location.x, loc_y, s=50, c=line_color, 
                                  marker='o', zorder=6)
            
            # Draw ECU-to-ECU backbone connections from comm_matrix
            if comm_matrix is not None and cable_types is not None and scs is not None:
                # Create SC to location mapping
                sc_to_loc = {sc_id: loc_id for sc_id, loc_id in assignments.items()}
                
                # Create cable_types lookup: interface_name -> interface object
                cable_dict = {name: iface for name, iface in cable_types.items()}
                
                # Track drawn connections to avoid duplicates
                drawn_connections = set()
                
                for comm_link in comm_matrix:
                    src_sc = comm_link.get('src')
                    dst_sc = comm_link.get('dst')
                    
                    if src_sc not in sc_to_loc or dst_sc not in sc_to_loc:
                        continue
                    
                    src_loc_id = sc_to_loc[src_sc]
                    dst_loc_id = sc_to_loc[dst_sc]
                    
                    # Skip if same location (not backbone)
                    if src_loc_id == dst_loc_id:
                        continue
                    
                    # Avoid duplicate connections
                    conn_key = tuple(sorted([src_loc_id, dst_loc_id]))
                    if conn_key in drawn_connections:
                        continue
                    drawn_connections.add(conn_key)
                    
                    # Get source and destination locations
                    if src_loc_id not in loc_dict or dst_loc_id not in loc_dict:
                        continue
                    
                    src_loc = loc_dict[src_loc_id]
                    dst_loc = loc_dict[dst_loc_id]
                    
                    src_y = -src_loc.location.y
                    dst_y = -dst_loc.location.y
                    
                    # Determine bus type from SC interface requirements
                    # Check both SCs to find compatible interface
                    src_sc_obj = next((s for s in scs if s.id == src_sc), None)
                    dst_sc_obj = next((s for s in scs if s.id == dst_sc), None)
                    
                    bus_type = 'CAN'  # Default
                    linestyle = '-'   # Default solid
                    linewidth = 2.5
                    
                    if src_sc_obj and src_sc_obj.interface_required:
                        bus_type = src_sc_obj.interface_required[0]
                    elif dst_sc_obj and dst_sc_obj.interface_required:
                        bus_type = dst_sc_obj.interface_required[0]
                    
                    # Set line style based on bus type
                    if bus_type == 'ETH':
                        linestyle = '--'    # Solid for Ethernet
                        linewidth = 3.5
                    elif bus_type == 'CAN':
                        linestyle = '--'   # Dashed for CAN
                        linewidth = 2.5
                    elif bus_type == 'FLEXRAY':
                        linestyle = '--'   # Dash-dot for FLEXRAY
                        linewidth = 2.5
                    elif bus_type == 'LIN':
                        linestyle = '--'    # Dotted for LIN
                        linewidth = 2
                    
                    line_color = interface_colors.get(bus_type, '#95A5A6')
                    
                    # Draw ECU-ECU backbone connection
                    ax.plot([src_loc.location.x, dst_loc.location.x], 
                           [src_y, dst_y],
                           color=line_color, linewidth=linewidth, linestyle=linestyle, 
                           alpha=0.7, zorder=2, label=f'ECU-ECU: {bus_type}' if bus_type not in [c.get('_label') for c in []] else '')
            
            # Plot all locations (active ones with highlight, inactive ones with less emphasis)
            for loc in locations:
                y_pos = -loc.location.y
                is_active = loc.id in active_locations
                
                if is_active:
                    # Active location: brighter, with border
                    ax.scatter(loc.location.x, y_pos, s=400, c='#FFE74C', 
                              marker='s', edgecolor='#FF6B35', linewidth=3, zorder=4,
                              label='Active Location' if loc.id == list(active_locations.keys())[0] else '')
                    
                    # Show number of assigned SCs
                    num_scs = len(active_locations[loc.id])
                    ax.text(loc.location.x, y_pos, f"{loc.id}\n({num_scs})",
                           fontsize=9, ha='center', va='center', fontweight='bold',
                           color='black', zorder=10)
                else:
                    # Inactive location: grayed out
                    ax.scatter(loc.location.x, y_pos, s=300, c='#D3D3D3',
                              marker='s', edgecolor='gray', linewidth=1.5, zorder=2, alpha=0.5,
                              label='Inactive Location' if loc == locations[0] else '')
                    
                    ax.text(loc.location.x, y_pos, loc.id,
                           fontsize=7, ha='center', va='center', fontweight='bold',
                           color='gray', zorder=5, alpha=0.5)
        
        elif ecus is not None:
            # Check mode: Candidate Sites (no assignments) or Assigned ECUs
            if assignments is None:
                # --- CANDIDATE SITES MODE ---
                unique_locations = []
                seen_coords = set()
                
                # Sort to ensure stable labeling
                sorted_ecus = sorted(ecus, key=lambda e: e.id)
                
                for ecu in sorted_ecus:
                    coord = (round(ecu.location.x, 2), round(ecu.location.y, 2))
                    if coord not in seen_coords:
                        seen_coords.add(coord)
                        unique_locations.append(ecu)
                
                for idx, ecu in enumerate(unique_locations, 1):
                    y_pos = -ecu.location.y
                    
                    label = 'Candidate Site' if idx == 1 else ""
                    ax.scatter(ecu.location.x, y_pos, s=350, c='lightgray', 
                              marker='s', edgecolor='black', linewidth=2, zorder=4, 
                              label=label)
                    
                    # Label as L1, L2... (ensure zorder is higher than scatter)
                    ax.text(ecu.location.x, y_pos, f"L{idx-1}", 
                           fontsize=8, ha='center', va='center', fontweight='bold', 
                           color='black', zorder=10)
            
            else:
                # --- ASSIGNED ECUS MODE ---
                ecu_type_colors = {
                    'HPC': '#FF6B6B',      # Red
                    'ZONE': "#050370",     # Cyan
                    'MCU': '#95E1D3'       # Mint green
                }
                
                assigned_ecu_ids = set(assignments.values())
                
                ecu_types_plotted = set()
                for ecu in ecus:
                    # Only show assigned ECUs
                    if ecu.id not in assigned_ecu_ids:
                        continue
                    
                    color = ecu_type_colors.get(ecu.type, '#FADBD8')
                    y_pos = -ecu.location.y
                    
                    label = f'ECU: {ecu.type}' if ecu.type not in ecu_types_plotted else ''
                    
                    ax.scatter(ecu.location.x, y_pos, s=200, c=color, 
                              marker='D', edgecolor='black', linewidth=2, zorder=4, 
                              alpha=0.7, label=label)
                    
                    # Label using the ID number from ECU_X
                    try:
                        short_id = ecu.id.split('_')[1]
                    except:
                        short_id = ecu.id
                        
                    ax.annotate(short_id, (ecu.location.x, y_pos), 
                               fontsize=8, ha='center', va='center', fontweight='bold', 
                               color='black')
                    
                    if ecu.type not in ecu_types_plotted:
                        ecu_types_plotted.add(ecu.type)
        
        # Add dimension annotations
        # Length annotation (vertical on the left)
        ax.annotate('', xy=(-vehicle_width/2 - 0.4, vehicle_length/2), xytext=(-vehicle_width/2 - 0.4, -vehicle_length/2),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=2.5))
        ax.text(-vehicle_width/2 - 0.7, 0, f'{vehicle_length}m', fontsize=12, weight='bold', 
               ha='right', va='center', rotation=90,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
        
        # Width annotation (horizontal at the bottom)
        ax.annotate('', xy=(vehicle_width/2, -vehicle_length/2 - 0.4), xytext=(-vehicle_width/2, -vehicle_length/2 - 0.4),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=2.5))
        ax.text(0, -vehicle_length/2 - 0.7, f'{vehicle_width}m', fontsize=12, weight='bold', 
               ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            patches.Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', linewidth=2, label='Vehicle Body'),
        ]
        
        if locations is not None and assignments is None:
            legend_elements.append(
                Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', markersize=12,
                       markeredgecolor='black', markeredgewidth=2, label='Candidate Site')
            )
        elif locations is not None and assignments is not None:
            # Add location legend for active/inactive locations
            legend_elements.extend([
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFE74C', markersize=12,
                       markeredgecolor='#FF6B35', markeredgewidth=2, label='Active Location'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#D3D3D3', markersize=12,
                       markeredgecolor='gray', markeredgewidth=1.5, alpha=0.5, label='Inactive Location'),
            ])
            # Add interface/bus type legend
            legend_elements.extend([
                Line2D([0], [0], color='#3498DB', linewidth=3, label='Bus: ETH (Ethernet)'),
                Line2D([0], [0], color='#E74C3C', linewidth=3, label='Bus: CAN'),
                Line2D([0], [0], color='#2ECC71', linewidth=3, label='Bus: FLEXRAY'),
                Line2D([0], [0], color='#F39C12', linewidth=3, label='Bus: LIN'),
            ])
            # Add ECU-ECU backbone legend
            legend_elements.extend([
                Line2D([0], [0], color='#3498DB', linewidth=3.5, linestyle='-', label='ECU-ECU: ETH (solid)'),
                Line2D([0], [0], color='#E74C3C', linewidth=2.5, linestyle='--', label='ECU-ECU: CAN (dashed)'),
                Line2D([0], [0], color='#2ECC71', linewidth=2.5, linestyle='-.', label='ECU-ECU: FLEXRAY (dash-dot)'),
                Line2D([0], [0], color='#F39C12', linewidth=2, linestyle=':', label='ECU-ECU: LIN (dotted)'),
            ])
        elif ecus is not None:
            if assignments is None:
                legend_elements.append(
                    Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', markersize=12,
                           markeredgecolor='black', markeredgewidth=2, label='Candidate Site')
                )
            else:
                legend_elements.extend([
                    Line2D([0], [0], marker='D', color='w', markerfacecolor='#FF6B6B', markersize=12,
                           markeredgecolor='black', markeredgewidth=2, label='ECU: HPC'),
                    Line2D([0], [0], marker='D', color='w', markerfacecolor='#050370', markersize=12,
                           markeredgecolor='black', markeredgewidth=2, label='ECU: ZONE'),
                    Line2D([0], [0], marker='D', color='w', markerfacecolor='#95E1D3', markersize=12,
                           markeredgecolor='black', markeredgewidth=2, label='ECU: MCU'),
                ])

        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71', markersize=12, markeredgecolor='black', markeredgewidth=2, label='CAMERA'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', markersize=12, markeredgecolor='black', markeredgewidth=2, label='LIDAR'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#F39C12', markersize=12, markeredgecolor='black', markeredgewidth=2, label='IMU'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#9B59B6', markersize=12, markeredgecolor='black', markeredgewidth=2, label='GPS'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#2ECC71', markersize=12, markeredgecolor='black', markeredgewidth=2, label='MOTOR'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#E74C3C', markersize=12, markeredgecolor='black', markeredgewidth=2, label='BRAKE'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#3498DB', markersize=12, markeredgecolor='black', markeredgewidth=2, label='STEERING'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#FFD700', markersize=12, markeredgecolor='black', markeredgewidth=2, label='LIGHT'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#F39C12', markersize=12, markeredgecolor='black', markeredgewidth=2, label='HVAC'),
        ])
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, ncol=2, framealpha=0.95, edgecolor='black', fancybox=True)
        
        # Set labels and title
        ax.set_xlabel('X (Left-Right) [meters]', fontsize=13, weight='bold')
        ax.set_ylabel('Y (Front-Back) [meters]', fontsize=13, weight='bold')
        
        title = "Vehicle Layout - Bird's Eye View (Top-Down)"
        if locations is not None and assignments is not None:
            active_count = len(set(assignments.values()))
            title += f" - {active_count} Active Locations"
        elif assignments:
            title += f" - {len(set(assignments.values()))} ECUs Assigned"
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Set limits dynamically based on assigned ECU positions if ECUs are shown
        if ecus is not None:
            shown_ecus = [e for e in ecus if not assignments or e.id in set(assignments.values())]
            if len(shown_ecus) > 0:
                ecu_x_coords = [e.location.x for e in shown_ecus]
                ecu_y_coords = [-e.location.y for e in shown_ecus]
                ecu_x_min, ecu_x_max = min(ecu_x_coords), max(ecu_x_coords)
                ecu_y_min, ecu_y_max = min(ecu_y_coords), max(ecu_y_coords)
                
                # Set limits with padding around ECU spread
                padding_x = max(1.5, abs(ecu_x_max - ecu_x_min) * 0.1)
                padding_y = max(1.0, abs(ecu_y_max - ecu_y_min) * 0.1)
                
                ax.set_xlim(min(ecu_x_min, -vehicle_width/2) - padding_x, max(ecu_x_max, vehicle_width/2) + padding_x)
                ax.set_ylim(min(ecu_y_min, -vehicle_length/2) - padding_y, max(ecu_y_max, vehicle_length/2) + padding_y)
            else:
                # Default limits if no assigned ECUs
                padding_x = 1.5
                padding_y = 1.0
                ax.set_xlim(-vehicle_width/2 - padding_x, vehicle_width/2 + padding_x)
                ax.set_ylim(-vehicle_length/2 - padding_y, vehicle_length/2 + padding_y)
        else:
            # Default limits if no ECUs
            padding_x = 1.5
            padding_y = 1.0
            ax.set_xlim(-vehicle_width/2 - padding_x, vehicle_width/2 + padding_x)
            ax.set_ylim(-vehicle_length/2 - padding_y, vehicle_length/2 + padding_y)
        
        ax.grid(True, alpha=0.4, linestyle=':', linewidth=0.8)
        ax.set_facecolor('#F8F9F9')
        
        plt.tight_layout()
        self.save_plot(filename)

    def display_solution_architecture(self, solution, scs, locations, filename="solution_architecture.png"):
        """
        Display the complete optimization solution architecture:
        - Each location as a container
        - Partitions within each location
        - SWs assigned to each partition
        - HW features and interfaces enabled at each location
        """
        sc_dict = {s.id: s for s in scs}
        location_dict = {l.id: l for l in locations}
        
        assignment = solution['assignment']  # {SC_id: location_id}
        partitions = solution['partitions']  # {SC_id: "LOC0_asil3_p0"}
        hw_features = solution['hw_features']  # ["HW_ACC@LOC0", ...]
        interfaces = solution['interfaces']  # ["ETH@LOC0", ...]
        
        # Group SCs by location and partition
        loc_partition_sws = {}  # {location_id: {partition_name: [SC_ids]}}
        loc_hw = {}  # {location_id: [hw_features]}
        loc_if = {}  # {location_id: [interfaces]}
        
        for sc_id, loc_id in assignment.items():
            if loc_id not in loc_partition_sws:
                loc_partition_sws[loc_id] = {}
            partition_name = partitions.get(sc_id, "unknown")
            if partition_name not in loc_partition_sws[loc_id]:
                loc_partition_sws[loc_id][partition_name] = []
            loc_partition_sws[loc_id][partition_name].append(sc_id)
        
        # Group HW features by location
        for hw_feat in hw_features:
            hw_name, loc_id = hw_feat.rsplit('@', 1)
            if loc_id not in loc_hw:
                loc_hw[loc_id] = []
            loc_hw[loc_id].append(hw_name)
        
        # Group interfaces by location
        for iface in interfaces:
            iface_name, loc_id = iface.rsplit('@', 1)
            if loc_id not in loc_if:
                loc_if[loc_id] = []
            loc_if[loc_id].append(iface_name)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(18, 12))
        
        # Color schemes
        asil_colors = {
            'ASIL-A': '#FFE5E5',
            'ASIL-B': '#FFD1D1',
            'ASIL-C': '#FFB3B3',
            'ASIL-D': '#FF9999',
            'QM': '#E8F4E8',
        }
        
        # Render locations and their contents
        num_locs = len(loc_partition_sws)
        loc_width = 12  # Width of each location box
        loc_height = 8  # Height of each location box
        spacing_x = 14
        spacing_y = 10
        
        start_x = 2
        start_y = (num_locs - 1) * spacing_y
        
        for loc_idx, (loc_id, partitions_dict) in enumerate(sorted(loc_partition_sws.items())):
            x_pos = start_x
            y_pos = start_y - loc_idx * spacing_y
            
            # Location box
            loc_box = FancyBboxPatch((x_pos, y_pos), loc_width, loc_height,
                                      boxstyle="round,pad=0.3",
                                      edgecolor='darkblue', facecolor='#E8F8FF',
                                      linewidth=3, alpha=0.9)
            ax.add_patch(loc_box)
            
            # Location title
            loc = location_dict.get(loc_id)
            title_text = f"📍 {loc_id}"
            ax.text(x_pos + loc_width/2, y_pos + loc_height - 0.6, title_text,
                   fontsize=12, ha='center', va='top', weight='bold')
            
            # HW Features section
            hw_text = f"Hardware: {', '.join(loc_hw.get(loc_id, ['None']))}"
            ax.text(x_pos + 0.3, y_pos + loc_height - 1.2, hw_text,
                   fontsize=8, ha='left', va='top', style='italic',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
            
            # Interfaces section
            if_text = f"Interfaces: {', '.join(loc_if.get(loc_id, ['None']))}"
            ax.text(x_pos + 0.3, y_pos + loc_height - 1.8, if_text,
                   fontsize=8, ha='left', va='top', style='italic',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.6))
            
            # Render partitions within location
            partition_y = y_pos + 3.8
            partition_height = 2.8
            partition_width = (loc_width - 0.6) / max(len(partitions_dict), 1)
            
            for part_idx, (partition_name, sc_ids) in enumerate(sorted(partitions_dict.items())):
                part_x = x_pos + 0.3 + part_idx * partition_width
                
                # Extract ASIL from partition name (e.g., "LOC0_asil3_p0")
                asil_level = 'ASIL-' + partition_name.split('asil')[1][0].upper() if 'asil' in partition_name else 'QM'
                asil_color = asil_colors.get(asil_level, '#E8E8E8')
                
                # Partition box
                part_box = FancyBboxPatch((part_x, partition_y - partition_height), 
                                         partition_width - 0.1, partition_height,
                                         boxstyle="round,pad=0.1",
                                         edgecolor='darkred', facecolor=asil_color,
                                         linewidth=2, alpha=0.85)
                ax.add_patch(part_box)
                
                # Partition header
                part_header = f"{asil_level}\n{partition_name.split('_')[-1]}"
                ax.text(part_x + partition_width/2 - 0.05, partition_y - 0.3, part_header,
                       fontsize=7, ha='center', va='top', weight='bold')
                
                # SWs in partition
                sw_list_text = "\n".join(sc_ids)
                ax.text(part_x + partition_width/2 - 0.05, partition_y - 1.0, sw_list_text,
                       fontsize=6, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Legend
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', label='ASIL-D', 
                   markerfacecolor='#FF9999', markersize=8),
            Line2D([0], [0], marker='s', color='w', label='ASIL-C', 
                   markerfacecolor='#FFB3B3', markersize=8),
            Line2D([0], [0], marker='s', color='w', label='ASIL-B', 
                   markerfacecolor='#FFD1D1', markersize=8),
            Line2D([0], [0], marker='s', color='w', label='ASIL-A', 
                   markerfacecolor='#FFE5E5', markersize=8),
            Line2D([0], [0], marker='s', color='w', label='QM', 
                   markerfacecolor='#E8F4E8', markersize=8),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Title
        ax.text(0.5, 0.98, 'LEGO Optimization Solution Architecture',
               transform=ax.transAxes, fontsize=14, ha='center', va='top',
               weight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax.set_xlim(-1, 20)
        ax.set_ylim(-2, (num_locs + 1) * spacing_y)
        ax.axis('off')
        
        plt.tight_layout()
        self.save_plot(filename)