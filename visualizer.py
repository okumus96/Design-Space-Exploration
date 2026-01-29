import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from tabulate import tabulate


class Visualization:
    def __init__(self):
        pass
    def display_data_summary(self, ecus, scs, sensors, actuators, cable_types, comm_matrix):
        print(f"\n Generated Data Summary:")
        print(f"   - ECUs: {len(ecus)}")
        print(f"   - Software Components: {len(scs)}")
        print(f"   - Sensors: {len(sensors)}")
        print(f"   - Actuators: {len(actuators)}")
        print(f"   - Communication Links: {len(comm_matrix)}")
        print(f"   - Bus Types: {len(cable_types)}")

    def display_assignments(self,assignments):
            # Summary by ECU
            print(f"\n   SWs assigned per ECU:")
            ecu_assignments = {}
            for sc_id, ecu_id in assignments.items():
                if ecu_id not in ecu_assignments:
                    ecu_assignments[ecu_id] = []
                ecu_assignments[ecu_id].append(sc_id)
            
            for ecu_id in sorted(ecu_assignments.keys()):
                assigned_scs = ecu_assignments[ecu_id]
                sc_list = ", ".join(assigned_scs)
                print(f"      - {ecu_id}: {len(assigned_scs)} SWs → [{sc_list}]")

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
        resource_cols = ['id', 'domain', 'cpu_req', 'ram_req', 'rom_req', 'asil_req', 'hw_required']
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

    def display_data(self,df_sensors, df_actuators, df_sc, df_ecu):

        # Display summary of generated Sensors and Actuators
        self.display_sensors(df_sensors)
        self.display_actuators(df_actuators)
        # Display summary of generated Software Components (SWs)
        self.display_scs(df_sc)

        # Display summary of generated ECUs
        self.display_ECUs(df_ecu)
    
    def display_assignments(self,solution_idx, solution,scs, ecus):
        assignments = solution['assignment']
        print("\n" + "-" * 80)
        print(f"SOLUTION {solution_idx}")
        print(f"  Hardware Cost: ${solution['hardware_cost']:.2f}")
        print(f"  Cable Length: {solution['cable_length']:.2f}m")
        print(f"  Total Cost: ${solution['hardware_cost']:.2f}")
        print(f"  ECUs Used: {solution['num_ecus_used']}")
        print("-" * 80)
        
        print(f"\nAssignment Summary:")
        print(f"   - Total SWs Assigned: {len(assignments)} / {len(scs)}")

    def plot_charts(self, sc_list, ecu_list, sensor_list, actuator_list):
        """Convert lists to DataFrames and plot charts"""
        # Convert lists to DataFrames if not already
        if isinstance(sc_list, list):
            df_sc = pd.DataFrame([vars(s) for s in sc_list])
        else:
            df_sc = sc_list
            
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

        # ECU Distribution
        sns.countplot(x='type', data=df_ecu, ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('Distribution of Candidate ECU Types')

        #  SWC Distribution in terms of Domain
        df_sc['type_guess'] = df_sc['id'].apply(lambda x: x.split('_')[-1])
        sns.countplot(x='type_guess', data=df_sc, ax=axes[0, 1], palette='magma')
        axes[0, 1].set_title('Distribution of SWC Workloads')

        # Sensor Type Distribution
        sns.countplot(x='type', data=df_sensors, ax=axes[1, 0], palette='Set2')
        axes[1, 0].set_title('Distribution of Sensor Types')
        axes[1, 0].set_xlabel('Sensor Type')
        axes[1, 0].set_ylabel('Count')

        # Actuator Type Distribution
        sns.countplot(x='type', data=df_actuators, ax=axes[1, 1], palette='Set1')
        axes[1, 1].set_title('Distribution of Actuator Types')
        axes[1, 1].set_xlabel('Actuator Type')
        axes[1, 1].set_ylabel('Count')

        plt.tight_layout()
        plt.show()

    def plot_sw_sensor_actuator_graph_final(self, scs, sensors, actuators, comm_matrix):
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
        plt.show()

    def visualize_pareto_front(self, pareto_solutions):
        """
        Create a visualization of the Pareto front showing trade-offs.
        
        Args:
            pareto_solutions: List of solutions from optimize_pareto_epsilon_constraint
        """
        if not pareto_solutions:
            print("No solutions to visualize")
            return
        
        # Extract data
        ecus = [sol['num_ecus_used'] for sol in pareto_solutions]
        costs = [sol['hardware_cost'] for sol in pareto_solutions]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Pareto Front (ECU vs Cost)
        ax1 = axes[0]
        ax1.scatter(ecus, costs, s=200, alpha=0.6, c='blue', edgecolors='black', linewidth=2)
        
        # Add solution numbers
        for i, (ecu, cost) in enumerate(zip(ecus, costs), 1):
            ax1.annotate(f'S{i}', (ecu, cost), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
        
        # Add line connecting solutions
        sorted_indices = sorted(range(len(ecus)), key=lambda i: ecus[i])
        sorted_ecus = [ecus[i] for i in sorted_indices]
        sorted_costs = [costs[i] for i in sorted_indices]
        ax1.plot(sorted_ecus, sorted_costs, 'b--', alpha=0.3, linewidth=1)
        
        ax1.set_xlabel('Number of ECUs Used', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
        ax1.set_title('Pareto Front: Cost vs ECU Count', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(int(min(ecus)), int(max(ecus))+1))
        
        # Plot 2: Efficiency (SCs per ECU)
        ax2 = axes[1]
        efficiency = [len(sol['assignment']) / sol['num_ecus_used'] for sol in pareto_solutions]
        cost_range = max(costs) - min(costs) if max(costs) != min(costs) else 1
        colors = plt.cm.RdYlGn([(c - min(costs)) / cost_range for c in costs])
        
        bars = ax2.bar(range(1, len(pareto_solutions)+1), efficiency, color=colors, 
                       edgecolor='black', linewidth=1.5, alpha=0.7)
        
        ax2.set_xlabel('Solution Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('SCs per ECU (Efficiency)', fontsize=12, fontweight='bold')
        ax2.set_title('Solution Efficiency Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(1, len(pareto_solutions)+1))
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, eff) in enumerate(zip(bars, efficiency)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{eff:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        #plt.savefig('pareto_front_analysis.png', dpi=300, bbox_inches='tight')
        #print("\n✓ Saved: pareto_front_analysis.png")
        plt.show()

    def visualize_optimization_result(self,scs, ecus, sensors, actuators, assignments):
        """
        ECU'ları kutular olarak görselleştir, SW'leri içinde göster
        """
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        from matplotlib.patches import ConnectionPatch
        
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
        plt.show()

    def plot_vehicle_layout_topdown(self, sensors, actuators, assignments=None, ecus=None, vehicle_length=4.5, vehicle_width=1.8):
        """
        Bird's eye view of vehicle layout showing sensors, actuators, and optionally ECUs.
        Displays the physical dimensions and locations of all components.
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
        
        # Plot ECUs (if provided, show only assigned ones)
        if ecus is not None:
            # ECU tiplerine göre farklı renkler
            ecu_type_colors = {
                'HPC': '#FF6B6B',      # Kırmızı
                'ZONE': "#050370",     # Cyan
                'MCU': '#95E1D3'       # Mint green
            }
            
            # Filter: only show ECUs that are in assignments
            assigned_ecu_ids = set()
            if assignments:
                assigned_ecu_ids = set(assignments.values())
            
            ecu_types_plotted = set()
            for ecu in ecus:
                # Skip ECUs that are not assigned (if assignments provided)
                if assignments and ecu.id not in assigned_ecu_ids:
                    continue
                
                color = ecu_type_colors.get(ecu.type, '#FADBD8')
                y_pos = -ecu.location.y
                
                label = f'ECU: {ecu.type}' if ecu.type not in ecu_types_plotted else ''
                
                ax.scatter(ecu.location.x, y_pos, s=200, c=color, 
                          marker='D', edgecolor='black', linewidth=2, zorder=4, 
                          alpha=0.7, label=label)
                
                # Label next to the marker (not on top)
                short_id = ecu.id.split('_')[1]
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
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#FF6B6B', markersize=12, 
                   markeredgecolor='black', markeredgewidth=2, label='ECU: HPC'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#050370', markersize=12, 
                   markeredgecolor='black', markeredgewidth=2, label='ECU: ZONE'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#95E1D3', markersize=12, 
                   markeredgecolor='black', markeredgewidth=2, label='ECU: MCU'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71', markersize=12, markeredgecolor='black', markeredgewidth=2, label='CAMERA'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', markersize=12, markeredgecolor='black', markeredgewidth=2, label='LIDAR'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#F39C12', markersize=12, markeredgecolor='black', markeredgewidth=2, label='IMU'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#9B59B6', markersize=12, markeredgecolor='black', markeredgewidth=2, label='GPS'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#2ECC71', markersize=12, markeredgecolor='black', markeredgewidth=2, label='MOTOR'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#E74C3C', markersize=12, markeredgecolor='black', markeredgewidth=2, label='BRAKE'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#3498DB', markersize=12, markeredgecolor='black', markeredgewidth=2, label='STEERING'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#FFD700', markersize=12, markeredgecolor='black', markeredgewidth=2, label='LIGHT'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#F39C12', markersize=12, markeredgecolor='black', markeredgewidth=2, label='HVAC'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, ncol=2, framealpha=0.95, edgecolor='black', fancybox=True)
        
        # Set labels and title
        ax.set_xlabel('X (Left-Right) [meters]', fontsize=13, weight='bold')
        ax.set_ylabel('Y (Front-Back) [meters]', fontsize=13, weight='bold')
        
        title = "Vehicle Layout - Bird's Eye View (Top-Down)"
        if assignments:
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
        plt.show()