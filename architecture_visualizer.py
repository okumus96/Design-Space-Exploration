import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re
import textwrap


def _build_compact_layered_positions(graph, layer_attr="layer", x_gap=2.1, y_gap=1.25):
    layers = {}
    for node, attrs in graph.nodes(data=True):
        layer = attrs.get(layer_attr, 0)
        layers.setdefault(layer, []).append(node)

    for layer_nodes in layers.values():
        layer_nodes.sort()

    positions = {}
    for layer in sorted(layers):
        layer_nodes = layers[layer]
        node_count = len(layer_nodes)
        center_offset = (node_count - 1) / 2.0
        for index, node in enumerate(layer_nodes):
            x = layer * x_gap
            y = (center_offset - index) * y_gap
            positions[node] = (x, y)
    return positions


def _shorten_label(node_id):
    short_name = (
        node_id.replace("SC_", "")
        .replace("Driver_", "")
        .replace("AW_", "")
        .replace("BC_", "")
        .replace("INFO_", "")
        .replace("CONN_", "")
    )
    short_name = re.sub(r"^\d+_", "", short_name)
    abbreviation_map = {
        "Localization": "Loc",
        "Perception": "Perc",
        "Planning": "Plan",
        "Control": "Ctrl",
        "Connectivity": "Conn",
        "Navigation": "Nav",
        "Detection": "Detect",
        "Recognition": "Recog",
        "Preprocessor": "Preproc",
        "Actuator": "Act",
        "Sensor": "Sens",
    }
    for source, target in abbreviation_map.items():
        short_name = short_name.replace(source, target)

    wrapped = textwrap.wrap(short_name, width=11, break_long_words=False)
    if len(wrapped) > 3:
        wrapped = wrapped[:3]
        wrapped[-1] = textwrap.shorten(wrapped[-1], width=11, placeholder="…")
    return "\n".join(wrapped)


def _label_prefix(node_type):
    prefix_map = {
        "sensor": "S",
        "actuator": "A",
        "drivetrain": "DT",
        "body_comfort": "BC",
        "infotainment": "I",
        "connectivity": "C",
        "autoware": "AW",
    }
    return prefix_map.get(node_type, "N")


def _resolve_node_style(node_id, node_type, layer_map):
    if node_type == "sensor":
        return '#A9CCE3', layer_map["sensor"]
    if node_type == "actuator":
        return '#F5B7B1', layer_map["actuator"]
    if node_type == "body_comfort":
        return '#A3E4D7', layer_map["control_and_apps"]
    if node_type == "drivetrain":
        return '#73C6B6', layer_map["drivetrain"]
    if node_type == "infotainment":
        layer = layer_map["telematics_media"]
        if "Navigation" in node_id:
            layer = layer_map["control_and_apps"]
        return '#D7BDE2', layer
    if node_type == "connectivity":
        layer = layer_map["telematics_media"]
        if "V2X" in node_id:
            layer = layer_map["fusion"]
        return '#F9E79F', layer
    if node_type == "autoware":
        if "Map" in node_id:
            return '#D5D8DC', layer_map["sensor"]
        if "Sensing" in node_id:
            return '#D2B4DE', layer_map["preprocessor"]
        if "Localization" in node_id or "Perception" in node_id:
            layer = layer_map["perception_loc"]
            if "Fusion" in node_id or "Grid" in node_id:
                layer = layer_map["fusion"]
            return '#82E0AA', layer
        if "Planning" in node_id:
            return '#F1C40F', layer_map["planning"]
        if "Control" in node_id or "System" in node_id:
            return '#E67E22', layer_map["control_and_apps"]
        if "Vehicle" in node_id:
            return '#CB4335', layer_map["control_and_apps"] + 0.5
    return 'white', 5


def _ensure_default_map_nodes(components):
    has_map_nodes = any(
        sc.get("type") == "autoware" and "Map" in sc.get("id", "")
        for sc in components
    )
    if has_map_nodes:
        return components

    defaults = [
        {
            "id": "AW_Map_PointCloud_Loader",
            "domain": "ADAS",
            "type": "autoware",
            "cpu_req": 1000,
            "ram_req": 4096,
            "rom_req": 8192,
            "asil_req": 2,
            "hw_required": [],
            "receives_from": []
        },
        {
            "id": "AW_Map_Lanelet2_Loader",
            "domain": "ADAS",
            "type": "autoware",
            "cpu_req": 800,
            "ram_req": 1024,
            "rom_req": 2048,
            "asil_req": 2,
            "hw_required": [],
            "receives_from": []
        },
    ]
    print("Map node bulunamadı, default map node'ları eklendi.")
    return components + defaults


def draw_complete_architecture_dag(json_filepath="full_architecture_v2.json", label_mode="code", add_default_map_if_missing=False):
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    
    components = data['software_components']
    if add_default_map_if_missing:
        components = _ensure_default_map_nodes(components)
    print(f"Toplam SW component sayısı: {len(components)}")
    G = nx.DiGraph()
    
    # Katmanları (Layer) belirleme (Soldan sağa akış)
    layer_map = {
        "sensor": 0,
        "preprocessor": 1,
        "perception_loc": 2,
        "fusion": 3,
        "planning": 4,
        "control_and_apps": 5, # Body, Info ve ADAS Kontrolü aynı hizada
        "telematics_media": 6,
        "drivetrain": 6.5,
        "actuator": 7
    }
    
    for sc in components:
        node_id = sc['id']
        node_type = sc.get('type', '')
        
        color, layer = _resolve_node_style(node_id, node_type, layer_map)
                
        G.add_node(node_id, color=color, layer=layer, node_type=node_type)
        
        for src in sc.get('receives_from', []):
            G.add_edge(src, node_id)
            
    # Daha kompakt ve kontrollü hiyerarşik çizim düzeni
    pos = _build_compact_layered_positions(G, layer_attr="layer", x_gap=2.1, y_gap=1.25)
    
    # İsimleri temizle/kısalt (Grafikte net okunabilmesi için)
    labels = {}
    label_map_rows = []
    prefix_counter = {}
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type", "")
        prefix = _label_prefix(node_type)
        prefix_counter[prefix] = prefix_counter.get(prefix, 0) + 1
        compact_code = f"{prefix}{prefix_counter[prefix]:02d}"
        short_label = _shorten_label(node)
        if label_mode == "code":
            labels[node] = compact_code
        elif label_mode == "short":
            labels[node] = short_label
        else:
            labels[node] = node
        label_map_rows.append((compact_code, short_label.replace("\n", " "), node))

    colors = [nx.get_node_attributes(G, 'color').get(node, 'white') for node in G.nodes()]

    # Çizim Ayarları (LaTeX'e uygun, daha okunur ve daha az whitespace)
    fig, ax = plt.subplots(figsize=(15, 8.5), constrained_layout=True)
    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_color=colors,
        node_size=1000,
        font_size=7,
        font_weight="bold",
        arrows=True,
        arrowsize=14,
        edge_color="#666666",
        width=1.2,
        linewidths=1.2,
        edgecolors="black",
        connectionstyle="arc3,rad=0.05",
        ax=ax,
    )

    legend_items = [
        Patch(facecolor='#A9CCE3', edgecolor='black', label='Sensor'),
        Patch(facecolor='#D5D8DC', edgecolor='black', label='Autoware Map'),
        Patch(facecolor='#F5B7B1', edgecolor='black', label='Actuator'),
        Patch(facecolor='#73C6B6', edgecolor='black', label='DriveTrain'),
        Patch(facecolor='#A3E4D7', edgecolor='black', label='Body Comfort'),
        Patch(facecolor='#D7BDE2', edgecolor='black', label='Infotainment'),
        Patch(facecolor='#F9E79F', edgecolor='black', label='Connectivity'),
        Patch(facecolor='#D2B4DE', edgecolor='black', label='Autoware Sensing/Preprocessor'),
        Patch(facecolor='#82E0AA', edgecolor='black', label='Autoware Perception/Localization/Fusion/Grid'),
        Patch(facecolor='#F1C40F', edgecolor='black', label='Autoware Planning'),
        Patch(facecolor='#E67E22', edgecolor='black', label='Autoware Control/System'),
        Patch(facecolor='#CB4335', edgecolor='black', label='Autoware Vehicle Cmd Gate'),
    ]
    fig.legend(
        handles=legend_items,
        title="Node Types",
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=True
    )
    
    ax.set_title("Whole-Vehicle Software Architecture DAG", fontsize=14, fontweight='bold', pad=8)
    ax.margins(x=0.02, y=0.04)
    ax.set_axis_off()
    
    fig.savefig("whole_vehicle_dag.png", dpi=400, bbox_inches='tight', pad_inches=0.05)
    fig.savefig("whole_vehicle_dag.pdf", bbox_inches='tight', pad_inches=0.05)
    if label_mode == "code":
        with open("whole_vehicle_dag_label_map.txt", "w", encoding="utf-8") as mapping_file:
            mapping_file.write("Code\tShort Name\tOriginal Node ID\n")
            for compact_code, short_name, original_id in sorted(label_map_rows, key=lambda item: item[0]):
                mapping_file.write(f"{compact_code}\t{short_name}\t{original_id}\n")
    print("Grafik 'whole_vehicle_dag.png' ve 'whole_vehicle_dag.pdf' olarak kaydedildi!")


def _parse_args():
    parser = argparse.ArgumentParser(description="Draw complete architecture DAG from a JSON file.")
    parser.add_argument(
        "--json",
        dest="json_filepath",
        default="full_architecture_v2.json",
        help="Path to architecture JSON file (default: full_architecture_v2.json)",
    )
    parser.add_argument(
        "--label-mode",
        choices=["code", "short", "full"],
        default="code",
        help="Label rendering mode: code (compact), short (abbreviated), full (original node id)",
    )
    parser.add_argument(
        "--add-default-map-if-missing",
        action="store_true",
        help="Automatically add default Autoware map nodes if they are missing in the input JSON",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    draw_complete_architecture_dag(
        json_filepath=arguments.json_filepath,
        label_mode=arguments.label_mode,
        add_default_map_if_missing=arguments.add_default_map_if_missing,
    )
    