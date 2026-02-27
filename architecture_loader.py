import json
import math
import re
from models import Point, Location, Sensor, Actuator, SoftwareComponent


def _generate_locations(num_locs, vehicle_dimensions):
    ecu_bounds = vehicle_dimensions.get('ecu_placement_bounds', {})
    x_min = ecu_bounds.get('x_min', -0.5)
    x_max = ecu_bounds.get('x_max', 0.5)
    y_min = ecu_bounds.get('y_min', -1.0)
    y_max = ecu_bounds.get('y_max', 1.0)

    cols = int(math.sqrt(num_locs))
    rows = (num_locs + cols - 1) // cols

    x_step = (x_max - x_min) / max(1, cols - 1) if cols > 1 else 0
    y_step = (y_max - y_min) / max(1, rows - 1) if rows > 1 else 0

    locations = []
    loc_idx = 0
    for r in range(rows):
        for c in range(cols):
            if loc_idx >= num_locs:
                break
            x_pos = x_min + c * x_step
            y_pos = y_max - r * y_step
            locations.append(
                Location(
                    id=f"LOC{loc_idx}",
                    location=Point(x_pos, y_pos),
                    health_factor=0.85,
                )
            )
            loc_idx += 1
    return locations


def _extract_device_id_from_driver_sc(sc_id):
    match = re.match(r"^SC_\d+_Driver_(.+)$", sc_id)
    if not match:
        return None
    return match.group(1)


def load_architecture_from_json(json_filepath, config_reader, num_locs=6):
    with open(json_filepath, 'r') as f:
        architecture = json.load(f)

    components = architecture.get('software_components', [])

    cable_types = config_reader.get_interfaces()
    vehicle_dimensions = config_reader.get_vehicle_dimensions()
    locations = _generate_locations(num_locs, vehicle_dimensions)

    hw = vehicle_dimensions['width'] / 2
    hl = vehicle_dimensions['length'] / 2

    sensors = []
    sensor_by_id = {}
    for item in config_reader.get_sensors_config():
        s = Sensor(
            id=item['id'],
            type=item['type'],
            interface=item['interface'],
            volume=float(item.get('volume', 1.0)),
            location=Point(item['location']['x_ratio'] * hw, item['location']['y_ratio'] * hl),
            max_latency=item.get('max_latency', None),
        )
        sensors.append(s)
        sensor_by_id[s.id] = s

    actuators = []
    actuator_by_id = {}
    for item in config_reader.get_actuators_config():
        a = Actuator(
            id=item['id'],
            type=item['type'],
            interface=item['interface'],
            volume=float(item.get('volume', 1.0)),
            location=Point(item['location']['x_ratio'] * hw, item['location']['y_ratio'] * hl),
            max_latency=item.get('max_latency', None),
        )
        actuators.append(a)
        actuator_by_id[a.id] = a

    scs = []
    sc_id_to_idx = {}

    for comp in components:
        sc = SoftwareComponent(
            id=comp['id'],
            domain=comp.get('domain', 'Unknown'),
            cpu_req=int(comp.get('cpu_req', 0)),
            ram_req=int(comp.get('ram_req', 0)),
            rom_req=int(comp.get('rom_req', 0)),
            asil_req=int(comp.get('asil_req', 0)),
            hw_required=list(comp.get('hw_required', [])),
        )

        sc_type = comp.get('type', '')
        if sc_type == 'sensor':
            dev_id = _extract_device_id_from_driver_sc(sc.id)
            if dev_id and dev_id in sensor_by_id:
                sc.sensors = [dev_id]
                sc.interface_required = [sensor_by_id[dev_id].interface]
        elif sc_type == 'actuator':
            dev_id = _extract_device_id_from_driver_sc(sc.id)
            if dev_id and dev_id in actuator_by_id:
                sc.actuators = [dev_id]
                sc.interface_required = [actuator_by_id[dev_id].interface]

        sc_id_to_idx[sc.id] = len(scs)
        scs.append(sc)

    comm_matrix = []
    for comp in components:
        dst = comp['id']
        for src in comp.get('receives_from', []):
            if src not in sc_id_to_idx or dst not in sc_id_to_idx:
                continue

            src_sc = scs[sc_id_to_idx[src]]
            volume = 10.0
            if src_sc.sensors:
                sensor_id = src_sc.sensors[0]
                sensor_obj = sensor_by_id.get(sensor_id)
                if sensor_obj is not None:
                    volume = float(sensor_obj.volume)
            elif src_sc.actuators:
                actuator_id = src_sc.actuators[0]
                actuator_obj = actuator_by_id.get(actuator_id)
                if actuator_obj is not None:
                    volume = float(actuator_obj.volume)

            comm_matrix.append({
                'src': src,
                'dst': dst,
                'volume': volume,
                'max_latency': 1.0,
            })

    return scs, comm_matrix, sensors, actuators, cable_types, locations
