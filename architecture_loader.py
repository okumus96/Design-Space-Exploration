import json
import re
from models import Point, Location, Sensor, Actuator, SoftwareComponent
from optimizer_utils import set_distance_table


def _load_locations(config_reader, num_locs=None):
    """Candidate ECU sites from configs/sites.json; num_locs keeps an evenly spread subset (None = all)."""
    sites = config_reader.get_sites()
    if num_locs is not None and 0 < num_locs < len(sites):
        step = len(sites) / float(num_locs)
        sites = [sites[int(i * step)] for i in range(num_locs)]
        print(f"  Using {num_locs} of the candidate sites")
    return [
        Location(
            id=s['id'],
            location=Point(s['position']['x'], s['position']['y'], s['position']['z'], ref=s['id']),
            health_factor=float(s.get('derating', 1.0)),
            zone=s.get('zone'),
            thermal_class=s.get('thermal_class'),
        )
        for s in sites
    ]


def _position(item):
    p = item['position']
    return Point(p['x'], p['y'], p['z'], ref=item['id'])


def _extract_device_id_from_driver_sc(sc_id):
    match = re.match(r"^SC_\d+_Driver_(.+)$", sc_id)
    if not match:
        return None
    return match.group(1)


def load_architecture_from_json(json_filepath, config_reader, num_locs=None):
    with open(json_filepath, 'r') as f:
        architecture = json.load(f)

    components = architecture.get('software_components', [])

    cable_types = config_reader.get_interfaces()
    locations = _load_locations(config_reader, num_locs)
    table = config_reader.get_distance_table()
    if table is not None:
        set_distance_table(*table)
        print(f"  Cable lengths from the channel graph ({len(table[0])} entities)")
    else:
        print("  WARNING: configs/distances.json not found, using Manhattan distances")

    sensors = []
    sensor_by_id = {}
    for item in config_reader.get_sensors_config():
        s = Sensor(
            id=item['id'],
            type=item['type'],
            interface=item['interface'],
            volume=float(item.get('volume', 1.0)),
            location=_position(item),
            rate_hz=float(item.get('rate_hz', 10.0)),
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
            location=_position(item),
            rate_hz=float(item.get('rate_hz', 10.0)),
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
            redundant_with=comp.get('redundant_with', None),
            rate_hz=float(comp.get('rate_hz', 10.0)),
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
            volume, rate_hz = 10.0, float(src_sc.rate_hz or 10.0)
            if src_sc.sensors:
                sensor_obj = sensor_by_id.get(src_sc.sensors[0])
                if sensor_obj is not None:
                    volume, rate_hz = float(sensor_obj.volume), float(sensor_obj.rate_hz or 10.0)
            elif src_sc.actuators:
                actuator_obj = actuator_by_id.get(src_sc.actuators[0])
                if actuator_obj is not None:
                    volume, rate_hz = float(actuator_obj.volume), float(actuator_obj.rate_hz or 10.0)

            comm_matrix.append({
                'src': src,
                'dst': dst,
                'volume': volume,                      # Mbit/s
                'kbit': volume * 1000.0 / rate_hz,     # size of one message (kbit) at the sender's message rate
            })

    return scs, comm_matrix, sensors, actuators, cable_types, locations
