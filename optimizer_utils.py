"""
Utility functions for the design space exploration optimizer.

This module contains helper functions used by the optimizer module,
including geometry distance calculations and lookup dictionary builders.
"""

from collections import deque


def get_distance(loc1, loc2):
    """
    Extract the manhattan distance between two locations.
    
    Takes two location objects (Point objects) and returns the manhattan distance 
    between them. Each location object must have a `.dist(other)` method that 
    returns a tuple of (manhattan_distance, euclidean_distance).
    
    Args:
        loc1: First location object (Point) with `.dist()` method, or None
        loc2: Second location object (Point) with `.dist()` method, or None
    
    Returns:
        float: The manhattan distance between the two locations, or 0.0 if either
               location is None
    """
    if loc1 is None or loc2 is None:
        return 0.0
    
    manhattan_dist, _ = loc1.dist(loc2)
    return manhattan_dist

def build_sensor_lookup(sensors):
    """
    Create a lookup dictionary for sensors by their ID.
    
    Args:
        sensors: List of sensor objects, each with an `id` attribute
    
    Returns:
        dict: Dictionary mapping sensor IDs to sensor objects
    """
    return {s.id: s for s in sensors}

def build_actuator_lookup(actuators):
    """
    Create a lookup dictionary for actuators by their ID.
    
    Args:
        actuators: List of actuator objects, each with an `id` attribute
    
    Returns:
        dict: Dictionary mapping actuator IDs to actuator objects
    """
    return {a.id: a for a in actuators}

def build_cost_map(cable_types):
    """
    Create a cost-per-meter lookup dictionary from cable types.
    
    Args:
        cable_types: Dict mapping cable type names to cable type objects
                    with 'cost_per_meter' attribute
    
    Returns:
        dict: Dictionary mapping cable type names to their cost_per_meter values
    """
    return {name: ct.cost_per_meter for name, ct in cable_types.items()}


def build_latency_map(cable_types):
    """
    Create a latency-per-meter lookup dictionary from cable types.
    
    Args:
        cable_types: Dict mapping cable type names to cable type objects
                    with 'latency_per_meter' attribute
    
    Returns:
        dict: Dictionary mapping cable type names to their latency_per_meter values
    """
    return {name: ct.latency_per_meter for name, ct in cable_types.items()}


def precompute_latency_infeasible_pairs(scs, locations, sensors, actuators, cable_types):
    """
    Precompute infeasible (sc_idx, loc_idx) pairs due to sensor/actuator max-latency constraints.

    This identifies which software components cannot be placed at which locations because 
    the latency required for sensor/actuator communication would exceed the maximum allowed latency.
    
    This replaces adding many explicit constraints like:
        sum_p z[i,j,a,p] == 0
    by directly setting the corresponding z variables' upper bounds to 0.
    
    Args:
        scs: List of software component objects, each with 'sensors' and 'actuators' attributes
        locations: List of location/ECU objects with a `.location` attribute (Point)
        sensors: List of sensor objects with 'id', 'location', 'interface', and 'max_latency' attributes
        actuators: List of actuator objects with 'id', 'location', 'interface', and 'max_latency' attributes
        cable_types: Dict mapping cable type names to cable type objects with 'latency_per_meter' attribute
    
    Returns:
        set: Set of (sc_index, location_index) tuples that are infeasible due to latency constraints
    """
    latency_map = {name: ct.latency_per_meter for name, ct in cable_types.items()}
    sensor_lookup = build_sensor_lookup(sensors)
    actuator_lookup = build_actuator_lookup(actuators)

    infeasible = set()
    n_sc = len(scs)
    n_locs = len(locations)

    for i in range(n_sc):
        # Check sensor latency constraints
        for s_id in getattr(scs[i], 'sensors', []) or []:
            sensor = sensor_lookup.get(s_id)
            if not sensor or not getattr(sensor, 'max_latency', None):
                continue
            # ETH sensors may attach to a switch/ECU node; do not preclude SWC placement based on
            # direct sensor->ECU wiring latency in that case.
            if getattr(sensor, 'interface', None) == 'ETH':
                continue
            if not getattr(sensor, 'location', None):
                continue
            for j in range(n_locs):
                if not getattr(locations[j], 'location', None):
                    continue
                dist = get_distance(sensor.location, locations[j].location)
                latency = dist * latency_map.get(getattr(sensor, 'interface', None), 0.0)
                if latency > sensor.max_latency:
                    infeasible.add((i, j))

        # Check actuator latency constraints
        for a_id in getattr(scs[i], 'actuators', []) or []:
            actuator = actuator_lookup.get(a_id)
            if not actuator or not getattr(actuator, 'max_latency', None):
                continue
            if getattr(actuator, 'interface', None) == 'ETH':
                continue
            if not getattr(actuator, 'location', None):
                continue
            for j in range(n_locs):
                if not getattr(locations[j], 'location', None):
                    continue
                dist = get_distance(actuator.location, locations[j].location)
                latency = dist * latency_map.get(getattr(actuator, 'interface', None), 0.0)
                if latency > actuator.max_latency:
                    infeasible.add((i, j))

    return infeasible


# ============================================================================
# COST CALCULATION UTILITIES
# ============================================================================

def calculate_sensor_actuator_cable_costs(z, scs, locations, sensors, actuators, cable_types, attach_s=None, attach_a=None, shared_attach_s=None, shared_attach_a=None, shared_trunk_len=None):
    """
    Calculate total cable costs and distances for sensor/actuator connections.
    
    Iterates through all assigned software components (z variables) and calculates
    the cost and distance of cables connecting sensors and actuators to their
    assigned locations.
    
    Args:
        z: Dict of z variables with keys (sc_idx, loc_idx, asil, partition)
        scs: List of software component objects, each with 'sensors' and 'actuators' attributes
        locations: List of location/ECU objects
        sensors: List of sensor objects with 'id', 'location', and 'interface' attributes
        actuators: List of actuator objects with 'id', 'location', and 'interface' attributes
        cable_types: Dict mapping cable type names to cable type objects with 'cost_per_meter'
    
    Returns:
        tuple: (cable_length, cable_cost)
            - cable_length: Total distance of all sensor/actuator cables (float)
            - cable_cost: Total cost of all sensor/actuator cables (float)
    """
    cable_length = 0.0
    cable_cost = 0.0
    cost_map = {name: ct.cost_per_meter for name, ct in cable_types.items()}
    sensor_lookup = build_sensor_lookup(sensors)
    actuator_lookup = build_actuator_lookup(actuators)
    
    shared_bus_ifaces = {'CAN', 'LIN', 'FLEXRAY'}

    for (i, j, a, p), var in z.items():
        if var.X > 0.5:
            sc = scs[i]
            
            # Sensor cable costs and distances
            for s_id in (sc.sensors or []):
                sensor = sensor_lookup.get(s_id)
                if sensor and sensor.location:
                    if attach_s is not None and getattr(sensor, 'interface', None) == 'ETH':
                        continue
                    if shared_attach_s is not None and getattr(sensor, 'interface', None) in shared_bus_ifaces:
                        continue
                    dist = get_distance(sensor.location, locations[j].location)
                    cable_length += dist
                    cable_cost += dist * cost_map.get(sensor.interface, 0.0)
            
            # Actuator cable costs and distances
            for a_id in (sc.actuators or []):
                actuator = actuator_lookup.get(a_id)
                if actuator and actuator.location:
                    if attach_a is not None and getattr(actuator, 'interface', None) == 'ETH':
                        continue
                    if shared_attach_a is not None and getattr(actuator, 'interface', None) in shared_bus_ifaces:
                        continue
                    dist = get_distance(actuator.location, locations[j].location)
                    cable_length += dist
                    cable_cost += dist * cost_map.get(actuator.interface, 0.0)

    # ETH attachment cables
    if attach_s:
        for (si, j), av in attach_s.items():
            if av.X > 0.5:
                s = sensors[si]
                if not getattr(s, 'location', None):
                    continue
                dist = get_distance(s.location, locations[j].location)
                cable_length += dist
                cable_cost += dist * cost_map.get(getattr(s, 'interface', 'ETH'), 0.0)

    if attach_a:
        for (ai, j), av in attach_a.items():
            if av.X > 0.5:
                aobj = actuators[ai]
                if not getattr(aobj, 'location', None):
                    continue
                dist = get_distance(aobj.location, locations[j].location)
                cable_length += dist
                cable_cost += dist * cost_map.get(getattr(aobj, 'interface', 'ETH'), 0.0)

    # Shared-bus attachments (CAN/LIN/FLEXRAY)
    if shared_attach_s:
        for (si, j), av in shared_attach_s.items():
            if av.X > 0.5:
                s = sensors[si]
                if not getattr(s, 'location', None):
                    continue
                iface = getattr(s, 'interface', None)
                if iface not in shared_bus_ifaces:
                    continue
                dist = get_distance(s.location, locations[j].location)
                cable_length += dist
                cable_cost += dist * cost_map.get(iface, 0.0)

    if shared_attach_a:
        for (ai, j), av in shared_attach_a.items():
            if av.X > 0.5:
                aobj = actuators[ai]
                if not getattr(aobj, 'location', None):
                    continue
                iface = getattr(aobj, 'interface', None)
                if iface not in shared_bus_ifaces:
                    continue
                dist = get_distance(aobj.location, locations[j].location)
                cable_length += dist
                cable_cost += dist * cost_map.get(iface, 0.0)

    # Shared trunk length contribution (one main segment per location/interface)
    if shared_trunk_len:
        for (_j, iface), tv in shared_trunk_len.items():
            trunk_len = float(getattr(tv, 'X', 0.0) or 0.0)
            if trunk_len <= 1e-9:
                continue
            cable_length += trunk_len
            cable_cost += trunk_len * cost_map.get(iface, 0.0)
    
    return cable_length, cable_cost


def calculate_partition_cost(y, partitions):
    """
    Calculate total partition cost based on number of opened partitions.
    
    Args:
        y: Dict of y variables with keys (location_idx, asil, partition)
        partitions: Dict with key 'cost' representing the cost per partition
    
    Returns:
        float: Total partition cost
    """
    num_partitions_opened = 0
    for (j, a, p), var in y.items():
        if var.X > 0.5:
            num_partitions_opened += 1
    
    partition_cost = num_partitions_opened * partitions.get('cost', 0)
    return partition_cost


def calculate_hardware_cost(hw_use, hardwares, locations):
    """
    Calculate total hardware cost based on opened hardware at locations.
    
    Args:
        hw_use: Dict of hw_use variables with keys (location_idx, hardware_name)
        hardwares: Dict mapping hardware names to their costs (float)
        locations: List of location/ECU objects
    
    Returns:
        tuple: (hw_cost, hw_opened list)
            - hw_cost: Total hardware cost (float)
            - hw_opened: List of opened hardware with format "HW_NAME@LOCATION_ID"
    """
    hw_opened = []
    for (j, h), var in hw_use.items():
        if var.X > 0.5:
            hw_opened.append(f"{h}@{locations[j].id}")
    
    hw_cost = sum(hardwares.get(hw_name.split('@')[0], 0) for hw_name in hw_opened)
    return hw_cost, hw_opened


def calculate_interface_cost(if_use, interfaces, locations):
    """
    Calculate total interface cost based on number of ports at locations.
    
    Args:
        if_use: Dict of if_use variables with keys (location_idx, interface_name)
        interfaces: Dict mapping interface names to interface objects with 'port_cost' attribute
        locations: List of location/ECU objects
    
    Returns:
        tuple: (if_cost, if_opened list)
            - if_cost: Total interface cost (float)
            - if_opened: List of opened interfaces with count, e.g., "3xETH@LOCATION_ID"
    """
    if_opened = []
    if_cost = 0.0
    for (j, i_name), var in if_use.items():
        if var.X > 0.1:  # Use a small epsilon for integer variables
            count = int(round(var.X))
            if_opened.append(f"{count}x{i_name}@{locations[j].id}")
            if_cost += count * interfaces[i_name].port_cost
    
    return if_cost, if_opened


def calculate_communication_cost(comm, locations, cable_types):
    """
    Calculate total communication cost for ECU-to-ECU connections.
    
    Calculates the cost of communication backbone links between locations
    based on distance and cable type.
    
    Args:
        comm: Dict of comm variables with keys (location_idx1, location_idx2, interface_name)
        locations: List of location/ECU objects
        cable_types: Dict mapping cable type names to cable type objects with 'cost_per_meter'
    
    Returns:
        float: Total communication cost
    """
    comm_cost = 0.0
    for (j1, j2, iface), var in comm.items():
        if var.X > 0.5:
            num_links = int(round(var.X))
            dist = get_distance(locations[j1].location, locations[j2].location)
            cost_per_link = dist * cable_types[iface].cost_per_meter
            comm_cost += num_links * cost_per_link
    
    return comm_cost


def extract_communication_links(comm, locations, min_link_value=0.5):
    """Extract opened ECU-to-ECU links from solved comm variables."""
    links = []
    for (j1, j2, iface), var in comm.items():
        if var.X > min_link_value:
            links.append({
                'src_loc': locations[j1].id,
                'dst_loc': locations[j2].id,
                'iface': iface,
                'count': int(round(var.X)),
            })
    return links



# ============================================================================
# SOLUTION EXTRACTION AND FORMATTING
# ============================================================================
def extract_solution(y, z, hw_use, if_use, comm, scs, locations, sensors, actuators,
                     cable_types, partitions, hardwares, interfaces, comm_matrix=None,
                     traffic_flows=None, flow=None, attach_s=None, attach_a=None,
                     shared_attach_s=None, shared_attach_a=None, shared_trunk_len=None):
        """
        Extract formatted solution dict from raw Gurobi model variables.
        
        This is a PRESENTATION LAYER function - converts raw optimization results 
        into a human-readable solution dictionary for visualization and reporting.
        
        Args:
            y: Dict of partition opening variables
            z: Dict of SC assignment variables  
            hw_use: Dict of hardware opening variables
            if_use: Dict of interface opening variables
            comm: Dict of communication backbone variables
            scs: List of software component objects
            locations: List of location/ECU objects
            sensors: List of sensor objects
            actuators: List of actuator objects
            cable_types: Dict of cable type objects
            partitions: Dict with partition configuration
            hardwares: Dict mapping hardware names to costs
            interfaces: Dict mapping interface names to interface objects
            comm_matrix: Optional list of communication requirements
        
        Returns:
            dict: Formatted solution dictionary with keys:
                - assignment: {SC_id: location_id}
                - partitions: {SC_id: partition_name}
                - hw_cost, if_cost, cable_cost, comm_cost, partition_cost
                - total_cost
                - hw_features, interfaces (opened features)
                - num_locations_used, num_partitions, cable_length
                - status: 'OPTIMAL'
        """
        assignment_map = {}
        partition_map = {}
        num_locations_used = 0
        num_partitions_opened = 0
        
        # Extract SC assignments and partitions from z
        for (i, j, a, p), var in z.items():
            if var.X > 0.5:
                assignment_map[scs[i].id] = locations[j].id
                partition_map[scs[i].id] = f"{locations[j].id}_asil{a}_p{p}"
        
        # DEBUG: Print SC placements and communication requirements
        print(f"\n[DEBUG] SC Placement Analysis:")
        print(f"  Total SCs assigned: {len(assignment_map)}")
        if comm_matrix:
            print(f"  Communication links required: {len(comm_matrix)}")
            for link in comm_matrix:
                src_id = link.get('src')
                dst_id = link.get('dst')
                vol = link.get('volume', 0)
                src_loc = assignment_map.get(src_id, 'NOT_ASSIGNED')
                dst_loc = assignment_map.get(dst_id, 'NOT_ASSIGNED')
                same_loc = (src_loc == dst_loc)
                print(f"    {src_id} ({src_loc}) -> {dst_id} ({dst_loc}) vol={vol} {'[SAME_LOC]' if same_loc else '[DIFFERENT_LOC]'}")
        
        # Extract partitions opened
        for (j, a, p), var in y.items():
            if var.X > 0.5:
                num_partitions_opened += 1
        
        # Extract HW and interfaces opened
        hw_cost, hw_opened = calculate_hardware_cost(hw_use, hardwares, locations)
        
        # Debug: Check if SWITCH was selected
        switches_selected = [hw for hw in hw_opened if hw.startswith("SWITCH")]
        all_hw_available = list(hardwares.keys()) if hardwares else []
        if switches_selected:
            print(f"[DEBUG] SWITCH selected at: {switches_selected}")
        elif "SWITCH" in all_hw_available:
            print(f"[DEBUG] WARNING: SWITCH available but NOT selected in optimal solution")
        
        if_cost, if_opened = calculate_interface_cost(if_use, interfaces, locations)
        
        # Count locations used
        locs_used = set()
        for (i, j, a, p), var in z.items():
            if var.X > 0.5:
                locs_used.add(j)
        num_locations_used = len(locs_used)
        
        # Calculate cable costs and distances
        cable_length, cable_cost = calculate_sensor_actuator_cable_costs(
            z, scs, locations, sensors, actuators, cable_types,
            attach_s=attach_s, attach_a=attach_a,
            shared_attach_s=shared_attach_s, shared_attach_a=shared_attach_a,
            shared_trunk_len=shared_trunk_len
        )
        
        # Calculate partition and communication costs
        partition_cost = calculate_partition_cost(y, partitions)
        comm_cost = calculate_communication_cost(comm, locations, cable_types)
        comm_links = extract_communication_links(comm, locations)

        # DEBUG: ETH peripheral attachments
        if attach_s:
            attached = []
            for (si, j), av in attach_s.items():
                if av.X > 0.5:
                    attached.append((sensors[si].id, locations[j].id))
            print("\n[DEBUG] ETH Sensor Attachments:")
            if not attached:
                print("  None")
            else:
                print(f"  Total: {len(attached)}")
                for sid, lid in attached[:20]:
                    print(f"  {sid} -> {lid}")
                if len(attached) > 20:
                    print(f"  ... ({len(attached) - 20} more)")

        if attach_a:
            attached = []
            for (ai, j), av in attach_a.items():
                if av.X > 0.5:
                    attached.append((actuators[ai].id, locations[j].id))
            print("\n[DEBUG] ETH Actuator Attachments:")
            if not attached:
                print("  None")
            else:
                print(f"  Total: {len(attached)}")
                for aid, lid in attached[:20]:
                    print(f"  {aid} -> {lid}")
                if len(attached) > 20:
                    print(f"  ... ({len(attached) - 20} more)")

        # DEBUG: Selected backbone links (comm)
        print(f"\n[DEBUG] Selected Backbone Links (comm):")
        if not comm_links:
            print("  No backbone links opened")
        else:
            # Stable sort by iface then endpoints
            def _lk_sort_key(lk):
                a, b = (lk['src_loc'], lk['dst_loc']) if lk['src_loc'] <= lk['dst_loc'] else (lk['dst_loc'], lk['src_loc'])
                return (lk.get('iface', ''), a, b)

            for lk in sorted(comm_links, key=_lk_sort_key):
                src_id = lk['src_loc']
                dst_id = lk['dst_loc']
                iface = lk.get('iface', '?')
                count = lk.get('count', 0)
                # Find indices for distance (safe fallback if ids are unexpected)
                src_idx = next((ii for ii, loc in enumerate(locations) if loc.id == src_id), None)
                dst_idx = next((ii for ii, loc in enumerate(locations) if loc.id == dst_id), None)
                if src_idx is not None and dst_idx is not None:
                    dist = get_distance(locations[src_idx].location, locations[dst_idx].location)
                    cpm = cable_types.get(iface).cost_per_meter if cable_types.get(iface) else 0.0
                    print(f"  {src_id} <-> {dst_id} iface={iface} count={count} dist={dist:.2f} cost/link={dist * cpm:.2f}")
                else:
                    print(f"  {src_id} <-> {dst_id} iface={iface} count={count}")

        # DEBUG: Traffic flow routing analysis (multi-hop detection)
        print(f"\n[DEBUG] Traffic Flow Routing Analysis:")
        if not traffic_flows or not flow:
            print("  Routing analysis unavailable (flow variables not provided to extract_solution)")
        else:
            # Map SC index -> location index for quick lookup
            sc_idx_to_loc = {}
            for (i, j, _a, _p), var in z.items():
                if var.X > 0.5:
                    sc_idx_to_loc[i] = j

            # Map location index -> has SWITCH
            loc_has_switch = {j: False for j in range(len(locations))}
            for (j, h), var in hw_use.items():
                if h == 'SWITCH' and var.X > 0.5:
                    loc_has_switch[j] = True

            # Build adjacency per traffic id once (only active arcs)
            adj_by_t = {tr['id']: {j: [] for j in range(len(locations))} for tr in traffic_flows if 'id' in tr}
            for (tt, u, v), fvar in flow.items():
                if fvar.X <= 0.5:
                    continue
                if tt not in adj_by_t:
                    continue
                adj_by_t[tt][u].append(v)

            def _shortest_path_nodes(adj, src, dst):
                if src == dst:
                    return [src]
                q = deque([src])
                prev = {src: None}
                while q:
                    u = q.popleft()
                    for v in adj.get(u, []):
                        if v in prev:
                            continue
                        prev[v] = u
                        if v == dst:
                            q.clear()
                            break
                        q.append(v)
                if dst not in prev:
                    return None
                # reconstruct
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path

            # Detailed per-traffic routing table (SC_SC + ETH sensor/actuator)
            print("  Traffic routes (only DIFFERENT_LOC):")

            total = 0
            different_loc = 0
            multi_hop = 0
            missing_paths = 0
            transit_nodes = set()
            transit_switch_nodes = set()

            # Optional edge-load summary (directed) for quick sanity checks
            edge_load = {}  # (u,v) -> total volume

            # Helpers to resolve peripheral attachment locations
            def _sensor_attach_loc(si):
                if not attach_s:
                    return None
                for j in range(len(locations)):
                    v = attach_s.get((si, j))
                    if v is not None and v.X > 0.5:
                        return j
                return None

            def _act_attach_loc(ai):
                if not attach_a:
                    return None
                for j in range(len(locations)):
                    v = attach_a.get((ai, j))
                    if v is not None and v.X > 0.5:
                        return j
                return None

            for tr in traffic_flows:
                if tr.get('type') not in ('SC_SC', 'SENS_SC', 'SC_ACT'):
                    continue
                total += 1

                t_id = tr.get('id')
                vol = tr.get('volume', 0)
                if t_id is None:
                    continue

                ttype = tr.get('type')
                src_loc = None
                dst_loc = None
                label = None

                if ttype == 'SC_SC':
                    src_i = tr.get('src_idx')
                    dst_i = tr.get('dst_idx')
                    if src_i is None or dst_i is None:
                        continue
                    src_loc = sc_idx_to_loc.get(src_i)
                    dst_loc = sc_idx_to_loc.get(dst_i)
                    if src_loc is None or dst_loc is None:
                        continue
                    label = f"SC_SC {scs[src_i].id}@{locations[src_loc].id} -> {scs[dst_i].id}@{locations[dst_loc].id}"

                elif ttype == 'SENS_SC':
                    si = tr.get('sensor_idx')
                    dst_i = tr.get('dst_idx')
                    if si is None or dst_i is None:
                        continue
                    src_loc = _sensor_attach_loc(si)
                    dst_loc = sc_idx_to_loc.get(dst_i)
                    if src_loc is None or dst_loc is None:
                        continue
                    label = f"SENS_SC {sensors[si].id}@{locations[src_loc].id} -> {scs[dst_i].id}@{locations[dst_loc].id}"

                else:  # SC_ACT
                    src_i = tr.get('src_idx')
                    ai = tr.get('act_idx')
                    if src_i is None or ai is None:
                        continue
                    src_loc = sc_idx_to_loc.get(src_i)
                    dst_loc = _act_attach_loc(ai)
                    if src_loc is None or dst_loc is None:
                        continue
                    label = f"SC_ACT {scs[src_i].id}@{locations[src_loc].id} -> {actuators[ai].id}@{locations[dst_loc].id}"

                if src_loc == dst_loc:
                    continue

                different_loc += 1
                adj = adj_by_t.get(t_id)
                path = _shortest_path_nodes(adj, src_loc, dst_loc) if adj else None
                if not path:
                    missing_paths += 1
                    print(
                        f"    t={t_id} {label} vol={vol} iface=ETH path=UNREACHABLE"
                    )
                    continue

                # Accumulate edge loads along the chosen path
                for u, v in zip(path, path[1:]):
                    edge_load[(u, v)] = edge_load.get((u, v), 0) + vol

                is_multihop = len(path) > 2
                if is_multihop:
                    multi_hop += 1
                    for mid in path[1:-1]:
                        transit_nodes.add(mid)
                        if loc_has_switch.get(mid, False):
                            transit_switch_nodes.add(mid)

                path_ids = [locations[j].id for j in path]
                via = []
                for mid in path[1:-1]:
                    if loc_has_switch.get(mid, False):
                        via.append(f"SWITCH@{locations[mid].id}")
                via_txt = (" via " + ", ".join(via)) if via else ""

                print(
                    f"    t={t_id} {label} vol={vol} iface=ETH path={' -> '.join(path_ids)}{via_txt}"
                )

            # Summary
            if different_loc == 0:
                print("  No modeled traffics across different locations")
            else:
                if multi_hop == 0:
                    print(f"  Summary: multi-hop=0 / different_loc={different_loc} (total traffics={total})")
                else:
                    print(f"  Summary: multi-hop={multi_hop} / different_loc={different_loc} (total traffics={total})")
                    if transit_nodes:
                        transit_ids = [locations[j].id for j in sorted(transit_nodes)]
                        print(f"  Transit nodes used: {transit_ids}")
                    if transit_switch_nodes:
                        transit_switch_ids = [locations[j].id for j in sorted(transit_switch_nodes)]
                        print(f"  Transit SWITCH nodes: {transit_switch_ids}")

                if missing_paths > 0:
                    print(f"  WARNING: {missing_paths} different-loc traffics had no reachable path in flow vars")

            # Edge load summary (print only if multi-hop or if links exist)
            if comm_links and edge_load:
                print("  Directed edge load summary (vol units):")
                # Sort by descending load
                for (u, v), load in sorted(edge_load.items(), key=lambda kv: (-kv[1], locations[kv[0][0]].id, locations[kv[0][1]].id))[:25]:
                    print(f"    {locations[u].id} -> {locations[v].id}: load={load}")
        
        total_cost = partition_cost + hw_cost + if_cost + cable_cost + comm_cost

        eth_sensor_attachments = {}
        if attach_s:
            for (si, j), av in attach_s.items():
                if av.X > 0.5:
                    eth_sensor_attachments[sensors[si].id] = locations[j].id

        eth_actuator_attachments = {}
        if attach_a:
            for (ai, j), av in attach_a.items():
                if av.X > 0.5:
                    eth_actuator_attachments[actuators[ai].id] = locations[j].id

        shared_sensor_attachments = {}
        if shared_attach_s:
            for (si, j), av in shared_attach_s.items():
                if av.X > 0.5:
                    shared_sensor_attachments[sensors[si].id] = locations[j].id

        shared_actuator_attachments = {}
        if shared_attach_a:
            for (ai, j), av in shared_attach_a.items():
                if av.X > 0.5:
                    shared_actuator_attachments[actuators[ai].id] = locations[j].id
        
        solution = {
            'assignment': assignment_map,
            'partitions': partition_map,
            'hardware_cost': hw_cost,  # For visualizer compatibility
            'hw_cost': hw_cost,
            'interface_cost': if_cost,
            'cable_cost': cable_cost,
            'comm_cost': comm_cost,
            'comm_links': comm_links,
            'eth_sensor_attachments': eth_sensor_attachments,
            'eth_actuator_attachments': eth_actuator_attachments,
            'shared_sensor_attachments': shared_sensor_attachments,
            'shared_actuator_attachments': shared_actuator_attachments,
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
