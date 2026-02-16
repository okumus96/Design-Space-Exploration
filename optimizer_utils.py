"""
Utility functions for the design space exploration optimizer.

This module contains helper functions used by the optimizer module,
including geometry distance calculations and lookup dictionary builders.
"""


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

def calculate_sensor_actuator_cable_costs(z, scs, locations, sensors, actuators, cable_types):
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
    
    for (i, j, a, p), var in z.items():
        if var.X > 0.5:
            sc = scs[i]
            
            # Sensor cable costs and distances
            for s_id in (sc.sensors or []):
                sensor = sensor_lookup.get(s_id)
                if sensor and sensor.location:
                    dist = get_distance(sensor.location, locations[j].location)
                    cable_length += dist
                    cable_cost += dist * cost_map.get(sensor.interface, 0.0)
            
            # Actuator cable costs and distances
            for a_id in (sc.actuators or []):
                actuator = actuator_lookup.get(a_id)
                if actuator and actuator.location:
                    dist = get_distance(actuator.location, locations[j].location)
                    cable_length += dist
                    cable_cost += dist * cost_map.get(actuator.interface, 0.0)
    
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
    Calculate total interface cost based on opened interfaces at locations.
    
    Args:
        if_use: Dict of if_use variables with keys (location_idx, interface_name)
        interfaces: Dict mapping interface names to interface objects with 'port_cost' attribute
        locations: List of location/ECU objects
    
    Returns:
        tuple: (if_cost, if_opened list)
            - if_cost: Total interface cost (float)
            - if_opened: List of opened interfaces with format "INTERFACE_NAME@LOCATION_ID"
    """
    if_opened = []
    for (j, i_name), var in if_use.items():
        if var.X > 0.5:
            if_opened.append(f"{i_name}@{locations[j].id}")
    
    if_cost = sum(interfaces[iface_name.split('@')[0]].port_cost for iface_name in if_opened)
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



# ============================================================================
# SOLUTION EXTRACTION AND FORMATTING
# ============================================================================
def extract_solution(y, z, hw_use, if_use, comm, w, scs, locations, sensors, actuators,
                     cable_types, partitions, hardwares, interfaces, comm_matrix=None):
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
            w: Dict of traffic flow variables
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
        cable_length, cable_cost = calculate_sensor_actuator_cable_costs(z, scs, locations, sensors, actuators, cable_types)
        
        # Calculate partition and communication costs
        partition_cost = calculate_partition_cost(y, partitions)
        comm_cost = calculate_communication_cost(comm, locations, cable_types)
        
        # DEBUG: Print traffic flow information
        print(f"\n[DEBUG] Traffic Flow Routing Analysis:")
        if w:
            multi_hop_traffic = 0
            for key, var in w.items():
                if var.X > 0.5:
                    multi_hop_traffic += 1
            if multi_hop_traffic > 0:
                print(f"  Multi-hop traffic found: {multi_hop_traffic} paths need routing through intermediate nodes")
            else:
                print(f"  NO multi-hop traffic detected - all traffic is direct (P2P)")
                print(f"  Reason: Source and destination nodes are always direct, no intermediate hops needed")
        
        total_cost = partition_cost + hw_cost + if_cost + cable_cost + comm_cost
        
        solution = {
            'assignment': assignment_map,
            'partitions': partition_map,
            'hardware_cost': hw_cost,  # For visualizer compatibility
            'hw_cost': hw_cost,
            'interface_cost': if_cost,
            'cable_cost': cable_cost,
            'comm_cost': comm_cost,
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
