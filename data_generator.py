import random
from models import Point, Sensor, Actuator, SoftwareComponent, CandidateECU, CableType


class VehicleDataGenerator:
    def __init__(self, num_ecus=10, num_scs=20, seed=42, config_reader=None):
        random.seed(seed)
        self.num_ecus = num_ecus
        self.num_scs = num_scs
        self.sensors = []
        self.actuators = []
        self.ecus = []
        self.scs = []
        self.comm_matrix = []

        if config_reader is None:
            raise ValueError("config_reader is required")
        else: 
            self.config_reader = config_reader
        
        # Vehicle dimensions from config
        dimensions = self.config_reader.get_vehicle_dimensions()
        self.VEHICLE_LENGTH = dimensions['length']
        self.VEHICLE_WIDTH = dimensions['width']
        
        # ECU placement bounds from config
        ecu_bounds = dimensions.get('ecu_placement_bounds')
        self.ECU_X_MIN = ecu_bounds['x_min']
        self.ECU_X_MAX = ecu_bounds['x_max']
        self.ECU_Y_MIN = ecu_bounds['y_min']
        self.ECU_Y_MAX = ecu_bounds['y_max']
        
        # Build cable types from config
        self.cable_types = self.config_reader.get_cable_types()
        
        # Get SC domain weights from config
        self.sc_domain_weights = self.config_reader.get_sc_domain_weights()

    def generate_sensors(self):
        """Generate sensors from configuration file"""
        hw = self.VEHICLE_WIDTH / 2
        hl = self.VEHICLE_LENGTH / 2
        
        sensors_config = self.config_reader.get_sensors_config()
        for sensor_config in sensors_config:
            location = Point(
                sensor_config['location']['x_ratio'] * hw,
                sensor_config['location']['y_ratio'] * hl
            )
            sensor = Sensor(
                id=sensor_config['id'],
                type=sensor_config['type'],
                interface=sensor_config['interface'],
                volume=sensor_config['volume'],
                location=location,
                max_latency=random.randint(5, 100)  # 5-100ms tolerance
            )
            self.sensors.append(sensor)

    def generate_actuators(self):
        """Generate actuators from configuration file"""
        hw = self.VEHICLE_WIDTH / 2
        hl = self.VEHICLE_LENGTH / 2
        
        actuators_config = self.config_reader.get_actuators_config()
        for actuator_config in actuators_config:
            location = Point(
                actuator_config['location']['x_ratio'] * hw,
                actuator_config['location']['y_ratio'] * hl
            )
            actuator = Actuator(
                id=actuator_config['id'],
                type=actuator_config['type'],
                interface=actuator_config['interface'],
                volume=actuator_config['volume'],
                location=location,
                max_latency=random.randint(5, 100)  # 5-100ms tolerance
            )
            self.actuators.append(actuator)

    def generate_scs(self, weights=None):
        """Generate software components with domain distribution from config"""
        if weights is None:
            weights = self.sc_domain_weights
        
        # Get domains from config
        domains = self.config_reader.get_domains()
        
        # Get domain configurations from config
        domain_configs = self.config_reader.get_sc_domain_configs()

        # Calculate exact count for each domain based on weights
        domain_counts = {}
        remaining = self.num_scs
        
        for i, domain in enumerate(domains):
            if i < len(domains) - 1:
                count = round(weights[i] * self.num_scs)
                domain_counts[domain] = count
                remaining -= count
            else:
                # Last domain gets remaining SCs to ensure exact total
                domain_counts[domain] = remaining
        
        # Create list of domains with exact counts
        assigned_domains = []
        for domain, count in domain_counts.items():
            assigned_domains.extend([domain] * count)
        
        # Calculate HW requirements assignment for each domain deterministically
        hw_assignments = {}  # domain -> list of hw_required lists
        for domain, count in domain_counts.items():
            domain_config = domain_configs.get(domain)
            if not domain_config:
                continue
                
            hw_requirements = domain_config.get('hw_requirements', {})
            hw_list = []
            
            for hw_name, probability in hw_requirements.items():
                # Calculate exact count of SCs that should have this HW requirement
                hw_count = round(probability * count)
                # Add this HW to the first hw_count SCs
                for j in range(count):
                    if j < hw_count:
                        if j >= len(hw_list):
                            hw_list.append([])
                        hw_list[j].append(hw_name)
                    else:
                        if j >= len(hw_list):
                            hw_list.append([])
            
            # Shuffle HW assignments within domain to randomize which SCs get which HW
            random.shuffle(hw_list)
            hw_assignments[domain] = hw_list
        
        # Shuffle to avoid ordered placement
        #random.shuffle(assigned_domains)
        
        # Track index per domain for HW assignment
        domain_indices = {domain: 0 for domain in domains}

        for i, domain in enumerate(assigned_domains):
            domain_config = domain_configs.get(domain)
            if not domain_config:
                continue
            
            # Get pre-assigned HW requirements for this SC
            domain_idx = domain_indices[domain]
            hw_req = hw_assignments.get(domain, [[]])[domain_idx] if domain in hw_assignments else []
            domain_indices[domain] += 1
            
            # Get resource requirements from config
            cpu_range = domain_config.get('cpu_range')
            ram_range = domain_config.get('ram_range')
            rom_range = domain_config.get('rom_range')
            asil_levels = domain_config.get('asil_levels')
            
            sc = SoftwareComponent(
                id=f"SC_{i}_{domain[:4].upper()}",
                domain=domain,
                cpu_req=random.randint(cpu_range[0], cpu_range[1]),
                ram_req=random.randint(ram_range[0], ram_range[1]),
                rom_req=random.randint(rom_range[0], rom_range[1]),
                asil_req=random.choice(asil_levels),
                hw_required=hw_req
            )
            
            self.scs.append(sc)
    
    def assign_redundancy(self):
        """
        Assign redundancy pairs for high-ASIL safety-critical components.
        Uses domain-specific redundancy ratios from software.json config.
        Redundant SCs must be placed on different ECUs for fault tolerance.
        """
        domain_configs = self.config_reader.get_sc_domain_configs()
        
        # Group high-ASIL SCs by domain
        high_asil_by_domain = {}
        for sc in self.scs:
            if sc.asil_req == 4:  # ASIL D only
                if sc.domain not in high_asil_by_domain:
                    high_asil_by_domain[sc.domain] = []
                high_asil_by_domain[sc.domain].append(sc)
        
        # Process each domain separately
        for domain, scs_in_domain in high_asil_by_domain.items():
            domain_config = domain_configs.get(domain, {})
            redundancy_ratio = domain_config.get('redundancy_ratio', 0.0)
            
            if redundancy_ratio <= 0:
                continue
            
            # Calculate number of redundant pairs for this domain
            num_redundant = max(1, int(len(scs_in_domain) * redundancy_ratio))
            redundant_candidates = random.sample(scs_in_domain, min(num_redundant, len(scs_in_domain)))
            
            # Create pairs within this domain
            for sc in redundant_candidates:
                candidates = [s for s in scs_in_domain 
                             if s != sc and not s.redundant_with]
                if candidates:
                    partner = random.choice(candidates)
                    sc.redundant_with = partner.id
                    partner.redundant_with = sc.id
   
    def assign_sensors_actuators(self):
        """
        Assign sensors/actuators to SCs:
        - Each SC gets EITHER 1 sensor OR 1 actuator, never both
        - Distribute evenly across SCs in each domain (round-robin style)
        """
        # Get assignment config from config reader
        assignment_config = self.config_reader.get_sc_sensor_actuator_assignments()
        
        sensor_assignments = assignment_config.get('sensor_assignments', {})
        actuator_assignments = assignment_config.get('actuator_assignments', {})
        
        # Initialize all SC sensors/actuators as empty
        for sc in self.scs:
            sc.sensors = []
            sc.actuators = []
        
        # Group SCs by domain for easy lookup
        domain_scs = {}
        for sc in self.scs:
            if sc.domain not in domain_scs:
                domain_scs[sc.domain] = []
            domain_scs[sc.domain].append(sc)
        
        # For each domain, combine sensors and actuators, then distribute
        for domain, scs_in_domain in domain_scs.items():
            if not scs_in_domain:
                continue
            
            # Shuffle for randomness
            random.shuffle(scs_in_domain)
            
            # Collect all sensors for this domain
            domain_sensor_ids = sensor_assignments.get(domain, [])
            domain_sensors = [s.id for s in self.sensors if s.id in domain_sensor_ids]
            
            # Collect all actuators for this domain
            domain_actuator_ids = actuator_assignments.get(domain, [])
            domain_actuators = [a.id for a in self.actuators if a.id in domain_actuator_ids]
            
            # Combine sensors and actuators with tags
            items = [('sensor', sid) for sid in domain_sensors] + [('actuator', aid) for aid in domain_actuators]
            
            # Shuffle combined list for randomness
            random.shuffle(items)
            
            # Distribute items ensuring 1 item per SC limit and Interface compatibility
            assigned_scs = set()
            
            # Helper to get item interface
            def get_item_interface(itype, iid):
                if itype == 'sensor':
                    return next((s.interface for s in self.sensors if s.id == iid), None)
                else:
                    return next((a.interface for a in self.actuators if a.id == iid), None)

            for item_type, item_id in items:
                item_interface = get_item_interface(item_type, item_id)
                
                # Find best candidate SC
                best_sc = None
                
                # Try to find an SC that Matches HW requirements (e.g. HW_ACC/DSP -> requires ETH)
                # And has no items yet
                candidates = [sc for sc in scs_in_domain if sc.id not in assigned_scs]
                
                # Filter candidates based on interface compatibility with their HW requirements
                valid_candidates = []
                for sc in candidates:
                    # If SC requires HW_ACC/DSP OR has huge CPU requirements (>30k), it must go on HPC (ETH only)
                    # Because only HPC handles >30k CPU, and HPC only offers ETH.
                    is_hpc_bound = ('HW_ACC' in sc.hw_required or 'DSP' in sc.hw_required or sc.cpu_req > 30000)
                    
                    if is_hpc_bound:
                        if item_interface == 'ETH':
                            valid_candidates.append(sc)
                    # If SC requires HSM or is generic, it goes on ZONE or MCU (CAN, ETH, LIN...)
                    else:
                        valid_candidates.append(sc)
                
                if valid_candidates:
                    best_sc = valid_candidates[0] # Pick first valid (already shuffled)
                elif candidates and item_interface != 'ETH': 
                    # If no valid candidates found but we have generic candidates and item is not ETH
                    # (Fallback for e.g. CAN item but only HW_ACC SCs left? Should not happen if config is sane)
                     pass

                if best_sc:
                    if item_type == 'sensor':
                        best_sc.sensors.append(item_id)
                    else:
                        best_sc.actuators.append(item_id)
                    assigned_scs.add(best_sc.id)
                else:
                    # Could not assign this item (no suitable empty SCs left)
                    # This respects "One SC = One Item", so excess items are dropped/unassigned
                    pass
        
        # Derive interface_required from assigned sensors/actuators
        for sc in self.scs:
            interfaces = set()
            
            # Collect interfaces from sensors
            for sensor_id in sc.sensors:
                sensor = next((s for s in self.sensors if s.id == sensor_id), None)
                if sensor:
                    interfaces.add(sensor.interface)
            
            # Collect interfaces from actuators
            for actuator_id in sc.actuators:
                actuator = next((a for a in self.actuators if a.id == actuator_id), None)
                if actuator:
                    interfaces.add(actuator.interface)
            
            sc.interface_required = list(interfaces)

    def generate_comm_matrix(self):
        """Generate communication matrix between SCs based on random probabilities"""
        # Get communication probabilities from config
        comm_config = self.config_reader.get_sc_communication_config()
        
        intra_prob = comm_config.get('intra_domain_probability')
        inter_prob = comm_config.get('inter_domain_probability')
        
        for i, src in enumerate(self.scs):
            for j, dst in enumerate(self.scs):
                if i != j:
                    prob = intra_prob if src.domain == dst.domain else inter_prob                 
                    if random.random() < prob:
                        volume = random.randint(1, 50)
                        max_latency = random.randint(10, 500)
                        self.comm_matrix.append({
                            'src': src.id,
                            'dst': dst.id,
                            'volume': volume,
                            'max_latency': max_latency,
                        })
    
    def generate_ecu(self):
        """Generate ECUs using types and placement bounds from configuration"""
        ecu_types = self.config_reader.get_ecu_types_config()
        
        # Calculate exact count for each ECU type based on probability
        type_counts = {}
        remaining = self.num_ecus
        
        for i, ecu_type_config in enumerate(ecu_types):
            if i < len(ecu_types) - 1:
                count = round(ecu_type_config['probability'] * self.num_ecus)
                type_counts[ecu_type_config['type']] = count
                remaining -= count
            else:
                # Last type gets remaining ECUs to ensure exact total
                type_counts[ecu_type_config['type']] = remaining
        
        # Create list of ECU type configs with exact counts
        ecu_type_list = []
        for ecu_type_config in ecu_types:
            count = type_counts[ecu_type_config['type']]
            ecu_type_list.extend([ecu_type_config] * count)
        
        # Shuffle to avoid ordered placement
        random.shuffle(ecu_type_list)
        
        # Place ECUs in a grid pattern using config bounds
        cols = int(self.num_ecus ** 0.5) 
        rows = (self.num_ecus + cols - 1) // cols
        
        x_step = (self.ECU_X_MAX - self.ECU_X_MIN) / max(1, cols - 1) if cols > 1 else 0
        y_step = (self.ECU_Y_MAX - self.ECU_Y_MIN) / max(1, rows - 1) if rows > 1 else 0

        ecu_idx = 0
        for r in range(rows):
            for c in range(cols):
                if ecu_idx >= self.num_ecus:
                    break
                    
                # Calculate grid position
                x_pos = self.ECU_X_MIN + c * x_step
                y_pos = self.ECU_Y_MAX - r * y_step # Top to bottom
                location = Point(x_pos, y_pos)
                
                # Get pre-assigned ECU type
                selected_type = ecu_type_list[ecu_idx]
                
                ecu = CandidateECU(
                    id=f"ECU_{ecu_idx}_{selected_type['type']}",
                    cpu_cap=selected_type['cpu_cap'],
                    ram_cap=selected_type['ram_cap'],
                    rom_cap=selected_type['rom_cap'],
                    max_containers=selected_type['max_containers'],
                    cost=selected_type['cost'],
                    type=selected_type['type'],
                    asil_level=selected_type['asil_level'],
                    hw_offered=selected_type['hw_offered'],
                    interface_offered=selected_type['interface_offered'],
                    location=location
                )
                self.ecus.append(ecu)
                ecu_idx += 1

    def generate_data(self):
        self.generate_sensors()
        self.generate_actuators()
        self.generate_scs()
        self.assign_redundancy()
        self.assign_sensors_actuators()
        self.generate_comm_matrix()
        self.generate_ecu()
        return self.ecus, self.scs, self.comm_matrix, self.sensors, self.actuators, self.cable_types
