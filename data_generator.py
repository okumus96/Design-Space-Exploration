import random
from models import Point, Location, Sensor, Actuator, SoftwareComponent, CandidateECU, Interface


class VehicleDataGenerator:
    def __init__(self, num_ecus=10, num_locs=20, num_scs=20, seed=42, config_reader=None):
        random.seed(seed)
        self.num_ecus = num_ecus
        self.num_scs = num_scs
        self.num_locs = num_locs
        self.sensors = []
        self.actuators = []
        self.ecus = []
        self.scs = []
        self.comm_matrix = []
        self.locations = []

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
        self.cable_types = self.config_reader.get_interfaces()
        
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
                max_latency=sensor_config.get('max_latency', random.randint(5, 100))
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
                max_latency=actuator_config.get('max_latency', random.randint(5, 100))
            )
            self.actuators.append(actuator)

    def generate_scs(self, weights=None):
        """
        Generate software components with domain distribution from config.
        
        Note: self.num_scs = number of OTHER DOMAIN SCs (not including Driver)
              Driver SCs = len(sensors) + len(actuators) = 22 (fixed)
              Total SCs = self.num_scs + 22
        """
        # First, generate Driver SCs for each sensor and actuator (1:1 mapping)
        num_sensors = len(self.sensors)
        num_actuators = len(self.actuators)
        num_driver_scs = num_sensors + num_actuators
        
        sc_index = 0
        
        # Create Driver domain SCs for sensors
        domain_config = self.config_reader.get_sc_domain_configs().get('Driver', {})
        if not domain_config:
            domain_config = {
                'cpu_range': [100, 500],
                'ram_range': [1000, 5000],
                'rom_range': [5000, 10000],
                'asil_levels': [0, 1],
                'hw_requirements': {}
            }
        
        for sensor in self.sensors:
            sc = SoftwareComponent(
                id=f"SC_{sc_index}_Driver_{sensor.id}",
                domain='Driver',
                cpu_req=random.randint(domain_config['cpu_range'][0], domain_config['cpu_range'][1]),
                ram_req=random.randint(domain_config['ram_range'][0], domain_config['ram_range'][1]),
                rom_req=random.randint(domain_config['rom_range'][0], domain_config['rom_range'][1]),
                asil_req=random.choice(domain_config.get('asil_levels', [0, 1])),
                hw_required=[]
            )
            sc.sensors = [sensor.id]
            sc.actuators = []
            sc.interface_required = [sensor.interface]
            self.scs.append(sc)
            sc_index += 1
        
        # Create Driver domain SCs for actuators
        for actuator in self.actuators:
            sc = SoftwareComponent(
                id=f"SC_{sc_index}_Driver_{actuator.id}",
                domain='Driver',
                cpu_req=random.randint(domain_config['cpu_range'][0], domain_config['cpu_range'][1]),
                ram_req=random.randint(domain_config['ram_range'][0], domain_config['ram_range'][1]),
                rom_req=random.randint(domain_config['rom_range'][0], domain_config['rom_range'][1]),
                asil_req=random.choice(domain_config.get('asil_levels', [0, 1])),
                hw_required=[]
            )
            sc.sensors = []
            sc.actuators = [actuator.id]
            sc.interface_required = [actuator.interface]
            self.scs.append(sc)
            sc_index += 1
        
        # Now generate OTHER domain SCs (self.num_scs = count of OTHER domain SCs)
        if weights is None:
            weights = self.sc_domain_weights
        
        # Get domains from config (excluding Driver)
        domains = [d for d in self.config_reader.get_domains() if d != 'Driver']
        
        # Get domain configurations from config
        domain_configs = self.config_reader.get_sc_domain_configs()
        
        # Number of SCs to create for other domains (self.num_scs is already the count for other domains)
        remaining_scs = self.num_scs

        # Calculate exact count for each domain based on weights
        domain_counts = {}
        remaining = remaining_scs
        
        for i, domain in enumerate(domains):
            if i < len(domains) - 1:
                count = round(weights[i] * remaining_scs)
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
        
        # Track index per domain for HW assignment
        domain_indices = {domain: 0 for domain in domains}

        for domain in assigned_domains:
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
                id=f"SC_{sc_index}_{domain[:4].upper()}",
                domain=domain,
                cpu_req=random.randint(cpu_range[0], cpu_range[1]),
                ram_req=random.randint(ram_range[0], ram_range[1]),
                rom_req=random.randint(rom_range[0], rom_range[1]),
                asil_req=random.choice(asil_levels),
                hw_required=hw_req
            )
            
            self.scs.append(sc)
            sc_index += 1
    
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
        Assign sensors/actuators to SCs.
        
        Driver SCs already have 1:1 device mapping from generate_scs().
        Other SCs (non-Driver domains) are left without devices unless config specifies otherwise.
        """
        # Driver SCs (indices 0 to num_sensors+num_actuators-1) already have 
        # sensors/actuators assigned during SCgeneration.
        # Non-Driver SCs start with empty sensors/actuators lists.
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
        """
        Generate communication matrix between SCs based on random probabilities.
        
        Special case: Driver SWs (sensors/actuators) should NOT communicate with each other.
        They only communicate with application SWs (ADAS, INFO, etc.)
        """
        # Get communication probabilities from config
        comm_config = self.config_reader.get_sc_communication_config()
        
        intra_prob = comm_config.get('intra_domain_probability')
        inter_prob = comm_config.get('inter_domain_probability')
        
        for i, src in enumerate(self.scs):
            for j, dst in enumerate(self.scs):
                if i != j:
                    # Filter: Driver-to-Driver communication is NOT allowed
                    if src.domain == 'Driver' and dst.domain == 'Driver':
                        continue
                    
                    # Use intra-domain probability if both are same domain (and not Driver)
                    # Use inter-domain probability otherwise
                    prob = intra_prob if src.domain == dst.domain else inter_prob                 
                    if random.random() < prob:
                        volume = random.randint(1, 50)
                        max_latency = random.uniform(0.2, 0.5) #random.randint(10, 500)
                        self.comm_matrix.append({
                            'src': src.id,
                            'dst': dst.id,
                            'volume': volume,
                            'max_latency': max_latency,
                        })

    def generate_locations(self):
        """Generate candidate locations (with id and Point coordinates)."""
        num_locations = self.num_locs

        cols = int(num_locations ** 0.5)
        rows = (num_locations + cols - 1) // cols

        x_step = (self.ECU_X_MAX - self.ECU_X_MIN) / max(1, cols - 1) if cols > 1 else 0
        y_step = (self.ECU_Y_MAX - self.ECU_Y_MIN) / max(1, rows - 1) if rows > 1 else 0

        self.locations = []
        loc_idx = 0
        for r in range(rows):
            for c in range(cols):
                if loc_idx >= num_locations:
                    break

                x_pos = self.ECU_X_MIN + c * x_step
                y_pos = self.ECU_Y_MAX - r * y_step  # Top to bottom
                self.locations.append(Location(id=f"LOC{loc_idx}", location=Point(x_pos, y_pos)))
                loc_idx += 1
    
    def generate_ecu(self):
        """Generate ECUs: One of each type at each candidate location to avoid placement bias."""
        ecu_types = self.config_reader.get_ecu_types_config()
        
        # We treat 'self.num_ecus' as the number of 'Candidate Sites/Locations'
        num_locations = self.num_ecus
        
        # Place Locations in a grid pattern using config bounds
        cols = int(num_locations ** 0.5) 
        rows = (num_locations + cols - 1) // cols
        
        x_step = (self.ECU_X_MAX - self.ECU_X_MIN) / max(1, cols - 1) if cols > 1 else 0
        y_step = (self.ECU_Y_MAX - self.ECU_Y_MIN) / max(1, rows - 1) if rows > 1 else 0

        self.ecus = []
        loc_idx = 0
        for r in range(rows):
            for c in range(cols):
                if loc_idx >= num_locations:
                    break
                    
                # Calculate grid position
                x_pos = self.ECU_X_MIN + c * x_step
                y_pos = self.ECU_Y_MAX - r * y_step # Top to bottom
                location = Point(x_pos, y_pos)
                
                # At this location, generate ONE of EACH ECU type
                for type_config in ecu_types:
                    # Create unique ID: ECU_L{loc_idx}_{Type}
                    ecu_id = f"ECU_L{loc_idx}_{type_config['type']}"
                    
                    ecu = CandidateECU(
                        id=ecu_id,
                        cpu_cap=type_config['cpu_cap'],
                        ram_cap=type_config['ram_cap'],
                        rom_cap=type_config['rom_cap'],
                        max_partitions=type_config['max_partitions'],
                        cost=type_config['cost'],
                        type=type_config['type'],
                        asil_level=type_config['asil_level'],
                        hw_offered=type_config['hw_offered'],
                        interface_offered=type_config['interface_offered'],
                        location=location,
                        system_overhead_percent=type_config.get('system_overhead_percent', 0.0),
                        partition_reservation_percent=type_config.get('partition_reservation_percent', 0.0)
                    )
                    self.ecus.append(ecu)
                
                loc_idx += 1

    def generate_data(self):
        self.generate_sensors()
        self.generate_actuators()
        self.generate_scs()
        self.assign_redundancy()
        self.assign_sensors_actuators()
        self.generate_comm_matrix()
        self.generate_locations()
        # ECU generation removed: LEGO model uses locations directly
        return self.scs, self.comm_matrix, self.sensors, self.actuators, self.cable_types, self.locations
