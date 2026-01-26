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
                location=location
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
                location=location
            )
            self.actuators.append(actuator)

    def generate_scs(self, weights=None):
        """Generate software components with domain distribution from config"""
        if weights is None:
            weights = self.sc_domain_weights
        
        domains = ["ADAS", "Infotainment", "VehicleDynamics", "BodyComfort", "Connectivity"]
        
        # Get domain configurations from config
        domain_configs = self.config_reader.get_sc_domain_configs()

        assigned_domains = random.choices(domains, weights=weights, k=self.num_scs)

        for i, domain in enumerate(assigned_domains):
            domain_config = domain_configs.get(domain)
            if not domain_config:
                continue
            
            # Probabilistically assign HW requirements
            hw_requirements = domain_config.get('hw_requirements', {})
            rand = random.random()
            cumulative = 0
            hw_req = []
            for prob_str, hw_list in sorted(hw_requirements.items(), key=lambda x: float(x[0]), reverse=True):
                cumulative += float(prob_str)
                if rand < cumulative:
                    hw_req = hw_list
                    break
            
            # Get resource requirements from config
            cpu_range = domain_config.get('cpu_range', [1000, 5000])
            ram_range = domain_config.get('ram_range', [256, 1024])
            rom_range = domain_config.get('rom_range', [10, 50])
            asil_levels = domain_config.get('asil_levels', [0])
            
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
   
    def assign_sensors_actuators(self):
        """
        Assign sensors/actuators to SCs based on domain-specific logic from config.
        """
        # Get assignment config from config reader
        assignment_config = self.config_reader.get_sc_sensor_actuator_assignments()
        
        access_probability = assignment_config.get('access_probability', {})
        sensor_type_assignments = assignment_config.get('sensor_types', {})
        actuator_type_assignments = assignment_config.get('actuator_types', {})
        
        for sc in self.scs:
            # Determine if this SC gets sensor/actuator access
            prob = access_probability.get(sc.domain, 0.0)
            if random.random() < prob:
                # Assign sensors by type - all sensors of allowed types
                domain_sensor_types = sensor_type_assignments.get(sc.domain, [])
                sc.sensors = [s.id for s in self.sensors if s.type in domain_sensor_types]
                
                # Assign actuators by type - all actuators of allowed types
                domain_actuator_types = actuator_type_assignments.get(sc.domain, [])
                sc.actuators = [a.id for a in self.actuators if a.type in domain_actuator_types]
            else:
                # No access
                sc.sensors = []
                sc.actuators = []

    def generate_comm_matrix(self):
        """Generate communication matrix between SCs based on random probabilities"""
        for i, src in enumerate(self.scs):
            for j, dst in enumerate(self.scs):
                if i != j and random.random() < 0.3:
                    volume = random.randint(1, 50)
                    if src.domain == dst.domain:
                        max_latency = random.randint(1, 10)
                    else:
                        max_latency = random.randint(10, 100)
                    self.comm_matrix.append({
                        'src': src.id,
                        'dst': dst.id,
                        'volume': volume,
                        'max_latency': max_latency,
                    })
    
    def generate_ecu(self):
        """Generate ECUs using types and placement bounds from configuration"""
        ecu_types = self.config_reader.get_ecu_types_config()
        
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
                
                # Select ECU type based on probability
                rand_val = random.random()
                cumulative_prob = 0
                selected_type = None
                
                for ecu_type_config in ecu_types:
                    cumulative_prob += ecu_type_config['probability']
                    if rand_val < cumulative_prob:
                        selected_type = ecu_type_config
                        break
                
                # Fallback to last type if not selected
                if selected_type is None:
                    selected_type = ecu_types[-1]
                
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
                    location=location
                )
                self.ecus.append(ecu)
                ecu_idx += 1

    def generate_data(self):
        self.generate_sensors()
        self.generate_actuators()
        self.generate_scs()
        self.assign_sensors_actuators()
        self.generate_comm_matrix()
        self.generate_ecu()
        return self.ecus, self.scs, self.comm_matrix, self.sensors, self.actuators, self.cable_types
