import random
from models import Point, Sensor, Actuator, SoftwareComponent, CandidateECU


class VehicleDataGenerator:
    # Vehicle dimensions (2D) - in meters
    VEHICLE_LENGTH = 4.5  # Front to rear
    VEHICLE_WIDTH = 1.8   # Left to right
    
    # ECU placement area (much larger than vehicle)
    ECU_AREA_EXTENT = 50  # Coordinates from -50 to +50

    def __init__(self, num_ecus=10, num_scs=20, seed=42):
        random.seed(seed)
        self.num_ecus = num_ecus
        self.num_scs = num_scs
        self.sensors = []
        self.actuators = []
        self.ecus = []
        self.scs = []

    def generate_sensors(self):
        """Generate sensors with fixed locations based on vehicle layout diagram"""
        # Normalized to vehicle dimensions
        # X: -VEHICLE_WIDTH/2 to +VEHICLE_WIDTH/2 (left to right)
        # Y: -VEHICLE_LENGTH/2 to +VEHICLE_LENGTH/2 (rear to front)
        hw = self.VEHICLE_WIDTH / 2
        hl = self.VEHICLE_LENGTH / 2
        
        sensors_config = [
            # Front sensors
            ('CAMERA', 'CAM_Rear_Left_Corner', 'ETH', 80.0, Point(-0.9 * hw, -0.80 * hl)),
            ('CAMERA', 'CAM_Front_Center', 'ETH', 80.0, Point(0.0, 0.5 * hl)),
            ('CAMERA', 'CAM_Rear_Right_Corner', 'ETH', 80.0, Point(0.9 * hw, -0.80 * hl)),
            ('CAMERA', 'CAM_Front_Left', 'ETH', 80.0, Point(-0.90 * hw, 0.45 * hl)),
            ('CAMERA', 'CAM_Front_Right', 'ETH', 80.0, Point(0.90 * hw, 0.45 * hl)),
            
            # Side sensors
            ('CAMERA', 'CAM_Left_Side', 'ETH', 80.0, Point(-0.90 * hw, 0.0)),
            ('CAMERA', 'CAM_Right_Side', 'ETH', 80.0, Point(0.90 * hw, 0.0)),
            ('CAMERA', 'CAM_Rear_Center', 'ETH', 80.0, Point(0.0, -0.85 * hl)),
            
            # LIDAR sensors (5 total)
            ('LIDAR', 'LIDAR_Front', 'ETH', 15.0, Point(-0.0 * hw, 0.95 * hl)),
            ('LIDAR', 'LIDAR_Left_Pillar', 'ETH', 15.0, Point(-0.90 * hw, 0.60 * hl)),
            ('LIDAR', 'LIDAR_Right_Pillar', 'ETH', 15.0, Point(0.90 * hw, 0.60 * hl)),
            ('LIDAR', 'LIDAR_Rear', 'ETH', 15.0, Point(0.0 * hw, -0.95 * hl)),
            ('LIDAR', 'LIDAR_Top', 'ETH', 15.0, Point(0.0, 0.25 * hl)),
            
            # Center sensors (IMU + GPS)
            ('IMU', 'IMU_Top_Center', 'CAN', 1.0, Point(0.0, -0.30 * hl)),
            ('GPS', 'GPS_Center', 'ETH', 2.0, Point(0.0, -0.4 * hl)),
        ]
        
        for idx, (s_type, name, interface, volume, location) in enumerate(sensors_config):
            sensor = Sensor(
                id=name,
                type=s_type,
                interface=interface,
                volume=volume,
                location=location
            )
            self.sensors.append(sensor)

    def generate_actuators(self):
        """Generate actuators with fixed locations on the vehicle"""
        hw = self.VEHICLE_WIDTH / 2
        hl = self.VEHICLE_LENGTH / 2
        
        actuators_config = [
            # Drivetrain and motion control actuators (front)
            ('STEERING', 'ACT_Steering', 'FLEXRAY', 1.0, Point(0.5*hw, 0.80 * hl)),
            ('MOTOR', 'ACT_Motor_Front', 'CAN', 3.0, Point(0.0, 0.80 * hl)),
            
            # Brake system (distributed)
            ('BRAKE', 'ACT_Brake_Front', 'CAN', 2.0, Point(0.4*hw, 0.85 * hl)),
            ('BRAKE', 'ACT_Brake_Rear', 'CAN', 2.0, Point(0.4*hw, -0.85 * hl)),
            
            # Comfort and HVAC (center cabin)
            ('HVAC', 'ACT_HVAC_Cabin', 'LIN', 0.5, Point(0.0, 0.70 * hl)),
            
            # Lighting actuators (2 total)
            ('LIGHT', 'ACT_Light_Front', 'LIN', 0.2, Point(0.6*hw, 0.85 * hl)),
            ('LIGHT', 'ACT_Light_Rear', 'LIN', 0.2, Point(0.6*hw, -0.85 * hl)),
        ]
        
        for idx, (a_type, name, interface, volume, location) in enumerate(actuators_config):
            actuator = Actuator(
                id=name,
                type=a_type,
                interface=interface,
                volume=volume,
                location=location
            )
            self.actuators.append(actuator)

    def generate_scs(self,weights = [0.15, 0.15, 0.20, 0.35, 0.15]):
        domains = ["ADAS", "Infotainment", "VehicleDynamics", "BodyComfort", "Connectivity"]
        
        # Domain-specific HW requirements with probability
        # Format: {domain: {probability: requirement_list}}
        domain_hw_config = {
            "ADAS": {
                0.60: ['HW_ACC', 'HW_ETH'],      # 60% need specific HW
                0.30: ['HW_ACC'],                 # 30% need just ACC
                0.10: []                          # 10% no specific HW
            },
            "Infotainment": {
                0.40: ['HW_ETH'],                 # 40% need ETH
                0.20: ['HW_BT'],                  # 20% need BT
                0.40: []                          # 40% no specific HW
            },
            "VehicleDynamics": {
                0.60: ['HW_CANFD'],               # 60% need CANFD (critical)
                0.40: []                          # 20% no specific HW
            },
            "BodyComfort": {
                0.30: ['HW_LIN'],                 # 50% need LIN
                0.30: ['HW_CANFD'],               # 30% need CANFD
                0.40: []                          # 20% no specific HW
            },
            "Connectivity": {
                0.20: ['HW_ETH'],                 # 50% need ETH
                0.80: []                          # 50% no specific HW
            }
        }

        assigned_domains = random.choices(domains, weights=weights, k=self.num_scs)

        for i, domain in enumerate(assigned_domains):
            # Probabilistically assign HW requirements
            config = domain_hw_config.get(domain, {0.5: []})
            rand = random.random()
            cumulative = 0
            hw_req = []
            for prob, hw_list in sorted(config.items(), reverse=True):
                cumulative += prob
                if rand < cumulative:
                    hw_req = hw_list
                    break
            
            if domain == "ADAS":
                sc = SoftwareComponent(
                    id=f"SC_{i}_ADAS", domain=domain,
                    cpu_req=random.randint(15000, 40000),
                    ram_req=random.randint(1000, 3000),
                    rom_req=random.randint(50, 200),
                    asil_req=random.choice([3, 4]),  # ASIL-C (3) ve ASIL-D (4)
                    hw_required=hw_req
                )
            elif domain == "Infotainment":
                sc = SoftwareComponent(
                    id=f"SC_{i}_INF", domain=domain,
                    cpu_req=random.randint(10000, 30000),
                    ram_req=random.randint(1000, 4000),
                    rom_req=random.randint(20, 100),
                    asil_req=random.choice([0, 1]),  # ASIL-QM, ASIL-A
                    hw_required=hw_req
                )
            elif domain == "VehicleDynamics":
                sc = SoftwareComponent(
                    id=f"SC_{i}_DYN", domain=domain,
                    cpu_req=random.randint(1000, 5000),
                    ram_req=random.randint(256, 512),
                    rom_req=random.randint(10, 50),
                    asil_req=4,  # ASIL-D
                    hw_required=hw_req
                )
            elif domain == "BodyComfort":
                sc = SoftwareComponent(
                    id=f"SC_{i}_BODY", domain=domain,
                    cpu_req=random.randint(200, 1000),
                    ram_req=random.randint(64, 256),
                    rom_req=random.randint(5, 20),
                    asil_req=random.choice([0, 1, 2]),  # ASIL-QM, ASIL-A, ASIL-B
                    hw_required=hw_req
                )
            elif domain == "Connectivity":
                sc = SoftwareComponent(
                    id=f"SC_{i}_CONN", domain=domain,
                    cpu_req=random.randint(3000, 8000),
                    ram_req=random.randint(256, 1024),
                    rom_req=random.randint(50, 200),
                    asil_req=random.choice([0, 1, 2]),  # Bağlantı kritik değil
                    hw_required=hw_req
                )
            
            self.scs.append(sc)
   
    def assign_sensors_actuators(self):
        """
        Assign sensors/actuators to SCs based on domain-specific logic.
        """
        # Access probability by domain
        access_probability = {
            "ADAS": 0.30,
            "VehicleDynamics": 0.70,
            "BodyComfort": 0.70,
            "Infotainment": 0.10,
            "Connectivity": 0.05
        }
        
        # Sensor type assignments by domain (if SC is selected for access)
        sensor_type_assignments = {
            "ADAS": ["CAMERA", "LIDAR", "RADAR"],
            "VehicleDynamics": ["RADAR"],
            "BodyComfort": ["LIDAR"],
            "Infotainment": ["CAMERA"],  # Rare, but possible
            "Connectivity": []
        }
        
        # Actuator type assignments by domain (if SC is selected for access)
        actuator_type_assignments = {
            "ADAS": ["BRAKE", "STEERING"],
            "VehicleDynamics": ["BRAKE", "STEERING", "MOTOR"],
            "BodyComfort": ["HVAC", "LIGHT"],
            "Infotainment": ["LIGHT"],
            "Connectivity": []
        }
        
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
        comm_matrix = []
        for i, src in enumerate(self.scs):
            for j, dst in enumerate(self.scs):
                if i != j and random.random() < 0.3:
                    volume = random.randint(1, 50)
                    if src.domain == dst.domain:
                        max_latency = random.randint(1, 10)
                    else:
                        max_latency = random.randint(10, 100)
                    comm_matrix.append({
                        'src': src.id,
                        'dst': dst.id,
                        'volume': volume,
                        'max_latency': max_latency,
                    })
        return comm_matrix
    
    def generate_ecu(self):
        """Generate ECUs strictly INSIDE the vehicle dimensions (X: ±0.8m, Y: ±2.0m)"""
        hpc_hw = ['HW_ACC', 'HW_ETH', 'HW_HSM']
        zone_hw = ['HW_CANFD', 'HW_FLEX', 'HW_ETH', 'HW_LIN',"HW_BT"]
        mcu_hw = ['HW_CANFD', 'HW_LIN']
        
        # Place ECUs in a grid pattern strictly inside the vehicle
        # Vehicle is approx 1.8m wide (X: -0.9 to 0.9) and 4.5m long (Y: -2.25 to 2.25)
        # We stay within safe margins: X: -0.8 to 0.8, Y: -2.0 to 2.0
        
        cols = int(self.num_ecus ** 0.5) 
        rows = (self.num_ecus + cols - 1) // cols
        
        x_min, x_max = -0.7, 0.7 
        y_min, y_max = -2.0, 2.0
        
        x_step = (x_max - x_min) / max(1, cols - 1) if cols > 1 else 0
        y_step = (y_max - y_min) / max(1, rows - 1) if rows > 1 else 0

        ecu_idx = 0
        for r in range(rows):
            for c in range(cols):
                if ecu_idx >= self.num_ecus:
                    break
                    
                # Calculate grid position inside vehicle
                x_pos = x_min + c * x_step
                y_pos = y_max - r * y_step # Top to bottom
                
                location = Point(x_pos, y_pos)
                
                r_val = random.random()
                
                r = random.random()
                
                if r < 0.10:
                    ecu = CandidateECU(
                        id=f"ECU_{ecu_idx}_HPC",
                        cpu_cap=200000,
                        ram_cap=64000,
                        rom_cap=256000,
                        max_containers=16,
                        cost=1000,
                        type='HPC',
                        asil_level=4,
                        hw_offered=hpc_hw,
                        location=location
                    )
                elif r < 0.40:
                    ecu = CandidateECU(
                        id=f"ECU_{ecu_idx}_SAFE",
                        cpu_cap=30000,
                        ram_cap=4000,
                        rom_cap=8000,
                        max_containers=4,
                        cost=300,
                        type='ZONE',
                        asil_level=4,
                        hw_offered=zone_hw,
                        location=location
                    )
                else:
                    ecu = CandidateECU(
                        id=f"ECU_{ecu_idx}_MCU",
                        cpu_cap=5000,
                        ram_cap=512,
                        rom_cap=2048,
                        max_containers=1,
                        cost=50,
                        type='MCU',
                        asil_level=3,
                        hw_offered=mcu_hw,
                        location=location
                    )
                self.ecus.append(ecu)
                ecu_idx += 1

    def generate_data(self):
        self.generate_sensors()
        self.generate_actuators()
        self.generate_scs()
        self.assign_sensors_actuators()
        comm_matrix = self.generate_comm_matrix()

        self.generate_ecu()
        return self.ecus, self.scs, comm_matrix, self.sensors, self.actuators
