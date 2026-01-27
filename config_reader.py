import json
import os
from models import CableType


class ConfigReader:
    """Reads and parses vehicle configuration from JSON file"""
    
    def __init__(self, config_dir='configs'):
        """
        Initialize config reader
        
        Args:
            config_dir: Directory containing configuration JSON files
        """
        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
        
        self.config_dir = config_dir
        
        # Load all config files
        self.vehicle_config = self._load_json('vehicle.json')
        self.ecu_types_config = self._load_json('ecu_types.json')
        self.sensors_config = self._load_json('sensors.json')
        self.actuators_config = self._load_json('actuators.json')
        self.software_config = self._load_json('software.json')
        self.bus_config = self._load_json('buses.json')
    
    def _load_json(self, filename):
        """Load a JSON file from config directory"""
        filepath = os.path.join(self.config_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _load_json(self, filename):
        """Load a JSON file from config directory"""
        filepath = os.path.join(self.config_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def get_vehicle_dimensions(self):
        """Get vehicle dimensions"""
        return self.vehicle_config['dimensions']
    
    def get_cable_types(self):
        """
        Get cable types from configuration
        
        Returns:
            Dictionary of cable types
        """
        cable_types = {}
        for name, specs in self.bus_config['cable_types'].items():
            cable_types[name] = CableType(
                name=name,
                cost_per_meter=specs['cost_per_meter'],
                latency_per_meter=specs['latency_per_meter'],
                weight_per_meter=specs['weight_per_meter']
            )
        return cable_types
    
    def get_sensors_config(self):
        """Get sensors configuration"""
        return self.sensors_config['sensors']
    
    def get_actuators_config(self):
        """Get actuators configuration"""
        return self.actuators_config['actuators']
    
    def get_ecu_types_config(self):
        """Get ECU types configuration"""
        return self.ecu_types_config['ecu_types']
    
    def get_sc_domain_weights(self):
        """Get software component domain weights"""
        weights_dict = self.software_config.get('sc_domain_weights')
        # Return as list in the expected order
        domains = list(weights_dict.keys())
        return [weights_dict[d] for d in domains]
    
    def get_domains(self):
        """Get list of software component domains"""
        return list(self.software_config.get('sc_domain_weights', {}).keys())
    
    def get_sc_domain_configs(self):
        """Get software component domain configurations"""
        return self.software_config.get('sc_domain_configs', {})
    
    def get_sc_sensor_actuator_assignments(self):
        """Get sensor/actuator assignment configurations for SCs"""
        return self.software_config.get('sc_sensor_actuator_assignments', {})
    
    def get_sc_communication_config(self):
        """Get software component communication configuration"""
        return self.software_config.get('sc_communication')
