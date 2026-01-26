import json
import os
from models import CableType


class ConfigReader:
    """Reads and parses vehicle configuration from JSON file"""
    
    def __init__(self, config_path):
        """
        Initialize config reader
        
        Args:
            config_path: Path to the configuration JSON file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def get_vehicle_dimensions(self):
        """Get vehicle dimensions"""
        return self.config['vehicle_dimensions']
    
    def get_cable_types(self, base_cost_per_meter=10.0):
        """
        Get cable types with cost calculation
        
        Args:
            base_cost_per_meter: Base cost multiplier
            
        Returns:
            Dictionary of cable types
        """
        cable_types = {}
        for name, specs in self.config['cable_types'].items():
            cable_types[name] = CableType(
                name=name,
                cost_per_meter=base_cost_per_meter * specs['cost_multiplier'],
                latency_per_meter=specs['latency_per_meter'],
                weight_per_meter=specs['weight_per_meter']
            )
        return cable_types
    
    def get_sensors_config(self):
        """Get sensors configuration"""
        return self.config['sensors']
    
    def get_actuators_config(self):
        """Get actuators configuration"""
        return self.config['actuators']
    
    def get_full_config(self):
        """Get full configuration dictionary"""
        return self.config
