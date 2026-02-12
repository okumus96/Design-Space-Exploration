"""
Data Models for ECU Optimization

Contains all dataclass definitions for the vehicle architecture components.
"""

import math
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float

    def dist(self, other):
        """Calculate Manhattan and Euclidean distances."""
        manhattan = abs(self.x - other.x) + abs(self.y - other.y)
        euclidean = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        return manhattan, euclidean

@dataclass
class Location:
    """Represents a physical location in the vehicle where components can be deployed."""
    id: str        # e.g., 'LOC0', 'LOC1'
    location: 'Point' = None

@dataclass
class Interface:
    name: str  # e.g., 'CAN', 'ETH'
    cost_per_meter: float
    latency_per_meter: float  # microseconds per meter
    weight_per_meter: float   # grams per meter
    capacity: float = float('inf')  # traffic capacity in the same units as comm_matrix[*]['volume']
    port_cost: float = 0.0    # cost per port/interface on ECU or SC

@dataclass
class Sensor:
    id: str
    type: str
    interface: str
    volume: float
    location: 'Point' = None
    max_latency: float = None  # ms

@dataclass
class Actuator:
    id: str
    type: str
    interface: str
    volume: float
    location: 'Point' = None
    max_latency: float = None  # ms

@dataclass
class SoftwareComponent:
    id: str
    domain: str
    cpu_req: int
    ram_req: int
    rom_req: int
    asil_req: int
    hw_required: list
    interface_required: list = None
    sensors: list = None
    actuators: list = None
    redundant_with: str = None  # ID of redundant SC pair

    def __post_init__(self):
        if self.interface_required is None:
            self.interface_required = []
        if self.sensors is None:
            self.sensors = []
        if self.actuators is None:
            self.actuators = []

@dataclass
class CandidateECU:
    id: str
    cpu_cap: int
    ram_cap: int
    rom_cap: int
    max_partitions: int
    cost: int
    type: str
    asil_level: int
    hw_offered: list
    interface_offered: list
    location: 'Point' = None
    system_overhead_percent: float = 0.0      # Base overhead for OS/Hypervisor
    partition_reservation_percent: float = 0.0 # Reserved capacity per partition
