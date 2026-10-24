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
    z: float = 0.0
    ref: str = None  # id of the site / peripheral this point belongs to (key of the cable distance table)

    def dist(self, other):
        """Calculate Manhattan and Euclidean distances (3D)."""
        dx, dy, dz = abs(self.x - other.x), abs(self.y - other.y), abs(self.z - other.z)
        return dx + dy + dz, math.sqrt(dx * dx + dy * dy + dz * dz)

@dataclass
class Location:
    """Represents a physical location in the vehicle where components can be deployed."""
    id: str        # e.g., 'dash_center', 'trunk_side_left_1'
    location: 'Point' = None
    health_factor: float = 1.0   # capacity derating of the site's thermal class (used by --derating)
    zone: str = None
    thermal_class: str = None

@dataclass
class Interface:
    name: str  # e.g., 'CAN', 'ETH'
    cost_per_meter: float
    latency_per_meter: float  # microseconds per meter
    weight_per_meter: float   # grams per meter
    capacity: float = float('inf')  # traffic capacity in the same units as comm_matrix[*]['volume']
    port_cost: float = 0.0    # cost per port/interface on ECU or SC
    hop_latency_ms: float = 0.0  # per-hop processing/queueing bound of a link of this type (switch forwarding, bus arbitration)
    frame_kbit: float = float('inf')  # longest frame (kbit): what a lower-priority message can block a higher one with

@dataclass
class Sensor:
    id: str
    type: str
    interface: str
    volume: float
    location: 'Point' = None
    rate_hz: float = 10.0      # message rate; message size (kbit) = volume (Mbit/s) * 1000 / rate_hz

@dataclass
class Actuator:
    id: str
    type: str
    interface: str
    volume: float
    location: 'Point' = None
    rate_hz: float = 10.0      # message rate; message size (kbit) = volume (Mbit/s) * 1000 / rate_hz

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
    rate_hz: float = 10.0       # output message rate; message size (kbit) = link volume (Mbit/s) * 1000 / rate_hz
    msg_kbit: float = None      # size of one output message (kbit); None = default link volume or the device stream
    tops_req: float = 0.0       # accelerator demand (TOPS, rated-equivalent) of a component that needs a TOPS feature (HW_ACC)
    fail_operational: bool = None  # must keep working after a single failure (standby); None = ASIL rule of uncertainty.json

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
