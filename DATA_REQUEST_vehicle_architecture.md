# Data I need from a real vehicle

I am building a tool that decides where the vehicle software runs (which ECU or computer), which hardware is used, how the network and the redundancy look, and what the whole thing costs. For that I need the following information from a real car. Anonymised IDs, ranges, percentages or relative numbers are fine everywhere.

## 1. Software components

- I need the number of software components in the whole vehicle, split by domain (body/comfort, occupant safety, chassis, powertrain, ADAS/AD, infotainment/HMI, connectivity/telematics, platform/diagnostics).
- I need to know what you count as one component (an AUTOSAR SW-C, a runnable, an Adaptive application, a Linux/Android process).
- I need the overall distributions over all components, as percentages:
  - the share of components per ASIL (QM, A, B, C, D);
  - the share of components per activation period (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 ms, event-triggered, angle-synchronous);
  - the distribution of execution time per activation (min, typical, max in µs) per period class;
  - the distribution of RAM and ROM per component (min, typical, max) per ASIL;
- For each software component I need:
  - the domain and the function it belongs to;
  - the ASIL (QM, A, B, C, D);
  - the activation: the period in ms, or event-triggered, or angle-synchronous;
  - the execution time per activation, worst case and average in µs, and the core type and clock it was measured on;
  - the CPU demand, as a share of a core or in DMIPS, if you specify it that way;
  - the RAM and the ROM/flash it needs (kB or MB);
  - the hardware it requires (GPU/NPU, DSP, HSM, lockstep core, special I/O);
  - the placement constraints (it must stay near a sensor or actuator, it must not share an ECU or a core with certain other components, it is a supplier black box that cannot be split).
- I need the list of function clusters that exist in the car and the number of components in each one.

## 2. Communication

- I need the list of signals or messages between components or between ECUs, with for each one:
  - the sender and the receiver(s);
  - the payload size in bytes;
  - the cycle time or the rate in Hz;
  - the bus or protocol (CAN FD, LIN, FlexRay, Ethernet SOME/IP, DDS, PCIe);
  - the priority or the message ID;
  - the latency requirement, if there is one.
- I need the overall distributions as well: the share of signals per bus type, the distribution of payload sizes and the distribution of cycle times.
- I need the total number of signals exchanged between ECUs.
- I need the bus loads that are actually reached on each bus, in percent.
- For the Ethernet backbone I need the link speeds, the number of switches, the TSN or priority classes in use and the per-hop latency budget.

## 3. Timing chains and safety times

- I need the end-to-end chains that have a timing requirement (sensor -> processing -> actuator), with for each chain:
  - the components in order;
  - the end-to-end latency bound in ms;
  - the data-age bound, if it differs;
  - whether the value is a requirement or a measurement.
- I need the distribution of chain lengths (number of components per chain) and of chain bounds, as percentages or ranges, if the full list cannot be shared.
- I need the FTTI (fault tolerant time interval) per hazard from the HARA, in ms, together with the ASIL of the hazard, for steering, braking, propulsion / unintended torque, airbag, battery management, lighting, AEB, ACC, lane keeping and automated driving.
- I need the fail-operational requirements:
  - which functions must keep working after a single ECU, zone, link or power failure;
  - the allowed switch-over time in ms;
  - which functions may be lost, and whether there is a limit on how many functions one failure may take down.

## 4. Sensors and actuators

- I need the sensor list with the type, the count, the interface (GMSL/CSI-2, Ethernet, CAN FD, LIN, analogue), the data rate in Mbit/s, the update rate in Hz, the ASIL, and the unit cost.
- I need the actuator list with the type, the interface and bus, the command rate, the ASIL, and the unit cost.
- I need to know where the driver or preprocessing software of each sensor and actuator runs today (in the smart sensor, in a zone controller, in a central computer).

## 5. Hardware and topology

- I need the current topology: the number and type of central computers, zone or domain controllers and small ECUs, and which domains run on which of them.
- For each compute class (central computer, zone controller, small ECU) I need:
  - the SoC or MCU, the CPU cores and clock, and the DMIPS or another performance figure;
  - the GPU/NPU performance in TOPS;
  - the RAM and the flash;
  - the number of CAN FD, LIN, FlexRay and Ethernet ports with their speeds, and PCIe;
  - the ASIL capability, the power consumption and the temperature grade;
  - how many partitions, VMs or cores can be opened on it, and at which ASIL.
- I need the design limits you use: the maximum CPU load per core, the maximum bus load, the maximum memory fill, and the headroom reserved for software updates, all in percent.
- I need the harness data: the cable types in use, the cost and the weight per metre, the connector cost, and the typical cable lengths between zones and central computers.
- I need the temperature classes per vehicle area (engine bay, cabin, roof, trunk, wheel arch) and the derating applied to compute units there.

## 6. Costs

- I need the unit cost per compute class (ECU, or SoC plus board).
- I need the cost per port, per cable metre and per installation place of an ECU (bracket, housing, power supply).
- I need the fixed cost of adding a new ECU to the vehicle.
- I need the weight of the compute units and of the harness, if weight matters in the design.

## 7. Growth and margins

- I need the planned software growth over the vehicle lifetime (OTA), as percent per year or as the headroom at start of production.
- I need the margins between the average demand and the design value (worst / mean ratios), per ASIL if they differ.
- I need the safety factor applied to the network load.
- I need the planned variation of AI inference times.

## 8. A reference design

- I need one real allocation: which component runs on which ECU or core today, its cost or cost split, the measured end-to-end latencies of a few chains, and the measured bus and CPU loads.
