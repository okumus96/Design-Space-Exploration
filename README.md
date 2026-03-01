# Uncertainty-Aware Design Space Exploration for Software-Defined Vehicles

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code and experimental datasets for the automated Design Space Exploration (DSE) of Automotive Electronic/Electrical (E/E) Architectures. The framework performs **Multi-Objective Mixed-Integer Linear Programming (MILP)** optimization to assign Software Components (SCs) to edge devices/Electronic Control Units (ECUs) across the spatial vehicle topology. 

It generates Pareto-optimal architectures by simultaneously minimizing:
1. **Total Deployment Cost** (Hardware utilization)
2. **Total Cable Length** (Wiring harness footprint)

> **Note:** This repository is intended as a supplementary artifact for academic peer review and subsequent publication. Please see the [Citation](#-citation) section for referencing this work.

---

## Key Features

* **Multi-Objective Optimization:** Evaluates the mathematically precise trade-offs between ECU hardware cost and physical cable layout footprint (extrapolating the Pareto capability frontier).
* **Advanced Constraint Satisfaction:** Incorporates real-world automotive constraints:
  * Hardware Capacities (Compute, Memory, Hardware Accelerators like DSP/NPU)
  * System Reliability (Redundancy Monitors ensuring fail-operational states)
  * Network Topologies (Domains, Zones, Central Compute boundaries)
* **Uncertainty & Robustness:** Integrates configuration parameters reflecting predictive AI contention thresholds and safety margins (`--uncertainty`).
* **Automated Publication-ready Visualizations:** Automatically generates spatial architecture diagrams, Pareto curves, and top-down hierarchical layout PDFs specifically tailored for IEEE/ACM publication standards (`--latex_topdown`).

---

## Repository Structure

```text
.
├── configs/                          # JSON definitions for vehicle topologies, ECUs, and architectures
│   ├── actuators.json                # Definitions of vehicle actuators and their physical zones
│   ├── buses.json                    # Network buses, bandwidth capacities, and connections
│   ├── full_architecture_*.json      # End-to-end architecture definitions connecting all entities
│   ├── hardwares.json                # Specific hardware accelerators (e.g., DSP, NPU) details
│   ├── partitions.json               # Logical software partitions
│   ├── sensors.json                  # Sensor specifications, data payloads, 
│   ├── software.json                 # Software Component (SC) specs, dependencies
│   └── vehicle.json                  # CVehicle topology and candidate locations
├── optimizer.py                      # Mathematical modeling (Gurobi constraints, objective formulation)
├── main.py                           # Orchestration script (CLI parser, solver invocation, pipeline)
├── architecture_loader.py            # Config parsers for JSON-based architecture specifications
├── visualizer.py / report.py         # Reporting engines for generating diagrams and Pareto graphs
├── results/                          # Auto-generated outputs (execution logs, graphs, raw .dat files)
└── design_space_exploration.sbatch   # SLURM batch file for High-Performance Computing (HPC)
```

---

## Requirements & Installation

### Prerequisites
* **Python 3.10+**: Ensure a modern Python environment (used virtual environments are highly recommended).
* **Gurobi Optimizer**: An active Gurobi license is *strictly required* for the MILP solver. Academic licenses are available for free at [Gurobi's Academic Program](https://www.gurobi.com/academia/academic-program-and-licenses/).

### General Setup

1. **Clone the repository and enter the directory:**
   ```bash
   git clone <repository_url>
   cd Design-Space-Exploration
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Gurobi License:**
   Ensure your license is correctly targeted and accessible to the environment:
   ```bash
   python -c "import gurobipy as gp; print('Gurobi version:', gp.gurobi.version())"
   ```

---

## Usage & Reproduction

### Local Execution (Quick Start)

To test the installation and run a baseline optimization using a compact architecture (generating 5 points on the Pareto front across 6 candidate locations):

```bash
python main.py \
  --num_locs 6 \
  --architecture_json ./configs/full_architecture_v2.json \
  --num_points 5 \
  --uncertainty \
  --time_limit 600 \
  --latex_topdown \
  --topdown_dpi 600 \
  --output_dir results/local_experiment
```

### HPC / SLURM Execution

For exhaustive computations, it is recommended to use a High-Performance Computing (HPC) cluster. We provide a pre-configured SLURM batch script:

```bash
sbatch design_space_exploration.sbatch
```
*Outputs, including standard out and error logs, will be automatically routed. SLURM logs are saved to `results/ecu_dse_<job_id>.txt`, whilst graphical outputs map to `results/experiment_<job_id>/`.*

---

##  Result Artifacts

Upon solver convergence or timing out limits, the pipeline yields analytical artifacts in the specified `--output_dir`:

1. **`pareto_front_analysis.pdf`**: The synthesized Pareto curve demonstrating Cost vs. Cable Length.
2. **`pareto_front_analysis_points.dat`**: The unformatted raw data coordinates corresponding to the Pareto front endpoints.
3. **`solution_architecture_*.png`**: Bipartite schematic representations demonstrating mapped assignments between Software Components and chosen ECUs.
4. **`vehicle_layout_solution_*.pdf`**: A top-down spatial map representing the physical installation footprint of the E/E design within the vehicle topology.
5. **`vehicle_layout_latex_compare_shared_legend.pdf`**: A side-by-side comparative layout graph explicitly designed for double-column publication templates.

---

##  Citation

*(Placeholder — to be updated upon paper publication)*

## License

This project is licensed under the MIT License.
