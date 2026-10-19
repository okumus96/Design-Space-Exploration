#!/usr/bin/env python3
"""
ECU Optimization and Visualization Pipeline

Main script that orchestrates the entire workflow:
1. Generate vehicle architecture data (possible locations, SCs, sensors, actuators)
2. Run Gurobi optimization for SW-to-ECU assignment
3. Visualize the optimization results
"""
import argparse
from config_reader import ConfigReader
from data_generator import VehicleDataGenerator
from architecture_loader import load_architecture_from_json
from optimizer import AssignmentOptimizer, UNCERTAINTY_BLOCKS
from visualizer import Visualization
from report import ReportGenerator
from tabulate import tabulate
import time
import os
import sys
import json
import socket
import subprocess
from datetime import datetime


def _git_info():
    """Return (commit hash, dirty flag) of the working tree, or (None, None) if unavailable."""
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_dir, text=True).strip())
        return commit, dirty
    except Exception:
        return None, None


def save_run_results(args, pareto_solutions, pareto_time_s):
    """Save run metadata (run_info.json) and every Pareto solution (solution_<i>.json) to args.output_dir."""
    import gurobipy as gp
    commit, dirty = _git_info()
    run_info = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'command': " ".join(sys.argv),
        'args': vars(args),
        'git_commit': commit,
        'git_dirty': dirty,
        'gurobi_version': ".".join(str(v) for v in gp.gurobi.version()),
        'python_version': sys.version.split()[0],
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'pareto_time_s': pareto_time_s,
        'num_solutions': len(pareto_solutions),
        'solutions': [
            {
                'file': f"solution_{idx}.json",
                'total_cost': sol.get('total_cost'),
                'cable_length': sol.get('cable_length'),
            }
            for idx, sol in enumerate(pareto_solutions, 1)
        ],
    }
    with open(os.path.join(args.output_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2, default=str)
    for idx, sol in enumerate(pareto_solutions, 1):
        with open(os.path.join(args.output_dir, f"solution_{idx}.json"), "w") as f:
            json.dump(sol, f, indent=2, default=str)
    print(f"Saved run_info.json and {len(pareto_solutions)} solution JSON file(s) to: {args.output_dir}")


def main(args):
    print("=" * 80)
    print("ECU OPTIMIZATION AND VISUALIZATION PIPELINE")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    config_reader = ConfigReader(args.config_dir)
    partitions = config_reader.get_partitions()

    # Uncertainty blocks: --uncertainty switches on all of them, otherwise each one by its own flag
    uncertainty = {k: bool(args.uncertainty or getattr(args, k)) for k in UNCERTAINTY_BLOCKS}
    uncertainty['params'] = config_reader.get_uncertainty_config()
    hardwares = config_reader.get_hardwares()
    interfaces = config_reader.get_interfaces()
    
    # Step 1: Prepare data
    print("\n" + "-" * 80)
    print("STEP 1: Preparing Vehicle Architecture Data")
    print("-" * 80)

    if args.architecture_json:
        print(f"Loading architecture from JSON: {args.architecture_json}")
        scs, comm_matrix, sensors, actuators, cable_types, locations = load_architecture_from_json(
            args.architecture_json,
            config_reader=config_reader,
            num_locs=args.num_locs,
        )
    else:
        generator = VehicleDataGenerator(num_locs=args.num_locs, num_scs=args.num_scs, seed=args.seed, config_reader=config_reader)
        scs, comm_matrix, sensors, actuators, cable_types, locations = generator.generate_data()
    
    # Summary and visualization of generated data
    visualizer = Visualization(save_dir=args.output_dir)
    reporter = ReportGenerator()
    reporter.display_data_summary(scs, sensors, actuators, cable_types, comm_matrix, locations=locations)
    reporter.display_data(sensors, actuators, scs, locations=locations, hardwares=hardwares, interface_costs=interfaces, partitions=partitions)

    # Step 2: Run optimization
    if hasattr(args, 'solver') and args.solver == 'z3':
        pass
    else:
        opt = AssignmentOptimizer()
    
    # Step 3: Run Pareto optimization
    start_time = time.time()
    pareto_solutions = opt.optimize_pareto_epsilon_constraint(
        scs, locations, sensors, actuators, cable_types, comm_matrix,
        partitions=partitions,
        hardwares=hardwares,
        interfaces=interfaces,
        backbone=config_reader.get_backbone_config(),
        phase2_time=args.phase2_time,
        gurobi_params=args.gurobi_params,
        branch_priority=args.branch_priority,
        num_points=args.num_points,
        solve_time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        lp_dir=args.output_dir if args.write_lp else None,
        uncertainty=uncertainty,
    )
    end_time = time.time()
    print(f"#"*80)
    print(f"Pareto optimization completed in {end_time - start_time:.2f} seconds.")
    print(f"#"*80)

    # Save results before plotting, so a plotting error does not lose them
    save_run_results(args, pareto_solutions, end_time - start_time)
    
    # Display detailed solution analysis
    if pareto_solutions:
        for idx, solution in enumerate(pareto_solutions, 1):
            reporter.display_solution_details(solution, scs, locations, sensors, actuators)

    # Pareto front figure (pareto_front_analysis.pdf)
    visualizer.visualize_pareto_front(pareto_solutions)

    for solution_idx, solution in enumerate(pareto_solutions, 1):
        reporter.display_assignments(solution['assignment'])

    # Figures inside the reference car: tools/vehicle_layout/solution_paths.py <output_dir>/solution_<i>.json, then
    # tools/vehicle_viz/build_vehicle.py --solution <that file> --render (Blender, see HANDOFF / README)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="ECU Optimization and Visualization Pipeline")
    argparser.add_argument("--num_locs", type=int, default=None, help="Use only this many of the candidate sites in configs/sites.json (default: all)")
    argparser.add_argument("--num_scs", type=int, default=20, help="Number of software components to generate")
    argparser.add_argument("--architecture_json", type=str, default=None, help="Load complete architecture from JSON and bypass data_generator")
    argparser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    argparser.add_argument("--config_dir", type=str, default="configs", help="Directory containing configuration JSON files")
    argparser.add_argument("--solver", type=str, default="gurobi", choices=["gurobi", "z3"], help="Solver to use")
    argparser.add_argument("--time_limit", type=int, default=43200, help="Time limit per solve in seconds (default: 12 h)")
    argparser.add_argument("--phase2_time", type=int, default=600, help="Minimum time for the second lexicographic phase of every solve, even when the first phase used up the time limit (default: 600 s)")
    argparser.add_argument("--uncertainty", "--uncertainity", dest="uncertainty", action="store_true", help="Enable all uncertainty-aware constraint blocks below")
    argparser.add_argument("--sw_margin", action="store_true", help="SW resource demand margins per ASIL / AI (configs/uncertainty.json: sw_margin)")
    argparser.add_argument("--derating", action="store_true", help="Partition capacity derating by location health factor (configs/uncertainty.json: derating)")
    argparser.add_argument("--redundancy", action="store_true", help="Redundant SC pairs must be placed at different locations")
    argparser.add_argument("--ai_contention", action="store_true", help="Limit AI-accelerated SCs per location (configs/uncertainty.json: ai_contention)")
    argparser.add_argument("--robust_latency", action="store_true", help="Robust link latency with safety factor and overheads (configs/uncertainty.json: robust_latency)")
    argparser.add_argument("--warm_start", action="store_true", help="Enable warm start for optimization")
    argparser.add_argument("--mip_gap", type=float, default=None, help="MIP Gap for Gurobi optimizer (default: Gurobi's own, 1e-4)")
    argparser.add_argument("--gurobi_params", type=str, default=None, help="Extra Gurobi parameters, e.g. \"MIPFocus=3,Cuts=2\" (experiments)")
    argparser.add_argument("--branch_priority", action="store_true", help="Branch on site activation first, then partitions/hardware (experiments)")
    argparser.add_argument("--output_dir", type=str, default="results", help="Directory to save visualization results")
    argparser.add_argument("--write_lp", action="store_true", help="Write the Gurobi model of each solve as model_<point>.lp.gz into output_dir")
    argparser.add_argument("--num_points", type=int, default=5, help="Number of Pareto points to generate (1 = only the min-cost extreme)")
    argparser.add_argument("--verbose", action="store_true", help="Enable verbose output during optimization")
    args = argparser.parse_args()
    main(args)
