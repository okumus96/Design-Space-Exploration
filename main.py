#!/usr/bin/env python3
"""
ECU Optimization and Visualization Pipeline

Main script that orchestrates the entire workflow:
1. Generate vehicle architecture data (ECUs, SCs, sensors, actuators)
2. Run Gurobi optimization for SW-to-ECU assignment
3. Visualize the optimization results
"""
import argparse
from config_reader import ConfigReader
from data_generator import VehicleDataGenerator
from optimizer import AssignmentOptimizer
from optimizer_z3 import AssignmentOptimizerZ3
from visualizer import Visualization
from tabulate import tabulate

def main(args):
    print("=" * 80)
    print("ECU OPTIMIZATION AND VISUALIZATION PIPELINE")
    print("=" * 80)
    
    # Step 1: Generate data
    print("\n" + "-" * 80)
    print("STEP 1: Generating Vehicle Architecture Data")
    print("-" * 80)

    config_reader = ConfigReader(args.config_dir)
    generator = VehicleDataGenerator(num_ecus=args.num_ecus, num_scs=args.num_scs, seed=args.seed, config_reader=config_reader)
    ecus, scs, comm_matrix, sensors, actuators, cable_types = generator.generate_data()
    
    # Summary and visualization of generated data
    visualizer = Visualization(save_dir=args.output_dir)
    visualizer.display_data_summary(ecus, scs, sensors, actuators, cable_types,comm_matrix)
    visualizer.display_data(sensors, actuators, scs, ecus)
    visualizer.plot_charts(scs, ecus, sensors, actuators)
    visualizer.plot_vehicle_layout_topdown(sensors, actuators, assignments=None, ecus=ecus, filename="initial_vehicle_layout.png")
    #visualizer.plot_sw_sensor_actuator_graph_final(scs, sensors, actuators, comm_matrix)

    #return
    # Step 2: Run optimization
    print("\n" + "-" * 80)

    if hasattr(args, 'solver') and args.solver == 'z3':
        print("STEP 2: Running Z3 Optimization")
        print("-" * 80)
        opt = AssignmentOptimizerZ3()
    else:
        print("STEP 2: Running Gurobi Optimization")
        print("-" * 80)
        opt = AssignmentOptimizer()
    
    # Generate Pareto front: HW Cost vs Cable Length
    pareto_solutions = opt.optimize(
        scs, ecus, sensors, actuators, cable_types, comm_matrix, num_points=args.num_points,
        include_cable_cost=True, enable_latency_constraints=True, warm_start=args.warm_start, time_limit=args.time_limit,
        mip_gap=args.mip_gap, verbose=args.verbose
    )
    
    # Visualize Pareto front
    visualizer.visualize_pareto_front(pareto_solutions)
    
    # Visualize and analyze each solution
    print("\n" + "=" * 80)
    print("VISUALIZATION OF EACH PARETO SOLUTION")
    print("=" * 80)
    
    for solution_idx, solution in enumerate(pareto_solutions, 1):
        visualizer.display_assignments(solution_idx,solution,scs,ecus)
        # Visualize this solution
        print(f"\n   Generating visualization for Solution {solution_idx}...")
        visualizer.visualize_optimization_result(scs, ecus, sensors, actuators, solution['assignment'], filename=f"optimization_result_solution_{solution_idx}.png")
        
        # Vehicle layout with assigned ECUs
        print(f"   Generating vehicle layout for Solution {solution_idx}...")
        visualizer.plot_vehicle_layout_topdown(sensors, actuators, solution['assignment'], ecus, filename=f"vehicle_layout_solution_{solution_idx}.png")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="ECU Optimization and Visualization Pipeline")
    argparser.add_argument("--num_ecus", type=int, default=30, help="Number of candidate ECUs to generate")
    argparser.add_argument("--num_scs", type=int, default=100, help="Number of software components to generate")
    argparser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    argparser.add_argument("--config_dir", type=str, default="configs", help="Directory containing configuration JSON files")
    argparser.add_argument("--solver", type=str, default="gurobi", choices=["gurobi", "z3"], help="Solver to use")
    argparser.add_argument("--time_limit", type=int, default=None, help="Time limit in seconds")
    argparser.add_argument("--warm_start", action="store_true", help="Enable warm start for optimization")
    argparser.add_argument("--mip_gap", type=float, default=None, help="MIP Gap for Gurobi optimizer")
    argparser.add_argument("--output_dir", type=str, default="results", help="Directory to save visualization results")
    argparser.add_argument("--num_points", type=int, default=5, help="Number of Pareto points to generate")
    argparser.add_argument("--verbose", action="store_true", help="Enable verbose output during optimization")
    args = argparser.parse_args()
    main(args)
