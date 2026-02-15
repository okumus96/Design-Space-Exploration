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
from optimizer import AssignmentOptimizer
from visualizer import Visualization
from tabulate import tabulate
import time

def main(args):
    print("=" * 80)
    print("ECU OPTIMIZATION AND VISUALIZATION PIPELINE")
    print("=" * 80)

    config_reader = ConfigReader(args.config_dir)
    partitions = config_reader.get_partitions()
    hardwares = config_reader.get_hardwares()
    interfaces = config_reader.get_interfaces()
    
    # Step 1: Generate data
    print("\n" + "-" * 80)
    print("STEP 1: Generating Vehicle Architecture Data")
    print("-" * 80)

    generator = VehicleDataGenerator(num_locs=args.num_locs, num_scs=args.num_scs, seed=args.seed, config_reader=config_reader)
    scs, comm_matrix, sensors, actuators, cable_types, locations = generator.generate_data()
    
    # Summary and visualization of generated data
    visualizer = Visualization(save_dir=args.output_dir)
    visualizer.display_data_summary(scs, sensors, actuators, cable_types, comm_matrix, locations=locations)
    visualizer.display_data(sensors, actuators, scs, locations=locations, hardwares=hardwares, interface_costs=interfaces, partitions=partitions)
    visualizer.plot_vehicle_layout_topdown( sensors, actuators, assignments=None, locations=locations, filename="initial_vehicle_layout.png")
    #visualizer.plot_sw_sensor_actuator_graph_final(scs, sensors, actuators, comm_matrix)
    #visualizer.plot_charts(scs, sensors, actuators)

    # Step 2: Run optimization
    if hasattr(args, 'solver') and args.solver == 'z3':
        pass
    else:
        opt = AssignmentOptimizer()
    
    # Generate Pareto front: HW Cost vs Cable Length
    start_time = time.time()
    pareto_solutions = opt.optimize(
        scs, locations, sensors, actuators, cable_types, comm_matrix,
        partitions=partitions,
        hardwares=hardwares,
        interfaces=interfaces
    )
    end_time = time.time()
    print(f"#"*80)
    print(f"Optimization completed in {end_time - start_time:.2f} seconds.")
    print(f"#"*80)
    
    # Display detailed solution analysis
    if pareto_solutions:
        for idx, solution in enumerate(pareto_solutions, 1):
            visualizer.display_solution_details(solution, scs, locations, sensors, actuators)

    # Visualize Pareto front
    visualizer.visualize_pareto_front(pareto_solutions)
    
    # Visualize and analyze each solution
    print("\n" + "=" * 80)
    print("VISUALIZATION OF EACH PARETO SOLUTION")
    print("=" * 80)
    
    for solution_idx, solution in enumerate(pareto_solutions, 1):
        visualizer.display_assignments(solution['assignment'])

        print(f"\n   Generating architecture visualization for Solution {solution_idx}...")
        visualizer.display_solution_architecture(solution, scs, locations, 
                                                 filename=f"solution_architecture_{solution_idx}.png")
        
        # Generate vehicle layout with active locations and bus connections
        print(f"   Generating vehicle layout for Solution {solution_idx}...")
        visualizer.plot_vehicle_layout_topdown(sensors, actuators, solution['assignment'], locations=locations,
                                               scs=scs, comm_matrix=comm_matrix, cable_types=cable_types,
                                               filename=f"vehicle_layout_solution_{solution_idx}.png")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="ECU Optimization and Visualization Pipeline")
    argparser.add_argument("--num_locs", type=int, default=20, help="Number of locations to generate")
    argparser.add_argument("--num_scs", type=int, default=20, help="Number of software components to generate")
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
