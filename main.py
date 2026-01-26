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

    config_reader = ConfigReader(args.config)
    generator = VehicleDataGenerator(num_ecus=args.num_ecus, num_scs=args.num_scs, seed=args.seed, config_reader=config_reader)
    ecus, scs, comm_matrix, sensors, actuators, cable_types = generator.generate_data()
    


    # Summary and visualzation of generated data
    visualizer = Visualization()
    visualizer.display_data_summary(ecus, scs, sensors, actuators, cable_types,comm_matrix)
    visualizer.display_data(sensors, actuators, scs, ecus)
    visualizer.plot_charts(scs, ecus, sensors, actuators)
    visualizer.plot_sw_sensor_actuator_graph_final(scs, sensors, actuators, comm_matrix)
    
    return

    # Step 2: Run optimization
    print("\n" + "-" * 80)
    print("STEP 2: Running Gurobi Optimization")
    print("-" * 80)
    opt = AssignmentOptimizer()
    
    
    # Generate Pareto front: Total Cost (Hardware + Cable) vs Load Balancing
    pareto_solutions = opt.optimize_pareto_cost_vs_loadbalance(
        scs, ecus, sensors, actuators, cable_types, comm_matrix, num_points=5
    )
    
    # Visualize Pareto front
    visualizer.visualize_pareto_front(pareto_solutions)
    
    # Visualize and analyze each solution
    print("\n" + "=" * 80)
    print("VISUALIZATION OF EACH PARETO SOLUTION")
    print("=" * 80)
    
    for solution_idx, solution in enumerate(pareto_solutions, 1):
        assignments = solution['assignment']
        
        print("\n" + "-" * 80)
        print(f"SOLUTION {solution_idx}")
        print(f"  Hardware Cost: ${solution['hardware_cost']:.2f}")
        print(f"  Cable Length: {solution['kpis']['total_length']:.2f}m")
        print(f"  Cable Cost: ${solution['kpis']['total_cost']:.2f} (Real)")
        print(f"    - Sensor→ECU: ${solution['kpis']['breakdown']['sensor']['cost']:.2f}")
        print(f"    - Actuator→ECU: ${solution['kpis']['breakdown']['actuator']['cost']:.2f}")
        print(f"    - ECU↔ECU: ${solution['kpis']['breakdown']['ecu_ecu']['cost']:.2f}")
        print(f"  Total Latency: {solution['kpis']['total_latency']*1000:.2f}us")
        print(f"  Total Weight: {solution['kpis']['total_weight']:.2f}kg")
        print(f"  Total Project Cost: ${solution['total_cost']:.2f}")
        print(f"  Max Utilization: {solution['max_utilization']:.1%}")
        print("-" * 80)
        
        print(f"\nAssignment Summary:")
        print(f"   - Total SWs Assigned: {len(assignments)} / {len(scs)}")
        print(f"   - ECUs Used: {solution['num_ecus_used']}")
        print(f"   - Max Utilization: {solution['max_utilization']:.1%}")
        visualizer.display_assignments(assignments)
        
        # Visualize this solution
        print(f"\n   Generating visualization for Solution {solution_idx}...")
        visualizer.visualize_optimization_result(scs, ecus, sensors, actuators, assignments)
        
        # Vehicle layout with assigned ECUs
        print(f"   Generating vehicle layout for Solution {solution_idx}...")
        visualizer.plot_vehicle_layout_topdown(sensors, actuators, assignments, ecus)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="ECU Optimization and Visualization Pipeline")
    argparser.add_argument("--num_ecus", type=int, default=30, help="Number of candidate ECUs to generate")
    argparser.add_argument("--num_scs", type=int, default=20, help="Number of software components to generate")
    argparser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    argparser.add_argument("--config", type=str, default="configs/vehicle_config.json", help="Path to vehicle configuration JSON file")
    argparser.add_argument("--sc_domain_weights", type=list, nargs=5, default=[0.15, 0.15, 0.20, 0.35, 0.15], help="Weights for SC domain distribution")
    args = argparser.parse_args()
    main(args)
