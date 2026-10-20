#!/bin/bash
#SBATCH --job-name=ecu_dse
#SBATCH --output=results/ecu_dse_%j.txt
#SBATCH --error=results/ecu_dse_%j.txt
#SBATCH --open-mode=append
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=96:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"  # submit from the repository root: sbatch design_space_exploration.sbatch
echo "Starting Design Space Exploration on $(hostname) with ${SLURM_CPUS_PER_TASK} CPUs..."

source ./dse/bin/activate
ARCH=./configs/full_architecture_v2.json
OUT="results/experiment_${SLURM_JOB_ID}"

# 1) Cost-cable Pareto front of the nominal design (5 points, lexicographic cost then cable)
python main.py --vehicle_layout --num_points 5 --architecture_json "$ARCH" --time_limit 3600 --phase2_time 600 --output_dir "$OUT/pareto"

# 2) Cost against software growth (levels of configs/uncertainty.json), margins/derating/redundancy/AI/robust latency on
python main.py --vehicle_layout --growth --sw_margin --derating --redundancy --ai_contention --robust_latency --architecture_json "$ARCH" --time_limit 3600 --phase2_time 600 --output_dir "$OUT/growth"

# 3) The same curve with single-failure tolerance (--uncertainty = all blocks incl. failures)
python main.py --vehicle_layout --uncertainty --architecture_json "$ARCH" --time_limit 7200 --phase2_time 600 --output_dir "$OUT/growth_failures"

echo "All experiments completed."
