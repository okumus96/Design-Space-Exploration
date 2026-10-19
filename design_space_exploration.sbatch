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
python main.py --verbose --num_points 5 --write_lp --architecture_json ./configs/full_architecture_v2.json --output_dir "results/experiment_${SLURM_JOB_ID}"

echo "All experiments completed."
