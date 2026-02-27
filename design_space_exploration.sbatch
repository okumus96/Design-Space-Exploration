#!/bin/bash
#SBATCH --job-name=ecu_dse
#SBATCH --output=/home/okumus/Design-Space-Exploration/results/ecu_dse_%j.txt
#SBATCH --error=/home/okumus/Design-Space-Exploration/results/ecu_dse_%j.txt
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=ultimate
#SBATCH --account=ultimate
#SBATCH --partition=ultimate
#SBATCH --gres=gpu:7g.79gb:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=96:00:00

set -euo pipefail
cd /home/okumus/Design-Space-Exploration
echo "Starting Design Space Exploration..."

# Run
source ./.optimization/bin/activate
srun --cpu-bind=none python main.py --num_locs 6 --verbose --show_bus_utilization --hide_peripheral_labels  --num_points 5 --architecture_json full_architecture_v3.json --output_dir "results/experiment_${SLURM_JOB_ID}" &

wait

echo "All experiments completed."
