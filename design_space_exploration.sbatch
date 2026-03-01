#!/bin/bash
#SBATCH --job-name=ecu_dse
#SBATCH --output=/path/to/your/Design-Space-Exploration/results/ecu_dse_%j.txt
#SBATCH --error=/path/to/your/Design-Space-Exploration/results/ecu_dse_%j.txt
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
cd /path/to/your/Design-Space-Exploration
echo "Starting Design Space Exploration..."

# Run
source ./.optimization/bin/activate
srun --cpu-bind=none python main.py --num_locs 6 --verbose --show_bus_utilization --hide_peripheral_labels --num_points 5 --architecture_json ./configs/full_architecture_v2.json --latex_topdown --topdown_dpi 600 --time_limit 600 --output_dir "results/experiment_${SLURM_JOB_ID}" &

wait

echo "All experiments completed."
