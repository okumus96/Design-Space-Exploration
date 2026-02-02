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
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=96:00:00

set -euo pipefail
cd /home/okumus/Design-Space-Exploration
echo "Starting Design Space Exploration..."

# Run
source ./.optimization/bin/activate
srun python3 main.py --num_ecus 30 --num_scs 100  --output_dir "results/experiment_${SLURM_JOB_ID}" &

wait

echo "All experiments completed."
