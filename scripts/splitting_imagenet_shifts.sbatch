#!/bin/bash
#SBATCH --job-name=splitting
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --requeue

python -m src.experiments.splitting_shifts.imagenet $1 $SLURM_ARRAY_TASK_ID $2