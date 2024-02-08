#!/bin/bash
#SBATCH --job-name=synthetic
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --partition=background
#SBATCH --requeue

python -m src.experiments.imagenet_synthetic_shift $1 $SLURM_ARRAY_TASK_ID
