# Trains a baseline model from scratch
jobOutput=$(sbatch --array=50 --output=logs/imagenet_synthetic_$1_%a.log scripts/imagenet_synthetic_shift.sbatch $1)
jobNum=$(echo "$jobOutput" | awk '{print $NF}')
# Evaluates the baseline model at different epochs
sbatch --array=51-85 --output=logs/imagenet_synthetic_$1_%a.log --dependency=$jobNum scripts/imagenet_synthetic_shift.sbatch $1