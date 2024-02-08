# Fine-tunes models with different learning rates
jobOutput=$(sbatch --array=0-29 --output=logs/imagenet_synthetic_$1_%a.log scripts/imagenet_synthetic_shift.sbatch $1)
jobNum=$(echo "$jobOutput" | awk '{print $NF}')
# Fine-tunes copies of each model according to the best learning rate
sbatch --array=30-49 --output=logs/imagenet_synthetic_$1_%a.log --dependency=$jobNum scripts/imagenet_synthetic_shift.sbatch $1