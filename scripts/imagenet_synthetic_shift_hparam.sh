sbatch --array=86-174 --output=logs/imagenet_synthetic_$1_%a.log scripts/imagenet_synthetic_shift.sbatch $1