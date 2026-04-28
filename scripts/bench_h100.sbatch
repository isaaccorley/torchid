#!/bin/bash
#SBATCH --job-name=torchid-h100
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/h100-%j.out
#SBATCH --error=logs/h100-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run --group validation python -m benchmarks.bench --out BENCHMARKS_H100.md
