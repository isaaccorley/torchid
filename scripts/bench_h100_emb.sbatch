#!/bin/bash
#SBATCH --job-name=torchid-h100-emb
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/h100-emb-%j.out
#SBATCH --error=logs/h100-emb-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run --group validation python -u -m benchmarks.bench \
  --sizes "10000,64;10000,768;10000,1024;20000,64;20000,768;20000,1024" \
  --skip-skdim --skip-cpu \
  --out BENCHMARKS_H100_emb.md
