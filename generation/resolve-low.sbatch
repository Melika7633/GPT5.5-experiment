#!/bin/bash
#SBATCH --job-name=gpt55_resolve_low
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.log
#SBATCH --error=slurm-%j.log

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

START="${START:-200}"
COUNT="${COUNT:-200}"
STOP=$((START + COUNT - 1))
LOG_DIR=$(printf "outputs/low/resolved/%04d-%04d" "$START" "$STOP")
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/resolve.log") 2>&1

module purge
module load Python/3.12.3-GCCcore-13.3.0
source "$VSC_DATA_VO_USER/venvs/vllm-gptoss-py312/bin/activate"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

python3 -m generation.resolve --condition low --start "$START" --count "$COUNT"
