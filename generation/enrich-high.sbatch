#!/bin/bash
#SBATCH --job-name=gpt55_meta_high
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set." >&2
  exit 2
fi

module purge
module load Python/3.12.3-GCCcore-13.3.0
source "$VSC_DATA_VO_USER/venvs/citation-audit/bin/activate"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

ACTION="${ACTION:-submit}"
case "$ACTION" in
  submit|status|collect) ;;
  *) echo "ACTION must be submit, status, or collect." >&2; exit 2 ;;
esac

python3 -m generation.enrich --condition high --start "${START:-200}" --count "${COUNT:-200}" "--$ACTION"
