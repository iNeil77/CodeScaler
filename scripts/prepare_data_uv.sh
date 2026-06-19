#!/bin/bash
# Dataset preparation (self-contained; run from anywhere after cloning).
# =============================================================================
# Produces the parquet files the training scripts expect:
#   datasets/DeepCoder/train.parquet            (training data)
#   datasets/DeepCoder/val.parquet              (validation data: codeforces + LCB v5 + v6)
#   ... plus the standalone eval benchmarks under datasets/Evaluation/.
#
# Like the train scripts, this resolves the repo root, syncs the locked uv
# environment, and runs everything inside it -- so a fresh clone can just do:
#   bash scripts/prepare_data.sh
# then launch training with scripts/train_codescaler.sh / train_themis.sh.
# =============================================================================
set -euo pipefail

# Resolve repo root from this script's location and create/activate the locked
# uv environment, so `python` below resolves to the project .venv. The prep
# scripts use repo-root-relative paths (./data, ./datasets), so we cd there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
uv sync --frozen
source .venv/bin/activate

# 1. Train + validation sets (DeepCoder): pulls the source datasets from the Hub
#    (incl. LiveCodeBench v6) and writes datasets/DeepCoder/train.parquet and
#    datasets/DeepCoder/val.parquet (val = codeforces + LCB v5 + LCB v6).
echo ">> preparing train + val datasets (data/prepare_deepcoder.py)"
python data/prepare_deepcoder.py

# 2. (Optional) standalone eval benchmarks under datasets/Evaluation/. Not used for
#    mid-training validation. Download must run before prepare_evaluation.py.
echo ">> downloading evaluation benchmarks (data/download_data.py)"
python data/download_data.py

echo ">> preparing standalone evaluation datasets (data/prepare_evaluation.py)"
python data/prepare_evaluation.py

echo ">> done. Train: datasets/DeepCoder/train.parquet | Val: datasets/DeepCoder/val.parquet"
