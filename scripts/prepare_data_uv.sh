#!/bin/bash
# Dataset preparation (self-contained; run from anywhere after cloning).
# =============================================================================
# Produces the parquet files the training scripts expect:
#   datasets/DeepCoder/train.parquet            (training data)
#   datasets/Evaluation/LiveCodeBench.parquet   (validation data)
#   ... plus datasets/DeepCoder/test.parquet and the other eval benchmarks.
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

# 1. Training set (DeepCoder): pulls the source datasets from the Hub and writes
#    datasets/DeepCoder/{train,test}.parquet. Independent of the JSON benchmarks.
echo ">> preparing training dataset (data/prepare_deepcoder.py)"
python data/prepare_deepcoder.py

# 2. Download the JSON evaluation benchmarks into ./data/*.json. Must run before
#    prepare_evaluation.py, which reads those files.
echo ">> downloading evaluation benchmarks (data/download_data.py)"
python data/download_data.py

# 3. Build the evaluation parquet files (incl. datasets/Evaluation/LiveCodeBench.parquet).
echo ">> preparing evaluation datasets (data/prepare_evaluation.py)"
python data/prepare_evaluation.py

echo ">> done. Train: datasets/DeepCoder/train.parquet | Val: datasets/Evaluation/LiveCodeBench.parquet"
