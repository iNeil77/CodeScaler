#!/bin/bash
# Dataset preparation.
# =============================================================================
# Produces the parquet files the training scripts expect:
#   datasets/DeepCoder/train.parquet            (training data)
#   datasets/DeepCoder/val.parquet              (validation data: codeforces + LCB v5 + v6)
#   ... plus the standalone eval benchmarks under datasets/Evaluation/.
#
# Runs with whatever Python environment is already active (conda, pip, venv, ...).
# It does NOT manage dependencies; the prep scripts use repo-root-relative paths
# (./data, ./datasets), so run this from the repo root. For a self-contained
# uv-managed run (auto env sync), use scripts/prepare_data_uv.sh.
# =============================================================================
set -euo pipefail

# 1. Train + validation sets (DeepCoder): pulls the source datasets from the Hub
#    (incl. LiveCodeBench v6) and writes datasets/DeepCoder/train.parquet and
#    datasets/DeepCoder/val.parquet (val = codeforces + LCB v5 + LCB v6).
echo ">> preparing train + val datasets (data/prepare_deepcoder.py)"
python data/prepare_deepcoder.py

# 2. (Optional) standalone eval benchmarks. Download the JSON benchmarks into
#    ./data/*.json (must run before prepare_evaluation.py), then build their parquets
#    under datasets/Evaluation/. Not used for mid-training validation.
echo ">> downloading evaluation benchmarks (data/download_data.py)"
python data/download_data.py

echo ">> preparing standalone evaluation datasets (data/prepare_evaluation.py)"
python data/prepare_evaluation.py

echo ">> done. Train: datasets/DeepCoder/train.parquet | Val: datasets/DeepCoder/val.parquet"
