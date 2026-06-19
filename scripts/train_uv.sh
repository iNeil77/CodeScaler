#!/bin/bash
# Sample uv-driven training launcher.
# =============================================================================
# Materializes the locked environment with `uv sync --frozen` and then launches
# one of the existing training recipes under `uv run`, so that `python`, `ray`,
# and `wandb` inside the recipe all resolve to the project's .venv -- no manual
# `conda activate` / `source .venv/bin/activate` needed.
#
# This is a thin wrapper: it does NOT duplicate the training hyperparameters. It
# just picks one of:
#   scripts/train_codescaler.sh          (default; CodeScaler-8B reward model)
#   scripts/train_themis.sh              (Themis reward model, single node)
#   scripts/train_themis_32b_multinode.sh (Themis-RM-32B, 4 nodes)
# and runs it inside the uv environment.
#
# USAGE (run from the repo root):
#   bash scripts/train_uv.sh                       # -> train_codescaler.sh
#   bash scripts/train_uv.sh themis                # -> train_themis.sh
#   bash scripts/train_uv.sh themis-32b-multinode  # -> train_themis_32b_multinode.sh
#
# Any extra args are forwarded to the underlying recipe as Hydra overrides, e.g.
#   bash scripts/train_uv.sh codescaler trainer.total_training_steps=10
#
# Prerequisite: uv installed (https://docs.astral.sh/uv/) and run from the repo
# root (where pyproject.toml / uv.lock live).
# =============================================================================
set -euo pipefail

# Resolve repo root from this script's location, so it works regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# Map the friendly recipe name to a script.
RECIPE="${1:-codescaler}"
case "$RECIPE" in
    codescaler)            TARGET="scripts/train_codescaler.sh" ;;
    themis)                TARGET="scripts/train_themis.sh" ;;
    themis-32b-multinode)  TARGET="scripts/train_themis_32b_multinode.sh" ;;
    *)
        echo "error: unknown recipe '$RECIPE'." >&2
        echo "       choose one of: codescaler | themis | themis-32b-multinode" >&2
        exit 1
        ;;
esac
shift || true  # drop the recipe name; remaining args are forwarded as Hydra overrides

# When a Ray driver is launched under `uv run`, Ray auto-activates a hook
# (RAY_ENABLE_UV_RUN_RUNTIME_ENV, default on) that rederives a uv runtime_env for the
# workers. The recipe passes runtime_env with working_dir=None, which makes that hook
# raise `TypeError: path_or_uri must be a string, got NoneType` before ray.init()
# returns. We already manage dependencies via the synced .venv (workers inherit it on
# this single node), so disable the hook.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# 1. Create/refresh the locked environment (.venv). --frozen errors out instead of
#    silently editing uv.lock, keeping the run reproducible.
echo ">> uv sync --frozen"
uv sync --frozen

# Always tear down the Ray cluster on exit (success, failure, or Ctrl-C). The recipe
# scripts start `ray start --head` but their trailing cleanup (`pkill -P -9
# $server_pid`) is broken (unset var), so without this the head + worker processes
# leak and keep holding GPUs/RAM after the run ends.
cleanup() {
    echo ">> ray stop (cleanup)"
    uv run --frozen --no-sync ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# 2. Launch the chosen recipe inside the uv environment. `uv run` puts .venv/bin on
#    PATH for the bash subprocess, so the `python -m recipe...` / `ray` / `wandb`
#    calls inside $TARGET use the locked interpreter and dependencies.
#    --no-sync because step 1 already synced. Not `exec`, so the trap above runs.
echo ">> uv run bash $TARGET $*"
uv run --frozen --no-sync bash "$TARGET" "$@"
