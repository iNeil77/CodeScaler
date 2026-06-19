export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Self-contained uv bootstrap: resolve repo root, create/activate the locked .venv
# so `python` below resolves to the project environment. (No Ray here, so no trap.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
uv sync --frozen
source .venv/bin/activate

MODEL_SHORT=Qwen3-8B
MODEL_PATH=Qwen/Qwen3-8B-Base
python recipe/codescaler/eval.py --model_short $MODEL_SHORT \
                                --model_path $MODEL_PATH \
                                --device_ids 0,1,2,3,4,5,6,7 \
                                --dataset codeforces
