export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Runs with whatever Python environment is already active, exactly as before. For a
# self-contained uv-managed run, use scripts/eval_uv.sh.

MODEL_SHORT=Qwen3-8B
MODEL_PATH=Qwen/Qwen3-8B-Base
python recipe/codescaler/eval.py --model_short $MODEL_SHORT \
                                --model_path $MODEL_PATH \
                                --device_ids 0,1,2,3,4,5,6,7 \
                                --dataset codeforces
