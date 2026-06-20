#!/bin/bash
# Themis-RM-32B GRPO Multi-Node Training Script (4 nodes, colocated layout)
# =============================================================================
# Trains Qwen3-8B-Base with the 32B Themis scalar reward model
# (project-themis/Themis-RM-32B) on a 4-node x 8-GPU AWS cluster (32 GPUs).
#
# LAYOUT (colocated / hybrid engine):
#   reward_model.enable_resource_pool=False (default) => the policy (actor+rollout)
#   and the Themis-32B reward model share ALL 32 GPUs. One Ray WorkerDict actor per
#   GPU holds a shard of both models; they time-share the GPU (rollout -> log_prob
#   -> reward scoring -> GRPO update). The Themis RM is FSDP-sharded AND CPU-offloaded
#   (the CodeScaler reward-model worker hardcodes CPUOffload), so its weights live in
#   host RAM and stream onto the GPUs only for the scoring forward pass.
#
# RAY vs TORCH MASTER:
#   The IP/port you supply below is the *Ray head* address (used to join the 4 nodes
#   into one Ray cluster). VeRL derives the torch.distributed MASTER_ADDR/MASTER_PORT
#   automatically from the rank-0 worker's placement group -- you do NOT set those.
#
# USAGE:
#   1. On every WORKER node (3 of them), first run:
#        HEAD_IP=<head-private-ip> HEAD_PORT=6379 ROLE=worker bash scripts/train_themis_32b_multinode.sh
#   2. On the HEAD node, run:
#        HEAD_IP=<head-private-ip> HEAD_PORT=6379 ROLE=head   bash scripts/train_themis_32b_multinode.sh
#   The head waits until all 32 GPUs have registered, then launches the driver.
#   Only the head runs the Python driver; workers just host Ray actors.
#
# OUT-OF-MEMORY LEVERS: see the big comment block at the bottom of this file.
# =============================================================================

set -x

# ============================================================================
# UV ENVIRONMENT BOOTSTRAP (self-contained; run from anywhere, on every node)
# ============================================================================
# Resolve the repo root from this script's location and create/activate the locked
# uv environment, so `python`/`ray` resolve to the project .venv. Run this on BOTH
# the head and the worker nodes. Activating the venv (instead of `uv run`) also
# avoids Ray's uv runtime-env hook, which errors on the recipe's working_dir=None.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
uv sync --frozen
source .venv/bin/activate

# ============================================================================
# CLUSTER / LAUNCH CONFIGURATION  (override via env)
# ============================================================================
HEAD_IP=${HEAD_IP:?Set HEAD_IP to the Ray head node private IP}
HEAD_PORT=${HEAD_PORT:-6379}      # Ray head port (default 6379)
ROLE=${ROLE:?Set ROLE=head or ROLE=worker}
n_gpus_per_node=8
n_nodes=4
WORLD_GPUS=$((n_gpus_per_node * n_nodes))   # 32

# ============================================================================
# AWS MULTI-NODE NETWORKING (EFA / NCCL)
# These are NOT in the repo; they matter a lot for inter-node FSDP collectives.
# Adjust NCCL_SOCKET_IFNAME to your ENI name (`ip -o link` / check your AMI).
# The security group must allow (a) Ray ports between nodes and (b) all traffic
# within the SG itself for EFA/NCCL (self-referencing SG rule).
# ============================================================================
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens}
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
# export NCCL_PROTO=simple   # uncomment if EFA throughput/hangs require it
export VLLM_USE_V1=1

# ----------------------------------------------------------------------------
# WORKER nodes: join the Ray cluster and idle. The head drives everything.
# ----------------------------------------------------------------------------
if [ "$ROLE" = "worker" ]; then
    ray stop
    ray start --address="${HEAD_IP}:${HEAD_PORT}" --num-gpus=${n_gpus_per_node}
    echo "Worker joined Ray head ${HEAD_IP}:${HEAD_PORT}. Leave this node running; the head launches training."
    exit 0
fi

if [ "$ROLE" != "head" ]; then
    echo "ROLE must be 'head' or 'worker' (got '$ROLE')"; exit 1
fi

# Head-only: tear down the local Ray head on exit (success, failure, or Ctrl-C) so
# the cluster doesn't leak. NOTE: this stops Ray on the HEAD node only; each WORKER
# node keeps its own Ray running (it exited above with ROLE=worker). After the run,
# stop the workers with `ray stop` on each worker node (or just terminate them).
cleanup() { echo ">> ray stop (cleanup, head)"; ray stop --force >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
dataset_name=DeepCoder
train_data=[$(pwd)/datasets/DeepCoder/train.parquet]
val_data=[$(pwd)/datasets/DeepCoder/val.parquet]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
model_name=Qwen/Qwen3-8B-Base
model_pretty_name=Qwen3-8B-Base
# 32B Themis reward model. The "themis" substring drives the AutoModelForSequence-
# Classification arch selection + functional-correctness system prompt in
# CodeScalerRewardModelWorker. Pre-stage this on a shared FSx/EFS mount or bake it
# into the AMI so all 4 nodes don't each re-download ~64GB.
rm_path=project-themis/Themis-RM-32B
rm_pretty_name=Themis-RM-32B

# ============================================================================
# TRAINING ALGORITHM CONFIGURATION
# ============================================================================
rl_alg=grpo
reward_manager=codescaler   # keep: this routes to CodeScalerRewardModelWorker

# ============================================================================
# SHARDING / MEMORY CONFIGURATION
# ============================================================================
# fsdp_size controls the FSDP device mesh (see OOM levers at bottom):
#   8  -> HYBRID_SHARD: shard within each node's 8 GPUs (NVLink), replicate across the
#         4 nodes. Faster (param all-gathers stay intra-node) but MORE per-GPU memory.
#   -1 -> FULL_SHARD across all 32 GPUs: LEAST per-GPU memory, but every all-gather
#         crosses the inter-node fabric. Use this first if you OOM.
fsdp_size=8
gpu_memory_utilization=0.7   # leave HBM headroom for FSDP gathers + RM paging
do_offload=True
strategy="fsdp"

# ============================================================================
# BATCH SIZE AND PARALLELIZATION CONFIGURATION
# ============================================================================
n=16
batch_size=128
ppo_mini_batch_size=64
ppo_micro_batch_size=16
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=2
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
tensor_model_parallel_size=1   # vLLM rollout TP; raise to 2/4 to cut rollout HBM
mask_observations=True
enable_mtrl=False
max_action_length=2048

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
max_prompt_length=4096
max_response_length=16384
max_obs_length=512
temperature=0.6
top_p=0.95
action_stop_tokens=''
max_turns=0
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6

export EXP_NAME="${dataset_name}-${reward_manager}-${model_pretty_name}-${rm_pretty_name}"
run_name=$EXP_NAME

export VERL_RUN_ID=$run_name
export WANDB_PROJECT="LibraRM"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROOT=$(pwd)
OUTPUT_DIR=$ROOT/logs/$WANDB_PROJECT/$EXP_NAME/$TIMESTAMP
mkdir -p $OUTPUT_DIR

# ----------------------------------------------------------------------------
# HEAD node: start the Ray head, wait for all 32 GPUs to register, then drive.
# ----------------------------------------------------------------------------
ray stop
ray start --head --node-ip-address="${HEAD_IP}" --port="${HEAD_PORT}" --num-gpus=${n_gpus_per_node}

# Block until all 4 nodes have joined; otherwise placement-group creation hangs/fails.
python - "$WORLD_GPUS" <<'PYEOF'
import ray, sys, time
ray.init(address="auto")
need = int(sys.argv[1])
while int(ray.cluster_resources().get("GPU", 0)) < need:
    print(f"waiting for GPUs: {int(ray.cluster_resources().get('GPU',0))}/{need}")
    time.sleep(3)
print(f"cluster ready: {need} GPUs registered")
PYEOF

python -m recipe.codescaler.main_codescaler \
    algorithm.adv_estimator=$rl_alg \
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.metric='seq_final_reward' \
    +algorithm.filter_groups.max_num_gen_batches=0 \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=512 \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.enable=True \
    reward_model.max_length=4096 \
    reward_model.model.path=$rm_path \
    reward_model.model.fsdp_config.fsdp_size=$fsdp_size \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    +reward_model.record_dir=$ROOT \
    +reward_model.check_ast=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$ppo_micro_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.loss_agg_mode='token-mean' \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=False \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_training_steps=250 "${@:1}" 2>&1 | tee $OUTPUT_DIR/train.log


# =============================================================================
# OUT-OF-MEMORY LEVERS (4 nodes x 8 GPUs, colocated Themis-32B)
# Ordered roughly by impact for the colocated layout. Pull the top ones first.
# =============================================================================
# 1. MOVE THE 32B RM TO ITS OWN NODES (biggest lever for colocation OOM).
#    reward_model.enable_resource_pool=True with its own pool, e.g. 3 policy nodes
#    + 1 RM node:
#       reward_model.enable_resource_pool=True \
#       reward_model.n_gpus_per_node=8 reward_model.nnodes=1 \
#       trainer.nnodes=3
#    The RM no longer competes for policy HBM/host-RAM at all.
#
# 2. fsdp_size=-1  (FULL_SHARD across all 32 GPUs). Minimizes per-GPU param/optimizer
#    memory (each GPU holds 1/32 of each model vs 1/8 under HYBRID). Costs inter-node
#    bandwidth, but is the cleanest fix when you simply don't fit. Set it for BOTH the
#    actor (actor_rollout_ref.actor.fsdp_config.fsdp_size) and the RM
#    (reward_model.model.fsdp_config.fsdp_size); the var above already does both.
#
# 3. Lower vLLM HBM reservation: actor_rollout_ref.rollout.gpu_memory_utilization
#    (0.7 -> 0.5/0.4). vLLM's KV cache is reserved up front; shrinking it frees HBM
#    for FSDP all-gathers and RM paging during colocation.
#
# 4. Shard the rollout too: actor_rollout_ref.rollout.tensor_model_parallel_size=2 (or 4)
#    so vLLM splits the policy across GPUs, cutting per-GPU rollout weights/activations.
#
# 5. Cut activation memory:
#      - actor_rollout_ref.actor.ppo_max_token_len_per_gpu (20480 -> 10240) under dynamic bsz
#      - actor_rollout_ref.actor.ppo_micro_batch_size / ppo_micro_batch_size_per_gpu
#      - log_prob_micro_batch_size_per_gpu (2 -> 1)
#      - gradient checkpointing + activation offload are already ON.
#
# 6. Cut RM activation memory (the 32B forward is the spike during scoring):
#      - reward_model.max_length (4096 -> 2048)
#      - reward_model.forward_max_token_len_per_gpu (inherits critic's; set lower)
#      - reward_model.micro_batch_size_per_gpu=1
#
# 7. Shorten sequences (activations scale with length everywhere):
#      - data.max_response_length (16384 is large; 8192 roughly halves activation peak)
#      - or shard long sequences: ulysses_sequence_parallel_size=2/4
#
# 8. Keep offload ON (already set): actor param/optimizer offload between steps, and
#    the RM is CPU-offloaded by the worker. Ensure each node has enough HOST RAM:
#    under HYBRID_SHARD a node's CPU holds a full 32B bf16 copy (~64GB) for the RM,
#    plus the actor's offloaded state. p4d/p5-class host RAM (>1TB) is comfortable;
#    smaller-RAM instances may need fsdp_size=-1 (lever 2) to spread the offload.
# =============================================================================
