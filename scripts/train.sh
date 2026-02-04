#!/bin/bash
# CodeScaler GRPO Correctness Reward Model Training Script
# This script trains a model using GRPO (Group Relative Policy Optimization) algorithm
# with CodeScaler reward model

set -x
ray stop
# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
dataset_name=DeepCoder
train_data=[$(pwd)/datasets/DeepCoder/train.parquet]
val_data=[$(pwd)/datasets/Evaluation/LiveCodeBench.parquet]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
model_name=Qwen/Qwen3-8B-Base
model_pretty_name=Qwen3-8B-Base
rm_path=LARK-Lab/CodeScaler-8B
rm_pretty_name=CodeScaler-8B
# ============================================================================
# TRAINING ALGORITHM CONFIGURATION
# ============================================================================
# RL algorithm: gae(ppo) or grpo
# Note: if grpo, then better set n>1 otherwise the group norm can not be effective
rl_alg=grpo
reward_manager=codescaler

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================
n_gpus_per_node=8
n_nodes=1
tensor_model_parallel_size=1
# Higher gpu_memory_utilization will likely cause vllm to OOM, so set to lower value
gpu_memory_utilization=0.8
# Control actor's fsdp offloading; if gpu_memory_utilization > 0.6, should be True to avoid OOM
do_offload=True
strategy="fsdp"

# ============================================================================
# BATCH SIZE AND PARALLELIZATION CONFIGURATION
# ============================================================================
n=8
batch_size=128
ppo_mini_batch_size=64
ppo_micro_batch_size=16
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=2
use_dynamic_bsz=True  # faster
# Set to 1 for normal verl behavior, otherwise it will cause OOM
ulysses_sequence_parallel_size=1
fsdp_size=-1
mask_observations=True # mask observations for kl loss and gradient descent
enable_mtrl=False # enable multi-turn training
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
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
export NCCL_NVLS_ENABLE=0

export WANDB_PROJECT="DeepCoder"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NUM_GPUS=8

ROOT=$(pwd)
OUTPUT_DIR=$ROOT/logs/$WANDB_PROJECT/$EXP_NAME/$TIMESTAMP
mkdir -p $OUTPUT_DIR

ray start --head --num-gpus ${NUM_GPUS}

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
    trainer.test_freq=10 \
    trainer.total_training_steps=250 "${@:1}" 2>&1 | tee $OUTPUT_DIR/train.log


pkill -P -9 $server_pid
kill -9 $kill $server_pid
