# Reward Models in the CodeScaler Recipe

This document describes how scalar reward models are wired into the CodeScaler
VeRL recipe end to end: the reward flow, how a reward-model (RM) family is
selected and loaded, how each family formats its input, how the scalar score is
produced and shaped, how the policy and reward models are placed on GPUs (single-
and multi-node), the FSDP sharding modes, out-of-memory levers, and how to add a
new RM family.

All file:line references point at this repository.

---

## 1. Reward flow at a glance

CodeScaler trains a policy with GRPO using a **scalar reward model** (a learned
"is this code correct?" scorer) instead of executing code against tests during
training. Execution-based verification is reserved for validation.

Per training step (`recipe/codescaler/codescaler_ray_trainer.py:1048-1109`):

1. **Reward-model forward pass.** If `reward_model.enable=True` and `rm_scores`
   is not already on the batch, the trainer calls the RM worker
   (`codescaler_ray_trainer.py:1050-1052`):
   ```python
   if self.use_rm and "rm_scores" not in batch.batch.keys():
       reward_tensor = self.rm_wg.compute_rm_score(batch)
       batch = batch.union(reward_tensor)
   ```
2. **Reward manager.** `CodeScalerRewardManager` turns `rm_scores` into the final
   token-level reward tensor (`codescaler_reward.py:187-351`), optionally async
   (`compute_reward_async.remote`, `codescaler_ray_trainer.py:1055`).
3. **Advantage / update.** The result is stored as
   `batch.batch["token_level_scores"]` (`codescaler_ray_trainer.py:1097`); KL
   penalty (off by default in the shipped scripts) and GRPO advantage estimation
   follow.

The scalar RM is served by `CodeScalerRewardModelWorker`
(`recipe/codescaler/codescaler_fsdp_workers.py`). The reward manager is selected
by `reward_model.reward_manager` (`codescaler` → `CodeScalerRewardManager`,
registered at `codescaler_reward.py:187`).

### Train vs. validation reward

`CodeScalerRewardManager.__call__` branches on `split`
(`codescaler_reward.py:235-351`):

- **Training** uses the RM scalar, shaped (`codescaler_reward.py:294-298`):
  ```python
  if split == 'train' and 'rm_scores' in data.batch.keys():
      reward_tensor[i, valid_response_length[i].item() - 1] = transform_score(
          sum(data[i].batch['rm_scores']).item(),
          extracted_codes[i] == EMTPY_STRING,   # empty-code guard -> 0
          self.reward_shaping)
  ```
- **Validation/test** falls back to real execution via `get_verified_score` →
  `check_correctness` (`codescaler_reward.py:204-232`, `300-311`).

Reward shaping (`codescaler_reward.py:177-185`) is a softplus with a hard zero for
responses that contain no extractable code:
```python
def transform_score(score, is_empty, reward_shaping=True):
    if not reward_shaping: return score
    if is_empty:           return 0.0
    else:                  return np.log(1 + np.exp(score))   # softplus
```

---

## 2. Reward-model family selection

`CodeScalerRewardModelWorker._build_model`
(`codescaler_fsdp_workers.py:103-218`) selects the RM architecture by a
**case-insensitive substring match on the resolved model path**
(`reward_model.model.path`, after `copy_to_local`). Each supported family is
matched **explicitly**, and an unrecognized path **raises** rather than silently
defaulting (`codescaler_fsdp_workers.py:140-188`):

| Substring in path | `self.rm_type` | Loader | Head / output |
|---|---|---|---|
| `acecoderm` | `acecoderm` | `AceCodeRM.from_pretrained` (fp32) | Qwen2 causal LM + value head |
| `codescaler` | `codescaler` | `AutoModelForTokenClassification` (bf16) | token-classification, `num_labels=1` |
| `themis` | `themis` | `AutoModelForSequenceClassification` (bf16) | sequence classifier, `num_labels=1` |
| *(none of the above)* | — | — | `raise ValueError` |

```python
if 'acecoderm' in local_path.lower():
    self.rm_type = "acecoderm"
    reward_module = AceCodeRM.from_pretrained(...)
elif 'codescaler' in local_path.lower():
    self.rm_type = "codescaler"
    reward_module = AutoModelForTokenClassification.from_pretrained(...)
elif 'themis' in local_path.lower():
    self.rm_type = "themis"
    reward_module = AutoModelForSequenceClassification.from_pretrained(...)
else:
    raise ValueError("Unrecognized reward model path ...")
```

Notes:

- The match runs on the **local** path returned by `copy_to_local`
  (`verl/utils/fs.py:195-216`). For non-HDFS paths (HF repo ids, local dirs) the
  string is returned unchanged; for HDFS paths the basename is preserved. So the
  canonical ids `LARK-Lab/CodeScaler-8B` and `project-themis/Themis-RM-*` always
  match.
- `model_config.num_labels = 1` is set for all families
  (`codescaler_fsdp_workers.py:124`); the reward module is cast to bf16 after load.
- `self.rm_type` is consumed downstream by the forward pass and the chat
  construction (sections 4–5).

> The `AceCodeRM` branch exists for compatibility with AceCoder checkpoints, but
> its `forward` currently has the score computation commented out
> (`recipe/codescaler/acecoder.py`); only the `codescaler` and `themis` paths are
> fully wired.

---

## 3. Input construction and chat templates

The RM does **not** see the policy's raw generation. The worker re-builds a clean
chat from the original problem statement plus the extracted code, using the RM's
*own* chat template.

### Two tokenizers and the re-templating switch

`reward_model.model.input_tokenizer` defaults to the policy model path
(`verl/trainer/config/reward_model/reward_model.yaml`). Because the RM and policy
tokenizers differ, `_do_switch_chat_template = True`
(`codescaler_fsdp_workers.py:112-120`), so `compute_rm_score` routes through
`_switch_chat_template` (`codescaler_fsdp_workers.py:333-411`) instead of reusing
the policy's token ids.

> If you set `input_tokenizer=null`, the switch is skipped and the RM consumes the
> policy's raw token ids directly — bypassing code extraction, re-templating, and
> (for Themis) the system prompt. Keep it set for the shipped recipes.

### Per-sample formatting (`_switch_chat_template`)

For each response (`codescaler_fsdp_workers.py:342-391`):

1. Decode the response with the policy tokenizer; strip EOS.
2. Pull the original problem statement from `non_tensor_batch['raw_prompt']`.
3. **Extract only the code block** — `extract_code_from_model(response,
   check_ast=...)` (`codescaler_utils.py:41-62`): a single ```` ``` ```` block,
   AST-validated; returns `""` if there isn't exactly one valid block. CoT and
   prose are discarded.
4. Build the chat and apply the RM chat template
   (`add_generation_prompt=False, tokenize=False`), then re-tokenize with the RM
   tokenizer, **right-padded / right-truncated** to `reward_model.max_length`.

The chat differs by family:

- **CodeScaler / AceCoder** (`codescaler_fsdp_workers.py:363-366`):
  ```python
  chat = [
      {"role": "user", "content": prompt},      # problem statement
      {"role": "assistant", "content": response} # extracted code
  ]
  ```
- **Themis** prepends a functional-correctness **system prompt**
  (`codescaler_fsdp_workers.py:373-377`):
  ```python
  if getattr(self, "rm_type", None) == "themis":
      chat = [
          {"role": "system", "content": THEMIS_FUNCTIONAL_CORRECTNESS_SYSTEM_PROMPT},
          *chat,
      ]
  ```
  The system prompt (`codescaler_fsdp_workers.py:17`) is quoted **verbatim** from
  the authoritative Themis evaluation suite
  (`SYSTEM_PROMPT_MAP['Functional_Correctness']` in
  [`coderewardbench-seqcls.py`](https://github.com/iNeil77/Themis/blob/main/Evaluation/Evaluation_Scripts/coderewardbench-seqcls.py)),
  matching its `system → user → assistant` structure and chat-template usage.

---

## 4. Score read-out (forward pass)

`_forward_micro_batch` (`codescaler_fsdp_workers.py:238-314`) produces one scalar
per sample, but the pooling differs by family:

- **Themis (sequence classifier).** The head already pools to `(batch_size, 1)`,
  so the worker reads it directly and **skips** the remove-padding/token-level
  path and the last-token gather (`codescaler_fsdp_workers.py:250-263`):
  ```python
  if getattr(self, "rm_type", None) == "themis":
      output = self.reward_module(input_ids=input_ids,
                                  attention_mask=attention_mask, use_cache=False)
      rm_score = output.logits.squeeze(-1)   # (batch_size, 1) -> (batch_size,)
      return rm_score
  ```
  The classifier locates the final token from the attention mask, so **right
  padding is required** — which `_switch_chat_template` already uses.
- **CodeScaler / AceCoder (token-level).** The model emits a per-token logit; the
  worker selects the **last valid token** (`codescaler_fsdp_workers.py:296-298`):
  ```python
  eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)
  rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
  ```
  (with an optional remove-padding/flash-varlen path for efficiency).

All families return `(batch_size,)`. `_expand_to_token_level`
(`codescaler_fsdp_workers.py:316-331`) then scatters the scalar onto the last
response token and slices to the response region, yielding `rm_scores`.

> **Doc note on CodeScaler-8B:** the model card and README load CodeScaler-8B with
> `AutoModelForSequenceClassification`, while the training worker loads it with
> `AutoModelForTokenClassification` + an explicit last-token gather. For Qwen3
> these are numerically equivalent (same `score` head; the gather reproduces
> seqcls pooling), so existing runs are correct. Themis is loaded as a true
> sequence classifier to match its authoritative inference script.

---

## 5. Training scripts

| Script | Reward model | Layout |
|---|---|---|
| `scripts/train_codescaler.sh` | `LARK-Lab/CodeScaler-8B` | single node, colocated |
| `scripts/train_themis.sh` | `project-themis/Themis-RM-8B` (any size) | single node, colocated |
| `scripts/train_themis_32b_multinode.sh` | `project-themis/Themis-RM-32B` | 4 nodes × 8 GPUs, colocated |

All three keep `reward_model.reward_manager=codescaler` (that routes to
`CodeScalerRewardModelWorker`, which holds the family dispatch) and differ mainly
in `reward_model.model.path`. The Themis scripts set `reward_model.max_length=4096`
to match the authoritative Themis inference default.

---

## 6. GPU placement: how the models are stored on a node

Two layers: **(A) which GPUs each model lives on** (Ray placement) and **(B) how a
model's parameters are split across those GPUs** (FSDP).

### A. Resource pools and role → pool mapping

`init_resource_pool_mgr` (`main_codescaler.py:167-190`) builds:
```python
global_pool = [trainer.n_gpus_per_node] * trainer.nnodes   # e.g. [8] or [8,8,8,8]
```
Each pool becomes Ray placement groups — **one PG per node, one bundle
(`{CPU:1, GPU:1}`) per GPU**, scheduled `STRICT_PACK` (a node's bundles stay on
that node). `world_size = sum(process_on_nodes)`.

Role mapping (`main_codescaler.py:185-227`):

| Role | Pool |
|---|---|
| ActorRollout | `global_pool` |
| Critic | `global_pool` (absent for GRPO — no critic) |
| RefPolicy | `global_pool` (only if KL is used; off in shipped scripts) |
| RewardModel | `global_pool` if `enable_resource_pool=False` (**default**), else `reward_pool` |

**Colocated (default).** Every role assigned to `global_pool` is grouped into one
`WorkerDict` (`verl/single_controller/ray/base.py`, `create_colocated_worker_cls`)
with **one Ray actor per GPU** holding a shard of the policy *and* a shard of the
reward model. They **time-share** the GPU: rollout → log-prob → reward scoring →
GRPO update, sequentially. This is the "hybrid engine."

**Separate pool (`enable_resource_pool=True`).** RewardModel →
`reward_pool = [reward_model.n_gpus_per_node] * reward_model.nnodes`, on distinct
GPUs (`main_codescaler.py:176-183, 216-219`). Policy and RM no longer contend for
memory and can run concurrently. Total GPUs = `sum(global_pool) + sum(reward_pool)`.

### B. FSDP sharding modes

Both the policy (`verl/workers/fsdp_workers.py`) and the RM
(`codescaler_fsdp_workers.py`) build a device mesh via
`create_device_mesh(world_size, fsdp_size)` (`fsdp_workers.py:98-105`) and map mesh
rank → strategy (`fsdp_workers.py:108-117`):

| `fsdp_size` | Mesh | Strategy | Meaning |
|---|---|---|---|
| `-1` (or ≥ world_size) | 1-D `(world_size,)` | `FULL_SHARD` (ZeRO-3) | every param/grad/optimizer state sharded across **all** GPUs |
| `k` (0 < k < world_size) | 2-D `(world_size//k, k)` | `HYBRID_SHARD` | FULL_SHARD **within** each k-GPU group, **replicate** across groups |

Per-family offload:

- **Actor (policy):** mixed precision; FSDP CPU-offload **off** during the step
  (`fsdp_workers.py:471-472` notes it breaks grad accumulation); the scripts set
  `param_offload`/`optimizer_offload=True` to offload **between** steps.
- **Reference policy:** `CPUOffload(offload_params=True)` (n/a in shipped scripts).
- **Reward model:** bf16, `classifier_dropout=0.0`, and **`CPUOffload(offload_params
  =True)` is hardcoded** in the RM FSDP wrap — its weights live in host RAM and
  stream to GPU only during `compute_rm_score`. This is what makes colocating an
  8B policy + (8B–32B) RM on the same GPUs feasible.

The **rollout (vLLM)** engine uses its own mesh `(dp, infer_tp)` with
`infer_tp = tensor_model_parallel_size × data_parallel_size`
(`fsdp_workers.py:578`). With `tensor_model_parallel_size=1` rollout is pure
data-parallel (one vLLM replica per GPU); FSDP-sharded actor weights are gathered
and resharded into vLLM's layout at the start of each rollout.

### Worked example — single node (`train_codescaler.sh`, 1 × 8 GPUs)

- `world_size=8`; one PG, 8 bundles, `STRICT_PACK`.
- `enable_resource_pool=False` → policy + RM colocated on all 8 GPUs, one
  `WorkerDict` actor per GPU.
- No critic (GRPO), no ref worker (KL off).
- `fsdp_size=-1` → 1-D 8-rank mesh → `FULL_SHARD` for both models.
- RM bf16 + CPU-offloaded; actor param/optimizer offloaded between steps.

---

## 7. Multi-node (AWS): `train_themis_32b_multinode.sh`

Colocated layout across **4 nodes × 8 GPUs = 32 GPUs**, training Qwen3-8B-Base
against Themis-RM-32B. The same pattern works for any RM — change `rm_path`.

### Ray vs. torch master

The `HEAD_IP`/`HEAD_PORT` you supply is the **Ray head** address (used to join the
4 nodes into one cluster). VeRL derives the `torch.distributed`
`MASTER_ADDR`/`MASTER_PORT` automatically from the rank-0 worker's placement group
(`verl/single_controller/ray/base.py:83-88`, `352-405`). You do **not** set torch's
master yourself.

### Launch procedure

```bash
# On each of the 3 WORKER nodes (join Ray and idle; they only host actors):
HEAD_IP=<head-private-ip> HEAD_PORT=6379 ROLE=worker bash scripts/train_themis_32b_multinode.sh

# On the HEAD node (starts Ray head, waits for all 32 GPUs, then drives training):
HEAD_IP=<head-private-ip> HEAD_PORT=6379 ROLE=head   bash scripts/train_themis_32b_multinode.sh
```

Only the head runs the Python driver (`TaskRunner` always runs on the head; workers
never execute it). The head **blocks until all 32 GPUs register**, otherwise
placement-group creation (`pg.ready()`) hangs/fails.

### Config deltas vs. single node

- `trainer.nnodes=4` → `global_pool=[8,8,8,8]`, `world_size=32`.
- `fsdp_size=8` applied to **both** the actor
  (`actor_rollout_ref.actor.fsdp_config.fsdp_size`) and the RM
  (`reward_model.model.fsdp_config.fsdp_size`) → `HYBRID_SHARD` (shard within a
  node's 8 GPUs over NVLink, replicate across the 4 nodes). Cross-node traffic is
  then only gradient all-reduce, not per-layer param all-gathers.
- `reward_model.model.fsdp_config.param_offload=True` (the worker also hardcodes
  CPU offload).

### AWS networking (not part of the recipe)

The script sets EFA/NCCL env that multi-node FSDP collectives need:
```bash
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_SOCKET_IFNAME=ens     # adjust to your ENI (ip -o link)
export NCCL_DEBUG=INFO
# export NCCL_PROTO=simple        # if EFA throughput/hangs require it
```
Also required:
- **Security group:** allow Ray ports between nodes **and** all traffic within the
  SG itself (self-referencing rule) for EFA/NCCL.
- **Homogeneity:** each node must expose ≥8 GPUs (`STRICT_PACK`); identical image,
  env, and code on all nodes.
- **Model staging:** pre-stage weights on shared FSx/EFS or bake into the AMI so
  the 4 nodes don't each re-download (~64 GB for the 32B RM). `copy_to_local`
  returns local/HF paths unchanged, so a shared-filesystem path works cleanly.

---

## 8. Out-of-memory levers (colocated 32B RM on 4 nodes)

Ordered roughly by impact (full list at the bottom of
`scripts/train_themis_32b_multinode.sh`):

1. **Give the reward model its own nodes** —
   `reward_model.enable_resource_pool=True` with `reward_model.nnodes` /
   `reward_model.n_gpus_per_node` (e.g. 3 policy nodes + 1 RM node, `trainer.nnodes=3`).
   Biggest win: the 32B RM stops competing for policy HBM/host-RAM.
2. **`fsdp_size=-1`** → `FULL_SHARD` across all 32 GPUs (each GPU holds 1/32 of each
   model vs 1/8 under HYBRID). Lowest per-GPU footprint; costs inter-node bandwidth.
3. **Lower `actor_rollout_ref.rollout.gpu_memory_utilization`** (0.7 → 0.5/0.4) to
   shrink vLLM's up-front KV reservation.
4. **`actor_rollout_ref.rollout.tensor_model_parallel_size=2/4`** to shard the
   policy inside vLLM.
5. **Cut activations** — `ppo_max_token_len_per_gpu`, micro-batch sizes,
   `log_prob_micro_batch_size_per_gpu` (gradient checkpointing + activation offload
   already on).
6. **Cut RM activations** — `reward_model.max_length` (4096 → 2048),
   `reward_model.forward_max_token_len_per_gpu`, `reward_model.micro_batch_size_per_gpu=1`.
7. **Shorten / shard sequences** — `data.max_response_length`, or
   `ulysses_sequence_parallel_size=2/4`.
8. **Keep offload on** — actor param/optimizer offload between steps; the RM is
   CPU-offloaded by the worker. Ensure host RAM is sufficient: under `HYBRID_SHARD`
   a node's CPU holds a full bf16 copy of the 32B RM (~64 GB) plus the actor's
   offloaded state. Comfortable on p4d/p5-class hosts; on smaller-RAM instances
   prefer `fsdp_size=-1` to spread the offload.

---

## 9. Adding a new reward-model family

1. **Architecture selection** — add an `elif '<name>' in local_path.lower():`
   branch in `_build_model` (`codescaler_fsdp_workers.py:140-188`) that sets
   `self.rm_type = "<name>"` and loads the model with the correct HF class/dtype.
   Leave the final `else: raise ValueError` as the catch-all.
2. **Score pooling** — if the model is not token-level, add an
   `if self.rm_type == "<name>":` branch in `_forward_micro_batch`
   (`codescaler_fsdp_workers.py:238-314`) that returns a `(batch_size,)` scalar
   (e.g. read pooled `logits`, bypassing the last-token gather). Mind padding side.
3. **Prompt formatting** — if the model needs a system prompt or a different chat
   shape, add an `if self.rm_type == "<name>":` branch in `_switch_chat_template`
   (`codescaler_fsdp_workers.py:333-411`). Keep using the RM's own chat template.
4. **Training script** — copy `scripts/train_themis.sh`, set `rm_path` to a path
   containing your substring, and adjust `reward_model.max_length`.

The `raise ValueError` guard ensures a new family is a deliberate, reviewable
change rather than a silent fallback.

---

## 10. Key config keys

| Key | Effect |
|---|---|
| `reward_model.enable` | turns on the served scalar RM worker |
| `reward_model.reward_manager` | `codescaler` → `CodeScalerRewardManager` + worker |
| `reward_model.model.path` | RM checkpoint; **also selects the family** (substring) |
| `reward_model.model.input_tokenizer` | policy tokenizer; non-null enables decode→extract→re-template |
| `reward_model.max_length` | RM input truncation (4096 for the shipped scripts) |
| `reward_model.model.fsdp_config.fsdp_size` | RM FSDP mesh: `-1` FULL_SHARD, `k` HYBRID_SHARD |
| `reward_model.model.fsdp_config.param_offload` | RM CPU param offload |
| `reward_model.enable_resource_pool` / `nnodes` / `n_gpus_per_node` | give the RM its own GPU pool |
| `reward_model.launch_reward_fn_async` | run the reward manager in a Ray worker |
| `reward_model.use_partial_reward` / `reward_shaping` | partial credit (test-time) / softplus shaping |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | actor FSDP mesh |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | vLLM rollout TP |
| `trainer.nnodes` / `trainer.n_gpus_per_node` | global pool size = `world_size` |
| `algorithm.adv_estimator` | `grpo` in the shipped scripts |
