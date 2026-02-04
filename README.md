<h2 align="center">
  CodeScaler: Scaling Code LLM Training and Test-Time Inference via Execution-Free Reward Models
</h2>

<p align="center">
  <a href="">
    <img
      src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&logoColor=red"
      alt="CodeScaler Paper on arXiv"
    />
  <a href="https://github.com/LARK-AI-Lab/CodeScaler">
    <img 
        src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white" 
        alt="GitHub Code"
    />
  </a>
  <a href="https://lark-ai-lab.github.io/codescaler.github.io/">
    <img 
        src="https://img.shields.io/badge/GitHub-Page-4078c0?logo=github&logoColor=white" 
        alt="GitHub Page"
    />
  </a>
  <a href="https://huggingface.co/collections/LARK-Lab/codescaler">
    <img 
        src="https://img.shields.io/badge/Datasets-Hugging%20Face%20Data-orange?logo=huggingface&logoColor=yellow" 
        alt="Datasets on Hugging Face"
    />
  </a>
  <a href="https://huggingface.co/collections/LARK-Lab/codescaler">
    <img 
        src="https://img.shields.io/badge/CodeScaler-Hugging%20Face%20Model-FFCC00?logo=huggingface&logoColor=yellow" 
        alt="CodeScaler on Hugging Face"
    />
  </a>

  
</p>

## 📊 Overview

<p align="center">
  <img src="assets/overview.png"  alt="Overview of models"  width="800">
</p>

- We propose **CodeScaler**, an execution-free reward model designed to scale both reinforcement learning training and test-time inference for code generation. **CodeScaler** is trained on carefully curated preference data derived from verified code problems and incorporates syntax-aware code extraction and validity-preserving reward shaping to ensure stable and robust optimization. 

- Across five coding benchmarks, **CodeScaler** improves Qwen3-8B-Base by an average of **+11.72** points, outperforming binary execution-based RL by **+1.82** points, and enables scalable reinforcement learning on synthetic datasets without any test cases. 

- At inference time, **CodeScaler** serves as an effective test-time scaling method, achieving performance comparable to unit test approaches while providing a **10×** reduction in latency. Moreover, **CodeScaler** surpasses existing reward models on RM-Bench not only in the code domain but also in general and reasoning domains.

## 📰 News

- **[2024-02]** 🎉 We have released the [code](https://github.com/LARK-AI-Lab/CodeScaler), [dataset](https://huggingface.co/collections/LARK-Lab/codescaler) and [models](https://huggingface.co/collections/LARK-Lab/codescaler) for CodeScaler!

<!-- - **[2024-02]** 📝 CodeScaler paper is available on arXiv. -->

## 📚 Datasets

- [CodeScalerPair-51K](https://huggingface.co/datasets/LARK-Lab/CodeScalerPair-51K): We construct high-quality preference data from on-policy training trajectories.

## 🤖 Models
 We release CodeScaler at different scales from 1.7B, 4B to 8B.
 - [CodeScaler-1.7B](https://huggingface.co/LARK-Lab/CodeScaler-1.7B): A reward model trained on CodeScalerPair-51K from Skywork/Skywork-Reward-V2-Qwen3-1.7B.

  - [CodeScaler-4B](https://huggingface.co/LARK-Lab/CodeScaler-4B): A reward model trained on CodeScalerPair-51K from Skywork/Skywork-Reward-V2-Qwen3-4B.

   - [CodeScaler-8B](https://huggingface.co/LARK-Lab/CodeScaler-8B): A reward model trained on CodeScalerPair-51K from Skywork/Skywork-Reward-V2-Qwen3-8B.

## 🚀 Quick Start

### ⚙️ Environment Setup

**Step 1: Clone the repository**

```bash
git clone https://github.com/LARK-AI-Lab/CodeScaler.git
cd CodeScaler
```

**Step 2: Create a conda environment**

```bash
conda create -n CodeScaler python==3.10.19
conda activate CodeScaler
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Install FlashAttention**

```bash
pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

> 💡 **Tip:** You can also install [FlashAttention](https://github.com/Dao-AILab/flash-attention) based on your specific PyTorch and CUDA versions for optimal performance.

### 📦 Data Preparation

Prepare the training and evaluation datasets:

```bash
# Prepare training dataset
python data/prepare_deepcoder.py

# Download and prepare evaluation dataset
python data/download_dataset.py
python data/prepare_evaluation.py
```

> 💡 **Tip:** The training dataset is based on DeepCoder training datasets, and evaluation includes multiple coding benchmarks.

### 🏋️ Training

Train Qwen3-8B-Base on DeepCoder dataset using CodeScaler as reward model:

```bash
# Login to Weights & Biases for experiment tracking
wandb login

# Start training
bash scripts/train.sh
```

> 💡 **Tip:** Check `scripts/train.sh` to customize hyperparameters such as learning rate, batch size, and training epochs.

### 📈 Evaluation

Evaluate your trained model:

```bash
# Run evaluation on benchmarks
bash scripts/eval.sh
```

### 💻 Use CodeScaler for RM Scoring
````python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = 'LARK-Lab/CodeScaler-8B'

tokenizer = AutoTokenizer.from_pretrained(model_path)
reward_model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
reward_model.eval()

question = """\
Given an integer array nums and an integer k, return the total number of continuous subarrays whose sum equals k.
A subarray is a contiguous part of the array.
For example:
```
Input:
nums = [1, 1, 1], k = 2

Output:
2
```
"""

# Correct solution using prefix sum approach
program_correct = """\
from collections import defaultdict

def subarraySum(nums, k):
    prefix = 0
    count = 0
    freq = defaultdict(int)
    freq[0] = 1  # Important: subarray starting from index 0

    for num in nums:
        prefix += num

        if prefix - k in freq:
            count += freq[prefix - k]

        freq[prefix] += 1

    return count
"""

# Incorrect solution using sliding window (fails on negative numbers)
program_wrong = """\
def subarraySum(nums, k):
    left = 0
    curr_sum = 0
    count = 0

    for right in range(len(nums)):
        curr_sum += nums[right]

        while curr_sum > k and left <= right:
            curr_sum -= nums[left]
            left += 1

        if curr_sum == k:
            count += 1

    return count
"""


convs = [
    [
        {
            "content": question,
            "role": "user",
        },
        {
            "role": "assistant",
            "content": program
        }
    ] for program in [program_correct, program_wrong]
]


texts = [
    tokenizer.apply_chat_template(conv, tokenize=False)
    for conv in convs
]

toks = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=2048,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = reward_model(
        input_ids=toks["input_ids"].to(device),
        attention_mask=toks["attention_mask"].to(device),
    )
    scores = outputs.logits.squeeze(-1).cpu().tolist()


print("RM Scores:", scores)
# RM Scores: [6.5424089431762695, -0.0312652587890625]
````

## 📝 Citation
```

```

