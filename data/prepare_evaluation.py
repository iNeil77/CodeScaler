from datasets import load_dataset
from system_prompts import *
import json
import os
from datasets import Dataset
from datasets import concatenate_datasets


def _num_proc(n_rows, min_chunk=256):
    """num_proc for datasets.map that scales with available cores while keeping each
    worker busy (>= min_chunk rows). Uses sched_getaffinity (true core budget)."""
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(cores, max(1, n_rows // min_chunk)))

def fetch_live_code_bench_system_prompt(prompt: str, starter_code: str | None = None):
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    prompt = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + prompt
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt

def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def convert_test(test_input, test_output):
    outputs = []
    for inp, out in zip(test_input, test_output):
        outputs.append({
            "input": inp,
            "output": out
        })
    return outputs
            
def _build_lcb_example(item):
    """Transform one LiveCodeBench record into the eval schema. Pure function, safe
    for datasets.map(num_proc=...)."""
    question = item["problem"]
    ori_question = question
    tests = json.loads(item["tests"])

    if item.get("metadata", {}):
        assert "func_name" in item["metadata"], f"Function name is not found, check if your LCB data is preprocessed correctly: {item['metadata']}"
        if isinstance(tests, dict):
            tests["metadata"] = item["metadata"]
        else:
            for test in tests:
                assert isinstance(test, dict), "Test is not a dict"
                test["metadata"] = item["metadata"]

    tests = json.dumps(tests)

    starter_code = item.get("starter_code", None)
    question = fetch_live_code_bench_system_prompt(question, starter_code)

    if isinstance(question, dict):
        question = json.dumps(question)

    return {
        "data_source": 'lcbv5',
        "question": ori_question,
        "prompt": [{"role": "user", "content": question}],
        "raw_prompt": [{"role": "user", "content": question}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": tests},
    }


def construct_lcb(ds_lcb):
    return ds_lcb.map(
        _build_lcb_example,
        num_proc=_num_proc(len(ds_lcb)),
        remove_columns=ds_lcb.column_names,
        # Keep Arrow write batches small: ground_truth test strings can be very large
        # and overflow pyarrow's 32-bit offset when chunks are combined.
        writer_batch_size=64,
        desc="Building lcbv5",
    )


def construct_test_dataset(ds, dataset_name):
    # ds here is a plain list loaded from JSON; build via from_list (small splits).
    outputs = []
    for item in ds:
        question = fetch_live_code_bench_system_prompt(item['question'])
        tests = convert_test(item['test_input'], item['test_output'])
        tests = json.dumps(tests)

        data = {
            "data_source": dataset_name,
            "question": item['question'],
            "prompt": [{"role": "user", "content": question}],
            "raw_prompt": [{"role": "user", "content": question}],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": tests},
        }
        outputs.append(data)

    ds = Dataset.from_list(outputs)
    return ds

def main():
    ds_lcbv5 = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "lcbv5")
    ds_codecontests = load_json("./data/CodeContests.json")
    ds_codeforces = load_json("./data/CodeForces.json")
    ds_livebench = load_json("./data/LiveBench.json")
    ds_mbpp = load_json("./data/MBPP.json")

    ds_all = []

    split = "test"
    ds_test_lcbv5 = construct_lcb(ds_lcbv5[split])
    ds_all.append(ds_test_lcbv5)
    ds_test_lcbv5.to_parquet("./datasets/Evaluation/LiveCodeBench.parquet")

    dataset_names = {'CodeContests': ds_codecontests,
                     'CodeForces': ds_codeforces,
                     'LiveBench': ds_livebench,
                     'MBPP': ds_mbpp}

    for dataset_name in dataset_names:
        ds_test = construct_test_dataset(dataset_names[dataset_name], dataset_name)
        ds_all.append(ds_test)
        ds_test.to_parquet(f"./datasets/Evaluation/{dataset_name}.parquet")

    # combine
    ds_combine = concatenate_datasets(ds_all)
    ds_combine.to_parquet("./datasets/Evaluation/All.parquet")


if __name__ == "__main__":
    main()