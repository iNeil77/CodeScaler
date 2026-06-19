from datasets import load_dataset
from system_prompts import *
import json
import os
from datasets import Dataset, concatenate_datasets


def _num_proc(n_rows, min_chunk=256):
    """Pick a num_proc for datasets.map that scales with available cores but keeps
    each worker meaningfully busy (>= min_chunk rows), avoiding process-spawn overhead
    on small splits. This machine exposes many cores (os.sched_getaffinity)."""
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


def _build_example(item, data_source):
    """Transform one raw record into the training schema. Pure function of (item,
    data_source) so it is safe to run under datasets.map(num_proc=...)."""
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

    if data_source == 'lcbv5':
        starter_code = item.get("starter_code", None)
        question = fetch_live_code_bench_system_prompt(question, starter_code)

    if isinstance(question, dict):
        question = json.dumps(question)

    return {
        "data_source": data_source,
        "question": ori_question,
        "prompt": [{"role": "user", "content": question}],
        "raw_prompt": [{"role": "user", "content": question}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": tests},
    }


def _map_source(ds, data_source):
    """Map a single source dataset to the target schema in parallel, dropping the
    original columns so the per-source schemas line up for concatenation."""
    return ds.map(
        lambda item: _build_example(item, data_source),
        num_proc=_num_proc(len(ds)),
        remove_columns=ds.column_names,
        # The ground_truth test strings can be very large (esp. TACO); keep the Arrow
        # write batches small so a single chunk's string column never exceeds the 2GB
        # offset limit (pyarrow "offset overflow while concatenating arrays").
        writer_batch_size=64,
        desc=f"Building {data_source}",
    )


def construct_train_dataset(ds_lcbv5, ds_primeintellect, ds_taco):
    # Mapped per source and concatenated in the original [lcbv5, primeintellect, taco]
    # order, so the downstream shuffle(seed=42) yields the same dataset as before.
    ds_list = [ds_lcbv5, ds_primeintellect, ds_taco]
    data_source_list = ['lcbv5', 'primeintellect', 'taco']
    mapped = [_map_source(ds, src) for ds, src in zip(ds_list, data_source_list)]
    return concatenate_datasets(mapped)


def construct_test_dataset(ds_codeforces, ds_lcbv5):
    ds_list = [ds_codeforces, ds_lcbv5]
    data_source_list = ['codeforces', 'lcbv5']
    mapped = [_map_source(ds, src) for ds, src in zip(ds_list, data_source_list)]
    return concatenate_datasets(mapped)

def main():
    ds_codeforces = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "codeforces")
    ds_lcbv5 = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "lcbv5")
    ds_primeintellect = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "primeintellect")
    ds_taco = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "taco")

    split = "train"
    ds_train = construct_train_dataset(ds_lcbv5[split], ds_primeintellect[split], ds_taco[split])
    
    split = "test"
    ds_test = construct_test_dataset(ds_codeforces[split], ds_lcbv5[split])

    ds_train = ds_train.shuffle(seed=42)
    ds_test = ds_test.shuffle(seed=42)

    ds_train.to_parquet("./datasets/DeepCoder/train.parquet")
    ds_test.to_parquet("./datasets/DeepCoder/test.parquet")

if __name__ == "__main__":
    main()