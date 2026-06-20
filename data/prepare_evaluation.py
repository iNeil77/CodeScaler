from datasets import load_dataset
from system_prompts import *
import json
import os
import uuid
from datasets import Dataset
from datasets import concatenate_datasets
from datasets import Value


def _cast_ground_truth_large(ds):
    """Store reward_model.ground_truth as Arrow large_string (64-bit offsets) so
    consumers can load/concatenate the whole split without overflowing pyarrow's 2GB
    int32 string-offset limit (lcbv5 ground_truth blobs reach ~200MB). Values unchanged."""
    feats = ds.features.copy()
    rm = feats["reward_model"]
    if rm["ground_truth"].dtype != "large_string":
        rm["ground_truth"] = Value("large_string")
        feats["reward_model"] = rm
        ds = ds.cast(feats)
    return ds


# Pinned HF dataset revisions (resolved 2026-06-20). Mirrors data/prepare_deepcoder.py
# and data/download_data.py.
HF_REVISIONS = {
    "agentica-org/DeepCoder-Preview-Dataset": "177913a7bd43791646ef6a43645caa3c871ab3db",
    "Gen-Verse/LiveBench": "fa070cf11dccf8bab0bdea01901649bc17222aeb",
    "Gen-Verse/CodeContests": "e06e6e140899dddc8ef841255db0b050b6da27d6",
    "Gen-Verse/CodeForces": "b3b3df092edb8f5412a6c4f83ec8753bcd9de943",
}

# Map data_source labels to the SOURCE prefix used in extra_info["id"] (SOURCE_UUID).
_SOURCE_PREFIX = {
    "lcbv5": "LCB_V5",
    "CodeContests": "CODECONTESTS",
    "CodeForces": "CODEFORCES",
    "LiveBench": "LIVEBENCH",
}


def _make_extra_info(data_source, index, split="test"):
    """verl extra_info dict. `id` is SOURCE_UUID (SOURCE capitalized, UUID is uuid4)."""
    prefix = _SOURCE_PREFIX.get(data_source, data_source.upper())
    return {
        "split": split,
        "index": index,
        "id": f"{prefix}_{uuid.uuid4()}",
    }


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


# stdin/stdout system prompt applied as a separate `system` turn to the evaluation
# sets, so they match the LiveCodeBench-style prompt used for training. Same content
# as fetch_live_code_bench_system_prompt's no-starter-code branch, problem text kept
# in the `user` turn. See data/prepare_deepcoder.py for the identical training-side
# constant.
STDIN_SYSTEM_PROMPT = (
    LCB_SYSTEM_MESSAGE_GENERIC
    + "\n\n"
    + f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
    + "```python\n# YOUR CODE HERE\n```\n\n"
    + "### Answer: (use the provided format with backticks)"
)


def _lcb_is_stdin(item) -> bool:
    """lcbv5 is mixed; keep only stdin/stdout problems (no starter_code, no functional
    func_name / testtype)."""
    if item.get("starter_code"):
        return False
    md = item.get("metadata") or {}
    if md.get("func_name"):
        return False
    try:
        tests = json.loads(item["tests"])
    except (KeyError, TypeError, ValueError):
        return True
    if isinstance(tests, list) and tests and isinstance(tests[0], dict):
        if tests[0].get("testtype") == "functional":
            return False
    return True

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
            
def _build_lcb_example(item, idx):
    """Transform one LiveCodeBench record into the eval schema. Pure function, safe
    for datasets.map(num_proc=..., with_indices=True)."""
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

    if isinstance(question, dict):
        question = json.dumps(question)

    # Separate stdin/stdout system turn; problem stays in the user turn. raw_prompt is
    # user-only so the reward-model worker (raw_prompt[0]['content']) scores the real
    # problem, not the instruction.
    user_msg = {"role": "user", "content": question}
    return {
        "data_source": 'lcbv5',
        "question": ori_question,
        "prompt": [{"role": "system", "content": STDIN_SYSTEM_PROMPT}, user_msg],
        "raw_prompt": [user_msg],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": tests},
        "extra_info": _make_extra_info("lcbv5", idx),
    }


def construct_lcb(ds_lcb):
    # Keep only stdin/stdout problems (drop function-style / starter-code items).
    n_before = len(ds_lcb)
    ds_lcb = ds_lcb.filter(_lcb_is_stdin, num_proc=_num_proc(len(ds_lcb)), desc="Filtering lcbv5 -> stdin only")
    if len(ds_lcb) != n_before:
        print(f"  lcbv5: kept {len(ds_lcb)}/{n_before} stdin/stdout problems (dropped {n_before - len(ds_lcb)} function-style)")
    return ds_lcb.map(
        _build_lcb_example,
        with_indices=True,
        num_proc=_num_proc(len(ds_lcb)),
        remove_columns=ds_lcb.column_names,
        # lcbv5 has individual ground_truth blobs up to ~200MB; write one example per
        # Arrow chunk so a chunk's string column never overflows pyarrow's 2GB offset
        # ("offset overflow while concatenating arrays").
        writer_batch_size=1,
        desc="Building lcbv5",
    )


def construct_test_dataset(ds, dataset_name):
    # ds here is a plain list loaded from JSON (CodeContests/CodeForces/LiveBench);
    # these benchmarks are stdin/stdout (exe_method='stdin'). Separate system turn,
    # user-only raw_prompt.
    outputs = []
    for idx, item in enumerate(ds):
        question = item['question']
        tests = convert_test(item['test_input'], item['test_output'])
        tests = json.dumps(tests)

        user_msg = {"role": "user", "content": question}
        data = {
            "data_source": dataset_name,
            "question": item['question'],
            "prompt": [{"role": "system", "content": STDIN_SYSTEM_PROMPT}, user_msg],
            "raw_prompt": [user_msg],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": tests},
            "extra_info": _make_extra_info(dataset_name, idx),
        }
        outputs.append(data)

    return Dataset.from_list(outputs)

def main():
    ds_lcbv5 = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "lcbv5",
                            revision=HF_REVISIONS["agentica-org/DeepCoder-Preview-Dataset"])
    ds_codecontests = load_json("./data/CodeContests.json")
    ds_codeforces = load_json("./data/CodeForces.json")
    ds_livebench = load_json("./data/LiveBench.json")

    ds_all = []

    split = "test"
    ds_test_lcbv5 = construct_lcb(ds_lcbv5[split])
    ds_all.append(ds_test_lcbv5)
    # Store ground_truth as large_string (lcbv5 blobs reach ~200MB) so HF can load and
    # concatenate the whole split without overflowing pyarrow's 2GB int32 offset limit.
    # batch_size=1 also keeps the *write* under the per-chunk limit.
    _cast_ground_truth_large(ds_test_lcbv5).to_parquet("./datasets/Evaluation/LiveCodeBench.parquet", batch_size=1)

    dataset_names = {'CodeContests': ds_codecontests,
                     'CodeForces': ds_codeforces,
                     'LiveBench': ds_livebench}

    for dataset_name in dataset_names:
        ds_test = construct_test_dataset(dataset_names[dataset_name], dataset_name)
        ds_all.append(ds_test)
        _cast_ground_truth_large(ds_test).to_parquet(f"./datasets/Evaluation/{dataset_name}.parquet", batch_size=64)

    # combine (includes lcbv5 -> large blobs)
    ds_combine = _cast_ground_truth_large(concatenate_datasets(ds_all))
    ds_combine.to_parquet("./datasets/Evaluation/All.parquet", batch_size=1)


if __name__ == "__main__":
    main()