from datasets import load_dataset
from system_prompts import *
import base64
import json
import os
import pickle
import zlib
from datasets import Dataset, concatenate_datasets


def load_lcb_v6(version_tag="release_v6", filename="test6.jsonl"):
    """Load the LiveCodeBench v6 code-generation problems and shape each record like
    the other DeepCoder sources, so it can flow through _map_source('lcbv6').

    Mirrors the official decoder
    (github.com/LiveCodeBench/LiveCodeBench .../benchmarks/code_generation.py):
    public_test_cases are plain JSON; private_test_cases are JSON, or fall back to
    base64 -> zlib.decompress -> pickle.loads -> json.loads. Each test is
    {input, output, testtype in {stdin, functional}}. fn_name comes from
    metadata.func_name; functional problems are dropped downstream by _is_stdin.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="livecodebench/code_generation_lite",
        repo_type="dataset",
        filename=filename,
    )

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            public = json.loads(rec["public_test_cases"])
            priv_raw = rec["private_test_cases"]
            try:
                private = json.loads(priv_raw)
            except Exception:
                private = json.loads(
                    pickle.loads(zlib.decompress(base64.b64decode(priv_raw.encode("utf-8"))))
                )

            metadata = rec["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata) if metadata else {}

            # Build the same [{input, output, testtype}] tests list shape used by lcbv5,
            # combining public + private cases.
            tests = [
                {"input": t["input"], "output": t["output"], "testtype": t.get("testtype", "stdin")}
                for t in (public + private)
            ]

            records.append(
                {
                    "problem": rec["question_content"],
                    "tests": json.dumps(tests),
                    "starter_code": rec.get("starter_code", "") or "",
                    "metadata": metadata,  # carries func_name for functional problems
                }
            )

    return Dataset.from_list(records)


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


# Stdin/stdout system prompt, applied as a separate `system` turn to the taco and
# primeintellect sources so they look like the LiveCodeBench-style prompt. Same
# content as fetch_live_code_bench_system_prompt's no-starter-code branch, but with
# the problem text kept out (it goes in the `user` turn). Tells the model to read
# stdin / write stdout and to emit its code in a single ```python``` markdown block.
STDIN_SYSTEM_PROMPT = (
    LCB_SYSTEM_MESSAGE_GENERIC
    + "\n\n"
    + f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
    + "```python\n# YOUR CODE HERE\n```\n\n"
    + "### Answer: (use the provided format with backticks)"
)

# Every primeintellect problem begins with this line (100% of the split); it is
# redundant once the LCB system prompt is added, so strip it from the user turn.
_PRIMEINTELLECT_PREAMBLE = "Solve the following coding problem using the programming language python:"


def _strip_primeintellect_preamble(question: str) -> str:
    stripped = question.lstrip()
    if stripped.startswith(_PRIMEINTELLECT_PREAMBLE):
        return stripped[len(_PRIMEINTELLECT_PREAMBLE):].lstrip()
    return question


def _is_stdin(item) -> bool:
    """True if a problem is stdin/stdout style (not function-call graded). We train
    and validate only on stdin/stdout problems. Detection covers all sources:
      - starter_code present  -> function-style (lcbv5 functional items)
      - tests dict has fn_name -> function-style (taco functional items)
      - tests list item testtype == 'functional' -> function-style (lcbv5)
    primeintellect and codeforces have none of these, so they pass through."""
    if item.get("starter_code"):
        return False
    md = item.get("metadata") or {}
    if md.get("func_name"):
        return False
    try:
        tests = json.loads(item["tests"])
    except (KeyError, TypeError, ValueError):
        return True
    if isinstance(tests, dict) and tests.get("fn_name"):
        return False
    if isinstance(tests, list) and tests and isinstance(tests[0], dict):
        if tests[0].get("testtype") == "functional":
            return False
    return True


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

    # All sources are filtered to stdin/stdout problems and get the same scheme: a
    # separate `system` turn carrying the stdin/stdout + markdown-block instruction,
    # so every prompt looks like LiveCodeBench. primeintellect's redundant leading
    # "Solve ... python:" line is stripped from the user turn.
    if data_source == 'primeintellect':
        question = _strip_primeintellect_preamble(question)

    if isinstance(question, dict):
        question = json.dumps(question)

    # `prompt` is what the policy sees (system + user). `raw_prompt` is the
    # user-only problem text: the reward-model worker reads raw_prompt[0]['content']
    # as the problem, so the system instruction must NOT leak into raw_prompt.
    user_msg = {"role": "user", "content": question}
    prompt = [{"role": "system", "content": STDIN_SYSTEM_PROMPT}, user_msg]

    return {
        "data_source": data_source,
        "question": ori_question,
        "prompt": prompt,
        "raw_prompt": [user_msg],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": tests},
    }


def _map_source(ds, data_source):
    """Map a single source dataset to the target schema in parallel, dropping the
    original columns so the per-source schemas line up for concatenation."""
    # Keep only stdin/stdout problems (drop function-call / starter-code items). lcbv5
    # and taco are mixed; primeintellect and codeforces are already all stdin (no-op).
    n_before = len(ds)
    ds = ds.filter(_is_stdin, num_proc=_num_proc(len(ds)), desc=f"Filtering {data_source} -> stdin only")
    if len(ds) != n_before:
        print(f"  {data_source}: kept {len(ds)}/{n_before} stdin/stdout problems (dropped {n_before - len(ds)} function-style)")
    # The ground_truth test strings can be very large: lcbv5 has individual blobs up to
    # ~200MB, so even a few per Arrow chunk overflow pyarrow's 2GB 32-bit offset
    # ("offset overflow while concatenating arrays"). Use batch size 1 for lcbv5 (small
    # split) and a small batch elsewhere.
    writer_batch_size = 1 if data_source == 'lcbv5' else 64
    return ds.map(
        lambda item: _build_example(item, data_source),
        num_proc=_num_proc(len(ds)),
        remove_columns=ds.column_names,
        writer_batch_size=writer_batch_size,
        desc=f"Building {data_source}",
    )


def construct_train_dataset(ds_lcbv5, ds_primeintellect, ds_taco):
    # Mapped per source and concatenated in the original [lcbv5, primeintellect, taco]
    # order, so the downstream shuffle(seed=42) yields the same dataset as before.
    ds_list = [ds_lcbv5, ds_primeintellect, ds_taco]
    data_source_list = ['lcbv5', 'primeintellect', 'taco']
    mapped = [_map_source(ds, src) for ds, src in zip(ds_list, data_source_list)]
    return concatenate_datasets(mapped)


def construct_val_dataset(ds_codeforces, ds_lcbv5, ds_lcbv6):
    # Validation set exec-evaluated mid-training: codeforces + LiveCodeBench v5 + v6,
    # tagged by data_source. All routed through the same stdin-only scheme.
    ds_list = [ds_codeforces, ds_lcbv5, ds_lcbv6]
    data_source_list = ['codeforces', 'lcbv5', 'lcbv6']
    mapped = [_map_source(ds, src) for ds, src in zip(ds_list, data_source_list)]
    return concatenate_datasets(mapped)

def main():
    ds_codeforces = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "codeforces")
    ds_lcbv5 = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "lcbv5")
    ds_primeintellect = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "primeintellect")
    ds_taco = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "taco")
    ds_lcbv6 = load_lcb_v6()

    split = "train"
    ds_train = construct_train_dataset(ds_lcbv5[split], ds_primeintellect[split], ds_taco[split])

    # Validation: codeforces + lcbv5 (DeepCoder test split) + lcbv6 (LiveCodeBench v6).
    ds_val = construct_val_dataset(ds_codeforces["test"], ds_lcbv5["test"], ds_lcbv6)

    ds_train = ds_train.shuffle(seed=42)
    ds_val = ds_val.shuffle(seed=42)

    # lcbv5/lcbv6 ground_truth blobs reach ~200MB; batch_size=1 avoids pyarrow's 2GB
    # offset overflow on write.
    ds_train.to_parquet("./datasets/DeepCoder/train.parquet", batch_size=1)
    ds_val.to_parquet("./datasets/DeepCoder/val.parquet", batch_size=1)

if __name__ == "__main__":
    main()