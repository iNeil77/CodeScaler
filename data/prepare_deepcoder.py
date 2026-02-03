from datasets import load_dataset
from system_prompts import *
import json
from datasets import Dataset

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


def construct_train_dataset(ds_lcbv5, ds_primeintellect, ds_taco):
    outputs = []
    ds_list = [ds_lcbv5, ds_primeintellect, ds_taco]
    data_source_list = ['lcbv5', 'primeintellect', 'taco']
    for i in range(len(ds_list)):
        ds = ds_list[i]
        data_source = data_source_list[i]
        for item in ds:
            question = item.pop("problem")
            ori_question = question
            tests = item.pop("tests")
            tests = json.loads(tests)

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

            data = {
                "data_source": data_source,
                "question": ori_question,
                "prompt": [{"role": "user", "content": question}],
                "raw_prompt": [{"role": "user", "content": question}],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": tests},
            }

            outputs.append(data)
    ds = Dataset.from_list(outputs)
    return ds
            
def construct_test_dataset(ds_codeforces, ds_lcbv5):
    outputs = []
    ds_list = [ds_codeforces, ds_lcbv5]
    data_source_list = ['codeforces', 'lcbv5']
    for i in range(len(ds_list)):
        ds = ds_list[i]
        data_source = data_source_list[i]
        for item in ds:
            question = item.pop("problem")
            ori_question = question
            tests = item.pop("tests")

            tests = json.loads(tests)

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

            data = {
                "data_source": data_source,
                "question": ori_question,
                "prompt": [{"role": "user", "content": question}],
                "raw_prompt": [{"role": "user", "content": question}],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": tests},
            }

            outputs.append(data)
    ds = Dataset.from_list(outputs)
    return ds

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