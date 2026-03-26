
import re
import json
import multiprocessing
import ast
import numpy as np

from recipe.codescaler.codescaler_reward_types import RewardOutput
from recipe.codescaler.livecodebench import run_test as lcb_run_test

# EMTPY_STRING = "This is an empty string."
EMTPY_STRING = ""


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except:
        return False



def extract_code_from_model_test(model_response: str):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.
        last_block (bool): if only extract the last code block

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return ""
    code = code_blocks[-1].strip()
    return code

def extract_code_from_model(model_response: str, check_ast=True):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.
        last_block (bool): if only extract the last code block
        check_ast (bool): if apply syntax check

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """

    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return ""
    if len(code_blocks) != 1:
        return ""
    code = code_blocks[-1].strip()
    if check_ast and not is_valid_python(code):
        return ""
    return code

def clean_code_main_block(code: str) -> str:
    """
    Removes `if __name__ == "__main__"` blocks from Python code.

    Args:
        code (str): The input Python code.

    Returns:
        str: Cleaned code without the main execution block.
    """
    code_lines = code.split("\n")
    filtered_lines = []
    skip_block = False

    for line in code_lines:
        if line.strip().startswith('if __name__ == "__main__"') or line.strip().startswith("if __name__ == '__main__'"):
            skip_block = True
            continue
        if skip_block:
            # Check if we're out of the block (less indentation)
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                skip_block = False
            else:
                continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)

def check_correctness(data, model_response, config, use_partial_reward=False):
    data_source = data.non_tensor_batch.get("data_source","")
    # verifiable = data.non_tensor_batch.get("verifiable", False)
    tests = json.loads(data.non_tensor_batch.get("reward_model",{}).get("ground_truth",""))

    # if not verifiable or tests is None:
        # use rm_score
    if tests is None:
        return RewardOutput(reward=config.format_error_reward, is_correct=False, metadata={"error": "No tests found"})

    model_code = extract_code_from_model(model_response)
    if model_code is None:
        return RewardOutput(reward=config.format_error_reward, is_correct=False, metadata={"error": "No code found in model response"})

    is_correct = False

    if data_source in ['taco']:
        tests = taco_to_lcb_format(tests)
        is_correct, test_details = lcb_check_correctness(tests, model_code, debug=False)
    elif data_source in ['lcbv5', 'primeintellect', 'codeforces', 'rstarcoder', 'kodcode']:
        is_correct, test_details = lcb_check_correctness(tests, model_code, debug=False)
    else:
        raise NotImplementedError(f"Dataset {data_source} not implemented")
    
    if use_partial_reward:
        return RewardOutput(reward=test_details['passed_tests']/test_details['total_tests'], is_correct=True, metadata=test_details)


    if is_correct:
        return RewardOutput(reward=config.correct_reward, is_correct=True, metadata=test_details)
    else:
        return RewardOutput(reward=config.incorrect_reward, is_correct=False, metadata=test_details)
    

def taco_to_lcb_format(tests):
    """
    Given a dictionary with keys "inputs" and "outputs", returns a list of test cases.
    Each test case is a dictionary with keys "input" and "output". If the lists are unequal,
    missing entries are filled by reusing the first element of the shorter list.

    Args:
        data (dict): A dictionary with keys "inputs" and "outputs", each mapped to a list of strings.

    Returns:
        list of dict: A list where each element is a dict with keys "input" and "output".
    """
    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])

    # Determine the number of test cases to create.
    n = max(len(inputs), len(outputs))

    test_cases = []
    for i in range(n):
        # Use the first element as a fallback if the list is shorter than n.
        inp = inputs[i] if i < len(inputs) else (inputs[0] if inputs else "")
        out = outputs[i] if i < len(outputs) else (outputs[0] if outputs else "")
        out = out[0] if isinstance(out, list) else out
        test_case: dict[str, Any] = {"input": inp, "output": out, "metadata": {}}
        if "fn_name" in tests:
            test_case["testtype"] = "functional"
            test_case["metadata"]["func_name"] = tests["fn_name"]
        test_cases.append(test_case)

    return test_cases

def postprocess_lcb_sample(sample):
    sample_inputs = [sample["input"] for sample in sample]
    sample_outputs = [sample["output"] for sample in sample]

    sample_dict = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name", None)
        assert fn_name is not None, f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}\nSample: {sample}"
        # Fill in the blank
        sample_dict["fn_name"] = fn_name

    sample = {
        "input_output": json.dumps(sample_dict),
    }
    return sample

def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = lcb_run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)

def lcb_check_correctness(sample, generation, timeout=2, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    assert len(sample) >= 1, "Sample must contain at least one test case"
    sample = postprocess_lcb_sample(sample)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)

    detailed_results = {"all_passed": False, "test_results": [], "total_tests": 0, "passed_tests": 0}

    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result.extend([[-1 for i in range(len(in_outs["inputs"]))]])
        detailed_results["total_tests"] = len(in_outs["inputs"])
        detailed_results["test_results"] = [{"input": inp, "expected": out, "passed": False, "error": "global timeout"} for inp, out in zip(in_outs["inputs"], in_outs["outputs"], strict=False)]
        if debug:
            print("global timeout")
        return False, detailed_results

    if not result:
        return False, detailed_results

    # Create detailed test results
    in_outs = json.loads(sample["input_output"])
    detailed_results["total_tests"] = len(result[0])
    detailed_results["test_results"] = [{"input": inp, "expected": out, "passed": res == True, "error": metadata_list[0].get("error", None), "error_message": metadata_list[0].get("error_message", None), "output": metadata_list[0].get("output", None)} for inp, out, res in zip(in_outs["inputs"], in_outs["outputs"], result[0], strict=False)]
    detailed_results["passed_tests"] = sum(1 for r in result[0] if r == True)
    detailed_results["all_passed"] = all(r == True for r in result[0])

    return all(x == True for x in result[0]), detailed_results

def load_json_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data
