# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import subprocess
import tempfile
import difflib
import asyncio
import regex as re
import hashlib
import random
import os
import json
import ast
import numpy as np
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import defaultdict
import time
import torch
import textwrap
from tqdm import tqdm
from verl import DataProto
import ray

import asyncio
from verl.utils.reward_score.prime_code import compute_score as prime_code_compute_score
from verl.workers.reward_manager.prime import parallel_compute_score_async
from verl.workers.reward_manager import register
from verl.utils.reward_score import default_compute_score
from recipe.codescaler import get_reward_manager_cls
from recipe.codescaler.codescaler_utils import check_correctness, extract_code_from_model, extract_code_from_model_test, EMTPY_STRING
from recipe.codescaler.codescaler_reward_types import *



def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        split: 'train' or 'val'
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_model_config = config.reward_model
    reward_manager_name = reward_model_config.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    # Try to get a custom reward function based on the configuration
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score
    record_dir = reward_model_config.get("record_dir", None) 
    use_partial_reward = reward_model_config.get("use_partial_reward", False)
    reward_shaping = reward_model_config.get("reward_shaping", True)

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(default_compute_score, sandbox_fusion_url=sandbox_url, concurrent_semaphore=_concurrent_semaphore)
        else:
            final_compute_score = default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    reward_manager = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        record_dir=record_dir,
        run_id=config.trainer.experiment_name,
        use_partial_reward=use_partial_reward,
        reward_shaping=reward_shaping,
        **reward_kwargs,
    )

    return reward_manager


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result["reward_extra_info"]
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)

def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()

def check_syntax(code_string):
    try:
        # Attempt to parse the code string
        ast.parse(code_string)
        return True
    # except SyntaxError as e:
    except Exception as e:
        # If a SyntaxError is raised, the code is not valid
        # print(f"Syntax error in code: {e}")
        return False
    
def parse_code(action: str, mode="all"):
    """
    Parse the raw action string (which is the llm response) into an actual action and its contents.
    Ensures that the parsed code is valid and safe for execution.
    
    Args:
        action: Raw action string containing Python code
        
    Returns:
        Tuple containing the extracted code and a validity flag
    """
    # Try to find Python code in various formats
    all_valid_python_code = re.findall(r"<python>(.*?)</python>", action, re.DOTALL)
    
    if not all_valid_python_code:
        all_valid_python_code = re.findall(r"```\n?python(.*?)```", action, re.DOTALL)
    
    if len(all_valid_python_code) == 0:
        return ""
    
    if mode == "all":
        parsed_code = "\n".join([code for code in all_valid_python_code if check_syntax(code)])
    elif mode == "first":
        # Use the first code block found
        parsed_code = all_valid_python_code[0]
    elif mode == "last":
        # Use the last code block found
        parsed_code = all_valid_python_code[-1]
    elif mode == "all_in_last_turn":
        # parse all the code blocks only in the last assistant turn
        # find the last assistant turn
        last_turn_start_idx = action.rfind('<|im_start|>assistant')
        if last_turn_start_idx == -1:
            last_turn = action
        else:
            last_turn = action[last_turn_start_idx:]
        all_valid_python_code = re.findall(r"<python>(.*?)</python>", last_turn, re.DOTALL)
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```\n?python(.*?)```", last_turn, re.DOTALL)
        if len(all_valid_python_code) == 0:
            return ""
        parsed_code = "\n".join([code for code in all_valid_python_code if check_syntax(code)])
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'all', 'first', 'last', or 'all_in_last_turn'.")
    
    parsed_code = parsed_code.strip(' \n')
    return parsed_code

def transform_score(score: float, is_empty: bool, reward_shaping=True):
    if not reward_shaping:
        return score
    
    if is_empty:
        return 0.0
    else:
        # return log(1+e^x)
        return np.log(1 + np.exp(score))

@register("codescaler")
class CodeScalerRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None, run_id=None, reward_fn_key='data_source', record_dir=None, use_partial_reward=False, reward_shaping=True) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.step_idx = None
        self.n_workers = 64
        self.binary = False
        self.parse_code_mode = "all_in_last_turn" # "all", "first", "last"
        self.config = RewardConfig()
        self.run_id = run_id
        self.use_partial_reward = use_partial_reward
        self.reward_shaping = reward_shaping
        if record_dir is not None:
            self.record_dir = Path(record_dir) / "verl_step_records" / self.run_id
 
    def get_verified_score(self, data: DataProto, responses_str, split='train'):
        scores = [{} for _ in range(len(data))]

        if split == 'test':
            use_partial_reward = False
        else:
            use_partial_reward = self.use_partial_reward

        # run temp bash
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_index = {
                executor.submit(check_correctness, entry, model_response, self.config, use_partial_reward): i 
                for i, (entry, model_response) in enumerate(zip(data, responses_str))
            }
            results = [None] * len(data)

            for future in tqdm(concurrent.futures.as_completed(future_to_index)):
                index = future_to_index[future]
                try:
                    updated_entry = future.result()
                    results[index] = updated_entry
                except Exception as e:
                    print(f'Error processing index {index}: {e}')
                    results[index] = RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

        for i in range(len(scores)):
            scores[i]['score'] = results[i].reward

        return scores
       
        
    def __call__(self, data: DataProto, return_dict=False, split='train'):
        """We will expand this function gradually based on the available datasets"""
        save_record = data.meta_info.get('save_record', True)

        if not hasattr(self, 'record_dir'):
            if hasattr(self, 'run_id'):
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / self.run_id
                self.record_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / f"acecoder-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
                self.record_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.record_dir.mkdir(parents=True, exist_ok=True)

        
        # check the last step index
        if self.step_idx is None:
            last_step_idx = 0
            for file in os.listdir(self.record_dir):
                if self.num_examine == 1:
                    if re.search(r"step-val-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
                else:
                    if re.search(r"step-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
            self.step_idx = last_step_idx + 1
        if data.meta_info.get('global_step', None) is not None:
            self.step_idx = data.meta_info['global_step']
                


        # TODO: implement new reward computing & statistic mechanism
        verified_scores = [{} for _ in range(len(data))]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
            
        already_print_data_sources = {}
        
        # retrieve the list of prompt_token_ids and their length
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        # retrieve the list of response ids and their valid length
        response_ids = data.batch['responses']
        valid_prompt_length = data.batch['attention_mask'][:, :prompt_length].sum(dim=-1)
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
            
        # batch decode the list of responses and prompts
        responses_str = [self.tokenizer.decode(response_ids[i][:valid_response_length[i].item()], skip_special_tokens=False) for i in range(len(data))]
        prompts_str = [self.tokenizer.decode(prompt_ids[i][-valid_prompt_length[i].item():], skip_special_tokens=False) for i in range(len(data))]
        
        
    

        if split == 'train' and 'rm_scores' in data.batch.keys():
            extracted_codes = [extract_code_from_model(response) for response in responses_str]
            for i in range(len(data)):
                reward_tensor[i, valid_response_length[i].item() - 1] = transform_score(sum(data[i].batch['rm_scores']).item(), extracted_codes[i]==EMTPY_STRING, self.reward_shaping)
            return reward_tensor

        # extract the answer for the list of responses
        extracted_codes = [extract_code_from_model_test(response) for response in responses_str]
        
        verified_scores = self.get_verified_score(data, responses_str, split)

        for i, score in enumerate(verified_scores):
            if isinstance(score, dict):
                reward_tensor[i, valid_response_length[i].item() - 1] = score['score']
                for k, v in score.items():
                    reward_extra_info[k].append(v)
            else:
                reward_tensor[i, valid_response_length[i].item() - 1] = score
        
        if save_record:
            # Save the records for each code response sample, which will be reported to wandb
            # if is list
            if isinstance(data[i].non_tensor_batch['raw_prompt'], list):
                problem = data[i].non_tensor_batch['raw_prompt'][0]['content']
            elif isinstance(data[i].non_tensor_batch['raw_prompt'], str):
                problem = data[i].non_tensor_batch['raw_prompt']

            to_save_records = [
                {
                    "data_source": data[i].non_tensor_batch.get("data_source", ""),
                    "problem": problem,
                    "prompt": prompts_str[i],
                    "response": responses_str[i],
                    "extracted_code": extracted_codes[i],
                    "verified_score": verified_scores[i]
                }
                for i in range(len(data))
            ]

            # Save the records to a file
            if self.num_examine == 1:
                temp_file = self.record_dir / f"step-val-{self.step_idx}.json"
            else:
                temp_file = self.record_dir / f"step-{self.step_idx}.json"
            self.step_idx += 1

            with open(temp_file, "w") as f:
                json.dump(to_save_records, f, indent=4)
            print(f"Step {self.step_idx}: saved {len(to_save_records)} records to {temp_file}")
 

        if return_dict: 
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor