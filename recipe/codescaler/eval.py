import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import asyncio
import multiprocessing as mp
from functools import partial

from tqdm import tqdm

from vllm import LLM, SamplingParams
import datasets
from datasets import load_dataset, Dataset
import uuid
from tenacity import retry, stop_after_attempt, wait_random
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent.futures

import gzip
from pathlib import Path
from typing import List, Union, Dict, Any

from recipe.codescaler.codescaler_utils import *

OUTPUT_ROOT = Path("./eval")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SAMPLING_PARAMS = SamplingParams(
    max_tokens=16384,
    temperature=0.6,
    top_p=0.95,
)

TENSOR_PARALLEL_SIZE = 1

DATASETS = {
    'codecontests': './datasets/Evaluation/CodeContests.parquet',
    'codeforces': './datasets/Evaluation/CodeForces.parquet',
    'livebench': './datasets/Evaluation/LiveBench.parquet',
    'livecodebench': './datasets/Evaluation/LiveCodeBench.parquet',
    'all': './datasets/Evaluation/All.parquet'
}


@retry(stop=stop_after_attempt(3), wait=wait_random(min=1, max=3))
def generate_batch(messages_batch, llm: LLM):
    try:
        outputs = llm.chat(messages_batch, SAMPLING_PARAMS)#, chat_template_kwargs= {"enable_thinking": False})
        results = []
        for out in outputs:
            if not out.outputs:
                results.append("")
            else:
                results.append(out.outputs[0].text)
        return results
    except Exception as e:
        logger.error(f"{e}")
        return [""] * len(messages_batch)


def infer_single_process(model_path, data_batch, device_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=16384,
        gpu_memory_utilization=0.95,
        enable_chunked_prefill=True,
    )
    
    messages_batch = []
    for d in data_batch:
        messages_batch.append(d['prompt'])

    texts = generate_batch(messages_batch, llm)

    all_completions = []
    for d, gen_text in zip(data_batch, texts):
        
        all_completions.append(
            {
                "uuid": d['uuid'],
                "question": d['question'],
                "completion": gen_text,
                "data_source": d['data_source'],
                "testcase": d['reward_model']['ground_truth']
            }
        )

    # Clean up GPU memory
    del llm
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    
    return all_completions


def run_model(model_short: str, model_path: str, n: int, batch_size: int = 32, dataset_name: str = "", device_ids: List[str] = None):
    logger.info("=" * 80)
    logger.info(f"Model Path: {model_path}")

    save_path = f"./eval/{model_short}/{model_short}_{dataset_name}.parquet"

    # Check if output already exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        output_ds = load_dataset("parquet", data_files=save_path)['train']
    else:
        ds = load_dataset("parquet",data_files=DATASETS[dataset_name])['train']
        ds = ds.add_column("uuid", [str(uuid.uuid4()) for _ in range(len(ds))])
        ds = datasets.concatenate_datasets([ds] * n)

        if device_ids is None:
            device_ids = ["0", "1", "2", "3", "4", "5", "6", "7"]  # Use 8 GPUs by default
        
        nums_process = len(device_ids)
        per_process_nums = len(ds) // nums_process

        # Use process pool for data parallel inference
        with ProcessPoolExecutor(max_workers=nums_process) as executor:
            futures = []
            for idx in range(nums_process):
                start = idx * per_process_nums
                if idx == nums_process - 1:  # Last process handles remaining data
                    end = len(ds)
                else:
                    end = start + per_process_nums
                
                # Split data into batches
                data_batch = [ds[i] for i in range(start, end)]
                futures.append(executor.submit(infer_single_process, model_path, data_batch, device_ids[idx]))
            
            # Collect all results
            all_completions = []
            for future in concurrent.futures.as_completed(futures):
                batch_results = future.result()
                all_completions.extend(batch_results)

        # Save completions batch - keep original logic
        output_ds = Dataset.from_list(all_completions)
        # output_ds.to_parquet(save_path)

    return output_ds

def process_single_item(args):
    """Process a single test item function for process pool"""
    i, item = args
    try:
        testcase = json.loads(item['testcase'])
        code = extract_code_from_model_test(item['completion'])
        result = lcb_check_correctness(testcase, code, 2, False)
        return i, result[0]
    except Exception as e:
        logger.error(f'Error processing index {i}: {e}')
        return i, False

def run_test(ds, model_short, max_workers=None, dataset_name: str = ""):
    """
    Optimized test function using process pool for concurrent processing
    """
    save_path = f"./eval/{model_short}/{model_short}_{dataset_name}_verified.parquet"

    if os.path.exists(save_path):
        output_ds = load_dataset("parquet", data_files=save_path)['train']
        cal_metrics(output_ds)
        return output_ds

    # Automatically determine optimal number of worker processes
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 16)  # Limit max processes to avoid resource overload
    
    logger.info(f"Using {max_workers} processes for concurrent testing")
    
    # Prepare data
    items_with_index = [(i, item) for i, item in enumerate(ds)]
    results = [None] * len(ds)

    # Use process pool for concurrent processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_item, item_data): item_data[0] 
            for item_data in items_with_index
        }
        
        # Collect results
        for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                          total=len(future_to_index), desc="Testing completions"):
            original_index = future_to_index[future]
            try:
                index, result = future.result()
                results[index] = result
            except Exception as e:
                logger.error(f'Error processing index {original_index}: {e}')
                results[original_index] = False

    # Assemble results
    verified_outputs = []
    for i, item in enumerate(ds):
        item_copy = dict(item)
        item_copy['verified_score'] = float(results[i])
        item_copy.pop('testcase')
        verified_outputs.append(item_copy)
        
    
    output_ds = Dataset.from_list(verified_outputs)
    cal_metrics(output_ds)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_ds.to_parquet(save_path)
    return output_ds
    
def cal_metrics(ds):
    datasource2acc = defaultdict(list)
    for item in ds:
        datasource2acc[item['data_source']].append(item['verified_score'])
    
    for datasource, acc_list in datasource2acc.items():
        print(f"{datasource}: {sum(acc_list) / len(acc_list)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--model_short", type=str, required=True, help="Model short name")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--only_run_model", action="store_true", help="Only run model without testing")
    parser.add_argument("--n", type=int, default=8, help="Number of repetitions")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--test_workers", type=int, default=None, help="Number of concurrent test processes")
    parser.add_argument("--dataset", type=str, choices=['codecontests', 'codeforces', 'livebench', 'livecodebench', 'all'])
    parser.add_argument("--device_ids", type=str, default=None, help="Comma-separated GPU device IDs (e.g., '0,1,2,3')")
    args = parser.parse_args()
    

    save_path = f"./eval/{args.model_short}/{args.model_short}_{args.dataset}_verified.parquet"

    if os.path.exists(save_path):
        output_ds = load_dataset("parquet", data_files=save_path)['train']
        cal_metrics(output_ds)
    else:

        logger.info(f"Output root directory: {OUTPUT_ROOT}")
        logger.info(f"Using tensor_parallel_size={TENSOR_PARALLEL_SIZE}")
        logger.info(f"Inference batch size: {args.batch_size}")
        if args.test_workers:
            logger.info(f"Concurrent test processes: {args.test_workers}")

        device_ids = None
        if args.device_ids:
            device_ids = args.device_ids.split(',')
            logger.info(f"Using GPU devices: {device_ids}")

        output_ds = run_model(args.model_short, args.model_path, args.n, args.batch_size, args.dataset, device_ids)

        if not args.only_run_model:
            output_ds = run_test(output_ds, args.model_short, args.test_workers, args.dataset)
            