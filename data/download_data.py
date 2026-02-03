import argparse
from huggingface_hub import hf_hub_download
import shutil

datasets = ["LiveBench","CodeContests","CodeForces","MBPP"]

split = "test"

for dataset in datasets:
    cached_path = hf_hub_download(
        repo_id=f"Gen-Verse/{dataset}",
        repo_type="dataset",
        filename=f"{split}/{dataset}.json"
    )
    shutil.copy(cached_path, f"./data/{dataset}.json")