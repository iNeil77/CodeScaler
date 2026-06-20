import argparse
from huggingface_hub import hf_hub_download
import shutil

datasets = ["LiveBench","CodeContests","CodeForces"]

# Pinned HF dataset revisions (resolved 2026-06-20) for reproducibility.
HF_REVISIONS = {
    "Gen-Verse/LiveBench": "fa070cf11dccf8bab0bdea01901649bc17222aeb",
    "Gen-Verse/CodeContests": "e06e6e140899dddc8ef841255db0b050b6da27d6",
    "Gen-Verse/CodeForces": "b3b3df092edb8f5412a6c4f83ec8753bcd9de943",
}

split = "test"

for dataset in datasets:
    repo_id = f"Gen-Verse/{dataset}"
    cached_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"{split}/{dataset}.json",
        revision=HF_REVISIONS[repo_id],
    )
    shutil.copy(cached_path, f"./data/{dataset}.json")
