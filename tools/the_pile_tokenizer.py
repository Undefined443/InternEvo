import warnings; warnings.filterwarnings("ignore", category=FutureWarning)  # noqa
import os
import argparse
import json
import os.path as osp
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import zstandard as zstd
import concurrent.futures as futures


def get_files(dataset_path):
    """Get files from input path

    Args:
        dataset_path (str): Path of dataset parquet file.
        model (str): Path of tokenizer.

    Yields:
        tuple: dumped processed data sample and length of tokens.
    """

    dataset_path = Path(dataset_path)
    files = {}
    if dataset_path.is_dir():
        files[".parquet"] = sorted(list(dataset_path.glob("*.parquet")))
        files[".jsonl.zst"] = sorted(list(dataset_path.glob("*.jsonl.zst")))
        files[".jsonl"] = sorted(list(dataset_path.glob("*.jsonl")))
    else:
        raise ValueError(f"Invalid dataset path: {dataset_path}")

    return files


def tokenize(sample, model, pile_set_name=None):
    """Tokenize input dataset

    Args:
        sample (dict): Input data sample.
        model (str): Path of tokenizer.

    Returns:
        tuple: dumped processed data sample and length of tokens.
    """
    token_ids = model.encode(sample)
    obj = {"tokens": token_ids}
    if pile_set_name:
        obj["pile_set_name"] = pile_set_name
    line = str.encode(json.dumps(obj) + "\n")
    return line, len(token_ids)


def dump_bin_meta_bin(dataset, path):
    """Dump processed dataset

    Args:
        samples (dict): Input data sample.
        path (str): Path for output dataset.
    """
    dir_path = Path(path)
    bin_path = Path(path).with_suffix(".bin")
    meta_path = Path(path).with_suffix(".bin.meta")
    dir_path.mkdir(exist_ok=True, parents=True)
    bin_file = open(bin_path, "wb")

    tokens = 0
    last_position = 0
    samples = 0
    meta = []

    for line, token_num in dataset:
        tokens += token_num
        bin_file.write(line)
        meta.append((last_position, token_num))
        last_position += len(line)
        samples += 1

    bin_file.close()
    np.save(open(meta_path, "wb"), meta)
    print(f"Wrote {samples} samples and {tokens} tokens into {bin_path}")


def generate_from_parquet(pf, model, position):
    df = pd.read_parquet(pf)
    total_lines = len(df)

    for row in tqdm(df.itertuples(), total=total_lines, desc=f"Processing {pf.name}", position=position, leave=False):
        yield tokenize(row.text, model)


def generate_from_jsonl_zst(jsonl_zst_file, model, position):
    with zstd.open(jsonl_zst_file, "rt") as f:
        for line in tqdm(f, desc=f"Processing {jsonl_zst_file.name}", position=position, leave=False):
            obj = json.loads(line)
            yield tokenize(obj["text"], model, obj["meta"]["pile_set_name"])


def generate_from_jsonl(jsonl_file, model, position):
    with open(jsonl_file, "rt") as f:
        for line in tqdm(f, desc=f"Processing {jsonl_file.name}", position=position, leave=False):
            obj = json.loads(line)
            yield tokenize(obj["text"], model, obj["meta"]["pile_set_name"])


def process_files(files, model, output_path):
    generate_funcs = {
        '.parquet': generate_from_parquet,
        '.jsonl.zst': generate_from_jsonl_zst,
        '.jsonl': generate_from_jsonl
    }

    cpu_count = os.cpu_count()
    max_workers = min(cpu_count * 2, 32)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_type, file_list in files.items():
            for position, file in enumerate(file_list):
                dataset = generate_funcs[file_type](file, model, position)
                output_file = output_path / file.stem
                executor.submit(dump_bin_meta_bin, dataset, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="path of dataset json file")
    parser.add_argument("output_path", type=str, help="path of processed dataset")
    parser.add_argument("--model", type=str, default="microsoft/mpnet-base", help="Hugging Face model name")

    args = parser.parse_args()
    model = AutoTokenizer.from_pretrained(args.model)

    files = get_files(args.dataset_path)
    process_files(files, model, args.output_path)
