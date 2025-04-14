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


def process(dataset_path, model):
    """Process data sample from input dataset

    Args:
        dataset_path (str): Path of dataset parquet file.
        model (str): Path of tokenizer.

    Yields:
        tuple: dumped processed data sample and length of tokens.
    """

    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        parquet_files = list(dataset_path.glob("*.parquet"))
        jsonl_zst_files = list(dataset_path.glob("*.zst"))
    else:
        raise ValueError(f"Invalid dataset path: {dataset_path}")

    return process_files(parquet_files, jsonl_zst_files, model)


def tokenize(sample, pile_set_name, model):
    """Tokenize input dataset

    Args:
        sample (dict): Input data sample.
        model (str): Path of tokenizer.

    Returns:
        tuple: dumped processed data sample and length of tokens.
    """
    token_ids = model.encode(sample)
    if len(token_ids) > model.model_max_length:
        token_ids = token_ids[: model.model_max_length]
    line = str.encode(json.dumps({"tokens": token_ids, "pile_set_name": pile_set_name}) + "\n")
    return line, len(token_ids)


def dump_bin_meta_bin(samples, path, split_ratio=0.1):
    """Dump processed dataset

    Args:
        samples (dict): Input data sample.
        path (str): Path for output dataset.
        split_ratio (float): Ratio for validation dataset splitting.
            Default to: 0.1.

    Returns:
        tuple: number of train/valid tokens of processed dataset,
            number of train/valid samples of processed dataset.
    """

    train_path = osp.join(path, "train/en/")
    valid_path = osp.join(path, "valid/en/")
    train_dir = Path(train_path)
    valid_dir = Path(valid_path)
    train_dir.mkdir(exist_ok=True, parents=True)
    valid_dir.mkdir(exist_ok=True, parents=True)
    train_f = open(train_dir.joinpath("dataset.bin"), "wb")
    valid_f = open(valid_dir.joinpath("dataset.bin"), "wb")

    train_tokens = 0
    valid_tokens = 0
    last_train_position = 0
    last_valid_position = 0
    train_samples = 0
    valid_samples = 0
    train_meta = []
    valid_meta = []

    sample_length = len(samples)
    np.random.seed(0)
    valid_indices = np.random.choice(range(sample_length), int(sample_length * split_ratio)).tolist()

    count = -1
    for line, token_num in samples:
        count += 1
        if count in valid_indices:
            valid_tokens += token_num
            valid_f.write(line)
            valid_meta.append((last_valid_position, token_num))
            last_valid_position += len(line)
            valid_samples += 1
        else:
            train_tokens += token_num
            train_f.write(line)
            train_meta.append((last_train_position, token_num))
            last_train_position += len(line)
            train_samples += 1

    train_f.close()
    valid_f.close()
    np.save(open(train_dir.joinpath("dataset.bin.meta"), "wb"), train_meta)
    np.save(open(valid_dir.joinpath("dataset.bin.meta"), "wb"), valid_meta)

    return train_tokens, valid_tokens, train_samples, valid_samples


def process_parquet_file(pf, model, pbar=None):
    df = pd.read_parquet(pf)
    results = []
    total_lines = len(df)

    for row in tqdm(df.itertuples(), total=total_lines, desc=f"Processing {os.path.basename(pf)}", leave=False):
        results.append(tokenize(row.text, model))

    if pbar:
        pbar.update(1)
    return results


def process_jsonl_zst_file(jsonl_zst_file, model, pbar=None):
    results = []

    with zstd.open(jsonl_zst_file, "rt") as f:
        for line in tqdm(f, desc=f"Processing {os.path.basename(jsonl_zst_file)}", leave=False):
            obj = json.loads(line)
            text = obj["text"]
            meta = obj["meta"]
            pile_set_name = meta["pile_set_name"]
            results.append(tokenize(text, pile_set_name, model))

    if pbar:
        pbar.update(1)
    return results


def process_files(parquet_files, jsonl_zst_files, model):
    cpu_count = os.cpu_count()
    max_workers = min(cpu_count * 2, 32)
    all_results = []

    total_files = len(parquet_files) + len(jsonl_zst_files)

    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        with tqdm(total=total_files, desc="Overall progress") as main_pbar:

            parquet_futures = [executor.submit(process_parquet_file, pf, model, main_pbar) for pf in parquet_files]
            zst_futures = [
                executor.submit(process_jsonl_zst_file, jsonl_zst_file, model, main_pbar)
                for jsonl_zst_file in jsonl_zst_files
            ]

            for future in futures.as_completed(parquet_futures + zst_futures):
                results = future.result()
                all_results.extend(results)

    return all_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="path of dataset json file")
    parser.add_argument("output_path", type=str, help="path of processed dataset")
    parser.add_argument("--split_ratio", type=float, default=0.1, help="ratio for validation dataset splitting")
    parser.add_argument("--model", type=str, default="microsoft/mpnet-base", help="Hugging Face model name")

    args = parser.parse_args()
    model = AutoTokenizer.from_pretrained(args.model)
    split_ratio = args.split_ratio
    samples = []

    dataset = process(args.dataset_path, model)

    train_tokens, valid_tokens, train_samples, valid_samples = dump_bin_meta_bin(
        dataset, args.output_path, args.split_ratio
    )
    print(f"number of train dataset: {train_samples}, number of train dataset token: {train_tokens}")
    print(f"number of validation dataset: {valid_samples}, number of validation dataset token: {valid_tokens}")
