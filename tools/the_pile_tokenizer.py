import argparse
import orjson
from pathlib import Path
import math

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import zstandard as zstd
import concurrent.futures
import psutil
from typing import Generator

CHUNK_SIZE = 1000


def get_files(dataset_path: str) -> dict:
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


def calculate_file_lines(file_path: Path) -> int:
    with zstd.open(file_path, "rt") as f:
        num_of_lines = sum(1 for _ in f)
    return num_of_lines


def tokenize_chunk(idx: int, chunk: pd.DataFrame, model: AutoTokenizer) -> tuple:
    texts = chunk["text"].tolist()
    outputs = model(texts)
    chunk["input_ids"] = outputs["input_ids"]
    chunk["line"] = chunk.apply(lambda x: orjson.dumps({"tokens": x["input_ids"], "pile_set_name": x["pile_set_name"]}) + b"\n", axis=1)  # noqa
    chunk["token_num"] = chunk.apply(lambda x: len(x["input_ids"]), axis=1)
    return idx, chunk


def tokenize_dataset(dataset: Generator[pd.DataFrame, None, None], model: AutoTokenizer, total: int) -> list:
    """Tokenize input dataset

    Args:
        sample (dict): Input data sample.
        model (str): Path of tokenizer.

    Returns:
        tuple: dumped processed data sample and length of tokens.
    """
    tokenset = [None] * total
    max_workers = get_optimal_workers()

    batch = []
    num_iter = 0
    i = 0

    with tqdm(total=total, desc="Tokenizing", position=1, leave=True) as pbar:
        for chunk in dataset:
            batch.append(chunk)
            i += 1
            if i % max_workers == 0:
                futures = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    for j, chunk in enumerate(batch):
                        idx = num_iter * max_workers + j
                        futures.append(executor.submit(tokenize_chunk, idx, chunk, model))
                    for future in concurrent.futures.as_completed(futures):
                        idx, chunk = future.result()
                        tokenset[idx] = chunk
                        pbar.update(1)
                num_iter += 1
                batch = []

        if batch:
            futures = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for j, chunk in enumerate(batch):
                    idx = num_iter * max_workers + j
                    futures.append(executor.submit(tokenize_chunk, idx, chunk, model))
                for future in concurrent.futures.as_completed(futures):
                    idx, chunk = future.result()
                    tokenset[idx] = chunk
                    pbar.update(1)

    return tokenset


def dump_bin_meta_bin(tokenset: Generator[list, None, None], path: Path, total: int):
    """Dump processed tokenset

    Args:
        samples (dict): Input data sample.
        path (str): Path for output tokenset.
    """
    dir_path = path.parent
    bin_path = path.with_suffix(".bin")
    meta_path = path.with_suffix(".bin.meta")
    dir_path.mkdir(exist_ok=True, parents=True)
    bin_file = open(bin_path, "wb")

    tokens = 0
    last_position = 0
    samples = 0
    meta = []

    for idx, chunk in tqdm(enumerate(tokenset), total=len(tokenset), desc=f"Dumping to {bin_path}", position=2, leave=True):  # chunk: [(line, token_num), ...]
        if chunk is None:
            print(f"Empty chunk at {idx}, skipping...")
            continue
        for line, token_num in zip(chunk["line"], chunk["token_num"]):
            tokens += token_num
            meta.append((last_position, token_num))
            last_position += len(line)
            samples += 1

        output_chunk = b"".join(line for line in chunk["line"])
        bin_file.write(output_chunk)

    bin_file.close()
    with open(meta_path, "wb") as f:
        np.save(f, meta)
    print(f"Wrote {samples} samples and {tokens} tokens into {bin_path}")


def generate_from_parquet(pf: Path, model: AutoTokenizer, position: int):
    df = pd.read_parquet(pf)
    total_lines = len(df)

    for row in tqdm(df.itertuples(), total=total_lines, desc=f"Processing {pf.name}", position=position, leave=True):
        yield tokenize_dataset(row.text, model)


def generate_from_jsonl_zst(jsonl_zst_file: Path, total: int) -> Generator[list, None, None]:  # noqa
    with zstd.open(jsonl_zst_file, "rt") as f:
        chunk = pd.DataFrame(columns=["text", "pile_set_name"])
        for line in tqdm(f, desc=f"Extracting {jsonl_zst_file.name}", total=total, position=0, leave=True):
            obj = orjson.loads(line)
            text = obj["text"]
            pile_set_name = obj["meta"]["pile_set_name"]
            record = pd.DataFrame({"text": [text], "pile_set_name": [pile_set_name]})
            chunk = pd.concat([chunk, record], ignore_index=True)

            if len(chunk) >= CHUNK_SIZE:
                yield chunk
                chunk = pd.DataFrame(columns=["text", "pile_set_name"])
        if not chunk.empty:
            yield chunk


def generate_from_jsonl(jsonl_file: Path, model: AutoTokenizer, position: int):
    with open(jsonl_file, "rt") as f:
        for line in tqdm(f, desc=f"Processing {jsonl_file.name}", position=position, leave=True):
            obj = orjson.loads(line)
            yield tokenize_dataset(obj["text"], model, obj["meta"]["pile_set_name"])


def get_optimal_workers() -> int:
    cpu_count = psutil.cpu_count(logical=False)
    memory = psutil.virtual_memory().available / 2**30
    memory_based_workers = int(memory)
    return min(cpu_count, memory_based_workers)


def process_files(files: dict, model: str, output_path: str):
    generate_funcs = {".parquet": generate_from_parquet, ".jsonl.zst": generate_from_jsonl_zst, ".jsonl": generate_from_jsonl}
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    for file_type, file_list in files.items():
        for file_path in file_list:
            total = calculate_file_lines(file_path)
            dataset = generate_funcs[file_type](file_path, total)  # 生成 chunk set
            chunk_num = math.ceil(total / CHUNK_SIZE)
            tokenset = tokenize_dataset(dataset, model, chunk_num)  # 从 chunk set 生成 token set
            output_file = output_path / file_path.stem
            dump_bin_meta_bin(tokenset, output_file, total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="path of dataset json file")
    parser.add_argument("output_path", type=str, help="path of processed dataset")
    parser.add_argument("--model", type=str, default="microsoft/mpnet-base", help="Hugging Face model name")

    args = parser.parse_args()

    files = get_files(args.dataset_path)
    model = AutoTokenizer.from_pretrained(args.model)
    process_files(files, model, args.output_path)
