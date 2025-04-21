import os
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import concurrent.futures
import psutil
from typing import Generator

CHUNK_SIZE = 100


def get_optimal_workers() -> int:
    cpu_count = os.cpu_count()
    memory = psutil.virtual_memory()
    memory_based_workers = int(memory.available / (2**30))
    return min(cpu_count * 2, memory_based_workers, 32)


def generate_chunks(file: Path, total: int) -> Generator[list, None, None]:
    with open(file, "rt") as f:
        chunk = ""
        for line in tqdm(f, desc=f"Generating chunks for {file.name}", total=total):
            if line == "<|endoftext|>\n":
                yield chunk
                chunk = ""
            else:
                chunk += line


def tokenize_chunk(idx, chunk: str, model: AutoTokenizer) -> list:
    output = model(chunk)
    input_ids = output["input_ids"]
    token_num = len(input_ids)
    line = str.encode(json.dumps({"tokens": input_ids}) + "\n")
    return idx, (line, token_num)


def process_file(file: Path, model: AutoTokenizer) -> list:
    max_workers = get_optimal_workers()
    if not file.exists():
        raise FileNotFoundError(f"File {file} does not exist")
    with open(file, "rt") as f:
        total = sum(1 for _ in f)
    chunks = generate_chunks(file, total)
    result = [None] * total
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, chunk in enumerate(chunks):
            futures.append(executor.submit(tokenize_chunk, idx, chunk, model))

    for future in futures:
        idx, (line, token_num) = future.result()
        result[idx] = (line, token_num)

    return result


def dump_bin_meta_bin(dataset: list, output_path: Path):
    dir_path = output_path.parent
    bin_path = output_path.with_suffix(".bin")
    meta_path = output_path.with_suffix(".bin.meta")
    dir_path.mkdir(exist_ok=True, parents=True)

    tokens = 0
    last_position = 0
    samples = 0
    bin = b""
    meta = []

    for line, token_num in tqdm(dataset, desc="Writing dataset", total=len(dataset)):
        tokens += token_num
        meta.append((last_position, token_num))
        last_position += len(line)
        samples += 1
        bin += line

    with open(bin_path, "wb") as f:
        f.write(bin)

    with open(meta_path, "wb") as f:
        np.save(f, meta)

    print(f"Wrote {samples} samples and {tokens} tokens into {bin_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="path of dataset txt file")
    parser.add_argument("output_path", type=str, help="path of processed dataset")
    parser.add_argument("--model", type=str, default="microsoft/mpnet-base", help="Hugging Face model name")

    args = parser.parse_args()
    model = AutoTokenizer.from_pretrained(args.model)

    file = Path(args.dataset_path)
    output_path = Path(args.output_path)
    dataset = process_file(file, model)
    dump_bin_meta_bin(dataset, output_path)
