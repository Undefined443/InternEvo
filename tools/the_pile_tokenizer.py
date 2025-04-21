import shutup

shutup.please()
import argparse
import orjson
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import zstandard as zstd
from typing import Generator
import pandas as pd


def generate_lines(file_path: str) -> Generator[dict, None, None]:
    _file = Path(file_path)
    if _file.suffix == ".zst":
        with zstd.open(file_path, "rt") as f:
            total = sum(1 for _ in f)

        with zstd.open(file_path, "rt") as f:
            for line in tqdm(f, total=total, desc=f"Processing {file_path}"):
                line = orjson.loads(line)
                yield line

    elif _file.suffix == ".parquet":
        df = pd.read_parquet(file_path)
        total = len(df)
        for _, line in tqdm(df.iterrows(), total=total, desc=f"Processing {file_path}"):
            text = line["text"]
            meta = {"pile_set_name": "minipile"}
            line = {"text": text, "meta": meta}
            yield line

    else:
        raise ValueError(f"Unsupported file extension: {_file} ({_file.suffix})")


def generate_samples(lines: Generator[dict, None, None]) -> Generator[dict, None, None]:
    for line in lines:
        text = line["text"]
        pile_set_name = line["meta"]["pile_set_name"]
        line = {"text": text, "pile_set_name": pile_set_name}
        yield line


def tokenize_samples(inputs: Generator[dict, None, None], model: str) -> Generator[dict, None, None]:
    model = AutoTokenizer.from_pretrained(model)
    for input in inputs:
        _text = input["text"]
        _output = model(_text)
        tokens = _output["input_ids"]
        pile_set_name = input["pile_set_name"]
        output = {"tokens": tokens, "pile_set_name": pile_set_name}
        yield output


def dump_outputs(outputs: Generator[list, None, None], output_path: str):
    _output_path = Path(output_path)
    bin_file = str(_output_path.with_suffix(".bin"))
    meta_file = str(_output_path.with_suffix(".bin.meta"))
    _output_dir = _output_path.parent
    _output_dir.mkdir(exist_ok=True, parents=True)

    sample_num = 0
    token_num = 0
    last_position = 0
    meta = []
    with open(bin_file, "wb") as f:
        for output in outputs:
            tokens = output["tokens"]
            pile_set_name = output["pile_set_name"]
            sample_num += 1
            token_num += len(tokens)
            meta.append((last_position, len(tokens)))
            line = orjson.dumps({"tokens": tokens, "pile_set_name": pile_set_name}) + b"\n"
            last_position += len(line)
            f.write(line)

    with open(meta_file, "wb") as f:
        np.save(f, meta)

    print(f"Wrote {sample_num} samples, {token_num} tokens to {bin_file}")


def process_file(input_file: str, model: str, output_dir: str):
    _input_file = Path(input_file)
    _output_dir = Path(output_dir)
    _output_dir.mkdir(exist_ok=True, parents=True)
    output_path = str(_output_dir / _input_file.stem)
    lines = generate_lines(input_file)
    samples = generate_samples(lines)
    outputs = tokenize_samples(samples, model)
    dump_outputs(outputs, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--model", type=str, default="microsoft/mpnet-base", help="Hugging Face model name")
    args = parser.parse_args()

    process_file(args.input_file, args.model, args.output_dir)
