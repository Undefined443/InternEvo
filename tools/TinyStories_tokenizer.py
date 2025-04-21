import shutup; shutup.please()
import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Generator
import orjson


def generate_samples(file: str) -> Generator[str, None, None]:
    with open(file, "rt") as f:
        total = sum(1 for _ in f)

    with open(file, "rt") as f:
        sample = ""
        for line in tqdm(f, desc=f"Generating samples for {file}", total=total):
            if line == "<|endoftext|>\n":
                yield sample
                sample = ""
            else:
                sample += line


def tokenize_samples(samples: Generator[str, None, None], model: str) -> Generator[list, None, None]:
    model = AutoTokenizer.from_pretrained(model)
    for input in samples:
        _output = model(input)
        tokens = _output["input_ids"]
        yield tokens


def dump_tokens(tokenset: Generator[list, None, None], output_path: str):
    _output_path = Path(output_path)
    bin_file = str(_output_path.with_suffix(".bin"))
    meta_file = str(_output_path.with_suffix(".bin.meta"))
    _output_dir = _output_path.parent
    _output_dir.mkdir(exist_ok=True, parents=True)

    token_num = 0
    last_position = 0
    sample_num = 0
    meta = []
    with open(bin_file, "wb") as f:
        for tokens in tokenset:
            sample_num += 1
            token_num += len(tokens)
            meta.append((last_position, len(tokens)))
            line = orjson.dumps({"tokens": tokens}) + b"\n"
            last_position += len(line)
            f.write(line)

    with open(meta_file, "wb") as f:
        np.save(f, meta)

    print(f"Wrote {sample_num} samples, {token_num} tokens to {bin_file}")


def process_file(input_file: str, model: str, output_dir: str):
    samples = generate_samples(input_file)
    tokens = tokenize_samples(samples, model)
    output_path = str(Path(output_dir) / Path(input_file).stem)
    dump_tokens(tokens, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--model", type=str, default="microsoft/mpnet-base", help="Hugging Face model name")
    args = parser.parse_args()

    process_file(args.input_path, args.model, args.output_path)
