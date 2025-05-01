import torch as th
from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer


def main(model_path, prompt):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=th.float16, attn_implementation="flash_attention_2").cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, repetition_penalty=2.0)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--prompt", type=str, default="Tell me a joke.")
    args = parser.parse_args()
    main(args.model_path, args.prompt)
