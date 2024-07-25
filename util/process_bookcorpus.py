# author: ddukic

import json
import multiprocessing
from functools import partial

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

block_size = 512


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in tqdm(examples.keys())}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in tqdm(range(0, total_length, block_size))]
        for k, t in concatenated_examples.items()
    }
    return result


def process_range(index, ranges, texts, tokenizer):
    if index != len(ranges) - 1:
        start, end = ranges[index], ranges[index + 1]
        sentences_tokenized_grouped = group_texts(tokenizer(texts[start:end]))
        with open(
            f"/home/ddukic/generative-ie/data/processed/bookcorpus/train_{index}.json",
            "w+",
        ) as f:
            json.dump(sentences_tokenized_grouped, f)


if __name__ == "__main__":
    dataset = load_dataset("bookcorpus")

    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base", add_prefix_space=True, do_lower_case=False
    )

    texts = dataset["train"]["text"]

    ranges = list(range(0, len(dataset["train"]), len(dataset["train"]) // 1000))

    with multiprocessing.Pool(processes=4) as pool:
        func = partial(process_range, ranges=ranges, texts=texts, tokenizer=tokenizer)
        pool.map(func, range(len(ranges)))
