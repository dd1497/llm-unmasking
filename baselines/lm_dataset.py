# author: ddukic

import json
import os

from constants import *
from datasets import load_dataset
from torch.utils import data
from tqdm import tqdm


class LMDataset(data.Dataset):
    def __init__(self, causal=False, small_sample=False):
        files = os.listdir(os.path.join(REPO_HOME, "data/processed/bookcorpus"))

        self.input_ids, self.attention_mask = [], []
        self.causal = causal

        if small_sample:
            files = files[:1]

        for file in tqdm(files):
            with open(
                os.path.join(REPO_HOME, "data/processed/bookcorpus", file), "r"
            ) as f:
                content = json.load(f)
                self.input_ids.extend(content["input_ids"])
                self.attention_mask.extend(content["attention_mask"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        result = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }

        if self.causal:
            result["labels"] = result["input_ids"].copy()

        return result


class LMDatasetTAPT(data.Dataset):
    def __init__(self, dataset_name, tokenizer, split="train", causal=True):
        self.causal = causal

        self.load_examples(dataset_name, split)

        tokenized_examples = tokenizer(self.examples)

        result = self.group_texts(tokenized_examples)

        self.input_ids, self.attention_mask = (
            result["input_ids"],
            result["attention_mask"],
        )

    def __len__(self):
        return len(self.input_ids)

    def group_texts(self, examples):
        block_size = 128
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

    def load_conll2003(self, split):
        data_split = load_dataset("conll2003", split=split)

        examples = data_split["tokens"]

        self.examples = [" ".join(x) for x in examples]

    def load_maven(self, split):
        if split == "train":
            fpath = REPO_HOME + "data/processed/maven/train.jsonl"
        elif split == "validation":
            fpath = REPO_HOME + "data/processed/maven/valid.jsonl"
        else:
            # split must be test
            fpath = REPO_HOME + "data/processed/maven/test.jsonl"

        examples = []

        with open(fpath, "r", encoding="utf-8") as f:
            data = [json.loads(ln) for ln in f.readlines()]
            for item in data:
                toks = [cnt["tokens"] for cnt in item["content"]]

                for token_sent in toks:
                    examples.append(" ".join(token_sent))

            self.examples = examples

    def load_ace(self, split):
        if split == "train":
            fpath = REPO_HOME + "data/processed/ace/train.json"
        elif split == "validation":
            fpath = REPO_HOME + "data/processed/ace/dev.json"
        else:
            # split must be test
            fpath = REPO_HOME + "data/processed/ace/test.json"

        examples = []

        with open(fpath, "r") as f:
            data = json.load(f)
            for item in data:
                words = item["words"]

                examples.append(" ".join(words))

        self.examples = examples

    def load_absa(self, dataset_name, split):
        from ast import literal_eval

        import pandas as pd

        if split == "train":
            fpath = REPO_HOME + "data/processed/" + dataset_name + "/train.csv"
        elif split == "validation":
            fpath = REPO_HOME + "data/processed/" + dataset_name + "/valid.csv"
        else:
            # split must be test
            fpath = REPO_HOME + "data/processed/" + dataset_name + "/test.csv"

        df = pd.read_csv(fpath)

        df["tokens"] = df["tokens"].apply(literal_eval)

        examples = df.tokens.tolist()

        self.examples = [" ".join(x) for x in examples]

    def load_examples(self, dataset_name, split):
        if "conll2003" in dataset_name:
            self.load_conll2003(split)
        elif "maven" in dataset_name:
            self.load_maven(split)
        elif "ace" in dataset_name:
            self.load_ace(split)
        elif "absa" in dataset_name:
            self.load_absa(dataset_name, split)

    def __getitem__(self, idx):
        result = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }

        if self.causal:
            result["labels"] = result["input_ids"].copy()

        return result
